#!/usr/bin/env python3
"""
coleta_meteo.py — Coleta dados meteorológicos horários (reanálise ERA5) de cada
edição de maratona pela API Open-Meteo Historical Weather
(archive-api.open-meteo.com).

Entradas:
    dados/brutos/maratonas_datas.csv                  — data, coordenadas e fuso de cada edição
    dados/processados/apoio/horarios_largada.csv      — horário de largada de cada edição

Saídas:
    dados/brutos/meteo/era5/hourly/<maratona>_<ano>_hourly.csv  — dados horários do dia da prova, incluindo o dia seguinte quando a janela de seis horas após a largada passa da meia-noite
    dados/brutos/meteo/era5/hourly/<maratona>_<ano>_hourly.json — parâmetros da consulta, metadados da resposta e hash SHA-256 do CSV

Os agregados da prova são calculados separadamente por
codigo/cap4/notebooks/03_agregados_horarios.py, sempre no intervalo entre a
largada de cada edição e as seis horas seguintes.

Uso:
    python codigo/etl/coleta_meteo.py              # coleta todas as edições pendentes
    python codigo/etl/coleta_meteo.py --forcar     # recoleta mesmo se o arquivo já existe
"""

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

# ── Caminhos ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATAS_CSV  = ROOT / "dados" / "brutos" / "maratonas_datas.csv"
HOURLY_DIR = ROOT / "dados" / "brutos" / "meteo" / "era5" / "hourly"

# ── API Open-Meteo ────────────────────────────────────────────────────────────
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wet_bulb_temperature_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]

def parametros_consulta(lat: float, lon: float, date: str, timezone: str, end_date: str | None = None) -> dict:
    return {
        "latitude":           lat,
        "longitude":          lon,
        "start_date":         date,
        "end_date":           end_date or date,
        "hourly":             ",".join(HOURLY_VARS),
        "timezone":           timezone,
        "wind_speed_unit":    "ms",
        "temperature_unit":   "celsius",
        "precipitation_unit": "mm",
        "models":             "era5",
    }


def query_open_meteo(lat: float, lon: float, date: str, timezone: str, end_date: str | None = None) -> pd.DataFrame:
    """Consulta ERA5 explicitamente e preserva os metadados da resposta."""
    params = parametros_consulta(lat, lon, date, timezone, end_date)
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["hourly"])
    if not set(["time", *HOURLY_VARS]).issubset(df.columns):
        raise ValueError("Resposta meteorologica sem todas as variaveis solicitadas")
    if df.empty or df[["time", *HOURLY_VARS]].isna().any().any():
        raise ValueError("Resposta meteorologica vazia ou com valores ausentes")
    df["time"] = pd.to_datetime(df["time"])
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        raise ValueError("Horarios meteorologicos fora de ordem ou duplicados")
    df.insert(0, "date", date)
    df.attrs["proveniencia"] = {
        "url": BASE_URL,
        "parametros": params,
        "coletado_em": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resposta": {k: v for k, v in data.items() if k != "hourly"},
    }
    return df


def cache_valido(path: Path, params: dict) -> bool:
    # O CSV so e reaproveitado se o recibo .json tiver os mesmos parametros de consulta e o hash conferir.
    metadata = path.with_suffix(".json")
    if not path.exists() or not metadata.exists():
        return False
    try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        return (
            saved["parametros"] == params
            and saved["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    except (OSError, ValueError, KeyError):
        return False


def salvar_coleta(df: pd.DataFrame, path: Path) -> None:
    """Grava dados e recibo; um par incompleto nunca e reutilizado como cache."""
    df.to_csv(path, index=False, lineterminator="\r\n")
    metadata = dict(df.attrs["proveniencia"])
    metadata["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_bytes(
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        .replace("\n", "\r\n").encode("utf-8")
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--forcar", action="store_true",
        help="recoleta mesmo se arquivo já existe"
    )
    args = parser.parse_args()

    HOURLY_DIR.mkdir(parents=True, exist_ok=True)

    # Lê o CSV de datas; ignora linhas de comentário (início com #)
    datas = pd.read_csv(DATAS_CSV, comment="#")
    largadas = pd.read_csv(ROOT / "dados/processados/apoio/horarios_largada.csv")
    datas = datas.merge(largadas[["maratona", "ano", "hora_largada"]], on=["maratona", "ano"], validate="one_to_one", how="left")
    if datas["hora_largada"].isna().any():
        raise ValueError("Edicao sem horario de largada para definir o fim da coleta")
    print(f"Edições carregadas: {len(datas)}")

    # Ordena por maratona e ano para facilitar o acompanhamento
    datas = datas.sort_values(["maratona", "ano"]).reset_index(drop=True)

    erros: list[str] = []

    for i, row in datas.iterrows():
        maratona = str(row["maratona"])
        ano      = int(row["ano"])
        data     = str(row["data"])
        lat      = float(row["latitude"])
        lon      = float(row["longitude"])
        tz       = str(row["timezone"])
        fim = (pd.Timestamp(f"{data} {row['hora_largada']}") + pd.Timedelta(hours=6)).ceil("h")
        end_date = fim.strftime("%Y-%m-%d")

        slug = f"{maratona}_{ano}"
        hourly_file = HOURLY_DIR / f"{slug}_hourly.csv"

        if cache_valido(hourly_file, parametros_consulta(lat, lon, data, tz, end_date)) and not args.forcar:
            print(f"  [cache] {slug}")
        else:
            print(f"  [{i+1}/{len(datas)}] {slug}  ({data})  lat={lat} lon={lon}")
            try:
                df_h = query_open_meteo(lat, lon, data, tz, end_date)
                salvar_coleta(df_h, hourly_file)
                time.sleep(0.4)  # pausa entre consultas a API gratuita; limites em https://open-meteo.com/en/pricing
            except requests.HTTPError as e:
                print(f"    ERRO HTTP {slug}: {e}")
                erros.append(slug)
                continue
            except Exception as e:
                print(f"    ERRO {slug}: {e}")
                erros.append(slug)
                continue

    if erros:
        print(f"\nATENCAO: erros em {len(erros)} edicao(oes): {erros}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
