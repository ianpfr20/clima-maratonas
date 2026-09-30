"""
Agregados climaticos por edicao, calculados a partir do clima horario ERA5
via Open-Meteo (dados/brutos/meteo/era5/hourly/<maratona>_<ano>_hourly.csv).

Inspirado no tutorial TensorFlow de series temporais em tres pontos:
- decomposicao da direcao do vento em componentes vetoriais Wx, Wy
  para evitar a descontinuidade em 360 -> 0;
- codificacao seno-cosseno do horario do pico de radiacao (`hora_pico_*`);
- representacao da janela de prova como sequencia que e resumida em
  estatisticas, em vez de manter o tempo bruto como input direto do
  modelo tabular.

Janela climatica:
- Regra unica: seis horas a partir da largada, [largada, largada + 6 horas).
- A largada por edicao vem de dados/processados/apoio/horarios_largada.csv
  (colunas: maratona, ano, hora_largada em formato HH:MM).
- Os registros horarios sao filtrados pelo timestamp completo, sem descartar
  os minutos da largada.
- Chuva e radiacao representam [timestamp - 1 hora, timestamp). Seus totais e
  medias usam a duracao de intersecao com a prova; fracoes pressupõem intensidade
  uniforme dentro da hora. Maximos se referem aos intervalos sobrepostos.
- Horario ausente, duplicado ou malformado interrompe o processamento; nao ha
  janela alternativa silenciosa.
- Cada CSV horario e conferido contra o recibo JSON gravado na coleta
  (modelo ERA5 e hash SHA-256 do arquivo).

Saidas:
- dados/processados/apoio/agregados_horarios.csv

Uso (a partir da raiz do repositorio):
    python codigo/cap4/notebooks/03_agregados_horarios.py
"""

from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
DIR_HOURLY = RAIZ / "dados" / "brutos" / "meteo" / "era5" / "hourly"
DIR_APOIO = RAIZ / "dados" / "processados" / "apoio"
ARQ_LARGADAS = DIR_APOIO / "horarios_largada.csv"
SAIDA = DIR_APOIO / "agregados_horarios.csv"

JANELA_HORAS = 6
LIMIARES_BULBO = (18.0, 20.0, 22.0)

PADRAO_NOME = re.compile(r"^(?P<maratona>[a-z_]+)_(?P<ano>\d{4})_hourly\.csv$")


def carregar_largadas() -> pd.DataFrame:
    if not ARQ_LARGADAS.exists():
        raise FileNotFoundError(
            f"Arquivo de horarios de largada nao encontrado: {ARQ_LARGADAS}"
        )

    df = pd.read_csv(ARQ_LARGADAS)
    obrigatorias = {"maratona", "ano", "hora_largada"}
    ausentes = obrigatorias.difference(df.columns)
    if ausentes:
        raise ValueError(
            "Colunas obrigatorias ausentes em horarios_largada.csv: "
            + ", ".join(sorted(ausentes))
        )

    df["maratona"] = df["maratona"].astype(str).str.strip().str.lower()
    df["ano"] = df["ano"].astype(int)
    duplicadas = df.duplicated(["maratona", "ano"], keep=False)
    if duplicadas.any():
        chaves = df.loc[duplicadas, ["maratona", "ano"]].drop_duplicates()
        raise ValueError(
            "Horarios de largada duplicados para: "
            + ", ".join(
                f"{row.maratona} {row.ano}" for row in chaves.itertuples(index=False)
            )
        )

    horario = pd.to_datetime(
        df["hora_largada"].astype(str).str.strip(),
        format="%H:%M",
        errors="coerce",
    )
    invalidas = horario.isna()
    if invalidas.any():
        valores = df.loc[invalidas, ["maratona", "ano", "hora_largada"]]
        raise ValueError(
            "Horarios de largada invalidos:\n" + valores.to_string(index=False)
        )

    df["largada_delta"] = (
        pd.to_timedelta(horario.dt.hour, unit="h")
        + pd.to_timedelta(horario.dt.minute, unit="m")
    )
    return df


def janela_para_edicao(
    maratona: str, ano: int, largadas: pd.DataFrame
) -> tuple[pd.Timedelta, pd.Timedelta]:
    """Retorna o intervalo [largada, largada + 6 horas) da edicao."""
    sel = largadas[(largadas["maratona"] == maratona) & (largadas["ano"] == ano)]
    if sel.empty:
        raise ValueError(f"Horario de largada ausente para {maratona} {ano}")
    inicio = sel.iloc[0]["largada_delta"]
    return inicio, inicio + pd.Timedelta(hours=JANELA_HORAS)


def listar_arquivos() -> list[Path]:
    return sorted(p for p in DIR_HOURLY.glob("*_hourly.csv"))


def metadata_arquivo(p: Path) -> tuple[str, int]:
    m = PADRAO_NOME.match(p.name)
    if m is None:
        raise ValueError(f"nome fora do padrao: {p.name}")
    return m["maratona"], int(m["ano"])


def recortar_janela(
    df: pd.DataFrame, inicio: pd.Timedelta, fim: pd.Timedelta
) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Arquivo horario contem timestamps invalidos")

    if df["time"].duplicated().any() or not df["time"].is_monotonic_increasing:
        raise ValueError("Arquivo horario contem timestamps duplicados ou fora de ordem")
    base = df["time"].iloc[0].normalize()
    instante_inicio = base + inicio
    instante_fim = base + fim
    df["hora"] = df["time"].dt.hour + df["time"].dt.minute / 60
    return df[
        (df["time"] >= instante_inicio) & (df["time"] < instante_fim)
    ].reset_index(drop=True)


def recortar_intervalos(
    df: pd.DataFrame, inicio: pd.Timedelta, fim: pd.Timedelta
) -> pd.DataFrame:
    """Intersecoes de [t-1h, t) com a prova para chuva e radiacao.

    Frações de hora pressupõem taxa/intensidade uniforme no intervalo de origem.
    O horario de pico identifica o centro do intervalo horario, nao um pico instantaneo.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    if df.empty or df["time"].duplicated().any() or not df["time"].is_monotonic_increasing:
        raise ValueError("Serie horaria vazia, duplicada ou fora de ordem")
    if not df["time"].eq(df["time"].dt.floor("h")).all():
        raise ValueError("Serie meteorologica fora da grade de horas cheias")
    base = df["time"].iloc[0].normalize()
    a, b = base + inicio, base + fim
    ends = df["time"]
    starts = ends - pd.Timedelta(hours=1)
    overlap = (ends.clip(upper=b) - starts.clip(lower=a)).dt.total_seconds().clip(lower=0) / 3600
    df["duracao_horas"] = overlap
    df = df[overlap > 0].copy().reset_index(drop=True)
    if not np.isclose(df["duracao_horas"].sum(), (b-a).total_seconds()/3600, rtol=0, atol=1e-10):
        raise ValueError("Cobertura incompleta dos intervalos de chuva e radiacao")
    if df[["precipitation", "shortwave_radiation"]].isna().any().any():
        raise ValueError("Chuva ou radiacao ausente no intervalo da prova")
    midpoints = df["time"] - pd.Timedelta(minutes=30)
    df["hora"] = midpoints.dt.hour + midpoints.dt.minute / 60
    return df


def horas_decimais(valor: pd.Timedelta) -> float:
    return valor.total_seconds() / 3600


def features_da_janela(df: pd.DataFrame, intervalos: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}

    f: dict[str, float] = {}
    f["n_horas_janela"] = len(df)

    f["temp_media"] = df["temperature_2m"].mean()
    f["temp_max"] = df["temperature_2m"].max()
    f["temp_min"] = df["temperature_2m"].min()
    f["temp_amplitude"] = f["temp_max"] - f["temp_min"]
    f["temp_std"] = df["temperature_2m"].std(ddof=0)

    f["bulbo_umido_med"] = df["wet_bulb_temperature_2m"].mean()
    f["bulbo_umido_max"] = df["wet_bulb_temperature_2m"].max()
    for limiar in LIMIARES_BULBO:
        col = f"bulbo_umido_horas_acima_{int(limiar)}"
        f[col] = float((df["wet_bulb_temperature_2m"] >= limiar).sum())

    f["umidade_med"] = df["relative_humidity_2m"].mean()
    f["umidade_max"] = df["relative_humidity_2m"].max()

    f["temp_aparente_med"] = df["apparent_temperature"].mean()
    f["temp_aparente_max"] = df["apparent_temperature"].max()

    f["orvalho_med"] = df["dew_point_2m"].mean()

    f["vento_med_ms"] = df["wind_speed_10m"].mean()
    f["vento_max_ms"] = df["wind_speed_10m"].max()
    f["rajada_max_ms"] = df["wind_gusts_10m"].max()

    angulo = np.deg2rad(df["wind_direction_10m"].astype(float))
    velocidade = df["wind_speed_10m"].astype(float)
    f["wx_med"] = float((velocidade * np.cos(angulo)).mean())
    f["wy_med"] = float((velocidade * np.sin(angulo)).mean())

    pesos = intervalos["duracao_horas"]
    f["precipitacao_total_mm"] = (intervalos["precipitation"] * pesos).sum()
    f["precipitacao_max_horaria_mm"] = intervalos["precipitation"].max()
    f["horas_com_chuva"] = float(pesos[intervalos["precipitation"] > 0.1].sum())

    f["radiacao_med_wm2"] = float(np.average(intervalos["shortwave_radiation"], weights=pesos))
    f["radiacao_max_wm2"] = intervalos["shortwave_radiation"].max()
    # Integral em Wh/m^2, ponderada pela duracao sobreposta de cada intervalo.
    f["radiacao_acumulada_whm2"] = (intervalos["shortwave_radiation"] * pesos).sum()

    primeira_metade = df.iloc[: max(1, len(df) // 2)]
    segunda_metade = df.iloc[max(1, len(df) // 2):]
    f["temp_derivada_meia_prova_c"] = float(
        segunda_metade["temperature_2m"].mean() - primeira_metade["temperature_2m"].mean()
    )
    f["bulbo_derivada_meia_prova_c"] = float(
        segunda_metade["wet_bulb_temperature_2m"].mean()
        - primeira_metade["wet_bulb_temperature_2m"].mean()
    )

    idx_pico = int(intervalos["shortwave_radiation"].idxmax())
    hora_pico = float(intervalos.loc[idx_pico, "hora"])
    f["hora_pico_radiacao"] = float(hora_pico)
    angulo_pico = 2 * np.pi * hora_pico / 24
    f["hora_pico_sin"] = float(np.sin(angulo_pico))
    f["hora_pico_cos"] = float(np.cos(angulo_pico))

    return f


def processar() -> pd.DataFrame:
    largadas = carregar_largadas()
    arquivos = listar_arquivos()
    print(f"[carga] {len(arquivos)} arquivos horarios em {DIR_HOURLY}")

    linhas = []
    for p in arquivos:
        recibo = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if recibo["parametros"].get("models") != "era5":
            raise ValueError(f"Arquivo fora da coleta ERA5: {p.name}")
        if hashlib.sha256(p.read_bytes()).hexdigest() != recibo["csv_sha256"]:
            raise ValueError(f"Arquivo horario diverge do recibo: {p.name}")
        maratona, ano = metadata_arquivo(p)
        inicio, fim = janela_para_edicao(maratona, ano, largadas)
        df = pd.read_csv(p)
        recorte = recortar_janela(df, inicio, fim)
        if len(recorte) != JANELA_HORAS:
            raise ValueError(
                f"Janela de {maratona} {ano} tem {len(recorte)} registros; "
                f"esperados {JANELA_HORAS}"
            )
        intervalos = recortar_intervalos(df, inicio, fim)
        feats = features_da_janela(recorte, intervalos)
        if not feats:
            raise ValueError(f"Janela climatica vazia para {maratona} {ano}")
        feats.update(
            {
                "maratona": maratona,
                "ano": ano,
                "janela_inicio_h": horas_decimais(inicio),
                "janela_fim_h": horas_decimais(fim),
                "fonte_janela": "dinamica",
            }
        )
        linhas.append(feats)

    print(f"[janela] largada + {JANELA_HORAS}h em {len(linhas)} edicoes")
    saida = pd.DataFrame(linhas)
    cols_chave = ["maratona", "ano", "janela_inicio_h", "janela_fim_h", "fonte_janela"]
    cols_demais = [c for c in saida.columns if c not in cols_chave]
    return saida[cols_chave + cols_demais].sort_values(["maratona", "ano"])


def main() -> None:
    DIR_APOIO.mkdir(parents=True, exist_ok=True)
    df = processar()
    df.to_csv(SAIDA, index=False, float_format="%.4f")
    print(f"[saida] {SAIDA} ({df.shape})")
    print()
    print("=== amostra ===")
    print(df.head(8).to_string(index=False))
    print()
    print("=== resumo numerico ===")
    print(df.describe(include="number").round(2).to_string())


if __name__ == "__main__":
    main()
