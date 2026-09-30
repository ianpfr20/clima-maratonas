"""
Monta a tabela de treino por edicao (uma linha por edicao): alvos = quantis
do tempo de prova; features = clima da janela de seis horas a partir da
largada + contexto da edicao.

Cada linha e uma edicao (maratona, ano). Os alvos sao os quantis do tempo
de chegada em MINUTOS (P10, P25, P50, P75, P90) e o IQR (P75 - P25). As
features de contexto (n_finishers, proporcao de homens,
codificacao ciclica do dia do ano) acompanham as 31 features climaticas da
janela [largada, largada+6h) vindas dos agregados horarios (script 03).

A tabela NAO inclui alvos em desvio historico (diferenca para a mediana
historica da prova). Calcular essa mediana aqui, sobre o conjunto inteiro,
vazaria informacao de teste para o treino. O desvio historico e calculado
dentro de cada particao da validacao, nos scripts de treino (12 e 13).

Convencao de hemisferio: maratonas do hemisferio sul recebem sinal -1 em
sin e cos do dia do ano, alinhando estacao em vez de calendario civil.

Entradas: dados/processados/marathon_canonical.csv e
dados/processados/apoio/agregados_horarios.csv.
Saida: tabela de treino por edicao (dataset_treino_edicoes.csv; caminho em
SAIDA).

Uso (a partir da raiz do repositorio):
    python codigo/cap4/notebooks/11_dataset_treino_edicoes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
CANONICO = RAIZ / "dados" / "processados" / "marathon_canonical.csv"
AGREGADOS = RAIZ / "dados" / "processados" / "apoio" / "agregados_horarios.csv"
SAIDA = RAIZ / "dataset117" / "dataset_treino_edicoes.csv"

PERIODO_ANO_DIAS = 365.2425

# Correcao de hemisferio: maratonas do sul recebem sinal -1 em sin e cos,
# alinhando estacao em vez de calendario civil.
MARATONAS_HEMISFERIO_SUL = {"rio", "durban"}

QUANTIS = [10, 25, 50, 75, 90]


def carregar_alvos_e_contexto() -> pd.DataFrame:
    canon = pd.read_csv(
        CANONICO,
        usecols=["maratona", "ano", "data_prova", "tempo_segundos", "genero"],
        parse_dates=["data_prova"],
    )
    canon = canon.dropna(subset=["tempo_segundos"])

    def por_edicao(g: pd.DataFrame) -> pd.Series:
        # alvos em MINUTOS
        t_min = g["tempo_segundos"].to_numpy() / 60.0
        out = {}
        for q in QUANTIS:
            out[f"p{q}"] = float(np.percentile(t_min, q))
        out["iqr"] = out["p75"] - out["p25"]
        out["n_finishers"] = len(g)
        out["pct_M"] = float((g["genero"] == "M").mean())
        out["data_prova"] = g["data_prova"].iloc[0]
        return pd.Series(out)

    ed = canon.groupby(["maratona", "ano"]).apply(por_edicao, include_groups=False)
    ed = ed.reset_index()

    # codificacao ciclica do dia do ano, com correcao de hemisferio
    ed["data_prova"] = pd.to_datetime(ed["data_prova"])
    dia_ano = ed["data_prova"].dt.dayofyear.astype(float)
    angulo = 2 * np.pi * dia_ano / PERIODO_ANO_DIAS
    sinal_hem = np.where(ed["maratona"].isin(MARATONAS_HEMISFERIO_SUL), -1.0, 1.0)
    ed["dia_ano_sin"] = sinal_hem * np.sin(angulo)
    ed["dia_ano_cos"] = sinal_hem * np.cos(angulo)
    ed["ln_n_finishers"] = np.log1p(ed["n_finishers"])
    ed["n_finishers"] = ed["n_finishers"].astype(int)
    return ed


def carregar_clima() -> tuple[pd.DataFrame, list[str]]:
    agr = pd.read_csv(AGREGADOS)
    cols_meta_drop = ["janela_inicio_h", "janela_fim_h", "fonte_janela", "n_horas_janela"]
    ausentes = set(cols_meta_drop).difference(agr.columns)
    if ausentes:
        raise ValueError(
            "Metadados da janela climatica ausentes: " + ", ".join(sorted(ausentes))
        )
    if not agr["fonte_janela"].astype(str).str.lower().eq("dinamica").all():
        raise ValueError("O dataset de treino aceita somente a janela largada + 6h")
    duracao = agr["janela_fim_h"] - agr["janela_inicio_h"]
    janela_valida = np.isclose(duracao, 6) & agr["n_horas_janela"].eq(6)
    if not janela_valida.all():
        invalidas = agr.loc[
            ~janela_valida,
            [
                "maratona",
                "ano",
                "janela_inicio_h",
                "janela_fim_h",
                "n_horas_janela",
            ],
        ]
        raise ValueError(
            "Edicoes fora da janela largada + 6h:\n" + invalidas.to_string(index=False)
        )
    agr = agr.drop(columns=cols_meta_drop)
    cols_clima = [c for c in agr.columns if c not in ("maratona", "ano")]
    return agr, cols_clima


def relatar_anti_joins(ed: pd.DataFrame, agr: pd.DataFrame) -> None:
    chaves_ed = ed[["maratona", "ano"]].drop_duplicates()
    chaves_agr = agr[["maratona", "ano"]].drop_duplicates()

    # presentes nos agregados e ausentes no canonico
    so_agr = chaves_agr.merge(chaves_ed, on=["maratona", "ano"], how="left", indicator=True)
    so_agr = so_agr[so_agr["_merge"] == "left_only"].drop(columns="_merge")
    # presentes no canonico e ausentes nos agregados
    so_ed = chaves_ed.merge(chaves_agr, on=["maratona", "ano"], how="left", indicator=True)
    so_ed = so_ed[so_ed["_merge"] == "left_only"].drop(columns="_merge")

    print(f"[dataset] edicoes no canonico = {len(chaves_ed)}, nos agregados = {len(chaves_agr)}")
    print(f"[dataset] anti-join: {len(so_agr)} edicao(oes) nos agregados sem par no canonico")
    for _, r in so_agr.sort_values(["maratona", "ano"]).iterrows():
        print(f"[dataset]   descartada (agregados sem canonico): {r['maratona']} {int(r['ano'])}")
    print(f"[dataset] anti-join: {len(so_ed)} edicao(oes) no canonico sem par nos agregados")
    for _, r in so_ed.sort_values(["maratona", "ano"]).iterrows():
        print(f"[dataset]   descartada (canonico sem agregados): {r['maratona']} {int(r['ano'])}")


def validar(df: pd.DataFrame, cols_clima: list[str]) -> None:
    print()
    print("=== validacoes ===")

    # 1) numero de linhas
    n = len(df)
    ok_n = n == 117
    print(f"[valida] linhas = {n} (esperado 117) -> {'OK' if ok_n else 'FALHA'}")

    # 2) monotonicidade dos quantis
    mono = (
        (df["p10"] < df["p25"]) & (df["p25"] < df["p50"])
        & (df["p50"] < df["p75"]) & (df["p75"] < df["p90"])
    )
    ok_mono = bool(mono.all())
    print(f"[valida] monotonicidade p10<p25<p50<p75<p90 em todas as linhas -> "
          f"{'OK' if ok_mono else 'FALHA'} ({int(mono.sum())}/{n})")
    if not ok_mono:
        for _, r in df[~mono].iterrows():
            print(f"[valida]   violacao: {r['maratona']} {int(r['ano'])}")

    # 3) NaN por coluna
    alvos = ["p10", "p25", "p50", "p75", "p90", "iqr"]
    nan_alvos = df[alvos].isna().sum()
    nan_alvos_total = int(nan_alvos.sum())
    print(f"[valida] NaN nos alvos = {nan_alvos_total} -> "
          f"{'OK' if nan_alvos_total == 0 else 'FALHA'}")
    if nan_alvos_total:
        print(nan_alvos[nan_alvos > 0].to_string())
    nan_feat = df[cols_clima].isna().sum()
    nan_feat = nan_feat[nan_feat > 0]
    if len(nan_feat):
        print(f"[valida] NaN em features climaticas (informativo):")
        print(nan_feat.to_string())
    else:
        print("[valida] sem NaN nas features climaticas")

    # 4) sanidade cruzada: p50 == mediana do tempo no canonico (em horas) * 60
    canon = pd.read_csv(CANONICO, usecols=["maratona", "ano", "tempo_segundos"])
    canon = canon.dropna(subset=["tempo_segundos"])
    med = (
        canon.groupby(["maratona", "ano"], as_index=False)
        .agg(mediana_tempo_h=("tempo_segundos", lambda s: s.median() / 3600))
    )
    chk = df[["maratona", "ano", "p50"]].merge(med, on=["maratona", "ano"], how="left")
    dif = (chk["p50"] - chk["mediana_tempo_h"] * 60).abs()
    ok_p50 = bool((dif < 1e-6).all())
    print(f"[valida] p50 == mediana_tempo_h*60 (tol 1e-6) -> "
          f"{'OK' if ok_p50 else 'FALHA'} (max dif = {dif.max():.2e})")


def main() -> None:
    ed = carregar_alvos_e_contexto()
    agr, cols_clima = carregar_clima()
    print(f"[dataset] {len(cols_clima)} features climaticas mantidas dos agregados")

    relatar_anti_joins(ed, agr)

    df = ed.merge(agr, on=["maratona", "ano"], how="inner")
    print(f"[dataset] {len(df)} edicoes apos merge inner")

    # ordenacao das colunas: identificadores, alvos, contexto, clima
    cols_id = ["maratona", "ano", "data_prova"]
    cols_alvo = ["p10", "p25", "p50", "p75", "p90", "iqr"]
    cols_ctx = ["n_finishers", "ln_n_finishers", "pct_M",
                "dia_ano_sin", "dia_ano_cos"]
    ordem = cols_id + cols_alvo + cols_ctx + cols_clima
    df = df[ordem]

    validar(df, cols_clima)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False)
    print()
    print(f"[csv] {SAIDA}")
    print(f"[csv] {len(df)} linhas x {len(df.columns)} colunas")
    print(f"[csv] p50 (min): min={df['p50'].min():.2f}, "
          f"mediana={df['p50'].median():.2f}, max={df['p50'].max():.2f}")


if __name__ == "__main__":
    main()
