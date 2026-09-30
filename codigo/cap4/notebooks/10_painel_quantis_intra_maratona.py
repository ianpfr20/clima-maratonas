"""
Painel de quantis do tempo de prova por edicao x clima, dentro de cada
maratona (analise intra-maratona do capitulo 4).

Para cada maratona com pelo menos N_MIN_EDICOES edicoes, regride os quantis
P10, P25, P50, P75 e P90 do tempo de chegada contra o bulbo umido medio da
edicao. O painel mede se a sensibilidade ao calor cresce da cauda rapida
para a cauda lenta da distribuicao, o efeito de degradacao diferenciada
documentado por Ely et al. e El Helou et al. A mediana (P50) e o IQR
(P75 - P25) fazem parte do painel.

Entradas:
  - tempos de chegada: dados/processados/marathon_canonical.csv (por atleta)
  - clima: dados/processados/apoio/agregados_horarios.csv
    (bulbo umido medio na janela de seis horas a partir da largada, ERA5 via
    Open-Meteo).

O clima vem do mesmo agregado integrado ao dataset canonico e usado na
modelagem, com a mesma definicao de janela.

Saida: dados/processados/apoio/painel_quantis_intra_maratona.csv, alem das
tabelas impressas no terminal.

Uso:
    python codigo/cap4/notebooks/10_painel_quantis_intra_maratona.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[3]
CANONICO = RAIZ / "dados" / "processados" / "marathon_canonical.csv"
AGREGADOS = RAIZ / "dados" / "processados" / "apoio" / "agregados_horarios.csv"

N_MIN_EDICOES = 5
QUANTIS = [10, 25, 50, 75, 90]
# Amplitude minima de bulbo umido (C) para a leitura de slope ser
# confiavel; abaixo disso a variancia em x e baixa demais e o
# coeficiente angular fica instavel (caso de Honolulu e Singapura).
AMP_MIN_CONFIAVEL = 5.0


def carregar() -> pd.DataFrame:
    tempos = pd.read_csv(CANONICO, usecols=["maratona", "ano", "tempo_segundos"])
    tempos = tempos.dropna(subset=["tempo_segundos"])

    def por_edicao(g: pd.DataFrame) -> pd.Series:
        t = g["tempo_segundos"].to_numpy() / 60.0  # minutos
        out = {"n_finishers": len(t), "win": t.min()}
        for q in QUANTIS:
            out[f"p{q}"] = np.percentile(t, q)
        return pd.Series(out)

    ed = tempos.groupby(["maratona", "ano"]).apply(por_edicao, include_groups=False)
    ed = ed.reset_index()
    ed["iqr"] = ed["p75"] - ed["p25"]

    clima = pd.read_csv(AGREGADOS, usecols=["maratona", "ano", "bulbo_umido_med"])
    df = ed.merge(clima, on=["maratona", "ano"], how="inner")
    return df


def slope_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 5:
        return (np.nan, np.nan)
    b = float(np.polyfit(x, y, 1)[0])
    r = float(stats.pearsonr(x, y)[0])
    return (b, r)


def main() -> None:
    df = carregar()
    cont = df.groupby("maratona").size()
    elegiveis = sorted(cont[cont >= N_MIN_EDICOES].index.tolist())

    linhas = []
    for m in elegiveis:
        sub = df[df["maratona"] == m]
        x = sub["bulbo_umido_med"].to_numpy()
        amp = float(x.max() - x.min())
        reg = {"maratona": m, "n": len(sub), "amp_bulbo": amp,
               "confiavel": amp >= AMP_MIN_CONFIAVEL}
        for q in QUANTIS:
            b, r = slope_r(x, sub[f"p{q}"].to_numpy())
            reg[f"slope_p{q}"] = b
            reg[f"r_p{q}"] = r
        b_iqr, r_iqr = slope_r(x, sub["iqr"].to_numpy())
        reg["slope_iqr"], reg["r_iqr"] = b_iqr, r_iqr
        b_win, r_win = slope_r(x, sub["win"].to_numpy())
        reg["r_win"] = r_win
        reg["cv_win"] = 100 * sub["win"].std() / sub["win"].mean()
        reg["cv_p50"] = 100 * sub["p50"].std() / sub["p50"].mean()
        linhas.append(reg)

    res = pd.DataFrame(linhas)
    # ordenar por slope da mediana (efeito climatico) decrescente, confiaveis primeiro
    res = res.sort_values(["confiavel", "slope_p50"], ascending=[False, False])

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    print("=== PAINEL DE QUANTIS: slope (min por C de bulbo umido) e r de Pearson ===")
    print("(janela de 6 h a partir da largada; * marca amplitude de bulbo < 5 C, slope nao confiavel)\n")
    print(f"{'maratona':<11}{'n':>3} {'amp':>5}  "
          f"{'P10':>11}{'P25':>11}{'P50':>11}{'P75':>11}{'P90':>11}{'IQR':>11}")
    for _, r in res.iterrows():
        flag = "" if r["confiavel"] else "*"
        cells = ""
        for q in QUANTIS:
            cells += f"{r[f'slope_p{q}']:>6.2f}/{r[f'r_p{q}']:>4.2f}"
        cells += f"{r['slope_iqr']:>6.2f}/{r['r_iqr']:>4.2f}"
        print(f"{r['maratona']+flag:<11}{int(r['n']):>3} {r['amp_bulbo']:>5.1f}  {cells}")

    print("\n=== TEMPO DO VENCEDOR (comparacao com a mediana como referencia da edicao) ===")
    print(f"{'maratona':<11}{'r_win':>7}{'r_p50':>7}{'CV_win%':>9}{'CV_p50%':>9}")
    for _, r in res.iterrows():
        print(f"{r['maratona']:<11}{r['r_win']:>7.2f}{r['r_p50']:>7.2f}"
              f"{r['cv_win']:>9.1f}{r['cv_p50']:>9.1f}")

    saida = RAIZ / "dados" / "processados" / "apoio" / "painel_quantis_intra_maratona.csv"
    res.to_csv(saida, index=False)
    print(f"\n[csv] {saida}")


if __name__ == "__main__":
    main()
