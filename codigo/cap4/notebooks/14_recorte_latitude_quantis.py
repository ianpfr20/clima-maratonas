"""
14_recorte_latitude_quantis.py - bateria do recorte homogeneo de latitude
=========================================================================

Objetivo
--------
Repete a bateria do desenho final (script 13) restrita as tres provas do
corpus situadas na mesma faixa de latitude (40-43 N) e com serie historica
longa (>= 15 edicoes): boston, chicago e nyc, ou seja, 49 das 117 edicoes
(dataset49). Perguntas definidas antes da execucao:

(i)   as familias de arvore continuam a frente no problema homogeneo?
(ii)  o ganho sobre a referencia historica aumenta no recorte?
(iii) reduzir a massa de treino (48 vs 116 edicoes por particao) prejudica
      o modelo?

Comparacao justa
----------------
O protocolo e o mesmo da bateria principal com as 117 edicoes (scripts
12/13, chamada aqui de "exp 1"): mesmas fabricas e configuracoes, mesma
semente, mesmo pre-processamento, mesma referencia historica (mediana
historica por prova ajustada so no treino) e nenhuma exclusao nova. So o
corpus muda.

O criterio e o MAE sobre AS MESMAS 49 edicoes, em tres trilhas:
  a) referencia historica - identica por construcao nos dois
     experimentos: as edicoes irmas de treino de cada edicao sao as mesmas
     na validacao com uma edicao reservada por vez sobre 117 edicoes
     (LOO-117) e sobre 49 (LOO-49). O script confere a igualdade
     numericamente;
  b) exp 1 restrito - predicoes LOO-117 do script 13 filtradas as 49;
  c) recorte - bateria LOO-49 deste script.

Leitura do resultado (definida antes dos numeros):
  - iguala: o corpus de treino nao tem efeito mensuravel;
  - melhora: um problema mais uniforme compensa ter menos dados;
  - piora: a heterogeneidade das 117 edicoes transfere sinal entre provas
    e o volume importa.
  O script relata a direcao em qualquer caso.

Verificacoes de reproducao (o script para se alguma falhar)
-----------------------------------------------------------
  1. a agregacao deste script reproduz grade_formulacoes_quantis.csv a
     partir de predicoes_grade_formulacoes.csv, celula a celula (tol
     1e-9);
  2. a referencia por edicao do LOO-49 e identica a ref_min do exp 1
     restrito as 49, e o MAE da celula referencia_historica coincide nas
     duas trilhas (tol 1e-9);
  3. as validacoes de cobertura do script 13 sao aplicadas a grade do
     recorte.

Ressalvas de interpretacao
--------------------------
  - A exclusao de maratona inteira (LOMO) no recorte retira 1 de 3 provas
    (~1/3 do treino), e a prova excluida nao tem historico proprio: a
    referencia cai no centro global das outras duas. O significado nao e
    o mesmo do LOMO com as 11 provas do corpus completo.
  - corte temporal no recorte: treino <= 2017, teste >= 2018; as tres
    provas tem historico no treino.
  - sensibilidade de horario atipico: boston 2005 e 2006 estao DENTRO do
    recorte; singapura 2019 fica fora por construcao.
  - p ~ n: 48 linhas de treino para 36 preditores; familias com pouca
    regularizacao (linear, mlp) podem degradar. Isso e resultado do
    experimento, nao defeito de execucao.

Saidas (dados/processados/apoio/)
---------------------------------
  recorte_grade_formulacoes_quantis.csv
  recorte_predicoes_grade_formulacoes.csv
  recorte_exp1_restrito_quantis.csv
  recorte_comparacao_tres_vias.csv
  recorte_atribuicao_blocos_quantis.csv (+ _celulas; + predicoes)
  recorte_robustez_formulacoes.csv (+ predicoes)
  recorte_sensibilidade_horario_largada.csv
  recorte_estabilidade_por_edicao.csv (+ _resumo)
  figura: codigo/cap4/figuras/fig_recorte_grade_formulacoes.png

Uso
---
  python codigo/cap4/notebooks/14_recorte_latitude_quantis.py
  [--smoke]  so ridge+gbm e p50; nao grava artefatos

Requer as saidas do script 13 (grade e predicoes com as 117 edicoes).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

# Uma thread por biblioteca numerica, como nos scripts 12 e 13. Precisa ser
# definido ANTES de importar numpy/sklearn (os modulos abaixo usam setdefault).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_AQUI = Path(__file__).resolve().parent
_CAMINHO_13 = _AQUI / "13_desenho_treino_quantis.py"


# Reaproveita funcoes e constantes do script 13. O carregamento e por caminho
# porque o nome do arquivo comeca com digito.
def _carregar_modulo_13():
    spec = importlib.util.spec_from_file_location("desenho13", _CAMINHO_13)
    if spec is None or spec.loader is None:
        raise ImportError(f"nao consegui carregar {_CAMINHO_13}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t13 = _carregar_modulo_13()
t12 = t13.t12

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

APOIO_DIR = t13.APOIO_DIR
FIG_DIR = t13.FIG_DIR
ALVOS = list(t13.ALVOS)
FAMILIAS = dict(t13.FAMILIAS)
TOL = t13.TOL

FORMULACAO_REFERENCIA = t13.FORMULACAO_REFERENCIA
FORMULACAO_ABSOLUTO = t13.FORMULACAO_ABSOLUTO
FORMULACAO_DESVIO = t13.FORMULACAO_DESVIO

# recorte: as tres provas do corpus na faixa 40-43 N, todas com >= 15
# edicoes (boston 42,4 N; chicago 41,9 N; nyc 40,7 N). O criterio de
# latitude seleciona exatamente as tres; berlin (52,5 N) e as demais
# ficam fora.
RECORTE = ("boston", "chicago", "nyc")
EDICOES_ESPERADAS = {"boston": 15, "chicago": 17, "nyc": 17}

SAIDA_GRADE = APOIO_DIR / "recorte_grade_formulacoes_quantis.csv"
SAIDA_PRED_GRADE = APOIO_DIR / "recorte_predicoes_grade_formulacoes.csv"
SAIDA_EXP1_RESTRITO = APOIO_DIR / "recorte_exp1_restrito_quantis.csv"
SAIDA_COMPARACAO = APOIO_DIR / "recorte_comparacao_tres_vias.csv"
SAIDA_BLOCOS = APOIO_DIR / "recorte_atribuicao_blocos_quantis.csv"
SAIDA_BLOCOS_CELULAS = APOIO_DIR / "recorte_atribuicao_blocos_celulas_quantis.csv"
SAIDA_PRED_BLOCOS = APOIO_DIR / "recorte_predicoes_atribuicao_blocos.csv"
SAIDA_ROBUSTEZ = APOIO_DIR / "recorte_robustez_formulacoes.csv"
SAIDA_PRED_ROBUSTEZ = APOIO_DIR / "recorte_predicoes_robustez_formulacoes.csv"
SAIDA_SENSIBILIDADE = APOIO_DIR / "recorte_sensibilidade_horario_largada.csv"
SAIDA_ESTABILIDADE = APOIO_DIR / "recorte_estabilidade_por_edicao.csv"
SAIDA_ESTABILIDADE_RESUMO = APOIO_DIR / "recorte_estabilidade_por_edicao_resumo.csv"
FIG_GRADE_RECORTE = FIG_DIR / "fig_recorte_grade_formulacoes.png"


# ---------------------------------------------------------------------------
# Corpus do recorte
# ---------------------------------------------------------------------------

def carregar_recorte() -> tuple[pd.DataFrame, list[str], list[str]]:
    df, cols_num, cols_cat = t12.carregar_dataset()
    df_r = df[df["maratona"].isin(RECORTE)].reset_index(drop=True)
    contagens = {k: int(v) for k, v in df_r["maratona"].value_counts().items()}
    if contagens != EDICOES_ESPERADAS:
        raise AssertionError(
            f"recorte inesperado: {contagens} (esperava {EDICOES_ESPERADAS})"
        )
    if len(df_r) != sum(EDICOES_ESPERADAS.values()):
        raise AssertionError(f"recorte com {len(df_r)} linhas; esperava 49")
    print(f"[recorte] {len(df_r)} edicoes: "
          + ", ".join(f"{m}={contagens[m]}" for m in RECORTE))
    return df_r, cols_num, cols_cat


# ---------------------------------------------------------------------------
# Verificacao 1: a agregacao reproduz os resultados gravados pelo script 13
# ---------------------------------------------------------------------------

def _agregados_de_predicoes(pred: pd.DataFrame) -> pd.DataFrame:
    """Recalcula a tabela de resultados celula a celula, com o mesmo codigo
    de metricas do script 13 (`_linha_resultado`)."""
    linhas: list[dict] = []
    for (formulacao, familia, alvo), g in pred.groupby(
        ["formulacao", "familia", "alvo"], sort=False
    ):
        linhas.append(t13._linha_resultado(
            formulacao, familia, alvo,
            g["real_min"].to_numpy(dtype=float),
            g["pred_min"].to_numpy(dtype=float),
            g["ref_min"].to_numpy(dtype=float),
        ))
    return pd.DataFrame(linhas)[t13.COLS_RESULT]


def validar_agregacao_contra_13(
    grade13: pd.DataFrame, pred13: pd.DataFrame
) -> None:
    rec = _agregados_de_predicoes(pred13)
    j = grade13.merge(
        rec, on=["formulacao", "familia", "alvo"],
        suffixes=("_pub", "_rec"), validate="1:1",
    )
    if len(j) != len(grade13):
        raise AssertionError(
            f"celulas nao pareadas na verificacao 1: {len(j)} vs {len(grade13)}"
        )
    max_diff = 0.0
    for col in ("mae_min", "rmse_min", "r2_abs", "r2_desvio"):
        d = float((j[f"{col}_pub"] - j[f"{col}_rec"]).abs().max())
        max_diff = max(max_diff, d)
        if d > TOL:
            raise AssertionError(
                f"verificacao 1 falhou: {col} diverge do resultado do script 13 (max {d:.2e})"
            )
    print(f"[valida] verificacao 1: agregacao reproduz o script 13 em {len(j)} celulas "
          f"(max diff {max_diff:.2e}) -> OK")


# ---------------------------------------------------------------------------
# Trilha b: exp 1 restrito as 49 edicoes do recorte
# ---------------------------------------------------------------------------

def exp1_restrito(
    pred13: pd.DataFrame, df_r: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chaves = set(zip(df_r["maratona"], df_r["ano"]))
    mask = np.array(
        [(m, a) in chaves for m, a in zip(pred13["maratona"], pred13["ano"])]
    )
    sel = pred13[mask].reset_index(drop=True)
    tamanhos = sel.groupby(["formulacao", "familia", "alvo"]).size()
    if set(tamanhos.unique()) != {len(df_r)}:
        raise AssertionError(
            "exp1 restrito sem cobertura uniforme de 49 edicoes por celula: "
            f"{sorted(tamanhos.unique())}"
        )
    res = _agregados_de_predicoes(sel)
    print(f"[exp1] restrito as {len(df_r)} edicoes do recorte: "
          f"{len(res)} celulas reagregadas")
    return res, sel


# ---------------------------------------------------------------------------
# Verificacao 2: a referencia historica e identica nas duas trilhas
# ---------------------------------------------------------------------------

def validar_baseline_identica(
    pred_exp1_r: pd.DataFrame, pred_recorte: pd.DataFrame, alvos: list[str]
) -> None:
    chave = ["alvo", "maratona", "ano"]
    filtro = (
        (pred_exp1_r["formulacao"] == FORMULACAO_REFERENCIA)
        & (pred_exp1_r["familia"] == "referencia_historica")
        & pred_exp1_r["alvo"].isin(alvos)
    )
    a = pred_exp1_r[filtro]
    b = pred_recorte[
        (pred_recorte["formulacao"] == FORMULACAO_REFERENCIA)
        & (pred_recorte["familia"] == "referencia_historica")
    ]
    j = a.merge(b, on=chave, suffixes=("_exp1", "_rec"), validate="1:1")
    esperado = b.groupby("alvo").size().sum()
    if len(j) != esperado:
        raise AssertionError(
            f"verificacao 2: pareamento incompleto ({len(j)} vs {esperado})"
        )
    max_diff = 0.0
    for col in ("real_min", "ref_min", "pred_min"):
        d = float((j[f"{col}_exp1"] - j[f"{col}_rec"]).abs().max())
        max_diff = max(max_diff, d)
        if d > TOL:
            raise AssertionError(
                f"verificacao 2 falhou: {col} difere entre LOO-117 e LOO-49 "
                f"(max {d:.2e}); a referencia historica NAO e identica"
            )
    print(f"[valida] verificacao 2: referencia por edicao identica nas duas "
          f"trilhas em {len(j)} pares (max diff {max_diff:.2e}) -> OK")


# ---------------------------------------------------------------------------
# Comparacao de tres vias (tabela principal da leitura)
# ---------------------------------------------------------------------------

def comparacao_tres_vias(
    grade13: pd.DataFrame,
    res_exp1_r: pd.DataFrame,
    grade_recorte: pd.DataFrame,
) -> pd.DataFrame:
    chave = ["formulacao", "familia", "alvo"]
    c117 = grade13[chave + ["mae_min", "rmse_min"]].rename(columns={
        "mae_min": "mae_corpus117", "rmse_min": "rmse_corpus117",
    })
    c1 = res_exp1_r[chave + ["mae_min", "rmse_min"]].rename(columns={
        "mae_min": "mae_exp1_49", "rmse_min": "rmse_exp1_49",
    })
    c2 = grade_recorte[chave + ["mae_min", "rmse_min"]].rename(columns={
        "mae_min": "mae_recorte_49", "rmse_min": "rmse_recorte_49",
    })
    comp = c117.merge(c1, on=chave, validate="1:1").merge(
        c2, on=chave, validate="1:1"
    )

    ref = comp[comp["familia"] == "referencia_historica"]
    mae_ref = dict(zip(ref["alvo"], ref["mae_exp1_49"]))
    # a referencia e identica nas duas trilhas (verificacao 2); um unico
    # vetor de referencia serve para os dois ganhos
    comp["mae_ref_49"] = comp["alvo"].map(mae_ref)
    comp["delta_recorte_vs_exp1"] = comp["mae_recorte_49"] - comp["mae_exp1_49"]
    comp["ganho_exp1_vs_ref"] = comp["mae_ref_49"] - comp["mae_exp1_49"]
    comp["ganho_recorte_vs_ref"] = comp["mae_ref_49"] - comp["mae_recorte_49"]
    return comp


# ---------------------------------------------------------------------------
# Relatorio: respostas as tres perguntas do recorte
# ---------------------------------------------------------------------------

def relatorio_adjudicacao(comp: pd.DataFrame, alvos: list[str]) -> None:
    print()
    print("=" * 72)
    print("RESULTADO (criterio: MAE nas mesmas 49 edicoes)")
    print("=" * 72)

    gbm = comp[
        (comp["formulacao"] == FORMULACAO_DESVIO)
        & (comp["familia"] == "gbm_histogramas")
    ].set_index("alvo")
    print()
    print("--- 1. Tabela de tres vias (gbm em desvio, configuracao principal) ---")
    tab = gbm.loc[[a for a in alvos if a in gbm.index],
                  ["mae_ref_49", "mae_exp1_49", "mae_recorte_49",
                   "delta_recorte_vs_exp1", "ganho_recorte_vs_ref"]]
    tab.loc["media"] = tab.mean()
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    print("--- 2. MAE medio nos alvos, por formulacao x familia (3 vias) ---")
    ag = (
        comp.groupby(["formulacao", "familia"])[
            ["mae_corpus117", "mae_exp1_49", "mae_recorte_49"]
        ].mean().sort_values("mae_recorte_49")
    )
    print(ag.to_string(float_format=lambda v: f"{v:.4f}"))

    corpo = comp[comp["formulacao"] == FORMULACAO_DESVIO]
    ranking = (
        corpo.groupby("familia")["mae_recorte_49"].mean().sort_values()
    )
    melhor = str(ranking.index[0])
    arvores = {"floresta_aleatoria", "gbm_histogramas"}
    print()
    print("--- 3. Perguntas do recorte ---")
    print(f"  (i) melhor familia em desvio no recorte: {melhor} "
          f"({'ARVORE' if melhor in arvores else 'NAO-ARVORE'})")

    ganho_exp1 = float(gbm["ganho_exp1_vs_ref"].mean())
    ganho_rec = float(gbm["ganho_recorte_vs_ref"].mean())
    print(f"  (ii) ganho medio vs referencia nas 49: exp1 {ganho_exp1:+.3f} min; "
          f"recorte {ganho_rec:+.3f} min")

    delta = float(gbm["delta_recorte_vs_exp1"].mean())
    sinais = gbm["delta_recorte_vs_exp1"]
    consistencia = int((np.sign(sinais) == np.sign(delta)).sum())
    if delta > 0:
        ramo = ("PIORA no recorte: treinar so com as 49 aumenta o MAE - "
                "a heterogeneidade das 117 transfere sinal entre provas")
    elif delta < 0:
        ramo = ("MELHORA no recorte: problema mais uniforme compensa menos "
                "dados - disputa de contextos no corpus heterogeneo")
    else:
        ramo = "IGUALA: sem efeito mensuravel do corpus de treino"
    print(f"  (iii) delta medio recorte - exp1 (gbm em desvio): {delta:+.3f} min; "
          f"mesmo sinal em {consistencia}/{len(sinais)} alvos")
    print(f"  ramo: {ramo}")


def figura_grade_recorte(grade: pd.DataFrame) -> None:
    ordem_alvos = [a for a in ALVOS if a in set(grade["alvo"])]
    corpo = grade[grade["formulacao"].isin([FORMULACAO_ABSOLUTO, FORMULACAO_DESVIO])]
    b2 = grade[grade["familia"] == "referencia_historica"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, formulacao in zip(axes, (FORMULACAO_ABSOLUTO, FORMULACAO_DESVIO)):
        g = corpo[corpo["formulacao"] == formulacao]
        sns.lineplot(
            data=g, x="alvo", y="mae_min", hue="familia",
            marker="o", ax=ax, sort=False,
            hue_order=sorted(g["familia"].unique()),
        )
        ax.plot(
            ordem_alvos,
            [float(b2[b2["alvo"] == a]["mae_min"].iloc[0]) for a in ordem_alvos],
            "k--", label="referencia historica",
        )
        ax.set_title(f"recorte 40-43 N - formulacao: {formulacao}")
        ax.set_xlabel("alvo")
        ax.set_ylabel("MAE (min)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_GRADE_RECORTE, dpi=150)
    plt.close(fig)
    print(f"[fig] {FIG_GRADE_RECORTE}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="so ridge+gbm e p50; nao grava artefatos")
    args = parser.parse_args()

    inicio = time.time()
    df_r, cols_num, cols_cat = carregar_recorte()

    grade13 = pd.read_csv(t13.SAIDA_GRADE)
    pred13 = pd.read_csv(t13.SAIDA_PRED_GRADE)
    validar_agregacao_contra_13(grade13, pred13)
    res_exp1_r, pred_exp1_r = exp1_restrito(pred13, df_r)

    if args.smoke:
        familias = {k: FAMILIAS[k] for k in ("ridge", "gbm_histogramas")}
        alvos = ["p50"]
        print("[modo] SMOKE: ridge+gbm, p50, sem gravacao de artefatos")
    else:
        familias = dict(FAMILIAS)
        alvos = list(ALVOS)

    familias_blocos = {k: FAMILIAS[k] for k in ("ridge", "gbm_histogramas")}

    grade_r, pred_r = t13.grade_formulacoes(df_r, cols_num, cols_cat,
                                            familias, alvos)
    t13.validar_cobertura_grade(df_r, grade_r, pred_r, alvos, familias)
    validar_baseline_identica(pred_exp1_r, pred_r, alvos)

    comp = comparacao_tres_vias(grade13, res_exp1_r, grade_r)

    blocos_res, decomposicao, pred_blocos = t13.atribuicao_blocos(
        df_r, cols_num, pred_r, familias_blocos, alvos
    )

    rob, pred_rob = t13.robustez(df_r, cols_num, cols_cat, alvos)
    print("[nota] LOMO no recorte: a prova excluida fica sem historico "
          "proprio e a referencia cai no centro global das outras duas; "
          "significado distinto do LOMO com as 11 provas do corpus completo.")

    sens = t13.sensibilidade_horario(df_r, cols_num, cols_cat,
                                     grade_r, pred_r, alvos)
    estab_det, estab_res = t13.estabilidade_por_edicao(
        pred_r, alvos, list(familias_blocos)
    )

    relatorio_adjudicacao(comp, alvos)

    if args.smoke:
        print()
        print(f"[ok] smoke concluido em {time.time() - inicio:.0f}s; nada gravado")
        return

    grade_r.to_csv(SAIDA_GRADE, index=False)
    pred_r.to_csv(SAIDA_PRED_GRADE, index=False)
    res_exp1_r.to_csv(SAIDA_EXP1_RESTRITO, index=False)
    comp.to_csv(SAIDA_COMPARACAO, index=False)
    decomposicao.to_csv(SAIDA_BLOCOS, index=False)
    blocos_res.to_csv(SAIDA_BLOCOS_CELULAS, index=False)
    pred_blocos.to_csv(SAIDA_PRED_BLOCOS, index=False)
    rob.to_csv(SAIDA_ROBUSTEZ, index=False)
    pred_rob.to_csv(SAIDA_PRED_ROBUSTEZ, index=False)
    sens.to_csv(SAIDA_SENSIBILIDADE, index=False)
    estab_det.to_csv(SAIDA_ESTABILIDADE, index=False)
    estab_res.to_csv(SAIDA_ESTABILIDADE_RESUMO, index=False)
    print()
    for p in (SAIDA_GRADE, SAIDA_PRED_GRADE, SAIDA_EXP1_RESTRITO,
              SAIDA_COMPARACAO, SAIDA_BLOCOS, SAIDA_BLOCOS_CELULAS,
              SAIDA_PRED_BLOCOS, SAIDA_ROBUSTEZ, SAIDA_PRED_ROBUSTEZ,
              SAIDA_SENSIBILIDADE, SAIDA_ESTABILIDADE,
              SAIDA_ESTABILIDADE_RESUMO):
        print(f"[csv] {p}")
    figura_grade_recorte(grade_r)

    print()
    print(f"[ok] bateria do recorte concluida em "
          f"{(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
