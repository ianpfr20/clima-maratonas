"""
Comparacao de desenhos de treinamento para os quantis P10-P90 por edicao.

Complementa o script 12 (12_treino_quantis_edicao.py): reutiliza a mesma
maquinaria (fabricas de modelos, pipeline, referencia historica, folds de
validacao com uma edicao reservada por vez em ordem de linha) importando o
modulo 12. Toda celula que tambem existe nos artefatos do 12 precisa
reproduzir exatamente os numeros gravados por ele; se a reproducao falhar,
nada e gravado.

Entradas:
  tabela de treino por edicao (dataset117), lida pelo loader do script 12
  dados/processados/apoio/resultados_treino_quantis.csv e
  dados/processados/apoio/predicoes_loo_quantis.csv   saidas do script 12
  dados/processados/apoio/horarios_largada.csv   horario de largada por edicao

O que se computa (o centro da referencia historica e sempre a mediana
historica do mesmo alvo):

1. GRADE COMPLETA formulacao x familia: as cinco familias (linear, Ridge,
   floresta aleatoria, gradient boosting por histogramas, perceptron
   multicamadas) nas DUAS formulacoes:
     absoluto  y                          features 36 numericas + indicadora;
     desvio    y - referencia_historica   features 36 numericas, SEM
               indicadora; predicao reconstruida somando a referencia do
               fold antes de qualquer metrica.
   Linhas de base: media global do treino e referencia historica (equivale
   a prever desvio zero). O script 12 cobre a formulacao em desvio apenas
   para Ridge e GBM; aqui as cinco familias rodam nas duas formulacoes.
   Metricas: MAE (primaria), RMSE, R2 na escala absoluta e R2 na escala do
   DESVIO. Na escala absoluta a variancia entre maratonas domina (a
   referencia historica tem R2 0,92-0,94); na escala do desvio a referencia
   preve desvio zero, mas seu R2 nao e necessariamente zero, porque o R2
   centraliza os desvios observados pela sua media. MAE e RMSE sao identicos
   nas duas escalas (a reconstrucao soma a mesma referencia aos dois lados).

2. ATRIBUICAO POR BLOCO: na formulacao em desvio, cruzamento 2x2 mantendo
   sempre o bloco temporal E = [ano, dia_ano_sin, dia_ano_cos]:
     E, E+C, E+S, E+C+S com C = 31 climaticas e S = [pct_M, ln_n_finishers].
   E+C+S nao e reajustada: reutiliza a celula da grade (mesmos folds), o que
   e verificado por identidade de colunas. Decomposicao em efeito do clima,
   efeito da composicao e interacao, como na ablacao 2x2 do script 12. O
   desenho fatorial evita bracos sequenciais, que tornariam a atribuicao
   dependente da ordem de inclusao dos blocos.

3. ROBUSTEZ NAS DUAS FORMULACOES: exclusao de maratona inteira (LOMO, 11
   maratonas) e corte temporal (treino ano<=2017, teste ano>=2018), nos
   cinco alvos e tambem para as variantes em desvio (a robustez do script 12
   cobre so o alvo absoluto em P50). No LOMO a maratona de teste nao tem
   historico no treino e a referencia cai no fallback global: e o teste de
   transferencia do sinal climatico para uma prova nova. As predicoes por
   edicao sao exportadas para conferencia edicao a edicao.

4. SENSIBILIDADE AO HORARIO DE LARGADA: edicoes cujo horario desvia >= 60
   minutos do horario modal da propria prova no corpus (regra calculada a
   partir de dados/processados/apoio/horarios_largada.csv) sao removidas e a
   validacao com uma edicao reservada por vez roda de novo com a referencia
   historica, Ridge e GBM nas duas formulacoes. Tambem se reporta o erro das
   edicoes excluidas na grade completa.

5. ESTABILIDADE POR EDICAO: para Ridge e GBM em desvio contra a referencia
   historica, delta de erro absoluto por edicao: fracao de edicoes que
   melhora, concentracao do ganho (ganho recalculado sem as 5 edicoes que
   mais melhoram) e media por maratona. Verifica se o ganho medio depende
   de poucas edicoes entre as 117.

Criterios de comparacao (definidos antes de examinar os resultados):
  1. primario: MAE medio nos cinco alvos, por (formulacao, familia);
  2. secundario: vitorias por alvo, RMSE e R2 na escala do desvio;
  3. robustez: o ranking se mantem no LOMO e no corte temporal?
  4. estabilidade: o ganho aparece na maioria das edicoes ou em poucas?
  5. sensibilidade: as leituras se mantem sem as edicoes de horario atipico?

Saidas (dados/processados/apoio/):
  grade_formulacoes_quantis.csv              resultados da grade (item 1)
  predicoes_grade_formulacoes.csv            predicoes por edicao, com ref_min
  atribuicao_blocos_quantis.csv              decomposicao 2x2 por bloco (item 2)
  atribuicao_blocos_celulas_quantis.csv      metricas de cada celula do 2x2
  predicoes_atribuicao_blocos.csv            predicoes por edicao das celulas novas
  robustez_formulacoes.csv                   LOMO + corte temporal (item 3)
  predicoes_robustez_formulacoes.csv         predicoes por edicao da robustez
  sensibilidade_horario_largada.csv          item 4
  estabilidade_por_edicao.csv                item 5, detalhe por edicao
  estabilidade_por_edicao_resumo.csv         item 5, resumo por familia e alvo
  codigo/cap4/figuras/fig_grade_formulacoes.png

Execucao, a partir da raiz do repositorio:
    python codigo/cap4/notebooks/13_desenho_treino_quantis.py
Modo rapido de verificacao (so Ridge+GBM, so P50, NAO grava artefatos):
    python codigo/cap4/notebooks/13_desenho_treino_quantis.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

# Uma thread por biblioteca numerica, como no script 12, para resultados
# reprodutiveis. Precisa vir antes de importar numpy/sklearn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_AQUI = Path(__file__).resolve().parent
_CAMINHO_12 = _AQUI / "12_treino_quantis_edicao.py"


def _carregar_modulo_12():
    spec = importlib.util.spec_from_file_location("treino12", _CAMINHO_12)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # registrar ANTES de executar: o @dataclass do 12 resolve anotacoes
    # adiadas via sys.modules[cls.__module__] e falha se o nome nao existir
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


t12 = _carregar_modulo_12()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

RAIZ = t12.RAIZ
APOIO_DIR = t12.APOIO_DIR
FIG_DIR = t12.FIG_DIR
ALVOS = list(t12.ALVOS)
CENTRO = t12.CENTRO_PRIMARIO  # mediana historica do mesmo alvo

HORARIOS = APOIO_DIR / "horarios_largada.csv"
LIMIAR_HORARIO_MIN = 60  # desvio do horario modal da prova que marca edicao atipica

SAIDA_GRADE = APOIO_DIR / "grade_formulacoes_quantis.csv"
SAIDA_PRED_GRADE = APOIO_DIR / "predicoes_grade_formulacoes.csv"
SAIDA_BLOCOS = APOIO_DIR / "atribuicao_blocos_quantis.csv"
SAIDA_PRED_BLOCOS = APOIO_DIR / "predicoes_atribuicao_blocos.csv"
SAIDA_ROBUSTEZ = APOIO_DIR / "robustez_formulacoes.csv"
SAIDA_PRED_ROBUSTEZ = APOIO_DIR / "predicoes_robustez_formulacoes.csv"
SAIDA_SENSIBILIDADE = APOIO_DIR / "sensibilidade_horario_largada.csv"
SAIDA_ESTABILIDADE = APOIO_DIR / "estabilidade_por_edicao.csv"
FIG_GRADE = FIG_DIR / "fig_grade_formulacoes.png"

ARTEFATOS_12_RESULTADOS = APOIO_DIR / "resultados_treino_quantis.csv"
ARTEFATOS_12_PREDICOES = APOIO_DIR / "predicoes_loo_quantis.csv"

# familia -> chave da fabrica de modelos no script 12
FAMILIAS = {
    "linear": "M1_linear",
    "ridge": "M2_ridge",
    "floresta_aleatoria": "M3_random_forest",
    "gbm_histogramas": "M4_hist_gbm",
    "perceptron_multicamadas": "M5_mlp",
}

# celulas ja calculadas pelo script 12 que a grade precisa reproduzir exatamente
EQUIVALENCIA_12_ABS = {fam: nome for fam, nome in FAMILIAS.items()}
EQUIVALENCIA_12_DEV = {
    "ridge": "M2a_ridge_anom",
    "gbm_histogramas": "M4a_hist_gbm_anom",
}

BLOCO_E = ["ano", "dia_ano_sin", "dia_ano_cos"]
BLOCO_S = ["pct_M", "ln_n_finishers"]

FORMULACAO_REFERENCIA = "referencia"
FORMULACAO_ABSOLUTO = "absoluto"
FORMULACAO_DESVIO = "desvio"

COLS_RESULT = [
    "formulacao", "familia", "alvo", "mae_min", "rmse_min", "r2_abs", "r2_desvio",
]
COLS_PRED = [
    "maratona", "ano", "alvo", "formulacao", "familia",
    "real_min", "pred_min", "ref_min",
]

TOL = 1e-9


# ---------------------------------------------------------------------------
# Nucleo compartilhado
# ---------------------------------------------------------------------------

def _r2_desvio(y_real: np.ndarray, y_pred: np.ndarray, refs: np.ndarray) -> float:
    return float(r2_score(y_real - refs, y_pred - refs))


def _linha_resultado(formulacao, familia, alvo, y_real, y_pred, refs) -> dict:
    m = t12.metricas(y_real, y_pred)
    return {
        "formulacao": formulacao,
        "familia": familia,
        "alvo": alvo,
        "mae_min": m["mae_min"],
        "rmse_min": m["rmse_min"],
        "r2_abs": m["r2"],
        "r2_desvio": _r2_desvio(y_real, y_pred, refs),
    }


def _linhas_predicao(df, alvo, formulacao, familia, y, pr, refs) -> list[dict]:
    return [
        {
            "maratona": df["maratona"].iloc[j],
            "ano": int(df["ano"].iloc[j]),
            "alvo": alvo,
            "formulacao": formulacao,
            "familia": familia,
            "real_min": float(y.iloc[j]),
            "pred_min": float(pr[j]),
            "ref_min": float(refs[j]),
        }
        for j in range(len(y))
    ]


# ---------------------------------------------------------------------------
# 1. Grade formulacao x familia (LOO)
# ---------------------------------------------------------------------------

def grade_formulacoes(
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    familias: dict[str, str],
    alvos: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print(f"=== GRADE formulacao x familia (LOO; centro={CENTRO}; "
          f"{len(familias)} familias x 2 formulacoes x {len(alvos)} alvos) ===")
    loo = LeaveOneOut()
    fabricas = t12.fabricas_modelos()
    X_com = df[cols_num + cols_cat]
    X_sem = df[cols_num]
    mar = df["maratona"]

    resultados: list[dict] = []
    predicoes: list[dict] = []

    for alvo in alvos:
        y = df[alvo]
        t0 = time.time()
        preds: dict[tuple[str, str], np.ndarray] = {
            (FORMULACAO_ABSOLUTO, fam): np.zeros(len(y)) for fam in familias
        }
        preds.update({
            (FORMULACAO_DESVIO, fam): np.zeros(len(y)) for fam in familias
        })
        preds_b1 = np.zeros(len(y))
        refs = np.zeros(len(y))

        for fold, (tr, te) in enumerate(loo.split(X_com), start=1):
            i = int(te[0])
            y_tr = y.iloc[tr]
            mar_tr = mar.iloc[tr]
            preds_b1[i] = float(y_tr.mean())
            referencia = t12.ajustar_referencia_historica(y_tr, mar_tr, CENTRO)
            y_anom_tr = y_tr.values - referencia.prever(mar_tr)
            centro_teste = float(referencia.prever([mar.iloc[i]])[0])
            refs[i] = centro_teste

            for fam, nome12 in familias.items():
                pipe = t12.montar_pipeline(fabricas[nome12](), cols_num, cols_cat)
                pipe.fit(X_com.iloc[tr], y_tr)
                preds[(FORMULACAO_ABSOLUTO, fam)][i] = float(
                    pipe.predict(X_com.iloc[[i]])[0]
                )

                pipe_dev = t12.montar_pipeline(fabricas[nome12](), cols_num, [])
                pipe_dev.fit(X_sem.iloc[tr], y_anom_tr)
                pred_anom = float(pipe_dev.predict(X_sem.iloc[[i]])[0])
                preds[(FORMULACAO_DESVIO, fam)][i] = float(
                    t12.reconstruir_predicao_absoluta(pred_anom, centro_teste).item()
                )

            if fold % 25 == 0 or fold == len(y):
                print(f"[grade] {alvo}: fold {fold}/{len(y)}", flush=True)

        # linhas de base: media global e a propria referencia do fold
        # (referencia historica = prever desvio zero)
        todos: dict[tuple[str, str], np.ndarray] = {
            (FORMULACAO_REFERENCIA, "media_global"): preds_b1,
            (FORMULACAO_REFERENCIA, "referencia_historica"): refs.copy(),
            **preds,
        }
        for (formulacao, fam), pr in todos.items():
            resultados.append(
                _linha_resultado(formulacao, fam, alvo, y.values, pr, refs)
            )
            predicoes.extend(
                _linhas_predicao(df, alvo, formulacao, fam, y, pr, refs)
            )
        print(f"[grade] {alvo} concluido em {time.time() - t0:.1f}s", flush=True)

    return (
        pd.DataFrame(resultados)[COLS_RESULT],
        pd.DataFrame(predicoes)[COLS_PRED],
    )


# ---------------------------------------------------------------------------
# Validacao de igualdade contra os artefatos do 12
# ---------------------------------------------------------------------------

def validar_grade_contra_12(
    grade: pd.DataFrame,
    pred_grade: pd.DataFrame,
    alvos: list[str],
    familias: dict[str, str],
) -> None:
    """Confere que as celulas tambem calculadas pelo script 12 reproduzem os valores gravados por ele."""
    res12 = pd.read_csv(ARTEFATOS_12_RESULTADOS)
    pred12 = pd.read_csv(ARTEFATOS_12_PREDICOES)

    # (formulacao, familia, modelo_12, protocolo_12) - o protocolo evita
    # colisao com as linhas C_LOMO/C_TEMPORAL dos mesmos modelos
    pares: list[tuple[str, str, str, str]] = []
    pares.append((FORMULACAO_REFERENCIA, "media_global", "B1_global", "A"))
    pares.append(
        (FORMULACAO_REFERENCIA, "referencia_historica", "B2_por_maratona", "A")
    )
    for fam in familias:
        pares.append((FORMULACAO_ABSOLUTO, fam, EQUIVALENCIA_12_ABS[fam], "A"))
        if fam in EQUIVALENCIA_12_DEV:
            pares.append((FORMULACAO_DESVIO, fam, EQUIVALENCIA_12_DEV[fam], "B"))

    max_diff_res = 0.0
    max_diff_pred = 0.0
    celulas = 0
    for formulacao, fam, modelo12, proto12 in pares:
        for alvo in alvos:
            meu = grade[
                (grade["formulacao"] == formulacao)
                & (grade["familia"] == fam)
                & (grade["alvo"] == alvo)
            ]
            deles = res12[
                (res12["protocolo"] == proto12)
                & (res12["modelo"] == modelo12)
                & (res12["alvo"] == alvo)
            ]
            if len(deles) == 0:
                raise AssertionError(f"celula do 12 ausente: {modelo12} {alvo}")
            if len(meu) != 1 or len(deles) != 1:
                raise AssertionError(
                    f"celula duplicada: {formulacao}/{fam}/{alvo} "
                    f"({len(meu)} minha, {len(deles)} no 12)"
                )
            for minha_col, col12 in (
                ("mae_min", "mae_min"), ("rmse_min", "rmse_min"), ("r2_abs", "r2"),
            ):
                d = abs(float(meu.iloc[0][minha_col]) - float(deles.iloc[0][col12]))
                max_diff_res = max(max_diff_res, d)
                if d > TOL:
                    raise AssertionError(
                        f"grade diverge do 12 em {modelo12} {alvo} {col12}: "
                        f"{meu.iloc[0][minha_col]!r} vs {deles.iloc[0][col12]!r}"
                    )

            minha_pred = pred_grade[
                (pred_grade["formulacao"] == formulacao)
                & (pred_grade["familia"] == fam)
                & (pred_grade["alvo"] == alvo)
            ].set_index(["maratona", "ano"])["pred_min"]
            deles_pred = pred12[
                (pred12["modelo"] == modelo12) & (pred12["alvo"] == alvo)
            ].set_index(["maratona", "ano"])["pred_min"]
            junto = pd.concat(
                [minha_pred.rename("minha"), deles_pred.rename("deles")],
                axis=1, join="inner",
            )
            if len(junto) != len(minha_pred) or len(junto) != len(deles_pred):
                raise AssertionError(
                    f"predicoes desalinhadas em {modelo12} {alvo}: "
                    f"{len(minha_pred)} vs {len(deles_pred)} (interseccao {len(junto)})"
                )
            d = float((junto["minha"] - junto["deles"]).abs().max())
            max_diff_pred = max(max_diff_pred, d)
            if d > TOL:
                raise AssertionError(
                    f"predicoes divergem do 12 em {modelo12} {alvo}: max |diff| = {d}"
                )
            celulas += 1

    print(f"[valida] grade reproduz o 12 em {celulas} celulas "
          f"(max diff resultados {max_diff_res:.2e}, predicoes {max_diff_pred:.2e}) -> OK")


def validar_cobertura_grade(
    df: pd.DataFrame, grade: pd.DataFrame, pred_grade: pd.DataFrame,
    alvos: list[str], familias: dict[str, str],
) -> None:
    n = len(df)
    esperado_celulas = (2 + 2 * len(familias)) * len(alvos)
    if len(grade) != esperado_celulas:
        raise AssertionError(
            f"grade tem {len(grade)} linhas; esperava {esperado_celulas}"
        )
    grupos = pred_grade.groupby(["formulacao", "familia", "alvo"], sort=False)
    for chave, g in grupos:
        if len(g) != n or g.duplicated(["maratona", "ano"]).any():
            raise AssertionError(f"cobertura de predicao invalida em {chave}")
    # a referencia por edicao e a mesma em toda celula do mesmo alvo
    ref_por_alvo = pred_grade.groupby(["alvo", "maratona", "ano"])["ref_min"].nunique()
    if int(ref_por_alvo.max()) != 1:
        raise AssertionError("ref_min difere entre celulas do mesmo alvo/edicao")
    # MAE recomputado das predicoes tem que bater com a tabela de resultados
    err = (pred_grade["real_min"] - pred_grade["pred_min"]).abs()
    mae_rec = (
        pred_grade.assign(ae=err)
        .groupby(["formulacao", "familia", "alvo"], sort=False)["ae"].mean()
    )
    for _, linha in grade.iterrows():
        chave = (linha["formulacao"], linha["familia"], linha["alvo"])
        if abs(float(mae_rec.loc[chave]) - float(linha["mae_min"])) > TOL:
            raise AssertionError(f"MAE da tabela nao bate com predicoes em {chave}")
    print("[valida] cobertura da grade (celulas, unicidade, ref unica, MAE) -> OK")


# ---------------------------------------------------------------------------
# 2. Atribuicao por bloco (E / E+C / E+S / E+C+S), formulacao em desvio
# ---------------------------------------------------------------------------

def _blocos_de(cols_num: list[str]) -> dict[str, list[str]]:
    resto = [c for c in cols_num if c not in BLOCO_E + BLOCO_S]
    return {"E": list(BLOCO_E), "C": resto, "S": list(BLOCO_S)}


def atribuicao_blocos(
    df: pd.DataFrame,
    cols_num: list[str],
    pred_grade: pd.DataFrame,
    familias_blocos: dict[str, str],
    alvos: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocos = _blocos_de(cols_num)
    print()
    print(f"=== ATRIBUICAO POR BLOCO (desvio; E={len(blocos['E'])}, "
          f"C={len(blocos['C'])}, S={len(blocos['S'])} colunas) ===")

    celulas = {
        "E": blocos["E"],
        "E+C": [c for c in cols_num if c in set(blocos["E"] + blocos["C"])],
        "E+S": [c for c in cols_num if c in set(blocos["E"] + blocos["S"])],
    }
    completo = [c for c in cols_num if c in set(blocos["E"] + blocos["C"] + blocos["S"])]
    if completo != list(cols_num):
        raise AssertionError("E+C+S nao reconstitui cols_num; blocos mal definidos")

    loo = LeaveOneOut()
    fabricas = t12.fabricas_modelos()
    mar = df["maratona"]
    resultados: list[dict] = []
    predicoes: list[dict] = []
    mae: dict[tuple[str, str, str], float] = {}

    for alvo in alvos:
        y = df[alvo]
        refs = np.zeros(len(y))
        preds: dict[tuple[str, str], np.ndarray] = {
            (fam, cel): np.zeros(len(y))
            for fam in familias_blocos for cel in celulas
        }
        for fold, (tr, te) in enumerate(loo.split(df), start=1):
            i = int(te[0])
            y_tr = y.iloc[tr]
            mar_tr = mar.iloc[tr]
            referencia = t12.ajustar_referencia_historica(y_tr, mar_tr, CENTRO)
            y_anom_tr = y_tr.values - referencia.prever(mar_tr)
            centro_teste = float(referencia.prever([mar.iloc[i]])[0])
            refs[i] = centro_teste
            for fam, nome12 in familias_blocos.items():
                for cel, cols in celulas.items():
                    X = df[cols]
                    pipe = t12.montar_pipeline(fabricas[nome12](), cols, [])
                    pipe.fit(X.iloc[tr], y_anom_tr)
                    pred_anom = float(pipe.predict(X.iloc[[i]])[0])
                    preds[(fam, cel)][i] = pred_anom + centro_teste
            if fold % 50 == 0 or fold == len(y):
                print(f"[blocos] {alvo}: fold {fold}/{len(y)}", flush=True)

        for (fam, cel), pr in preds.items():
            m = t12.metricas(y.values, pr)
            mae[(fam, cel, alvo)] = m["mae_min"]
            resultados.append({
                "familia": fam, "celula": cel, "alvo": alvo,
                "mae_min": m["mae_min"], "rmse_min": m["rmse_min"],
                "r2_abs": m["r2"],
                "r2_desvio": _r2_desvio(y.values, pr, refs),
            })
            predicoes.extend([
                {
                    "maratona": df["maratona"].iloc[j],
                    "ano": int(df["ano"].iloc[j]),
                    "alvo": alvo, "familia": fam, "celula": cel,
                    "real_min": float(y.iloc[j]),
                    "pred_min": float(pr[j]),
                    "ref_min": float(refs[j]),
                }
                for j in range(len(y))
            ])

    # celula E+C+S vem da grade (mesmos folds, mesma fabrica, mesmas colunas)
    for fam in familias_blocos:
        sel = pred_grade[
            (pred_grade["formulacao"] == FORMULACAO_DESVIO)
            & (pred_grade["familia"] == fam)
        ]
        for alvo in alvos:
            g = sel[sel["alvo"] == alvo]
            m = t12.metricas(g["real_min"].values, g["pred_min"].values)
            mae[(fam, "E+C+S", alvo)] = m["mae_min"]
            resultados.append({
                "familia": fam, "celula": "E+C+S", "alvo": alvo,
                "mae_min": m["mae_min"], "rmse_min": m["rmse_min"],
                "r2_abs": m["r2"],
                "r2_desvio": _r2_desvio(
                    g["real_min"].values, g["pred_min"].values, g["ref_min"].values
                ),
            })

    res = pd.DataFrame(resultados)

    # decomposicao 2x2 por (familia, alvo)
    linhas: list[dict] = []
    for fam in familias_blocos:
        for alvo in alvos:
            m_e = mae[(fam, "E", alvo)]
            m_ec = mae[(fam, "E+C", alvo)]
            m_es = mae[(fam, "E+S", alvo)]
            m_ecs = mae[(fam, "E+C+S", alvo)]
            clima_sem_comp = m_e - m_ec
            clima_com_comp = m_es - m_ecs
            comp_sem_clima = m_e - m_es
            comp_com_clima = m_ec - m_ecs
            interacao = clima_sem_comp - clima_com_comp
            check = comp_sem_clima - comp_com_clima
            if abs(interacao - check) > 1e-12:
                raise AssertionError(
                    f"decomposicao 2x2 inconsistente em {fam}/{alvo}"
                )
            linhas.append({
                "familia": fam, "alvo": alvo,
                "mae_E": m_e, "mae_EC": m_ec, "mae_ES": m_es, "mae_ECS": m_ecs,
                "efeito_clima_sem_composicao": clima_sem_comp,
                "efeito_clima_com_composicao": clima_com_comp,
                "efeito_composicao_sem_clima": comp_sem_clima,
                "efeito_composicao_com_clima": comp_com_clima,
                "interacao": interacao,
            })
    decomposicao = pd.DataFrame(linhas)
    return res, decomposicao, pd.DataFrame(predicoes)


# ---------------------------------------------------------------------------
# 3. Robustez: LOMO + corte temporal, duas formulacoes, cinco alvos
# ---------------------------------------------------------------------------

def _prever_particao(
    nome: str,
    fabricas: dict,
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    tr_mask: np.ndarray,
    te_mask: np.ndarray,
    alvo: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicao e referencia (por edicao de teste) para uma particao."""
    y = df[alvo]
    mar = df["maratona"]
    y_tr = y[tr_mask]
    mar_tr = mar[tr_mask]
    mar_te = mar[te_mask].values
    referencia = t12.ajustar_referencia_historica(y_tr, mar_tr, CENTRO)
    refs_te = referencia.prever(mar_te)

    if nome == "media_global":
        return np.full(int(te_mask.sum()), float(y_tr.mean())), refs_te
    if nome == "referencia_historica":
        return refs_te.copy(), refs_te

    familia, formulacao = nome.rsplit("_", 1)
    fab = fabricas[FAMILIAS[familia]]
    if formulacao == "abs":
        X_com = df[cols_num + cols_cat]
        pipe = t12.montar_pipeline(fab(), cols_num, cols_cat)
        pipe.fit(X_com[tr_mask], y_tr)
        return pipe.predict(X_com[te_mask]), refs_te
    if formulacao == "dev":
        X_sem = df[cols_num]
        y_anom_tr = y_tr.values - referencia.prever(mar_tr)
        pipe = t12.montar_pipeline(fab(), cols_num, [])
        pipe.fit(X_sem[tr_mask], y_anom_tr)
        return pipe.predict(X_sem[te_mask]) + refs_te, refs_te
    raise ValueError(f"modelo desconhecido: {nome}")


MODELOS_ROBUSTEZ = [
    "media_global", "referencia_historica",
    "ridge_abs", "gbm_histogramas_abs",
    "ridge_dev", "gbm_histogramas_dev",
]


def robustez(
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    alvos: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("=== ROBUSTEZ (LOMO + corte temporal; duas formulacoes; "
          f"{len(alvos)} alvos) ===")
    fabricas = t12.fabricas_modelos()
    mar = df["maratona"]
    resultados: list[dict] = []
    predicoes: list[dict] = []

    def registrar(protocolo, alvo, nome, te_mask, pr, refs_te):
        y_te = df[alvo][te_mask]
        m = t12.metricas(y_te.values, pr)
        resultados.append({
            "protocolo": protocolo, "alvo": alvo, "modelo": nome,
            "mae_min": m["mae_min"], "rmse_min": m["rmse_min"],
            "r2_abs": m["r2"],
            "r2_desvio": _r2_desvio(y_te.values, np.asarray(pr, float), refs_te),
        })
        sub = df[te_mask]
        predicoes.extend([
            {
                "protocolo": protocolo,
                "maratona": sub["maratona"].iloc[j],
                "ano": int(sub["ano"].iloc[j]),
                "alvo": alvo, "modelo": nome,
                "real_min": float(y_te.iloc[j]),
                "pred_min": float(pr[j]),
                "ref_min": float(refs_te[j]),
            }
            for j in range(len(y_te))
        ])

    # LOMO
    maratonas = sorted(mar.unique())
    print(f"[rob] LOMO: {len(maratonas)} maratonas")
    for alvo in alvos:
        preds_lomo = {nome: np.full(len(df), np.nan) for nome in MODELOS_ROBUSTEZ}
        refs_lomo = np.full(len(df), np.nan)
        for m_out in maratonas:
            te_mask = (mar == m_out).values
            tr_mask = ~te_mask
            for nome in MODELOS_ROBUSTEZ:
                pr, refs_te = _prever_particao(
                    nome, fabricas, df, cols_num, cols_cat, tr_mask, te_mask, alvo
                )
                preds_lomo[nome][te_mask] = pr
                refs_lomo[te_mask] = refs_te
        for nome in MODELOS_ROBUSTEZ:
            if np.isnan(preds_lomo[nome]).any():
                raise AssertionError(f"LOMO sem cobertura completa em {nome}/{alvo}")
            registrar(
                "LOMO", alvo, nome, np.ones(len(df), dtype=bool),
                preds_lomo[nome], refs_lomo,
            )

    # corte temporal
    tr_mask = (df["ano"] <= 2017).values
    te_mask = (df["ano"] >= 2018).values
    print(f"[rob] temporal: treino<=2017 (n={int(tr_mask.sum())}), "
          f"teste>=2018 (n={int(te_mask.sum())})")
    sem_historico = sorted(
        set(mar[te_mask].unique()) - set(mar[tr_mask].unique())
    )
    if sem_historico:
        print(f"[rob] temporal: sem historico no treino (fallback global): "
              f"{', '.join(sem_historico)}")
    for alvo in alvos:
        for nome in MODELOS_ROBUSTEZ:
            pr, refs_te = _prever_particao(
                nome, fabricas, df, cols_num, cols_cat, tr_mask, te_mask, alvo
            )
            registrar("TEMPORAL", alvo, nome, te_mask, pr, refs_te)

    return pd.DataFrame(resultados), pd.DataFrame(predicoes)


def validar_robustez_contra_12(rob: pd.DataFrame) -> None:
    """Confere que as celulas absolutas de P50 reproduzem a robustez do script 12 (protocolo C)."""
    res12 = pd.read_csv(ARTEFATOS_12_RESULTADOS)
    equivalencia = {
        ("LOMO", "media_global"): ("C_LOMO", "B1_global"),
        ("LOMO", "referencia_historica"): ("C_LOMO", "B2_por_maratona"),
        ("LOMO", "ridge_abs"): ("C_LOMO", "M2_ridge"),
        ("LOMO", "gbm_histogramas_abs"): ("C_LOMO", "M4_hist_gbm"),
        ("TEMPORAL", "media_global"): ("C_TEMPORAL", "B1_global"),
        ("TEMPORAL", "referencia_historica"): ("C_TEMPORAL", "B2_por_maratona"),
        ("TEMPORAL", "ridge_abs"): ("C_TEMPORAL", "M2_ridge"),
        ("TEMPORAL", "gbm_histogramas_abs"): ("C_TEMPORAL", "M4_hist_gbm"),
    }
    max_diff = 0.0
    for (meu_proto, meu_nome), (proto12, modelo12) in equivalencia.items():
        meu = rob[
            (rob["protocolo"] == meu_proto) & (rob["modelo"] == meu_nome)
            & (rob["alvo"] == "p50")
        ]
        deles = res12[
            (res12["protocolo"] == proto12) & (res12["modelo"] == modelo12)
            & (res12["alvo"] == "p50")
        ]
        if len(meu) != 1 or len(deles) != 1:
            raise AssertionError(f"celula de robustez ausente: {meu_proto}/{meu_nome}")
        for minha_col, col12 in (
            ("mae_min", "mae_min"), ("rmse_min", "rmse_min"), ("r2_abs", "r2"),
        ):
            d = abs(float(meu.iloc[0][minha_col]) - float(deles.iloc[0][col12]))
            max_diff = max(max_diff, d)
            if d > TOL:
                raise AssertionError(
                    f"robustez diverge do 12 em {proto12}/{modelo12}/{col12}: "
                    f"{meu.iloc[0][minha_col]!r} vs {deles.iloc[0][col12]!r}"
                )
    print(f"[valida] robustez reproduz o protocolo C do 12 em 8 celulas "
          f"(max diff {max_diff:.2e}) -> OK")


# ---------------------------------------------------------------------------
# 4. Sensibilidade de horario de largada
# ---------------------------------------------------------------------------

def edicoes_horario_atipico(df: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_csv(HORARIOS)
    h["minutos"] = (
        h["hora_largada"].str.split(":").str[0].astype(int) * 60
        + h["hora_largada"].str.split(":").str[1].astype(int)
    )
    corpus = df[["maratona", "ano"]].copy()
    j = corpus.merge(h, on=["maratona", "ano"], how="left", validate="1:1")
    if j["minutos"].isna().any():
        faltando = j[j["minutos"].isna()][["maratona", "ano"]]
        raise AssertionError(
            f"horario de largada ausente para edicoes do corpus:\n{faltando}"
        )
    modal = j.groupby("maratona")["minutos"].agg(lambda s: s.mode().iloc[0])
    j["desvio_min"] = (j["minutos"] - j["maratona"].map(modal)).abs()
    j["atipica"] = j["desvio_min"] >= LIMIAR_HORARIO_MIN
    return j[["maratona", "ano", "hora_largada", "desvio_min", "atipica"]]


MODELOS_SENSIBILIDADE = [
    "referencia_historica",
    "ridge_abs", "gbm_histogramas_abs",
    "ridge_dev", "gbm_histogramas_dev",
]


def sensibilidade_horario(
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    grade: pd.DataFrame,
    pred_grade: pd.DataFrame,
    alvos: list[str],
) -> pd.DataFrame:
    print()
    print("=== SENSIBILIDADE: horario de largada atipico ===")
    marcacao = edicoes_horario_atipico(df)
    atipicas = marcacao[marcacao["atipica"]]
    print("[sens] edicoes atipicas (desvio >= "
          f"{LIMIAR_HORARIO_MIN} min do horario modal da prova):")
    for _, r in atipicas.iterrows():
        print(f"[sens]   {r['maratona']} {r['ano']} ({r['hora_largada']}, "
              f"desvio {int(r['desvio_min'])} min)")

    # erro das edicoes atipicas na grade completa (percentil do |erro|)
    chaves = set(zip(atipicas["maratona"], atipicas["ano"]))
    pred = pred_grade.copy()
    pred["ae"] = (pred["real_min"] - pred["pred_min"]).abs()
    pred["atipica"] = [
        (m, a) in chaves for m, a in zip(pred["maratona"], pred["ano"])
    ]
    for fam in ("gbm_histogramas",):
        sel = pred[(pred["formulacao"] == FORMULACAO_DESVIO) & (pred["familia"] == fam)]
        for _, r in atipicas.iterrows():
            e = sel[(sel["maratona"] == r["maratona"]) & (sel["ano"] == r["ano"])]
            pct = [
                float((sel[sel["alvo"] == alvo]["ae"]
                       <= float(e[e["alvo"] == alvo]["ae"].iloc[0])).mean())
                for alvo in alvos
            ]
            print(f"[sens]   |erro| de {r['maratona']} {r['ano']} no gbm em desvio: "
                  f"percentil medio {100 * float(np.mean(pct)):.0f} nos alvos")

    mask = np.array(
        [(m, a) not in chaves for m, a in zip(df["maratona"], df["ano"])]
    )
    df_sem = df[mask].reset_index(drop=True)
    print(f"[sens] reexecucao LOO com {len(df_sem)} edicoes "
          f"(excluidas {int((~mask).sum())})")

    loo = LeaveOneOut()
    fabricas = t12.fabricas_modelos()
    mar = df_sem["maratona"]
    X_com = df_sem[cols_num + cols_cat]
    X_sem_cat = df_sem[cols_num]
    linhas: list[dict] = []
    for alvo in alvos:
        y = df_sem[alvo]
        preds = {nome: np.zeros(len(y)) for nome in MODELOS_SENSIBILIDADE}
        for tr, te in loo.split(df_sem):
            i = int(te[0])
            y_tr = y.iloc[tr]
            mar_tr = mar.iloc[tr]
            referencia = t12.ajustar_referencia_historica(y_tr, mar_tr, CENTRO)
            y_anom_tr = y_tr.values - referencia.prever(mar_tr)
            centro_teste = float(referencia.prever([mar.iloc[i]])[0])
            for nome in MODELOS_SENSIBILIDADE:
                if nome == "referencia_historica":
                    preds[nome][i] = centro_teste
                    continue
                familia, forma = nome.rsplit("_", 1)
                fab = fabricas[FAMILIAS[familia]]
                if forma == "abs":
                    pipe = t12.montar_pipeline(fab(), cols_num, cols_cat)
                    pipe.fit(X_com.iloc[tr], y_tr)
                    preds[nome][i] = float(pipe.predict(X_com.iloc[[i]])[0])
                else:
                    pipe = t12.montar_pipeline(fab(), cols_num, [])
                    pipe.fit(X_sem_cat.iloc[tr], y_anom_tr)
                    preds[nome][i] = float(pipe.predict(X_sem_cat.iloc[[i]])[0]) + centro_teste
        for nome, pr in preds.items():
            m = t12.metricas(y.values, pr)
            cheio = grade[
                (grade["alvo"] == alvo)
                & (
                    ((grade["formulacao"] == FORMULACAO_REFERENCIA)
                     & (grade["familia"] == nome))
                    | ((grade["formulacao"] == FORMULACAO_ABSOLUTO)
                       & (grade["familia"] == nome.removesuffix("_abs"))
                       & nome.endswith("_abs"))
                    | ((grade["formulacao"] == FORMULACAO_DESVIO)
                       & (grade["familia"] == nome.removesuffix("_dev"))
                       & nome.endswith("_dev"))
                )
            ]
            linhas.append({
                "alvo": alvo, "modelo": nome,
                "mae_sem_atipicas": m["mae_min"],
                "mae_corpus_completo": float(cheio.iloc[0]["mae_min"]),
                "delta": m["mae_min"] - float(cheio.iloc[0]["mae_min"]),
                "n_edicoes": len(df_sem),
            })
        print(f"[sens] {alvo} concluido", flush=True)
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 5. Estabilidade por edicao (ganho sobre a referencia historica)
# ---------------------------------------------------------------------------

def estabilidade_por_edicao(
    pred_grade: pd.DataFrame, alvos: list[str], familias: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("=== ESTABILIDADE POR EDICAO (delta de |erro| vs referencia) ===")
    b2 = pred_grade[
        (pred_grade["formulacao"] == FORMULACAO_REFERENCIA)
        & (pred_grade["familia"] == "referencia_historica")
    ].copy()
    b2["ae_b2"] = (b2["real_min"] - b2["pred_min"]).abs()
    b2 = b2[["maratona", "ano", "alvo", "ae_b2"]]

    detalhes: list[pd.DataFrame] = []
    resumo: list[dict] = []
    for fam in familias:
        sel = pred_grade[
            (pred_grade["formulacao"] == FORMULACAO_DESVIO)
            & (pred_grade["familia"] == fam)
        ].copy()
        sel["ae_modelo"] = (sel["real_min"] - sel["pred_min"]).abs()
        j = sel.merge(b2, on=["maratona", "ano", "alvo"], validate="1:1")
        j["delta_ae"] = j["ae_b2"] - j["ae_modelo"]  # >0 = modelo melhora
        detalhes.append(
            j[["maratona", "ano", "alvo", "familia", "ae_b2", "ae_modelo", "delta_ae"]]
        )
        for alvo in alvos:
            ja = j[j["alvo"] == alvo]
            ganho_medio = float(ja["delta_ae"].mean())
            top5 = ja.nlargest(5, "delta_ae")
            sem_top5 = ja[~ja.index.isin(top5.index)]
            resumo.append({
                "familia": fam, "alvo": alvo,
                "pct_edicoes_melhora": float((ja["delta_ae"] > 0).mean()),
                "ganho_mae_medio": ganho_medio,
                "ganho_mae_sem_top5": float(sem_top5["delta_ae"].mean()),
                "maratona_maior_ganho": str(
                    ja.groupby("maratona")["delta_ae"].mean().idxmax()
                ),
                "maratona_menor_ganho": str(
                    ja.groupby("maratona")["delta_ae"].mean().idxmin()
                ),
            })
    det = pd.concat(detalhes, ignore_index=True)
    res = pd.DataFrame(resumo)
    for _, r in res.iterrows():
        print(f"[estab] {r['familia']} {r['alvo']}: melhora em "
              f"{100 * r['pct_edicoes_melhora']:.0f}% das edicoes; ganho medio "
              f"{r['ganho_mae_medio']:.2f} min ({r['ganho_mae_sem_top5']:.2f} sem top-5)")
    return det, res


# ---------------------------------------------------------------------------
# Relatorio de decisao e figura
# ---------------------------------------------------------------------------

def relatorio_decisao(grade: pd.DataFrame, decomposicao: pd.DataFrame,
                      rob: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("RELATORIO DE COMPARACAO (criterios descritos no docstring)")
    print("=" * 72)

    print()
    print("--- 1. MAE medio nos cinco alvos, por formulacao x familia ---")
    tab = (
        grade.groupby(["formulacao", "familia"])["mae_min"].mean()
        .sort_values().reset_index()
    )
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    b2 = grade[grade["familia"] == "referencia_historica"]
    mae_b2 = {a: float(b2[b2["alvo"] == a]["mae_min"].iloc[0]) for a in b2["alvo"]}
    print()
    print("--- 2. Melhor familia por alvo e formulacao (delta vs referencia) ---")
    corpo = grade[grade["formulacao"].isin([FORMULACAO_ABSOLUTO, FORMULACAO_DESVIO])]
    for alvo in sorted(corpo["alvo"].unique()):
        for formulacao in (FORMULACAO_ABSOLUTO, FORMULACAO_DESVIO):
            g = corpo[(corpo["alvo"] == alvo) & (corpo["formulacao"] == formulacao)]
            melhor = g.loc[g["mae_min"].idxmin()]
            print(f"  {alvo} {formulacao:9s}: {melhor['familia']:24s} "
                  f"MAE {melhor['mae_min']:.3f} "
                  f"(ref {mae_b2[alvo]:.3f}; delta {melhor['mae_min'] - mae_b2[alvo]:+.3f})")

    print()
    print("--- 3. Atribuicao por bloco (medias nos cinco alvos) ---")
    ag = decomposicao.groupby("familia")[
        ["mae_E", "mae_EC", "mae_ES", "mae_ECS",
         "efeito_clima_sem_composicao", "efeito_clima_com_composicao",
         "efeito_composicao_sem_clima", "efeito_composicao_com_clima",
         "interacao"]
    ].mean()
    print(ag.to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    print("--- 4. Robustez (MAE medio nos cinco alvos) ---")
    rtab = (
        rob.groupby(["protocolo", "modelo"])["mae_min"].mean()
        .reset_index().sort_values(["protocolo", "mae_min"])
    )
    print(rtab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def figura_grade(grade: pd.DataFrame) -> None:
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
        ax.set_title(f"formulacao: {formulacao}")
        ax.set_xlabel("alvo")
        ax.set_ylabel("MAE (min)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_GRADE, dpi=150)
    plt.close(fig)
    print(f"[fig] {FIG_GRADE}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="so Ridge+GBM e P50; nao grava artefatos")
    args = parser.parse_args()

    inicio = time.time()
    df, cols_num, cols_cat = t12.carregar_dataset()

    if args.smoke:
        familias = {k: FAMILIAS[k] for k in ("ridge", "gbm_histogramas")}
        alvos = ["p50"]
        print("[modo] SMOKE: ridge+gbm, p50, sem gravacao de artefatos")
    else:
        familias = dict(FAMILIAS)
        alvos = list(ALVOS)

    familias_blocos = {k: FAMILIAS[k] for k in ("ridge", "gbm_histogramas")}

    grade, pred_grade = grade_formulacoes(df, cols_num, cols_cat, familias, alvos)
    validar_grade_contra_12(grade, pred_grade, alvos, familias)
    validar_cobertura_grade(df, grade, pred_grade, alvos, familias)

    blocos_res, decomposicao, pred_blocos = atribuicao_blocos(
        df, cols_num, pred_grade, familias_blocos, alvos
    )

    rob, pred_rob = robustez(df, cols_num, cols_cat, alvos)
    if "p50" in alvos:
        validar_robustez_contra_12(rob)

    sens = sensibilidade_horario(df, cols_num, cols_cat, grade, pred_grade, alvos)
    estab_det, estab_res = estabilidade_por_edicao(
        pred_grade, alvos, list(familias_blocos)
    )

    relatorio_decisao(grade, decomposicao, rob)

    if args.smoke:
        print()
        print(f"[ok] smoke concluido em {time.time() - inicio:.0f}s; nada gravado")
        return

    grade.to_csv(SAIDA_GRADE, index=False)
    pred_grade.to_csv(SAIDA_PRED_GRADE, index=False)
    # celulas e decomposicao tem granularidades distintas; gravadas separadas:
    # a decomposicao (uma linha por familia x alvo, com as quatro MAEs) e o
    # artefato de leitura; as celulas completas (com RMSE/R2) sao apoio.
    decomposicao.to_csv(SAIDA_BLOCOS, index=False)
    blocos_res.to_csv(
        SAIDA_BLOCOS.with_name("atribuicao_blocos_celulas_quantis.csv"), index=False
    )
    pred_blocos.to_csv(SAIDA_PRED_BLOCOS, index=False)
    rob.to_csv(SAIDA_ROBUSTEZ, index=False)
    pred_rob.to_csv(SAIDA_PRED_ROBUSTEZ, index=False)
    sens.to_csv(SAIDA_SENSIBILIDADE, index=False)
    estab_det.to_csv(SAIDA_ESTABILIDADE, index=False)
    estab_res.to_csv(
        SAIDA_ESTABILIDADE.with_name("estabilidade_por_edicao_resumo.csv"), index=False
    )
    print()
    for p in (SAIDA_GRADE, SAIDA_PRED_GRADE, SAIDA_BLOCOS, SAIDA_PRED_BLOCOS,
              SAIDA_ROBUSTEZ, SAIDA_PRED_ROBUSTEZ, SAIDA_SENSIBILIDADE,
              SAIDA_ESTABILIDADE):
        print(f"[csv] {p}")
    figura_grade(grade)

    print()
    print(f"[ok] comparacao de desenhos concluida em {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
