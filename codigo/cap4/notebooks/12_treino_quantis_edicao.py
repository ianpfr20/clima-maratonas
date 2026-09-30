"""
Treinamento e avaliacao dos modelos de quantis do tempo de prova por edicao
(Capitulo 4 da monografia). Cada alvo e um dos quantis P10, P25, P50, P75 e
P90 do tempo de conclusao de uma edicao, em minutos.

Entrada: tabela de treino por edicao (dataset117: 117 edicoes, uma linha por
(maratona, ano); caminho na constante DATASET), gerada por
codigo/cap4/notebooks/11_dataset_treino_edicoes.py. Features = 31 variaveis
climaticas da janela de seis horas a partir da largada (ERA5 via Open-Meteo)
mais o contexto da edicao [pct_M, ln_n_finishers, ano, dia_ano_sin,
dia_ano_cos]; categorica = maratona.

Modelos (os codigos aparecem nas colunas `modelo` das saidas):
  B1  media global do alvo no treino
  B2  referencia historica: mediana do mesmo quantil na mesma maratona,
      calculada so no treino (mediana global se a maratona nao esta no treino)
  M1  regressao linear
  M2  ridge (RidgeCV alphas=[0.01,0.1,1,10,100])
  M3  random forest (400 arvores, min_samples_leaf=2, seed 42)
  M4  gradient boosting por histograma (max_iter=300, lr=0.05, max_depth=4,
      min_samples_leaf=4, seed 42)
  M5  MLP ((64,32), relu, alpha=1e-3, max_iter=1000, seed 42)

Protocolos:
  A  Validacao com uma edicao reservada por vez (leave-one-out, 117 folds),
     7 modelos, 5 alvos. MAE e a metrica primaria; RMSE (min) e R2 sao
     secundarios.
  B  Desvio historico (o clima explica o afastamento da referencia
     historica?): em cada fold, y_anom = y - referencia historica ajustada no
     treino. Ridge e gradient boosting (codigos M2a e M4a) treinam sobre
     y_anom SEM a indicadora de maratona, so com as numericas; a predicao
     volta a escala absoluta somando a referencia. O centro vem so do treino
     do fold, sem vazamento. Metricas sobre o y reconstruido (comparaveis ao A).
  C  Robustez, so para P50 (para conter custo):
     - exclusao de maratona inteira (LOMO, 11 folds) com B1, B2, M2 e M4.
       Como a maratona de teste some do treino, o B2 usa a mediana global do
       alvo calculada no treino.
     - corte temporal: treino ano<=2017, teste ano>=2018, mesmos 4 modelos.
  D  Ablacao 2x2 (forma do alvo x indicadora de maratona). M2a e M4a mudam
     duas coisas em relacao a M2 e M4: centralizam o alvo na referencia
     historica e retiram a indicadora de maratona. O protocolo D ajusta as
     duas celulas restantes, nos mesmos folds leave-one-out:
       M2s / M4s    alvo absoluto, so as numericas, SEM a indicadora;
       M2ai / M4ai  alvo em desvio historico, numericas MAIS a indicadora.
     Com as quatro celulas, o ganho de M4a sobre M4 se decompoe em efeito de
     centralizacao, efeito de retirada da indicadora e interacao. As celulas
     em desvio historico voltam a escala absoluta antes de qualquer metrica.

Sensibilidade: os metodos que dependem da referencia historica sao
repetidos com a media no lugar da mediana, com os mesmos folds, features,
sementes e hiperparametros.

Custo: o MLP usa max_iter=1000. Com 117 folds x 7 modelos x 5 alvos, 2000
iteracoes tornam a execucao muito lenta, e 1000 bastam para esta amostra
pequena.

Execucao no Windows: as bibliotecas numericas ficam limitadas a 1 thread e o
random forest roda com n_jobs=1. Com n_jobs=-1 o backend loky trava dentro
do leave-one-out (processo vivo, CPU=0, sem escrever CSV). Em serie, a
execucao e deterministica e leva cerca de 40 minutos num processador de
notebook.

Saidas:
  dados/processados/apoio/resultados_treino_quantis.csv
  dados/processados/apoio/predicoes_loo_quantis.csv  (protocolo primario)
  dados/processados/apoio/resultados_sensibilidade_media_quantis.csv
  dados/processados/apoio/predicoes_sensibilidade_media_quantis.csv
  dados/processados/apoio/comparacao_centralizacao_quantis.csv
  dados/processados/apoio/ablacao_alvo_indicadora_quantis.csv
  codigo/cap4/figuras/fig_mae_por_quantil.png
  codigo/cap4/figuras/fig_sensibilidade_centralizacao.png
  codigo/cap4/figuras/fig_ablacao_alvo_indicadora.png
  no terminal: tabela consolidada, deltas vs B2 por alvo e sintese final.

Uso (a partir da raiz do repositorio):
    python codigo/cap4/notebooks/12_treino_quantis_edicao.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

# Limita as bibliotecas BLAS/OpenMP a 1 thread: no Windows, o leave-one-out
# (585 ajustes) com n_jobs=-1 e o excesso de threads do MLP/BLAS travam o
# backend loky (processo vivo, CPU=0, sem progresso). Precisa ser definido
# ANTES de importar numpy/sklearn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[3]
DATASET = RAIZ / "dataset117" / "dataset_treino_edicoes.csv"
APOIO_DIR = RAIZ / "dados" / "processados" / "apoio"
FIG_DIR = RAIZ / "codigo" / "cap4" / "figuras"
APOIO_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAIDA_RESULTADOS = APOIO_DIR / "resultados_treino_quantis.csv"
SAIDA_PREDICOES = APOIO_DIR / "predicoes_loo_quantis.csv"
SAIDA_RESULTADOS_MEDIA = APOIO_DIR / "resultados_sensibilidade_media_quantis.csv"
SAIDA_PREDICOES_MEDIA = APOIO_DIR / "predicoes_sensibilidade_media_quantis.csv"
SAIDA_COMPARACAO_CENTRO = APOIO_DIR / "comparacao_centralizacao_quantis.csv"
SAIDA_ABLACAO = APOIO_DIR / "ablacao_alvo_indicadora_quantis.csv"
FIG_MAE = FIG_DIR / "fig_mae_por_quantil.png"
FIG_SENSIBILIDADE = FIG_DIR / "fig_sensibilidade_centralizacao.png"
FIG_ABLACAO = FIG_DIR / "fig_ablacao_alvo_indicadora.png"

SEED = 42
ALVOS = ["p10", "p25", "p50", "p75", "p90"]
MLP_MAX_ITER = 1000  # ver nota de custo na docstring
CENTRO_PRIMARIO = "mediana"
CENTRO_SENSIBILIDADE = "media"
CENTRO_NAO_APLICAVEL = "nao_se_aplica"
ESTATISTICAS_CENTRO = {
    CENTRO_PRIMARIO: "median",
    CENTRO_SENSIBILIDADE: "mean",
}

COLUNAS_RESULTADOS = [
    "protocolo", "alvo", "modelo", "estatistica_centro",
    "mae_min", "rmse_min", "r2",
]
COLUNAS_PREDICOES = [
    "maratona", "ano", "alvo", "modelo", "estatistica_centro",
    "real_min", "pred_min",
]

# Ablacao 2x2 do protocolo D. Cada familia aparece nas quatro combinacoes de
# (forma do alvo) x (indicadora de maratona): `abs`/`anom` para alvo
# absoluto ou em desvio historico, `com`/`sem` para a indicadora.
FAMILIAS_ABLACAO = {
    "ridge": {
        "abs_com": "M2_ridge",
        "abs_sem": "M2s_ridge_semcat",
        "anom_com": "M2ai_ridge_anom_cat",
        "anom_sem": "M2a_ridge_anom",
    },
    "hist_gbm": {
        "abs_com": "M4_hist_gbm",
        "abs_sem": "M4s_hist_gbm_semcat",
        "anom_com": "M4ai_hist_gbm_anom_cat",
        "anom_sem": "M4a_hist_gbm_anom",
    },
}
PROTOCOLO_DA_CELULA = {
    "abs_com": "A",
    "abs_sem": "D",
    "anom_com": "D",
    "anom_sem": "B",
}
COLUNAS_ABLACAO = [
    "estatistica_centro", "familia", "alvo",
    "mae_abs_com", "mae_abs_sem", "mae_anom_com", "mae_anom_sem",
    "ganho_total", "efeito_centralizacao", "efeito_retirada_indicadora",
    "interacao", "efeito_centralizacao_sem_indicadora",
    "efeito_retirada_indicadora_anomalia",
    "rmse_abs_com", "rmse_abs_sem", "rmse_anom_com", "rmse_anom_sem",
]

# O MLP nao converge em 1000 iteracoes nesta amostra pequena (esperado e
# inofensivo); silencia a ConvergenceWarning para nao repetir centenas de
# avisos no log do leave-one-out.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Definicao de features e fabricas de modelos
# ---------------------------------------------------------------------------

def carregar_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(DATASET)
    cols_id = ["maratona", "ano", "data_prova"]
    cols_alvo = ["p10", "p25", "p50", "p75", "p90", "iqr"]
    cols_ctx_num = ["pct_M", "ln_n_finishers", "ano",
                    "dia_ano_sin", "dia_ano_cos"]
    # clima = tudo que nao e id, alvo ou contexto explicito
    exclui = set(cols_id + cols_alvo + ["n_finishers"] + cols_ctx_num)
    cols_clima = [c for c in df.columns if c not in exclui]
    cols_num = cols_clima + cols_ctx_num
    cols_cat = ["maratona"]
    print(f"[dataset] {len(df)} edicoes; {len(cols_clima)} features climaticas; "
          f"{len(cols_num)} numericas; 1 categorica")
    return df, cols_num, cols_cat


def fabricas_modelos() -> dict:
    """Fabricas dos modelos sklearn (uma funcao por modelo, sem estado)."""
    return {
        "M1_linear": lambda: LinearRegression(),
        "M2_ridge": lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        # n_jobs=1: no Windows o loky com n_jobs=-1 trava dentro dos 585
        # ajustes do leave-one-out (ver docstring). Em serie e estavel e
        # rapido o bastante com 400 arvores e n<=116 amostras por fold.
        "M3_random_forest": lambda: RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=SEED, n_jobs=1),
        # random_state nao tem efeito aqui: com n=116, `do_early_stopping_`
        # resolve para False (limiar de 10.000 amostras) e a subamostragem do
        # binning so ocorre acima de 200.000. Fica por simetria com as outras
        # fabricas; a semente so age de fato no random forest e no MLP.
        "M4_hist_gbm": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=4, random_state=SEED),
        "M5_mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            alpha=1e-3, max_iter=MLP_MAX_ITER, random_state=SEED,
            early_stopping=False),
    }


def fabricas_ablacao() -> dict:
    """Celulas faltantes do 2x2, com as fabricas identicas as do protocolo A.

    Cada valor e `(fabrica, alvo_em_anomalia, usa_indicadora)`.
    """
    base = fabricas_modelos()
    return {
        "M2s_ridge_semcat": (base["M2_ridge"], False, False),
        "M4s_hist_gbm_semcat": (base["M4_hist_gbm"], False, False),
        "M2ai_ridge_anom_cat": (base["M2_ridge"], True, True),
        "M4ai_hist_gbm_anom_cat": (base["M4_hist_gbm"], True, True),
    }


def montar_pipeline(modelo, cols_num: list[str], cols_cat: list[str]) -> Pipeline:
    # Nenhuma coluna do dataset tem valor ausente (ver validacao em
    # 11_dataset_treino_edicoes.py); o imputador fica como protecao caso uma
    # fonte futura introduza lacunas.
    pipe_num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    transformers = [("num", pipe_num, cols_num)]
    if cols_cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat)
        )
    pre = ColumnTransformer(transformers=transformers)
    return Pipeline([("pre", pre), ("m", modelo)])


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae_min": float(mean_absolute_error(y_real, y_pred)),
        "rmse_min": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2": float(r2_score(y_real, y_pred)),
    }


# ---------------------------------------------------------------------------
# Referencia historica por fold (unica implementacao para todos os protocolos)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenciaHistorica:
    """Centro do mesmo alvo ajustado exclusivamente sobre uma porcao de treino."""

    por_maratona: pd.Series
    global_alvo: float
    estatistica: str

    def prever(self, maratonas: pd.Series | np.ndarray | list[str]) -> np.ndarray:
        """Aplica centros por maratona e o fallback global a provas ausentes."""
        valores = pd.Series(np.asarray(maratonas, dtype=object))
        return (
            valores.map(self.por_maratona)
            .fillna(self.global_alvo)
            .to_numpy(dtype=float)
        )


def ajustar_referencia_historica(
    y_treino: pd.Series | np.ndarray,
    maratonas_treino: pd.Series | np.ndarray,
    estatistica: str,
) -> ReferenciaHistorica:
    """Ajusta mediana ou media do mesmo alvo usando somente linhas de treino."""
    if estatistica not in ESTATISTICAS_CENTRO:
        raise ValueError(
            f"Estatistica de centro invalida: {estatistica!r}; "
            f"use {sorted(ESTATISTICAS_CENTRO)}"
        )

    y = np.asarray(y_treino, dtype=float)
    maratonas = np.asarray(maratonas_treino, dtype=object)
    if len(y) == 0 or len(y) != len(maratonas):
        raise ValueError("Treino da referencia deve ser nao vazio e alinhado")
    if not np.isfinite(y).all():
        raise ValueError("Alvo de treino da referencia contem valor nao finito")

    agregador = ESTATISTICAS_CENTRO[estatistica]
    treino = pd.DataFrame({"maratona": maratonas, "alvo": y})
    por_maratona = treino.groupby("maratona", sort=False)["alvo"].agg(agregador)
    global_alvo = float(treino["alvo"].agg(agregador))
    return ReferenciaHistorica(por_maratona, global_alvo, estatistica)


def reconstruir_predicao_absoluta(
    predicao_anomalia: float | np.ndarray,
    referencia: float | np.ndarray,
) -> np.ndarray:
    """Reconstrui a escala absoluta antes do calculo de qualquer metrica."""
    return np.asarray(predicao_anomalia, dtype=float) + np.asarray(referencia, dtype=float)


def predicoes_referencia_loo(
    y: pd.Series,
    maratonas: pd.Series,
    estatistica: str,
) -> np.ndarray:
    """Prediz cada edicao pelo centro historico ajustado no treino do fold."""
    predicoes = np.zeros(len(y), dtype=float)
    for tr, te in LeaveOneOut().split(y):
        referencia = ajustar_referencia_historica(
            y.iloc[tr], maratonas.iloc[tr], estatistica
        )
        i = int(te[0])
        predicoes[i] = referencia.prever([maratonas.iloc[i]])[0]
    return predicoes


# ---------------------------------------------------------------------------
# Protocolo A: LOO padrao, 7 modelos, 5 alvos
# ---------------------------------------------------------------------------

def protocolo_a(
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    estatistica_centro: str = CENTRO_PRIMARIO,
):
    print()
    print(
        "=== PROTOCOLO A (LOO padrao, 7 modelos x 5 alvos; "
        f"B2={estatistica_centro}) ==="
    )
    loo = LeaveOneOut()
    fabricas = fabricas_modelos()
    X = df[cols_num + cols_cat]
    mar = df["maratona"]

    resultados = []
    predicoes = []  # formato longo

    for alvo in ALVOS:
        y = df[alvo]
        print(f"[A] alvo {alvo}...")

        # baselines
        preds_b1 = np.zeros(len(y))
        preds_b2 = predicoes_referencia_loo(y, mar, estatistica_centro)
        preds_ml = {nome: np.zeros(len(y)) for nome in fabricas}

        for tr, te in loo.split(X):
            i = te[0]
            y_tr = y.iloc[tr]
            media_global = float(y_tr.mean())
            preds_b1[i] = media_global
            for nome, fab in fabricas.items():
                pipe = montar_pipeline(fab(), cols_num, cols_cat)
                pipe.fit(X.iloc[tr], y_tr)
                preds_ml[nome][i] = pipe.predict(X.iloc[[i]])[0]

        todos = {"B1_global": preds_b1, "B2_por_maratona": preds_b2, **preds_ml}
        for nome, pr in todos.items():
            m = metricas(y.values, pr)
            centro = (
                estatistica_centro
                if nome == "B2_por_maratona"
                else CENTRO_NAO_APLICAVEL
            )
            resultados.append({"protocolo": "A", "alvo": alvo, "modelo": nome,
                               "estatistica_centro": centro,
                               "mae_min": m["mae_min"], "rmse_min": m["rmse_min"],
                               "r2": m["r2"]})
            for j in range(len(y)):
                predicoes.append({
                    "maratona": df["maratona"].iloc[j], "ano": int(df["ano"].iloc[j]),
                    "alvo": alvo, "modelo": nome,
                    "estatistica_centro": centro,
                    "real_min": float(y.iloc[j]), "pred_min": float(pr[j]),
                })

    return pd.DataFrame(resultados), pd.DataFrame(predicoes)


# ---------------------------------------------------------------------------
# Sensibilidade do B2: mesma referencia LOO, trocando apenas o centro
# ---------------------------------------------------------------------------

def protocolo_a_baseline_historica(
    df: pd.DataFrame,
    estatistica_centro: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print(
        "=== SENSIBILIDADE DO B2 (LOO; "
        f"centro={estatistica_centro}) ==="
    )
    resultados = []
    predicoes = []
    mar = df["maratona"]

    for alvo in ALVOS:
        y = df[alvo]
        pr = predicoes_referencia_loo(y, mar, estatistica_centro)
        m = metricas(y.values, pr)
        resultados.append({
            "protocolo": "A",
            "alvo": alvo,
            "modelo": "B2_por_maratona",
            "estatistica_centro": estatistica_centro,
            **m,
        })
        for j in range(len(y)):
            predicoes.append({
                "maratona": df["maratona"].iloc[j],
                "ano": int(df["ano"].iloc[j]),
                "alvo": alvo,
                "modelo": "B2_por_maratona",
                "estatistica_centro": estatistica_centro,
                "real_min": float(y.iloc[j]),
                "pred_min": float(pr[j]),
            })
    return pd.DataFrame(resultados), pd.DataFrame(predicoes)


# ---------------------------------------------------------------------------
# Protocolo B: anomalia (M2a, M4a), so numericas, sobre y - centro_maratona
# ---------------------------------------------------------------------------

def protocolo_b(
    df: pd.DataFrame,
    cols_num: list[str],
    estatistica_centro: str,
):
    print()
    print(
        "=== PROTOCOLO B (anomalia: y - centro_maratona; "
        f"centro={estatistica_centro}; M2a, M4a) ==="
    )
    loo = LeaveOneOut()
    X = df[cols_num]  # sem categorica
    mar = df["maratona"]

    modelos_b = {
        "M2a_ridge_anom": lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        "M4a_hist_gbm_anom": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=4, random_state=SEED),
    }

    resultados = []
    predicoes = []

    for alvo in ALVOS:
        y = df[alvo]
        print(f"[B-{estatistica_centro}] alvo {alvo}...", flush=True)
        preds = {nome: np.zeros(len(y)) for nome in modelos_b}

        for fold, (tr, te) in enumerate(loo.split(X), start=1):
            i = te[0]
            y_tr = y.iloc[tr]
            mar_tr = mar.iloc[tr]
            referencia = ajustar_referencia_historica(
                y_tr, mar_tr, estatistica_centro
            )
            centros_tr = referencia.prever(mar_tr)
            y_anom_tr = y_tr.values - centros_tr
            centro_teste = referencia.prever([mar.iloc[i]])[0]
            for nome, fab in modelos_b.items():
                pipe = montar_pipeline(fab(), cols_num, [])
                pipe.fit(X.iloc[tr], y_anom_tr)
                pred_anom = pipe.predict(X.iloc[[i]])[0]
                preds[nome][i] = reconstruir_predicao_absoluta(
                    pred_anom, centro_teste
                ).item()
            if fold % 25 == 0 or fold == len(y):
                print(
                    f"[B-{estatistica_centro}] {alvo}: fold {fold}/{len(y)}",
                    flush=True,
                )

        for nome, pr in preds.items():
            m = metricas(y.values, pr)
            resultados.append({"protocolo": "B", "alvo": alvo, "modelo": nome,
                               "estatistica_centro": estatistica_centro,
                               "mae_min": m["mae_min"], "rmse_min": m["rmse_min"],
                               "r2": m["r2"]})
            for j in range(len(y)):
                predicoes.append({
                    "maratona": df["maratona"].iloc[j], "ano": int(df["ano"].iloc[j]),
                    "alvo": alvo, "modelo": nome,
                    "estatistica_centro": estatistica_centro,
                    "real_min": float(y.iloc[j]), "pred_min": float(pr[j]),
                })

    return pd.DataFrame(resultados), pd.DataFrame(predicoes)


# ---------------------------------------------------------------------------
# Protocolo C: robustez so para P50 (LOMO + split temporal)
# ---------------------------------------------------------------------------

def _prever_fold(
    nome_modelo,
    fabricas,
    X_tr,
    y_tr,
    mar_tr,
    X_te,
    mar_te,
    estatistica_centro,
):
    """Retorna predicoes para um par (treino, teste) dado o nome do modelo."""
    media_global = float(y_tr.mean())
    if nome_modelo == "B1_global":
        return np.full(len(X_te), media_global)
    if nome_modelo == "B2_por_maratona":
        referencia = ajustar_referencia_historica(
            y_tr, mar_tr, estatistica_centro
        )
        return referencia.prever(mar_te)
    pipe = montar_pipeline(fabricas[nome_modelo](), X_tr.columns.tolist()[:-1], ["maratona"])
    # X_tr / X_te ja contem cols_num + maratona na ordem certa
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def protocolo_c(
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    estatistica_centro: str,
    modelos_c: list[str] | None = None,
):
    print()
    print(
        "=== PROTOCOLO C (robustez, so P50; "
        f"B2={estatistica_centro}) ==="
    )
    alvo = "p50"
    y = df[alvo]
    X = df[cols_num + cols_cat]
    mar = df["maratona"]
    fabricas = fabricas_modelos()
    if modelos_c is None:
        modelos_c = ["B1_global", "B2_por_maratona", "M2_ridge", "M4_hist_gbm"]
    resultados = []

    # --- C1: LOMO (leave-one-marathon-out) ---
    maratonas = sorted(mar.unique())
    print(f"[C] LOMO: {len(maratonas)} maratonas (folds)")
    preds_lomo = {nome: np.full(len(y), np.nan) for nome in modelos_c}
    for m_out in maratonas:
        te_mask = (mar == m_out).values
        tr_mask = ~te_mask
        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr = y[tr_mask]
        mar_tr, mar_te = mar[tr_mask], mar[te_mask]
        for nome in modelos_c:
            p = _prever_fold(
                nome, fabricas, X_tr, y_tr, mar_tr, X_te, mar_te.values,
                estatistica_centro,
            )
            preds_lomo[nome][te_mask] = p
    if "B2_por_maratona" in modelos_c:
        print(
            f"[C] LOMO: B2 usou fallback global por {estatistica_centro} "
            f"nos {len(maratonas)} folds (maratona de teste ausente do treino)"
        )
    for nome in modelos_c:
        met = metricas(y.values, preds_lomo[nome])
        centro = (
            estatistica_centro
            if nome == "B2_por_maratona"
            else CENTRO_NAO_APLICAVEL
        )
        resultados.append({"protocolo": "C_LOMO", "alvo": alvo, "modelo": nome,
                           "estatistica_centro": centro,
                           "mae_min": met["mae_min"], "rmse_min": met["rmse_min"],
                           "r2": met["r2"]})

    # --- C2: split temporal ---
    tr_mask = (df["ano"] <= 2017).values
    te_mask = (df["ano"] >= 2018).values
    n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
    print(f"[C] split temporal: treino ano<=2017 (n={n_tr}), teste ano>=2018 (n={n_te})")
    X_tr, X_te = X[tr_mask], X[te_mask]
    y_tr, y_te = y[tr_mask], y[te_mask]
    mar_tr, mar_te = mar[tr_mask], mar[te_mask]
    for nome in modelos_c:
        p = _prever_fold(
            nome, fabricas, X_tr, y_tr, mar_tr, X_te, mar_te.values,
            estatistica_centro,
        )
        met = metricas(y_te.values, p)
        centro = (
            estatistica_centro
            if nome == "B2_por_maratona"
            else CENTRO_NAO_APLICAVEL
        )
        resultados.append({"protocolo": "C_TEMPORAL", "alvo": alvo, "modelo": nome,
                           "estatistica_centro": centro,
                           "mae_min": met["mae_min"], "rmse_min": met["rmse_min"],
                           "r2": met["r2"]})

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Protocolo D: ablacao 2x2 (forma do alvo) x (indicadora de maratona)
# ---------------------------------------------------------------------------

def protocolo_d(
    df: pd.DataFrame,
    cols_num: list[str],
    cols_cat: list[str],
    estatistica_centro: str,
    incluir_absolutos: bool = True,
):
    """Ajusta as celulas que faltavam para separar centralizacao e indicadora.

    As celulas de alvo absoluto nao usam a referencia historica e por isso sao
    ajustadas uma unica vez, no protocolo primario; `incluir_absolutos=False`
    reexecuta apenas as celulas afetadas pelo centro na trilha de
    sensibilidade.
    """
    print()
    print(
        "=== PROTOCOLO D (ablacao 2x2 alvo x indicadora; "
        f"centro={estatistica_centro}) ==="
    )
    loo = LeaveOneOut()
    X_com = df[cols_num + cols_cat]
    X_sem = df[cols_num]
    mar = df["maratona"]

    celulas = {
        nome: celula
        for nome, celula in fabricas_ablacao().items()
        if celula[1] or incluir_absolutos
    }

    resultados = []
    predicoes = []

    for alvo in ALVOS:
        y = df[alvo]
        print(f"[D-{estatistica_centro}] alvo {alvo}...", flush=True)
        preds = {nome: np.zeros(len(y)) for nome in celulas}

        for fold, (tr, te) in enumerate(loo.split(X_com), start=1):
            i = te[0]
            y_tr = y.iloc[tr]
            mar_tr = mar.iloc[tr]
            referencia = ajustar_referencia_historica(
                y_tr, mar_tr, estatistica_centro
            )
            y_anom_tr = y_tr.values - referencia.prever(mar_tr)
            centro_teste = referencia.prever([mar.iloc[i]])[0]
            for nome, (fab, anomalia, com_indicadora) in celulas.items():
                X = X_com if com_indicadora else X_sem
                pipe = montar_pipeline(
                    fab(), cols_num, cols_cat if com_indicadora else []
                )
                pipe.fit(X.iloc[tr], y_anom_tr if anomalia else y_tr)
                pred = pipe.predict(X.iloc[[i]])[0]
                preds[nome][i] = (
                    reconstruir_predicao_absoluta(pred, centro_teste).item()
                    if anomalia
                    else float(pred)
                )
            if fold % 25 == 0 or fold == len(y):
                print(
                    f"[D-{estatistica_centro}] {alvo}: fold {fold}/{len(y)}",
                    flush=True,
                )

        for nome, pr in preds.items():
            m = metricas(y.values, pr)
            centro = (
                estatistica_centro
                if celulas[nome][1]
                else CENTRO_NAO_APLICAVEL
            )
            resultados.append({"protocolo": "D", "alvo": alvo, "modelo": nome,
                               "estatistica_centro": centro,
                               "mae_min": m["mae_min"], "rmse_min": m["rmse_min"],
                               "r2": m["r2"]})
            for j in range(len(y)):
                predicoes.append({
                    "maratona": df["maratona"].iloc[j], "ano": int(df["ano"].iloc[j]),
                    "alvo": alvo, "modelo": nome,
                    "estatistica_centro": centro,
                    "real_min": float(y.iloc[j]), "pred_min": float(pr[j]),
                })

    return pd.DataFrame(resultados), pd.DataFrame(predicoes)


# ---------------------------------------------------------------------------
# Relatorio: deltas vs B2, sintese, figura
# ---------------------------------------------------------------------------

def relatar_deltas_e_sintese(res_a: pd.DataFrame, res_b: pd.DataFrame) -> None:
    print()
    print(
        "=== DELTAS DE MAE vs B2 MEDIANA "
        "(por alvo; negativo = ganha da linha de base) ==="
    )
    res_ab = pd.concat([res_a, res_b], ignore_index=True)
    alvos_vencidos = 0
    detalhes_vitoria = []
    for alvo in ALVOS:
        sub = res_ab[res_ab["alvo"] == alvo]
        mae_b2 = sub[sub["modelo"] == "B2_por_maratona"]["mae_min"].iloc[0]
        print(f"\n[{alvo}] B2 mediana MAE = {mae_b2:.3f} min")
        comp = sub[sub["modelo"] != "B2_por_maratona"].copy()
        comp["delta_vs_b2"] = comp["mae_min"] - mae_b2
        comp = comp.sort_values("delta_vs_b2")
        for _, r in comp.iterrows():
            marca = "  <== ganha" if r["delta_vs_b2"] < 0 else ""
            print(f"    {r['modelo']:<20} MAE={r['mae_min']:6.3f}  "
                  f"delta={r['delta_vs_b2']:+.3f}{marca}")
        melhor = comp.iloc[0]
        if melhor["delta_vs_b2"] < 0:
            alvos_vencidos += 1
            detalhes_vitoria.append((alvo, melhor["modelo"], melhor["delta_vs_b2"]))

    print()
    print("=== SINTESE FINAL ===")
    print(f"Em {alvos_vencidos}/{len(ALVOS)} alvos algum modelo com clima supera o B2.")
    if detalhes_vitoria:
        for alvo, mod, d in detalhes_vitoria:
            print(f"  {alvo}: melhor = {mod}, ganho de {-d:.3f} min de MAE sobre o B2")
    else:
        print("  Nenhum modelo com clima bateu o B2 em MAE em nenhum alvo.")


def figura_mae_por_quantil(res_a: pd.DataFrame, res_b: pd.DataFrame) -> None:
    # Protocolo A (todos) + M2a/M4a do B. B2 vira linha de referencia.
    dados = pd.concat([res_a, res_b], ignore_index=True)
    dados = dados[dados["alvo"].isin(ALVOS)].copy()

    ordem_alvos = ALVOS
    modelos_linha = [m for m in dados["modelo"].unique() if m != "B2_por_maratona"]
    # ordena modelos por MAE medio para uma legenda estavel
    ordem_mod = (
        dados[dados["modelo"].isin(modelos_linha)]
        .groupby("modelo")["mae_min"].mean().sort_values().index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ordem_alvos))
    paleta = sns.color_palette("viridis", len(ordem_mod))
    for cor, mod in zip(paleta, ordem_mod):
        sub = dados[dados["modelo"] == mod].set_index("alvo").reindex(ordem_alvos)
        ax.plot(x, sub["mae_min"].values, marker="o", label=mod, color=cor)

    # B2 como linha de referencia destacada
    b2 = dados[dados["modelo"] == "B2_por_maratona"].set_index("alvo").reindex(ordem_alvos)
    ax.plot(x, b2["mae_min"].values, marker="s", color="#c0392b", lw=2.5,
            ls="--", label="B2_por_maratona (mediana; ref.)", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ordem_alvos])
    ax.set_xlabel("Quantil alvo")
    ax.set_ylabel("MAE (min) - LOO")
    ax.set_title("MAE por modelo x quantil (protocolo primario por mediana)\n"
                 "B2 (mediana por maratona) em vermelho tracejado como referencia")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_MAE, dpi=140)
    plt.close(fig)
    print(f"[fig] {FIG_MAE}")


def comparar_centralizacoes(
    resultados_primarios: pd.DataFrame,
    resultados_media: pd.DataFrame,
) -> pd.DataFrame:
    """Compara somente os metodos afetados pela referencia historica."""
    modelos = ["B2_por_maratona", "M2a_ridge_anom", "M4a_hist_gbm_anom"]
    chaves = ["protocolo", "alvo", "modelo"]
    metricas_cols = ["mae_min", "rmse_min", "r2"]

    mediana = resultados_primarios[
        resultados_primarios["modelo"].isin(modelos)
        & resultados_primarios["protocolo"].isin(["A", "B"])
    ][chaves + metricas_cols].copy()
    media = resultados_media[
        resultados_media["modelo"].isin(modelos)
        & resultados_media["protocolo"].isin(["A", "B"])
    ][chaves + metricas_cols].copy()
    mediana = mediana.rename(columns={c: f"{c}_mediana" for c in metricas_cols})
    media = media.rename(columns={c: f"{c}_media" for c in metricas_cols})
    comparacao = mediana.merge(media, on=chaves, how="outer", validate="one_to_one")

    if comparacao.isna().any().any():
        raise AssertionError("Comparacao mediana x media ficou incompleta")
    comparacao["ganho_mae_mediana_vs_media"] = (
        comparacao["mae_min_media"] - comparacao["mae_min_mediana"]
    )
    comparacao["ganho_rmse_mediana_vs_media"] = (
        comparacao["rmse_min_media"] - comparacao["rmse_min_mediana"]
    )
    ordem_modelo = {modelo: i for i, modelo in enumerate(modelos)}
    ordem_alvo = {alvo: i for i, alvo in enumerate(ALVOS)}
    comparacao["_ord_modelo"] = comparacao["modelo"].map(ordem_modelo)
    comparacao["_ord_alvo"] = comparacao["alvo"].map(ordem_alvo)
    return (
        comparacao.sort_values(["_ord_modelo", "_ord_alvo"])
        .drop(columns=["_ord_modelo", "_ord_alvo"])
        .reset_index(drop=True)
    )


def relatar_sensibilidade(comparacao: pd.DataFrame) -> None:
    print()
    print("=== MEDIANA PRIMARIA x MEDIA DE SENSIBILIDADE ===")
    cols = [
        "alvo", "modelo", "mae_min_mediana", "mae_min_media",
        "ganho_mae_mediana_vs_media",
    ]
    print(comparacao[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    agregado = (
        comparacao.groupby("modelo", sort=False)[
            ["mae_min_mediana", "mae_min_media", "ganho_mae_mediana_vs_media"]
        ]
        .mean()
        .reset_index()
    )
    print("=== MAE MEDIO NOS CINCO ALVOS ===")
    print(agregado.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def figura_sensibilidade_centralizacao(comparacao: pd.DataFrame) -> None:
    modelos = ["B2_por_maratona", "M2a_ridge_anom", "M4a_hist_gbm_anom"]
    titulos = {
        "B2_por_maratona": "Referencia historica",
        "M2a_ridge_anom": "Ridge sobre o desvio historico",
        "M4a_hist_gbm_anom": "Gradient boosting sobre o desvio historico",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    x = np.arange(len(ALVOS))
    for ax, modelo in zip(axes, modelos):
        sub = comparacao[comparacao["modelo"] == modelo].set_index("alvo").reindex(ALVOS)
        ax.plot(
            x, sub["mae_min_mediana"], marker="o", lw=2.2,
            label="mediana (primaria)", color="#1f77b4",
        )
        ax.plot(
            x, sub["mae_min_media"], marker="s", lw=1.9, ls="--",
            label="media (sensibilidade)", color="#e67e22",
        )
        ax.set_title(titulos[modelo])
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in ALVOS])
        ax.set_xlabel("quantil alvo")
        ax.set_ylabel("MAE (min) - LOO")
        ax.legend(fontsize=8)
    fig.suptitle("Sensibilidade do MAE a estatistica da referencia historica")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_SENSIBILIDADE, dpi=140)
    plt.close(fig)
    print(f"[fig] {FIG_SENSIBILIDADE}")


def _valor_resultado(
    resultados: pd.DataFrame,
    protocolo: str,
    alvo: str,
    modelo: str,
    metrica: str,
) -> float:
    linha = resultados[
        (resultados["protocolo"] == protocolo)
        & (resultados["alvo"] == alvo)
        & (resultados["modelo"] == modelo)
    ]
    if len(linha) != 1:
        raise AssertionError(
            f"Esperada uma linha para {protocolo}/{alvo}/{modelo}; obtidas {len(linha)}"
        )
    return float(linha.iloc[0][metrica])


def decompor_ablacao(
    resultados_primarios: pd.DataFrame,
    resultados_media: pd.DataFrame,
) -> pd.DataFrame:
    """Separa centralizacao, retirada da indicadora e interacao no 2x2.

    Todos os efeitos sao reducoes de MAE em minutos: positivo significa que a
    mudanca melhora a predicao. As celulas de alvo absoluto nao dependem da
    estatistica de centro e por isso vem sempre do conjunto primario.
    """
    linhas = []
    for centro, resultados_anomalia in (
        (CENTRO_PRIMARIO, resultados_primarios),
        (CENTRO_SENSIBILIDADE, resultados_media),
    ):
        for familia, modelos in FAMILIAS_ABLACAO.items():
            for alvo in ALVOS:
                mae = {}
                rmse = {}
                for celula, modelo in modelos.items():
                    fonte = (
                        resultados_primarios
                        if celula.startswith("abs")
                        else resultados_anomalia
                    )
                    protocolo = PROTOCOLO_DA_CELULA[celula]
                    mae[celula] = _valor_resultado(
                        fonte, protocolo, alvo, modelo, "mae_min"
                    )
                    rmse[celula] = _valor_resultado(
                        fonte, protocolo, alvo, modelo, "rmse_min"
                    )
                ganho_total = mae["abs_com"] - mae["anom_sem"]
                efeito_centralizacao = mae["abs_com"] - mae["anom_com"]
                efeito_indicadora = mae["abs_com"] - mae["abs_sem"]
                linhas.append({
                    "estatistica_centro": centro,
                    "familia": familia,
                    "alvo": alvo,
                    "mae_abs_com": mae["abs_com"],
                    "mae_abs_sem": mae["abs_sem"],
                    "mae_anom_com": mae["anom_com"],
                    "mae_anom_sem": mae["anom_sem"],
                    "ganho_total": ganho_total,
                    "efeito_centralizacao": efeito_centralizacao,
                    "efeito_retirada_indicadora": efeito_indicadora,
                    "interacao": (
                        ganho_total - efeito_centralizacao - efeito_indicadora
                    ),
                    "efeito_centralizacao_sem_indicadora": (
                        mae["abs_sem"] - mae["anom_sem"]
                    ),
                    "efeito_retirada_indicadora_anomalia": (
                        mae["anom_com"] - mae["anom_sem"]
                    ),
                    "rmse_abs_com": rmse["abs_com"],
                    "rmse_abs_sem": rmse["abs_sem"],
                    "rmse_anom_com": rmse["anom_com"],
                    "rmse_anom_sem": rmse["anom_sem"],
                })
    return pd.DataFrame(linhas)[COLUNAS_ABLACAO]


def relatar_ablacao(ablacao: pd.DataFrame) -> None:
    print()
    print(
        "=== ABLACAO 2x2: FORMA DO ALVO x INDICADORA DE MARATONA "
        "(efeitos em reducao de MAE, min; positivo = melhora) ==="
    )
    efeitos = [
        "ganho_total", "efeito_centralizacao", "efeito_retirada_indicadora",
        "interacao",
    ]
    for centro in (CENTRO_PRIMARIO, CENTRO_SENSIBILIDADE):
        sub_centro = ablacao[ablacao["estatistica_centro"] == centro]
        for familia in FAMILIAS_ABLACAO:
            sub = sub_centro[sub_centro["familia"] == familia]
            print()
            print(f"[{familia} | centro={centro}] MAE das quatro celulas")
            print(
                sub[[
                    "alvo", "mae_abs_com", "mae_abs_sem",
                    "mae_anom_com", "mae_anom_sem",
                ]].to_string(index=False, float_format=lambda v: f"{v:.4f}")
            )
            print(f"[{familia} | centro={centro}] decomposicao do ganho")
            print(
                sub[["alvo"] + efeitos].to_string(
                    index=False, float_format=lambda v: f"{v:+.4f}"
                )
            )
            medias = sub[efeitos].mean()
            print(
                f"  media nos cinco alvos: total {medias['ganho_total']:+.4f}; "
                f"centralizacao {medias['efeito_centralizacao']:+.4f}; "
                f"retirada da indicadora "
                f"{medias['efeito_retirada_indicadora']:+.4f}; "
                f"interacao {medias['interacao']:+.4f}"
            )


def figura_ablacao(ablacao: pd.DataFrame) -> None:
    celulas = {
        "mae_abs_com": ("alvo absoluto, com indicadora", "#c0392b", "s", "-"),
        "mae_abs_sem": ("alvo absoluto, sem indicadora", "#e67e22", "^", "--"),
        "mae_anom_com": ("anomalia, com indicadora", "#2980b9", "v", "--"),
        "mae_anom_sem": ("anomalia, sem indicadora", "#27ae60", "o", "-"),
    }
    sub_primario = ablacao[ablacao["estatistica_centro"] == CENTRO_PRIMARIO]
    familias = list(FAMILIAS_ABLACAO)
    fig, axes = plt.subplots(1, len(familias), figsize=(12, 4.8), sharex=True)
    x = np.arange(len(ALVOS))
    for ax, familia in zip(axes, familias):
        sub = (
            sub_primario[sub_primario["familia"] == familia]
            .set_index("alvo")
            .reindex(ALVOS)
        )
        for coluna, (rotulo, cor, marcador, estilo) in celulas.items():
            ax.plot(x, sub[coluna].values, marker=marcador, ls=estilo,
                    lw=2.0, color=cor, label=rotulo)
        ax.set_title(familia)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in ALVOS])
        ax.set_xlabel("quantil alvo")
        ax.set_ylabel("MAE (min) - LOO")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Ablacao 2x2: forma do alvo x indicadora de maratona "
        "(centro primario por mediana)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_ABLACAO, dpi=140)
    plt.close(fig)
    print(f"[fig] {FIG_ABLACAO}")


def validar_ablacao(ablacao: pd.DataFrame) -> None:
    """Confere a integridade aritmetica e a fonte das celulas do 2x2."""
    esperado = len(FAMILIAS_ABLACAO) * len(ALVOS) * 2
    if len(ablacao) != esperado:
        raise AssertionError(
            f"Ablacao: {len(ablacao)} linhas; esperado {esperado}"
        )
    if ablacao.isna().any().any():
        raise AssertionError("Ablacao: valores ausentes")
    colunas_mae = ["mae_abs_com", "mae_abs_sem", "mae_anom_com", "mae_anom_sem"]
    if not (ablacao[colunas_mae] > 0).all().all():
        raise AssertionError("Ablacao: MAE nao positivo")

    soma = (
        ablacao["efeito_centralizacao"]
        + ablacao["efeito_retirada_indicadora"]
        + ablacao["interacao"]
    )
    if (soma - ablacao["ganho_total"]).abs().max() > 1e-9:
        raise AssertionError("Ablacao: decomposicao nao soma o ganho total")
    caminho_alternativo = (
        ablacao["efeito_retirada_indicadora"]
        + ablacao["efeito_centralizacao_sem_indicadora"]
    )
    if (caminho_alternativo - ablacao["ganho_total"]).abs().max() > 1e-9:
        raise AssertionError("Ablacao: caminhos do 2x2 discordam")

    # As celulas de alvo absoluto nao usam referencia historica: precisam ser
    # identicas nas duas trilhas de centro.
    chaves = ["familia", "alvo"]
    por_centro = ablacao.set_index(["estatistica_centro"] + chaves).sort_index()
    diferenca = (
        por_centro.loc[CENTRO_PRIMARIO, ["mae_abs_com", "mae_abs_sem"]]
        - por_centro.loc[CENTRO_SENSIBILIDADE, ["mae_abs_com", "mae_abs_sem"]]
    )
    if diferenca.abs().to_numpy().max() > 0:
        raise AssertionError(
            "Ablacao: celulas de alvo absoluto divergem entre trilhas de centro"
        )
    # Pontos de controle de reprodutibilidade: medias da decomposicao nos
    # cinco alvos obtidas com o dataset117 publicado. As identidades
    # aritmeticas acima independem destes valores.
    esperado_decomposicao = {'ridge': (1.5531, 1.5589, -4.459, 4.4532),
     'hist_gbm': (1.988, 2.066, -0.3551, 0.2772)}
    efeitos = [
        "ganho_total", "efeito_centralizacao",
        "efeito_retirada_indicadora", "interacao",
    ]
    primario = ablacao[ablacao["estatistica_centro"] == CENTRO_PRIMARIO]
    for familia, esperados in esperado_decomposicao.items():
        medias = primario[primario["familia"] == familia][efeitos].mean()
        for efeito, valor_esperado in zip(efeitos, esperados):
            if abs(float(medias[efeito]) - valor_esperado) > 7e-4:
                raise AssertionError(
                    f"Regressao ablacao/{familia}/{efeito}: "
                    f"{float(medias[efeito]):.6f} != {valor_esperado:.4f}"
                )
    print(
        "[valida] ablacao 2x2: soma dos efeitos, celulas absolutas e "
        "decomposicao media -> OK"
    )


def validar_regressoes(
    resultados_primarios: pd.DataFrame,
    resultados_media: pd.DataFrame,
) -> None:
    """Confere os resultados contra valores de referencia antes de salvar artefatos.

    Os valores fixos abaixo sao os obtidos com o dataset117 publicado; uma
    divergencia acima da tolerancia indica que a entrada ou o ambiente mudou.
    """
    baseline = {
        CENTRO_SENSIBILIDADE: {
            "p10": (5.0343, 7.1057),
            "p25": (6.2803, 8.5503),
            "p50": (8.2009, 10.5667),
            "p75": (9.9261, 12.8888),
            "p90": (11.2331, 15.1752),
        },
        CENTRO_PRIMARIO: {
            "p10": (4.8741, 6.8759),
            "p25": (5.7611, 8.4379),
            "p50": (7.6551, 10.6971),
            "p75": (9.9103, 13.2821),
            "p90": (11.2365, 15.4406),
        },
    }
    ridge_mae = {'media': {'p10': 4.8965,
               'p25': 5.8824,
               'p50': 7.2631,
               'p75': 8.632,
               'p90': 9.8623},
     'mediana': {'p10': 4.8473,
                 'p25': 5.687,
                 'p50': 6.981,
                 'p75': 8.5261,
                 'p90': 9.8273}}
    ridge_delta_rmse = {'p10': -0.0925, 'p25': -0.1346, 'p50': -0.1192, 'p75': -0.1052, 'p90': -0.0535}
    por_centro = {
        CENTRO_PRIMARIO: resultados_primarios,
        CENTRO_SENSIBILIDADE: resultados_media,
    }
    tolerancia = 7e-4

    for centro, esperado_por_alvo in baseline.items():
        resultados = por_centro[centro]
        for alvo, (mae_esperado, rmse_esperado) in esperado_por_alvo.items():
            mae = _valor_resultado(resultados, "A", alvo, "B2_por_maratona", "mae_min")
            rmse = _valor_resultado(resultados, "A", alvo, "B2_por_maratona", "rmse_min")
            if abs(mae - mae_esperado) > tolerancia:
                raise AssertionError(
                    f"Regressao B2/{centro}/{alvo}/MAE: {mae:.6f} != {mae_esperado:.4f}"
                )
            if abs(rmse - rmse_esperado) > tolerancia:
                raise AssertionError(
                    f"Regressao B2/{centro}/{alvo}/RMSE: {rmse:.6f} != {rmse_esperado:.4f}"
                )

    for centro, esperado_por_alvo in ridge_mae.items():
        resultados = por_centro[centro]
        for alvo, esperado in esperado_por_alvo.items():
            mae = _valor_resultado(resultados, "B", alvo, "M2a_ridge_anom", "mae_min")
            if abs(mae - esperado) > tolerancia:
                raise AssertionError(
                    f"Regressao M2a/{centro}/{alvo}/MAE: {mae:.6f} != {esperado:.4f}"
                )

    for alvo, delta_esperado in ridge_delta_rmse.items():
        rmse_mediana = _valor_resultado(
            resultados_primarios, "B", alvo, "M2a_ridge_anom", "rmse_min"
        )
        rmse_media = _valor_resultado(
            resultados_media, "B", alvo, "M2a_ridge_anom", "rmse_min"
        )
        delta = rmse_mediana - rmse_media
        if abs(delta - delta_esperado) > tolerancia:
            raise AssertionError(
                f"Regressao M2a/{alvo}/delta RMSE: {delta:.6f} != {delta_esperado:.4f}"
            )

    # Celulas do protocolo D. As duas de alvo absoluto nao dependem do centro
    # e so existem na trilha primaria.
    ablacao_mae = {'nao_se_aplica': {'M2s_ridge_semcat': {'p10': 7.048,
                                            'p25': 9.163,
                                            'p50': 12.5335,
                                            'p75': 16.8169,
                                            'p90': 20.3674},
                       'M4s_hist_gbm_semcat': {'p10': 5.4563,
                                               'p25': 6.1477,
                                               'p50': 8.2388,
                                               'p75': 10.6947,
                                               'p90': 12.7887}},
     'mediana': {'M2ai_ridge_anom_cat': {'p10': 4.8313,
                                         'p25': 5.6815,
                                         'p50': 6.9799,
                                         'p75': 8.529,
                                         'p90': 9.8181},
                 'M4ai_hist_gbm_anom_cat': {'p10': 4.14,
                                            'p25': 5.135,
                                            'p50': 6.413,
                                            'p75': 7.4964,
                                            'p90': 8.0363}},
     'media': {'M2ai_ridge_anom_cat': {'p10': 4.8806,
                                       'p25': 5.8642,
                                       'p50': 7.2498,
                                       'p75': 8.6204,
                                       'p90': 9.8455},
               'M4ai_hist_gbm_anom_cat': {'p10': 4.2396,
                                          'p25': 5.3238,
                                          'p50': 6.9455,
                                          'p75': 7.327,
                                          'p90': 7.5667}}}
    for centro, por_modelo in ablacao_mae.items():
        resultados = (
            resultados_media
            if centro == CENTRO_SENSIBILIDADE
            else resultados_primarios
        )
        for modelo, por_alvo in por_modelo.items():
            for alvo, esperado in por_alvo.items():
                mae = _valor_resultado(resultados, "D", alvo, modelo, "mae_min")
                if abs(mae - esperado) > tolerancia:
                    raise AssertionError(
                        f"Regressao {modelo}/{centro}/{alvo}/MAE: "
                        f"{mae:.6f} != {esperado:.4f}"
                    )
    print(
        "[valida] pontos de regressao B2, M2a e celulas do protocolo D -> OK"
    )


def validar_cobertura(
    df: pd.DataFrame,
    resultados_primarios: pd.DataFrame,
    predicoes_primarias: pd.DataFrame,
    resultados_media: pd.DataFrame,
    predicoes_media: pd.DataFrame,
) -> None:
    modelos_primarios = {
        "B1_global", "B2_por_maratona", "M1_linear", "M2_ridge",
        "M3_random_forest", "M4_hist_gbm", "M5_mlp",
        "M2a_ridge_anom", "M4a_hist_gbm_anom",
        "M2s_ridge_semcat", "M4s_hist_gbm_semcat",
        "M2ai_ridge_anom_cat", "M4ai_hist_gbm_anom_cat",
    }
    modelos_media = {
        "B2_por_maratona", "M2a_ridge_anom", "M4a_hist_gbm_anom",
        "M2ai_ridge_anom_cat", "M4ai_hist_gbm_anom_cat",
    }
    # primario: A 35 + B 10 + C 8 + D 20; sensibilidade: A 5 + B 10 + C 2 + D 10
    conjuntos = [
        ("primario", resultados_primarios, predicoes_primarias, modelos_primarios, 73),
        ("sensibilidade_media", resultados_media, predicoes_media, modelos_media, 27),
    ]
    esperado_real = df[["maratona", "ano"] + ALVOS].melt(
        id_vars=["maratona", "ano"],
        value_vars=ALVOS,
        var_name="alvo",
        value_name="real_esperado",
    )

    for rotulo, resultados, predicoes, modelos, n_resultados in conjuntos:
        if len(resultados) != n_resultados:
            raise AssertionError(
                f"{rotulo}: {len(resultados)} resultados; esperado {n_resultados}"
            )
        if resultados.duplicated(["protocolo", "alvo", "modelo"]).any():
            raise AssertionError(f"{rotulo}: resultados duplicados")
        if predicoes.isna().any().any() or resultados.isna().any().any():
            raise AssertionError(f"{rotulo}: valores ausentes inesperados")
        if set(predicoes["modelo"].unique()) != modelos:
            raise AssertionError(f"{rotulo}: conjunto de modelos inesperado")
        contagens = predicoes.groupby(["modelo", "alvo"]).size()
        if len(contagens) != len(modelos) * len(ALVOS) or not contagens.eq(117).all():
            raise AssertionError(f"{rotulo}: cobertura diferente de 117 por modelo/alvo")
        if predicoes.duplicated(["maratona", "ano", "alvo", "modelo"]).any():
            raise AssertionError(f"{rotulo}: predicoes duplicadas")

        conferencia = predicoes.merge(
            esperado_real,
            on=["maratona", "ano", "alvo"],
            how="left",
            validate="many_to_one",
        )
        if conferencia["real_esperado"].isna().any():
            raise AssertionError(f"{rotulo}: edicao/alvo fora do dataset de entrada")
        max_dif = (conferencia["real_min"] - conferencia["real_esperado"]).abs().max()
        if max_dif > 1e-12:
            raise AssertionError(f"{rotulo}: real_min diverge do alvo absoluto")

    afetados = {
        "B2_por_maratona", "M2a_ridge_anom", "M4a_hist_gbm_anom",
        "M2ai_ridge_anom_cat", "M4ai_hist_gbm_anom_cat",
    }
    centro_primario = predicoes_primarias["modelo"].isin(afetados)
    if not predicoes_primarias.loc[centro_primario, "estatistica_centro"].eq(
        CENTRO_PRIMARIO
    ).all():
        raise AssertionError("Predicoes primarias sem rotulo de mediana")
    if not predicoes_primarias.loc[~centro_primario, "estatistica_centro"].eq(
        CENTRO_NAO_APLICAVEL
    ).all():
        raise AssertionError("Modelos absolutos receberam rotulo historico indevido")
    if not predicoes_media["estatistica_centro"].eq(CENTRO_SENSIBILIDADE).all():
        raise AssertionError("Sensibilidade sem rotulo de media")
    print("[valida] cobertura, unicidade, escala absoluta e rotulos -> OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    df, cols_num, cols_cat = carregar_dataset()

    # Protocolo primario: mediana historica do mesmo alvo, sempre ajustada no treino.
    res_a, pred_a = protocolo_a(
        df, cols_num, cols_cat, estatistica_centro=CENTRO_PRIMARIO
    )
    res_b, pred_b = protocolo_b(
        df, cols_num, estatistica_centro=CENTRO_PRIMARIO
    )
    res_c = protocolo_c(
        df, cols_num, cols_cat, estatistica_centro=CENTRO_PRIMARIO
    )
    res_d, pred_d = protocolo_d(
        df, cols_num, cols_cat, estatistica_centro=CENTRO_PRIMARIO
    )

    # Analise de sensibilidade: repete somente os metodos afetados pelo centro,
    # com os mesmos folds, features, sementes e hiperparametros.
    res_a_media, pred_a_media = protocolo_a_baseline_historica(
        df, CENTRO_SENSIBILIDADE
    )
    res_b_media, pred_b_media = protocolo_b(
        df, cols_num, estatistica_centro=CENTRO_SENSIBILIDADE
    )
    res_c_media = protocolo_c(
        df,
        cols_num,
        cols_cat,
        estatistica_centro=CENTRO_SENSIBILIDADE,
        modelos_c=["B2_por_maratona"],
    )
    res_d_media, pred_d_media = protocolo_d(
        df,
        cols_num,
        cols_cat,
        estatistica_centro=CENTRO_SENSIBILIDADE,
        incluir_absolutos=False,
    )

    resultados = pd.concat(
        [res_a, res_b, res_c, res_d], ignore_index=True
    )[COLUNAS_RESULTADOS]
    predicoes = pd.concat(
        [pred_a, pred_b, pred_d], ignore_index=True
    )[COLUNAS_PREDICOES]
    resultados_media = pd.concat(
        [res_a_media, res_b_media, res_c_media, res_d_media], ignore_index=True
    )[COLUNAS_RESULTADOS]
    predicoes_media = pd.concat(
        [pred_a_media, pred_b_media, pred_d_media], ignore_index=True
    )[COLUNAS_PREDICOES]
    comparacao = comparar_centralizacoes(resultados, resultados_media)
    ablacao = decompor_ablacao(resultados, resultados_media)

    # Nenhum artefato e substituido antes de a execucao completa passar pelos
    # pontos de controle de regressao e pelas invariantes de cobertura.
    validar_regressoes(resultados, resultados_media)
    validar_cobertura(
        df, resultados, predicoes, resultados_media, predicoes_media
    )
    validar_ablacao(ablacao)

    resultados.to_csv(SAIDA_RESULTADOS, index=False)
    predicoes.to_csv(SAIDA_PREDICOES, index=False)
    resultados_media.to_csv(SAIDA_RESULTADOS_MEDIA, index=False)
    predicoes_media.to_csv(SAIDA_PREDICOES_MEDIA, index=False)
    comparacao.to_csv(SAIDA_COMPARACAO_CENTRO, index=False)
    ablacao.to_csv(SAIDA_ABLACAO, index=False)
    print()
    print(f"[csv] {SAIDA_RESULTADOS} ({len(resultados)} linhas)")
    print(f"[csv] {SAIDA_PREDICOES} ({len(predicoes)} linhas)")
    print(f"[csv] {SAIDA_RESULTADOS_MEDIA} ({len(resultados_media)} linhas)")
    print(f"[csv] {SAIDA_PREDICOES_MEDIA} ({len(predicoes_media)} linhas)")
    print(f"[csv] {SAIDA_COMPARACAO_CENTRO} ({len(comparacao)} linhas)")
    print(f"[csv] {SAIDA_ABLACAO} ({len(ablacao)} linhas)")

    print()
    print("=== TABELA CONSOLIDADA DE METRICAS ===")
    tab = resultados.copy()
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    relatar_deltas_e_sintese(res_a, res_b)
    relatar_sensibilidade(comparacao)
    relatar_ablacao(ablacao)
    figura_mae_por_quantil(res_a, res_b)
    figura_sensibilidade_centralizacao(comparacao)
    figura_ablacao(ablacao)

    print()
    print("[ok] treinamento de quantis por edicao concluido")


if __name__ == "__main__":
    main()
