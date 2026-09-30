"""
ETL do dataset unico de maratonas.

Le os resultados individuais de cada maratona, aplica o recorte de edicoes e
os filtros de qualidade, junta os agregados meteorologicos (ERA5 via
Open-Meteo, janela de seis horas a partir da largada) e grava o dataset
canônico no nivel atleta-edicao.

Entradas:
  - dados/brutos/maratonas/<maratona>/                  (resultados individuais por maratona)
  - dados/brutos/kaggle_temp/Results.csv                (Fall Marathons: Maui, Savannah, Long Beach)
  - dados/brutos/maratonas_datas.csv                    (data, coordenadas e fuso de cada edicao)
  - dados/processados/apoio/agregados_horarios.csv      (gerado por codigo/cap4/notebooks/03_agregados_horarios.py)

Saidas:
  - dados/processados/marathon_canonical.csv
  - dados/processados/marathon_climate.csv              (visao legado, colunas reduzidas)
  - dados/processados/apoio/*.csv                       (manifestos de fontes, mapeamento de colunas e exclusoes)

Escopo do canônico:
  - apenas maratonas standalone de 42.195 km com data de prova local,
    metadados de evento definidos neste script e resultados individuais
    disponiveis em dados/brutos/.
  - inclui Berlin, Boston, Chicago, NYC, Honolulu, Singapore, Maui,
    Savannah, Long Beach, Rio e Durban.

Escopo de apoio:
  - exporta manifestos e uma visao padronizada das Fall Marathons
    selecionadas, ja integrada ao canônico.

Execucao:
    python codigo/etl/etl_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BRUTOS = ROOT / "dados" / "brutos"
MARATONAS_DIR = BRUTOS / "maratonas"
DATAS_CSV = BRUTOS / "maratonas_datas.csv"
FALL_CSV = BRUTOS / "kaggle_temp" / "Results.csv"

PROCESSADOS = ROOT / "dados" / "processados"
APOIO_DIR = PROCESSADOS / "apoio"
METEO_CSV = APOIO_DIR / "agregados_horarios.csv"

OUT_CANONICAL = PROCESSADOS / "marathon_canonical.csv"
OUT_LEGACY = PROCESSADOS / "marathon_climate.csv"
OUT_FALL_SUPPORT = APOIO_DIR / "fall_marathons_selecionadas_padronizado.csv"

# 2020 e 2021 ficam fora: edicoes canceladas ou com capacidade reduzida por COVID.
EXCLUIR_ANOS = {2020, 2021}
JANELA_CLIMATICA_HORAS = 6

MAPEAMENTO_COLUNAS_METEO = {
    "temp_media": "temp_media",
    "temp_max": "temp_max",
    "temp_min": "temp_min",
    "bulbo_umido_med": "bulbo_umido_med",
    "umidade_med": "umidade_media_pct",
    "orvalho_med": "orvalho_medio_c",
    "temp_aparente_med": "temp_aparente_med",
    "vento_med_ms": "vento_med_ms",
    "vento_max_ms": "vento_max_ms",
    "rajada_max_ms": "rajada_max_ms",
    "precipitacao_total_mm": "precipitacao_mm",
    "radiacao_med_wm2": "radiacao_media_wm2",
}

# Codificacao ciclica do dia do ano (mesma convencao de
# codigo/cap4/notebooks/11_dataset_treino_edicoes.py): angulo =
# 2*pi*dia_ano/PERIODO_ANO_DIAS, com correcao de hemisferio (sinal -1 em sin e
# cos) para maratonas do sul, alinhando estacao em vez de calendario civil.
PERIODO_ANO_DIAS = 365.2425
MARATONAS_HEMISFERIO_SUL = {"rio", "durban"}
FALL_FONTE = "kaggle:runningwithrock/2010-2019-fall-marathons"
FALL_TIPO_COLETA = "dataset_publico_kaggle"
SINGAPORE_FONTE = "sportsplits:singapore-marathon"
SINGAPORE_TIPO_COLETA = "scraping_resultados_web"
RIO_FONTE = "oficial:maratona-do-rio"
RIO_TIPO_COLETA = "coleta_oficial_api_html_pdf"
DURBAN_FONTE = "oficial:durban-international-marathon"
DURBAN_TIPO_COLETA = "coleta_oficial_pdf_html_xlsx"
BERLIN_2019_ANO = 2019
BERLIN_2019_CSV = MARATONAS_DIR / "berlin" / "marathon-results_berlin_2019.csv"
BERLIN_2019_FONTE = "github:AndrewMillerOnline/marathon-results"
BERLIN_2019_TIPO_COLETA = "dataset_publico_github"
MARATONAS_CANONICAS = [
    "berlin",
    "boston",
    "chicago",
    "nyc",
    "honolulu",
    "singapore",
    "maui",
    "savannah",
    "long_beach",
    "rio",
    "durban",
]

ESCOPO = {
    "berlin": (2005, 2019),
    "boston": (2005, 2019),
    "chicago": (2005, 2023),
    "nyc": (2005, 2024),
    "honolulu": (2005, 2019),
    "singapore": (2014, 2023),
    "maui": (2010, 2014),
    "savannah": (2011, 2019),
    "long_beach": (2010, 2019),
    "rio": (2023, 2025),
    "durban": (2022, 2025),
}

META_EVENTO = {
    "berlin": {
        "cidade": "Berlim",
        "pais_evento": "Alemanha",
        "fonte": "kaggle:aiaiaidavid/berlin-marathons-data",
        "tipo_coleta": "dataset_publico_kaggle",
    },
    "boston": {
        "cidade": "Boston",
        "pais_evento": "Estados Unidos",
        "fonte": "github:adrian3/Boston-Marathon-Data-Project",
        "tipo_coleta": "dataset_publico_github",
    },
    "chicago": {
        "cidade": "Chicago",
        "pais_evento": "Estados Unidos",
        "fonte": "kaggle:runningwithrock/chicago-marathon-results",
        "tipo_coleta": "dataset_publico_kaggle",
    },
    "nyc": {
        "cidade": "Nova York",
        "pais_evento": "Estados Unidos",
        "fonte": "kaggle:runningwithrock/nyc-marathon-results-all-years",
        "tipo_coleta": "dataset_publico_kaggle",
    },
    "honolulu": {
        "cidade": "Honolulu",
        "pais_evento": "Estados Unidos",
        "fonte": "zenodo:10.5281/zenodo.6959864",
        "tipo_coleta": "dataset_publico_zenodo",
    },
    "singapore": {
        "cidade": "Singapura",
        "pais_evento": "Singapura",
        "fonte": SINGAPORE_FONTE,
        "tipo_coleta": SINGAPORE_TIPO_COLETA,
    },
    "maui": {
        "cidade": "Maui",
        "pais_evento": "Estados Unidos",
        "fonte": FALL_FONTE,
        "tipo_coleta": FALL_TIPO_COLETA,
    },
    "savannah": {
        "cidade": "Savannah",
        "pais_evento": "Estados Unidos",
        "fonte": FALL_FONTE,
        "tipo_coleta": FALL_TIPO_COLETA,
    },
    "long_beach": {
        "cidade": "Long Beach",
        "pais_evento": "Estados Unidos",
        "fonte": FALL_FONTE,
        "tipo_coleta": FALL_TIPO_COLETA,
    },
    "rio": {
        "cidade": "Rio de Janeiro",
        "pais_evento": "Brasil",
        "fonte": RIO_FONTE,
        "tipo_coleta": RIO_TIPO_COLETA,
    },
    "durban": {
        "cidade": "Durban",
        "pais_evento": "Africa do Sul",
        "fonte": DURBAN_FONTE,
        "tipo_coleta": DURBAN_TIPO_COLETA,
    },
}

META_FALL = {
    "Maui Marathon": {
        "maratona": "maui",
        "cidade": "Maui",
        "pais_evento": "Estados Unidos",
    },
    "RnR Savannah Marathon": {
        "maratona": "savannah",
        "cidade": "Savannah",
        "pais_evento": "Estados Unidos",
    },
    "Long Beach International City Marathon": {
        "maratona": "long_beach",
        "cidade": "Long Beach",
        "pais_evento": "Estados Unidos",
    },
    "Long Beach Marathon": {
        "maratona": "long_beach",
        "cidade": "Long Beach",
        "pais_evento": "Estados Unidos",
    },
}

COLUNAS_CANONICAS = [
    "evento_id",
    "maratona",
    "ano",
    "data_prova",
    # Codificacao ciclica do dia do ano da prova, com correcao de hemisferio.
    # Fica junto ao bloco de prova/edicao, logo apos data_prova, por ser
    # derivada apenas dela e da maratona (ver anexar_sazonalidade_dia_ano).
    "dia_ano_sin",
    "dia_ano_cos",
    "cidade",
    "pais_evento",
    "fonte",
    "tipo_coleta",
    "genero",
    "tempo_segundos",
    "tempo_original",
    "pais_atleta",
    "distancia_km",
    "clima_disponivel",
    "temp_media",
    "temp_max",
    "temp_min",
    "bulbo_umido_med",
    "umidade_media_pct",
    "orvalho_medio_c",
    "temp_aparente_med",
    "vento_med_ms",
    "vento_max_ms",
    "rajada_max_ms",
    "precipitacao_mm",
    "radiacao_media_wm2",
    # Posicao relativa do atleta na sua edicao; coluna derivada, calculada
    # apos os filtros de qualidade por anexar_percentil_edicao.
    "percentil_edicao",
]

COLUNAS_LEGADO = [
    "maratona",
    "ano",
    "genero",
    "tempo_segundos",
    "pais",
    "temp_media",
    "temp_max",
    "temp_min",
    "bulbo_umido_med",
    "umidade_media_pct",
    "orvalho_medio_c",
    "temp_aparente_med",
    "vento_med_ms",
    "vento_max_ms",
    "rajada_max_ms",
    "precipitacao_mm",
    "radiacao_media_wm2",
]

MANIFESTO_FONTES = [
    {
        "fonte_id": "berlin_kaggle",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "berlin",
        "origem_url": "https://www.kaggle.com/datasets/aiaiaidavid/berlin-marathons-data",
        "licenca": "CC0",
        "caminho_local": "dados/brutos/maratonas/berlin/Berlin_Marathon_data_1974_2019.csv",
        "observacoes": "Resultados individuais de 2005 a 2018; em 2019 a fonte traz apenas registros masculinos e e substituida por berlin_github_2019.",
    },
    {
        "fonte_id": "berlin_github_2019",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "berlin",
        "origem_url": "https://github.com/AndrewMillerOnline/marathon-results/blob/2e6219da8dcb4b1ef51a78b9412e5ab4aa9e85af/Berlin/results-2019.csv",
        "licenca": "MIT",
        "caminho_local": "dados/brutos/maratonas/berlin/marathon-results_berlin_2019.csv",
        "observacoes": "Edicao de 2019 completa (homens e mulheres); 2018 da mesma fonte coincide com o Kaggle.",
    },
    {
        "fonte_id": "boston_github",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "boston",
        "origem_url": "https://github.com/adrian3/Boston-Marathon-Data-Project",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/boston/results*.csv",
        "observacoes": "Dois schemas distintos entre 2005-2014 e 2015-2019.",
    },
    {
        "fonte_id": "chicago_kaggle",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "chicago",
        "origem_url": "https://www.kaggle.com/datasets/runningwithrock/chicago-marathon-results",
        "licenca": "MIT",
        "caminho_local": "dados/brutos/maratonas/chicago/Chicago Marathon Results.csv",
        "observacoes": "Campo Finish em segundos e Finish Time em HH:MM:SS.",
    },
    {
        "fonte_id": "nyc_kaggle",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "nyc",
        "origem_url": "https://www.kaggle.com/datasets/runningwithrock/nyc-marathon-results-all-years",
        "licenca": "MIT",
        "caminho_local": "dados/brutos/maratonas/nyc/NYC Marathon Results.csv",
        "observacoes": "Campo Finish em segundos e Finish Time em HH:MM:SS.",
    },
    {
        "fonte_id": "honolulu_zenodo",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "honolulu",
        "origem_url": "https://zenodo.org/records/6959864",
        "licenca": "CC BY 4.0",
        "caminho_local": "dados/brutos/maratonas/zenodo_albrecht/all_marathons_final.csv",
        "observacoes": "Country veio fragmentado como endereco e foi descartado no canônico.",
    },
    {
        "fonte_id": "singapore_sportsplits",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "singapore",
        "origem_url": "https://www.sportsplits.com/",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/singapore/Singapore Marathon Results.csv",
        "observacoes": "Datas confirmadas no cabecalho do evento; canonico usa net_time, com gun_time como alternativa quando o tempo liquido esta ausente.",
    },
    {
        "fonte_id": "fall_kaggle",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "maui|savannah|long_beach",
        "origem_url": "https://www.kaggle.com/datasets/runningwithrock/2010-2019-fall-marathons",
        "licenca": "MIT",
        "caminho_local": "dados/brutos/kaggle_temp/Results.csv",
        "observacoes": "Integrado ao canonico apos registrar datas exatas por edicao e resolver o join meteorologico local.",
    },
    {
        "fonte_id": "rio_chiptiming",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "rio",
        "origem_url": "https://maratonadorio.com.br/resultados",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/rio/rio_2023_oficial.csv; dados/brutos/maratonas/rio/rio_2024_oficial.csv",
        "observacoes": "Fluxo oficial do front-end via metadado do evento, catalogo de resultados e listagens paginadas; 2023 inclui elite extraida de PDF oficial.",
    },
    {
        "fonte_id": "rio_runking",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "rio",
        "origem_url": "https://resultados.runking.com.br/maratona-do-rio/maratona-do-rio-2025",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/rio/rio_2025_oficial.csv",
        "observacoes": "Paginacao HTML explicita por page e limit; payload de resultados vem cifrado no HTML e foi reproduzido localmente.",
    },
    {
        "fonte_id": "durban_2022_pdf",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "durban",
        "origem_url": "https://durbanmarathon.co.za/wp-content/uploads/2023/03/Durban-International-Marathon-42km-2022.pdf",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/durban/durban_2022_oficial.csv",
        "observacoes": "PDF oficial com sobreposicao entre 42km e SA Marathon Champs; consolidado local deduplicado antes da integracao.",
    },
    {
        "fonte_id": "durban_2024_finishtime",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "durban",
        "origem_url": "https://results.finishtime.co.za/results.aspx?CId=35&RId=4552&EId=1&dt=0",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/durban/durban_2024_oficial.csv",
        "observacoes": "Listagem paginada oficial 42.2km coletada via navegador por causa da protecao anti-bot da pagina.",
    },
    {
        "fonte_id": "durban_2025_xlsx",
        "status": "integrado_canonico",
        "categoria": "resultados",
        "maratona": "durban",
        "origem_url": "https://durbanmarathon.co.za/wp-content/uploads/2025/05/DIM2025-42km-results.xlsx",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "dados/brutos/maratonas/durban/durban_2025_oficial.csv",
        "observacoes": "Planilha oficial 42km com data de nascimento, categoria e pais de residencia.",
    },
    {
        "fonte_id": "open_meteo_era5",
        "status": "integrado_canonico",
        "categoria": "meteorologia",
        "maratona": "todas_integradas",
        "origem_url": "https://archive-api.open-meteo.com/v1/archive",
        "licenca": "CC BY 4.0",
        "caminho_local": "dados/processados/apoio/agregados_horarios.csv",
        "observacoes": "Agregados meteorologicos integrados ao canonico pela janela unica entre a largada de cada edicao e as seis horas seguintes.",
    },
    {
        "fonte_id": "london_zenodo",
        "status": "avaliado_nao_integrado",
        "categoria": "candidata_resultados",
        "maratona": "london",
        "origem_url": "https://zenodo.org/records/10960982",
        "licenca": "CC BY 4.0",
        "caminho_local": "",
        "observacoes": "Dataset 2018-2023 sem 2020, sem elite e sem nao concluintes; estrutura restrita para o escopo atual.",
    },
    {
        "fonte_id": "lagos_official_results",
        "status": "avaliado_nao_integrado",
        "categoria": "candidata_quente",
        "maratona": "lagos",
        "origem_url": "https://lagoscitymarathon.com/race-results/",
        "licenca": "nao informada / sem licenca aberta explicita na fonte",
        "caminho_local": "",
        "observacoes": "Pagina oficial agrega links por ano/genero/categoria; serie fragmentada demais para integrar resultados individuais de forma consistente.",
    },
    {
        "fonte_id": "polar_circle_official",
        "status": "avaliado_nao_integrado",
        "categoria": "candidata_fria",
        "maratona": "polar_circle",
        "origem_url": "https://polar-circle-marathon.com/race-results",
        "licenca": "nao informada",
        "caminho_local": "",
        "observacoes": "Arquivo oficial de resultados por ano; fonte avaliada e nao integrada.",
    },
    {
        "fonte_id": "antarctic_ice_official",
        "status": "avaliado_nao_integrado",
        "categoria": "candidata_fria",
        "maratona": "antarctic_ice",
        "origem_url": "https://www.icemarathon.com/years/2024",
        "licenca": "nao informada",
        "caminho_local": "",
        "observacoes": "Competidores e resultados por edicao, mas base temporaria dificulta reproducao meteorologica.",
    },
    {
        "fonte_id": "north_pole_official",
        "status": "avaliado_nao_integrado",
        "categoria": "candidata_fria",
        "maratona": "north_pole",
        "origem_url": "https://www.npmarathon.com/years/2025",
        "licenca": "nao informada",
        "caminho_local": "",
        "observacoes": "Competidores e resultados por edicao, com percurso sobre gelo marinho movel.",
    },
    {
        "fonte_id": "antarctica_marathon_official",
        "status": "avaliado_nao_integrado",
        "categoria": "candidata_fria",
        "maratona": "antarctica_marathon",
        "origem_url": "https://marathontours.com/en-us/antarctica-marathon-results/",
        "licenca": "nao informada",
        "caminho_local": "",
        "observacoes": "Resultado oficial resumido e link para resultados completos; evento organizado em viagens (voyages) em alguns anos.",
    },
    {
        "fonte_id": "ironman_results_official",
        "status": "exploratorio_nao_integrado",
        "categoria": "trilha_ironman",
        "maratona": "ironman",
        "origem_url": "https://www.ironman.com/races/xclusive-challenge/race/results",
        "licenca": "nao informada",
        "caminho_local": "",
        "observacoes": "Resultados oficiais exibem colunas de split incluindo RUN para provas IRONMAN.",
    },
]

MAPEAMENTO_COLUNAS = [
    ("berlin_kaggle", "YEAR", "canonica", "ano", "copiar"),
    ("berlin_kaggle", "GENDER", "canonica", "genero", "male/female -> M/F"),
    ("berlin_kaggle", "TIME", "canonica", "tempo_original", "copiar"),
    ("berlin_kaggle", "TIME", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("berlin_kaggle", "COUNTRY", "canonica", "pais_atleta", "trim; muitos NaN"),
    ("berlin_github_2019", "gender", "canonica", "genero", "M/W -> M/F"),
    ("berlin_github_2019", "time_full", "canonica", "tempo_original", "copiar"),
    ("berlin_github_2019", "time_full", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("berlin_github_2019", "nationality", "canonica", "pais_atleta", "trim"),
    ("boston_github_2005_2014", "gender", "canonica", "genero", "copiar"),
    ("boston_github_2005_2014", "official_time", "canonica", "tempo_original", "copiar"),
    ("boston_github_2005_2014", "seconds", "canonica", "tempo_segundos", "copiar"),
    ("boston_github_2005_2014", "residence", "apoio", "residence_fonte", "manter apenas em apoio se necessario"),
    ("boston_github_2015_2019", "gender", "canonica", "genero", "copiar"),
    ("boston_github_2015_2019", "official_time", "canonica", "tempo_original", "copiar"),
    ("boston_github_2015_2019", "seconds", "canonica", "tempo_segundos", "copiar"),
    ("boston_github_2015_2019", "country_residence", "canonica", "pais_atleta", "copiar"),
    ("boston_github_2015_2019", "5k..40k", "apoio", "splits_boston", "nao entram no canônico"),
    ("chicago_kaggle", "Year", "canonica", "ano", "copiar"),
    ("chicago_kaggle", "Gender", "canonica", "genero", "copiar"),
    ("chicago_kaggle", "Finish Time", "canonica", "tempo_original", "copiar"),
    ("chicago_kaggle", "Finish", "canonica", "tempo_segundos", "copiar"),
    ("chicago_kaggle", "Country", "canonica", "pais_atleta", "copiar"),
    ("chicago_kaggle", "Overall", "apoio", "colocacao_geral", "nao entra no canônico"),
    ("nyc_kaggle", "Year", "canonica", "ano", "copiar"),
    ("nyc_kaggle", "Gender", "canonica", "genero", "M/W -> M/F; X excluido"),
    ("nyc_kaggle", "Finish Time", "canonica", "tempo_original", "copiar"),
    ("nyc_kaggle", "Finish", "canonica", "tempo_segundos", "copiar"),
    ("nyc_kaggle", "Country", "canonica", "pais_atleta", "copiar"),
    ("nyc_kaggle", "State", "apoio", "estado_fonte", "nao entra no canônico"),
    ("honolulu_zenodo", "Year", "canonica", "ano", "copiar"),
    ("honolulu_zenodo", "Sex", "canonica", "genero", "copiar"),
    ("honolulu_zenodo", "Gun Time", "canonica", "tempo_original", "copiar"),
    ("honolulu_zenodo", "Time", "canonica", "tempo_segundos", "minutos -> segundos"),
    ("honolulu_zenodo", "Country", "descartado", "", "campo inutilizavel como pais do atleta"),
    ("singapore_sportsplits", "ano", "canonica", "ano", "copiar"),
    ("singapore_sportsplits", "genero", "canonica", "genero", "copiar"),
    ("singapore_sportsplits", "net_time", "canonica", "tempo_original", "usar net_time; fallback para gun_time se necessario"),
    ("singapore_sportsplits", "net_time|gun_time", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("singapore_sportsplits", "representing", "canonica", "pais_atleta", "sigla de pais/representacao preservada da fonte"),
    ("fall_kaggle", "Year", "canonica", "ano", "copiar"),
    ("fall_kaggle", "Gender", "canonica", "genero", "copiar"),
    ("fall_kaggle", "Finish", "canonica", "tempo_segundos", "copiar"),
    ("fall_kaggle", "Finish", "apoio", "tempo_original", "mantido como string do valor bruto"),
    ("fall_kaggle", "Age Bracket", "apoio", "faixa_etaria_fonte", "nao entra no canônico"),
    ("fall_kaggle", "Race", "apoio", "nome_evento_fonte", "usado para alias e recorte"),
    ("rio_chiptiming", "ano", "canonica", "ano", "copiar"),
    ("rio_chiptiming", "genero", "canonica", "genero", "copiar"),
    ("rio_chiptiming", "tempo_liquido|tempo_bruto", "canonica", "tempo_original", "preferir tempo liquido; fallback para bruto"),
    ("rio_chiptiming", "tempo_liquido|tempo_bruto", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("rio_chiptiming", "pais_atleta", "canonica", "pais_atleta", "apenas elite 2023 traz nacionalidade explicita"),
    ("rio_runking", "genero", "canonica", "genero", "derivado do filtro F/M da URL"),
    ("rio_runking", "tempo_liquido|tempo_bruto", "canonica", "tempo_original", "usar liquidTime; fallback para rawTime"),
    ("rio_runking", "tempo_liquido|tempo_bruto", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("rio_runking", "pais_atleta", "canonica", "pais_atleta", "sigla de nacionalidade extraida do HTML oficial"),
    ("durban_2022_pdf", "tempo_bruto", "canonica", "tempo_original", "copiar do PDF 42km apos deduplicacao"),
    ("durban_2022_pdf", "tempo_bruto", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("durban_2024_finishtime", "tempo_liquido|tempo_bruto", "canonica", "tempo_original", "preferir net time; fallback para bruto"),
    ("durban_2024_finishtime", "tempo_liquido|tempo_bruto", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("durban_2025_xlsx", "Net time|Finish time", "canonica", "tempo_original", "preferir net time; fallback para finish time"),
    ("durban_2025_xlsx", "Net time|Finish time", "canonica", "tempo_segundos", "HH:MM:SS -> segundos"),
    ("durban_2025_xlsx", "Resident country", "canonica", "pais_atleta", "copiar"),
]

ALIASES_EXCLUSOES = [
    {
        "tipo": "escopo",
        "alvo": "tokyo",
        "acao": "descartado",
        "status": "fora_do_canonico",
        "motivo": "Scraping dificil e ganho pequeno de variabilidade em relacao ao corpus ja coletado.",
    },
    {
        "tipo": "escopo",
        "alvo": "paris",
        "acao": "descartado",
        "status": "fora_do_canonico",
        "motivo": "Scraping fragmentado e faixa climatica redundante com o conjunto ja integrado.",
    },
    {
        "tipo": "escopo",
        "alvo": "polar_circle",
        "acao": "avaliado_nao_integrado",
        "status": "avaliado_extra",
        "motivo": "Arquivo oficial de resultados e local terrestre estavel em Kangerlussuaq.",
    },
    {
        "tipo": "escopo",
        "alvo": "north_pole",
        "acao": "avaliado_nao_integrado",
        "status": "avaliado_extra",
        "motivo": "Percurso sobre gelo marinho movel reduz comparabilidade meteorologica.",
    },
    {
        "tipo": "escopo",
        "alvo": "antarctic_ice",
        "acao": "avaliado_nao_integrado",
        "status": "avaliado_extra",
        "motivo": "Base temporaria e campo muito pequeno dificultam reproducao de clima por edicao.",
    },
    {
        "tipo": "escopo",
        "alvo": "antarctica_marathon",
        "acao": "avaliado_nao_integrado",
        "status": "avaliado_extra",
        "motivo": "Resultados existem, mas a organizacao em viagens (voyages) exige regra adicional para nao misturar edicoes.",
    },
    {
        "tipo": "alias",
        "alvo": "Long Beach International City Marathon -> Long Beach Marathon",
        "acao": "unificar_em_long_beach",
        "status": "integrado_canonico",
        "motivo": "Mesma prova com nome diferente entre 2010-2013 e 2014-2019.",
    },
    {
        "tipo": "ano",
        "alvo": "2020",
        "acao": "excluir",
        "status": "fora_do_canonico",
        "motivo": "Ano cancelado ou metodologicamente inconsistente por COVID.",
    },
    {
        "tipo": "ano",
        "alvo": "2021",
        "acao": "excluir",
        "status": "fora_do_canonico",
        "motivo": "Capacidade reduzida e vies de selecao em maratonas presenciais pos-COVID.",
    },
    {
        "tipo": "linha",
        "alvo": "nyc.Gender in {X, NaN}",
        "acao": "excluir",
        "status": "fora_do_canonico",
        "motivo": "Nao ha regra consolidada para comparar com os demais datasets atuais.",
    },
    {
        "tipo": "campo",
        "alvo": "honolulu.Country",
        "acao": "descartar_campo",
        "status": "integrado_canonico",
        "motivo": "Valores sao fragmentos de endereco e nao pais do atleta.",
    },
    {
        "tipo": "ano",
        "alvo": "singapore_2024",
        "acao": "nao_integrar",
        "status": "fora_do_canonico",
        "motivo": "URLs provaveis do SportSplits retornaram 404 na coleta; sem bruto local confirmado.",
    },
    {
        "tipo": "regra",
        "alvo": "rio",
        "acao": "geral_por_genero_mais_elite_complementar",
        "status": "integrado_canonico",
        "motivo": "Canonico usa geral masculino + geral feminino; elite entra apenas quando houver atleta ausente das listagens gerais.",
    },
    {
        "tipo": "escopo",
        "alvo": "lagos",
        "acao": "nao_integrado",
        "status": "fora_do_canonico",
        "motivo": "Serie localizada continua fragmentada em top-10, top-20 e planilhas curtas, sem resultados individuais completos.",
    },
    {
        "tipo": "ano",
        "alvo": "durban_2023",
        "acao": "nao_integrar",
        "status": "fora_do_canonico",
        "motivo": "A evidencia localizada permaneceu restrita a top por categoria, sem resultados individuais completos.",
    },
    {
        "tipo": "campo",
        "alvo": "durban_2022_pdf",
        "acao": "deduplicar_42km_vs_sa_champs",
        "status": "integrado_canonico",
        "motivo": "PDF oficial mistura a linha geral 42km com a camada SA Marathon Champs; o bruto local foi colapsado para uma linha por finisher.",
    },
    {
        "tipo": "linha",
        "alvo": "nyc_2012",
        "acao": "excluir",
        "status": "fora_do_canonico",
        "motivo": "Edicao cancelada pelo furacao Sandy.",
    },
]

AVALIACAO_FRIAS = [
    {
        "maratona": "polar_circle",
        "resultado_oficial_url": "https://polar-circle-marathon.com/race-results",
        "anos_mencionados_oficialmente": "2001-2025",
        "campos_observados": "links anuais de resultados; volume anual; local Kangerlussuaq",
        "localizacao": "Kangerlussuaq, Groenlandia",
        "clima_reproduzivel": "sim, em principio",
        "decisao": "avaliada_nao_integrada",
        "motivo": "Melhor combinacao de frio extremo, local terrestre estavel e arquivo oficial de resultados.",
    },
    {
        "maratona": "antarctic_ice",
        "resultado_oficial_url": "https://www.icemarathon.com/years/2024",
        "anos_mencionados_oficialmente": "2005-2026",
        "campos_observados": "competidores e resultados por ano",
        "localizacao": "base temporaria na Antartida",
        "clima_reproduzivel": "nao, ou apenas com alto esforco adicional",
        "decisao": "avaliada_nao_integrada",
        "motivo": "Campo muito pequeno e dificuldade de reproduzir coordenadas/meteorologia por edicao.",
    },
    {
        "maratona": "north_pole",
        "resultado_oficial_url": "https://www.npmarathon.com/years/2025",
        "anos_mencionados_oficialmente": "2003-2026",
        "campos_observados": "competidores por ano e secao de race results",
        "localizacao": "gelo marinho no Polo Norte",
        "clima_reproduzivel": "nao",
        "decisao": "avaliada_nao_integrada",
        "motivo": "Localizacao operacionalmente movel reduz comparabilidade com maratonas terrestres.",
    },
    {
        "maratona": "antarctica_marathon",
        "resultado_oficial_url": "https://marathontours.com/en-us/antarctica-marathon-results/",
        "anos_mencionados_oficialmente": "1995-2025",
        "campos_observados": "vencedores, temperatura, vento, numero de participantes e link para resultados completos",
        "localizacao": "King George Island, Antartida",
        "clima_reproduzivel": "parcial",
        "decisao": "avaliada_nao_integrada",
        "motivo": "Evento muito pequeno; a organizacao em viagens (voyages) exige tratamento adicional antes de integrar.",
    },
]

IRONMAN_SPIKE = [
    {
        "fonte_url": "https://www.ironman.com/races/xclusive-challenge/race/results",
        "evidencia": "Pagina oficial de resultados exibe PLACE, BIB, NAME, REP, AGE, TIME, SWIM, T1, BIKE, T2 e RUN.",
        "status": "avaliado_nao_integrado",
        "decisao": "manter_corpus_exploratorio_separado",
        "motivo": "O split de corrida e extraivel, mas nao representa maratona standalone.",
    }
]


def hms_para_segundos(series: pd.Series) -> pd.Series:
    def _parse(value: object) -> float:
        text = str(value).strip()
        parts = text.split(":")
        if len(parts) != 3:
            return np.nan
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            return np.nan

    return series.apply(_parse)


def limpar_texto(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "nan", "NaN", "-0", "None"}:
        return np.nan
    return text


def limpar_pais(series: pd.Series) -> pd.Series:
    return series.apply(limpar_texto)


def escolher_tempo_principal(
    principal: pd.Series,
    fallback: pd.Series | None = None,
) -> pd.Series:
    base = principal.apply(limpar_texto).replace("00:00:00", np.nan)
    if fallback is None:
        return base
    reserva = fallback.apply(limpar_texto).replace("00:00:00", np.nan)
    return base.combine_first(reserva)


def chave_resultado(df: pd.DataFrame, tempo_col: pd.Series) -> pd.Series:
    bib = df["bib"].fillna(-1).astype(str).str.strip()
    nome = (
        df["nome"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    tempo = tempo_col.fillna("").astype(str).str.strip()
    return np.where(df["bib"].notna(), "bib:" + bib, "nome_tempo:" + nome + "|" + tempo)


def ler_edicoes() -> pd.DataFrame:
    df = pd.read_csv(DATAS_CSV, comment="#")
    df = df.rename(columns={"data": "data_prova"})
    df["evento_id"] = df["maratona"] + "_" + df["ano"].astype(str)
    meta_rows = []
    for maratona, meta in META_EVENTO.items():
        meta_rows.append(
            {
                "maratona": maratona,
                "cidade": meta["cidade"],
                "pais_evento": meta["pais_evento"],
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    df = df.merge(meta_df, on="maratona", how="left")
    df["status_dataset"] = np.where(
        df["maratona"].isin(MARATONAS_CANONICAS),
        "integrado_canonico",
        "fora_do_canonico",
    )
    return df


def preparar_schema_base(df: pd.DataFrame, maratona: str) -> pd.DataFrame:
    base = df.copy()
    base["maratona"] = maratona
    base["evento_id"] = base["maratona"] + "_" + base["ano"].astype(str)
    base["distancia_km"] = 42.195
    return base


def load_berlin() -> pd.DataFrame:
    log.info("Berlin: carregando...")
    path = MARATONAS_DIR / "berlin" / "Berlin_Marathon_data_1974_2019.csv"
    df = pd.read_csv(path, low_memory=False)
    ini, fim = ESCOPO["berlin"]
    df = df[df["YEAR"].between(ini, fim) & (df["YEAR"] != BERLIN_2019_ANO)].copy()
    df = df[~df["YEAR"].isin(EXCLUIR_ANOS)]
    df["GENDER"] = df["GENDER"].str.strip().map({"male": "M", "female": "F"})
    df = df[df["GENDER"].isin(["M", "F"])]
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df = df.dropna(subset=["AGE"])
    kaggle = pd.DataFrame(
        {
            "ano": df["YEAR"].astype(int),
            "genero": df["GENDER"].astype(str),
            "tempo_segundos": hms_para_segundos(df["TIME"]),
            "tempo_original": df["TIME"].astype(str).str.strip(),
            "pais_atleta": limpar_pais(df["COUNTRY"]),
        }
    )
    out = pd.concat([kaggle, load_berlin_2019()], ignore_index=True)
    log.info("Berlin: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "berlin")


def load_berlin_2019() -> pd.DataFrame:
    df = pd.read_csv(BERLIN_2019_CSV, low_memory=False)
    df = df[df["year"] == BERLIN_2019_ANO].copy()
    df["gender"] = df["gender"].str.strip().map({"M": "M", "W": "F"})
    df = df[df["gender"].isin(["M", "F"])]
    return pd.DataFrame(
        {
            "ano": df["year"].astype(int),
            "genero": df["gender"].astype(str),
            "tempo_segundos": hms_para_segundos(df["time_full"]),
            "tempo_original": df["time_full"].astype(str).str.strip(),
            "pais_atleta": limpar_pais(df["nationality"]),
            "fonte": BERLIN_2019_FONTE,
            "tipo_coleta": BERLIN_2019_TIPO_COLETA,
        }
    )


def load_boston() -> pd.DataFrame:
    log.info("Boston: carregando...")
    frames = []
    for ano in range(2005, 2020):
        path = MARATONAS_DIR / "boston" / f"results{ano}.csv"
        if not path.exists():
            log.warning("Boston %s: arquivo ausente - pulando", ano)
            continue
        df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
        row = pd.DataFrame(
            {
                "ano": ano,
                "genero": df["gender"].astype(str).str.strip(),
                "tempo_segundos": pd.to_numeric(df["seconds"], errors="coerce"),
                "tempo_original": df["official_time"].astype(str).str.strip(),
                "pais_atleta": np.nan,
            }
        )
        if ano >= 2015 and "country_residence" in df.columns:
            row["pais_atleta"] = limpar_pais(df["country_residence"])
        frames.append(row)

    out = pd.concat(frames, ignore_index=True)
    out = out[~out["ano"].isin(EXCLUIR_ANOS)]
    log.info("Boston: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "boston")


def load_chicago() -> pd.DataFrame:
    log.info("Chicago: carregando...")
    path = MARATONAS_DIR / "chicago" / "Chicago Marathon Results.csv"
    df = pd.read_csv(path, low_memory=False)
    ini, fim = ESCOPO["chicago"]
    df = df[df["Year"].between(ini, fim)].copy()
    df = df[~df["Year"].isin(EXCLUIR_ANOS)]
    out = pd.DataFrame(
        {
            "ano": df["Year"].astype(int),
            "genero": df["Gender"].astype(str).str.strip(),
            "tempo_segundos": pd.to_numeric(df["Finish"], errors="coerce"),
            "tempo_original": df["Finish Time"].astype(str).str.strip(),
            "pais_atleta": limpar_pais(df["Country"]),
        }
    )
    log.info("Chicago: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "chicago")


def load_nyc() -> pd.DataFrame:
    log.info("NYC: carregando...")
    path = MARATONAS_DIR / "nyc" / "NYC Marathon Results.csv"
    df = pd.read_csv(path, low_memory=False)
    ini, fim = ESCOPO["nyc"]
    df = df[df["Year"].between(ini, fim)].copy()
    # 2012 foi cancelada pelo furacao Sandy.
    df = df[~df["Year"].isin({2012} | EXCLUIR_ANOS)]
    df["genero"] = df["Gender"].map({"M": "M", "W": "F"})
    df = df[df["genero"].isin(["M", "F"])]
    out = pd.DataFrame(
        {
            "ano": df["Year"].astype(int),
            "genero": df["genero"].astype(str),
            "tempo_segundos": pd.to_numeric(df["Finish"], errors="coerce"),
            "tempo_original": df["Finish Time"].astype(str).str.strip(),
            "pais_atleta": limpar_pais(df["Country"]),
        }
    )
    log.info("NYC: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "nyc")


def load_honolulu() -> pd.DataFrame:
    log.info("Honolulu: carregando...")
    path = MARATONAS_DIR / "zenodo_albrecht" / "all_marathons_final.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Marathon"] == "Honolulu"].copy()
    ini, fim = ESCOPO["honolulu"]
    df = df[df["Year"].between(ini, fim)].copy()
    df = df[~df["Year"].isin(EXCLUIR_ANOS)]
    df["genero"] = df["Sex"].map({"M": "M", "F": "F"})
    df = df[df["genero"].isin(["M", "F"])]
    # Nesta fonte, Time vem em minutos.
    tempo_segundos = (pd.to_numeric(df["Time"], errors="coerce") * 60).round()
    out = pd.DataFrame(
        {
            "ano": df["Year"].astype(int),
            "genero": df["genero"].astype(str),
            "tempo_segundos": tempo_segundos,
            "tempo_original": df["Gun Time"].astype(str).str.strip(),
            "pais_atleta": np.nan,
        }
    )
    log.info("Honolulu: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "honolulu")


def load_singapore() -> pd.DataFrame:
    log.info("Singapore: carregando...")
    path = MARATONAS_DIR / "singapore" / "Singapore Marathon Results.csv"
    df = pd.read_csv(path, low_memory=False)
    ini, fim = ESCOPO["singapore"]
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df = df[df["ano"].between(ini, fim)].copy()
    df = df[~df["ano"].isin(EXCLUIR_ANOS)]
    df["genero"] = df["genero"].astype(str).str.strip()
    df = df[df["genero"].isin(["M", "F"])].copy()
    tempo_original = escolher_tempo_principal(df["net_time"], df["gun_time"])
    out = pd.DataFrame(
        {
            "ano": df["ano"].astype(int),
            "genero": df["genero"].astype(str),
            "tempo_segundos": hms_para_segundos(tempo_original),
            "tempo_original": tempo_original,
            "pais_atleta": limpar_pais(df["representing"]),
        }
    )
    log.info("Singapore: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "singapore")


def load_rio() -> pd.DataFrame:
    log.info("Rio: carregando...")
    path = MARATONAS_DIR / "rio" / "Rio Marathon Results.csv"
    df = pd.read_csv(path, low_memory=False)
    ini, fim = ESCOPO["rio"]
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df = df[df["ano"].between(ini, fim)].copy()
    df = df[~df["ano"].isin(EXCLUIR_ANOS)]
    df = df[df["nome"].notna() | df["bib"].notna()].copy()

    selected_frames = []
    for ano in sorted(df["ano"].dropna().astype(int).unique().tolist()):
        year_df = df[df["ano"] == ano].copy()
        # Em 2025 usa apenas a listagem oficial do Runking. Nos demais anos, usa
        # as listagens gerais por genero e acrescenta da elite so quem nao aparece nelas.
        if ano == 2025:
            selected_frames.append(year_df[year_df["fonte_base"] == "runking_html"].copy())
            continue

        general = year_df[year_df["resultado_nome"].isin(["Geral feminino", "Geral masculino"])].copy()
        elite = year_df[year_df["resultado_nome"].astype(str).str.contains("Elite", case=False, na=False)].copy()
        general_tempo = escolher_tempo_principal(general["tempo_liquido"], general["tempo_bruto"])
        elite_tempo = escolher_tempo_principal(elite["tempo_liquido"], elite["tempo_bruto"])
        general_keys = set(chave_resultado(general, general_tempo))
        elite_keys = pd.Series(chave_resultado(elite, elite_tempo), index=elite.index)
        elite = elite[~elite_keys.isin(general_keys)].copy()
        selected_frames.append(pd.concat([general, elite], ignore_index=True))

    selected = pd.concat(selected_frames, ignore_index=True)
    tempo_original = escolher_tempo_principal(selected["tempo_liquido"], selected["tempo_bruto"])
    out = pd.DataFrame(
        {
            "ano": selected["ano"].astype(int),
            "genero": selected["genero"].astype(str).str.strip(),
            "tempo_segundos": hms_para_segundos(tempo_original),
            "tempo_original": tempo_original,
            "pais_atleta": limpar_pais(selected["pais_atleta"]),
        }
    )
    out = out[out["genero"].isin(["M", "F"])].copy()
    log.info("Rio: %s registros apos regra canonica", f"{len(out):,}")
    return preparar_schema_base(out, "rio")


def load_durban() -> pd.DataFrame:
    log.info("Durban: carregando...")
    path = MARATONAS_DIR / "durban" / "Durban Marathon Results.csv"
    df = pd.read_csv(path, low_memory=False)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df = df[df["ano"].isin([2022, 2024, 2025])].copy()
    df = df[~df["ano"].isin(EXCLUIR_ANOS)]
    df["finish_status_norm"] = df["finish_status"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["finish_status_norm"] == "finished"].copy()
    tempo_original = escolher_tempo_principal(df["tempo_liquido"], df["tempo_bruto"])
    out = pd.DataFrame(
        {
            "ano": df["ano"].astype(int),
            "genero": df["genero"].astype(str).str.strip(),
            "tempo_segundos": hms_para_segundos(tempo_original),
            "tempo_original": tempo_original,
            "pais_atleta": limpar_pais(df["pais_atleta"]),
        }
    )
    out = out[out["genero"].isin(["M", "F"])].copy()
    log.info("Durban: %s registros", f"{len(out):,}")
    return preparar_schema_base(out, "durban")


def carregar_fall_padronizado() -> pd.DataFrame:
    log.info("Fall Marathons: carregando recorte selecionado...")
    df = pd.read_csv(
        FALL_CSV,
        usecols=["Race", "Year", "Gender", "Finish", "Age Bracket"],
        low_memory=False,
    )
    df = df[df["Race"].isin(META_FALL)].copy()
    df["maratona"] = df["Race"].map(lambda value: META_FALL[value]["maratona"])
    df["cidade"] = df["Race"].map(lambda value: META_FALL[value]["cidade"])
    df["pais_evento"] = df["Race"].map(lambda value: META_FALL[value]["pais_evento"])
    df["ano"] = df["Year"].astype(int)
    df = df[~df["ano"].isin(EXCLUIR_ANOS)]
    df["evento_id"] = df["maratona"] + "_" + df["ano"].astype(str)
    df["genero"] = df["Gender"].astype(str).str.strip()
    df = df[df["genero"].isin(["M", "F"])]
    df["tempo_segundos"] = pd.to_numeric(df["Finish"], errors="coerce")
    df["tempo_original"] = df["Finish"].astype(str).str.strip()
    df["pais_atleta"] = np.nan
    df["fonte"] = FALL_FONTE
    df["tipo_coleta"] = FALL_TIPO_COLETA
    df["distancia_km"] = 42.195
    df = df[
        [
            "evento_id",
            "maratona",
            "ano",
            "cidade",
            "pais_evento",
            "fonte",
            "tipo_coleta",
            "genero",
            "tempo_segundos",
            "tempo_original",
            "pais_atleta",
            "Age Bracket",
            "Race",
            "distancia_km",
        ]
    ].rename(
        columns={
            "Age Bracket": "faixa_etaria_fonte",
            "Race": "nome_evento_fonte",
        }
    )
    log.info("Fall Marathons: %s registros padronizados", f"{len(df):,}")
    return df


def load_fall_canonico(fall_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Fall Marathons: promovendo recorte ao canônico...")
    out = fall_df[
        [
            "evento_id",
            "maratona",
            "ano",
            "genero",
            "tempo_segundos",
            "tempo_original",
            "pais_atleta",
            "distancia_km",
        ]
    ].copy()
    log.info("Fall Marathons: %s registros no canônico", f"{len(out):,}")
    return out


def carregar_canonico(fall_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            load_berlin(),
            load_boston(),
            load_chicago(),
            load_nyc(),
            load_honolulu(),
            load_singapore(),
            load_fall_canonico(fall_df),
            load_rio(),
            load_durban(),
        ],
        ignore_index=True,
    )


def anexar_metadados_evento(df: pd.DataFrame, edicoes: pd.DataFrame) -> pd.DataFrame:
    meta = edicoes[
        [
            "evento_id",
            "maratona",
            "ano",
            "data_prova",
            "cidade",
            "pais_evento",
            "latitude",
            "longitude",
            "timezone",
            "nota",
        ]
    ]
    out = df.merge(meta, on=["evento_id", "maratona", "ano"], how="left")
    fonte_padrao = out["maratona"].map(lambda value: META_EVENTO[value]["fonte"])
    coleta_padrao = out["maratona"].map(lambda value: META_EVENTO[value]["tipo_coleta"])
    out["fonte"] = out["fonte"].fillna(fonte_padrao) if "fonte" in out else fonte_padrao
    out["tipo_coleta"] = (
        out["tipo_coleta"].fillna(coleta_padrao) if "tipo_coleta" in out else coleta_padrao
    )
    return out


def carregar_agregados_meteo() -> pd.DataFrame:
    """Carrega e valida a unica fonte climatica aceita pelo pipeline."""
    meteo = pd.read_csv(METEO_CSV)
    colunas_obrigatorias = {
        "maratona",
        "ano",
        "janela_inicio_h",
        "janela_fim_h",
        "fonte_janela",
        "n_horas_janela",
        *MAPEAMENTO_COLUNAS_METEO.keys(),
    }
    ausentes = colunas_obrigatorias.difference(meteo.columns)
    if ausentes:
        raise ValueError(
            "Colunas ausentes nos agregados meteorologicos: "
            + ", ".join(sorted(ausentes))
        )

    duplicadas = meteo.duplicated(["maratona", "ano"], keep=False)
    if duplicadas.any():
        chaves = meteo.loc[duplicadas, ["maratona", "ano"]].drop_duplicates()
        raise ValueError(
            "Agregados meteorologicos duplicados para:\n"
            + chaves.to_string(index=False)
        )

    fontes = meteo["fonte_janela"].astype(str).str.strip().str.lower()
    if not fontes.eq("dinamica").all():
        invalidas = meteo.loc[~fontes.eq("dinamica"), ["maratona", "ano", "fonte_janela"]]
        raise ValueError(
            "O pipeline aceita somente a janela dinamica largada + 6h:\n"
            + invalidas.to_string(index=False)
        )

    duracao = meteo["janela_fim_h"] - meteo["janela_inicio_h"]
    duracao_ok = np.isclose(duracao, JANELA_CLIMATICA_HORAS)
    amostras_ok = meteo["n_horas_janela"].eq(JANELA_CLIMATICA_HORAS)
    if not (duracao_ok & amostras_ok).all():
        invalidas = meteo.loc[
            ~(duracao_ok & amostras_ok),
            [
                "maratona",
                "ano",
                "janela_inicio_h",
                "janela_fim_h",
                "n_horas_janela",
            ],
        ]
        raise ValueError(
            "Agregados fora da janela largada + 6h:\n"
            + invalidas.to_string(index=False)
        )

    meteo["ano"] = meteo["ano"].astype(int)
    keep = meteo.rename(columns=MAPEAMENTO_COLUNAS_METEO)
    return keep[["maratona", "ano", *MAPEAMENTO_COLUNAS_METEO.values()]]


def anexar_meteo(df: pd.DataFrame) -> pd.DataFrame:
    keep = carregar_agregados_meteo()
    out = df.merge(keep, on=["maratona", "ano"], how="left")
    out["clima_disponivel"] = out["temp_media"].notna()
    return out


def tipar_canonico(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ano"] = out["ano"].astype("Int32")
    out["tempo_segundos"] = pd.to_numeric(out["tempo_segundos"], errors="coerce").astype("Int32")
    out["distancia_km"] = out["distancia_km"].astype(float)
    out["clima_disponivel"] = out["clima_disponivel"].astype(bool)
    return out


def anexar_sazonalidade_dia_ano(df: pd.DataFrame) -> pd.DataFrame:
    """Codificacao ciclica do dia do ano da prova, com correcao de hemisferio.

    Materializa no canônico as colunas dia_ano_sin e dia_ano_cos usadas na
    modelagem por edicao. dia_ano = dayofyear de data_prova; o angulo e
    2*pi*dia_ano/PERIODO_ANO_DIAS. Maratonas do hemisferio sul recebem sinal
    -1 em sin e cos, alinhando estacao em vez de calendario civil. Convencao
    identica a do script 11_dataset_treino_edicoes.py, para que os valores por
    edicao (maratona, data_prova) coincidam entre os dois artefatos."""
    out = df.copy()
    data_prova = pd.to_datetime(out["data_prova"])
    dia_ano = data_prova.dt.dayofyear.astype(float)
    angulo = 2 * np.pi * dia_ano / PERIODO_ANO_DIAS
    sinal_hem = np.where(out["maratona"].isin(MARATONAS_HEMISFERIO_SUL), -1.0, 1.0)
    out["dia_ano_sin"] = sinal_hem * np.sin(angulo)
    out["dia_ano_cos"] = sinal_hem * np.cos(angulo)
    return out


def anexar_percentil_edicao(df: pd.DataFrame) -> pd.DataFrame:
    """Posicao relativa do tempo do atleta dentro da sua edicao (maratona,
    ano). `rank(pct=True)` devolve a escala (0, 1]: sem empate o mais rapido
    recebe 1/n, nunca zero, e o mais lento recebe 1. E a variavel de posicao
    relativa usada no nivel atleta-edicao (interacao entre clima e percentil).
    Coluna derivada, calculada apos os filtros de qualidade, quando ja nao ha
    tempos nulos nem nao-positivos."""
    out = df.copy()
    tempo = pd.to_numeric(out["tempo_segundos"], errors="coerce")
    out["percentil_edicao"] = (
        tempo.groupby([out["maratona"], out["ano"]])
        .rank(pct=True, method="average")
        .round(6)
    )
    return out


def validar_canonico(df: pd.DataFrame) -> None:
    if not (df["distancia_km"] == 42.195).all():
        raise ValueError("Existe registro fora de 42.195 km no canônico.")
    if (df["tempo_segundos"].isna()).any() or not (df["tempo_segundos"] > 0).all():
        raise ValueError("Existem tempos invalidos no canônico.")
    if df["fonte"].isna().any() or (df["fonte"].astype(str).str.strip() == "").any():
        raise ValueError("Existem registros sem fonte preenchida.")
    if df["evento_id"].isna().any() or df["data_prova"].isna().any():
        raise ValueError("Existem edicoes sem chave resolvida no canônico.")
    if df["percentil_edicao"].isna().any():
        raise ValueError("Existem percentis de edicao nulos no canônico.")
    # A invariante segue a escala real de rank(pct=True), que e (0, 1]. O limite
    # inferior estrito e o que distingue esta transformacao de (rank-1)/(n-1).
    if not df["percentil_edicao"].between(0, 1).all():
        raise ValueError("percentil_edicao fora de (0, 1] no canônico.")
    if (df["percentil_edicao"] <= 0).any():
        raise ValueError("percentil_edicao nao positivo no canônico.")


def gerar_visao_legado(df: pd.DataFrame) -> pd.DataFrame:
    legado = df.rename(columns={"pais_atleta": "pais"})
    return legado[COLUNAS_LEGADO]


def build_fall_support(fall_df: pd.DataFrame, edicoes: pd.DataFrame) -> pd.DataFrame:
    log.info("Fall Marathons: exportando apoio com integração concluída...")
    meta = edicoes[["evento_id", "data_prova", "latitude", "longitude", "timezone", "nota"]].copy()
    meteo = carregar_agregados_meteo()[["maratona", "ano", "temp_media"]]
    meteo["clima_disponivel"] = meteo["temp_media"].notna()
    meteo = meteo[["maratona", "ano", "clima_disponivel"]].drop_duplicates()

    df = fall_df.merge(meta, on="evento_id", how="left")
    df = df.merge(meteo, on=["maratona", "ano"], how="left")
    df["clima_disponivel"] = df["clima_disponivel"].fillna(False)
    df["status_integracao"] = np.where(df["clima_disponivel"], "integrado_canonico", "sem_clima")
    df = df[
        [
            "evento_id",
            "maratona",
            "ano",
            "data_prova",
            "cidade",
            "pais_evento",
            "fonte",
            "tipo_coleta",
            "genero",
            "tempo_segundos",
            "tempo_original",
            "faixa_etaria_fonte",
            "nome_evento_fonte",
            "distancia_km",
            "status_integracao",
            "clima_disponivel",
            "latitude",
            "longitude",
            "timezone",
            "nota",
        ]
    ]
    log.info("Fall Marathons: %s registros no apoio", f"{len(df):,}")
    return df


def escrever_manifestos(edicoes: pd.DataFrame, fall_support: pd.DataFrame) -> None:
    APOIO_DIR.mkdir(parents=True, exist_ok=True)

    manifesto = pd.DataFrame(MANIFESTO_FONTES)
    manifesto.to_csv(APOIO_DIR / "fontes_manifesto.csv", index=False)

    mapping_rows = [
        {
            "fonte_id": fonte_id,
            "coluna_origem": coluna_origem,
            "classificacao": classificacao,
            "coluna_destino": coluna_destino,
            "transformacao": transformacao,
        }
        for fonte_id, coluna_origem, classificacao, coluna_destino, transformacao in MAPEAMENTO_COLUNAS
    ]
    pd.DataFrame(mapping_rows).to_csv(APOIO_DIR / "mapeamento_colunas.csv", index=False)

    edicoes_integradas = edicoes[edicoes["maratona"].isin(MARATONAS_CANONICAS)].copy()
    edicoes_integradas.to_csv(APOIO_DIR / "edicoes_maratonas.csv", index=False)

    pd.DataFrame(ALIASES_EXCLUSOES).to_csv(APOIO_DIR / "aliases_exclusoes.csv", index=False)
    pd.DataFrame(AVALIACAO_FRIAS).to_csv(APOIO_DIR / "avaliacao_candidatas_frias.csv", index=False)
    pd.DataFrame(IRONMAN_SPIKE).to_csv(APOIO_DIR / "ironman_spike.csv", index=False)
    fall_support.to_csv(OUT_FALL_SUPPORT, index=False)


def salvar_saidas(canonico: pd.DataFrame) -> None:
    PROCESSADOS.mkdir(parents=True, exist_ok=True)
    canonico.to_csv(OUT_CANONICAL, index=False)
    gerar_visao_legado(canonico).to_csv(OUT_LEGACY, index=False)


def resumo_por_maratona(df: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        df.groupby("maratona")
        .agg(
            edicoes=("ano", "nunique"),
            finishers=("tempo_segundos", "count"),
            clima_ok=("clima_disponivel", "sum"),
            tempo_mediano_min=("tempo_segundos", lambda values: round(values.median() / 60, 1)),
        )
        .reset_index()
    )
    return resumo


def aplicar_filtros_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    antes = len(out)
    out = out.dropna(subset=["tempo_segundos", "genero"])
    out = out[out["tempo_segundos"] > 0]
    removidos = antes - len(out)
    if removidos:
        log.info("Filtros de qualidade: removidos %s registros invalidos", f"{removidos:,}")
    return out


def main() -> None:
    log.info("=== ETL dataset unico de maratonas ===")
    edicoes = ler_edicoes()
    fall_df = carregar_fall_padronizado()

    canonico = carregar_canonico(fall_df)
    canonico = aplicar_filtros_qualidade(canonico)
    canonico = anexar_metadados_evento(canonico, edicoes)
    canonico = anexar_meteo(canonico)
    canonico = tipar_canonico(canonico)
    canonico = anexar_sazonalidade_dia_ano(canonico)
    canonico = anexar_percentil_edicao(canonico)
    canonico = canonico[COLUNAS_CANONICAS]
    validar_canonico(canonico)

    fall_support = build_fall_support(fall_df, edicoes)

    salvar_saidas(canonico)
    escrever_manifestos(edicoes, fall_support)

    log.info("Canônico salvo em %s", OUT_CANONICAL)
    log.info("Legado salvo em %s", OUT_LEGACY)
    log.info("Apoio salvo em %s", APOIO_DIR)
    log.info("Shape canônico: %s linhas x %s colunas", f"{canonico.shape[0]:,}", canonico.shape[1])
    log.info("Shape Fall apoio: %s linhas x %s colunas", f"{fall_support.shape[0]:,}", fall_support.shape[1])
    log.info("\n%s", resumo_por_maratona(canonico).to_string(index=False))

    if not canonico["clima_disponivel"].all():
        faltantes = canonico.loc[~canonico["clima_disponivel"], ["maratona", "ano"]].drop_duplicates()
        log.warning("Edicoes sem clima no canônico:\n%s", faltantes.to_string(index=False))


if __name__ == "__main__":
    main()
