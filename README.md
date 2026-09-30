# Clima e desempenho coletivo em maratonas

Código e dados do Projeto de Graduação *Impacto das condições climáticas no desempenho coletivo em maratonas: uma análise de dados*, de Ian Patello de Freitas, Engenharia de Computação e Informação, Escola Politécnica da Universidade Federal do Rio de Janeiro (UFRJ), sob orientação de Flávio Luis de Mello.

O trabalho representa cada edição de maratona pelos percentis 10, 25, 50, 75 e 90 dos tempos de conclusão, integra as condições meteorológicas da janela de seis horas iniciada na largada e avalia se essas condições acrescentam capacidade preditiva ao histórico de cada prova.

## Datasets

Os dois datasets têm uma linha por edição, 45 colunas e nenhum valor ausente.

| Pasta | Edições | Maratonas |
| --- | --- | --- |
| `dataset117/` | 117 | Berlim, Boston, Chicago, Durban, Honolulu, Long Beach, Maui, Nova York, Rio de Janeiro, Savannah e Singapura, de 2005 a 2025 |
| `dataset49/` | 49 | Boston (15), Chicago (17) e Nova York (17) |

O `dataset49` é o subconjunto do `dataset117` formado pelas maratonas situadas entre 40 °N e 43 °N com pelo menos quinze edições. Ele preserva todas as colunas e seleciona apenas as linhas dessas três provas.

Cada pasta contém `dataset_treino_edicoes.csv` e, em `resultados/`, as tabelas de erro e as previsões fora da amostra usadas na monografia.

### Colunas

- **Identificação:** `maratona`, `ano`, `data_prova`.
- **Alvos, em minutos:** `p10`, `p25`, `p50`, `p75`, `p90`. Os quantis estão na forma absoluta; a referência histórica usada na modelagem é calculada dentro de cada divisão de validação, apenas com as edições de treino.
- **Descritivas:** `iqr` (intervalo interquartil) e `n_finishers` (número de concluintes).
- **Contexto da edição:** `ano`, `dia_ano_sin`, `dia_ano_cos`, `ln_n_finishers` (logaritmo natural de um mais o número de concluintes) e `pct_M` (proporção de homens entre os concluintes).
- **Clima (31 colunas):** temperatura do ar, bulbo úmido, umidade relativa, temperatura aparente, ponto de orvalho, vento e rajada (em m/s, com componentes `wx_med` e `wy_med`), precipitação, radiação de onda curta e derivadas na janela da prova. Os sufixos `_med`, `_max`, `_min` e `_std` indicam média, máximo, mínimo e desvio padrão na janela; `bulbo_umido_horas_acima_N` conta as horas com bulbo úmido acima de N °C.

### Códigos nas tabelas de resultados

As tabelas de resultados usam códigos curtos. Nos arquivos gerados pelo script 12 (`resultados_treino_quantis.csv` e `predicoes_loo_quantis.csv`), a coluna `modelo` assume os valores abaixo.

| Código | Significado |
| --- | --- |
| `B1_global` | Média global do quantil nas edições de treino |
| `B2_por_maratona` | Referência histórica: mediana do mesmo quantil na mesma maratona, calculada só com as edições de treino; quando a maratona não aparece no treino, usa a mediana global desse quantil no treino |
| `M1_linear`, `M2_ridge`, `M3_random_forest`, `M4_hist_gbm`, `M5_mlp` | Regressão linear, Ridge, floresta aleatória, gradient boosting por histogramas e perceptron multicamadas sobre o quantil absoluto, com a identificação da maratona |
| `M2a_ridge_anom`, `M4a_hist_gbm_anom` | Ridge e gradient boosting sobre o desvio em relação à referência histórica, sem a identificação da maratona; a previsão volta aos minutos somando a referência |
| `M2s_ridge_semcat`, `M4s_hist_gbm_semcat`, `M2ai_ridge_anom_cat`, `M4ai_hist_gbm_anom_cat` | Variantes que separam o efeito da formulação em desvio do efeito de retirar a identificação da maratona: quantil absoluto sem a identificação (`s`) e desvio com a identificação (`ai`) |

Nos arquivos dos scripts 13 e 14, `familia` identifica o método (`media_global`, `referencia_historica`, `linear`, `ridge`, `floresta_aleatoria`, `gbm_histogramas`, `perceptron_multicamadas`) e `formulacao` indica o alvo: `absoluto`, `desvio` (em relação à referência histórica) ou `referencia` (as duas linhas de base). Nas tabelas de robustez e de sensibilidade ao horário de largada, os sufixos `_abs` e `_dev` de `modelo` indicam essas mesmas formulações; `protocolo` distingue a exclusão de uma maratona inteira (`LOMO`) do corte temporal (`TEMPORAL`, treino até 2017 e teste a partir de 2018). Na atribuição por grupos de variáveis, `E` é o bloco de calendário (`ano`, `dia_ano_sin`, `dia_ano_cos`), `C` o bloco climático e `S` a composição dos concluintes (`pct_M`, `ln_n_finishers`). Nos arquivos do `dataset49`, `exp1` se refere aos modelos treinados nas 117 edições e avaliados nas 49 edições do recorte.

## Estrutura

```
codigo/
  etl/            coleta meteorológica (Open-Meteo, modelo ERA5) e pipeline de ETL
  cap4/           agregados meteorológicos, dataset por edição, análise descritiva e modelagem
dados/
  brutos/         datas e coordenadas das edições; registros horários ERA5 com recibo de cada consulta
  processados/    horários de largada e agregados meteorológicos por edição
dataset117/       dataset de 117 edições e resultados
dataset49/        dataset de 49 edições e resultados
```

| Script | Gera |
| --- | --- |
| `codigo/etl/coleta_meteo.py` | registros horários ERA5 e recibos em `dados/brutos/meteo/era5/hourly/` |
| `codigo/cap4/notebooks/03_agregados_horarios.py` | `dados/processados/apoio/agregados_horarios.csv` |
| `codigo/etl/etl_pipeline.py` | dataset canônico por atleta (exige os resultados brutos, não distribuídos) |
| `codigo/cap4/notebooks/10_painel_quantis_intra_maratona.py` | `painel_quantis_intra_maratona.csv` (exige o dataset canônico) |
| `codigo/cap4/notebooks/11_dataset_treino_edicoes.py` | `dataset117/dataset_treino_edicoes.csv` (exige o dataset canônico) |
| `codigo/cap4/notebooks/12_treino_quantis_edicao.py` | `resultados_treino_quantis.csv` e `predicoes_loo_quantis.csv` |
| `codigo/cap4/notebooks/13_desenho_treino_quantis.py` | demais arquivos de `dataset117/resultados/` |
| `codigo/cap4/notebooks/14_recorte_latitude_quantis.py` | arquivos de `dataset49/resultados/` |

## Reprodução

Ambiente usado: Python 3.12.10 no Windows 11, com as versões de `requirements.txt`.

```
pip install -r requirements.txt
python codigo/cap4/notebooks/12_treino_quantis_edicao.py
python codigo/cap4/notebooks/13_desenho_treino_quantis.py
python codigo/cap4/notebooks/14_recorte_latitude_quantis.py
```

Os scripts 12 a 14 partem de `dataset117/dataset_treino_edicoes.csv`, devem ser executados nessa ordem e gravam tabelas em `dados/processados/apoio/` e figuras em `codigo/cap4/figuras/`. As bibliotecas numéricas são limitadas a uma thread e a semente é fixa; a execução completa leva cerca de duas horas. Uma nova execução reproduz as tabelas de modelagem publicadas em `resultados/` com as mesmas linhas e colunas e diferenças numéricas de no máximo 10⁻¹³ minuto, efeito da aritmética de ponto flutuante; em quatro arquivos, linhas com erros praticamente empatados podem sair em outra ordem. Por isso a comparação deve ser feita por valor, e não pelos hashes de `SHA256SUMS`, que identificam os arquivos publicados. A exceção é `dataset117/resultados/painel_quantis_intra_maratona.csv`, gerado pelo script 10 a partir do dataset canônico, que não é distribuído.

Os registros por atleta não estão no repositório. O dataset canônico tem cerca de 770 MB e reúne resultados de fontes com licenças diferentes, algumas sem licença explícita de redistribuição. Os resultados das fontes oficiais de Rio de Janeiro, Durban e Singapura foram coletados localmente, conforme o capítulo 3 da monografia, e os coletores não são distribuídos. Para reconstruir o dataset canônico, é preciso obter as fontes listadas nesse capítulo, gravá-las em `dados/brutos/maratonas/` e executar `codigo/etl/etl_pipeline.py` e, em seguida, `codigo/cap4/notebooks/11_dataset_treino_edicoes.py`.

## Licenças

- Código: MIT (`LICENSE`).
- Datasets e resultados: [Creative Commons Atribuição 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.pt-br). As variáveis meteorológicas derivam da reanálise ERA5 do Copernicus Climate Change Service, obtida pela API Historical Weather da Open-Meteo, ambas sob CC BY 4.0. Os quantis de tempo são estatísticas agregadas por edição, calculadas a partir das fontes de resultados citadas na monografia.

## Citação

I. P. de Freitas, "Impacto das condições climáticas no desempenho coletivo em maratonas: uma análise de dados," Projeto de Graduação, Escola Politécnica, Universidade Federal do Rio de Janeiro, Rio de Janeiro, 2026.
