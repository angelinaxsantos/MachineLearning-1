# Projeto Prático 1: Aprendizagem Supervisionada (Airbnb Lisboa)

Machine Learning, 2026/27. Previsão do preço por noite (regressão linear) e deteção de anúncios caros (regressão logística), com modelos interpretáveis implementados do zero com gradient descent.

## Dados

- **Fonte:** Inside Airbnb (https://insideairbnb.com/get-the-data/), ficheiro `listings.csv.gz`.
- **Cidade:** Lisboa.
- **Versão:** recolha de junho de 2026 (`scrape_id` 20260623040800).
- **Dimensão:** 24876 anúncios e 90 colunas. O ficheiro (cerca de 14 MB) está incluído no repositório, para os resultados serem reproduzidos com exatamente os mesmos dados.
- **Limpeza inicial:**
  - 12 colunas totalmente vazias são removidas, entre elas `host_response_rate` e `instant_bookable`, que neste ficheiro não têm nenhum valor preenchido.
  - 2240 anúncios sem preço são removidos, por o preço ser o alvo. Ficam 22636 anúncios.

## Estrutura

```
listings.csv.gz            dados originais (incluídos no repositório)
data_preparation.ipynb     limpeza, engenharia de variáveis, split e matrizes finais
linear_regression.ipynb    regressão linear (preço)
logistic_regression.ipynb  regressão logística (anúncio caro)
data_prepared/             gerada pelo data_preparation.ipynb (6 subpastas)
```

Os módulos pedidos no enunciado (`data_preparation`, `linear_regression`, `logistic_regression`) foram entregues como notebooks.

## Como reproduzir

1. **Requisitos:** Python 3.12 e as bibliotecas `pandas`, `numpy`, `matplotlib` e `jupyter` (ou `ipykernel`).
   ```
   pip install pandas numpy matplotlib jupyter
   ```
2. **Dados:** o `listings.csv.gz` já está incluído, na mesma pasta dos notebooks (o nome do ficheiro está na variável `INPUT_PATH` do `data_preparation.ipynb`). Para usar outra recolha, descarregá-la em https://insideairbnb.com/get-the-data/ e substituir o ficheiro.
3. **Executar, pela ordem seguinte e cada um do início ao fim (Run All):**
   1. `data_preparation.ipynb`: cria a pasta `data_prepared/` com as 6 combinações de pré-processamento.
   2. `linear_regression.ipynb`
   3. `logistic_regression.ipynb`

Os dois últimos notebooks só leem a pasta `data_prepared/`, por isso dependem do primeiro. A aleatoriedade está fixada (`SEED = 42`), pelo que os resultados são reproduzíveis.

## Pré-processamento

- **Split:** 70% treino (15846), 15% validação (3395) e 15% teste (3395), com baralhamento fixo.
- **Valores em falta:** a falta de reviews (`review_scores_*` e `reviews_per_month` vazios quando não há reviews) é distinguida da falta real de informação através da variável `has_reviews`.
- **Outliers:** detetados em `price`, `minimum_nights` e `accommodates` com dois critérios robustos, o IQR (1,5 × amplitude interquartil) e o z-score (|z| > 3).
- **Variáveis:** `amenities` passa a `n_amenities`; as categorias raras de `property_type` e `neighbourhood_cleansed` são agrupadas em `Other`; as categóricas são codificadas em one-hot; as numéricas são padronizadas.
- **Estratégias comparadas** (cada uma gera a pasta `data_prepared/<missing>_<outlier>/`):

  | Valores em falta | Outliers |
  |---|---|
  | `simple`: mediana global | `none`: sem tratamento |
  | `grouped`: mediana por bairro e tipo de quarto | `cap`: winsorização pelos limites do IQR |
  | | `drop`: remoção das linhas outlier (só no treino) |

  Os limites do capping, as medianas de imputação, as colunas one-hot e a escala são calculados com o conjunto de treino.

## Modelos e experiências

Todos os modelos usam mini-batch gradient descent implementado à mão, com regularização L2 opcional.

**Regressão linear** (alvo: preço)
- Estudo cruzado das 6 combinações e escolha automática da pasta principal.
- Preço bruto contra log-preço, termos polinomiais (graus 2 e 3) e interações.
- Regularização L2, k-fold (5 folds), modelo final, resíduos, erro por bairro e tipo de quarto, e tabela de coeficientes.
- Métricas: MAE, RMSE e R².

**Regressão logística** (alvo: `high_price`)
- **Classe positiva:** anúncio no top 20% de preço dentro do seu grupo `room_type` × `neighbourhood_cleansed`.
- Mesmas experiências da regressão linear, com métricas de classificação: accuracy, precision, recall, F1, matriz de confusão, ROC-AUC e PR-AUC.
- Análise do limiar de decisão (escolhido na validação), erros por grupo e tabela de coeficientes com odds ratios.

A pasta principal é escolhida em cada notebook pelos resultados do respetivo estudo cruzado (maior R² na regressão e maior ROC-AUC na classificação), por isso pode diferir entre os dois.

## Resultados principais

| Modelo | Conjunto | Resultado |
|---|---|---|
| Regressão linear (`grouped_cap`) | Teste | R² = 0,611; MAE = 41,5 €; RMSE = 56,1 € |
| Regressão logística (`simple_none`, limiar 0,23) | Teste | ROC-AUC = 0,854; PR-AUC = 0,649; recall = 0,705; precision = 0,493; F1 = 0,580 |

Os notebooks de regressão linear e de regressão logística contêm, a seguir às tabelas e gráficos, a discussão dos resultados.
