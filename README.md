
# Tech Challenge - Fase 1

## Sistema Inteligente de Apoio à Classificação de Tumores Mamários

Projeto desenvolvido para a pós-graduação em
Desenvolvimento com Inteligência Artificial da FIAP.

## Objetivo

Desenvolver e comparar modelos supervisionados
de Machine Learning para classificar amostras
de massas mamárias como benignas ou malignas.

## Dataset

Breast Cancer Wisconsin (Diagnostic)

Fonte:
https://archive.ics.uci.edu/dataset/17

## Tecnologias

- Python 3.13
- Pandas e NumPy
- Scikit-learn
- Matplotlib e Seaborn
- SHAP
- Jupyter Notebook
- Streamlit

## Preparação do ambiente

Criar um ambiente virtual Python 3.13
e instalar as dependências:

python -m pip install -r requirements-lock.txt

## Organização

- data: dados originais e processados
- notebooks: experimentos e análises
- src: algoritmos e processamento
- models: modelos treinados
- app: interface demonstrativa
- reports: gráficos e documentação
- tests: testes automatizados

## Status

Etapa 2 - Preparação do ambiente de desenvolvimento.

## Observação

Este projeto tem finalidade acadêmica e não
substitui avaliações ou diagnósticos médicos.



## Etapa 3 - Analise Exploratoria dos Dados

A analise exploratoria foi realizada utilizando
o dataset Breast Cancer Wisconsin Diagnostic,
disponibilizado pela UCI Machine Learning Repository.

### Caracterizacao do dataset

- Total de amostras: 569
- Caracteristicas preditoras: 30
- Amostras benignas: 357 (62,74%)
- Amostras malignas: 212 (37,26%)

### Qualidade dos dados

Foram realizadas verificacoes de:

- Tipos de dados
- Valores ausentes
- Registros duplicados
- Identificadores repetidos
- Valores nao finitos
- Valores negativos
- Valores iguais a zero

Nao foram identificados valores ausentes,
nao finitos, negativos ou registros duplicados
segundo os criterios investigados.

Foram encontradas 78 ocorrencias de valores zero
em caracteristicas relacionadas a concavidade
e pontos concavos. Esses valores foram preservados.

### Analise estatistica

Foram calculadas estatisticas descritivas
das 30 caracteristicas numericas.

Tambem foram realizadas comparacoes entre
as amostras benignas e malignas.

As analises revelaram diferencas nas distribuicoes
de raio, area, concavidade e pontos concavos.

### Analise de correlacao

Foram investigados 435 pares unicos de
caracteristicas utilizando Pearson.

Principais resultados:

- radius_mean x perimeter_mean: 0,998
- radius_worst x perimeter_worst: 0,994
- radius_mean x area_mean: 0,987

As correlacoes elevadas sugerem informacoes
parcialmente redundantes entre alguns atributos.

Nenhuma caracteristica foi removida apenas
com base na correlacao.

### Graficos produzidos

Os seis graficos estao armazenados em
reports/figures:

1. Distribuicao dos diagnosticos
2. Boxplot do raio medio
3. Histograma do raio medio
4. Boxplots das caracteristicas
5. Correlacao entre quatro caracteristicas
6. Heatmap das 30 caracteristicas

### Notebook

Arquivo:
notebooks/01_exploracao_dados.ipynb

### Reproducao

Apos configurar o ambiente virtual e instalar
as dependencias, abra o notebook no VS Code,
selecione o kernel Python do projeto e execute
todas as celulas em sequencia.

### Resultado

A analise exploratoria foi validada utilizando
verificacoes automaticas de integridade
e existencia dos graficos.

Os resultados representam uma investigacao
academica, nao uma validacao clinica.


## Etapa 4 - Preparacao dos Dados para Machine Learning

### Variaveis preditoras e variavel-alvo

Foram selecionadas as 30 caracteristicas morfologicas
como variaveis preditoras (X).

A coluna de diagnostico foi codificada em:

- 0: Benigno
- 1: Maligno

O identificador da amostra foi excluido das
variaveis de treinamento.

### Divisao dos dados

Foi utilizada uma separacao estratificada,
com semente aleatoria 42.

- Treinamento: 398 amostras
- Validacao: 85 amostras
- Teste: 86 amostras

As proporcoes das classes foram preservadas
aproximadamente nos tres conjuntos.

Nao foram identificados indices compartilhados
entre treinamento, validacao e teste.

### Padronizacao

Foi utilizado o StandardScaler para padronizar
as caracteristicas numericas.

Os parametros foram calculados exclusivamente
com os dados de treinamento.

O conjunto de validacao recebeu a mesma
transformacao, evitando o ajuste com dados
externos ao treinamento.

### Pipelines

Foram configurados dois pipelines:

1. Regressao Logistica
   - StandardScaler
   - LogisticRegression

2. Random Forest
   - RandomForestClassifier
   - 200 arvores inicialmente configuradas

Os classificadores foram validados quanto
a estrutura e permanecem sem treinamento.

### Notebook

notebooks/02_preparacao_modelagem.ipynb

### Proxima etapa

Treinamento e avaliacao dos classificadores,
utilizando os conjuntos previamente separados.

O conjunto de teste sera reservado para
a avaliacao final do modelo selecionado.
