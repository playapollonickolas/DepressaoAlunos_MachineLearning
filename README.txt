# Análise de Depressão em Estudantes

Projeto de Machine Learning aplicado à área de saúde mental, desenvolvido para identificar fatores de risco para depressão em estudantes universitários.

## Descrição

Este projeto utiliza técnicas de Machine Learning e Processamento de Linguagem Natural (PLN) para analisar dados relacionados à saúde mental de estudantes. O sistema realiza predições baseadas em múltiplos fatores acadêmicos, sociais e comportamentais.

## Tecnologias

- Python 3.x
- Scikit-learn
- Pandas e NumPy
- Streamlit
- Plotly
- NLTK
- Imbalanced-learn

## Instalação

1. Clone o repositório
2. Instale as dependências:

python -m pip install -r requirements.txt

## Como Usar

### 1. Análise Exploratória

Execute o notebook `01_analise_exploratoria.ipynb` para explorar os dados e gerar visualizações.

### 2. Treinamento do Modelo

Execute o notebook `02_treinamento_modelo.ipynb` para treinar os modelos e selecionar o melhor.

### 3. Aplicação Web

Execute a aplicação Streamlit:

streamlit run app.py

## Funcionalidades

- Análise exploratória completa dos dados
- Comparação de múltiplos algoritmos de Machine Learning
- Predição de risco de depressão
- Visualizações interativas
- Interface web amigável

## Dataset

O projeto utiliza o Student Depression Dataset disponível no Kaggle, que contém informações sobre diversos aspectos da vida estudantil que podem estar relacionados à depressão.

## Métricas de Avaliação

O modelo é avaliado utilizando as seguintes métricas:

- Acurácia
- Precisão
- Recall
- F1-Score
- AUC-ROC
- Cross-validation

## Autor

Apolo Nicolas Ferreira Tenório