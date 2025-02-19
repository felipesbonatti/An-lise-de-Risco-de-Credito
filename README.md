# Analise-de-Risco-de-Credito

[![GitHub last commit](https://img.shields.io/github/last-commit/felipesbonatti/Analise-de-Risco-de-Credito?style=flat-square)](https://github.com/felipesbonatti/Analise-de-Risco-de-Credito)
[![GitHub repo size](https://img.shields.io/github/repo-size/felipesbonatti/Analise-de-Risco-de-Credito?style=flat-square)](https://github.com/felipesbonatti/Analise-de-Risco-de-Credito)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="Logo GitHub" width="100">
</p>

---

## 📌 Sobre o Projeto

O código fornecido é um exemplo completo de um pipeline de machine learning para avaliação de risco de crédito. Ele inclui etapas desde o carregamento e processamento de dados até o treinamento e avaliação de modelos, além de uma API para servir previsões de risco de crédito. Abaixo está uma explicação detalhada das principais seções do código:

---

## 🎯 Objetivo

O objetivo do código fornecido é criar um pipeline completo de machine learning para avaliação de risco de crédito. O pipeline inclui várias etapas, desde o carregamento e processamento de dados até o treinamento e avaliação de modelos, além de disponibilizar uma API para servir previsões de risco de crédito. As etapas detalhadas são:

Configuração Inicial: Define as dependências do projeto no arquivo requirements.txt, garantindo que todas as bibliotecas necessárias estão instaladas com as versões corretas.

Processamento de Dados: Inclui funções para carregar dados de diferentes fontes (banco de dados SQL e arquivos CSV), tratar valores faltantes e outliers, e dividir os dados em conjuntos de treino, validação e teste. Este processamento é essencial para preparar os dados para a modelagem.

Engenharia de Features: Identifica os tipos de features (numéricas e categóricas), cria novas features a partir das existentes, e seleciona as features mais importantes. Também cria pipelines de pré-processamento para preparar as features para o treinamento dos modelos.

Treinamento de Modelos: Inclui funções para treinar diferentes modelos de machine learning (Árvore de Decisão, Regressão Logística, Random Forest e XGBoost), avaliar esses modelos usando métricas padrão, e salvar/carregar modelos treinados. Também inclui métodos para explicar as predições dos modelos usando SHAP ou LIME.

API para Consulta de Risco de Crédito: Configura uma API Flask para servir previsões de risco de crédito. A API possui endpoints para verificar a saúde do serviço (/health), realizar predições de risco de crédito (/predict), e explicar predições específicas (/explain). A API carrega o modelo treinado, o preprocessador e as configurações necessárias para realizar predições e fornecer explicações.

Em resumo, o código visa fornecer uma solução completa e automatizada para a avaliação de risco de crédito, desde a preparação dos dados até a disponibilização de uma API para consumo externo.

---

## ⚙️ Solução Entregue

A solução é um pipeline completo de machine learning para avaliação de risco de crédito, abordando:

Configuração Inicial:

Dependências listadas em requirements.txt.
Processamento de Dados (data_processing.py):

Carregamento de dados de SQL e CSV.
Tratamento de valores faltantes e outliers.
Divisão de dados em treino, validação e teste.
Engenharia de Features (feature_engineering.py):

Identificação e criação de features.
Criação de pipelines de pré-processamento.
Seleção de features importantes.
Treinamento de Modelos (model_training.py):

Treinamento de modelos (Árvore de Decisão, Regressão Logística, Random Forest, XGBoost).
Avaliação e explicação das predições dos modelos.
Salvamento e carregamento de modelos.
API para Predições (app.py):

API Flask com endpoints para verificar saúde, realizar predições e explicar predições.

---

## 📊 Resultados

Automatização do pipeline de ML.
Flexibilidade e extensibilidade.
Explicabilidade das predições.
Disponibilização de predições via API.


---

## 🛠️ Tecnologias Utilizadas

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white" alt="PySpark">
  <img src="https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=postgresql&logoColor=white" alt="SQL">
</div>

### Linguagem:
Python

### Bibliotecas:
pandas
numpy
scikit-learn
xgboost
flask
matplotlib
seaborn
joblib
sqlalchemy
pytest
shap
lime
jupyterlab

### Banco de Dados:
SQL (via SQLAlchemy)

### Ferramenta de Análise:
JupyterLab



