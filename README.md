# Impacto do Dropout na Generalizacao e Atribuicao de Features (SHAP) em Classificadores Neurais PyTorch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://github.com/shap/shap)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Este repositorio contem o codigo-fonte, pipeline de dados, rotinas de treino e experimentos da pesquisa sobre o **impacto da regularizacao estocastica por Dropout na capacidade de generalizacao e na estrutura de atribuicao de features (via SHAP)** em redes neurais profundas (MLP) para classificacao binaria tabular.

---

## 1. Visao Geral e Motivacao Cientifica

O **Dropout** (Srivastava et al., 2014) e uma tecnica de regularizacao em aprendizado profundo que atua desativando estocasticamente neuronios durante o treino com probabilidade $p$. Teoricamente, o Dropout:

1. **Evita a Co-Adaptacao de Neuronios**: Impede que neuronios dependam exclusivamente da presenca mutua de ativacoes especificas para corrigir erros.
2. **Atua como um Ensemble Implicito**: Treina exponencialmente muitas sub-redes esparsas que compartilham pesos, realizando uma media geometrica de predicoes no teste.
3. **Regulariza Representacoes Latentes**: Induz a dispersao da importancia preditiva por multiplos caminhos sinapticos, reduzindo a sensibilidade a ruidos espurios.

Neste projeto, essa dinamica e avaliada empiricamente em dois datasets tabulares de referencia, comparando uma arquitetura **Baseline ($p=0.0$)** contra uma variante **Regularizada com Dropout ($p=0.20$)**, com explicabilidade pos-hoc atraves de **SHapley Additive exPlanations (SHAP)**.

---

## 2. Datasets de Estudo

| Dataset | Instancias | Features Originais | Features Pos-Encoding | Tipo de Dados | Prevalencia Positiva |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Adult Census Income** (OpenML 1590) | 45.222 | 14 | 104 | Misto (Categorico One-Hot e Continuo Normalizado) | 24.8% (>50K) |
| **Spambase UCI** (OpenML 44) | 4.601 | 57 | 57 | Continuo Denso (Frequencias de palavras e simbolos) | 39.4% (Spam) |

- **Particionamento Estratificado**: 70% Treino, 15% Validacao e 15% Teste (held-out test set nao utilizado em nenhuma etapa de ajuste de hiperparametros).

---

## 3. Arquitetura da Rede Neural (PyTorch)

Para garantir comparabilidade rigorosa (controlled experiment), ambas as redes compartilham a mesma topologia, inicializacao deterministica de pesos (Kaiming Normal) e rotina de otimizacao:

```text
Entrada (d) -> Linear(128) -> BatchNorm1d -> ReLU -> [Dropout(p)] 
            -> Linear(64)  -> BatchNorm1d -> ReLU -> [Dropout(p)] 
            -> Linear(32)  -> BatchNorm1d -> ReLU 
            -> Linear(1)   -> Logit de Saida
```

- **Otimizador**: AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$).
- **Scheduler**: `ReduceLROnPlateau` (fator 0.5, paciencia de 4 epocas monitorando a perda de validacao).
- **Funcao de Custo**: `BCEWithLogitsLoss`.

---

## 4. Principais Resultados Empiricos

### 4.1 Desempenho no Dataset Spambase (57 Features)

| Metrica de Avaliacao | Baseline ($p=0.0$) | Dropout ($p=0.20$) | Variacao ($\Delta$) | Destaque Cientifico |
| :--- | :---: | :---: | :---: | :--- |
| **Acuracia (Teste)** | 92.19% | **93.20%** | **+1.01%** | Aumento superior a 1% na acuracia global |
| **Precisao** | 90.37% | **91.51%** | **+1.14%** | Menor taxa de falsos positivos |
| **Recall (Sensibilidade)** | 89.71% | **91.18%** | **+1.47%** | Maior taxa de deteccao da classe minoritaria |
| **F1-Score** | 90.04% | **91.34%** | **+1.30%** | Desempenho harmonico superior |
| **ROC-AUC** | 0.9774 | **0.9803** | **+0.0029** | Excelente discriminacao estocastica |
| **PR-AUC (Avg. Precision)**| 0.9632 | **0.9701** | **+0.0069** | Superioridade da curva Precisao-Revocacao |
| **Log-Loss** | 0.1910 | **0.1844** | **-0.0066** | Menor incerteza nas predicoes de teste |
| **Generalization Gap** | **+0.1353** | **+0.0335** | **-75.2%** | **Supressao severa de overfitting** |

### 4.2 Desempenho no Dataset Adult Census Income (104 Features)

| Metrica de Avaliacao | Baseline ($p=0.0$) | Dropout ($p=0.20$) | Variacao ($\Delta$) | Destaque Cientifico |
| :--- | :---: | :---: | :---: | :--- |
| **Acuracia (Teste)** | 84.88% | **85.55%** | **+0.67%** | Melhor acuracia obtida no censo |
| **Precisao** | 73.24% | **74.20%** | **+0.96%** | Reducao consistente de falsos positivos |
| **Recall (Sensibilidade)** | 61.39% | **63.89%** | **+2.50%** | Forte sensibilidade na classe de alta renda |
| **F1-Score** | 66.80% | **68.66%** | **+1.86%** | Equilibrio robusto entre precisao e sensibilidade |
| **ROC-AUC** | 0.9086 | **0.9103** | **+0.0017** | Melhor ordenacao de probabilidade |
| **PR-AUC (Avg. Precision)**| 0.7758 | **0.7818** | **+0.0060** | Robustez sob desbalanceamento |
| **Generalization Gap** | **+0.1252** | **+0.0353** | **-71.8%** | **Reducao expressiva de overfitting** |

---

## 5. Analise Explicativa com SHAP (SHapley Additive exPlanations)

Utilizando o `shap.GradientExplainer` adaptado para modelos PyTorch com background amostral representativo:

1. **Atenuacao de Ruido Amostral**: No dataset Adult, a variavel `fnlwgt` (peso amostral que representa ruido populacional sem relacao causal com renda) figurava em 16º lugar no Baseline. Com o Dropout, caiu 9 posicoes no ranking, demonstrando que a regularizacao estocastica penaliza a memorizacao de ruido espurio.
2. **Combate a Co-Adaptacao de Termos**: No Spambase, enquanto o Baseline concentrava seu peso quase exclusivamente em termos obvios (`word_freq_remove`, `word_freq_free`), o Dropout distribuiu o sinal para variaveis contextuais institucionais (`word_freq_edu`, `word_freq_george`, `word_freq_hpl`), estimulando caminhos de decisao mais robustos.
3. **Distribuicao da Importancia (Curva de Pareto e Entropia)**: A entropia normalizada de atribuicao aumentou e a concentracao Gini reduziu sob regularizacao por Dropout, validando a hipotese de representacoes mais distribuidas.

---

## 6. Estrutura do Repositorio

```text
README.md                              # Documentacao completa do experimento
requirements.txt                       # Dependencias do projeto
run_experiment.py                      # Script principal de orquestracao CLI
src/                                   # Modulos do pipeline
    __init__.py
    dataset.py                         # Carga do OpenML, One-Hot, normalizacao e DataLoaders
    models.py                          # Arquitetura MLP PyTorch parametrizada
    trainer.py                         # Loop de treinamento, metricas e avaliacao
    shap_analysis.py                   # Pipeline SHAP (GradientExplainer) e metricas Gini/Entropia
    visualizer.py                      # Gerador de graficos cientificos em 300 DPI
results/                               # Resultados e figuras do dataset Adult Census
    fig1_training_curves.png
    fig2_roc_pr_confusion.png
    fig3_shap_beeswarm_comparison.png
    fig4_shap_feature_importance.png
    fig5_shap_distribution_metrics.png
    metrics_comparison_summary.csv
    shap_feature_importance_comparison.csv
    training_history_comparison.csv
results_spambase/                      # Resultados e figuras do dataset Spambase
    fig1_training_curves.png
    fig2_roc_pr_confusion.png
    fig3_shap_beeswarm_comparison.png
    fig4_shap_feature_importance.png
    fig5_shap_distribution_metrics.png
    metrics_comparison_summary.csv
    shap_feature_importance_comparison.csv
    training_history_comparison.csv
```

---

## 7. Instalacao e Execucao

### 7.1 Pre-requisitos
Recomenda-se Python 3.10 ou superior. Instale as dependencias:

```bash
pip install -r requirements.txt
```

### 7.2 Executar Experimento no Dataset Adult Census

```bash
python run_experiment.py --dataset adult --dropout 0.20 --epochs 30 --output_dir results
```

### 7.3 Executar Experimento no Dataset Spambase

```bash
python run_experiment.py --dataset spambase --dropout 0.20 --epochs 35 --output_dir results_spambase
```

### 7.4 Parametros da Linha de Comando (CLI)

| Argumento | Tipo | Padrao | Descricao |
| :--- | :---: | :---: | :--- |
| `--dataset` | `str` | `adult` | Dataset a utilizar (`adult`, `spambase`, `synthetic`) |
| `--dropout` | `float` | `0.20` | Taxa de Dropout da rede regularizada ($p$) |
| `--epochs` | `int` | `35` | Numero de epocas de treinamento |
| `--batch_size` | `int` | `256` | Tamanho do mini-batch |
| `--lr` | `float` | `0.001` | Taxa de aprendizado inicial do AdamW |
| `--shap_test_samples` | `int` | `400` | Amostras do conjunto de teste a explicar com SHAP |
| `--shap_bg_samples` | `int` | `150` | Amostras de referencia de background para SHAP |
| `--output_dir` | `str` | `results` | Diretorio de destino para figuras (300 DPI) e CSVs |
| `--seed` | `int` | `42` | Semente pseudoaleatoria para reprodutibilidade |

---

## 8. Figuras Geradas para Publicacao (300 DPI)

Todas as figuras sao salvas em alta resolucao prontas para submissao em conferencias e periodicos cientificos:

1. **Figura 1 (`fig1_training_curves.png`)**: Dinamica de perda (BCE), evolucao do Generalization Gap ($Loss_{val} - Loss_{train}$) e curvas de ROC-AUC/Acuracia ao longo das epocas.
2. **Figura 2 (`fig2_roc_pr_confusion.png`)**: Curvas ROC, Precision-Recall e Matrizes de Confusao normalizadas lado a lado.
3. **Figura 3 (`fig3_shap_beeswarm_comparison.png`)**: Summary Beeswarm plot do SHAP comparando a dispersao das principais features.
4. **Figura 4 (`fig4_shap_feature_importance.png`)**: Ranking horizontal de Importancia Global ($mean(|SHAP|)$).
5. **Figura 5 (`fig5_shap_distribution_metrics.png`)**: Curva de Pareto de Atribuicao Cumulativa e Quantificacao de Regularizacao de Representacao (Indice de Gini e Entropia).

---

## 9. Licenca

Este projeto e disponibilizado sob a licenca MIT.
