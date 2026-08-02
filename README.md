# Impacto do Dropout na Generalização e Atribuição de Features (SHAP) em Classificadores Neurais PyTorch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://github.com/shap/shap)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Este repositório contém o código-fonte, pipeline de dados, rotinas de treino e experimentos da pesquisa sobre o **impacto da regularização estocástica por Dropout na capacidade de generalização e na estrutura de atribuição de features (via SHAP)** em redes neurais profundas (MLP) para classificação binária tabular.

---

## 1. Visão Geral e Motivação Científica

O **Dropout** (Srivastava et al., 2014) é uma técnica de regularização em aprendizado profundo que atua desativando estocasticamente neurônios durante o treino com probabilidade $p$. Teoricamente, o Dropout:

1. **Evita a Co-Adaptação de Neurônios**: Impede que neurônios dependam exclusivamente da presença mútua de ativações específicas para corrigir erros.
2. **Atua como um Ensemble Implícito**: Treina exponencialmente muitas sub-redes esparsas que compartilham pesos, realizando uma média geométrica de predições no teste.
3. **Regulariza Representações Latentes**: Induz a dispersão da importância preditiva por múltiplos caminhos sinápticos, reduzindo a sensibilidade a ruídos espúrios.

Neste projeto, essa dinâmica é avaliada empiricamente em dois datasets tabulares de referência, comparando uma arquitetura **Baseline ($p=0.0$)** contra uma variante **Regularizada com Dropout ($p=0.20$)**, com explicabilidade pós-hoc através de **SHapley Additive exPlanations (SHAP)**.

---

## 2. Datasets de Estudo

| Dataset | Instâncias | Features Originais | Features Pós-Encoding | Tipo de Dados | Prevalência Positiva |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Adult Census Income** (OpenML 1590) | 45.222 | 14 | 104 | Misto (Categórico One-Hot e Contínuo Normalizado) | 24.8% (>50K) |
| **Spambase UCI** (OpenML 44) | 4.601 | 57 | 57 | Contínuo Denso (Frequências de palavras e símbolos) | 39.4% (Spam) |

- **Particionamento Estratificado**: 70% Treino, 15% Validação e 15% Teste (held-out test set não utilizado em nenhuma etapa de ajuste de hiperparâmetros).

---

## 3. Arquitetura da Rede Neural (PyTorch)

Para garantir comparabilidade rigorosa (controlled experiment), ambas as redes compartilham a mesma topologia, inicialização determinística de pesos (Kaiming Normal) e rotina de otimização:

```text
Entrada (d) -> Linear(128) -> BatchNorm1d -> ReLU -> [Dropout(p)] 
            -> Linear(64)  -> BatchNorm1d -> ReLU -> [Dropout(p)] 
            -> Linear(32)  -> BatchNorm1d -> ReLU 
            -> Linear(1)   -> Logit de Saída
```

- **Otimizador**: AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$).
- **Scheduler**: `ReduceLROnPlateau` (fator 0.5, paciência de 4 épocas monitorando a perda de validação).
- **Função de Custo**: `BCEWithLogitsLoss`.

---

## 4. Principais Resultados Empíricos

### 4.1 Desempenho no Dataset Spambase (57 Features)

| Métrica de Avaliação | Baseline ($p=0.0$) | Dropout ($p=0.20$) | Variação ($\Delta$) | Destaque Científico |
| :--- | :---: | :---: | :---: | :--- |
| **Acurácia (Teste)** | 92.19% | **93.20%** | **+1.01%** | Aumento superior a 1% na acurácia global |
| **Precisão** | 90.37% | **91.51%** | **+1.14%** | Menor taxa de falsos positivos |
| **Recall (Sensibilidade)** | 89.71% | **91.18%** | **+1.47%** | Maior taxa de detecção da classe minoritária |
| **F1-Score** | 90.04% | **91.34%** | **+1.30%** | Desempenho harmônico superior |
| **ROC-AUC** | 0.9774 | **0.9803** | **+0.0029** | Excelente discriminação estocástica |
| **PR-AUC (Avg. Precision)**| 0.9632 | **0.9701** | **+0.0069** | Superioridade da curva Precisão-Revocação |
| **Log-Loss** | 0.1910 | **0.1844** | **-0.0066** | Menor incerteza nas predições de teste |
| **Generalization Gap** | **+0.1353** | **+0.0335** | **-75.2%** | **Supressão severa de overfitting** |

### 4.2 Desempenho no Dataset Adult Census Income (104 Features)

| Métrica de Avaliação | Baseline ($p=0.0$) | Dropout ($p=0.20$) | Variação ($\Delta$) | Destaque Científico |
| :--- | :---: | :---: | :---: | :--- |
| **Acurácia (Teste)** | 84.88% | **85.55%** | **+0.67%** | Melhor acurácia obtida no censo |
| **Precisão** | 73.24% | **74.20%** | **+0.96%** | Redução consistente de falsos positivos |
| **Recall (Sensibilidade)** | 61.39% | **63.89%** | **+2.50%** | Forte sensibilidade na classe de alta renda |
| **F1-Score** | 66.80% | **68.66%** | **+1.86%** | Equilíbrio robusto entre precisão e sensibilidade |
| **ROC-AUC** | 0.9086 | **0.9103** | **+0.0017** | Melhor ordenação de probabilidade |
| **PR-AUC (Avg. Precision)**| 0.7758 | **0.7818** | **+0.0060** | Robustez sob desbalanceamento |
| **Generalization Gap** | **+0.1252** | **+0.0353** | **-71.8%** | **Redução expressiva de overfitting** |

---

## 5. Análise Explicativa com SHAP (SHapley Additive exPlanations)

Utilizando o `shap.GradientExplainer` adaptado para modelos PyTorch com background amostral representativo:

1. **Atenuação de Ruído Amostral**: No dataset Adult, a variável `fnlwgt` (peso amostral que representa ruído populacional sem relação causal com renda) figurava em 16º lugar no Baseline. Com o Dropout, caiu 9 posições no ranking, demonstrando que a regularização estocástica penaliza a memorização de ruído espúrio.
2. **Combate à Co-Adaptação de Termos**: No Spambase, enquanto o Baseline concentrava seu peso quase exclusivamente em termos óbvios (`word_freq_remove`, `word_freq_free`), o Dropout distribuiu o sinal para variáveis contextuais institucionais (`word_freq_edu`, `word_freq_george`, `word_freq_hpl`), estimulando caminhos de decisão mais robustos.
3. **Distribuição da Importância (Curva de Pareto e Entropia)**: A entropia normalizada de atribuição aumentou e a concentração Gini reduziu sob regularização por Dropout, validando a hipótese de representações mais distribuídas.

---

## 6. Estrutura do Repositório

```text
README.md                              # Documentação completa do experimento
requirements.txt                       # Dependências do projeto
run_experiment.py                      # Script principal de orquestração CLI
src/                                   # Módulos do pipeline
    __init__.py
    dataset.py                         # Carga do OpenML, One-Hot, normalização e DataLoaders
    models.py                          # Arquitetura MLP PyTorch parametrizada
    trainer.py                         # Loop de treinamento, métricas e avaliação
    shap_analysis.py                   # Pipeline SHAP (GradientExplainer) e métricas Gini/Entropia
    visualizer.py                      # Gerador de gráficos científicos em 300 DPI
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

## 7. Instalação e Execução

### 7.1 Pré-requisitos
Recomenda-se Python 3.10 ou superior. Instale as dependências:

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

### 7.4 Parâmetros da Linha de Comando (CLI)

| Argumento | Tipo | Padrão | Descrição |
| :--- | :---: | :---: | :--- |
| `--dataset` | `str` | `adult` | Dataset a utilizar (`adult`, `spambase`, `synthetic`) |
| `--dropout` | `float` | `0.20` | Taxa de Dropout da rede regularizada ($p$) |
| `--epochs` | `int` | `35` | Número de épocas de treinamento |
| `--batch_size` | `int` | `256` | Tamanho do mini-batch |
| `--lr` | `float` | `0.001` | Taxa de aprendizado inicial do AdamW |
| `--shap_test_samples` | `int` | `400` | Amostras do conjunto de teste a explicar com SHAP |
| `--shap_bg_samples` | `int` | `150` | Amostras de referência de background para SHAP |
| `--output_dir` | `str` | `results` | Diretório de destino para figuras (300 DPI) e CSVs |
| `--seed` | `int` | `42` | Semente pseudoaleatória para reprodutibilidade |

---

## 8. Figuras Geradas para Publicação (300 DPI)

Todas as figuras são salvas em alta resolução prontas para submissão em conferências e periódicos científicos:

1. **Figura 1 (`fig1_training_curves.png`)**: Dinâmica de perda (BCE), evolução do Generalization Gap ($Loss_{val} - Loss_{train}$) e curvas de ROC-AUC/Acurácia ao longo das épocas.
2. **Figura 2 (`fig2_roc_pr_confusion.png`)**: Curvas ROC, Precision-Recall e Matrizes de Confusão normalizadas lado a lado.
3. **Figura 3 (`fig3_shap_beeswarm_comparison.png`)**: Summary Beeswarm plot do SHAP comparando a dispersão das principais features.
4. **Figura 4 (`fig4_shap_feature_importance.png`)**: Ranking horizontal de Importância Global ($mean(|SHAP|)$).
5. **Figura 5 (`fig5_shap_distribution_metrics.png`)**: Curva de Pareto de Atribuição Cumulativa e Quantificação de Regularização de Representação (Índice de Gini e Entropia).

---

## 9. Licença

Este projeto é disponibilizado sob a licença MIT.
