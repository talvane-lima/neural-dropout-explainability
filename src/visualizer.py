"""
Publication-Quality Visualization Module (300 DPI).
Generates ready-to-publish scientific figures for academic papers:
1. Training & Convergence Dynamics (Loss, Accuracy, ROC-AUC, Generalization Gap)
2. ROC, Precision-Recall & Normalized Confusion Matrices
3. Side-by-Side SHAP Summary Beeswarm Plots
4. Comparative Global Feature Importance (Top-20 features)
5. Attribution Dispersion, Gini Index & Pareto Cumulative Curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import shap


# Configure publication-grade styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Custom scientific palette
COLOR_BASE = "#D95F02"   # Deep Orange/Coral for Baseline (No Dropout)
COLOR_DROP = "#1B9E77"   # Teal/Emerald for Dropout 30%
COLOR_ACCENT = "#7570B3" # Soft Purple


class PublicationVisualizer:
    """
    Handles plotting and saving publication-ready figures.
    """
    def __init__(self, output_dir: str = "results", dropout_rate: float = 0.20):
        self.output_dir = output_dir
        self.dropout_rate = dropout_rate
        self.drop_pct_label = f"Dropout {int(dropout_rate*100)}%"
        self.drop_p_label = f"Dropout (p={dropout_rate:.2f})"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_training_curves(
        self,
        history_base: Dict[str, list],
        history_drop: Dict[str, list],
        filename: str = "fig1_training_curves.png"
    ) -> str:
        """Plots training and validation loss, accuracy, and generalization gap."""
        epochs = history_base['epoch']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Loss & Generalization Gap
        ax1 = axes[0]
        ax1.plot(epochs, history_base['train_loss'], label='Baseline (Train)', color=COLOR_BASE, linestyle='--', alpha=0.7, lw=1.8)
        ax1.plot(epochs, history_base['val_loss'], label='Baseline (Val)', color=COLOR_BASE, lw=2.4)
        ax1.plot(epochs, history_drop['train_loss'], label=f'{self.drop_pct_label} (Train)', color=COLOR_DROP, linestyle='--', alpha=0.7, lw=1.8)
        ax1.plot(epochs, history_drop['val_loss'], label=f'{self.drop_pct_label} (Val)', color=COLOR_DROP, lw=2.4)
        ax1.set_title('(A) Binary Cross-Entropy Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

        # 2. Generalization Gap (Val Loss - Train Loss)
        ax2 = axes[1]
        ax2.plot(epochs, history_base['gen_gap'], label=f'Baseline Gap (Final: {history_base["gen_gap"][-1]:.3f})', color=COLOR_BASE, lw=2.2)
        ax2.plot(epochs, history_drop['gen_gap'], label=f'{self.drop_pct_label} Gap (Final: {history_drop["gen_gap"][-1]:.3f})', color=COLOR_DROP, lw=2.2)
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.7)
        ax2.set_title('(B) Generalization Gap ($Loss_{val} - Loss_{train}$)', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

        # 3. Validation ROC-AUC & Accuracy
        ax3 = axes[2]
        ax3.plot(epochs, history_base['val_roc_auc'], label=f'Baseline ROC-AUC (Max: {max(history_base["val_roc_auc"]):.3f})', color=COLOR_BASE, lw=2.2)
        ax3.plot(epochs, history_drop['val_roc_auc'], label=f'{self.drop_pct_label} ROC-AUC (Max: {max(history_drop["val_roc_auc"]):.3f})', color=COLOR_DROP, lw=2.2)
        ax3.plot(epochs, history_base['val_acc'], label='Baseline Acc', color=COLOR_BASE, linestyle=':', alpha=0.6, lw=1.5)
        ax3.plot(epochs, history_drop['val_acc'], label=f'{self.drop_pct_label} Acc', color=COLOR_DROP, linestyle=':', alpha=0.6, lw=1.5)
        ax3.set_title('(C) Validation ROC-AUC & Accuracy', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Metric Score')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(frameon=True, facecolor='white', framealpha=0.9)

        plt.suptitle(f"Comparative Training Dynamics: Baseline vs. {self.drop_pct_label}", fontsize=15, fontweight='bold', y=1.03)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[OK] Saved {filepath}")
        return filepath

    def plot_roc_pr_confusion(
        self,
        metrics_base: Dict[str, Any],
        metrics_drop: Dict[str, Any],
        filename: str = "fig2_roc_pr_confusion.png"
    ) -> str:
        """Plots ROC curves, PR curves, and normalized confusion matrices side-by-side."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. ROC Curves
        ax1 = axes[0, 0]
        ax1.plot(metrics_base['fpr'], metrics_base['tpr'],
                 label=f"Baseline (AUC = {metrics_base['roc_auc']:.4f})", color=COLOR_BASE, lw=2.5)
        ax1.plot(metrics_drop['fpr'], metrics_drop['tpr'],
                 label=f"{self.drop_pct_label} (AUC = {metrics_drop['roc_auc']:.4f})", color=COLOR_DROP, lw=2.5)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
        ax1.set_title('(A) Receiver Operating Characteristic (ROC)', fontweight='bold')
        ax1.set_xlabel('False Positive Rate (1 - Specificity)')
        ax1.set_ylabel('True Positive Rate (Sensitivity)')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='lower right', frameon=True)

        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        ax2.plot(metrics_base['recall_curve'], metrics_base['precision_curve'],
                 label=f"Baseline (PR-AUC = {metrics_base['pr_auc']:.4f})", color=COLOR_BASE, lw=2.5)
        ax2.plot(metrics_drop['recall_curve'], metrics_drop['precision_curve'],
                 label=f"{self.drop_pct_label} (PR-AUC = {metrics_drop['pr_auc']:.4f})", color=COLOR_DROP, lw=2.5)
        ax2.set_title('(B) Precision-Recall Curve', fontweight='bold')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='lower left', frameon=True)

        # 3. Confusion Matrix Baseline
        ax3 = axes[1, 0]
        cm_base_norm = metrics_base['confusion_matrix_norm']
        cm_base_raw = metrics_base['confusion_matrix']
        annot_base = np.array([[f"{val:.1%}\n(n={raw})" for val, raw in zip(row_norm, row_raw)]
                               for row_norm, row_raw in zip(cm_base_norm, cm_base_raw)])
        sns.heatmap(cm_base_norm, annot=annot_base, fmt="", cmap="Oranges", cbar=False, ax=ax3,
                    xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
        ax3.set_title(f"(C) Confusion Matrix: Baseline (Acc: {metrics_base['accuracy']*100:.1f}%)", fontweight='bold')
        ax3.set_xlabel('Predicted Label')
        ax3.set_ylabel('True Label')

        # 4. Confusion Matrix Dropout
        ax4 = axes[1, 1]
        cm_drop_norm = metrics_drop['confusion_matrix_norm']
        cm_drop_raw = metrics_drop['confusion_matrix']
        annot_drop = np.array([[f"{val:.1%}\n(n={raw})" for val, raw in zip(row_norm, row_raw)]
                               for row_norm, row_raw in zip(cm_drop_norm, cm_drop_raw)])
        sns.heatmap(cm_drop_norm, annot=annot_drop, fmt="", cmap="Greens", cbar=False, ax=ax4,
                    xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
        ax4.set_title(f"(D) Confusion Matrix: {self.drop_pct_label} (Acc: {metrics_drop['accuracy']*100:.1f}%)", fontweight='bold')
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')

        plt.suptitle("Held-Out Test Performance: Discrimination & Classification Matrix", fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[OK] Saved {filepath}")
        return filepath

    def plot_shap_beeswarm_comparison(
        self,
        shap_values_base: np.ndarray,
        shap_values_drop: np.ndarray,
        X_df: pd.DataFrame,
        top_n: int = 15,
        filename: str = "fig3_shap_beeswarm_comparison.png"
    ) -> str:
        """Plots side-by-side SHAP beeswarm summary plots for Baseline vs Dropout."""
        # Clean feature names for publication readability
        clean_df = X_df.copy()
        clean_df.columns = [c.replace('_', ' ') for c in clean_df.columns]

        fig = plt.figure(figsize=(20, 9))
        
        # Subplot 1: Baseline Beeswarm
        ax1 = fig.add_subplot(1, 2, 1)
        plt.sca(ax1)
        shap.summary_plot(
            shap_values_base,
            clean_df,
            max_display=top_n,
            show=False,
            plot_size=None,
            color_bar=True
        )
        plt.title(f"(A) Baseline Network (No Dropout, p=0.0)\nHigh attribution concentration & brittle co-adaptation", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("SHAP value (Impact on log-odds output)")

        # Subplot 2: Dropout Beeswarm
        ax2 = fig.add_subplot(1, 2, 2)
        plt.sca(ax2)
        shap.summary_plot(
            shap_values_drop,
            clean_df,
            max_display=top_n,
            show=False,
            plot_size=None,
            color_bar=True
        )
        plt.title(f"(B) Regularized Network ({self.drop_p_label})\nSmoothed attribution & distributed feature representations", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("SHAP value (Impact on log-odds output)")

        plt.suptitle(f"SHAP Beeswarm Explanation: Feature Attribution Comparison (Top {top_n} Features)", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[OK] Saved {filepath}")
        return filepath

    def plot_shap_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "fig4_shap_feature_importance.png"
    ) -> str:
        """Plots grouped horizontal bar chart of mean absolute SHAP values for top features."""
        df_top = importance_df.head(top_n).iloc[::-1]  # Reverse for top-to-bottom horizontal bars
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(df_top))
        height = 0.38
        
        clean_labels = [f.replace('_', ' ') for f in df_top['Feature']]

        rects1 = ax.barh(y_pos + height/2, df_top['Baseline_Mean_Abs_SHAP'], height,
                         label='Baseline (p=0.0)', color=COLOR_BASE, alpha=0.85, edgecolor='black', linewidth=0.5)
        rects2 = ax.barh(y_pos - height/2, df_top['Dropout_Mean_Abs_SHAP'], height,
                         label=self.drop_p_label, color=COLOR_DROP, alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_labels, fontweight='medium')
        ax.set_xlabel('Mean Absolute SHAP Value: $\\frac{1}{N} \\sum |\\phi_j|$ (Global Importance)')
        ax.set_title(f"Comparative Global Feature Importance (Top {top_n} Features)", fontweight='bold', fontsize=14, pad=15)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[OK] Saved {filepath}")
        return filepath

    def plot_shap_distribution_metrics(
        self,
        attribution_metrics: Dict[str, Any],
        filename: str = "fig5_shap_distribution_metrics.png"
    ) -> str:
        """
        Plots scientific dispersion metrics:
        1. Gini index and Entropy bar comparison.
        2. Pareto Cumulative Attribution Curve (Lorenz curve of feature weights).
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        mean_base = np.sort(attribution_metrics['mean_abs_baseline'])[::-1]
        mean_drop = np.sort(attribution_metrics['mean_abs_dropout'])[::-1]

        # Cumulative attribution percentage
        cum_base = np.cumsum(mean_base) / np.sum(mean_base) * 100
        cum_drop = np.cumsum(mean_drop) / np.sum(mean_drop) * 100
        feature_ranks = np.arange(1, len(mean_base) + 1)

        # 1. Pareto / Cumulative Attribution Curve
        ax1 = axes[0]
        ax1.plot(feature_ranks, cum_base, label=f"Baseline (Top 5: {attribution_metrics['top5_ratio_baseline']*100:.1f}%)",
                 color=COLOR_BASE, lw=2.5)
        ax1.plot(feature_ranks, cum_drop, label=f"{self.drop_pct_label} (Top 5: {attribution_metrics['top5_ratio_dropout']*100:.1f}%)",
                 color=COLOR_DROP, lw=2.5)
        ax1.plot([1, len(mean_base)], [100/len(mean_base), 100], 'k--', alpha=0.5, label='Uniform Equality Line')
        ax1.set_title('(A) Cumulative Attribution Distribution (Pareto Curve)', fontweight='bold')
        ax1.set_xlabel('Number of Ranked Features (Top $k$)')
        ax1.set_ylabel('Cumulative % of Total Model Attribution')
        ax1.set_xlim(1, min(40, len(mean_base)))
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='lower right', frameon=True)

        # 2. Gini & Entropy Comparison Bars
        ax2 = axes[1]
        metrics_names = ['Gini Concentration Index\n(Lower = Less Co-adaptation)', 'Normalized Attribution Entropy\n(Higher = More Distributed)']
        base_vals = [attribution_metrics['gini_baseline'], attribution_metrics['entropy_baseline']]
        drop_vals = [attribution_metrics['gini_dropout'], attribution_metrics['entropy_dropout']]

        x = np.arange(len(metrics_names))
        width = 0.35

        rects1 = ax2.bar(x - width/2, base_vals, width, label='Baseline (p=0.0)', color=COLOR_BASE, alpha=0.85, edgecolor='black', lw=0.5)
        rects2 = ax2.bar(x + width/2, drop_vals, width, label=self.drop_p_label, color=COLOR_DROP, alpha=0.85, edgecolor='black', lw=0.5)

        for rect in rects1:
            height = rect.get_height()
            ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        for rect in rects2:
            height = rect.get_height()
            ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names, fontweight='medium')
        ax2.set_ylabel('Metric Value [0, 1]')
        ax2.set_ylim(0, 1.15)
        ax2.set_title('(B) Representation Regularization Quantification', fontweight='bold')
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', frameon=True)

        plt.suptitle("Dropout Impact on Feature Co-Adaptation & Representation Smoothness", fontsize=15, fontweight='bold', y=1.03)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[OK] Saved {filepath}")
        return filepath
