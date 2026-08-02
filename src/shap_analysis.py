"""
SHAP (SHapley Additive exPlanations) Analysis Module for PyTorch Tabular Classifiers.
Quantifies feature attributions, feature co-adaptation, and attribution distribution metrics:
- Global Feature Importance (mean |SHAP|)
- Gini Index of Attribution Concentration
- Normalized Feature Entropy (Distributed representation metric)
- Top Feature Rank Shifts
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
import torch
import torch.nn as nn
import shap


class PyTorchModelWrapper(nn.Module):
    """Wraps model to ensure eval mode and float tensor CPU/GPU compatibility for SHAP."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ShapExplainerPipeline:
    """
    Computes and analyzes SHAP values for Baseline vs Dropout PyTorch classifiers.
    """
    def __init__(
        self,
        baseline_model: nn.Module,
        dropout_model: nn.Module,
        feature_names: List[str],
        background_samples: int = 150,
        test_samples: int = 500,
        device: Optional[torch.device] = None,
        random_state: int = 42
    ):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.baseline_model = baseline_model.to(self.device).eval()
        self.dropout_model = dropout_model.to(self.device).eval()
        self.feature_names = feature_names
        self.background_samples = background_samples
        self.test_samples = test_samples
        self.random_state = random_state

        self.shap_values_baseline: Optional[np.ndarray] = None
        self.shap_values_dropout: Optional[np.ndarray] = None
        self.X_explained: Optional[np.ndarray] = None
        self.X_explained_df: Optional[pd.DataFrame] = None

    def compute_shap_values(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes SHAP values using GradientExplainer on PyTorch models.
        """
        print(f"\n========================================================")
        print(f"Computing SHAP explanations via GradientExplainer...")
        print(f"========================================================")
        
        np.random.seed(self.random_state)
        
        # Select background dataset (representative reference for Shapley baselines)
        bg_idx = np.random.choice(len(X_train), size=min(self.background_samples, len(X_train)), replace=False)
        X_bg = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(self.device)

        # Select test samples to explain
        test_size = min(self.test_samples, len(X_test))
        test_idx = np.random.choice(len(X_test), size=test_size, replace=False)
        self.X_explained = X_test[test_idx]
        self.X_explained_df = pd.DataFrame(self.X_explained, columns=self.feature_names)
        
        X_exp_tensor = torch.tensor(self.X_explained, dtype=torch.float32).to(self.device)

        # 1. Baseline Model Explainer
        print(f"Explaining Baseline model (Dropout 0%)...")
        explainer_base = shap.GradientExplainer(self.baseline_model, X_bg)
        shap_base_raw = explainer_base.shap_values(X_exp_tensor)
        if isinstance(shap_base_raw, list):
            self.shap_values_baseline = np.array(shap_base_raw[0])
        elif isinstance(shap_base_raw, np.ndarray):
            self.shap_values_baseline = shap_base_raw.squeeze()
        else:
            self.shap_values_baseline = np.array(shap_base_raw).squeeze()

        # 2. Dropout Model Explainer
        print(f"Explaining Dropout model (Dropout 30%)...")
        explainer_drop = shap.GradientExplainer(self.dropout_model, X_bg)
        shap_drop_raw = explainer_drop.shap_values(X_exp_tensor)
        if isinstance(shap_drop_raw, list):
            self.shap_values_dropout = np.array(shap_drop_raw[0])
        elif isinstance(shap_drop_raw, np.ndarray):
            self.shap_values_dropout = shap_drop_raw.squeeze()
        else:
            self.shap_values_dropout = np.array(shap_drop_raw).squeeze()

        print(f"[OK] SHAP computed successfully! Shape: {self.shap_values_baseline.shape}")
        return self.shap_values_baseline, self.shap_values_dropout, self.X_explained

    def calculate_attribution_metrics(self) -> Dict[str, Any]:
        """
        Calculates scientific metrics quantifying attribution concentration,
        Gini index, entropy, and feature importance rankings.
        """
        if self.shap_values_baseline is None or self.shap_values_dropout is None:
            raise ValueError("Must run compute_shap_values before calculating metrics.")

        # Mean Absolute SHAP (Global Feature Importance)
        mean_abs_base = np.mean(np.abs(self.shap_values_baseline), axis=0)
        mean_abs_drop = np.mean(np.abs(self.shap_values_dropout), axis=0)

        # Gini Index of Feature Importance
        gini_base = self._calculate_gini(mean_abs_base)
        gini_drop = self._calculate_gini(mean_abs_drop)

        # Normalized Shannon Entropy of Attributions
        entropy_base = self._calculate_normalized_entropy(mean_abs_base)
        entropy_drop = self._calculate_normalized_entropy(mean_abs_drop)

        # Top 5 Concentration Ratio (Percentage of total attribution captured by top 5 features)
        top5_base_ratio = np.sum(np.sort(mean_abs_base)[-5:]) / (np.sum(mean_abs_base) + 1e-12)
        top5_drop_ratio = np.sum(np.sort(mean_abs_drop)[-5:]) / (np.sum(mean_abs_drop) + 1e-12)

        # Feature Importance DataFrame
        df_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Baseline_Mean_Abs_SHAP': mean_abs_base,
            'Dropout_Mean_Abs_SHAP': mean_abs_drop,
        })
        df_importance['Baseline_Rank'] = df_importance['Baseline_Mean_Abs_SHAP'].rank(ascending=False).astype(int)
        df_importance['Dropout_Rank'] = df_importance['Dropout_Mean_Abs_SHAP'].rank(ascending=False).astype(int)
        df_importance['Rank_Change'] = df_importance['Baseline_Rank'] - df_importance['Dropout_Rank']
        
        # Sort by baseline importance
        df_importance = df_importance.sort_values(by='Baseline_Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

        metrics = {
            'gini_baseline': gini_base,
            'gini_dropout': gini_drop,
            'entropy_baseline': entropy_base,
            'entropy_dropout': entropy_drop,
            'top5_ratio_baseline': top5_base_ratio,
            'top5_ratio_dropout': top5_drop_ratio,
            'importance_df': df_importance,
            'mean_abs_baseline': mean_abs_base,
            'mean_abs_dropout': mean_abs_drop
        }

        return metrics

    @staticmethod
    def _calculate_gini(array: np.ndarray) -> float:
        """Computes Gini coefficient of an array representing attribution distribution."""
        arr = np.sort(np.maximum(array, 0.0))
        n = len(arr)
        if n == 0 or np.sum(arr) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))

    @staticmethod
    def _calculate_normalized_entropy(array: np.ndarray) -> float:
        """Computes normalized Shannon entropy (H / log(D)) of feature attributions in [0, 1]."""
        arr = np.maximum(array, 0.0)
        total = np.sum(arr)
        if total == 0:
            return 0.0
        p = arr / total
        p = p[p > 0]
        H = -np.sum(p * np.log(p))
        max_H = np.log(len(array))
        return H / max_H if max_H > 0 else 0.0
