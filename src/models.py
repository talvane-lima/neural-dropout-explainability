"""
Neural Network Architectures for Tabular Binary Classification in PyTorch.
Implements parameter-matched MLPs for controlled Dropout effect studies:
- Baseline (p=0.0, no dropout)
- Regularized (p=0.30 dropout)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification on tabular representations.
    
    Architecture:
    Input -> [Linear -> BatchNorm1d -> ReLU -> Dropout(p)] x N -> Linear(hidden[-1], 1)
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.0,
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
            
        # Final classification head (outputs raw logit)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Kaiming normal distribution for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (shape: [batch_size, 1])."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns positive class probability via Sigmoid activation."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def build_models(
    in_features: int,
    hidden_dims: List[int] = [128, 64, 32],
    dropout_rate: float = 0.20,
    seed: int = 42
):
    """
    Instantiates both Baseline (Dropout 0.0) and Regularized (Dropout p) models
    with identical initial architectural configurations.
    """
    torch.manual_seed(seed)
    baseline_model = TabularMLP(
        in_features=in_features,
        hidden_dims=hidden_dims,
        dropout_rate=0.0,
        use_batchnorm=True
    )
    
    torch.manual_seed(seed)
    dropout_model = TabularMLP(
        in_features=in_features,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batchnorm=True
    )
    
    return baseline_model, dropout_model
