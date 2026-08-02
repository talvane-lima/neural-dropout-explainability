"""
Training and Evaluation Pipeline with Convergence Tracking and Scientific Metrics.
Calculates comprehensive classification metrics for research publications.
"""

import time
import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)


class ModelTrainer:
    """
    Handles PyTorch model training, validation tracking per epoch,
    early stopping, and thorough testing evaluation.
    """
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "Model",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        self.model_name = model_name
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.history: Dict[str, list] = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_roc_auc': [],
            'val_f1': [],
            'gen_gap': []
        }
        self.best_model_weights = None
        self.best_val_loss = float('inf')

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 40,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Runs the training loop across specified epochs."""
        if verbose:
            print(f"\n========================================================")
            print(f"[*] Training {self.model_name} on {self.device} for {epochs} epochs")
            print(f"========================================================")
            
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training Phase
            self.model.train()
            train_loss_accum = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss_accum += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                train_correct += (preds == y_batch).sum().item()
                train_total += X_batch.size(0)

            train_loss = train_loss_accum / train_total
            train_acc = train_correct / train_total

            # Validation Phase
            val_metrics = self.evaluate_loader(val_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            val_auc = val_metrics['roc_auc']
            val_f1 = val_metrics['f1']
            gen_gap = val_loss - train_loss

            self.scheduler.step(val_loss)

            # Checkpoint best weights based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

            # Store history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_roc_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            self.history['gen_gap'].append(gen_gap)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% AUC: {val_auc:.4f} | Gap: {gen_gap:+.4f}")

        # Load best weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        elapsed = time.time() - start_time
        if verbose:
            print(f"[OK] Training completed in {elapsed:.2f}s. Best Val Loss: {self.best_val_loss:.4f}")
            
        return self.history

    def evaluate_loader(self, loader: DataLoader) -> Dict[str, float]:
        """Computes loss, accuracy, ROC-AUC, and F1 on a dataloader."""
        self.model.eval()
        loss_accum = 0.0
        total = 0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                loss_accum += loss.item() * X_batch.size(0)
                total += X_batch.size(0)
                
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(float)
                targets = y_batch.cpu().numpy().flatten()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        loss = loss_accum / total
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        return {
            'loss': loss,
            'accuracy': acc,
            'f1': f1,
            'roc_auc': auc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Computes complete research evaluation metrics on the held-out test partition."""
        eval_res = self.evaluate_loader(test_loader)
        y_true = eval_res['y_true']
        y_pred = eval_res['y_pred']
        y_prob = eval_res['y_prob']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

        metrics = {
            'model_name': self.model_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss,
            'brier_score': brier,
            'confusion_matrix': cm,
            'confusion_matrix_norm': cm_norm,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        return metrics
