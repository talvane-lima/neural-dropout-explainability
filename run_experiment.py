"""
Main Experiment Orchestration Script.
Executes the comparative empirical study between PyTorch Binary Classifiers:
1. Baseline Neural Network (p=0.0, no dropout)
2. Regularized Neural Network (p=0.30 dropout)
Performs training metric tracking, test evaluation, and SHAP explainability analysis.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from src.dataset import TabularDataModule
from src.models import build_models
from src.trainer import ModelTrainer
from src.shap_analysis import ShapExplainerPipeline
from src.visualizer import PublicationVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Dropout vs SHAP Explainability Study")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "spambase", "synthetic"],
                        help="Dataset to use for the experiment (default: adult)")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Number of training epochs (default: 35)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DataLoaders (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for AdamW (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.20,
                        help="Dropout probability for regularized model (default: 0.20)")
    parser.add_argument("--shap_test_samples", type=int, default=400,
                        help="Number of test samples to explain with SHAP (default: 400)")
    parser.add_argument("--shap_bg_samples", type=int, default=150,
                        help="Number of background reference samples for SHAP (default: 150)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save publication figures and tables (default: results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def set_seed(seed: int = 42):
    """Ensures full determinism across numpy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("      RESEARCH EXPERIMENT: IMPACT OF DROPOUT ON NEURAL NETWORKS      ")
    print("      Classification Performance & SHAP Representation Analysis     ")
    print("=" * 70)
    print(f"Device:               {device}")
    print(f"Dataset:              {args.dataset}")
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Dropout Rate:         {args.dropout * 100:.1f}%")
    print(f"Random Seed:          {args.seed}")
    print(f"Output Directory:     {args.output_dir}")
    print("=" * 70)

    # 1. Dataset Loading & Preprocessing
    data_module = TabularDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        random_state=args.seed
    )
    X_train, y_train, X_val, y_val, X_test, y_test = data_module.load_and_preprocess()
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    in_features = data_module.input_dim
    feature_names = data_module.feature_names

    # 2. Build Models (Baseline vs Dropout)
    hidden_architecture = [128, 64, 32]
    torch.manual_seed(args.seed)
    baseline_model, dropout_model = build_models(
        in_features=in_features,
        hidden_dims=hidden_architecture,
        dropout_rate=args.dropout,
        seed=args.seed
    )

    # 3. Train Baseline Model (p=0.0)
    trainer_baseline = ModelTrainer(
        model=baseline_model,
        model_name="Baseline (p=0.0)",
        learning_rate=args.lr,
        device=device
    )
    history_baseline = trainer_baseline.fit(train_loader, val_loader, epochs=args.epochs)
    metrics_baseline = trainer_baseline.evaluate_test(test_loader)

    # 4. Train Dropout Model (p=args.dropout)
    trainer_dropout = ModelTrainer(
        model=dropout_model,
        model_name=f"Dropout (p={args.dropout:.2f})",
        learning_rate=args.lr,
        device=device
    )
    history_dropout = trainer_dropout.fit(train_loader, val_loader, epochs=args.epochs)
    metrics_dropout = trainer_dropout.evaluate_test(test_loader)

    # 5. SHAP Explainability Analysis
    shap_pipeline = ShapExplainerPipeline(
        baseline_model=trainer_baseline.model,
        dropout_model=trainer_dropout.model,
        feature_names=feature_names,
        background_samples=args.shap_bg_samples,
        test_samples=args.shap_test_samples,
        device=device,
        random_state=args.seed
    )
    shap_base, shap_drop, X_exp = shap_pipeline.compute_shap_values(X_train, X_test)
    attribution_metrics = shap_pipeline.calculate_attribution_metrics()

    # 6. Generate Publication Visualizations (300 DPI)
    visualizer = PublicationVisualizer(output_dir=args.output_dir, dropout_rate=args.dropout)
    print(f"\n[*] Generating high-resolution publication figures...")
    
    fig1 = visualizer.plot_training_curves(history_baseline, history_dropout)
    fig2 = visualizer.plot_roc_pr_confusion(metrics_baseline, metrics_dropout)
    fig3 = visualizer.plot_shap_beeswarm_comparison(shap_base, shap_drop, shap_pipeline.X_explained_df, top_n=15)
    fig4 = visualizer.plot_shap_feature_importance(attribution_metrics['importance_df'], top_n=20)
    fig5 = visualizer.plot_shap_distribution_metrics(attribution_metrics)

    # 7. Compile Results and Tables
    summary_data = [
        {
            "Model": "Baseline (p=0.0)",
            "Accuracy": f"{metrics_baseline['accuracy']*100:.2f}%",
            "Precision": f"{metrics_baseline['precision']*100:.2f}%",
            "Recall": f"{metrics_baseline['recall']*100:.2f}%",
            "F1-Score": f"{metrics_baseline['f1']*100:.2f}%",
            "ROC-AUC": f"{metrics_baseline['roc_auc']:.4f}",
            "PR-AUC": f"{metrics_baseline['pr_auc']:.4f}",
            "Log-Loss": f"{metrics_baseline['log_loss']:.4f}",
            "Brier Score": f"{metrics_baseline['brier_score']:.4f}",
            "Generalization Gap (Loss)": f"{history_baseline['gen_gap'][-1]:+.4f}",
            "Gini Concentration": f"{attribution_metrics['gini_baseline']:.4f}",
            "Attribution Entropy": f"{attribution_metrics['entropy_baseline']:.4f}",
            "Top-5 Feature Share": f"{attribution_metrics['top5_ratio_baseline']*100:.1f}%"
        },
        {
            "Model": f"Dropout (p={args.dropout:.2f})",
            "Accuracy": f"{metrics_dropout['accuracy']*100:.2f}%",
            "Precision": f"{metrics_dropout['precision']*100:.2f}%",
            "Recall": f"{metrics_dropout['recall']*100:.2f}%",
            "F1-Score": f"{metrics_dropout['f1']*100:.2f}%",
            "ROC-AUC": f"{metrics_dropout['roc_auc']:.4f}",
            "PR-AUC": f"{metrics_dropout['pr_auc']:.4f}",
            "Log-Loss": f"{metrics_dropout['log_loss']:.4f}",
            "Brier Score": f"{metrics_dropout['brier_score']:.4f}",
            "Generalization Gap (Loss)": f"{history_dropout['gen_gap'][-1]:+.4f}",
            "Gini Concentration": f"{attribution_metrics['gini_dropout']:.4f}",
            "Attribution Entropy": f"{attribution_metrics['entropy_dropout']:.4f}",
            "Top-5 Feature Share": f"{attribution_metrics['top5_ratio_dropout']*100:.1f}%"
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    
    # Save CSV tables
    summary_csv_path = os.path.join(args.output_dir, "metrics_comparison_summary.csv")
    importance_csv_path = os.path.join(args.output_dir, "shap_feature_importance_comparison.csv")
    history_csv_path = os.path.join(args.output_dir, "training_history_comparison.csv")

    df_summary.to_csv(summary_csv_path, index=False)
    attribution_metrics['importance_df'].to_csv(importance_csv_path, index=False)
    
    df_hist = pd.DataFrame({
        'epoch': history_baseline['epoch'],
        'base_train_loss': history_baseline['train_loss'],
        'base_val_loss': history_baseline['val_loss'],
        'base_val_auc': history_baseline['val_roc_auc'],
        'base_gen_gap': history_baseline['gen_gap'],
        'drop_train_loss': history_dropout['train_loss'],
        'drop_val_loss': history_dropout['val_loss'],
        'drop_val_auc': history_dropout['val_roc_auc'],
        'drop_gen_gap': history_dropout['gen_gap']
    })
    df_hist.to_csv(history_csv_path, index=False)

    print("\n" + "=" * 70)
    print("                     FINAL EXPERIMENTAL RESULTS                     ")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)
    print(f"\n[OK] All figures and tables successfully saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
