"""
Dataset Loading and Preprocessing Module for PyTorch Tabular Binary Classification.
Supports:
- Adult Census Income (OpenML, ~48.8k instances, ~105 one-hot features)
- Spambase (OpenML, 4.6k instances, 57 continuous features)
- Synthetic Redundant Benchmark (for controlled feature co-adaptation studies)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader


class TabularDataModule:
    """
    Handles data acquisition, preprocessing, encoding, normalization,
    and PyTorch DataLoader generation with reproducible splits.
    """
    def __init__(
        self,
        dataset_name: str = "adult",
        batch_size: int = 256,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.feature_names: List[str] = []
        self.input_dim: int = 0
        self.scaler: Optional[StandardScaler] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        
        self.X_train_np: Optional[np.ndarray] = None
        self.y_train_np: Optional[np.ndarray] = None
        self.X_val_np: Optional[np.ndarray] = None
        self.y_val_np: Optional[np.ndarray] = None
        self.X_test_np: Optional[np.ndarray] = None
        self.y_test_np: Optional[np.ndarray] = None
        
        self.X_test_df: Optional[pd.DataFrame] = None

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and prepares the specified dataset."""
        print(f"[*] Loading dataset '{self.dataset_name}'...")
        
        if self.dataset_name in ["adult", "census"]:
            X_df, y_ser = self._load_adult()
            X_processed, feature_names = self._preprocess_mixed_data(X_df)
            y_arr = (y_ser == ">50K").astype(np.float32).values
        elif self.dataset_name in ["spam", "spambase"]:
            X_df, y_ser = self._load_spambase()
            feature_names = list(X_df.columns)
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_df.values.astype(np.float32))
            y_arr = y_ser.astype(np.float32).values
        elif self.dataset_name in ["synthetic", "redundant"]:
            X_arr, y_arr, feature_names = self._generate_synthetic()
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_arr)
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}. Choose from 'adult', 'spambase', 'synthetic'.")

        self.feature_names = feature_names
        self.input_dim = X_processed.shape[1]
        
        # Split: Train (1 - val - test), Val (val), Test (test)
        # First split train+val vs test
        test_fraction = self.test_size
        val_fraction_adjusted = self.val_size / (1.0 - test_fraction)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_processed, y_arr,
            test_size=test_fraction,
            stratify=y_arr,
            random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_fraction_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )

        self.X_train_np = X_train.astype(np.float32)
        self.y_train_np = y_train.astype(np.float32)
        self.X_val_np = X_val.astype(np.float32)
        self.y_val_np = y_val.astype(np.float32)
        self.X_test_np = X_test.astype(np.float32)
        self.y_test_np = y_test.astype(np.float32)

        self.X_test_df = pd.DataFrame(self.X_test_np, columns=self.feature_names)

        print(f"Dataset prepared: {X_processed.shape[0]} total samples, {self.input_dim} features.")
        print(f"    Train set: {self.X_train_np.shape[0]} samples (Positives: {np.mean(self.y_train_np)*100:.1f}%)")
        print(f"    Val set:   {self.X_val_np.shape[0]} samples (Positives: {np.mean(self.y_val_np)*100:.1f}%)")
        print(f"    Test set:  {self.X_test_np.shape[0]} samples (Positives: {np.mean(self.y_test_np)*100:.1f}%)")
        
        return self.X_train_np, self.y_train_np, self.X_val_np, self.y_val_np, self.X_test_np, self.y_test_np

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Returns PyTorch DataLoaders for train, val, and test partitions."""
        if self.X_train_np is None:
            self.load_and_preprocess()

        train_dataset = TensorDataset(
            torch.from_numpy(self.X_train_np),
            torch.from_numpy(self.y_train_np).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(self.X_val_np),
            torch.from_numpy(self.y_val_np).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(self.X_test_np),
            torch.from_numpy(self.y_test_np).unsqueeze(1)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def _load_adult(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads and cleans Adult Census dataset from OpenML."""
        data = fetch_openml('adult', version=2, as_frame=True)
        df = data.frame.dropna()
        
        # Target column cleaning
        target_col = data.target_names[0] if data.target_names else 'class'
        y = df[target_col].astype(str).str.strip().str.replace('.', '', regex=False)
        X = df.drop(columns=[target_col])
        return X, y

    def _load_spambase(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads Spambase dataset from OpenML."""
        data = fetch_openml(name='spambase', version=1, as_frame=True)
        df = data.frame.dropna()
        target_col = data.target_names[0] if data.target_names else 'class'
        y = df[target_col].astype(float)
        X = df.drop(columns=[target_col])
        return X, y

    def _generate_synthetic(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generates high-dimensional synthetic tabular data with redundant features."""
        n_samples = 10000
        n_features = 50
        n_informative = 20
        n_redundant = 20
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=2,
            weights=[0.6, 0.4],
            flip_y=0.03,
            class_sep=0.8,
            random_state=self.random_state
        )
        
        feature_names = [f"feat_{i:02d}_info" for i in range(n_informative)] + \
                        [f"feat_{i:02d}_redundant" for i in range(n_redundant)] + \
                        [f"feat_{i:02d}_noise" for i in range(n_features - n_informative - n_redundant)]
        return X.astype(np.float32), y.astype(np.float32), feature_names

    def _preprocess_mixed_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Applies StandardScaler to numericals and OneHotEncoder to categoricals."""
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(df)
        
        # Get output feature names
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        all_feature_names = num_cols + encoded_cat_names
        
        # Clean feature names for clear plots
        cleaned_names = [f.replace(' ', '_').replace('<=', 'le_').replace('>=', 'ge_').replace('>', 'gt_').replace('<', 'lt_') for f in all_feature_names]

        self.preprocessor = preprocessor
        return X_processed.astype(np.float32), cleaned_names
