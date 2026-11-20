"""
Preprocessing Module for F1 Race Predictor
Handles feature selection, missing value imputation, scaling, and chronological splitting.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

# The core pre-race feature columns used for ML modeling
FEATURE_COLUMNS = [
    "grid_position",
    "qualifying_position",
    "driver_avg_pos_last_3",
    "driver_avg_pos_last_5",
    "driver_recent_form_score",
    "driver_career_starts",
    "driver_career_podiums",
    "driver_career_podium_rate",
    "driver_career_win_rate",
    "driver_recent_dnf_rate_last_5",
    "driver_qualifying_avg_last_5",
    "driver_points_prior",
    "constructor_avg_pos_last_3",
    "constructor_avg_pos_last_5",
    "constructor_recent_performance_score",
    "constructor_career_podium_rate",
    "constructor_recent_dnf_rate_last_5",
    "constructor_qualifying_avg_last_5",
    "constructor_points_prior",
    "qualifying_diff_team",
    "driver_circuit_avg_finish",
    "driver_circuit_podium_rate",
    "driver_circuit_starts",
    "constructor_circuit_podium_rate",
    "circuit_altitude",
    "circuit_total_historical_races",
]

TARGET_COLUMN = "podium"


class F1Preprocessor:
    """
    Encapsulates preprocessing logic, imputation, and scaling.
    Ensures identical transformations during training and real-time inference.
    """
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        X_df = X[self.feature_columns] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_columns)
        imputed = self.imputer.fit_transform(X_df)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("F1Preprocessor must be fitted before transforming data.")
        
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X_df = X[self.feature_columns] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_columns)
        imputed = self.imputer.transform(X_df)
        scaled = self.scaler.transform(imputed)
        return scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, filepath="models/preprocessing.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath="models/preprocessing.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        return joblib.load(filepath)


def get_chronological_splits(features_df, train_start=2010, train_end=2021, val_year=2022, test_start=2023):
    """
    Splits the dataset strictly chronologically:
    - Train: train_start <= year <= train_end (e.g. 2010-2021)
    - Validation: year == val_year (e.g. 2022)
    - Test: year >= test_start (e.g. 2023-2024)
    """
    train_mask = (features_df["year"] >= train_start) & (features_df["year"] <= train_end)
    val_mask = (features_df["year"] == val_year)
    test_mask = (features_df["year"] >= test_start)

    train_df = features_df[train_mask].copy()
    val_df = features_df[val_mask].copy()
    test_df = features_df[test_mask].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].values

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    return (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df)


if __name__ == "__main__":
    features_df = pd.read_csv("data/processed/f1_features.csv")
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = get_chronological_splits(features_df)
    
    print(f"Train split (2010-2021): {len(X_tr)} samples, {y_tr.mean():.3f} podium rate")
    print(f"Val split (2022):        {len(X_va)} samples, {y_va.mean():.3f} podium rate")
    print(f"Test split (2023-2024):  {len(X_te)} samples, {y_te.mean():.3f} podium rate")
    
    preprocessor = F1Preprocessor()
    X_tr_scaled = preprocessor.fit_transform(X_tr)
    preprocessor.save("models/preprocessing.pkl")
