"""
Training Pipeline for F1 Race Predictor
Trains multiple ML classifiers using chronological validation and saves the best model.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.preprocessing import F1Preprocessor, get_chronological_splits, FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import evaluate_predictions, generate_all_figures


def train_and_compare_models(data_path="data/processed/f1_features.csv", models_dir="models", reports_dir="reports"):
    """
    Main training function:
    1. Loads processed features
    2. Splits chronologically (Train: 2010-2021, Val: 2022, Test: 2023-2024)
    3. Fits preprocessor on Train
    4. Trains 5 candidate classifiers
    5. Evaluates on validation and test sets
    6. Identifies best model and saves artifacts
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)

    print("=" * 70)
    print("F1 RACE PREDICTOR -- MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/5] Loading pre-race features from {data_path}...")
    if not os.path.exists(data_path):
        from src.data_loader import load_raw_data, get_base_race_dataframe
        from src.feature_engineering import generate_prerace_features, save_processed_features
        raw_data = load_raw_data("data/raw")
        base_df = get_base_race_dataframe(raw_data)
        features_df, _, _, _, _ = generate_prerace_features(base_df)
        save_processed_features(features_df, data_path)
    else:
        features_df = pd.read_csv(data_path)

    # 2. Chronological Split
    print("\n[2/5] Creating chronological Train / Validation / Test splits...")
    (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df) = get_chronological_splits(
        features_df, train_start=2010, train_end=2021, val_year=2022, test_start=2023
    )

    print(f"  • Training   (2010–2021): {len(X_train)} samples | Podium Rate: {y_train.mean():.2%}")
    print(f"  • Validation (2022):       {len(X_val)} samples | Podium Rate: {y_val.mean():.2%}")
    print(f"  • Testing    (2023–2024):  {len(X_test)} samples | Podium Rate: {y_test.mean():.2%}")

    # 3. Preprocessing
    print("\n[3/5] Fitting preprocessor on training data...")
    preprocessor = F1Preprocessor(FEATURE_COLUMNS)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    preprocessor.save(os.path.join(models_dir, "preprocessing.pkl"))

    # Compute scale_pos_weight for imbalanced podium class (~12.7% positive)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = neg_count / max(1, pos_count)

    # 4. Model Candidates Definition
    print("\n[4/5] Training 5 Classification Algorithms...")
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_split=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=pos_weight * 0.7,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in candidate_models.items():
        print(f"  [+] Training {name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        # Evaluate on Test Set
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_test_prob = model.predict(X_test_scaled)
        
        y_test_pred = (y_test_prob >= 0.35).astype(int)
        metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    
    # Sort by F1 then ROC-AUC
    results_df = results_df.sort_values(by=["F1", "ROC-AUC"], ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS ON TEST SET (2023-2024)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save results table
    results_csv_path = os.path.join(reports_dir, "model_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nModel comparison metrics saved to {results_csv_path}")

    # 5. Select Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_metrics = results_df.iloc[0]

    print(f"\n[BEST MODEL SELECTED]: {best_model_name}")
    print(f"   * F1-Score:  {best_metrics['F1']:.4f}")
    print(f"   * ROC-AUC:   {best_metrics['ROC-AUC']:.4f}")
    print(f"   * Precision: {best_metrics['Precision']:.4f}")
    print(f"   * Recall:    {best_metrics['Recall']:.4f}")
    print(f"   * Accuracy:  {best_metrics['Accuracy']:.4f}")

    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Serialized best model saved to {best_model_path}")

    # Save model metadata
    metadata = {
        "best_model_name": best_model_name,
        "metrics": best_metrics.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(metadata, os.path.join(models_dir, "model_metadata.pkl"))

    # Generate all evaluation charts
    print("\n[5/5] Generating publication-quality visualization charts...")
    generate_all_figures(
        trained_models,
        preprocessor,
        features_df,
        X_test,
        y_test,
        output_dir=os.path.join(reports_dir, "figures"),
    )

    print("\nTraining pipeline completed successfully!")
    return best_model, preprocessor, results_df


if __name__ == "__main__":
    train_and_compare_models()
