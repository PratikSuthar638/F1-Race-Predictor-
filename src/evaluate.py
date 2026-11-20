"""
Evaluation Module for F1 Race Predictor
Calculates metrics and generates visual charts for the ML models and F1 analytics.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Apply sleek modern dark/F1 aesthetic styling for figures
plt.style.use("dark_background")
F1_RED = "#e10600"
F1_DARK = "#151820"
F1_ACCENT = "#00d2be"
F1_ORANGE = "#ff8000"
F1_GRAY = "#a0a4b8"


def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Computes a dictionary of standard classification metrics.
    """
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5, 4),
    }


def generate_all_figures(models_dict, preprocessor, features_df, X_test, y_test, output_dir="reports/figures"):
    """
    Generates all 8+ required visual charts and saves them to reports/figures/.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating analytics and model evaluation figures in {output_dir}...")

    # 1. Chart 1: Podiums by Driver (Top 10 Modern/All-time)
    plt.figure(figsize=(10, 5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    driver_podiums = (
        features_df[features_df["podium"] == 1]["driver_full_name"]
        .value_counts()
        .head(10)
    )
    sns.barplot(x=driver_podiums.values, y=driver_podiums.index, color=F1_RED, ax=ax)
    plt.title("Top 10 Formula 1 Drivers by Total Podiums", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.xlabel("Total Podium Finishes", fontsize=11, color=F1_GRAY)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart1_podiums_by_driver.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 2. Chart 2: Podiums by Constructor (Top 10)
    plt.figure(figsize=(10, 5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    team_podiums = (
        features_df[features_df["podium"] == 1]["constructor_name"]
        .value_counts()
        .head(10)
    )
    sns.barplot(x=team_podiums.values, y=team_podiums.index, color=F1_ACCENT, ax=ax)
    plt.title("Top 10 Formula 1 Constructors by Total Podiums", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.xlabel("Total Podium Finishes", fontsize=11, color=F1_GRAY)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart2_podiums_by_constructor.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 3. Chart 3: Grid Position vs Podium Probability
    plt.figure(figsize=(9, 5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    grid_stats = features_df[features_df["grid_position"].between(1, 20)].groupby("grid_position")["podium"].mean().reset_index()
    sns.barplot(data=grid_stats, x="grid_position", y="podium", color="#e10600", ax=ax, alpha=0.85)
    plt.plot(range(len(grid_stats)), grid_stats["podium"], color="#ffffff", marker="o", linewidth=2)
    plt.title("Starting Grid Position vs Historical Podium Probability", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.xlabel("Starting Grid Position", fontsize=11, color=F1_GRAY)
    plt.ylabel("Podium Conversion Rate", fontsize=11, color=F1_GRAY)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart3_grid_vs_podium.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 4. Chart 4: Driver Recent Form vs Actual Finish
    plt.figure(figsize=(9, 5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    sample_df = features_df[features_df["year"] >= 2018].sample(min(1500, len(features_df[features_df["year"] >= 2018])), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="driver_recent_form_score",
        y="positionOrder",
        hue="podium",
        palette={1: F1_RED, 0: F1_GRAY},
        alpha=0.6,
        ax=ax,
    )
    plt.title("Driver Recent Form Score vs Race Finishing Position", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.xlabel("Driver Recent Form Score (0 - 10)", fontsize=11, color=F1_GRAY)
    plt.ylabel("Finishing Position (1st = Top)", fontsize=11, color=F1_GRAY)
    plt.gca().invert_yaxis()
    plt.legend(title="Podium", labels=["No Podium", "Podium"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart4_driver_recent_form.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 5. Chart 5: Constructor Performance vs Podium Probability
    plt.figure(figsize=(9, 5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    features_df["team_perf_bin"] = pd.cut(features_df["constructor_recent_performance_score"], bins=5)
    team_bin_stats = features_df.groupby("team_perf_bin", observed=False)["podium"].mean().reset_index()
    sns.barplot(x=team_bin_stats.index, y=team_bin_stats["podium"], color=F1_ORANGE, ax=ax)
    plt.xticks(ticks=range(len(team_bin_stats)), labels=[str(c) for c in team_bin_stats["team_perf_bin"]], rotation=15)
    plt.title("Constructor Performance Rating vs Podium Rate", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.xlabel("Car Performance Score Range", fontsize=11, color=F1_GRAY)
    plt.ylabel("Podium Rate", fontsize=11, color=F1_GRAY)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart5_constructor_performance.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 6. Chart 6: Model Comparison Bar Chart
    eval_results = []
    X_test_scaled = preprocessor.transform(X_test)
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.predict(X_test_scaled)
        y_pred = (y_prob >= 0.35).astype(int)  # Calibrated threshold for imbalanced F1 podium
        m = evaluate_predictions(y_test, y_pred, y_prob)
        m["Model"] = name
        eval_results.append(m)
    
    results_df = pd.DataFrame(eval_results)
    
    plt.figure(figsize=(11, 5.5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    melted = results_df.melt(id_vars=["Model"], value_vars=["F1", "ROC-AUC", "Accuracy", "Precision", "Recall"], var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="mako", ax=ax)
    plt.title("Machine Learning Models Comparison on F1 Test Set (2023-2024)", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.xlabel("")
    plt.ylabel("Metric Score", fontsize=11, color=F1_GRAY)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart6_model_comparison.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 7. Chart 7: Feature Importance Bar Chart
    best_model_name = results_df.sort_values(by=["F1", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    best_model = models_dict[best_model_name]
    
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])

    if importances is not None:
        feat_imp_df = pd.DataFrame({
            "Feature": preprocessor.feature_columns,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).head(12)

        plt.figure(figsize=(10, 6), facecolor=F1_DARK)
        ax = plt.gca()
        ax.set_facecolor(F1_DARK)
        sns.barplot(data=feat_imp_df, x="Importance", y="Feature", color=F1_RED, ax=ax)
        plt.title(f"Top 12 Engineered Pre-Race Feature Importances ({best_model_name})", fontsize=14, fontweight="bold", color="white", pad=15)
        plt.xlabel("Relative Feature Importance", fontsize=11, color=F1_GRAY)
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "chart7_feature_importance.png"), dpi=200, facecolor=F1_DARK)
        plt.close()

    # 8. Chart 8: Confusion Matrix Heatmap (Best Model)
    best_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(X_test_scaled)
    best_pred = (best_prob >= 0.35).astype(int)
    cm = confusion_matrix(y_test, best_pred)

    plt.figure(figsize=(6.5, 5.5), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["No Podium (0)", "Podium (1)"],
        yticklabels=["No Podium (0)", "Podium (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Test Confusion Matrix - {best_model_name}", fontsize=13, fontweight="bold", color="white", pad=15)
    plt.xlabel("Predicted Class", fontsize=11, color=F1_GRAY)
    plt.ylabel("Actual True Class", fontsize=11, color=F1_GRAY)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart8_confusion_matrix.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    # 9. Chart 9: Multi-Model ROC Curves
    plt.figure(figsize=(8, 6), facecolor=F1_DARK)
    ax = plt.gca()
    ax.set_facecolor(F1_DARK)
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, p)
            auc = roc_auc_score(y_test, p)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", color=F1_GRAY, alpha=0.6)
    plt.title("ROC Curves Comparison Across Models (Test Set 2023-2024)", fontsize=13, fontweight="bold", color="white", pad=15)
    plt.xlabel("False Positive Rate", fontsize=11, color=F1_GRAY)
    plt.ylabel("True Positive Rate", fontsize=11, color=F1_GRAY)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart9_roc_curves.png"), dpi=200, facecolor=F1_DARK)
    plt.close()

    print("All figures successfully generated.")
    return results_df
