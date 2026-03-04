import os
from typing import Any, Dict, Optional

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")


def extract_importance(best_pipeline: Pipeline) -> Optional[pd.DataFrame]:
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_)
        if getattr(importances, "ndim", 1) > 1:
            importances = importances.mean(axis=0)
    else:
        return None

    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    cleaned_names = [name.split("__", 1)[-1] for name in feature_names]
    if len(cleaned_names) != len(importances):
        return None

    importance_df = pd.DataFrame({"feature": cleaned_names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).head(12)


def save_performance_plot(
    run_dir: str,
    task_type: str,
    y_true: pd.Series,
    predictions,
    best_model_name: str,
    best_metrics: Dict[str, Any],
) -> None:
    plt.figure(figsize=(8, 6))

    if task_type == "classification":
        labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
        cm = confusion_matrix(y_true.astype(str), pd.Series(predictions).astype(str), labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{best_model_name} Confusion Matrix")
    else:
        y_true_values = pd.Series(y_true).astype(float)
        prediction_values = pd.Series(predictions).astype(float)
        sns.scatterplot(x=y_true_values, y=prediction_values, alpha=0.7)
        low = float(min(y_true_values.min(), prediction_values.min()))
        high = float(max(y_true_values.max(), prediction_values.max()))
        plt.plot([low, high], [low, high], linestyle="--", color="red")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(
            f"{best_model_name} Predicted vs Actual\n"
            f"R2={best_metrics['R2']:.3f} RMSE={best_metrics['RMSE']:.3f}"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot.png"))
    plt.close()


def save_importance_plot(run_dir: str, importance_df: Optional[pd.DataFrame], best_model_name: str) -> None:
    if importance_df is None or importance_df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"{best_model_name} Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "importance.png"))
    plt.close()
