from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.io import load_dataset, save_dataframe

TARGET_COLUMN = "fail"
EXCLUDED_COLUMNS = {TARGET_COLUMN, "machine_health_level", "health_numeric"}
SAMPLE_SIZE = 200


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    columns = [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]
    if not columns:
        raise ValueError("No numeric columns available for interpretability.")
    return columns


def load_best_model(project_root: Path):
    metadata_path = project_root / "models" / "supervised_models_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)
    comparison = pd.read_csv(project_root / "reports" / "phase5" / "supervised_model_comparison.csv")
    best_row = comparison[comparison["split"] == "test"].sort_values(["f1", "pr_auc", "recall"], ascending=False).iloc[0]
    best_model_name = str(best_row["model"])
    model_info = metadata["model_summaries"][best_model_name]
    model = joblib.load(model_info["model_path"])
    return model, metadata, model_info, best_model_name, float(best_row["f1"])


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[-1]
    if hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            return values[:, :, -1]
        return values
    return shap_values


def save_bar_plot(series: pd.Series, title: str, output_path: Path, color: str) -> None:
    plt.figure(figsize=(10, 6))
    series.sort_values().plot(kind="barh", color=color)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    report_dir = project_root / "reports" / "phase6"
    figure_dir = project_root / "reports" / "figures" / "phase6"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model, metadata, model_info, best_model_name, best_test_f1 = load_best_model(project_root)
    feature_columns = metadata["feature_columns"]
    train_df = load_dataset(project_root / "data" / "processed" / "train_features.csv")
    test_df = load_dataset(project_root / "data" / "processed" / "test_features.csv")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Selected model does not expose feature_importances_.")

    impurity_importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    save_dataframe(
        impurity_importance.reset_index().rename(columns={"index": "feature", 0: "importance"}),
        report_dir / "feature_importance.csv",
    )
    save_bar_plot(
        impurity_importance,
        f"{best_model_name} feature importance",
        figure_dir / "feature_importance.png",
        color="#1f77b4",
    )

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(sample)
    shap_values = normalize_shap_values(shap_values_raw)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_columns).sort_values(ascending=False)
    save_dataframe(
        shap_importance.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}),
        report_dir / "shap_importance.csv",
    )
    save_bar_plot(
        shap_importance,
        f"{best_model_name} mean(|SHAP|)",
        figure_dir / "shap_importance.png",
        color="#ff7f0e",
    )

    shap_summary = pd.DataFrame(
        {
            "metric": [
                "best_model",
                "best_test_roc_auc",
                "best_test_f1",
                "sample_rows_used",
                "feature_count",
            ],
            "value": [
                best_model_name,
                round(float(model_info["test_roc_auc"]), 4),
                round(float(best_test_f1), 4),
                len(sample),
                len(feature_columns),
            ],
        }
    )
    save_dataframe(shap_summary, report_dir / "interpretability_summary.csv")

    print("Phase 6 interpretability completed")
    print("Best model:", best_model_name)
    print("Top SHAP feature:", shap_importance.index[0])
    print("Top importance feature:", impurity_importance.index[0])


if __name__ == "__main__":
    main()
