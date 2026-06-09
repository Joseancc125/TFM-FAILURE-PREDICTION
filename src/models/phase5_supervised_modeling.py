from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src.utils.io import load_dataset, save_dataframe

TARGET_COLUMN = "fail"
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "machine_health_level",
    "health_numeric",
}


@dataclass(frozen=True)
class SupervisedPaths:
    processed_dir: Path
    model_dir: Path
    report_dir: Path
    metrics_path: Path
    predictions_path: Path
    metadata_path: Path


def get_paths(project_root: Path) -> SupervisedPaths:
    processed_dir = project_root / "data" / "processed"
    model_dir = project_root / "models"
    report_dir = project_root / "reports" / "phase5"
    return SupervisedPaths(
        processed_dir=processed_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        metrics_path=report_dir / "supervised_model_comparison.csv",
        predictions_path=processed_dir / "supervised_test_predictions.csv",
        metadata_path=model_dir / "supervised_models_metadata.json",
    )


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    feature_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for supervised modeling.")
    return feature_columns


def build_models(train_df: pd.DataFrame) -> dict[str, object]:
    positive_count = int(train_df[TARGET_COLUMN].sum())
    negative_count = int((train_df[TARGET_COLUMN] == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
            max_depth=None,
            min_samples_leaf=2,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
            verbosity=-1,
        ),
    }


def fit_predict_oof(model: object, features: pd.DataFrame, target: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(
        model,
        features,
        target,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return oof_proba


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index]), float(f1_values[best_index])


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def build_confusion_values(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def metrics_row(model_name: str, split_name: str, threshold: float, metrics: dict[str, float], confusion: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [model_name],
            "split": [split_name],
            "threshold": [round(threshold, 4)],
            "roc_auc": [round(metrics["roc_auc"], 4)],
            "pr_auc": [round(metrics["pr_auc"], 4)],
            "precision": [round(metrics["precision"], 4)],
            "recall": [round(metrics["recall"], 4)],
            "f1": [round(metrics["f1"], 4)],
            "tn": [confusion["tn"]],
            "fp": [confusion["fp"]],
            "fn": [confusion["fn"]],
            "tp": [confusion["tp"]],
        }
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(paths.processed_dir / "train_features.csv")
    test_df = load_dataset(paths.processed_dir / "test_features.csv")

    feature_columns = resolve_feature_columns(train_df)
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    models = build_models(train_df)
    comparison_frames: list[pd.DataFrame] = []
    predictions_output = test_df[[TARGET_COLUMN]].copy()
    metadata: dict[str, dict[str, object]] = {
        "feature_columns": feature_columns,
        "model_summaries": {},
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...", flush=True)
        oof_probabilities = fit_predict_oof(model, X_train, y_train)
        best_threshold, oof_f1 = find_best_threshold(y_train, oof_probabilities)
        oof_metrics = evaluate_predictions(y_train, oof_probabilities, best_threshold)
        oof_confusion = build_confusion_values(y_train, oof_probabilities, best_threshold)

        model.fit(X_train, y_train)
        train_probabilities = model.predict_proba(X_train)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]

        train_metrics = evaluate_predictions(y_train, train_probabilities, best_threshold)
        test_metrics = evaluate_predictions(y_test, test_probabilities, best_threshold)
        train_confusion = build_confusion_values(y_train, train_probabilities, best_threshold)
        test_confusion = build_confusion_values(y_test, test_probabilities, best_threshold)

        comparison_frames.append(metrics_row(model_name, "oof_train", best_threshold, {**oof_metrics, "f1": oof_f1}, oof_confusion))
        comparison_frames.append(metrics_row(model_name, "test", best_threshold, test_metrics, test_confusion))

        predictions_output[f"{model_name}_probability"] = test_probabilities
        predictions_output[f"{model_name}_prediction"] = (test_probabilities >= best_threshold).astype(int)

        model_path = paths.model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        metadata["model_summaries"][model_name] = {
            "best_threshold": round(best_threshold, 4),
            "oof_f1": round(oof_f1, 4),
            "train_roc_auc": round(train_metrics["roc_auc"], 4),
            "test_roc_auc": round(test_metrics["roc_auc"], 4),
            "model_path": str(model_path),
        }

        print(f"Finished {model_name} with test F1 {round(test_metrics['f1'], 4)}", flush=True)

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    save_dataframe(comparison_df, paths.metrics_path)
    save_dataframe(predictions_output, paths.predictions_path)

    with paths.metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, ensure_ascii=False)

    best_model_row = (
        comparison_df[comparison_df["split"] == "test"]
        .sort_values(["f1", "pr_auc", "recall"], ascending=False)
        .iloc[0]
    )

    print("Phase 5 supervised modeling completed")
    print("Feature columns used:", len(feature_columns))
    print("Best test model:", best_model_row["model"])
    print("Best test F1:", round(float(best_model_row["f1"]), 4))
    print("Best test recall:", round(float(best_model_row["recall"]), 4))


if __name__ == "__main__":
    main()
