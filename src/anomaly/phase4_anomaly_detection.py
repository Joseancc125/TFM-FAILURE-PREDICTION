from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import load_dataset, save_dataframe

TARGET_COLUMN = "fail"
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "machine_health_level",
    "health_numeric",
}
FEATURE_COLUMNS = [
    "footfall",
    "tempMode",
    "AQ",
    "USS",
    "CS",
    "VOC",
    "RP",
    "IP",
    "Temperature",
    "air_risk",
    "thermal_load",
    "sensor_gap",
    "mechanical_pressure",
    "air_pressure_ratio",
    "control_balance",
    "operating_intensity",
    "sensor_stability",
    "risk_score",
    "machine_health_index",
]


@dataclass(frozen=True)
class AnomalyPaths:
    processed_dir: Path
    train_output: Path
    test_output: Path
    isolation_model: Path
    lof_model: Path
    report: Path


def get_paths(project_root: Path) -> AnomalyPaths:
    processed_dir = project_root / "data" / "processed"
    return AnomalyPaths(
        processed_dir=processed_dir,
        train_output=processed_dir / "train_anomaly_scores.csv",
        test_output=processed_dir / "test_anomaly_scores.csv",
        isolation_model=processed_dir / "isolation_forest_model.joblib",
        lof_model=processed_dir / "lof_model.joblib",
        report=project_root / "reports" / "phase4" / "anomaly_detection_report.csv",
    )


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    available_columns = [column for column in FEATURE_COLUMNS if column in df.columns and column not in EXCLUDED_COLUMNS]
    if not available_columns:
        raise ValueError("No usable feature columns were found for anomaly detection.")
    return available_columns


def build_isolation_forest(contamination: float) -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )


def safe_n_neighbors(train_size: int) -> int:
    return max(10, min(35, train_size // 20 or 10))


def score_predictions(y_true: pd.Series, anomaly_flag: pd.Series, anomaly_score: pd.Series) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, anomaly_score),
        "pr_auc": average_precision_score(y_true, anomaly_score),
        "precision": precision_score(y_true, anomaly_flag, zero_division=0),
        "recall": recall_score(y_true, anomaly_flag, zero_division=0),
        "f1": f1_score(y_true, anomaly_flag, zero_division=0),
    }


def summarize_results(model_name: str, split_name: str, metrics: dict[str, float], contamination: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [model_name],
            "split": [split_name],
            "contamination": [round(contamination, 4)],
            "roc_auc": [round(metrics["roc_auc"], 4)],
            "pr_auc": [round(metrics["pr_auc"], 4)],
            "precision": [round(metrics["precision"], 4)],
            "recall": [round(metrics["recall"], 4)],
            "f1": [round(metrics["f1"], 4)],
        }
    )


def append_scores(df: pd.DataFrame, model_name: str, raw_scores: pd.Series, anomaly_flag: pd.Series) -> pd.DataFrame:
    scored_df = df.copy()
    scored_df[f"{model_name}_anomaly_score"] = pd.Series(raw_scores, index=df.index)
    scored_df[f"{model_name}_anomaly_flag"] = pd.Series(anomaly_flag, index=df.index).astype(int)
    return scored_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(paths.processed_dir / "train_features.csv")
    test_df = load_dataset(paths.processed_dir / "test_features.csv")

    feature_columns = resolve_feature_columns(train_df)
    contamination = 0.15
    lof_neighbors = safe_n_neighbors(len(train_df))

    isolation_model = build_isolation_forest(contamination=contamination)
    isolation_model.fit(train_df[feature_columns])

    lof_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lof",
                LocalOutlierFactor(
                    n_neighbors=lof_neighbors,
                    contamination=contamination,
                    novelty=True,
                ),
            ),
        ]
    )
    lof_model.fit(train_df[feature_columns])

    isolation_train_raw = -isolation_model.score_samples(train_df[feature_columns])
    isolation_test_raw = -isolation_model.score_samples(test_df[feature_columns])
    isolation_train_flag = isolation_model.predict(train_df[feature_columns]) == -1
    isolation_test_flag = isolation_model.predict(test_df[feature_columns]) == -1

    lof_train_raw = -lof_model.score_samples(train_df[feature_columns])
    lof_test_raw = -lof_model.score_samples(test_df[feature_columns])
    lof_train_flag = lof_model.predict(train_df[feature_columns]) == -1
    lof_test_flag = lof_model.predict(test_df[feature_columns]) == -1

    train_scored = append_scores(train_df, "isolation_forest", isolation_train_raw, pd.Series(isolation_train_flag, index=train_df.index))
    train_scored = append_scores(train_scored, "lof", lof_train_raw, pd.Series(lof_train_flag, index=train_df.index))
    test_scored = append_scores(test_df, "isolation_forest", isolation_test_raw, pd.Series(isolation_test_flag, index=test_df.index))
    test_scored = append_scores(test_scored, "lof", lof_test_raw, pd.Series(lof_test_flag, index=test_df.index))

    save_dataframe(train_scored, paths.train_output)
    save_dataframe(test_scored, paths.test_output)
    joblib.dump(isolation_model, paths.isolation_model)
    joblib.dump(lof_model, paths.lof_model)

    isolation_train_metrics = score_predictions(train_df[TARGET_COLUMN], train_scored["isolation_forest_anomaly_flag"], train_scored["isolation_forest_anomaly_score"])
    isolation_test_metrics = score_predictions(test_df[TARGET_COLUMN], test_scored["isolation_forest_anomaly_flag"], test_scored["isolation_forest_anomaly_score"])
    lof_train_metrics = score_predictions(train_df[TARGET_COLUMN], train_scored["lof_anomaly_flag"], train_scored["lof_anomaly_score"])
    lof_test_metrics = score_predictions(test_df[TARGET_COLUMN], test_scored["lof_anomaly_flag"], test_scored["lof_anomaly_score"])

    report = pd.concat(
        [
            summarize_results("isolation_forest", "train", isolation_train_metrics, contamination),
            summarize_results("isolation_forest", "test", isolation_test_metrics, contamination),
            summarize_results("lof", "train", lof_train_metrics, contamination),
            summarize_results("lof", "test", lof_test_metrics, contamination),
        ],
        ignore_index=True,
    )
    save_dataframe(report, paths.report)

    print("Phase 4 anomaly detection completed")
    print(f"Feature columns used: {len(feature_columns)}")
    print("Contamination used:", round(contamination, 4))
    print("Isolation Forest test F1:", round(isolation_test_metrics["f1"], 4))
    print("LOF test F1:", round(lof_test_metrics["f1"], 4))


if __name__ == "__main__":
    main()
