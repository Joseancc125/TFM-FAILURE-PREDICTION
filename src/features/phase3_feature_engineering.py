from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.io import load_dataset, save_dataframe

RAW_FEATURE_COLUMNS = [
    "footfall",
    "tempMode",
    "AQ",
    "USS",
    "CS",
    "VOC",
    "RP",
    "IP",
    "Temperature",
]
TARGET_COLUMN = "fail"
RISK_COLUMNS = ["VOC", "AQ", "USS", "Temperature", "footfall"]


@dataclass(frozen=True)
class FeatureEngineeringPaths:
    processed_dir: Path
    train_features: Path
    test_features: Path
    full_features: Path
    risk_scaler: Path
    thresholds: Path
    report: Path


def get_paths(project_root: Path) -> FeatureEngineeringPaths:
    processed_dir = project_root / "data" / "processed"
    return FeatureEngineeringPaths(
        processed_dir=processed_dir,
        train_features=processed_dir / "train_features.csv",
        test_features=processed_dir / "test_features.csv",
        full_features=processed_dir / "feature_engineered_dataset.csv",
        risk_scaler=processed_dir / "risk_scaler.joblib",
        thresholds=processed_dir / "health_thresholds.json",
        report=project_root / "reports" / "phase3" / "feature_engineering_report.csv",
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered_df = df.copy()
    engineered_df["air_risk"] = engineered_df["AQ"] + engineered_df["VOC"]
    engineered_df["thermal_load"] = engineered_df["Temperature"] + engineered_df["tempMode"]
    engineered_df["sensor_gap"] = engineered_df["AQ"] - engineered_df["USS"]
    engineered_df["mechanical_pressure"] = engineered_df["footfall"] / (engineered_df["RP"] + 1)
    engineered_df["air_pressure_ratio"] = engineered_df["VOC"] / (engineered_df["AQ"] + 1)
    engineered_df["control_balance"] = engineered_df["CS"] - engineered_df["IP"]
    engineered_df["operating_intensity"] = (
        engineered_df["footfall"] + engineered_df["RP"] + engineered_df["Temperature"]
    ) / 3
    engineered_df["sensor_stability"] = engineered_df[["AQ", "USS", "CS", "VOC", "IP"]].std(axis=1)
    engineered_df["risk_score"] = 0.0
    engineered_df["machine_health_index"] = 0.0
    engineered_df["machine_health_level"] = ""
    return engineered_df


def fit_risk_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_df[RISK_COLUMNS])
    return scaler


def compute_risk_score(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.Series:
    scaled = pd.DataFrame(
        scaler.transform(df[RISK_COLUMNS]),
        columns=RISK_COLUMNS,
        index=df.index,
    )
    voc_risk = scaled["VOC"]
    aq_risk = scaled["AQ"]
    temperature_risk = scaled["Temperature"]
    uss_risk = 1 - scaled["USS"]
    footfall_risk = 1 - scaled["footfall"]
    risk_score = (
        0.35 * voc_risk
        + 0.25 * aq_risk
        + 0.20 * uss_risk
        + 0.10 * temperature_risk
        + 0.10 * footfall_risk
    )
    return risk_score.clip(0.0, 1.0)


def build_health_thresholds(risk_score: pd.Series) -> dict[str, float]:
    return {
        "level_1": float(risk_score.quantile(0.20)),
        "level_2": float(risk_score.quantile(0.40)),
        "level_3": float(risk_score.quantile(0.60)),
        "level_4": float(risk_score.quantile(0.80)),
    }


def assign_health_level(score: float, thresholds: dict[str, float]) -> str:
    if score <= thresholds["level_1"]:
        return "Level 1 - Healthy"
    if score <= thresholds["level_2"]:
        return "Level 2 - Low Risk"
    if score <= thresholds["level_3"]:
        return "Level 3 - Moderate Risk"
    if score <= thresholds["level_4"]:
        return "Level 4 - High Risk"
    return "Level 5 - Critical"


def finalize_features(df: pd.DataFrame, scaler: MinMaxScaler, thresholds: dict[str, float]) -> pd.DataFrame:
    engineered_df = add_engineered_features(df)
    engineered_df["risk_score"] = compute_risk_score(engineered_df, scaler)
    engineered_df["machine_health_index"] = (1 - engineered_df["risk_score"]).clip(0.0, 1.0)
    engineered_df["machine_health_level"] = engineered_df["risk_score"].apply(
        lambda score: assign_health_level(score, thresholds)
    )
    engineered_df["health_numeric"] = engineered_df["machine_health_level"].map(
        {
            "Level 1 - Healthy": 5,
            "Level 2 - Low Risk": 4,
            "Level 3 - Moderate Risk": 3,
            "Level 4 - High Risk": 2,
            "Level 5 - Critical": 1,
        }
    )
    return engineered_df


def build_report(train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    engineered_columns = [
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
        "health_numeric",
    ]
    report = pd.DataFrame(
        {
            "metric": [
                "train_rows",
                "test_rows",
                "full_rows",
                "engineered_columns_created",
                "train_fail_rate_pct",
                "test_fail_rate_pct",
                "full_fail_rate_pct",
            ],
            "value": [
                len(train_df),
                len(test_df),
                len(full_df),
                len(engineered_columns),
                round(train_df[TARGET_COLUMN].mean() * 100, 2),
                round(test_df[TARGET_COLUMN].mean() * 100, 2),
                round(full_df[TARGET_COLUMN].mean() * 100, 2),
            ],
        }
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = get_paths(project_root)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(paths.processed_dir / "train_dataset.csv")
    test_df = load_dataset(paths.processed_dir / "test_dataset.csv")
    full_df = load_dataset(paths.processed_dir / "clean_dataset.csv")

    scaler = fit_risk_scaler(train_df)
    train_with_risk = add_engineered_features(train_df)
    train_with_risk["risk_score"] = compute_risk_score(train_with_risk, scaler)
    thresholds = build_health_thresholds(train_with_risk["risk_score"])

    train_features = finalize_features(train_df, scaler, thresholds)
    test_features = finalize_features(test_df, scaler, thresholds)
    full_features = finalize_features(full_df, scaler, thresholds)

    save_dataframe(train_features, paths.train_features)
    save_dataframe(test_features, paths.test_features)
    save_dataframe(full_features, paths.full_features)
    joblib.dump(scaler, paths.risk_scaler)
    with paths.thresholds.open("w", encoding="utf-8") as file_handle:
        json.dump(thresholds, file_handle, indent=2, ensure_ascii=False)

    report = build_report(train_features, test_features, full_features)
    save_dataframe(report, paths.report)

    print("Phase 3 feature engineering completed")
    print(f"Train feature rows: {len(train_features)}")
    print(f"Test feature rows: {len(test_features)}")
    print("Health thresholds:")
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
