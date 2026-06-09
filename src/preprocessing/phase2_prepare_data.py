from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import load_dataset, save_dataframe

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
]
TARGET_COLUMN = "fail"


@dataclass(frozen=True)
class PreparedDataPaths:
    processed_dir: Path
    cleaned_dataset: Path
    train_dataset: Path
    test_dataset: Path
    preprocessor: Path
    quality_report: Path


def build_preprocessor() -> ColumnTransformer:
    numeric_features = FEATURE_COLUMNS
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    cleaned_df[FEATURE_COLUMNS + [TARGET_COLUMN]] = cleaned_df[FEATURE_COLUMNS + [TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce")
    return cleaned_df


def build_quality_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "metric": [
                "rows_original",
                "rows_cleaned",
                "duplicates_removed",
                "missing_values_original",
                "missing_values_cleaned",
            ],
            "value": [
                len(original_df),
                len(cleaned_df),
                len(original_df) - len(cleaned_df),
                int(original_df.isna().sum().sum()),
                int(cleaned_df.isna().sum().sum()),
            ],
        }
    )
    return report


def get_paths(project_root: Path) -> PreparedDataPaths:
    processed_dir = project_root / "data" / "processed"
    return PreparedDataPaths(
        processed_dir=processed_dir,
        cleaned_dataset=processed_dir / "clean_dataset.csv",
        train_dataset=processed_dir / "train_dataset.csv",
        test_dataset=processed_dir / "test_dataset.csv",
        preprocessor=processed_dir / "preprocessor.joblib",
        quality_report=project_root / "reports" / "phase2" / "data_preparation_report.csv",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "data.csv"
    paths = get_paths(project_root)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.quality_report.parent.mkdir(parents=True, exist_ok=True)

    original_df = load_dataset(raw_data_path)
    cleaned_df = clean_dataset(original_df)

    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=0.2,
        random_state=42,
        stratify=cleaned_df[TARGET_COLUMN],
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(train_df[FEATURE_COLUMNS])

    save_dataframe(cleaned_df, paths.cleaned_dataset)
    save_dataframe(train_df.reset_index(drop=True), paths.train_dataset)
    save_dataframe(test_df.reset_index(drop=True), paths.test_dataset)
    joblib.dump(preprocessor, paths.preprocessor)

    quality_report = build_quality_report(original_df, cleaned_df)
    save_dataframe(quality_report, paths.quality_report)

    print("Phase 2 data preparation completed")
    print(f"Original rows: {len(original_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Duplicates removed: {len(original_df) - len(cleaned_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")


if __name__ == "__main__":
    main()
