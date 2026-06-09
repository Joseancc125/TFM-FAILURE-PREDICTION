from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
        "missing_values": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
        "unique_values": [df[column].nunique(dropna=True) for column in df.columns],
    })
    return summary


def build_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    summary = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "ratio_pct": (counts.values / len(df) * 100).round(2),
    })
    return summary


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    return numeric_df.describe().T


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "data.csv"
    reports_dir = project_root / "reports" / "phase1"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(raw_data_path)

    quality_summary = build_quality_summary(df)
    target_summary = build_target_summary(df)
    numeric_summary = build_numeric_summary(df)

    save_dataframe(quality_summary, reports_dir / "data_quality_summary.csv")
    save_dataframe(target_summary, reports_dir / "target_summary.csv")
    save_dataframe(numeric_summary.reset_index().rename(columns={"index": "feature"}), reports_dir / "numeric_summary.csv")

    print("Phase 1 EDA completed")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Target distribution:")
    print(target_summary.to_string(index=False))


if __name__ == "__main__":
    main()
