
"""Build modeling datasets for the NYC 311 complaint-resolution project.

This script mirrors the stable, repeatable logic from the feature-engineering
notebook:
- load the exploratory-analysis parquet
- drop leakage-prone fields
- select model features
- split train/test with stratification
- fit the preprocessing pipeline
- save processed train/test matrices, targets, and the fitted preprocessor
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

CONTINUOUS_FEATURES = [
    "latitude",
    "longitude",
]

LOW_CARDINALITY_FEATURES = [
    "agency",
    "complaint_type",
    "borough",
    "location_type",
    "complaint_hr",
    "complaint_day",
]

TARGET_COLUMN = "resolution_in_wk"
LEAKAGE_COLUMNS = ["resolution_time_days"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/02_nyc_info_exploratory_analysis.parquet"),
        help="Path to the parquet file produced by the exploratory-analysis stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where processed train/test datasets will be written.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where the fitted preprocessor will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the test split.",
    )
    return parser.parse_args()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                LOW_CARDINALITY_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(args.input_path)

    required_columns = (
        CONTINUOUS_FEATURES
        + LOW_CARDINALITY_FEATURES
        + [TARGET_COLUMN]
        + LEAKAGE_COLUMNS
    )
    validate_columns(data, required_columns)

    # Mirror notebook logic: remove leakage-prone duration column.
    data = data.drop(columns=LEAKAGE_COLUMNS).copy()

    X = data[CONTINUOUS_FEATURES + LOW_CARDINALITY_FEATURES].copy()
    y = data[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    X_train_df.to_parquet(args.output_dir / "03_X_train_processed.parquet")
    X_test_df.to_parquet(args.output_dir / "03_X_test_processed.parquet")
    y_train.to_frame(name=TARGET_COLUMN).to_parquet(args.output_dir / "03_y_train.parquet")
    y_test.to_frame(name=TARGET_COLUMN).to_parquet(args.output_dir / "03_y_test.parquet")

    # Helpful extras for debugging / inference parity.
    X_train.to_parquet(args.output_dir / "03_X_train_raw.parquet")
    X_test.to_parquet(args.output_dir / "03_X_test_raw.parquet")

    joblib.dump(preprocessor, args.models_dir / "preprocessor.joblib")

    print("Saved:")
    print(f"  - {args.output_dir / '03_X_train_processed.parquet'}")
    print(f"  - {args.output_dir / '03_X_test_processed.parquet'}")
    print(f"  - {args.output_dir / '03_y_train.parquet'}")
    print(f"  - {args.output_dir / '03_y_test.parquet'}")
    print(f"  - {args.models_dir / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
