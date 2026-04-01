
"""Train and compare models for the NYC 311 complaint-resolution project.

This script mirrors the stable, repeatable logic from the modeling notebook:
- load processed train/test data
- fit Logistic Regression, Random Forest, and XGBoost
- evaluate each model with ROC-AUC, PR-AUC, precision, recall, and F1
- save a comparison table and the selected best model

By default, the script selects the best model by PR-AUC. If you want to mirror
the notebook exactly, pass --preferred-model random_forest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

TARGET_COLUMN = "resolution_in_wk"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing processed train/test parquet files.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where model artifacts and metrics will be saved.",
    )
    parser.add_argument(
        "--preferred-model",
        choices=["auto", "logistic_regression", "random_forest", "xgboost"],
        default="auto",
        help="Model to save as the primary artifact. 'auto' selects best PR-AUC.",
    )
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_parquet(data_dir / "03_X_train_processed.parquet")
    X_test = pd.read_parquet(data_dir / "03_X_test_processed.parquet")
    y_train = pd.read_parquet(data_dir / "03_y_train.parquet")[TARGET_COLUMN]
    y_test = pd.read_parquet(data_dir / "03_y_test.parquet")[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def build_models(y_train: pd.Series) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    positive = report["1"]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "precision": float(positive["precision"]),
        "recall": float(positive["recall"]),
        "f1_score": float(positive["f1-score"]),
    }


def choose_best_model(metrics_df: pd.DataFrame, preferred_model: str) -> str:
    if preferred_model != "auto":
        if preferred_model not in metrics_df["model_name"].values:
            raise ValueError(f"Requested model '{preferred_model}' is unavailable.")
        return preferred_model

    return (
        metrics_df.sort_values(
            by=["pr_auc", "roc_auc", "f1_score"],
            ascending=False,
        )
        .iloc[0]["model_name"]
    )


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    models = build_models(y_train)

    fitted_models: dict[str, object] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        metrics = evaluate_model(model, X_test, y_test)
        metrics_rows.append({"model_name": model_name, **metrics})

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="pr_auc", ascending=False)
    metrics_df.to_csv(args.models_dir / "model_comparison.csv", index=False)

    selected_model_name = choose_best_model(metrics_df, args.preferred_model)
    selected_model = fitted_models[selected_model_name]

    joblib.dump(selected_model, args.models_dir / f"{selected_model_name}.joblib")
    joblib.dump(selected_model, args.models_dir / "selected_model.joblib")

    metadata = {
        "selected_model_name": selected_model_name,
        "selection_mode": args.preferred_model,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    (args.models_dir / "selected_model_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("Model comparison:")
    print(metrics_df.round(4).to_string(index=False))
    print()
    print(f"Saved selected model: {args.models_dir / 'selected_model.joblib'}")
    print(f"Selected model name: {selected_model_name}")


if __name__ == "__main__":
    main()
