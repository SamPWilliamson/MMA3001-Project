"""Train and evaluate baseline and candidate models under chronological split.

This module enforces time-series validation principles (no random shuffling),
fits Baseline KNN, Decision Tree, Gaussian Process, and Random Forest models,
applies physical constraints, evaluates predictive and operational metrics,
and saves predictions and metric summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

from execution.models import (
    BaselineKNN,
    DecisionTreeModel,
    GaussianProcessModel,
    RandomForestModel,
    clamp_physical_bounds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_and_evaluate")

BASELINE_FEATURES = ["co2_ppm", "time_of_day_hours"]
FULL_FEATURES = [
    "co2_ppm",
    "co2_rate_ppm_min",
    "co2_above_ambient",
    "co2_exposure_integral_3h",
    "temp_c",
    "temp_rate_c_min",
    "humidity_rh",
    "humidity_rate_rh_min",
    "time_of_day_hours",
    "day_of_week",
    "is_operating_hours",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
]


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform strict chronological train/test split to prevent temporal data leakage.

    Args:
        df: Input DataFrame sorted chronologically.
        train_ratio: Fraction of earlier contiguous period for training.

    Returns:
        Tuple of (df_train, df_test).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_train = int(len(df) * train_ratio)
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_test = df.iloc[n_train:].copy().reset_index(drop=True)

    logger.info("Chronological split: %d train records (%s to %s), %d test records (%s to %s)",
                len(df_train), df_train["timestamp"].min(), df_train["timestamp"].max(),
                len(df_test), df_test["timestamp"].min(), df_test["timestamp"].max())
    return df_train, df_test


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_operating_hours: np.ndarray,
) -> Dict[str, Any]:
    """Compute regression and operational classification metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Derived binary occupancy (headcount >= 1 vs 0)
    b_true = (y_true >= 0.5).astype(int)
    b_pred = (y_pred >= 0.5).astype(int)

    acc = float(accuracy_score(b_true, b_pred))
    prec = float(precision_score(b_true, b_pred, zero_division=0))
    rec = float(recall_score(b_true, b_pred, zero_division=0))
    f1 = float(f1_score(b_true, b_pred, zero_division=0))

    # Operating hours operational risk (False Negatives during core occupied hours)
    op_mask = is_operating_hours.astype(bool)
    if op_mask.sum() > 0:
        b_true_op = b_true[op_mask]
        b_pred_op = b_pred[op_mask]

        # False negative: room occupied (b_true_op == 1) but predicted empty (b_pred_op == 0)
        occupied_op = (b_true_op == 1).sum()
        fn_op = ((b_true_op == 1) & (b_pred_op == 0)).sum()
        fnr_op = float(fn_op / occupied_op) if occupied_op > 0 else 0.0

        # False positive: room empty (b_true_op == 0) but predicted occupied (b_pred_op == 1)
        unoccupied_op = (b_true_op == 0).sum()
        fp_op = ((b_true_op == 0) & (b_pred_op == 1)).sum()
        fpr_op = float(fp_op / unoccupied_op) if unoccupied_op > 0 else 0.0
    else:
        fnr_op = 0.0
        fpr_op = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "binary_accuracy": acc,
        "binary_precision": prec,
        "binary_recall": rec,
        "binary_f1": f1,
        "operating_hours_false_negative_rate": fnr_op,
        "operating_hours_false_positive_rate": fpr_op,
    }


def train_and_compare(
    features_parquet: str | Path,
    output_preds_parquet: str | Path,
    output_metrics_json: str | Path,
) -> Dict[str, Any]:
    """Train all models, generate test predictions, and calculate metrics."""
    df = pd.read_parquet(features_parquet)
    df_train, df_test = chronological_split(df, train_ratio=0.8)

    y_train = df_train["headcount"].values
    y_test = df_test["headcount"].values
    is_op_test = df_test["is_operating_hours"].values

    # Clean feature matrices
    available_full = [f for f in FULL_FEATURES if f in df.columns]
    available_base = [f for f in BASELINE_FEATURES if f in df.columns]

    X_train_base = df_train[available_base].values
    X_test_base = df_test[available_base].values

    X_train_full = df_train[available_full].values
    X_test_full = df_test[available_full].values

    results: Dict[str, Any] = {}
    preds_df = df_test[["timestamp", "headcount", "is_occupied", "is_operating_hours", "co2_ppm", "temp_c"]].copy()

    # 1. Baseline KNN
    logger.info("Training Baseline KNN...")
    knn = BaselineKNN(n_neighbors=5)
    knn.fit(X_train_base, y_train)
    y_pred_knn = knn.predict(X_test_base)
    preds_df["pred_knn"] = y_pred_knn
    results["Baseline KNN (CO2 + Time)"] = compute_metrics(y_test, y_pred_knn, is_op_test)

    # 2. Decision Tree Regressor
    logger.info("Training Decision Tree Regressor...")
    dt = DecisionTreeModel(max_depth=5, min_samples_leaf=20)
    dt.fit(X_train_full, y_train)
    y_pred_dt = dt.predict(X_test_full)
    preds_df["pred_decision_tree"] = y_pred_dt
    results["Decision Tree (Full Features)"] = compute_metrics(y_test, y_pred_dt, is_op_test)

    # 3. Gaussian Process Regressor
    logger.info("Training Gaussian Process Regressor...")
    gpr = GaussianProcessModel(max_train_samples=2500)
    gpr.fit(X_train_full, y_train)
    y_pred_gpr, y_std_gpr = gpr.predict(X_test_full, return_std=True)
    preds_df["pred_gaussian_process"] = y_pred_gpr
    preds_df["std_gaussian_process"] = y_std_gpr
    results["Gaussian Process (Full Features)"] = compute_metrics(y_test, y_pred_gpr, is_op_test)

    # 4. Random Forest Regressor
    logger.info("Training Random Forest Regressor...")
    rf = RandomForestModel(n_estimators=100, max_depth=10)
    rf.fit(X_train_full, y_train)
    y_pred_rf = rf.predict(X_test_full)
    preds_df["pred_random_forest"] = y_pred_rf
    results["Random Forest (Full Features)"] = compute_metrics(y_test, y_pred_rf, is_op_test)

    # Save outputs
    out_preds = Path(output_preds_parquet)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_parquet(out_preds, index=False)
    logger.info("Saved test predictions to %s", out_preds)

    out_metrics = Path(output_metrics_json)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved comparison metrics to %s", out_metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models on chronological split.")
    parser.add_argument("--features", default=".tmp/g57_features_matrix.parquet", help="Feature parquet path.")
    parser.add_argument("--output-preds", default=".tmp/model_predictions.parquet", help="Predictions parquet path.")
    parser.add_argument("--output-metrics", default=".tmp/model_comparison_metrics.json", help="Metrics JSON path.")

    args = parser.parse_args()
    train_and_compare(args.features, args.output_preds, args.output_metrics)


if __name__ == "__main__":
    main()
