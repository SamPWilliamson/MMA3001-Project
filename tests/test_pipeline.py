"""Integration and regression tests for models and end-to-end data workflow."""

import numpy as np
import pandas as pd
import pytest

from execution.models import (
    BaselineKNN,
    DecisionTreeModel,
    GaussianProcessModel,
    RandomForestModel,
    clamp_physical_bounds,
)
from execution.train_and_evaluate import chronological_split, compute_metrics


def test_clamp_physical_bounds():
    """Verify that predictions are strictly clamped to [0, max_capacity]."""
    preds = np.array([-5.2, -0.01, 0.0, 15.3, 40.0, 65.0, 102.5])
    clamped = clamp_physical_bounds(preds, max_capacity=40.0)

    assert np.all(clamped >= 0.0), "Clamped predictions must be >= 0."
    assert np.all(clamped <= 40.0), "Clamped predictions must be <= max_capacity."
    assert clamped[0] == 0.0
    assert clamped[-1] == 40.0


def test_chronological_split_no_leakage():
    """Verify chronological split ordering and non-overlap."""
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "headcount": np.random.randint(0, 10, 100),
    })

    df_train, df_test = chronological_split(df, train_ratio=0.8)

    assert len(df_train) == 80
    assert len(df_test) == 20
    # No date in test split should precede or equal any date in train split
    assert df_test["timestamp"].min() > df_train["timestamp"].max(), "Data leakage detected: test overlaps train."


def test_baseline_knn_fit_predict():
    """Verify BaselineKNN training, prediction shape, and non-negative bounds."""
    np.random.seed(42)
    X = np.random.uniform(400, 1200, (100, 2))  # CO2 and time
    y = np.random.poisson(lam=3.0, size=100)

    knn = BaselineKNN(n_neighbors=3)
    knn.fit(X, y)
    preds = knn.predict(X[:10])

    assert len(preds) == 10
    assert np.all(preds >= 0.0)


def test_decision_tree_fit_predict():
    """Verify DecisionTreeModel regularized depth and output properties."""
    np.random.seed(42)
    X = np.random.uniform(10, 30, (100, 5))
    y = np.random.uniform(0, 15, 100)

    dt = DecisionTreeModel(max_depth=3, min_samples_leaf=5)
    dt.fit(X, y)
    preds = dt.predict(X)

    assert len(preds) == 100
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 40.0)


def test_gaussian_process_fit_predict_with_uncertainty():
    """Verify GaussianProcessModel provides mean and positive standard deviation."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, (50, 3))
    y = 0.5 * X[:, 0] + np.random.normal(0, 0.2, 50)

    gpr = GaussianProcessModel(max_train_samples=50)
    gpr.fit(X, y)
    mean, std = gpr.predict(X[:5], return_std=True)

    assert len(mean) == 5
    assert len(std) == 5
    assert np.all(std > 0.0), "GPR predictive standard deviation must be positive."
    assert np.all(mean >= 0.0)


def test_compute_metrics_operational_risk():
    """Verify operational risk metric calculation during building operating hours."""
    y_true = np.array([0, 0, 1, 2, 0, 5])
    # Model mistakenly predicts 0 when true headcount is 5 (False Negative)
    y_pred = np.array([0, 0, 1, 2, 0, 0])
    is_operating = np.array([1, 1, 1, 1, 0, 1])

    metrics = compute_metrics(y_true, y_pred, is_operating)

    assert "rmse" in metrics
    assert "operating_hours_false_negative_rate" in metrics
    # In operating hours, true occupied cases: indices 2, 3, 5 (total 3).
    # False negatives: index 5 (1 false negative out of 3 occupied). FNR = 1/3 = ~0.333
    assert np.isclose(metrics["operating_hours_false_negative_rate"], 1.0 / 3.0)
