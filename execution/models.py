"""Model definition and training module for Room G.57 occupancy prediction.

Implements baseline, interpretable, probabilistic, and ensemble models:
    - BaselineKNN: 2-feature baseline (raw CO2 and time-of-day).
    - DecisionTreeModel: Pruned CART decision tree (Week 5).
    - GaussianProcessModel: Gaussian Process Regressor with RBF kernel and uncertainty estimation.
    - RandomForestModel: Random Forest ensemble regressor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("models")

MAX_ROOM_CAPACITY = 40.0  # Monash Room G.57 maximum capacity constraint


def clamp_physical_bounds(y_pred: np.ndarray, max_capacity: float = MAX_ROOM_CAPACITY) -> np.ndarray:
    """Apply physical engineering constraints to predicted occupancy headcount.

    Headcount cannot be negative (floor at 0.0) and cannot exceed the room's
    fire/building code physical maximum capacity.

    Args:
        y_pred: 1D array of continuous headcount predictions.
        max_capacity: Room capacity ceiling.

    Returns:
        Clamped array satisfying 0.0 <= y_pred <= max_capacity.
    """
    return np.clip(y_pred, 0.0, max_capacity)


class BaselineKNN(BaseEstimator, RegressorMixin):
    """Simple 2-feature baseline model using K-Nearest Neighbors.

    Uses only raw CO2 and time-of-day to provide a natural benchmark
    as scoped in Week 3 project component notes.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance")

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineKNN:
        """Fit baseline on CO2 and time-of-day features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict headcount clamped to physical bounds."""
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return clamp_physical_bounds(preds)


class DecisionTreeModel(BaseEstimator, RegressorMixin):
    """Interpretable CART Decision Tree regressor with pruning regularization.

    Limits max_depth to prevent overfitting on measurement noise (Week 5).
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return clamp_physical_bounds(preds)


class GaussianProcessModel(BaseEstimator, RegressorMixin):
    """Gaussian Process Regressor with closed-form uncertainty quantification.

    Delivers both mean headcount expectation mu(x) and epistemic uncertainty sigma(x).
    Uses RBF + WhiteKernel noise estimation.
    """

    def __init__(self, max_train_samples: int = 2500):
        self.max_train_samples = max_train_samples
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-2, 1e1))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            n_restarts_optimizer=2,
            random_state=42,
            normalize_y=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessModel:
        # Standardize features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Subsample if dataset exceeds scaling limits (O(N^3) complexity)
        if len(X_scaled) > self.max_train_samples:
            idx = np.linspace(0, len(X_scaled) - 1, self.max_train_samples, dtype=int)
            X_fit = X_scaled[idx]
            y_fit = y_scaled[idx]
        else:
            X_fit = X_scaled
            y_fit = y_scaled

        logger.info("Fitting Gaussian Process Regressor on %d points...", len(X_fit))
        self.model.fit(X_fit, y_fit)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_scaled = self.scaler_X.transform(X)
        if return_std:
            y_scaled_mean, y_scaled_std = self.model.predict(X_scaled, return_std=True)
            y_mean = self.scaler_y.inverse_transform(y_scaled_mean.reshape(-1, 1)).flatten()
            y_std = y_scaled_std * self.scaler_y.scale_[0]
            return clamp_physical_bounds(y_mean), y_std
        else:
            y_scaled_mean = self.model.predict(X_scaled, return_std=False)
            y_mean = self.scaler_y.inverse_transform(y_scaled_mean.reshape(-1, 1)).flatten()
            return clamp_physical_bounds(y_mean)


class RandomForestModel(BaseEstimator, RegressorMixin):
    """Ensemble Random Forest regressor for high nonlinear accuracy."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return clamp_physical_bounds(preds)
