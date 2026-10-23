"""Numerical methods library implemented from first principles.

This module provides deterministic implementations of numerical differentiation,
spline interpolation, numerical integration, and out-of-distribution detection,
aligned with MMA3001 syllabus concepts (Weeks 3, 6, and 7).

Modules:
    - central_difference_derivative: 2nd-order finite difference numerical differentiation.
    - NaturalCubicSpline: Natural cubic spline interpolator with tridiagonal solver.
    - composite_simpsons_rule: Composite Simpson's 1/3 integration with error order O(h^4).
    - cumulative_trapezoidal_rule: Running cumulative numerical integration.
    - OutOfDistributionDetector: Domain envelope validator for query support.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


def central_difference_derivative(
    x: np.ndarray,
    y: np.ndarray,
    smooth_window: Optional[int] = None,
) -> np.ndarray:
    r"""Compute first derivative using 2nd-order central differences with endpoint handling.

    Interior points:
        $$f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{x_{i+1} - x_{i-1}} + \mathcal{O}(h^2)$$

    Endpoints (2nd-order single-sided forward / backward differences):
        $$f'(x_0) \approx \frac{-3f(x_0) + 4f(x_1) - f(x_2)}{2h} + \mathcal{O}(h^2)$$
        $$f'(x_{n-1}) \approx \frac{3f(x_{n-1}) - 4f(x_{n-2}) + f(x_{n-3})}{2h} + \mathcal{O}(h^2)$$

    Args:
        x: Strictly monotonic 1D array of coordinates (e.g. time in minutes).
        y: 1D array of sampled signal values (e.g. CO2 in ppm).
        smooth_window: Optional odd integer window size for pre-smoothing moving average.
            Prevents numerical noise amplification where error scales as O(epsilon / h).

    Returns:
        1D array of derivatives dydx with same length as y.

    Raises:
        ValueError: If x and y have mismatched lengths or fewer than 2 elements.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError(f"Lengths of x ({len(x)}) and y ({len(y)}) must match.")
    if len(x) < 2:
        raise ValueError("At least 2 points required for numerical differentiation.")

    # Optional pre-smoothing to mitigate O(epsilon / h) error explosion on noisy sensor data
    if smooth_window is not None and smooth_window >= 3:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        y_padded = np.pad(y, pad, mode="edge")
        weights = np.ones(w) / w
        y_calc = np.convolve(y_padded, weights, mode="valid")
    else:
        y_calc = y.copy()

    n = len(x)
    dydx = np.zeros(n, dtype=np.float64)

    # Check if grid is uniform
    dx_diffs = np.diff(x)
    is_uniform = np.allclose(dx_diffs, dx_diffs[0], rtol=1e-4)

    if is_uniform and n >= 3:
        h = dx_diffs[0]
        # Interior points (2nd-order central differences)
        dydx[1:-1] = (y_calc[2:] - y_calc[:-2]) / (2.0 * h)
        # Boundary points (2nd-order 3-point single-sided forward and backward differences)
        dydx[0] = (-3.0 * y_calc[0] + 4.0 * y_calc[1] - y_calc[2]) / (2.0 * h)
        dydx[-1] = (3.0 * y_calc[-1] - 4.0 * y_calc[-2] + y_calc[-3]) / (2.0 * h)
    else:
        # Non-uniform grid formulation
        for i in range(1, n - 1):
            h_f = x[i + 1] - x[i]
            h_b = x[i] - x[i - 1]
            # General 3-point Lagrange finite difference derivative
            dydx[i] = (
                -h_f / (h_b * (h_b + h_f)) * y_calc[i - 1]
                + (h_f - h_b) / (h_b * h_f) * y_calc[i]
                + h_b / (h_f * (h_b + h_f)) * y_calc[i + 1]
            )
        # Endpoints
        dydx[0] = (y_calc[1] - y_calc[0]) / (x[1] - x[0])
        dydx[-1] = (y_calc[-1] - y_calc[-2]) / (x[-1] - x[-2])

    return dydx


class NaturalCubicSpline:
    r"""Piecewise natural cubic spline interpolator with tridiagonal system solver.

    Constructs cubic polynomials $S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$
    satisfying continuity of function, first, and second derivatives ($C^2$),
    with natural boundary conditions $S''_0(x_0) = 0$ and $S''_{n-1}(x_{n-1}) = 0$.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Fit natural cubic spline to knots (x, y).

        Args:
            x: Monotonically increasing knot positions.
            y: Function values at knots.
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

        if len(self.x) < 3:
            raise ValueError("Natural cubic spline requires at least 3 knot points.")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("Knot array x must be strictly monotonically increasing.")

        self._build_spline()

    def _build_spline(self) -> None:
        """Solve tridiagonal linear system for second derivative coefficients."""
        n = len(self.x)
        h = np.diff(self.x)

        # Tridiagonal system matrix A and RHS vector alpha
        # A * c = alpha
        diag = np.zeros(n)
        upper = np.zeros(n - 1)
        lower = np.zeros(n - 1)
        alpha = np.zeros(n)

        # Natural boundary condition at x_0: S''(x_0) = 0 -> c_0 = 0
        diag[0] = 1.0

        for i in range(1, n - 1):
            lower[i - 1] = h[i - 1]
            diag[i] = 2.0 * (h[i - 1] + h[i])
            upper[i] = h[i]
            alpha[i] = 3.0 * ((self.y[i + 1] - self.y[i]) / h[i] - (self.y[i] - self.y[i - 1]) / h[i - 1])

        # Natural boundary condition at x_{n-1}: S''(x_{n-1}) = 0 -> c_{n-1} = 0
        diag[-1] = 1.0

        # Thomas algorithm for tridiagonal system
        c = self._thomas_solver(lower, diag, upper, alpha)

        # Compute polynomial coefficients for each interval [x_i, x_{i+1}]
        self.a = self.y[:-1]
        self.c = c[:-1]
        self.b = np.zeros(n - 1)
        self.d = np.zeros(n - 1)

        for i in range(n - 1):
            self.b[i] = (self.y[i + 1] - self.y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
            self.d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    @staticmethod
    def _thomas_solver(
        lower: np.ndarray,
        diag: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Execute Thomas algorithm to solve tridiagonal linear system."""
        n = len(diag)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = rhs[0] / diag[0]

        for i in range(1, n - 1):
            m = diag[i] - lower[i - 1] * c_prime[i - 1]
            c_prime[i] = upper[i] / m
            d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / m

        d_prime[-1] = (rhs[-1] - lower[-1] * d_prime[-2]) / (diag[-1] - lower[-1] * c_prime[-2])

        sol = np.zeros(n)
        sol[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            sol[i] = d_prime[i] - c_prime[i] * sol[i + 1]

        return sol

    def __call__(self, x_query: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Interpolate function values at query positions."""
        is_scalar = np.isscalar(x_query)
        xq = np.atleast_1d(x_query).astype(np.float64)
        yq = np.zeros_like(xq)

        # Find interval for each query point using searchsorted
        idx = np.searchsorted(self.x, xq) - 1
        idx = np.clip(idx, 0, len(self.a) - 1)

        dx = xq - self.x[idx]
        yq = self.a[idx] + self.b[idx] * dx + self.c[idx] * (dx**2) + self.d[idx] * (dx**3)

        return float(yq[0]) if is_scalar else yq


def composite_simpsons_rule(y: np.ndarray, dx: float) -> float:
    r"""Compute definite integral using composite Simpson's 1/3 rule.

    $$\int_a^b f(x) dx \approx \frac{h}{3} \left[ f_0 + 4 \sum_{i=1,3,\dots}^{n-1} f_i + 2 \sum_{i=2,4,\dots}^{n-2} f_i + f_n \right]$$

    Args:
        y: 1D array of function values at uniform intervals.
        dx: Uniform node spacing h.

    Returns:
        Approximated definite integral.

    Raises:
        ValueError: If y has fewer than 3 elements.
    """
    y = np.asarray(y, dtype=np.float64)
    n_intervals = len(y) - 1

    if n_intervals < 2:
        raise ValueError("Simpson's rule requires at least 3 nodes (2 intervals).")

    # If even number of intervals, standard composite Simpson's
    if n_intervals % 2 == 0:
        integral = (dx / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])
    else:
        # Odd number of intervals: apply Simpson's rule on first n-1 intervals, trapezoid on last
        simp_part = (dx / 3.0) * (y[0] + 4.0 * np.sum(y[1:-2:2]) + 2.0 * np.sum(y[2:-2:2]) + y[-2])
        trap_part = 0.5 * dx * (y[-2] + y[-1])
        integral = simp_part + trap_part

    return float(integral)


def cumulative_trapezoidal_rule(y: np.ndarray, dx: float) -> np.ndarray:
    """Compute cumulative numerical integral using the composite trapezoidal rule.

    Args:
        y: 1D array of sampled values.
        dx: Sampling step size.

    Returns:
        1D array of cumulative integral values with initial value 0.
    """
    y = np.asarray(y, dtype=np.float64)
    res = np.zeros_like(y)
    res[1:] = np.cumsum(0.5 * dx * (y[:-1] + y[1:]))
    return res


class OutOfDistributionDetector:
    """Domain envelope detector to identify out-of-distribution unsupported queries.

    Ensures that queries outside the calibrated operating regime of the sensor
    distribution return explicit rejection flags rather than silent extrapolation.
    """

    def __init__(self, training_data: np.ndarray, feature_names: List[str], std_multiplier: float = 3.5):
        """Fit feature envelope bounds on training dataset.

        Args:
            training_data: 2D array of shape (n_samples, n_features).
            feature_names: Names corresponding to columns.
            std_multiplier: Bounds envelope width in standard deviations.
        """
        self.training_data = np.asarray(training_data, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.std_multiplier = std_multiplier

        # Calculate bounding envelope
        self.min_bounds = np.min(self.training_data, axis=0)
        self.max_bounds = np.max(self.training_data, axis=0)
        self.means = np.mean(self.training_data, axis=0)
        self.stds = np.std(self.training_data, axis=0) + 1e-8

    def validate_query(self, query: np.ndarray) -> Dict[str, Any]:
        """Validate whether a query vector is supported by the training domain.

        Args:
            query: 1D or 2D query feature vector.

        Returns:
            Dictionary with 'supported' boolean, 'confidence' metric, and 'flags' list.
        """
        q = np.asarray(query, dtype=np.float64).flatten()
        flags = []

        # 1. Physical hard bounds check
        for i, name in enumerate(self.feature_names):
            if q[i] < self.min_bounds[i] or q[i] > self.max_bounds[i]:
                flags.append(f"{name} ({q[i]:.2f}) outside training range [{self.min_bounds[i]:.2f}, {self.max_bounds[i]:.2f}]")

        # 2. Mahalanobis / Z-score envelope check
        z_scores = np.abs((q - self.means) / self.stds)
        max_z = float(np.max(z_scores))
        if max_z > self.std_multiplier:
            flags.append(f"Z-score outlier: max z-score {max_z:.2f} exceeds threshold {self.std_multiplier}")

        is_supported = len(flags) == 0
        confidence = float(np.clip(1.0 - (max_z / (2.0 * self.std_multiplier)), 0.0, 1.0))

        return {
            "supported": is_supported,
            "confidence": confidence,
            "flags": flags,
        }
