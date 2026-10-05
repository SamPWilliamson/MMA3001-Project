"""Unit tests for numerical methods library (MMA3001 syllabus verification)."""

import numpy as np
import pytest

from execution.numerical_methods import (
    NaturalCubicSpline,
    OutOfDistributionDetector,
    central_difference_derivative,
    composite_simpsons_rule,
    cumulative_trapezoidal_rule,
)


def test_central_difference_derivative_analytical():
    """Verify central difference derivative against analytical derivative of sin(t)."""
    h = 0.02
    t = np.arange(0.0, 2.0 * np.pi, h)
    y = np.sin(t)
    analytical_dy = np.cos(t)

    calc_dy = central_difference_derivative(t, y)

    # Interior points should have O(h^2) error
    interior_error = np.max(np.abs(calc_dy[1:-1] - analytical_dy[1:-1]))
    assert interior_error < 5e-4, f"Central difference interior error {interior_error} exceeds tolerance."

    # Endpoints have O(h) error
    assert np.abs(calc_dy[0] - analytical_dy[0]) < 2e-2
    assert np.abs(calc_dy[-1] - analytical_dy[-1]) < 2e-2


def test_natural_cubic_spline_knot_interpolation():
    """Verify that natural cubic spline passes exactly through all knot points."""
    x = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
    y = np.array([2.0, 3.5, 1.2, 5.8, 4.0])

    spline = NaturalCubicSpline(x, y)
    y_interp = spline(x)

    np.testing.assert_allclose(y_interp, y, atol=1e-10, err_msg="Spline must interpolate knots exactly.")


def test_natural_cubic_spline_smoothness():
    """Verify that natural cubic spline interpolates intermediate points smoothly."""
    x = np.linspace(0.0, np.pi, 6)
    y = np.sin(x)

    spline = NaturalCubicSpline(x, y)
    x_dense = np.linspace(0.0, np.pi, 50)
    y_dense = spline(x_dense)

    # All values should remain bounded in [-0.1, 1.1]
    assert np.all(y_dense >= -0.1)
    assert np.all(y_dense <= 1.1)


def test_composite_simpsons_rule_cubic_polynomial():
    """Verify Simpson's 1/3 rule exactness on cubic polynomial (degree <= 3).

    Integral of (3x^2 + 2x + 1) from 0 to 2 is [x^3 + x^2 + x]_0^2 = 8 + 4 + 2 = 14.0.
    """
    x = np.linspace(0.0, 2.0, 11)  # 10 intervals (even number)
    dx = x[1] - x[0]
    y = 3.0 * x**2 + 2.0 * x + 1.0

    integral = composite_simpsons_rule(y, dx)
    assert np.isclose(integral, 14.0, atol=1e-10), f"Calculated integral {integral} does not match 14.0"


def test_composite_simpsons_odd_intervals():
    """Verify Simpson's rule handling of odd number of intervals via trapezoid end-cap."""
    x = np.linspace(0.0, 2.0, 10)  # 9 intervals (odd number)
    dx = x[1] - x[0]
    y = 2.0 * x

    # Integral of 2x from 0 to 2 is [x^2]_0^2 = 4.0
    integral = composite_simpsons_rule(y, dx)
    assert np.isclose(integral, 4.0, atol=1e-10)


def test_cumulative_trapezoidal_rule():
    """Verify running cumulative trapezoidal integration."""
    x = np.linspace(0.0, 4.0, 5)  # dx = 1.0
    y = np.ones(5) * 2.0  # Constant f(x) = 2.0 -> integral is 2x

    cum_int = cumulative_trapezoidal_rule(y, dx=1.0)
    expected = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(cum_int, expected, atol=1e-10)


def test_out_of_distribution_detector():
    """Verify that OOD detector approves supported inputs and flags anomalous queries."""
    # Training envelope: CO2 ~ [400, 1200], Temp ~ [18, 26], Hour ~ [0, 24]
    training_data = np.array([
        [420.0, 19.0, 8.0],
        [600.0, 21.0, 12.0],
        [850.0, 23.0, 15.0],
        [1100.0, 25.0, 17.0],
        [450.0, 20.0, 20.0],
    ])
    feature_names = ["CO2", "Temperature", "Hour"]
    detector = OutOfDistributionDetector(training_data, feature_names, std_multiplier=3.0)

    # 1. In-distribution query
    q_normal = np.array([700.0, 22.0, 14.0])
    res_normal = detector.validate_query(q_normal)
    assert res_normal["supported"] is True
    assert len(res_normal["flags"]) == 0

    # 2. Out-of-distribution extreme CO2 query
    q_spike = np.array([2800.0, 22.0, 14.0])
    res_spike = detector.validate_query(q_spike)
    assert res_spike["supported"] is False
    assert any("CO2" in f for f in res_spike["flags"])
