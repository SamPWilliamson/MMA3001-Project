# Directive: Feature Engineering & Syllabus Numerical Methods

## 1. Goal
Transform the aligned Room G.57 time-series data into a feature-rich engineering dataset by applying core syllabus numerical methods:
1. **Spline Interpolation** (Week 3) for smooth sensor signal reconstruction over minor missing intervals.
2. **Finite Difference Differentiation** (Week 7) to compute rates of change $\frac{d[\text{CO}_2]}{dt}$ and $\frac{d[T]}{dt}$, representing human metabolic expiration flux and thermal build-up.
3. **Numerical Integration** (Week 6) using composite Simpson's / Trapezoidal rules to compute cumulative daily exposure ($\int \text{CO}_2 dt$) and ventilation renewal dynamics.
4. **Calendar & Physical Context Features**: time-of-day, day-of-week, working-hour boolean (08:00–18:00), and ambient baseline differentials.

## 2. Inputs
- `.tmp/g57_merged_data.parquet`: Cleaned and synchronized 5-minute time series from Directive 01.

## 3. Tools & Execution Scripts
- `execution/numerical_methods.py`: Pure numerical methods library containing:
  - `central_difference_derivative(x, y)`: $\mathcal{O}(h^2)$ central difference interior scheme with $\mathcal{O}(h)$ forward/backward boundary conditions.
  - `cubic_spline_interpolator(x, y)`: Tridiagonal matrix solver constructing natural cubic splines $S_i(x)$ with continuous first and second derivatives ($C^2$).
  - `composite_simpsons_rule(y, dx)`: Closed 3-point Newton-Cotes composite integration rule with $\mathcal{O}(h^4)$ local error order.
  - `out_of_distribution_detector(query_point, domain_envelope)`: Convex bounding box and Mahalanobis distance check flagging queries unsupported by training domain.
- `execution/feature_engineering.py`: CLI driver that orchestrates the numerical routines across the merged dataset and outputs the feature matrix.

## 4. Outputs & Deliverables
- Intermediate: `.tmp/g57_features_matrix.parquet`.
- Feature metadata: `.tmp/feature_manifest.json` (feature names, units, descriptions, summary statistics).

## 5. Workflow / Step-by-Step Instructions
1. Run `python execution/feature_engineering.py --input ".tmp/g57_merged_data.parquet" --output ".tmp/g57_features_matrix.parquet"`
2. Verify that newly engineered columns exist:
   - `co2_rate_central` ($\text{ppm}/\text{min}$ via 2nd-order central difference)
   - `temp_rate_central` ($^\circ\text{C}/\text{min}$)
   - `co2_integrated_6h` ($\text{ppm}\cdot\text{hr}$ via composite Simpson's rule)
   - `time_of_day_hours` ($[0, 24)$)
   - `is_operating_hours` ($\{0, 1\}$)
3. Check for any infinite or unhandled NaN values in `.tmp/g57_features_matrix.parquet`.

## 6. Edge Cases & Learnings
- Differentiation error amplification: Finite differences are notoriously sensitive to high-frequency measurement noise ($\frac{\Delta \epsilon}{h}$). Pre-smooth raw $\text{CO}_2$ with a small Gaussian / Savitzky-Golay window or cubic smoothing spline prior to differentiation to maintain numerical stability.
- Simpson's rule interval requirement: Composite Simpson's requires an even number of intervals (odd number of nodes). If an odd number of intervals is present, combine composite Simpson's rule with a final Trapezoidal step.
