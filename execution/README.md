# Numerical Methods & Machine Learning Execution Modules

Core Python engineering modules implementing first-principles numerical algorithms, streaming data ingestion, physics-informed feature derivation, candidate regression architectures, and hardware-efficiency profiling.

## Module Structure

1. **`numerical_methods.py`**
   - Natural cubic spline interpolation for variable-interval alignment ($\mathcal{O}(h^4)$ error).
   - Higher-order finite difference stencils (4th-order central difference $\mathcal{O}(h^4)$, backward $\mathcal{O}(h^2)$, forward $\mathcal{O}(h^2)$) with dynamic step sizes ($h_i \neq h_{i+1}$).
   - Composite numerical quadrature (Simpson's 1/3 and 3/8 rules) for cumulative energy and air exchange integrals.
   - Out-of-Distribution (OOD) envelope detection via Mahalanobis distance with physical thresholding ($\chi^2_{p, 0.999}$).

2. **`data_pipeline.py`**
   - Chunked, out-of-core streaming parser for campus telemetry (180k environmental records and 9.1M motion counter transitions).
   - Strict temporal boundaries, physical sanity clipping, and missing interval imputation.

3. **`feature_engineering.py`**
   - Physics-informed derivative extraction ($dCO_2/dt$, $dT/dt$, $dRH/dt$) capturing occupancy respiration dynamics.
   - Temporal windowing, rolling variance, and cyclical solar/diurnal encodings.

4. **`models.py`**
   - Candidate regression estimators: K-Nearest Neighbours (KNN), Decision Tree, Gaussian Process Regressor (with analytical uncertainty $\pm 2\sigma$), and Random Forest.

5. **`train_and_evaluate.py`**
   - Strict chronological 80/20 train/test evaluation (zero forward-looking data leakage).
   - Tracking of RMSE, MAE, $R^2$, and operational classification metrics (Empty vs Occupied).

6. **`evaluation_and_profiling.py`**
   - High-resolution CPU latency benchmarking (microsecond scale), peak resident memory (tracemalloc), and theoretical FLOP estimation.

7. **`visualizations.py`**
   - Publication-quality analytical figures matching academic IEEE/Elsevier formatting.
