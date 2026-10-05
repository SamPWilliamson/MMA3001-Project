# Directive: Validation, Engineering Credibility & Code Profiling

## 1. Goal
Execute thorough, defensible engineering validation and profiling on the occupancy models and numerical methods:
1. Physical feasibility verification (headcount non-negativity and room capacity bound).
2. Out-of-Distribution (OOD) test queries (flagging unsupported inputs when sensor readings fall outside training envelope).
3. Operational failure mode analysis: False Negatives during building operating hours (predicting 0 when room is occupied -> HVAC hazard).
4. Profiling execution latency, memory footprint, and theoretical computational complexity (FLOPS and arithmetic intensity per query).

## 2. Inputs
- `.tmp/model_predictions.parquet`: Model predictions from Directive 03.
- `.tmp/g57_features_matrix.parquet`: Feature dataset.

## 3. Tools & Execution Scripts
- `execution/evaluation_and_profiling.py`:
  - `evaluate_operational_risk(y_true, y_pred, is_operating_hours)`: Detailed confusion matrix, False Negative Rate (FNR) and False Positive Rate (FPR) during core business hours.
  - `profile_inference_performance(models, sample_queries)`: Uses Python's `timeit` and `tracemalloc` to record per-query latency ($\mu\text{s}$) and peak memory consumption (KB/MB).
  - `compute_arithmetic_complexity()`: Analyzes theoretical asymptotic complexity and FLOP counts for KNN search ($\mathcal{O}(N \cdot D)$), Decision Tree traversal ($\mathcal{O}(\text{depth})$), and GPR inference ($\mathcal{O}(N \cdot D)$).
  - `test_unsupported_inputs()`: Synthesizes anomalous sensor events (e.g. $\text{CO}_2 = 3000\text{ ppm}$, $\text{Temp} = 45^\circ\text{C}$, midnight queries) and verifies that the OOD module returns explicit `"unsupported"` rejections.

## 4. Outputs & Deliverables
- Intermediate: `.tmp/validation_report.json`
- Profiling benchmarks: `.tmp/profiling_report.txt`

## 5. Workflow / Step-by-Step Instructions
1. Run `python execution/evaluation_and_profiling.py --predictions ".tmp/model_predictions.parquet" --output-val ".tmp/validation_report.json" --output-prof ".tmp/profiling_report.txt"`
2. Verify all validation criteria, OOD tests, and profiling metrics are logged.

## 6. Edge Cases & Learnings
- In building operations, False Negatives (unoccupied prediction during active occupancy) carry far higher cost than False Positives (over-ventilating an empty room). Metrics must weight FN consequences explicitly.
