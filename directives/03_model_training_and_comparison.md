# Directive: Model Training and Multi-Model Comparison

## 1. Goal
Train, compare, and benchmark multiple machine learning and baseline models for Room G.57 occupancy prediction under a strict chronological train/test split. Produce quantitative comparisons across accuracy, latency, memory, and interpretability.

## 2. Inputs
- `.tmp/g57_features_matrix.parquet`: Feature matrix containing raw sensor signals, numerical derivatives ($\frac{d[\text{CO}_2]}{dt}$), integrated exposure, and temporal indices.
- Target: `headcount` ($\ge 0$) and binary `is_occupied` ($\ge 1$).

## 3. Tools & Execution Scripts
- `execution/models.py`: Model definition suite implementing:
  1. `BaselineKNN`: K-Nearest Neighbors regressor ($k=5$) conditioned only on raw $\text{CO}_2$ and time-of-day.
  2. `DecisionTreeModel`: Pruned CART regression tree (max depth constrained, Week 5) for high interpretability.
  3. `GaussianProcessModel`: Gaussian Process Regressor with Matérn / RBF + WhiteKernel noise estimator, producing both mean estimate $\mu(x)$ and epistemic uncertainty $\sigma(x)$.
  4. `RandomForestModel`: Ensemble bagging model providing robust non-linear feature interaction learning.
- `execution/train_and_evaluate.py`: Orchestrator script executing:
  - Chronological time-based split: e.g. first 80% contiguous duration for training, final 20% contiguous block (2 weeks) held out for validation (matching real deployment).
  - Model fitting, prediction generation, and metric calculation.
  - Physical constraint post-processing: clamping predictions to $[0, \text{Capacity}_{\max}]$.

## 4. Outputs & Deliverables
- Intermediate: `.tmp/model_predictions.parquet` (timestamp, ground truth headcount, predictions for all models, GPR $\sigma$).
- Performance Summary: `.tmp/model_comparison_metrics.json` (RMSE, MAE, $R^2$, training time, inference latency).

## 5. Workflow / Step-by-Step Instructions
1. Run `python execution/train_and_evaluate.py --features ".tmp/g57_features_matrix.parquet" --output-preds ".tmp/model_predictions.parquet" --output-metrics ".tmp/model_comparison_metrics.json"`
2. Verify all 4 models completed training and generated predictions on the withheld test split.
3. Review `.tmp/model_comparison_metrics.json` to confirm baseline vs candidate performance improvement.

## 6. Edge Cases & Learnings
- Data leakage prevention: Never use random train/test split or standard k-fold cross-validation on time-series IoT data. Future data points must never be visible to the past.
- Gaussian Process scaling: GPR matrix inversion scales as $\mathcal{O}(N^3)$. If training points exceed $5,000$, downsample training points via uniform temporal subsampling or k-means anchor selection to ensure fast convergence.
