# Directive: Generate Report Figures, Documentation, and Test Suite

## 1. Goal
Produce all required deliverables for the final submission:
1. High-resolution report figures (multi-panel time series, derivative dynamics, uncertainty bands, parity plots, confusion matrices, Pareto frontier).
2. Automated HTML documentation of all source code via `pdoc`.
3. Comprehensive test suite via `pytest` validating numerical methods and pipeline integrity.
4. Complete 7-12 page Written Project Report and 7-minute Presentation Slide deck structure.

## 2. Inputs
- `.tmp/model_predictions.parquet`, `.tmp/g57_features_matrix.parquet`, `.tmp/model_comparison_metrics.json`, `.tmp/validation_report.json`.

## 3. Tools & Execution Scripts
- `execution/visualizations.py`: Generates publication-quality charts saved to `deliverables/figures/`:
  - `fig1_environmental_time_series.png`
  - `fig2_numerical_derivative_dynamics.png`
  - `fig3_gpr_uncertainty_envelope.png`
  - `fig4_parity_plots_comparison.png`
  - `fig5_operational_confusion_matrices.png`
  - `fig6_pareto_accuracy_vs_latency.png`
- `tests/test_numerical_methods.py`: Unit tests for numerical schemes.
- `tests/test_pipeline.py`: End-to-end integration tests.

## 4. Outputs & Deliverables
- Figures in `deliverables/figures/`
- HTML docs in `docs/`
- Test report in `.tmp/test_report.txt`
- Final report document in `deliverables/MMA3001_Project_Report.md` (and compiled PDF/DOCX)
- Presentation slides and talking script in `deliverables/Presentation_Demonstration_Script.md`

## 5. Workflow / Step-by-Step Instructions
1. Run `python execution/visualizations.py`
2. Run `pytest tests/ -v --junitxml=.tmp/test_report.xml > .tmp/test_report.txt`
3. Run `pdoc execution -o docs/`
4. Assemble written report adhering to rubric criteria.
