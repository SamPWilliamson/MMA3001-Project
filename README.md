# Physics-Informed Numerical Differentiation & Machine Learning for Smart Facility Occupancy Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamPWilliamson/MMA3001-Project/blob/main/MMA3001_Project_Notebook.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Unit Tests](https://img.shields.io/badge/pytest-13%20passed-brightgreen.svg)]()
[![Documentation](https://img.shields.io/badge/pdoc-HTML5%20Docs-blue.svg)](https://sampwilliamson.github.io/MMA3001-Project/)
[![Unit Code](https://img.shields.io/badge/Monash-MMA3001-orange.svg)]()

Engineering implementation, numerical methods library, benchmarking suite, and deliverables for the **MMA3001 Numerical Methods and Machine Learning Major Project** (Monash University, Department of Mechanical and Aerospace Engineering).

Target Facility: **Monash Building 69, Room G.57 (Smart Infrastructure / Living Lab)**.

> 🚀 **Interactive Project Resources:**
> * **Live Web Documentation:** [https://sampwilliamson.github.io/MMA3001-Project/](https://sampwilliamson.github.io/MMA3001-Project/)
> * **Google Colab Executable Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamPWilliamson/MMA3001-Project/blob/main/MMA3001_Project_Notebook.ipynb)
> * **GitHub Repository:** [https://github.com/SamPWilliamson/MMA3001-Project](https://github.com/SamPWilliamson/MMA3001-Project)

---

## Repository Structure & Overview

The repository is organised into modular, testable components designed for complete end-to-end reproducibility:

```
├── MMA3001_Project_Notebook.ipynb     # Interactive Google Colab "Home Base" Project Dossier
├── data/                              # Lightweight reproducible benchmark telemetry
│   └── sample_g57_features.parquet    # Synchronised 2-week campus slice (160 KB)

├── Assignment Details/                # Unit Specification & Sensor Metadata
│   ├── MMA3001_2026_Project_Brief.pdf # Official project guidelines and rubrics
│   ├── MMA3001 Project Datasets.pdf   # Technical dataset documentation
│   └── Datasets/                      # Sensor identifiers and physical placement map
├── execution/                         # Deterministic Python Engineering Modules
│   ├── numerical_methods.py           # First-principles numerical routines (splines, derivatives, quadrature, OOD)
│   ├── data_pipeline.py               # Out-of-core streaming ETL pipeline (180k env + 9.1M counter rows)
│   ├── feature_engineering.py         # Physics-informed rate-of-change and quadrature calculations
│   ├── models.py                      # Candidate ML models (KNN, Decision Tree, GPR with sigma, Random Forest)
│   ├── train_and_evaluate.py          # Chronological 80/20 train/test evaluation & metric tracking
│   ├── evaluation_and_profiling.py    # Latency (us), peak memory (KB), FLOP estimation & OOD verification
│   └── visualizations.py              # Publication-quality figure generation (Matplotlib/Seaborn)
├── deliverables/                      # Project Deliverables & Academic Outputs
│   ├── MMA3001_Project_Report.md      # Comprehensive 10-section academic project report
│   ├── Presentation_Demonstration_Script.md # 8-slide deck structure, 7-min oral script & Q&A defense guide
│   └── figures/                       # Publication-quality vector/raster analytical figures
│       ├── fig1_environmental_time_series.png
│       ├── fig2_numerical_derivative_dynamics.png
│       ├── fig3_gpr_uncertainty_envelope.png
│       ├── fig4_parity_plots_comparison.png
│       ├── fig5_operational_confusion_matrices.png
│       └── fig6_pareto_accuracy_vs_latency.png
├── docs/                              # Compiled interactive HTML5 source documentation (Live: https://sampwilliamson.github.io/MMA3001-Project/)
│   ├── index.html
│   └── execution/
├── tests/                             # Unit & Integration test suite
│   ├── test_numerical_methods.py      # Verification of Taylor stencils, Simpson's quadrature, splines
│   └── test_pipeline.py               # Verification of physical bounds, leakage absence, OOD flags
├── .tmp/                              # Intermediate processed data (Parquet, metrics JSON; git-ignored)
└── README.md                          # Repository guide and reproduction workflow
```

---

## Key Results & Engineering Summary

Across **14,121 unseen held-out test intervals** under strict chronological evaluation:

| Model Architecture | Input Features | Test $R^2$ | RMSE (people) | MAE (people) | Operating Hours False Neg Rate | Inference Latency ($\mu\text{s}$) | Peak RAM Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline KNN ($K=5$)** | Raw $CO_2$ + Time | 0.1305 | 6.83 | 4.33 | 1.88% | 706.47 $\mu\text{s}$ | 25.53 KB |
| **Decision Tree ($\text{depth}=5$)** | Full Feature Suite (16) | 0.4832 | 5.27 | 3.11 | **0.00%** | **150.15 $\mu\text{s}$** | **5.84 KB** |
| **Gaussian Process (RBF)** | Full Feature Suite (16) | **0.4840** | **5.26** | 3.08 | 2.61% | 460.73 $\mu\text{s}$ | 1,570.10 KB |
| **Random Forest ($B=50$)** | Full Feature Suite (16) | 0.4730 | 5.32 | **3.07** | **0.00%** | 23,318.19 $\mu\text{s}$ | 176.91 KB |

**Primary Engineering Findings:**
1. **Physics-informed features** ($\frac{d[CO_2]}{dt}$, $\int \Delta C\, dt$) improved variance explained by **+271%** over raw $CO_2$.
2. **Zero Operating-Hours False Negatives:** The Decision Tree and Random Forest achieved a 0.00% False Negative Rate during university working hours, guaranteeing ventilation dampers remain open whenever occupants are present.
3. **Pareto-Optimal Edge Feasibility:** The Decision Tree achieves near-identical accuracy to the Gaussian Process while executing in **150 $\mu\text{s}$** with **$5.84\ \text{KB}$ RAM**, making it fully deployable on low-power ARM Cortex-M microcontrollers.
4. **Safety Envelope:** The Out-of-Distribution detector traps sensor dropouts, HVAC thermal failures, and crowding surges before actuation.

---

## Quickstart & Reproduction Workflow

### 1. Prerequisites & Environment Setup
Clone this repository and create a Python 3.11 virtual environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
# (or pip install numpy scipy pandas pyarrow scikit-learn matplotlib seaborn pytest pdoc python-dotenv)
```

### 2. Run the Automated Unit Test Suite
Verify that all numerical stencils, splines, quadrature, and pipeline tests pass:

```powershell
.\.venv\Scripts\pytest.exe -v
```

### 3. End-to-End Pipeline Reproduction

Execute the full deterministic data pipeline from raw telemetry to publication figures:

```powershell
# Step 1: Clean and synchronize 180k env + 9.1M occupancy rows
.\.venv\Scripts\python.exe -m execution.data_pipeline `
    --env "Content/5EnvSensor_MayToDec2024_180kRows.csv" `
    --occ "Content/5occupancySensor_MayToDec2024_9MRows.csv" `
    --output .tmp/g57_merged_data.parquet

# Step 2: Extract numerical derivatives, splines, and exposure features
.\.venv\Scripts\python.exe -m execution.feature_engineering `
    --input .tmp/g57_merged_data.parquet `
    --output .tmp/g57_features_matrix.parquet

# Step 3: Train and evaluate models under chronological 80/20 split
.\.venv\Scripts\python.exe -m execution.train_and_evaluate `
    --features .tmp/g57_features_matrix.parquet `
    --output-metrics .tmp/model_comparison_metrics.json `
    --output-preds .tmp/model_predictions.parquet

# Step 4: Run computational profiling, FLOP calculations, and OOD tests
.\.venv\Scripts\python.exe -m execution.evaluation_and_profiling `
    --features .tmp/g57_features_matrix.parquet `
    --output-val .tmp/validation_report.json `
    --output-prof .tmp/profiling_report.txt

# Step 5: Render all 6 publication figures to deliverables/figures/
.\.venv\Scripts\python.exe -m execution.visualizations `
    --features .tmp/g57_features_matrix.parquet `
    --preds .tmp/model_predictions.parquet `
    --metrics .tmp/model_comparison_metrics.json `
    --output-dir deliverables/figures
```

### 4. Inspect Interactive Source Code Documentation
Open the compiled documentation in your default browser:

```powershell
# Open interactive HTML documentation
start docs/index.html
```

---
