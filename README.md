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

The project is structured for **100% reproducible execution** across both cloud (zero-installation) and local workstation environments.

### 🌟 Recommended: One-Click Cloud Execution (Google Colab)

The primary and simplest way to reproduce the entire project—from first-principles numerical verification and machine learning benchmarking to automated unit testing and documentation compilation—is via the interactive **Google Colab "Home Base" Notebook**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamPWilliamson/MMA3001-Project/blob/main/MMA3001_Project_Notebook.ipynb)

1. Open the notebook by clicking the **Open In Colab** badge above.
2. Select **Runtime → Run all** (`Ctrl + F9` / `Cmd + F9`).

**What the notebook executes automatically end-to-end:**
* 📦 **Cloud Environment Scaffolding:** Clones the repository and installs all runtime dependencies (`pyarrow`, `scikit-learn`, `matplotlib`, `seaborn`, `pytest`, `pdoc`).
* 📂 **Telemetry Ingestion:** Ingests the bundled reproducible campus benchmark dataset ([`data/sample_g57_features.parquet`](data/sample_g57_features.parquet)).
* 🧮 **Numerical Methods Verification:** Numerically proves Taylor 2nd-order derivative $\mathcal{O}(h^2)$ convergence, Simpson's 1/3 quadrature $\mathcal{O}(h^4)$ error scaling, and natural cubic spline knot interpolation.
* ⏱️ **Temporal Splitting:** Partitions telemetry into strict chronological 80% train / 20% test sets (preventing temporal data leakage).
* 🏆 **Model Benchmarking:** Trains and evaluates Baseline KNN, Pruned Decision Tree, Gaussian Process Regressor (with analytical uncertainty), and Random Forest, outputting the complete evaluation scorecard.
* 📊 **Diagnostic Engineering Plots:** Renders sensor dynamic responses with numerical derivatives, GPR 95% Bayesian confidence envelopes, and parity plots.
* ⚡ **Hardware Profiling:** Computes microsecond CPU latency, peak resident memory footprints, and microcontroller edge feasibility.
* 🛡️ **Fail-Safe Safety Envelope:** Traps simulated Out-of-Distribution (OOD) anomalies (sensor disconnects, fire surges, HVAC freezing) before BMS actuation.
* 📄 **Interactive Documentation:** Live-compiles HTML5 docstring documentation to `docs/` via `pdoc`.
* ✅ **Automated Unit Testing:** Executes the full 13-test test suite via `pytest` directly in the notebook (verifying 100% test pass rate).

---

### 💻 Local Reproduction & Developer Workflow

If you prefer executing on your local machine:

#### 1. Clone Repository & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/SamPWilliamson/MMA3001-Project.git
cd MMA3001-Project

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

#### 2. Option A: Run the Notebook Locally
Launch JupyterLab or VSCode and execute all cells in [`MMA3001_Project_Notebook.ipynb`](MMA3001_Project_Notebook.ipynb):
```bash
jupyter notebook MMA3001_Project_Notebook.ipynb
```
Select **Kernel → Restart and Run All Cells**.

#### 3. Option B: Run the Automated Unit Test Suite (`pytest`)
Verify all numerical stencils, Simpson's quadrature, cubic splines, physical non-negativity bounds, temporal partition leakage absence, and OOD fault detection:

```bash
pytest -v tests/
```
*(Confirms all 13 independent unit and integration tests pass).*

#### 4. Option C: Modular CLI Execution (Using Bundled Benchmark Data)
To run individual deterministic modules directly using the included reproducible dataset ([`data/sample_g57_features.parquet`](data/sample_g57_features.parquet)):

```bash
# Step 1: Train and evaluate models under chronological 80/20 split
python -m execution.train_and_evaluate \
    --features data/sample_g57_features.parquet \
    --output-metrics .tmp/model_comparison_metrics.json \
    --output-preds .tmp/model_predictions.parquet

# Step 2: Run computational profiling, latency benchmarks, and OOD tests
python -m execution.evaluation_and_profiling \
    --features data/sample_g57_features.parquet \
    --output-val .tmp/validation_report.json \
    --output-prof .tmp/profiling_report.txt

# Step 3: Render publication diagnostic figures to deliverables/figures/
python -m execution.visualizations \
    --features data/sample_g57_features.parquet \
    --preds .tmp/model_predictions.parquet \
    --metrics .tmp/model_comparison_metrics.json \
    --output-dir deliverables/figures

# Step 4: Compile interactive HTML5 source code documentation
pdoc -o docs -d google --math execution
```

> [!NOTE]
> **Full Raw Telemetry ETL (Optional):**  
> The uncompressed raw university IoT telemetry dumps (180,000 environmental records and 9.1 million doorway counter transitions in `Content/`) exceed GitHub file size limits and are internal to Monash University. The repository includes [`data/sample_g57_features.parquet`](data/sample_g57_features.parquet) so that all models, figures, and unit tests reproduce immediately out-of-the-box.  
> If you have access to the original raw CSV files, you can re-run the out-of-core streaming ingestion and feature derivation from scratch:
> ```bash
> python -m execution.data_pipeline --env "path/to/5EnvSensor.csv" --occ "path/to/5occupancySensor.csv" --output .tmp/g57_merged_data.parquet
> python -m execution.feature_engineering --input .tmp/g57_merged_data.parquet --output .tmp/g57_features_matrix.parquet
> ```

---

## AI Acknowledgment & Academic Integrity Declaration

This project utilized Artificial Intelligence (AI) tools in compliance with the Monash University Student Academic Integrity Policy.

**Methodology of AI Use:**
*   **Tools:** Google DeepMind Antigravity / Gemini Advanced Agentic Coding System.
*   **Tasks:** Automated boilerplate scaffolding of Google-style Python docstrings, Markdown report formatting, LaTeX equation typography, and Matplotlib plotting configuration.

**Human Oversight and Engineering Responsibility:**
*   All mathematical formulations (including the Taylor series error derivations, mass-balance differential modeling, composite Simpson's quadrature, and tridiagonal Thomas algorithm implementations), data processing logic, chronological train/test partitioning, out-of-distribution bounds, and engineering interpretations were formulated, reviewed, verified, and validated by the author.
*   The human author maintained full oversight and made all final decisions regarding the integration and modification of all code and analysis.
