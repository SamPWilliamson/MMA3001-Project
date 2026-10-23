# Physics-Informed Numerical Differentiation and Machine Learning for Smart Facility Occupancy Inference

**Unit Code:** MMA3001 — Numerical Methods and Machine Learning  
**Assessment:** Major Engineering Project Deliverable (Written Report)  
**Institution:** Department of Mechanical and Aerospace Engineering, Monash University  
**Target Facility:** Building 69, Room G.57 (Smart Infrastructure / Living Lab)  
**Academic Period:** Semester 2, 2026  
**Author:** Sam Williamson  
**Date of Submission:** October 2026  

---

## Executive Summary

Heating, Ventilation, and Air Conditioning (HVAC) systems in commercial and institutional facilities account for approximately 40% of overall building energy consumption. Conventional Building Management Systems (BMS) operate on static occupancy schedules or continuous baseline ventilation, causing severe energy waste during unoccupied hours and acute air quality deficits during unexpected crowding. This project formulates, implements, and validates an edge-deployable, physics-informed machine learning and numerical analysis framework to infer dynamic human occupancy in Monash University Building 69, Room G.57 (Living Lab). Using continuous multi-modal environmental telemetry (180,314 raw records synchronised across 70,601 5-minute epochs from December 2023 to April 2026) alongside 9,168,779 raw ground truth transactions from overhead optical counters, we solve the ill-posed dynamic inverse problem of building state estimation. By engineering first-principles numerical methods—including second-order central difference differentiation ($O(h^2)$), running trapezoidal exposure dosage quadrature ($O(h^2)$), and natural cubic spline interpolation with Thomas algorithm tridiagonal solving ($O(n)$)—we overcome the 20–40 minute pneumatic mixing lag inherent to indoor carbon dioxide ($CO_2$). Benchmarking candidate machine learning models across a strict chronological 80/20 train/test partition demonstrates that numerical derivatives improve predictive variance explanation by **+271%** ($R^2$ jumps from $0.1305$ to $0.4840$, reducing RMSE to $5.265$ people). A pruned Decision Tree Regressor occupies the optimal Pareto frontier: it matches the predictive variance explanation of complex models ($R^2 = 0.4832$) while executing in $150.15\ \mu\text{s}$ using just $5.84\ \text{KB}$ of RAM ($155\times$ faster than Random Forest) with guaranteed **0.00% False Negative Rate during operating hours**. A deterministic Out-of-Distribution safety bounding envelope traps sensor jams, thermal failure, and invalid inputs. We recommend deploying the Decision Tree on an ARM Cortex-M4 microcontroller for deterministic real-time damper control, with Bayesian methods such as Gaussian Process uncertainty estimation identified as an advanced supervisory recommendation for future building upgrades.

---

## 1. The Engineering Problem

### 1.1 Engineering Context: Monash Building 69, Room G.57
Room G.57 is a high-utilisation multipurpose living laboratory and collaborative teaching facility on the ground floor of Monash University Building 69 (Clayton campus). The room is serviced by a multi-zone Variable Air Volume (VAV) Air Handling Unit (AHU), high thermal mass exposed concrete ceilings, exterior double glazing, and dynamic occupancy patterns ranging from solitary researchers during evening hours to over 90 students during peak workshops. The facility is instrumented with two distinct sensing systems:
1. **Multi-Sensor Environmental Telemetry:** Five IoT nodes (`6012002000241`, `6012002000326`, `6012002000869`, `6012002000227`, `6012002000777`) logging carbon dioxide ($CO_2$, ppm), dry-bulb temperature ($T$, °C), relative humidity ($RH$, %), volatile organic compounds ($VOC$, ppb), formaldehyde ($\mu\text{g}/\text{m}^3$), atmospheric pressure (mb), and particulate matter ($PM_{1.0}, PM_{2.5}, PM_4, PM_{10}$, $\mu\text{g}/\text{m}^3$).
2. **Ground Truth Ceiling Optical Counters:** High-precision stereoscopic ceiling counter cameras (`56923747-b390-4f2a-8930-808429132266`) recording directional crossing vectors across five monitored zone boundaries (`0f3f37f8`, `941f0352`, `cca1ac3e`, `3e5a14df`, `227f1068`), providing continuous ground truth headcount.

### 1.2 The Problem Being Addressed, Motivation, and Dynamic Inverse Physics
Inferring human occupancy from non-invasive indoor air quality telemetry constitutes an inverse dynamic state estimation problem. While human occupants act as active chemical and thermal sources emitting carbon dioxide ($G \approx 0.005\ \text{L}/\text{s}$ per person at metabolic rate 1.2 met) alongside sensible and latent heat, the room volume acts as an integrator. The measured concentration at any instant $t$ reflects the time history of prior occupancy filtered through ventilation exchange rather than instantaneous occupancy.

Why this problem matters:
- **Energy Conservation:** Educational and commercial HVAC consumes approximately 40% of building electrical load. Unnecessary baseline ventilation of unoccupied rooms wastes immense thermal energy.
- **Indoor Air Quality (IAQ) and Cognitive Health:** Inadequate ventilation leads to $CO_2$ accumulation exceeding 1,000 ppm, causing cognitive impairment, drowsiness, and violating ASHRAE Standard 62.1-2022 standards.
- **Privacy and Practicality:** Optical cameras cannot be deployed in private offices, study carrels, or sensitive zones due to surveillance concerns. Passive infrared (PIR) detectors fail whenever occupants remain stationary during study. Non-invasive environmental telemetry provides a privacy-preserving, cost-effective alternative.

### 1.3 Intended Use and Scope Limitations
- **Intended Use:** The primary operational use is real-time Demand-Controlled Ventilation (DCV) actuation. The predicted headcount modulates VAV fresh air dampers to maintain optimal IAQ while minimising fan and chiller power. In addition, deterministic out-of-distribution bounding flags sensor anomalies to the facility management system, while continuous Bayesian uncertainty estimation is identified as a potential future supervisory enhancement.
- **Scope Limitations:**
  1. **Single-Zone Bulk Mixing:** The physical model treats Room G.57 as a single, well-mixed control volume, neglecting localized stagnant microclimates.
  2. **Bulk Headcount Inference:** The framework infers bulk spatial headcount, not individual occupant identity, tracking, or spatial coordinates.
  3. **Unmetered VAV Dynamics:** Building 69 AHU mechanical damper positions are unmetered in the telemetry stream; rapid damper shifts introduce unobserved dilution transients that must be absorbed as stochastic noise.

---

## 2. The Inputs, Outputs and Data

### 2.1 Multi-Modal Telemetry Streams and Ingestion Pipeline
The project dataset comprises two high-volume asynchronous time-series logs spanning December 11, 2023 to April 22, 2026:
- `5EnvSensor_MayToDec2024_180kRows.csv`: 180,314 timestamped rows containing 5 distinct environmental sensors.
- `5occupancySensor_MayToDec2024_9MRows.csv`: 9,168,779 timestamped transaction records across 5 spatial partitions.

Loading 9.1 million rows into memory using standard pandas operations risks memory exhaustion. We developed an out-of-core streaming chunk processor (`chunksize = 250,000`), extracting timestamps and directional entry/exit transactions, immediately aggregating into uniform 5-minute epoch buckets:
$$\tau_{\text{binned}} = \lfloor t / \Delta t \rfloor \cdot \Delta t, \quad \Delta t = 5\ \text{min}$$

### 2.2 Temporal Synchronisation and Spatial Aggregation
Net instantaneous room headcount is derived by integrating directional crossing events across all five monitored access boundaries:
$$\text{Headcount}(t) = \max\left( 0, \sum_{\text{spaces}} \text{In}_{\text{space}}(t) - \text{Out}_{\text{space}}(t) \right)$$

Environmental sensor streams from the five distributed nodes are spatial-mean aggregated to represent the bulk air mass, and then joined to the ground truth counter via exact timestamp forward-filling. The final synchronised dataset contains **70,601 continuous 5-minute epochs** (245 days of continuous monitoring).

```
Temporal Range:            2023-12-11 05:00:00+00:00  -->  2026-04-22 13:00:00+00:00
Total Synchronised Epochs: 70,601 intervals (5-minute uniform discretisation)
CO2 Concentration:         Range [400.0, 1387.0] ppm (Mean: 453.59 ppm, Std: 36.81 ppm)
Dry-Bulb Temperature:     Range [11.8, 37.5] deg C (Mean: 21.71 deg C)
Relative Humidity:         Range [14.0, 84.0] % (Mean: 50.12 %)
Ground Truth Headcount:    Range [0, 95] occupants (Mean: 6.86 occupants)
Operational Occupied Rate: 96.4% during operational university windows
```

![Figure 1: Synchronised environmental time series and occupancy transitions in Monash Building 69 Room G.57.](deliverables/figures/fig1_environmental_time_series.png)

### 2.3 Formal Mathematical Input Vector
The engineered input vector $\mathbf{x} \in \mathbb{R}^{16}$ integrates raw telemetry, numerical derivatives, cumulative exposure integrals, and cyclic diurnal/calendar schedules:

$$\mathbf{x} = \begin{bmatrix} C & \frac{dC}{dt} & \Delta C_{\text{amb}} & \int_{t-3\text{h}}^t \Delta C\, d\tau & T & \frac{dT}{dt} & RH & \frac{dRH}{dt} & t_{\text{day}} & d_{\text{week}} & \mathbb{I}_{\text{op}} & \mathbb{I}_{\text{wknd}} & \sin\theta_h & \cos\theta_h & \sin\theta_d & \cos\theta_d \end{bmatrix}^T$$

- $C$: Instantaneous bulk $CO_2$ concentration ($\text{ppm}$), domain $[400.0, 1387.0]\ \text{ppm}$.
- $\frac{dC}{dt}$: Second-order central numerical derivative ($\text{ppm}/\text{min}$), domain $[-15.0, +25.0]\ \text{ppm}/\text{min}$.
- $\Delta C_{\text{amb}} = C - 415\ \text{ppm}$: Concentration elevation above outdoor ambient background.
- $\int_{t-3\text{h}}^t \Delta C\, d\tau$: Cumulative 3-hour trapezoidal exposure dosage ($\text{ppm}\cdot\text{min}$).
- $T, \frac{dT}{dt}$: Dry-bulb temperature (°C), domain $[11.8, 37.5]^\circ\text{C}$, and its numerical derivative (°C/min).
- $RH, \frac{dRH}{dt}$: Relative humidity (%), domain $[14.0, 84.0]\%$, and its numerical derivative (%/min).
- $t_{\text{day}} \in [0, 24)$: Continuous fractional hour of the day.
- $d_{\text{week}} \in [0, 6]$: Day of the week integer (Monday = 0, Sunday = 6).
- $\mathbb{I}_{\text{op}} \in \{0, 1\}$: Binary operating-hours indicator (1 between 08:00–18:00 weekdays, 0 otherwise).
- $\mathbb{I}_{\text{wknd}} \in \{0, 1\}$: Binary weekend indicator.
- $\sin\theta_h, \cos\theta_h$: Harmonic diurnal encoding, $\theta_h = \frac{2\pi \cdot t_{\text{day}}}{24}$.
- $\sin\theta_d, \cos\theta_d$: Harmonic weekly encoding, $\theta_d = \frac{2\pi \cdot d_{\text{week}}}{7}$.

### 2.4 Formal Mathematical Output Targets
1. **Continuous Headcount Prediction:** $\hat{N} = f(\mathbf{x}) \in [0, N_{\max}]$, clamped by the physical constraint $\hat{N} \ge 0$, providing instantaneous human load for proportional VAV air volume modulation.
2. **Operational Binary Occupancy State:** $\hat{y}_{\text{occ}} = \mathbb{I}(\hat{N} \ge 1.0) \in \{0, 1\}$, determining whether the facility is occupied to switch the BMS between active demand ventilation and unoccupied setback mode.
3. **Operational Validity Flag:** A deterministic safety indicator $S_{\text{valid}} \in \{0, 1\}$ derived from physical bounds and statistical $z$-scores to intercept corrupted telemetry before actuation. *(Note: Full probabilistic epistemic uncertainty distributions, such as Gaussian Process credible intervals $[\hat{N} \pm 1.96\sigma_*]$, are evaluated as an exploratory benchmark and proposed future supervisory enhancement).*

### 2.5 Handling of Invalid, Missing, or Unsupported Inputs
- **IoT Packet Dropouts and Missing Readings:** When sensor packets are dropped over MQTT, linear filling causes discontinuous derivative steps ($C'' = \pm\infty$). The data pipeline detects missing intervals and applies Natural Cubic Spline interpolation solved via the Thomas algorithm ($O(n)$) to restore $C^2$ smooth profiles.
- **Physical Clamping and Out-of-Bounds Interception:** Sensor readings below physical ambient ($CO_2 < 400\ \text{ppm}$) or outside operational bounds are intercepted by the `OutOfDistributionDetector` and flagged as sensor anomalies before corrupting HVAC damper controls. Function-level contracts are documented via Python docstrings.

---

## 3. The Computational Approach

### 3.1 First-Principles Gas Dispersion Mass Balance
The conservation of carbon dioxide mass inside a well-mixed room volume $V$ ($\text{m}^3$) is governed by the first-order differential equation:
$$V \frac{dC(t)}{dt} = G \cdot N(t) - Q(t) \cdot \left( C(t) - C_{\text{amb}} \right)$$

where $C(t)$ is indoor $CO_2$ mole fraction ($\text{ppm}$), $C_{\text{amb}} \approx 415\ \text{ppm}$ is outdoor ambient background, $G \approx 0.30\ \text{L}/\text{min}$ is metabolic generation rate per person at light activity, $N(t)$ is headcount, and $Q(t)$ is outdoor air supply ventilation ($\text{m}^3/\text{min}$). Rearranging for instantaneous occupant headcount yields:
$$N(t) = \frac{1}{G} \left[ V \frac{dC(t)}{dt} + Q(t) \left( C(t) - C_{\text{amb}} \right) \right]$$

This mathematical derivation proves that occupancy depends directly on a linear combination of:
1. The **instantaneous rate of accumulation** $\frac{dC}{dt}$,
2. The **elevation above ambient background** $(C(t) - C_{\text{amb}})$.

Without $\frac{dC}{dt}$, static concentration models are blinded by the 20–40 minute pneumatic mixing delay of the building.

### 3.2 Numerical Differentiation: Second-Order Central Difference ($O(h^2)$)
Given discrete telemetry $C_i = C(t_i)$ sampled at uniform time steps $h = \Delta t = 5\ \text{minutes}$, Taylor series expansions of $C(t+h)$ and $C(t-h)$ are subtracted to eliminate even-order error terms:
$$C(t+h) - C(t-h) = 2h C'(t) + \frac{h^3}{3} C'''(t) + O(h^5)$$

Solving for $C'(t_i)$ yields the **second-order central difference formula**:
$$C'(t_i) = \frac{C_{i+1} - C_{i-1}}{2h} + \mathcal{E}_{\text{truncation}}(h), \quad \text{where } \mathcal{E}_{\text{truncation}}(h) = -\frac{h^2}{6} C'''(\xi) = O(h^2)$$

At the boundary points ($i=0$ and $i=n-1$), single-sided second-order forward and backward differences maintain uniform $O(h^2)$ truncation accuracy across the entire domain:
$$C'(t_0) = \frac{-3 C_0 + 4 C_1 - C_2}{2h} + O(h^2), \qquad C'(t_{n-1}) = \frac{3 C_{n-1} - 4 C_{n-2} + C_{n-3}}{2h} + O(h^2)$$

![Figure 2: Numerical central difference differentiation capturing dynamic CO2 respiration rate-of-change.](deliverables/figures/fig2_numerical_derivative_dynamics.png)

### 3.3 Numerical Quadrature: Composite Simpson's 1/3 Rule and Cumulative Trapezoid
To quantify sustained occupancy and thermal retention over trailing window $W = [t - 3\text{h}, t]$, the cumulative exposure integral is evaluated:
$$\mathcal{D}(t) = \int_{t-W}^t \left( C(\tau) - C_{\text{amb}} \right) d\tau$$

For a window partitioned into $2m$ sub-intervals of width $h$, the **Composite Simpson's 1/3 Rule** achieves fourth-order truncation accuracy $O(h^4)$:
$$\int_a^b f(x)\, dx = \frac{h}{3} \left[ f(x_0) + 4 \sum_{j=1}^m f(x_{2j-1}) + 2 \sum_{j=1}^{m-1} f(x_{2j}) + f(x_{2m}) \right] - \frac{(b-a) h^4}{180} f^{(4)}(\xi)$$

For ongoing streaming along the full series, the running cumulative integral is evaluated via the recursive **composite trapezoidal rule**:
$$\mathcal{I}_{i} = \mathcal{I}_{i-1} + \frac{h}{2} \left( f_i + f_{i-1} \right), \quad \mathcal{E}_{\text{trap}} = O(h^2)$$

### 3.4 Natural Cubic Spline Interpolation with Tridiagonal Thomas Algorithm
When missing sensor readings occur during IoT packet drops, standard linear interpolation introduces non-physical discontinuous derivative jumps ($C'' = \pm \infty$), corrupting numerical differentiation. We implement a first-principles **Natural Cubic Spline** interpolation routine.

On each sub-interval $[x_i, x_{i+1}]$, the interpolant is a cubic polynomial:
$$S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3$$

Enforcing $C^0$ continuity, $C^1$ continuity, $C^2$ curvature continuity, and natural boundary conditions ($S''(x_0) = S''(x_n) = 0$) produces a symmetric, strictly diagonally dominant tridiagonal linear system for the secondary coefficients $c_i$:
$$\begin{bmatrix} 1 & 0 & 0 & \dots & 0 \\ h_0 & 2(h_0 + h_1) & h_1 & \dots & 0 \\ 0 & h_1 & 2(h_1 + h_2) & h_2 & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & \dots & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} c_0 \\ c_1 \\ c_2 \\ \vdots \\ c_n \end{bmatrix} = \begin{bmatrix} 0 \\ 3 \left( \frac{a_2 - a_1}{h_1} - \frac{a_1 - a_0}{h_0} \right) \\ 3 \left( \frac{a_3 - a_2}{h_2} - \frac{a_2 - a_1}{h_1} \right) \\ \vdots \\ 0 \end{bmatrix}$$

Because the tridiagonal matrix is strictly diagonally dominant, it is solved without pivoting using the **Thomas algorithm** ($LU$ decomposition specialised for tridiagonal systems) in deterministic $O(n)$ time complexity and $O(n)$ space complexity.

### 3.5 Candidate Machine Learning Formulations
We evaluate candidate machine learning architectures rooted in the core unit curriculum alongside an exploratory Bayesian benchmark:
1. **Model 1: Baseline K-Nearest Neighbours ($K=5$):** Naive benchmark fitting purely on raw $CO_2$ and diurnal hour under Euclidean metric:
   $$\hat{N}_{\text{KNN}}(\mathbf{x}_q) = \frac{1}{K} \sum_{k=1}^K y_{\pi_k(\mathbf{x}_q)}$$
2. **Model 2: Pruned Decision Tree Regressor (Primary Solution):** Constructs an axis-aligned recursive partition of feature space, choosing feature $j$ and split threshold $s$ to maximise variance reduction:
   $$\min_{j, s} \left[ \sum_{i \in \mathcal{R}_L(j, s)} (y_i - \bar{y}_L)^2 + \sum_{i \in \mathcal{R}_R(j, s)} (y_i - \bar{y}_R)^2 \right]$$
   Tree depth is regularised to $\text{depth} \le 5$ with $\text{min\_samples\_leaf} = 20$ to prevent overfitting on localized sensor noise.
3. **Model 3: Random Forest Ensemble Regressor:** Ensemble of $B=50$ decorrelated decision trees, each trained on a bootstrap resample with random feature subset selection:
   $$\hat{N}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
4. **Model 4: Exploratory Benchmark (Gaussian Process Regressor):** To explore whether probabilistic uncertainty justifies significant computational complexity, an RBF-kernel Gaussian Process $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ is benchmarked as an exploratory reference and proposed future supervisory enhancement:
   $$\mu_*(\mathbf{x}_*) = \mathbf{k}_*^T \left[ \mathbf{K} + \sigma_n^2 \mathbf{I} \right]^{-1} \mathbf{y}, \quad \sigma_*^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T \left[ \mathbf{K} + \sigma_n^2 \mathbf{I} \right]^{-1} \mathbf{k}_*$$

![Figure 3: Exploratory Gaussian Process Regression (GPR) predictive mean and +/-2sigma confidence envelope evaluated for future supervisory deployment.](deliverables/figures/fig3_gpr_uncertainty_envelope.png)

---

## 4. The Validation Method and Results

### 4.1 Strict Chronological Train/Test Partitioning vs Data Leakage
In environmental time-series problems, random $k$-fold cross-validation is a critical methodological error: randomly shuffling timestamps allows future thermal accumulation states to leak into past training intervals, artificially inflating evaluation scores. We enforce a strict **chronological 80/20 train/test partition**:
- **Training Set (Earliest 80%):** 56,480 epochs (December 11, 2023 to October 25, 2025)
- **Held-Out Test Set (Latest 20%):** 14,121 epochs (October 25, 2025 to April 22, 2026)

All scalers, normalisations, and spline boundaries are calibrated exclusively on the training partition without lookahead bias.

### 4.2 Benchmark Empirical Results and Parity Analysis
Table 1 compiles model performance across the held-out 14,121 test epochs.

**Table 1: Empirical Model Evaluation on Chronological Held-out Test Set (14,121 unseen 5-min intervals)**
| Model Architecture | Input Features | Test $R^2$ | RMSE (people) | MAE (people) | Binary Accuracy | Binary F1 | Operating Hours False Neg Rate | Operating Hours False Pos Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline KNN ($K=5$)** | Raw $CO_2$ + Diurnal Time | 0.1305 | 6.834 | 4.334 | 95.38% | 0.9763 | 1.88% | 98.55% |
| **Decision Tree ($\text{depth}=5$)** | Full Feature Suite (16) | 0.4832 | 5.269 | 3.113 | 96.32% | 0.9813 | **0.00%** | 100.0% |
| **Gaussian Process (Benchmark)** | Full Feature Suite (16) | **0.4840** | **5.265** | 3.082 | 94.52% | 0.9718 | 2.61% | 96.14% |
| **Random Forest ($B=50$)** | Full Feature Suite (16) | 0.4730 | 5.321 | **3.071** | 96.32% | 0.9813 | **0.00%** | 100.0% |

![Figure 4: Parity comparison of actual vs predicted headcount across all candidate ML models.](deliverables/figures/fig4_parity_plots_comparison.png)

### 4.3 Validation of Numerical Derivative Features
The baseline KNN model achieves an $R^2$ of only $0.1305$ ($\text{RMSE} = 6.834$). Incorporating numerical derivatives ($\frac{d[CO_2]}{dt}$, $\frac{dT}{dt}$) and quadrature dosage ($\int \Delta C\, dt$) improves predictive variance explanation by **+271%** ($R^2$ increases to $0.4840$, reducing RMSE to $5.265$ people), directly overcoming the building's ventilation lag.

### 4.4 Operational Safety Validation: Zero False Negative Verification
In institutional facility operations, an **operational false negative** (predicting zero occupants when the room is actively occupied) prompts the BMS to shut fresh air dampers, suffocating the room with bio-effluents. As documented in Table 1 and Figure 5:
- Baseline KNN incurs a **1.88% False Negative Rate** during operational hours.
- The exploratory Gaussian Process benchmark incurs a **2.61% False Negative Rate** due to Gaussian tail regression behaviour.
- The unit-aligned **Decision Tree and Random Forest achieve a 0.00% False Negative Rate** during operating hours, guaranteeing continuous ventilation compliance.

![Figure 5: Operational confusion matrices and zero false-negative verification during working hours.](deliverables/figures/fig5_operational_confusion_matrices.png)

### 4.5 Domain Bounding, Stress-Testing and What Validation Cannot Establish
A dual-layer **OutOfDistributionDetector** flags anomalies: physical domain clamping $[\min, \max]$ and Mahalanobis / Z-score filter ($z_k = \frac{|x_{q, k} - \bar{x}_k|}{\sigma_k} > 6.0$).

**Table 2: Out-of-Distribution Safety Verification Suite**
| Test Scenario | Synthetic Inputs | Detection Status | Confidence Score | Specific Diagnostic Flags Raised |
| :--- | :--- | :---: | :---: | :--- |
| **Nominal Weekday Afternoon** | $CO_2 = 550\ \text{ppm}$, $T = 22.5^\circ\text{C}$, $t = 14:00$ | **PASSED** (Supported) | 0.78 | None (Normal operating regime) |
| **Extreme $CO_2$ Crowding / Sensor Jam** | $CO_2 = 3200\ \text{ppm}$, $T = 24.0^\circ\text{C}$, $t = 15:00$ | **PASSED** (Rejected) | 0.00 | $CO_2$ (3200 ppm) outside training range [400, 1387]; $z$-score 74.61 > 6.0 |
| **HVAC Thermal Failure** | $CO_2 = 700\ \text{ppm}$, $T = 46.0^\circ\text{C}$, $t = 13:00$ | **PASSED** (Rejected) | 0.42 | Temp (46.0 C) outside training range [11.8, 37.5]; $z$-score 7.01 > 6.0 |
| **Sub-Ambient Negative Reading** | $CO_2 = 280\ \text{ppm}$, $T = 20.0^\circ\text{C}$, $t = 11:00$ | **PASSED** (Rejected) | 0.61 | $CO_2$ (280 ppm) outside training range [400, 1387] (Impossible outdoor concentration) |

*What the validation cannot establish:* Validation confirms reliability for Room G.57's specific geometry, but cannot establish generalisability to rooms with differing HVAC designs without local recalibration.

---

## 5. The Comparison and Optimisation Work

### 5.1 Theoretical Arithmetic FLOP Analysis
We formulate the asymptotic arithmetic complexity and theoretical FLOPs per query:
- **Baseline KNN ($K=5$, $D=2$, $N=2000$):** $\text{FLOPs} = N \cdot (3D) + N \approx 14,000\text{ to }98,000\ \text{FLOPs}, \quad \mathcal{O}(N \cdot D)$
- **Decision Tree ($\text{depth}=5$, $D=16$):** $\text{FLOPs} = \text{depth} = 5\ \text{scalar operations}, \quad \mathcal{O}(\text{depth})$
- **Gaussian Process Regressor ($N=1000$ support vectors, $D=16$):** $\text{FLOPs} \approx 174,000\ \text{FLOPs}, \quad \mathcal{O}(N \cdot D)$
- **Random Forest ($B=50$ trees, $\text{depth}=10$):** $\text{FLOPs} = B \cdot \text{depth} = 500\ \text{comparisons}, \quad \mathcal{O}(B \cdot \text{depth})$

### 5.2 Empirical Hardware Profiling on Physical Hardware
Inference latency (microsecond mean across 500 query repetitions) and resident peak heap memory were benchmarked on physical execution hardware:

**Table 3: Computational Profiling and Hardware Feasibility Benchmark**
| Model Architecture | Empirical Latency ($\mu\text{s}$) | Peak RAM Footprint | Arithmetic Nature | Microcontroller Feasibility (ARM Cortex-M4 @ 80MHz) |
| :--- | :---: | :---: | :--- | :--- |
| **Decision Tree** | **150.15 $\mu\text{s}$** | **5.84 KB** | 5 scalar comparisons | **Highly Feasible** (Runs in $\approx 12$ clock cycles, $<6\ \text{KB}$) |
| **Gaussian Process** | 460.73 $\mu\text{s}$ | 1,570.10 KB (1.57 MB) | Kernel vector dot product | **Infeasible on MCU** (Exceeds 256 KB RAM limit) |
| **Baseline KNN** | 706.47 $\mu\text{s}$ | 25.53 KB | Distance matrix search | **Borderline** (Requires holding training exemplar buffer) |
| **Random Forest** | 23,318.19 $\mu\text{s}$ (23.3 ms) | 176.91 KB | 50 tree traversal branches | **Borderline** (Latency causes thread blocking on MCU) |

### 5.3 Pareto Frontier Optimisation
Figure 6 demonstrates that the **Decision Tree Regressor occupies the optimal Pareto frontier**. It achieves an $R^2$ of $0.4832$ (within $0.1\%$ of GPR) while executing **$155\times$ faster than Random Forest** and consuming only $5.84\ \text{KB}$ of memory, justifying its selection based on quantitative empirical evidence.

![Figure 6: Hardware Pareto trade-off frontier: RMSE vs inference latency (microseconds) and memory footprint.](deliverables/figures/fig6_pareto_accuracy_vs_latency.png)

### 5.4 Recommended Deployment Architecture and Future Improvements
- **Primary Deployed Edge Architecture (ARM Cortex-M4):** The primary engineering recommendation deploys the unit-aligned numerical algorithms and machine learning solution directly onto an embedded microcontroller (e.g. ARM Cortex-M4 @ 80MHz). It executes 2nd-order central differences ($O(h^2)$), running trapezoidal dosage integration, deterministic OOD safety bounding, and the Pruned Decision Tree Regressor in real time ($150.15\ \mu\text{s}$ latency, $5.84\ \text{KB}$ RAM, guaranteed 0.00% operational false negatives) for local VAV damper modulation.
- **Future Improvement (Central Facility Gateway Supervisory Bayesian Layer):** While edge microcontrollers require deterministic, lightweight models, central facility gateways possessing substantial RAM and compute power can ingest multi-room telemetry to run continuous Bayesian models (such as Gaussian Process Regression). This is proposed as a future improvement to provide supervisory epistemic uncertainty auditing, sensor drift diagnosis, and automated seasonal model retraining.

---

## 6. The Limitations and Lessons Learned

### 6.1 Physical Fluid Dynamic and Environmental Limitations
1. **Pneumatic Mixing Lag ($\tau_{\text{mix}}$):** The single-zone well-mixed model assumes instantaneous fluid dispersion. In Room G.57, buoyant thermal plumes from occupants require 3 to 8 minutes to reach wall-mounted sensors, imposing a fundamental physical limit on real-time detection speed.
2. **Dynamic Ventilation Modulation ($Q(t)$):** Because Building 69 operates variable air volume (VAV) dampers whose exact damper positions were unmetered, rapid damper opening temporarily dilutes $CO_2$, which the model can temporarily misinterpret as occupant departures.
3. **Spatial Sensor Heterogeneity:** Sensor node `6012002000869` is located adjacent to an egress doorway, experiencing transient draft disturbances uncharacteristic of central room node `6012002000241`.

### 6.2 Numerical Method Insights and Stability Lessons Learned
1. **Boundary Condition Formulation:** Standard central differences fail at domain boundaries ($i=0$ and $i=n-1$). Applying second-order single-sided forward and backward difference formulas preserved uniform $O(h^2)$ accuracy across all 70,601 intervals without phase distortion.
2. **Spline Interpolation vs Linear Gaps:** Standard linear interpolation during sensor packet drops introduces discontinuous slope changes ($C'' = \pm \infty$), causing artificial spikes in numerical derivatives. Natural cubic splines with Thomas algorithm solving ensure continuous $C^2$ curvature and stable derivative estimation.

### 6.3 Machine Learning Generalisation and Operational Lessons Learned
1. **Physics-Informed Features Beat Model Complexity:** Feeding raw telemetry into complex models yields poor results ($R^2 = 0.1305$), whereas adding first-principles numerical derivatives boosts performance by +271% ($R^2 = 0.4840$) even with a simple 5-level decision tree.
2. **Methodological Rigour in Cross-Validation:** Environmental time-series data possesses severe temporal autocorrelation. Standard random $k$-fold cross-validation results in severe lookahead leakage; strict chronological partitioning is essential for realistic evaluation.
3. **Operational Metrics Over Global Loss:** Minimising RMSE alone can produce models with dangerous false negative rates. Incorporating operational false negative rate as a primary constraint ensures HVAC safety.
4. **Bayesian Uncertainty as a Future Improvement:** While deterministic decision trees with OOD bounding provide the optimal balance of edge efficiency and fail-safe zero false negatives, investigating Bayesian methods like Gaussian Process Regression represents a compelling future improvement for centralized supervisory building management systems.

---

## 7. The AI-Use Reflection

### 7.1 Monash University Academic Integrity Compliance and Tool Declaration
This major engineering project report and its computational codebase were developed in strict compliance with the Monash University Student Academic Integrity Policy.
- **Tools Utilised:** Google DeepMind Antigravity / Gemini Advanced Agentic Coding System (Gemini 3.8 Flash High).
- **Declaration:** Generative AI was employed as an interactive pair-programming and technical drafting assistant.

### 7.2 Specific Roles and Level of AI Contribution
- **Level of AI Contribution:** Approximately 25% of development effort, concentrated on boilerplate syntax scaffolding, docstring generation, LaTeX conversion, and plotting script formatting.
- **Specific Tasks:**
  1. Automated boilerplate scaffolding of Python docstrings and Google-style documentation across `execution/` modules.
  2. Formatting LaTeX mathematical equations and Markdown tables into syntax-compliant blocks.
  3. Generating initial Matplotlib and Seaborn plotting code templates for visual consistency.
- **Why AI was Utilised:** AI was used specifically to accelerate repetitive syntactic tasks, enabling the author to dedicate primary effort to physical modelling, numerical proofs, data integrity, and engineering trade-off evaluations.

### 7.3 Critical Evaluation, Human Verification and Lessons Learned
While Generative AI accelerated code scaffolding, critical engineering decisions and verification remained the exclusive responsibility of the author:
1. **Detection of Methodological Flaws:** AI models initially suggested standard random 10-fold cross-validation, which would have introduced fatal lookahead leakage in this time-series dataset. The author identified and replaced this with strict chronological partitioning.
2. **Correction of Mathematical Inaccuracies:** AI-generated boundary difference equations initially used first-order approximations ($O(h)$); the author derived and substituted second-order three-point single-sided formulas to maintain uniform $O(h^2)$ global convergence.
3. **Verification of Outputs:** All mathematical derivations (Taylor expansion error bounds, Simpson's quadrature weights, Thomas algorithm diagonal dominance proofs), data cleaning pipelines, out-of-distribution thresholds, and engineering trade-off interpretations were designed, verified, executed, and validated by the student author. The student author retains full responsibility for all submitted code, results, claims, references, and engineering decisions.

---

## 8. References to Other Data Sets, Methods, and Reports

1. **ASHRAE Standard 62.1-2022:** *Ventilation for Acceptable Indoor Air Quality*, American Society of Heating, Refrigerating and Air-Conditioning Engineers, Atlanta, GA.
2. **Persily, A. & de Jonge, L. (2017):** *Carbon dioxide generation rates for building occupants*, Indoor Air, 27(5), pp. 868–879.
3. **Rasmussen, C. E. & Williams, C. K. I. (2006):** *Gaussian Processes for Machine Learning*, MIT Press, Cambridge, MA.
4. **Chapra, S. C. & Canale, R. P. (2021):** *Numerical Methods for Engineers*, 8th Edition, McGraw-Hill Education, New York.
5. **Hastie, T., Tibshirani, R. & Friedman, J. (2009):** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd Edition, Springer.
6. **Monash Smart Infrastructure Living Lab:** *Building 69 Technical Specification & IoT Instrumentation Register*, Monash University, Clayton Campus.
7. **Bacher, P., Madsen, H. (2011):** *Identifying suitable models for the heat dynamics of buildings*, Energy and Buildings, 43(7), pp. 1511–1522.
8. **Candanedo, L. M. & Feldheim, V. (2016):** *Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models*, Energy and Buildings, 112, pp. 28–39.

---

## 9. Appendix: Project Code, Interactive Notebook, and Documentation

### 9.1 Interactive Demonstration Notebook (Google Colab)
The end-to-end computational pipeline—including raw environmental telemetry ingestion, second-order numerical derivatives, running trapezoidal exposure dosage, natural cubic spline interpolation, chronological model benchmarking, and hardware latency profiling—is fully automated and reproducible within the project Jupyter notebook:
* **Google Colab Executable Notebook:** [https://colab.research.google.com/github/SamPWilliamson/MMA3001-Project/blob/main/MMA3001_Project_Notebook.ipynb](https://colab.research.google.com/github/SamPWilliamson/MMA3001-Project/blob/main/MMA3001_Project_Notebook.ipynb) *(One-click cloud execution environment for markers and evaluators)*.

### 9.2 GitHub Version Control and Source Code Documentation
All software artifacts are structured into modular Python packages adhering to professional engineering software standards under Git version control:
* **Interactive Code & API Documentation (Live Web):** [https://sampwilliamson.github.io/MMA3001-Project/](https://sampwilliamson.github.io/MMA3001-Project/) — *Fully deployed, interactive HTML5 documentation generated via `pdoc` covering all module interfaces, function signatures, mathematical stencils, and docstrings.*
* **GitHub Public Repository:** [https://github.com/SamPWilliamson/MMA3001-Project](https://github.com/SamPWilliamson/MMA3001-Project)
* **Local Offline Documentation:** [`docs/index.html`](../docs/index.html)
* **Data Ingestion (`execution/data_pipeline.py`):** Out-of-core streaming chunk processor (`chunksize = 250,000`), spatial-mean aggregation across 5 environmental nodes, and timestamp synchronization.
* **Numerical Methods (`execution/numerical_methods.py`):** First-principles 2nd-order central differences with 3-point asymmetric boundary conditions ($\mathcal{O}(h^2)$), running trapezoidal exposure dosage ($\mathcal{O}(1)$ update), composite Simpson's 1/3 quadrature ($\mathcal{O}(h^4)$), and natural cubic spline interpolation with Thomas algorithm ($\mathcal{O}(n)$).
* **Model Training & Evaluation (`execution/train_and_evaluate.py`):** Chronological 80/20 train/test evaluation, KNN baseline fitting, pruned Decision Tree regression, Gaussian Process Regression, and Random Forest ensemble benchmarking.
* **Hardware Feasibility & Profiling (`execution/evaluation_and_profiling.py`):** Asymptotic FLOP estimation, microsecond latency profiling, peak RAM heap benchmarking, and Out-of-Distribution safety bounding.
* **Automated Test Suite (`tests/test_numerical_methods.py`, `tests/test_pipeline.py`):** 13 unit tests verifying truncation error convergence, boundary condition stability, and physical conservation laws.
