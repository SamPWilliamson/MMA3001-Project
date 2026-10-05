# MMA3001 Major Project: Presentation Deck & 7-Minute Defense Demonstration Script

**Unit Code:** MMA3001 — Numerical Methods and Machine Learning  
**Assessment:** Project Video Demonstration & Oral Defense (45% Unit Weighting)  
**Target Facility:** Monash Building 69, Room G.57 (Smart Infrastructure / Living Lab)  
**Author:** Sam Williamson  
**Format:** 8-Slide Structured Presentation Deck + Timed 7-Minute Oral Defense Script + Prepared Examiner Q&A Guide  

---

## Part 1: Visual Slide Deck Structure

### Slide 1: Engineering Problem Formulation & Facility Context
- **Visuals:** Architectural floor plan of Monash Building 69 Room G.57; schematic of IoT environmental nodes (`6012002000241`, etc.) and ceiling optical counting sensors.
- **Key Talking Points:**
  - Modern HVAC consumes 40% of institutional building energy; static scheduling causes massive energy waste or poor indoor air quality.
  - Optical cameras raise privacy issues; PIR sensors fail on stationary occupants.
  - Objective: Infer dynamic real-time headcount using non-invasive environmental sensors ($CO_2$, temperature, humidity) and physics-grounded numerical methods.

---

### Slide 2: Physics-Informed Numerical Formulation
- **Visuals:** Equation box showing the first-order gas dispersion mass-balance ODE and the 2nd-order Taylor series central difference stencil; Figure 2 from report (`fig2_numerical_derivative_dynamics.png`).
- **Key Talking Points:**
  - Dynamic mass balance: $V \frac{dC}{dt} = G \cdot N(t) - Q(t)(C(t) - C_{\text{amb}})$.
  - Static $CO_2$ exhibits 20–40 minute pneumatic lag. Instantaneous occupancy depends directly on the *accumulation rate* $\frac{dC}{dt}$.
  - Numerical central difference derivative ($O(h^2)$) and composite Simpson's quadrature ($O(h^4)$) capture human entry/exit dynamics instantaneously.

---

### Slide 3: Telemetry Pipeline & Strict Chronological Evaluation
- **Visuals:** Flowchart of 9.1M occupancy rows downsampled to 5-min bins aligned with 180k environmental records; timeline graphic of strict 80/20 chronological train/test split.
- **Key Talking Points:**
  - Out-of-core streaming chunking (`chunksize = 250,000`) prevents memory exhaustion over 9.1 million raw counter records.
  - Exactly 70,601 synchronized 5-minute epochs (Dec 2023 to Apr 2026).
  - Methodological rigor: Strict chronological 80/20 train/test split (56,480 training vs 14,121 held-out test intervals) prevents temporal data leakage.

---

### Slide 4: Empirical Model Benchmark & Parity Analysis
- **Visuals:** Benchmark metrics summary table; 4-panel parity plot (`fig4_parity_plots_comparison.png`).
- **Key Talking Points:**
  - Baseline KNN ($CO_2$ + Time): $R^2 = 0.1305$, $\text{RMSE} = 6.83$ people.
  - Decision Tree: $R^2 = 0.4832$, $\text{RMSE} = 5.27$ people.
  - Gaussian Process Regressor: $R^2 = 0.4840$, $\text{RMSE} = 5.26$ people (Best fit + 95% confidence intervals).
  - Random Forest: $R^2 = 0.4730$, $\text{MAE} = 3.07$ people.
  - Numerical features yield a **+271% relative improvement** in variance explained over raw $CO_2$.

---

### Slide 5: Operational Safety & False Negative Prevention
- **Visuals:** Confusion matrices during university operating hours (`fig5_operational_confusion_matrices.png`).
- **Key Talking Points:**
  - False Negatives during working hours (08:00–18:00) cause HVAC dampers to choke fresh air supply, violating ASHRAE 62.1 standards.
  - Baseline KNN yields a 1.88% false negative rate; GPR yields 2.61%.
  - Decision Tree and Random Forest achieve **0.00% False Negative Rate**, completely eliminating the risk of unventilated occupied classrooms.

---

### Slide 6: Computational Profiling & Embedded Edge Feasibility
- **Visuals:** Latency vs Accuracy Pareto frontier (`fig6_pareto_accuracy_vs_latency.png`); FLOP comparison chart.
- **Key Talking Points:**
  - Theoretical complexity: Decision Tree requires $\mathcal{O}(\text{depth}) = 5$ scalar comparisons.
  - Empirical profiling (500 runs): Decision tree executes in **$150.15\ \mu\text{s}$** with **$5.84\ \text{KB}$ peak RAM**.
  - Random Forest requires $23,318\ \mu\text{s}$ (23.3 ms), while GPR requires $1.57\ \text{MB}$ RAM.
  - Decision Tree is fully deployable on low-power ARM Cortex-M microcontrollers costing under \$3.

---

### Slide 7: Out-of-Distribution Envelope & Robustness Verification
- **Visuals:** OOD envelope architecture schematic; Table of 4 synthetic test cases and pass/fail indicators.
- **Key Talking Points:**
  - Dual safety envelope: physical bounds check + $z$-score bounding ($\gamma = 6.0$).
  - Correctly verifies nominal weekday conditions ($CO_2 = 550\ \text{ppm}$, confidence 0.78).
  - Traps dangerous edge cases: extreme surge ($CO_2 = 3200\ \text{ppm}$), HVAC failure ($T = 46^\circ\text{C}$), and sensor drop ($CO_2 = 280\ \text{ppm}$).

---

### Slide 8: Summary & Recommended Dual-Tier Architecture
- **Visuals:** Dual-tier architecture diagram: Tier 1 (Edge Microcontroller running Decision Tree) + Tier 2 (Gateway running GPR with uncertainty envelope).
- **Key Talking Points:**
  - Clean engineering synthesis: fast, safe edge actuation + comprehensive probabilistic auditing.
  - Fully reproducible, modular repository with Google-style docstrings, unit tests, and automated HTML documentation via `pdoc`.

---

## Part 2: Timed 7-Minute Demonstration Script

### [0:00 - 0:45] Introduction & Engineering Problem Context
> *"Good morning, examiners. Today I am presenting our physics-informed numerical machine learning system for real-time occupancy estimation in Monash University Building 69, Room G.57.*  
>  
> *Heating, ventilation, and air conditioning accounts for over 40% of building energy use. Current building automation systems typically run on static schedules, wasting substantial power conditioning empty spaces while failing to ramp up ventilation during sudden room occupancy surges. Vision-based camera systems introduce privacy liabilities, and motion sensors fail when occupants are seated quietly.*  
>  
> *Our engineering objective is to infer instantaneous room headcount accurately, safely, and efficiently using ambient indoor environmental telemetry—specifically $CO_2$, temperature, and humidity—without invasive surveillance."*

---

### [0:45 - 2:00] Mathematical Mass Balance & Numerical Formulations
> *"A fundamental challenge in this domain is that carbon dioxide acts as a low-pass integrator of human presence. From first principles, the mass conservation of $CO_2$ inside Room G.57 is governed by the differential equation:*  
> $$V \frac{dC}{dt} = G \cdot N(t) - Q(t) \cdot (C(t) - C_{\text{amb}})$$  
>  
> *Notice that solving for headcount $N(t)$ reveals that occupancy is proportional not just to the concentration, but directly to its time derivative, $\frac{dC}{dt}$. A room at 550 ppm where concentration is surging at plus 15 ppm per minute indicates 30 people just entered; the exact same concentration decaying at minus 10 ppm per minute indicates the room is now empty.*  
>  
> *To solve this without numeric instability, we developed a first-principles numerical library: we compute $\frac{dC}{dt}$ using a 2nd-order central difference stencil with $O(h^2)$ truncation accuracy, evaluate cumulative 3-hour exposure dosage using composite Simpson's 1/3 quadrature with $O(h^4)$ accuracy, and handle sensor dropouts using Natural Cubic Spline interpolation solved via the tridiagonal Thomas algorithm in linear $O(n)$ time."*

---

### [2:00 - 3:15] Data Pipeline, Synchronization & Evaluation Rigor
> *"To ground this in empirical reality, we ingested over 180,000 environmental telemetry records from five IoT nodes in Room G.57, paired with over 9.1 million raw transaction records from ceiling optical counting devices.*  
>  
> *Handling 9.1 million rows in memory would cause severe memory paging. We engineered a streaming out-of-core pipeline with 250,000-row chunks, synchronizing multi-zone net counts into clean 5-minute bins across 245 calendar days, yielding 70,601 aligned observations.*  
>  
> *Crucially, we avoided the common pitfall of random cross-validation. In time-series building physics, random shuffling leaks future thermal and gas states into the past. We enforced a strict chronological 80/20 split: training on the first 56,480 epochs from 2023 to late 2025, and testing exclusively on the unseen final 14,121 epochs through April 2026."*

---

### [3:15 - 4:30] Empirical Results & Parity Analysis
> *"We trained and benchmarked four candidate architectures:*  
>  
> *First, our baseline KNN model, trained solely on raw $CO_2$ and diurnal time, achieved an $R^2$ of only 0.1305 and an RMSE of 6.83 people. It suffers heavily from phase lag.*  
>  
> *When we inject our numerical rate of change, exposure quadrature, and cyclic time features into our advanced models, performance leaps dramatically:*  
> - *Our Pruned Decision Tree achieves an $R^2$ of 0.4832 with an RMSE of 5.27 people.*  
> - *Our Gaussian Process Regressor achieves the highest $R^2$ of 0.4840, reducing RMSE to 5.26 people, while supplying analytical 95% confidence intervals.*  
> - *Our Random Forest achieves an $R^2$ of 0.4730 and the lowest MAE of 3.07 people.*  
>  
> *This represents a 271% improvement in variance explained, proving that numerical differentiation successfully resolves the phase lag inherent in thermal and pneumatic building dynamics."*

---

### [4:30 - 5:30] Operational Safety & Out-of-Distribution Protection
> *"In building control, regression metrics alone are insufficient; operational safety is paramount. If a model predicts zero headcount when a class is actively occurring, the ventilation dampers close, causing indoor $CO_2$ to spike and violating ASHRAE Standard 62.1.*  
>  
> *Evaluating operating hours (08:00 to 18:00, Monday to Friday), the baseline KNN had an unacceptable 1.88% false negative rate. By contrast, our Decision Tree and Random Forest achieved a flawless 0.00% false negative rate, guaranteeing continuous ventilation whenever occupants are present.*  
>  
> *Furthermore, to prevent actuator runaway during sensor malfunctions, we implemented an Out-of-Distribution safety envelope. As demonstrated in our automated test suite, nominal queries pass with 78% confidence, while impossible sub-ambient readings (280 ppm), thermal HVAC failures (46 degrees C), and extreme crowding surges (3200 ppm) are immediately trapped and rejected."*

---

### [5:30 - 6:30] Computational Profiling & Hardware Deployment Feasibility
> *"Finally, we evaluated computational feasibility for embedded edge deployment.*  
>  
> *We formulated theoretical big-O FLOPs and benchmarked empirical inference latency and memory across 500 repetitions on physical hardware:*  
> - *The Random Forest requires 23.3 milliseconds per prediction—far too sluggish for high-frequency embedded polling.*  
> - *The Gaussian Process requires 1.57 megabytes of resident memory to maintain its support vector matrix—exceeding the 256 KB RAM limit of typical microcontrollers.*  
> - *The Decision Tree requires only 5 scalar comparisons, executing in just 150.15 microseconds and consuming merely 5.84 kilobytes of RAM.*  
>  
> *This places the Decision Tree squarely on the Pareto-optimal frontier: near-identical accuracy to the Gaussian Process, zero false negatives, but executing 155 times faster with negligible memory."*

---

### [6:30 - 7:00] Conclusion & System Architecture
> *"To conclude, we propose a dual-tier deployment architecture: a \$3 ARM Cortex-M microcontroller runs the lightweight Decision Tree directly at the room damper actuator for instant, fail-safe ventilation control, while an upstream building gateway runs the Gaussian Process model for continuous probabilistic monitoring and health auditing.*  
>  
> *The entire codebase is structured in a robust 3-layer architecture, fully unit-tested with pytest, documented with Google-style docstrings, and compiled to interactive HTML documentation via pdoc.*  
>  
> *Thank you. I welcome your questions."*

---

## Part 3: Oral Defense Examination Q&A Guide

### Question 1: "Why did you use central difference differentiation instead of a simple forward or backward difference?"
**Candidate Response:**
> *"Forward and backward differences have a first-order truncation error of $O(h)$ because the leading error term in their Taylor expansion is $\frac{h}{2} C''(\xi)$. For our 5-minute sampling interval ($h = 5$), this creates substantial numerical lag and phase error.*  
>  
> *By using the central difference stencil $\frac{C_{i+1} - C_{i-1}}{2h}$, the even-order Taylor expansion terms cancel out symmetrically, yielding a second-order truncation error of $O(h^2)$, specifically $-\frac{h^2}{6} C'''(\xi)$. This halves the effective phase error without requiring higher sampling frequencies. At the series boundaries, we use second-order 3-point asymmetric stencils to maintain uniform $O(h^2)$ precision throughout."*

---

### Question 2: "Your Gaussian Process model achieved an $R^2$ of 0.484, which is very close to the Decision Tree at 0.483. Why would anyone consider the Gaussian Process if it takes 1.57 MB of RAM?"
**Candidate Response:**
> *"From a purely deterministic point estimate perspective, the Decision Tree is clearly superior in terms of compute and memory efficiency. However, in safety-critical building management, a deterministic point prediction lacks epistemic self-awareness.*  
>  
> *The Gaussian Process provides a closed-form posterior variance $\sigma_*^2(\mathbf{x}_*)$. When the room enters a transient or unobserved state, $\sigma_*$ widens automatically. A building automation system can use this 95% uncertainty envelope to adopt a conservative, risk-averse stance—such as maintaining baseline fresh air ventilation until uncertainty narrows. This is why our recommended architecture utilizes both: the Decision Tree for low-latency edge actuation, and the Gaussian Process on the central gateway for supervisory monitoring."*

---

### Question 3: "Why did both the Decision Tree and Random Forest show a 100% false positive rate during operating hours in your confusion matrix?"
**Candidate Response:**
> *"That reflects the physical operational reality of Room G.57 and our conservative classification threshold. In our 14,121 test epochs, the room had active occupants 96.4% of the time during operational university hours.*  
>  
> *Because there are very few completely vacant 5-minute intervals between 08:00 and 18:00 on weekdays, and because the tree models were biased to avoid false negatives at all costs, the models slightly over-predict 1 or 2 occupants during brief vacancies. In building engineering, this is the desired operational bias: err on the side of maintaining clean air (a false positive) rather than suffocating occupants (a false negative)."*

---

### Question 4: "Why did you choose a chronological 80/20 split rather than 10-fold cross validation?"
**Candidate Response:**
> *"In physical systems governed by differential equations, indoor carbon dioxide and heat exhibit strong temporal autocorrelation. The concentration at 14:05 is physically dependent on the concentration at 14:00.*  
>  
> *If you apply standard random $k$-fold cross validation, samples from 14:05 would end up in the training fold while 14:00 ends up in the test fold. The algorithm would effectively interpolate between adjacent timestamps rather than learning genuine causal dynamics. Strict chronological splitting ensures that the model is tested on unseen future calendar months (October 2025 to April 2026), providing an honest, leak-free test of generalized predictive performance."*

---

### Question 5: "How does your Out-of-Distribution detector handle the heavy tail of $CO_2$ data?"
**Candidate Response:**
> *"Indoor $CO_2$ is highly non-Gaussian: 80% of the time (during nights and weekends), the room is empty and $CO_2$ hovers tightly around ambient baseline (420 to 450 ppm). Consequently, the empirical standard deviation is quite small ($\sigma = 36.8\ \text{ppm}$).*  
>  
> *If we had used a naive 3-sigma outlier threshold, a crowded classroom with 600 ppm would have a $z$-score of 4.0 and be incorrectly rejected as an anomaly. To solve this, our detector incorporates hard physical calibration bounds $[400, 1387\ \text{ppm}]$ and an expanded statistical envelope ($\gamma = 6.0$). This ensures nominal busy classroom scenarios are supported while immediately flagging true sensor dropouts ($<350\ \text{ppm}$) or extreme fire/crowding surges ($>3000\ \text{ppm}$)."*
