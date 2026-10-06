# Smart Room Occupancy Estimation: Plain-English Understanding Guide

**Unit Code:** MMA3001 — Numerical Methods and Machine Learning  
**Target Space:** Monash University Building 69, Room G.57 (Smart Infrastructure / Living Lab)  
**Author:** Sam Williamson  
**Date:** October 2026  
**Companion PDF Document:** [`MMA3001_Project_Understanding_Guide.pdf`](./MMA3001_Project_Understanding_Guide.pdf)

---

## Welcome & How to Read This Guide

If you have never opened the project brief, never seen the raw sensor spreadsheets, and have never written a line of machine learning code—**this guide was written specifically for you**.

It answers four straightforward questions:
1. **The Problem:** What real-world challenge were we trying to solve, and why does it matter?
2. **The Data:** What raw measurements were we given, and how did we clean and synchronize them?
3. **The Method:** Why does standard "naive" AI fail, and how did physics and numerical methods solve it?
4. **The Results:** Which models won, how fast and accurate were they, and what should engineers do next?

---

## Part 1: What Problem Were We Trying to Solve?

### The Real-World Scenario
Imagine walking into a large university building in the middle of winter:
- In **Room A**, 60 students are crowded together working on an engineering project. The air feels hot, stuffy, and sluggish because the ventilation system isn't supplying enough fresh oxygen.
- Three doors down in **Room B**, a large lecture theatre is completely empty. Yet, industrial heating and ventilation fans are blasting at full power, blowing heated air into empty space and burning hundreds of kilowatt-hours of electricity.

This is an enormous real-world problem:
- **Heating, Ventilation, and Air Conditioning (HVAC)** accounts for **approximately 40% of all energy consumed by university and commercial buildings**.
- Traditional building management systems (BMS) operate on dumb, static timers: they switch fans on at 8:00 AM and shut them off at 6:00 PM regardless of whether anyone is actually inside.

### Why Not Just Use Cameras or Motion Sensors?
You might ask: *"Why not just put security cameras or motion sensors on the ceiling?"*  
Engineers face three huge roadblocks with those approaches:
1. **Privacy & Ethical Liabilities:** Students, researchers, and lecturers strongly object to video surveillance in classrooms. Video feeds create severe legal liabilities, GDPR/privacy risks, and cybersecurity hazards.
2. **Motion Sensors (PIR) Fail on Seated Occupants:** Standard infrared motion detectors only register moving heat sources. If 40 students sit quietly listening to a lecture or taking a 2-hour exam, the sensor assumes the room is empty and shuts off fresh air!
3. **High Hardware Cost & Blind Spots:** Equipping every sub-zone with commercial stereoscopic vision cameras is expensive and prone to blind spots behind partitions and pillars.

> ### 🎯 The Core Engineering Question
> **Can we accurately count the number of people in a room using only simple, non-invasive environmental sensors—measuring invisible properties like Carbon Dioxide ($CO_2$), temperature, and humidity—without using any cameras or invading anyone's privacy?**

### The Target Space: Monash Building 69, Room G.57
Our study focuses strictly on **Room G.57 on the ground floor of Building 69 at Monash University (Clayton Campus)**. Room G.57 is a high-utilization "Living Lab" collaborative teaching facility instrumented with five environmental IoT sensing nodes and ceiling optical doorway counting sensors.

---

## Part 2: What Data Did We Have & How Did We Clean It?

We were provided with real campus telemetry collected continuously from **December 11, 2023 through April 22, 2026** (over two full years of real campus life). The data arrived in two massive, uncoordinated files:

### 1. The Environmental Sensor Log (180,314 rows)
Collected by five IoT sensing boxes scattered throughout Room G.57 (nodes `6012002000241`, `6012002000326`, `6012002000869`, `6012002000227`, and `6012002000777`). These sensors recorded:
- **Carbon Dioxide ($CO_2$):** Measured in parts per million (ppm). Outdoor fresh air is ~415 ppm. Inside a crowded classroom, it often rises above 1000–1200 ppm.
- **Dry-Bulb Temperature ($T$):** Degrees Celsius (°C).
- **Relative Humidity ($RH$):** Percentage (%RH).
- **Particulate Matter & Air Contaminants:** Dust particles ($PM_{1.0}, PM_{2.5}, PM_{10}$) and Volatile Organic Compounds ($VOC$).

### 2. The Ceiling People Counter Log (9,168,779 rows!)
A high-precision ceiling optical counter (`56923747-b390-4f2a-8930-808429132266`) pointed at doorways, recording a transaction timestamp every time a person walked into or out of five sub-room spaces (`0f3f37f8`, `941f0352`, `cca1ac3e`, `3e5a14df`, `227f1068`). Over two years, this sensor logged **over 9.1 million individual crossing events**.

### The Big Data Challenge & Out-of-Core Streaming
Trying to open a 9.1-million-row file directly in Excel or standard pandas causes an immediate memory crash (`MemoryError`). Furthermore, the doorway counter recorded random fractional seconds whenever people walked through, while the environmental sensors logged every few minutes.

> ### ⚙️ How We Solved the Big Data Challenge
> We engineered a streaming Python pipeline that processed the 9.1 million rows in **chunks of 250,000 rows at a time**. For each chunk, it aggregated net entries and exits, and rounded timestamps into clean **5-minute time buckets**. We then forward-filled the environmental sensor readings to match each 5-minute bucket.  
> **The Result:** Exactly **70,601 clean, synchronized 5-minute intervals** spanning December 2023 to April 2026. Zero memory crashes, fully reproducible.

### The Golden Rule: Strict Chronological Testing (No Time Travel!)
In many beginner machine learning tutorials, people randomly shuffle their data before training and testing. **In building engineering, randomly shuffling time-series data is a critical error.**

Because heat and carbon dioxide take time to build up and dissipate, the air at 2:05 PM already contains the heat and gas exhaled at 2:00 PM. If you randomly shuffle rows, the model uses data from 2:05 PM to "predict" 2:00 PM—which is cheating!

To ensure 100% scientific honesty, we enforced a **strict chronological 80/20 train/test split**:
- **Training Set (Earliest 80%):** 56,480 time steps (December 11, 2023 to October 25, 2025). The AI models learned exclusively from this past data.
- **Held-Out Test Set (Latest 20%):** 14,121 time steps (October 25, 2025 to April 22, 2026). The AI models were evaluated strictly on unseen future months.

---

## Part 3: The Physics Problem & Why Naive AI Fails

Why can't you just give raw $CO_2$ numbers to a standard AI model and let it make predictions? Because indoor air behaves like a **leaky bucket**.

> ### 🪣 The "Leaky Bucket" Analogy
> Think of a classroom as a bucket with a small drain hole at the bottom:
> - **People breathing in the room** is like turning on a water tap pouring into the bucket.
> - **The HVAC ventilation fan** is like the drain hole letting water escape.
> 
> When 30 students walk into an empty classroom, the bucket does not instantly fill to the brim. It takes **20 to 40 minutes** for $CO_2$ to slowly rise from 415 ppm to 700 ppm! Conversely, when everyone leaves at the end of class, the $CO_2$ does not vanish instantly—it takes 45 minutes for the ventilation fans to flush the stale air out.

### Why a Naive AI Gets Fooled
If you only feed the raw $CO_2$ concentration (e.g. "600 ppm") into an AI model, the model is completely blind to context:
- **Scenario A:** 35 students just walked in 5 minutes ago. $CO_2$ is currently at 600 ppm, but *rising rapidly at $+15\ \text{ppm}/\text{min}$*.
- **Scenario B:** Everyone walked out 20 minutes ago. $CO_2$ is still at 600 ppm, but *falling at $-10\ \text{ppm}/\text{min}$* as fans blow fresh air.

A naive model looks at 600 ppm and cannot tell whether the room is packed with people or completely empty! This is why a naive baseline model gets a terrible accuracy score ($R^2 = 0.13$).

### The Secret Sauce: Physics-Informed Mass Balance
From conservation of mass, the rate of change of carbon dioxide inside a room volume $V$ is governed by:
$$V \frac{dC}{dt} = G \cdot N(t) - Q(t) \cdot (C(t) - C_{\text{amb}})$$

Rearranging for instantaneous headcount $N(t)$:
$$N(t) = \frac{1}{G} \left[ V \frac{dC}{dt} + Q(t) (C(t) - C_{\text{amb}}) \right]$$

This physical equation proves that **instantaneous headcount depends directly on the slope (the derivative $\frac{dC}{dt}$)**! To give our AI the physics it needed, we implemented three first-principles numerical algorithms:

1. **Numerical Differentiation (2nd-Order Central Difference):**  
   Using Taylor series expansions, we implemented the central difference stencil:
   $$\frac{dC}{dt} \approx \frac{C_{i+1} - C_{i-1}}{2h}$$
   This cancels out even-order truncation errors ($O(h^2)$ precision). As shown in the project figures, positive spikes immediately signal people arriving, cutting straight through the 30-minute pneumatic lag!

2. **Numerical Quadrature (Simpson's 1/3 Rule & Cumulative Trapezoid):**  
   To measure chronic cumulative exposure over the past 3 hours, we computed numerical integrals of concentration elevation ($\int \Delta C\, dt$) using **Composite Simpson's 1/3 Rule** ($O(h^4)$ precision) and cumulative trapezoidal integration.

3. **Natural Cubic Splines with Tridiagonal Thomas Algorithm:**  
   When IoT sensors experience occasional packet drops, linear interpolation creates sharp, jagged corners that ruin derivative calculations. We implemented a **Natural Cubic Spline** routine that solves the tridiagonal matrix in linear $O(n)$ time using the Thomas algorithm, ensuring smooth $C^2$ continuous curves.

---

## Part 4: The 4 Machine Learning Contenders

We trained and benchmarked four distinct machine learning architectures:

1. **Baseline K-Nearest Neighbours (KNN, $K=5$):**  
   *The Naive Benchmark.* Looks through past history to find the 5 closest readings based strictly on raw $CO_2$ and time of day, then averages their headcounts. Demonstrates what happens when an AI lacks physics and numerical rate-of-change features.
2. **Pruned Decision Tree Regressor ($\text{depth} = 5$):**  
   *The Fast Rule-Based Thinker.* Learns a flowchart of smart yes/no questions (e.g. *"Is it a weekday?"*, *"Is $CO_2$ rising faster than 1.5 ppm/min?"*, *"Is temperature above 22°C?"*). Constrained to 5 levels, it executes in exactly 5 simple comparisons without matrix multiplications.
3. **Gaussian Process Regressor (GPR with RBF Kernel):**  
   *The Honest Probabilistic Expert.* Unlike normal AI that only gives a single point guess, a Gaussian Process outputs a **full bell curve (predictive mean and 95% confidence bounds)** for every single prediction! For example, it doesn't just say *"14 people"*; it says *"14 people, with 95% confidence that the true number is between 11 and 17 people"*.
4. **Random Forest Ensemble ($B=50$ Trees):**  
   *The Committee of Experts.* Creates 50 diverse decision trees, each trained on a random slice of data. All 50 trees vote, and their predictions are averaged together to resist sensor noise.

---

## Part 5: What Results Were Produced? (The Answers)

We evaluated all four models across the **14,121 unseen held-out test intervals** (October 2025 to April 2026). Here is the final head-to-head scorecard across accuracy, operational safety, speed, and memory:

### The Final Benchmark Scorecard

| Model Architecture | Input Features | Accuracy ($R^2$) | Average Error (MAE) | Operating Hours Miss Rate (FNR) | Inference Latency | Peak RAM Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline KNN ($K=5$)** | Raw $CO_2$ + Time | 0.1305 *(Poor)* | 4.33 people | 1.88% *(Dangerous)* | 706.5 $\mu\text{s}$ | 25.5 KB |
| **Decision Tree ($\text{depth}=5$)** | Full Physics (16) | 0.4832 *(+270%)* | 3.11 people | **0.00% (Perfect)** | **150.2 $\mu\text{s}$ (Fastest)** | **5.8 KB (Tiny)** |
| **Gaussian Process** | Full Physics (16) | **0.4840 (Best)** | 3.08 people | 2.61% | 460.7 $\mu\text{s}$ | 1,570 KB *(1.5 MB)* |
| **Random Forest ($B=50$)** | Full Physics (16) | 0.4730 *(+262%)* | **3.07 people** | **0.00% (Perfect)** | 23,318 $\mu\text{s}$ *(Slow)* | 176.9 KB |

---

### Four Key Takeaways from the Results

1. **Physics Features Tripled Explanation Power (+271% Boost):**  
   The baseline KNN model managed an $R^2$ of only 0.13. Introducing our numerical derivatives and quadrature exposure boosted $R^2$ to 0.484. The AI can now distinguish whether a room is filling up or clearing out.
2. **Flawless Operational Safety (0.00% Miss Rate):**  
   In a real building, predicting "0 people" during active class hours causes fresh air dampers to shut down, suffocating the classroom. During university working hours (08:00 to 18:00, Monday to Friday), our **Decision Tree and Random Forest achieved a 0.00% False Negative Rate**—they never once suffocated an active classroom!
3. **The Decision Tree is the Speed & Memory Champion:**  
   While the Random Forest took 23,318 microseconds (23.3 milliseconds) to make a prediction, the Decision Tree took just **150 microseconds (0.00015 seconds!)** and used merely **5.8 KB of RAM**. It achieves essentially the same accuracy as the complex Gaussian Process, but runs 155 times faster than Random Forest!
4. **Out-of-Distribution Protection Passed All Tests:**  
   We built a safety envelope to detect malfunctioning sensors. When tested on extreme synthetic faults—like an impossible 280 ppm reading, a 46°C heating runaway, or a 3200 ppm smoke/crowd surge—the system immediately flagged and rejected them rather than trusting corrupted data.

---

## Part 6: The Final Engineering Recommendation

If you were the facility manager of Monash Building 69, which model should you actually put into production? The answer is pairing the best two models into a **Dual-Tier Architecture**:

> ### 🏗️ The Recommended Dual-Tier Deployment Architecture
> - **Tier 1: On-the-Wall Edge Microcontroller (ARM Cortex-M4 @ $3):**  
>   Runs the **Decision Tree model** directly inside the wall thermostat sensor. Because it requires only 5.8 KB of RAM and executes in 150 microseconds, it runs locally without internet. It controls ventilation dampers in real-time with **zero false negatives**, keeping air clean and fans efficient.
> - **Tier 2: Building Central Server / Cloud Gateway:**  
>   Runs the **Gaussian Process Regressor** every 15 minutes. It analyzes long-term trends, calculates 95% confidence intervals, checks for sensor drift, and flags maintenance alerts when uncertainty spikes.

### Code Quality & Verification Standards
- **13/13 Automated Unit Tests:** Verified with `pytest` in 3.32 seconds (testing Taylor expansions, Simpson's quadrature, cubic splines, and data bounds).
- **Interactive HTML Documentation:** Compiled with `pdoc` into `docs/index.html` with Google-style docstrings and formulas.
- **Git Hygiene:** Multi-gigabyte raw CSVs are safely ignored in `.gitignore`, while deterministic code and figures are fully tracked.

---

## Part 7: Plain-English Glossary of Key Terms

| Technical Term | Plain-English Meaning |
| :--- | :--- |
| **$CO_2$ (Carbon Dioxide)** | A colorless gas exhaled by humans (~4% of exhaled breath). Fresh outdoor air is ~415 ppm. Inside crowded, poorly ventilated rooms, it rises above 1000 ppm, causing sluggishness and headache. |
| **PPM (Parts Per Million)** | The unit of gas concentration. 400 ppm means 400 molecules of $CO_2$ for every 1 million air molecules. |
| **Headcount** | The true physical count of human occupants currently inside Room G.57 at any given moment. |
| **Numerical Differentiation** | Using discrete math to calculate how fast a value is changing (the slope). Central difference uses points before and after to get an accurate derivative without phase lag. |
| **Numerical Quadrature (Integration)** | Calculating the total accumulated area under a curve (e.g. total $CO_2$ exposure over 3 hours) using Simpson's Rule or Trapezoid Rule. |
| **Cubic Spline** | A mathematical technique that smoothly connects separate data points with flexible cubic curves, preventing jagged artificial jumps when sensor packets drop. |
| **$R^2$ (Coefficient of Determination)** | A score from 0.0 to 1.0 showing how much variance the model explains. 0.13 is very poor; 0.48 is a strong, physics-grounded real-world fit. |
| **MAE & RMSE** | Average prediction error in number of people. An MAE of 3.07 means predictions are typically within ~3 people of ground truth. |
| **False Negative Rate (FNR)** | Predicting 0 people when the room is actually occupied. In building ventilation, this is dangerous because fresh air dampers close. Our tree models achieved 0.00% FNR. |
| **Inference Latency** | The time it takes for a model to make one prediction. 150 $\mu\text{s}$ means 150 millionths of a second (0.00015 s). |
| **Out-of-Distribution (OOD)** | When an incoming sensor reading is completely crazy or outside what the model ever saw in training (e.g. 3200 ppm or 46°C). An OOD detector catches this and raises an alert. |

---

*Compiled for Sam Williamson — MMA3001 Major Project, Department of Mechanical and Aerospace Engineering, Monash University.*
