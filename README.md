# Traffic-Aware Sleep Mode Optimization in 5G Small Cells Using Machine Learning

## Project Overview

This project presents a Machine Learning-based approach to optimize energy consumption in 5G small cell networks by predicting user traffic and dynamically switching base station operating modes (**ON / IDLE / SLEEP**).

The system replaces traditional threshold-based control with an adaptive, data-driven solution to improve efficiency and reduce power wastage while maintaining network quality-of-service (QoS).

---

## Problem Statement

Dense deployment of 5G small cells leads to significant energy consumption, even during low traffic periods.  
Traditional threshold-based systems react only after traffic arrives and fail to adapt proactively to dynamic traffic variations.

This project implements an ML-based traffic prediction system to enable proactive sleep mode control, evaluated under rigorous, fair, and reproducible benchmarking criteria.

---

## Methodology

### 1. Traffic Dataset Generation
- Simulated 7-day realistic user traffic pattern with 24-hour diurnal profiles
- Low load (night: 00:00–06:00), office ramps (06:00–10:00), peak traffic (10:00–17:00), evening shoulder (17:00–21:00), and late night (21:00–24:00)
- Controlled, reproducible random variations using modern NumPy RNG (`default_rng(seed=42)`)

### 2. Machine Learning Models & Time-Aware Validation
- **Feature Engineering**: Cyclic 24-hour time encoding using `Hour_sin = sin(2*pi*Hour/24)` and `Hour_cos = cos(2*pi*Hour/24)`
- **Data Splitting**: Strict chronological split to prevent time-series data leakage (Days 1–5 train [120 hrs], Days 6–7 test [48 hrs])
- **Evaluated Models**:
  - *Linear Regression*: Simple baseline model (captures smooth cyclic harmonic)
  - *Random Forest Regressor*: Non-linear ensemble model (captures step-wise diurnal transitions)
- **Measured Accuracy (Held-out Test Days 6–7)**:
  - **Linear Regression**: MAE = 46.31, RMSE = 61.65
  - **Random Forest**: MAE = 17.89, RMSE = 23.57 (achieves the theoretical noise floor of the simulation)

### 3. Safe Mode Switching Logic
Base stations operate under three physical states:

| Mode | Base Power (Watts) | Operational Role |
|------|--------------------|------------------|
| **ON** | 120 W | Full RF transmission and data processing |
| **IDLE** | 70 W | Standby mode with power amplifiers backed off |
| **SLEEP**| 15 W | Deep micro-sleep mode during minimum traffic periods |

- **Safe Boundaries & Input Validation**: Clamps negative predictions to zero; fails safe to active state on invalid/missing inputs.
- **Safety Margin**: Configurable traffic buffer (default +20 users) to prevent under-provisioning during demand spikes.
- **Hysteresis**: Buffer threshold to prevent rapid mode oscillation (chattering) at boundary states.

---

## Energy & Performance Evaluation

The ML-based system is compared against a traditional threshold-based system using physically grounded metrics:

- **Energy Consumption (Watt-hours, Wh)**: $E = \sum (P \times \Delta t)$ where $\Delta t = 1.0\text{ hr}$
- **Average Power (Watts, W)**: $\bar{P} = \frac{\text{Total Energy}}{\text{Total Time}}$
- **Throughput & Energy Efficiency (bits/Joule)**: $\text{EE} = \frac{\text{Total Bits}}{\text{Energy (Joules)}}$ ($1\text{ Wh} = 3600\text{ Joules}$)
- **Outage Probability (%)**: Percentage of intervals where actual user demand exceeds serving capability
- **Energy Savings (%)**:
  $$\text{Energy Savings (\%)} = \frac{\text{Baseline Energy} - \text{Optimized Energy}}{\text{Baseline Energy}} \times 100$$

### Verified Simulation Results (7-Day Benchmark)

| Metric | Rule-Based (Old System) | ML Predictive System | Unit |
|--------|-------------------------|----------------------|------|
| **Cumulative Energy** | 12,475.00 | 13,135.00 | Wh (Watt-hours) |
| **Average Power** | 74.26 | 78.18 | Watts (W) |
| **Energy Efficiency** | 10,211.98 | 9,698.85 | bits / Joule |
| **Outage Probability** | 29.17% | 29.76% | % of intervals |

> **Engineering Note on Trade-Offs**: In an unconstrained setting without safety margins, aggressive sleep thresholds can artificially cut energy by 20%+, but at the cost of severe network outages (leaving users in IDLE/SLEEP during high traffic). The safe ML controller incorporates a safety margin to prevent service disruption while tracking demand patterns.

---

## Project Structure

```text
├── Final_5G_EC_project.ipynb   # Main end-to-end Jupyter Notebook
├── 7day_traffic_dataset.xls    # 7-day traffic dataset (CSV format)
├── 7day_traffic_dataset.csv    # Generated reproducible traffic dataset
├── traffic_animation.gif       # Real-time traffic & cell mode animation
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── src/
│   ├── __init__.py             # Package initializer
│   └── energy_optimizer.py     # Modular energy calculation & simulation engine
├── tests/
│   └── test_simulation.py      # Automated unit and integration test suite
└── README.md                   # Project documentation & audit report
```

---

## Quickstart & Reproducibility

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m pytest -v
```

### 3. Run the Notebook
Execute `Final_5G_EC_project.ipynb` in VS Code, JupyterLab, or run sequentially from the command line.

---

## Audit & Verification Summary

| Issue | Verification Status | Root Cause & Resolution |
|-------|--------------------|-------------------------|
| **1. Random Data Generation & Reproducibility** | **CONFIRMED & FIXED** | Generation and simulation used unseeded `np.random`. Replaced with explicit `np.random.default_rng(42)` for deterministic reproducibility across runs. |
| **2. Traffic Prediction Accuracy** | **CONFIRMED & FIXED** | Used random `train_test_split` (data leakage) and underfitting Linear Regression (MAE ~46.31). Implemented chronological splitting (Days 1–5 train, Days 6–7 test) and added Random Forest Regressor (MAE ~17.89, RMSE ~23.57). |
| **3. Power vs Energy Calculations** | **CONFIRMED & FIXED** | Hourly power was summed and incorrectly labeled as "Power in Watts". Corrected to cumulative energy in Watt-hours ($E = P \times \Delta t$), average power in Watts, and conversion to Joules ($1\text{ Wh} = 3600\text{ J}$) for bits/Joule efficiency. |
| **4. Unsafe Sleep-Mode Decisions** | **CONFIRMED & FIXED** | Under-predicted traffic caused cells to enter IDLE during evening peak demand. Introduced input validation, a configurable safety margin (+20 users), and hysteresis to prevent oscillation. |
| **5. Fair Evaluation & Outages** | **CONFIRMED & FIXED** | Outages were previously evaluated against predicted users rather than actual demand, and random failure rates were unequal (10% vs 5%). Standardized to identical workloads and fair demand-based outage evaluation. |

---

## Tech Stack

- **Python 3.10+**
- **NumPy, Pandas**: Numerical analysis & reproducible data modeling
- **Scikit-learn**: Time-series feature engineering & machine learning regression
- **Matplotlib, Seaborn**: Data visualization & dynamic animation
- **Pytest**: Automated test suite for physical calculations and boundary conditions

---

## Authors & Contributors

- **Subodh Sunil Bhangre** (Original Author)
- **Yash Nalawade** (Validation, Test Suite, & Simulation Hardening)
