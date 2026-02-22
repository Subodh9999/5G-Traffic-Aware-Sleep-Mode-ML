# Traffic-Aware Sleep Mode Optimization in 5G Small Cells Using Machine Learning

## Project Overview

This project presents a Machine Learning-based approach to optimize energy consumption in 5G small cell networks by predicting user traffic and dynamically switching base station operating modes (ON / IDLE / SLEEP).

The system replaces traditional rule-based control with an adaptive, data-driven solution to improve efficiency and reduce power wastage.

---

##  Problem Statement

Dense deployment of 5G small cells leads to high energy consumption, even during low traffic periods.  
Traditional threshold-based systems fail to adapt to dynamic traffic variations.

This project proposes an intelligent ML-based traffic prediction system to enable adaptive sleep mode control.

---

##  Methodology

### 1️⃣ Traffic Dataset Generation
- Simulated 7-day realistic user traffic pattern
- Modeled office peak and night low-load behavior
- Added controlled randomness for realistic variation

### 2️⃣ Machine Learning Model
- Supervised Learning (Linear Regression)
- Cyclic time encoding using:
  - `sin(hour)`
  - `cos(hour)`
- Input Features:
  - Day  
  - Hour_sin  
  - Hour_cos  
- Output:
  - Predicted User Load

### 3️⃣ Intelligent Mode Switching

Dynamic decision based on predicted traffic:

| Traffic Load | Mode |
|-------------|------|
| High        | ON   |
| Medium      | IDLE |
| Low         | SLEEP |

---

##  Performance Evaluation

The ML-based system was compared with a traditional rule-based system using:

-  Power Consumption (Watts)
-  Energy Efficiency (bits/Joule)
-  Outage Probability (%)

###  Observed Improvements
- Reduced overall power consumption  
- Improved energy efficiency  
- Lower outage probability  
- More stable performance under traffic variation  

---

##  Visualizations Included

- 7-day traffic pattern graph  
- Actual vs Predicted traffic comparison  
- ML-based mode switching heatmap  
- System comparison graphs  
- Real-time traffic animation (GIF)  

---

##  Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

##  Key Learning Outcomes

- Applied supervised machine learning for network optimization  
- Implemented cyclic feature engineering for time-series modeling  
- Designed performance comparison framework  
- Simulated realistic telecom power and outage behavior  

---

##  Author

**Subodh Sunil Bhangre**  
BE Electronics & Telecommunication Engineering  

---

