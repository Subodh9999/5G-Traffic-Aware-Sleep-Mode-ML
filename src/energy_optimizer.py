"""
5G Traffic-Aware Sleep Mode Optimization Module
Provides reproducible data generation, feature engineering,
safe mode decision logic, energy calculations, and fair performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Base station physical power parameters (in Watts)
POWER_ON: float = 120.0
POWER_IDLE: float = 70.0
POWER_SLEEP: float = 15.0
MAX_CAPACITY: int = 450


def generate_traffic_data(
    days: int = 7,
    hours: int = 24,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic 7-day hourly user traffic dataset reproducibly.
    
    Parameters:
        days: Number of days to simulate (default 7).
        hours: Number of hours per day (default 24).
        seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame with Day, Hour, Users, Hour_sin, and Hour_cos.
    """
    rng = np.random.default_rng(seed)
    data = []

    for d in range(days):
        for h in range(hours):
            if 0 <= h < 6:
                base = 40
            elif 6 <= h < 10:
                base = 200
            elif 10 <= h < 17:
                base = 520
            elif 17 <= h < 21:
                base = 350
            else:
                base = 150

            noise = int(rng.integers(-40, 40))
            users = max(0, base + noise)
            data.append([d + 1, h, users])

    df = pd.DataFrame(data, columns=["Day", "Hour", "Users"])
    df = add_cyclic_features(df)
    return df


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic sine and cosine hour features for 24-hour diurnal patterns.
    """
    df = df.copy()
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    return df


def split_traffic_data(
    df: pd.DataFrame,
    test_days: int = 2,
    chronological: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split traffic dataset into training and test sets.
    
    If chronological=True, splits sequentially by days to avoid time-series data leakage.
    For 7 days with test_days=2: Days 1-5 train (120 hrs), Days 6-7 test (48 hrs).
    """
    feature_cols = ["Day", "Hour_sin", "Hour_cos"]
    X = df[feature_cols]
    y = df["Users"]

    if chronological:
        max_day = df["Day"].max()
        train_mask = df["Day"] <= (max_day - test_days)
        test_mask = df["Day"] > (max_day - test_days)
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask]
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_days / df["Day"].max(), random_state=random_state)


def decide_cell_mode(
    predicted_users: float,
    on_threshold: float = 300.0,
    idle_threshold: float = 150.0,
    safety_margin: float = 0.0,
    current_mode: Optional[str] = None,
    hysteresis: float = 20.0
) -> str:
    """
    Safely determine base station operational mode with bounds checking,
    safety margin, and hysteresis to prevent chattering.
    
    Parameters:
        predicted_users: Predicted traffic demand.
        on_threshold: Traffic threshold for ON mode.
        idle_threshold: Traffic threshold for IDLE mode.
        safety_margin: Additive buffer to prevent under-provisioning.
        current_mode: Previous mode, used for hysteresis.
        hysteresis: Buffer below threshold before switching to a lower power mode.
    """
    # Guard against negative or NaN predictions
    if np.isnan(predicted_users):
        return "ON"  # Fail-safe to active mode on missing data
    
    effective_users = max(0.0, float(predicted_users)) + safety_margin

    if current_mode == "ON":
        # Hysteresis: only leave ON if demand drops below on_threshold - hysteresis
        if effective_users > (on_threshold - hysteresis):
            return "ON"
        elif effective_users > idle_threshold:
            return "IDLE"
        else:
            return "SLEEP"
    elif current_mode == "IDLE":
        if effective_users > on_threshold:
            return "ON"
        elif effective_users > (idle_threshold - hysteresis):
            return "IDLE"
        else:
            return "SLEEP"
    else:
        # Default or transitioning from SLEEP
        if effective_users > on_threshold:
            return "ON"
        elif effective_users > idle_threshold:
            return "IDLE"
        else:
            return "SLEEP"


def decide_rule_based_mode(
    users: float,
    on_threshold: float = 300.0,
    idle_threshold: float = 150.0
) -> str:
    """
    Baseline rule-based operational mode decision based on observed users.
    """
    if users > on_threshold:
        return "ON"
    elif users > idle_threshold:
        return "IDLE"
    else:
        return "SLEEP"


def calculate_interval_energy(power_watts: float, interval_hours: float = 1.0) -> float:
    """
    Calculate energy in Watt-hours (Wh) for an interval of given duration.
    E (Wh) = P (W) * t (hours).
    """
    if interval_hours <= 0:
        raise ValueError("Interval duration must be strictly positive.")
    return power_watts * interval_hours


def convert_energy_wh_to_joules(energy_wh: float) -> float:
    """
    Convert Watt-hours to Joules: 1 Wh = 3600 J.
    """
    return energy_wh * 3600.0


def calculate_energy_savings(baseline_energy: float, optimized_energy: float) -> float:
    """
    Calculate percentage energy savings:
    Energy Savings (%) = ((Baseline - Optimized) / Baseline) * 100
    
    Safely handles zero or negative baseline energy.
    """
    if baseline_energy <= 0:
        if baseline_energy == 0 and optimized_energy == 0:
            return 0.0
        raise ValueError(f"Invalid baseline energy: {baseline_energy}. Must be positive.")
    return ((baseline_energy - optimized_energy) / baseline_energy) * 100.0


def simulate_network_strategy(
    df: pd.DataFrame,
    strategy: str = "ml",
    model: Any = None,
    interval_hours: float = 1.0,
    seed: int = 42,
    on_threshold: float = 300.0,
    idle_threshold: float = 150.0,
    safety_margin: float = 20.0,
    use_hysteresis: bool = True,
    power_noise_std: float = 0.0,
    stochastic_failure_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Fair and reproducible network performance simulation over a traffic workload.
    
    Both ML and Rule-based strategies receive the exact same workload and
    are evaluated using identical outage and power measurement criteria.
    """
    rng = np.random.default_rng(seed)
    
    total_energy_wh = 0.0
    total_data_bits = 0.0
    outages = 0
    mode_history = []
    power_history = []
    
    current_mode = None
    
    for _, row in df.iterrows():
        users = float(row["Users"])
        
        # Determine mode
        if strategy == "ml":
            if model is None:
                raise ValueError("A trained model must be provided for ML strategy.")
            test_feat = pd.DataFrame([{
                "Day": row["Day"],
                "Hour_sin": row["Hour_sin"],
                "Hour_cos": row["Hour_cos"]
            }])
            pred_users = float(model.predict(test_feat)[0])
            mode = decide_cell_mode(
                pred_users,
                on_threshold=on_threshold,
                idle_threshold=idle_threshold,
                safety_margin=safety_margin,
                current_mode=current_mode if use_hysteresis else None
            )
        elif strategy == "rule_based":
            mode = decide_rule_based_mode(
                users,
                on_threshold=on_threshold,
                idle_threshold=idle_threshold
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        current_mode = mode
        mode_history.append(mode)
        
        # Base power by mode
        if mode == "ON":
            base_p = POWER_ON
        elif mode == "IDLE":
            base_p = POWER_IDLE
        else:
            base_p = POWER_SLEEP
            
        # Optional fair symmetric power noise
        if power_noise_std > 0:
            p = base_p * (1.0 + rng.normal(0, power_noise_std))
        else:
            p = base_p
            
        power_history.append(p)
        interval_energy = calculate_interval_energy(p, interval_hours)
        total_energy_wh += interval_energy
        
        # Throughput assumption (e.g., 10 Mbits per user per hour)
        total_data_bits += users * 10.0
        
        # Fair outage evaluation based on ACTUAL USER DEMAND
        # 1. Functional outage: demand requires active transmission, but cell is in SLEEP or IDLE
        if users > on_threshold and mode != "ON":
            outages += 1
        elif users > idle_threshold and mode == "SLEEP":
            outages += 1
        elif users > MAX_CAPACITY:
            outages += 1
        elif stochastic_failure_rate > 0 and users > on_threshold:
            if rng.random() < stochastic_failure_rate:
                outages += 1

    total_intervals = len(df)
    total_hours = total_intervals * interval_hours
    avg_power_w = total_energy_wh / total_hours if total_hours > 0 else 0.0
    energy_joules = convert_energy_wh_to_joules(total_energy_wh)
    energy_eff_bits_per_j = total_data_bits / energy_joules if energy_joules > 0 else 0.0
    outage_prob = outages / total_intervals if total_intervals > 0 else 0.0
    
    return {
        "strategy": strategy,
        "total_energy_wh": total_energy_wh,
        "average_power_w": avg_power_w,
        "total_energy_joules": energy_joules,
        "total_data_bits": total_data_bits,
        "energy_efficiency_bits_per_joule": energy_eff_bits_per_j,
        "outage_count": outages,
        "total_intervals": total_intervals,
        "outage_probability": outage_prob,
        "mode_history": mode_history,
        "power_history": power_history
    }
