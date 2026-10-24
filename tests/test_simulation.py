"""
Automated Unit and Integration Tests for 5G Traffic-Aware Sleep Mode Optimization.
Tests cover:
- Issue 1: Reproducibility and RNG controls.
- Issue 2: Chronological time-aware splitting, features, and model metrics.
- Issue 3: Power vs Energy units, Watt-hours conversion, Joules, and savings calculations.
- Issue 4: Safe sleep-mode decisions, boundary conditions, safety margins, and hysteresis.
- Issue 5: Fair evaluation, identical workloads, and objective outage metrics.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.energy_optimizer import (
    generate_traffic_data,
    add_cyclic_features,
    split_traffic_data,
    decide_cell_mode,
    decide_rule_based_mode,
    calculate_interval_energy,
    convert_energy_wh_to_joules,
    calculate_energy_savings,
    simulate_network_strategy,
    POWER_ON,
    POWER_IDLE,
    POWER_SLEEP,
    MAX_CAPACITY
)


# ==========================================
# Issue 1: Reproducibility & RNG Controls
# ==========================================

def test_reproducible_traffic_generation():
    """Verify that using the same seed produces identical datasets."""
    df1 = generate_traffic_data(days=7, hours=24, seed=42)
    df2 = generate_traffic_data(days=7, hours=24, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    """Verify that different seeds produce distinct noise patterns."""
    df1 = generate_traffic_data(days=7, hours=24, seed=42)
    df2 = generate_traffic_data(days=7, hours=24, seed=99)
    assert not df1["Users"].equals(df2["Users"])


# ==========================================
# Issue 2: Traffic Prediction & Splitting
# ==========================================

def test_chronological_split_avoids_leakage():
    """Verify that chronological splitting separates earlier days from later days."""
    df = generate_traffic_data(days=7, hours=24, seed=42)
    X_train, X_test, y_train, y_test = split_traffic_data(df, test_days=2, chronological=True)
    
    assert len(X_train) == 120  # Days 1 to 5: 5 * 24 = 120
    assert len(X_test) == 48    # Days 6 to 7: 2 * 24 = 48
    assert X_train["Day"].max() == 5
    assert X_test["Day"].min() == 6


def test_model_metrics_evaluation():
    """Verify calculation of MAE and RMSE on test split."""
    df = generate_traffic_data(days=7, hours=24, seed=42)
    X_train, X_test, y_train, y_test = split_traffic_data(df, test_days=2, chronological=True)
    
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = root_mean_squared_error(y_test, y_pred_lr)
    
    assert mae_lr > 0
    assert rmse_lr > mae_lr  # Mathematical property of RMSE vs MAE


# ==========================================
# Issue 3: Power and Energy Calculations
# ==========================================

def test_calculate_interval_energy():
    """Verify power (W) * duration (h) = energy (Wh)."""
    assert calculate_interval_energy(120.0, 1.0) == 120.0
    assert calculate_interval_energy(120.0, 0.5) == 60.0
    assert calculate_interval_energy(70.0, 2.0) == 140.0


def test_invalid_interval_duration():
    """Ensure non-positive interval duration raises ValueError."""
    with pytest.raises(ValueError):
        calculate_interval_energy(100.0, 0.0)
    with pytest.raises(ValueError):
        calculate_interval_energy(100.0, -1.0)


def test_energy_wh_to_joules():
    """Verify conversion: 1 Wh = 3600 Joules."""
    assert convert_energy_wh_to_joules(1.0) == 3600.0
    assert convert_energy_wh_to_joules(10.0) == 36000.0


def test_calculate_energy_savings():
    """Verify energy savings percentage formula."""
    # 100 Wh baseline, 80 Wh optimized -> 20% savings
    savings = calculate_energy_savings(100.0, 80.0)
    assert pytest.approx(savings, rel=1e-5) == 20.0
    
    # Baseline equal to optimized -> 0% savings
    assert calculate_energy_savings(100.0, 100.0) == 0.0
    
    # Zero baseline handling
    with pytest.raises(ValueError):
        calculate_energy_savings(0.0, 50.0)
    with pytest.raises(ValueError):
        calculate_energy_savings(-10.0, 50.0)


# ==========================================
# Issue 4: Safe Sleep-Mode Decisions
# ==========================================

def test_safe_mode_boundary_conditions():
    """Test decisions under negative, zero, and missing predictions."""
    # Negative prediction should be clamped to 0 -> SLEEP
    assert decide_cell_mode(-50.0) == "SLEEP"
    # Zero prediction -> SLEEP
    assert decide_cell_mode(0.0) == "SLEEP"
    # NaN prediction should fail-safe to ON
    assert decide_cell_mode(float("nan")) == "ON"


def test_safety_margin_prevents_underprovisioning():
    """Verify that safety margin promotes mode to ON when prediction is close to threshold."""
    # Without safety margin, 290 users is IDLE (threshold 300)
    assert decide_cell_mode(290.0, on_threshold=300.0, safety_margin=0.0) == "IDLE"
    # With safety margin of 20, 290 + 20 = 310 -> ON
    assert decide_cell_mode(290.0, on_threshold=300.0, safety_margin=20.0) == "ON"


def test_hysteresis_prevents_chattering():
    """Verify hysteresis avoids dropping to lower state immediately on slight dip."""
    # Currently ON, demand dips slightly below 300 to 290
    # With hysteresis 20, on_threshold - hysteresis = 280 -> stays ON
    mode = decide_cell_mode(290.0, on_threshold=300.0, current_mode="ON", hysteresis=20.0)
    assert mode == "ON"
    
    # If demand drops below 280 (e.g. 270), transitions to IDLE
    mode = decide_cell_mode(270.0, on_threshold=300.0, current_mode="ON", hysteresis=20.0)
    assert mode == "IDLE"


# ==========================================
# Issue 5: Fair Simulation & Outage Evaluation
# ==========================================

def test_fair_evaluation_same_workload():
    """Verify both strategies run against identical workloads."""
    df = generate_traffic_data(days=7, hours=24, seed=42)
    X_train, X_test, y_train, y_test = split_traffic_data(df, test_days=2, chronological=True)
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    
    test_df = df[df["Day"] > 5].copy()
    
    sim_rule = simulate_network_strategy(test_df, strategy="rule_based")
    sim_ml = simulate_network_strategy(test_df, strategy="ml", model=rf, safety_margin=20.0)
    
    # Both processed the exact same number of intervals and total throughput
    assert sim_rule["total_intervals"] == sim_ml["total_intervals"]
    assert sim_rule["total_data_bits"] == sim_ml["total_data_bits"]
    
    # Energy savings can be computed safely
    savings = calculate_energy_savings(sim_rule["total_energy_wh"], sim_ml["total_energy_wh"])
    assert isinstance(savings, float)
    assert -100.0 <= savings <= 100.0
