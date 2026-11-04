"""
Automated unit tests for the live real-time simulation engine and state controller.
Verifies hour-by-hour continuous ticking, hysteresis transitions, and live state updates.
"""

import pytest
import numpy as np
import pandas as pd
from src.energy_optimizer import (
    decide_cell_mode,
    calculate_interval_energy,
    convert_energy_wh_to_joules,
    POWER_ON,
    POWER_IDLE,
    POWER_SLEEP
)


def test_live_diurnal_24h_cycle():
    """
    Simulate a full 24-hour live ticking cycle and verify expected mode transitions.
    """
    # 24-hour demand profile: night -> morning -> midday peak -> evening -> night
    hourly_demands = [
        35, 30, 25, 28, 40, 80,      # 00:00 - 05:00: deep night (SLEEP expected)
        160, 210, 260, 290,          # 06:00 - 09:00: morning ramp (IDLE expected)
        340, 480, 520, 510, 490, 420,# 10:00 - 15:00: peak hours (ON expected)
        330, 280, 220, 180,          # 16:00 - 19:00: evening shoulder (hysteresis keeps ON then IDLE)
        140, 90, 60, 35              # 20:00 - 23:00: drop to night (SLEEP expected)
    ]
    
    current_mode = "SLEEP"
    mode_history = []
    
    for demand in hourly_demands:
        current_mode = decide_cell_mode(
            predicted_users=demand,
            on_threshold=300.0,
            idle_threshold=150.0,
            safety_margin=20.0,
            current_mode=current_mode,
            hysteresis=20.0
        )
        mode_history.append(current_mode)
        
    # Night hours should be in SLEEP
    assert mode_history[0] == "SLEEP"
    assert mode_history[2] == "SLEEP"
    
    # Peak midday hours (10:00 - 15:00) should be ON
    assert mode_history[10] == "ON"
    assert mode_history[12] == "ON"
    
    # End of day should safely transition back to SLEEP
    assert mode_history[-1] == "SLEEP"


def test_live_loop_step_energy_tracking():
    """
    Verify live-step energy calculations match physical accumulation.
    """
    modes = ["SLEEP", "IDLE", "ON", "ON"]
    dt_hours = 1.0
    cumulative_wh = 0.0
    
    power_map = {"ON": POWER_ON, "IDLE": POWER_IDLE, "SLEEP": POWER_SLEEP}
    
    for m in modes:
        p = power_map[m]
        step_wh = calculate_interval_energy(p, dt_hours)
        cumulative_wh += step_wh
        
    expected_wh = POWER_SLEEP + POWER_IDLE + POWER_ON + POWER_ON
    assert cumulative_wh == pytest.approx(expected_wh)
    assert convert_energy_wh_to_joules(cumulative_wh) == pytest.approx(expected_wh * 3600.0)

