"""
Generate high-resolution visual report images for the Audit & Verification Summary.
Produces:
1. audit_verification_table.png - High quality visual table matching the audit summary
2. audit_verification_dashboard.png - 5-issue benchmark & verification visual dashboard
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.energy_optimizer import (
    generate_traffic_data,
    split_traffic_data,
    simulate_network_strategy,
    POWER_ON,
    POWER_IDLE,
    POWER_SLEEP
)

ARTIFACT_DIR = r"C:\Users\yashn\.gemini\antigravity-ide\brain\32477a53-c7d7-44a2-b5fa-d418ec04eb70"


def generate_table_image():
    fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
    ax.axis('off')
    
    # Title
    fig.text(0.06, 0.93, "Audit & Verification Summary", fontsize=20, fontweight='bold', color='#111827', family='sans-serif')
    
    # Table headers and contents
    headers = ["Issue", "Verification Status", "Root Cause & Resolution"]
    
    rows = [
        [
            "1. Random Data\nGeneration &\nReproducibility",
            "CONFIRMED\n& FIXED",
            "Generation and simulation used unseeded np.random. Replaced with\nexplicit np.random.default_rng(42) for deterministic reproducibility\nacross runs."
        ],
        [
            "2. Traffic Prediction\nAccuracy",
            "CONFIRMED\n& FIXED",
            "Used random train_test_split (data leakage) and underfitting Linear\nRegression (MAE ~46.31). Implemented chronological splitting (Days 1-5\ntrain, Days 6-7 test) and added Random Forest Regressor\n(MAE ~17.89, RMSE ~23.57)."
        ],
        [
            "3. Power vs Energy\nCalculations",
            "CONFIRMED\n& FIXED",
            "Hourly power was summed and incorrectly labeled as 'Power in Watts'.\nCorrected to cumulative energy in Watt-hours (E = P x dt), average\npower in Watts, and conversion to Joules (1 Wh = 3600 J) for bits/Joule\nefficiency."
        ],
        [
            "4. Unsafe Sleep-\nMode Decisions",
            "CONFIRMED\n& FIXED",
            "Under-predicted traffic caused cells to enter IDLE during evening peak\ndemand. Introduced input validation, a configurable safety margin (+20\nusers), and hysteresis to prevent oscillation."
        ],
        [
            "5. Fair Evaluation &\nOutages",
            "CONFIRMED\n& FIXED",
            "Outages were previously evaluated against predicted users rather than\nactual demand, and random failure rates were unequal (10% vs 5%).\nStandardized to identical workloads and fair demand-based outage\nevaluation."
        ]
    ]

    # Layout geometry
    y_start = 0.86
    row_heights = [0.125, 0.145, 0.145, 0.135, 0.145]
    col_widths = [0.22, 0.16, 0.54]
    x_positions = [0.06, 0.28, 0.44]

    # Header row background
    header_rect = patches.FancyBboxPatch(
        (0.05, y_start), 0.90, 0.055,
        boxstyle="square,pad=0",
        facecolor="#F3F4F6", edgecolor="#E5E7EB", linewidth=1.5
    )
    ax.add_patch(header_rect)

    for i, h in enumerate(headers):
        ha = 'center' if i == 1 else 'left'
        x_pos = x_positions[i] + (col_widths[i]/2 if i == 1 else 0.015)
        ax.text(x_pos, y_start + 0.027, h, fontsize=12, fontweight='bold', color='#374151', va='center', ha=ha)

    # Content rows
    curr_y = y_start
    for r_idx, (row, r_h) in enumerate(zip(rows, row_heights)):
        curr_y -= r_h
        bg_color = "#FFFFFF" if r_idx % 2 == 0 else "#F9FAFB"
        
        row_rect = patches.FancyBboxPatch(
            (0.05, curr_y), 0.90, r_h,
            boxstyle="square,pad=0",
            facecolor=bg_color, edgecolor="#E5E7EB", linewidth=1.0
        )
        ax.add_patch(row_rect)

        # Issue title
        ax.text(x_positions[0] + 0.015, curr_y + r_h / 2, row[0], fontsize=10.5, fontweight='bold', color='#1F2937', va='center')

        # Status Badge (Pill)
        badge_w, badge_h = 0.13, r_h * 0.55
        badge_x = x_positions[1] + (col_widths[1] - badge_w) / 2
        badge_y = curr_y + (r_h - badge_h) / 2
        
        badge_rect = patches.FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.2
        )
        ax.add_patch(badge_rect)
        ax.text(badge_x + badge_w / 2, badge_y + badge_h / 2, row[1], fontsize=9.5, fontweight='bold', color="#047857", ha='center', va='center')

        # Resolution details
        ax.text(x_positions[2] + 0.015, curr_y + r_h / 2, row[2], fontsize=9.5, color='#4B5563', va='center', family='sans-serif', linespacing=1.35)

    plt.tight_layout()
    out_path = "audit_verification_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def generate_dashboard_image():
    # Generate data
    df = generate_traffic_data(days=7, hours=24, seed=42)
    X_train, X_test, y_train, y_test = split_traffic_data(df, test_days=2, chronological=True)
    
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    
    sim_rule = simulate_network_strategy(df, strategy="rule_based", seed=42)
    sim_ml = simulate_network_strategy(df, strategy="ml", model=rf, seed=42)

    fig = plt.figure(figsize=(16, 12), dpi=180)
    fig.suptitle("5G Small Cell Traffic-Aware Sleep Mode: Audit & Verification Dashboard", fontsize=18, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, top=0.92, bottom=0.06, left=0.07, right=0.95)

    # 1. Issue 1: Traffic Pattern & Reproducibility
    ax1 = fig.add_subplot(gs[0, 0])
    df_seed99 = generate_traffic_data(days=7, hours=24, seed=99)
    hours_24 = range(24)
    ax1.plot(hours_24, df.iloc[0:24]["Users"], label="Seed 42 (Reproducible)", color="#2563EB", lw=2.2)
    ax1.plot(hours_24, df_seed99.iloc[0:24]["Users"], label="Seed 99 (Noise Shift)", color="#F59E0B", lw=1.8, linestyle='--')
    ax1.set_title("Issue 1: Controlled RNG & Diurnal Profile (24h)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Active Users")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Issue 2: ML Model Accuracy (Held-out Test Days 6-7)
    ax2 = fig.add_subplot(gs[0, 1])
    test_slice = slice(0, 48)
    time_pts = range(48)
    ax2.plot(time_pts, y_test.values[test_slice], label="Actual Traffic", color="#111827", lw=1.5, alpha=0.7)
    ax2.plot(time_pts, y_pred_rf[test_slice], label="Random Forest (MAE: 17.89)", color="#10B981", lw=2)
    ax2.plot(time_pts, y_pred_lr[test_slice], label="Linear Reg (MAE: 46.31)", color="#EF4444", lw=1.5, linestyle=":")
    ax2.set_title("Issue 2: Time-Series Test Predictions (Days 6-7)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Test Interval (Hours)")
    ax2.set_ylabel("Predicted Users")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Issue 3: Energy vs Power
    ax3 = fig.add_subplot(gs[1, 0])
    labels_eng = ["Baseline (Rule-Based)", "Safe ML System"]
    energy_vals = [sim_rule["total_energy_wh"], sim_ml["total_energy_wh"]]
    bars3 = ax3.bar(labels_eng, energy_vals, color=["#3B82F6", "#10B981"], width=0.45)
    ax3.set_title("Issue 3: Cumulative Energy Consumption (7 Days)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Energy (Watt-hours, Wh)")
    ax3.set_ylim(0, 16000)
    for bar in bars3:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 350, f"{bar.get_height():,.1f} Wh", ha='center', fontweight='bold', fontsize=10)
    ax3.grid(axis='y', linestyle=":", alpha=0.6)

    # 4. Issue 4: Safe Decision Boundaries & Transitions
    ax4 = fig.add_subplot(gs[1, 1])
    user_range = np.linspace(0, 450, 400)
    # Decisions without safety margin
    modes_clean = [("ON" if u > 300 else ("IDLE" if u > 150 else "SLEEP")) for u in user_range]
    # Decisions with +20 safety margin
    modes_safe = [("ON" if (u + 20) > 300 else ("IDLE" if (u + 20) > 150 else "SLEEP")) for u in user_range]
    
    mode_map = {"SLEEP": 15, "IDLE": 70, "ON": 120}
    ax4.plot(user_range, [mode_map[m] for m in modes_clean], label="Unbuffered Mode (Under-provision risk)", color="#EF4444", lw=2, linestyle='--')
    ax4.plot(user_range, [mode_map[m] for m in modes_safe], label="Safe Controller (+20 User Margin)", color="#10B981", lw=2.2)
    ax4.axvline(300, color='gray', linestyle=':', label="Nominal 300 Threshold")
    ax4.set_title("Issue 4: Safe Controller Boundary (Power vs Demand)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Traffic Demand (Users)")
    ax4.set_ylabel("Operating Power (Watts)")
    ax4.legend(loc="upper left", fontsize=8.5)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # 5. Issue 5: Fair Benchmark Metrics
    ax5 = fig.add_subplot(gs[2, :])
    metrics = ["Energy Efficiency (bits/Joule)", "Outage Probability (%)", "Average Power (Watts)"]
    rule_vals = [sim_rule["energy_efficiency_bits_per_joule"] / 100, sim_rule["outage_probability"] * 100, sim_rule["average_power_w"]]
    ml_vals = [sim_ml["energy_efficiency_bits_per_joule"] / 100, sim_ml["outage_probability"] * 100, sim_ml["average_power_w"]]
    
    x = np.arange(len(metrics))
    width = 0.32
    
    b1 = ax5.bar(x - width/2, rule_vals, width, label="Rule-Based (Old)", color="#3B82F6")
    b2 = ax5.bar(x + width/2, ml_vals, width, label="ML Predictive (Safe)", color="#10B981")
    
    ax5.set_xticks(x)
    ax5.set_xticklabels([
        "Energy Efficiency (x100 bits/J)\n[10,211.98 vs 9,698.85]",
        "Outage Probability (%)\n[29.17% vs 29.76%]",
        "Average Power (Watts)\n[74.26 W vs 78.18 W]"
    ], fontsize=10.5, fontweight='bold')
    ax5.set_title("Issue 5: Fair and Objective 7-Day Performance Evaluation", fontsize=12, fontweight='bold')
    ax5.legend(loc="upper right", fontsize=10)
    ax5.grid(axis='y', linestyle=":", alpha=0.6)
    
    for bar in b1:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.2f}", ha='center', fontsize=9.5, fontweight='bold')
    for bar in b2:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.2f}", ha='center', fontsize=9.5, fontweight='bold')

    out_path = "audit_verification_dashboard.png"
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    t_path = generate_table_image()
    d_path = generate_dashboard_image()
    
    # Copy both to artifacts directory so they can be embedded in markdown
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    shutil.copy2(t_path, os.path.join(ARTIFACT_DIR, t_path))
    shutil.copy2(d_path, os.path.join(ARTIFACT_DIR, d_path))
    print("Copied images to artifact directory!")
