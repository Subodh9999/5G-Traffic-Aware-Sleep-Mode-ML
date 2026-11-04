"""
dashboard.py — Interactive dashboard for
"Traffic-Aware Sleep Mode Optimization in 5G Small Cells Using Machine Learning"

Run:
    streamlit run dashboard.py

Deps:
    streamlit, plotly, pandas, numpy, scikit-learn
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure repository root is on sys.path for cloud deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------------------
# Page + constants
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="5G Sleep-Mode Optimizer", page_icon="📡", layout="wide")

MODE_COLORS = {"ON": "#e74c3c", "IDLE": "#f1c40f", "SLEEP": "#2ecc71"}
MODE_RANK = {"SLEEP": 0, "IDLE": 1, "ON": 2}
STRATEGY_LABELS = {
    "always_on": "Always ON (baseline)",
    "rule": "Rule-Based (reactive)",
    "ml": "ML Predictive (proposed)",
}

_PROFILE_X = np.array([0, 5, 6, 10, 13, 17, 18, 21, 23, 24], dtype=float)
_PROFILE_Y = np.array([40, 40, 90, 520, 540, 520, 400, 350, 150, 40], dtype=float)


# ----------------------------------------------------------------------------------
# Core simulation (same logic as src/energy_optimizer.py)
# ----------------------------------------------------------------------------------
def _raw_mode(demand: float, on_thr: float, idle_thr: float) -> str:
    if demand > on_thr:
        return "ON"
    if demand >= idle_thr:
        return "IDLE"
    return "SLEEP"


def _controller(demand, on_thr, idle_thr, margin, hyst, use_margin, use_hyst):
    """Maps a demand series -> array of modes using safety margin + hysteresis."""
    d_eff = np.asarray(demand, dtype=float) + (margin if use_margin else 0.0)
    modes = []
    prev = _raw_mode(d_eff[0], on_thr, idle_thr)
    for d in d_eff:
        if use_hyst:
            if prev == "ON":
                if d <= on_thr - hyst:
                    prev = "IDLE" if d >= idle_thr - hyst else "SLEEP"
            elif prev == "IDLE":
                if d >= on_thr + hyst:
                    prev = "ON"
                elif d <= idle_thr - hyst:
                    prev = "SLEEP"
            else:  # SLEEP
                if d >= on_thr + hyst:
                    prev = "ON"
                elif d >= idle_thr + hyst:
                    prev = "IDLE"
        else:
            prev = _raw_mode(d, on_thr, idle_thr)
        modes.append(prev)
    return np.array(modes)


def _evaluate(actual, modes, cfg):
    actual = np.asarray(actual, dtype=float)
    power = np.array([cfg["power"][m] for m in modes], dtype=float)
    cap = np.array([cfg["cap"][m] for m in modes], dtype=float)

    served = np.minimum(actual, cap)
    outage = actual > cap
    energy = power * cfg["dt"]
    bits = served * cfg["bits_per_user_hour"]

    total_energy = float(energy.sum())
    switches = int((np.asarray(modes)[1:] != np.asarray(modes)[:-1]).sum())

    return {
        "power": power,
        "energy": energy,
        "cum_energy": np.cumsum(energy),
        "outage": outage,
        "served": served,
        "bits": bits,
        "total_energy": total_energy,
        "avg_power": float(power.mean()),
        "ee": float(bits.sum() / (total_energy * 3600.0)) if total_energy > 0 else 0.0,
        "outage_pct": float(outage.mean() * 100.0),
        "switches": switches,
    }


@st.cache_data(show_spinner=False)
def run_pipeline(cfg_json: str):
    cfg = json.loads(cfg_json)
    days = int(cfg["days"])
    n = days * 24
    t = np.arange(n)
    hod = t % 24
    dow = (t // 24) % 7

    # --- 24h diurnal profile + weekday/weekend modulation -------------------------
    base = np.interp(hod, _PROFILE_X, _PROFILE_Y) * cfg["scale"]
    base = base * np.where(dow >= 5, cfg["weekend_factor"], 1.0)

    # --- measurement noise --------------------------------------------------------
    rng = np.random.default_rng(int(cfg["seed"]))
    actual = base + rng.normal(0.0, cfg["noise_frac"] * base + cfg["noise_abs"])
    actual = np.clip(actual, 0.0, None)

    # --- cyclic time features -----------------------------------------------------
    feats = np.column_stack(
        [np.sin(2 * np.pi * hod / 24.0), np.cos(2 * np.pi * hod / 24.0)]
    )

    n_train = int(np.clip(cfg["train_days"] * 24, 24, n - 24))

    # --- model training + chronological evaluation --------------------------------
    model_stats, preds = {}, {}
    for name, mdl in (
        ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
        ("Linear Regression", LinearRegression()),
    ):
        mdl.fit(feats[:n_train], actual[:n_train])
        p = np.clip(mdl.predict(feats), 0.0, None)
        preds[name] = p
        model_stats[name] = {
            "mae": float(mean_absolute_error(actual[n_train:], p[n_train:])),
            "rmse": float(np.sqrt(mean_squared_error(actual[n_train:], p[n_train:]))),
        }

    pred = preds[cfg["model"]]

    # --- controllers --------------------------------------------------------------
    modes_ml = _controller(
        pred, cfg["on_thr"], cfg["idle_thr"], cfg["margin"], cfg["hyst"],
        cfg["use_margin"], cfg["use_hyst"],
    )
    modes_rule = _controller(
        actual, cfg["on_thr"], cfg["idle_thr"], 0.0, cfg["hyst"],
        False, cfg["use_hyst"],
    )
    modes_on = np.array(["ON"] * n)

    df = pd.DataFrame(
        {
            "t": t,
            "hour": hod,
            "day": t // 24,
            "dow": dow,
            "actual": actual,
            "pred": pred,
            "mode_always_on": modes_on,
            "mode_rule": modes_rule,
            "mode_ml": modes_ml,
        }
    )

    # --- physical evaluation ------------------------------------------------------
    stats = {}
    for key, col in (
        ("always_on", "mode_always_on"),
        ("rule", "mode_rule"),
        ("ml", "mode_ml"),
    ):
        ev = _evaluate(df["actual"].to_numpy(), df[col].to_numpy(), cfg)
        df[f"power_{key}"] = ev["power"]
        df[f"cumenergy_{key}"] = ev["cum_energy"]
        df[f"outage_{key}"] = ev["outage"]
        stats[key] = {
            "total_energy": ev["total_energy"],
            "avg_power": ev["avg_power"],
            "ee": ev["ee"],
            "outage_pct": ev["outage_pct"],
            "switches": ev["switches"],
            "mode_hours": {m: int((df[col] == m).sum()) for m in ("ON", "IDLE", "SLEEP")},
        }

    return df, model_stats, stats, n_train


# ----------------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------------
def mode_badge(mode: str) -> str:
    c = MODE_COLORS[mode]
    return (
        f"<span style='background:{c};color:#111;padding:6px 16px;border-radius:10px;"
        f"font-weight:800;font-size:1.05rem;letter-spacing:.5px'>{mode}</span>"
    )


def base_layout(fig, height=420, **kw):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **kw,
    )
    return fig


# ----------------------------------------------------------------------------------
# Sidebar — every knob exposed
# ----------------------------------------------------------------------------------
st.sidebar.title("⚙️ Simulation Controls")

with st.sidebar.expander("📊 Traffic generator", expanded=True):
    days = st.slider("Days to simulate", 2, 14, 7)
    seed = st.number_input("Random seed", 0, 9999, 42, 1)
    scale = st.slider("Traffic scale (×)", 0.5, 1.5, 1.0, 0.05)
    weekend_factor = st.slider("Weekend factor (×)", 0.5, 1.2, 0.88, 0.01)
    noise_frac = st.slider("Relative noise (σ)", 0.0, 0.30, 0.05, 0.01)
    noise_abs = st.slider("Absolute noise (σ, users)", 0.0, 60.0, 15.0, 1.0)

with st.sidebar.expander("🤖 Model", expanded=True):
    model_name = st.selectbox("Predictor", ["Random Forest", "Linear Regression"])
    train_days = st.slider("Training days (chronological split)", 1, days - 1, min(5, days - 1))

with st.sidebar.expander("🛡️ Safe controller", expanded=True):
    on_thr = st.slider("ON threshold (users)", 100, 600, 300, 10)
    idle_thr = st.slider("IDLE threshold (users)", 50, on_thr - 10, min(150, on_thr - 10), 10)
    margin = st.slider("Safety margin (+ users)", 0, 80, 20, 5)
    hyst = st.slider("Hysteresis buffer (users)", 0, 80, 20, 5)
    use_margin = st.checkbox("Enable safety margin", value=True)
    use_hyst = st.checkbox("Enable hysteresis", value=True)

with st.sidebar.expander("🔌 Hardware profile"):
    p_on = st.number_input("ON power (W)", 50, 400, 120, 5)
    p_idle = st.number_input("IDLE power (W)", 10, 200, 70, 5)
    p_sleep = st.number_input("SLEEP power (W)", 1, 60, 15, 1)
    cap_on = st.number_input("ON capacity (users)", 100, 2000, 500, 25)
    cap_idle = st.number_input("IDLE capacity (users)", 50, 1000, 250, 25)
    cap_sleep = st.number_input("SLEEP capacity (users)", 10, 500, 100, 10)
    bits_per_user_hour = st.number_input(
        "Bits per user-hour", 1e5, 1e9, 1e7, 1e6, format="%.0f"
    )

cfg = {
    "days": int(days),
    "seed": int(seed),
    "scale": float(scale),
    "weekend_factor": float(weekend_factor),
    "noise_frac": float(noise_frac),
    "noise_abs": float(noise_abs),
    "model": model_name,
    "train_days": int(train_days),
    "on_thr": float(on_thr),
    "idle_thr": float(idle_thr),
    "margin": float(margin),
    "hyst": float(hyst),
    "use_margin": bool(use_margin),
    "use_hyst": bool(use_hyst),
    "power": {"ON": float(p_on), "IDLE": float(p_idle), "SLEEP": float(p_sleep)},
    "cap": {"ON": float(cap_on), "IDLE": float(cap_idle), "SLEEP": float(cap_sleep)},
    "bits_per_user_hour": float(bits_per_user_hour),
    "dt": 1.0,
}

df, model_stats, stats, n_train = run_pipeline(json.dumps(cfg))

# ----------------------------------------------------------------------------------
# Header + KPI strip
# ----------------------------------------------------------------------------------
st.title("📡 5G Small-Cell Sleep-Mode Optimizer")
st.caption(
    "Proactive ML-driven ON / IDLE / SLEEP controller — interactive simulation dashboard"
)

ml, rule, always = stats["ml"], stats["rule"], stats["always_on"]
savings = (1 - ml["total_energy"] / always["total_energy"]) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total energy (ML)", f"{ml['total_energy']:,.0f} Wh",
          f"{ml['total_energy'] - rule['total_energy']:+,.0f} vs rule")
k2.metric("Avg power (ML)", f"{ml['avg_power']:.1f} W",
          f"{ml['avg_power'] - always['avg_power']:+.1f} vs always-ON")
k3.metric("Energy saved vs always-ON", f"{savings:.1f} %")
k4.metric("Outage probability", f"{ml['outage_pct']:.2f} %",
          f"{ml['outage_pct'] - rule['outage_pct']:+.2f} vs rule", delta_color="inverse")
k5.metric("Mode switches", f"{ml['switches']}")

st.divider()

# ----------------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Traffic & Prediction", "🔀 Modes & Power", "▶️ Live Simulation",
     "🔋 Energy & KPIs", "🧪 Diagnostics"]
)

# ============================ TAB 1 — TRAFFIC & PREDICTION ========================
with tab1:
    c1, c2 = st.columns([3, 1])

    with c2:
        st.markdown("#### Model accuracy")
        mdf = pd.DataFrame(model_stats).T.reset_index().rename(columns={"index": "Model"})
        mdf["MAE"] = mdf["mae"].round(2)
        mdf["RMSE"] = mdf["rmse"].round(2)
        st.dataframe(mdf[["Model", "MAE", "RMSE"]], hide_index=True, use_container_width=True)
        st.caption(
            f"Trained on hours 0–{n_train} • evaluated on hours {n_train}–{len(df)} "
            "(held-out, no leakage)"
        )
        st.info(
            "Cyclic features `sin(2πh/24)`, `cos(2πh/24)` let the tree model learn "
            "that 23:00 is adjacent to 00:00."
        )

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["t"], y=df["actual"], name="Actual traffic",
                                 line=dict(color="#1f77b4", width=1.6)))
        fig.add_trace(go.Scatter(x=df["t"], y=df["pred"], name=f"{model_name} prediction",
                                 line=dict(color="#ff7f0e", width=1.6, dash="dash")))
        fig.add_hline(y=on_thr, line_dash="dot", line_color=MODE_COLORS["ON"],
                      annotation_text="ON threshold")
        fig.add_hline(y=idle_thr, line_dash="dot", line_color=MODE_COLORS["IDLE"],
                      annotation_text="IDLE threshold")
        fig.add_vrect(x0=n_train, x1=len(df) - 1, fillcolor="rgba(0,0,0,0.07)",
                      line_width=0, annotation_text="Held-out test", annotation_position="top left")
        # night shading
        for d in range(int(days)):
            fig.add_vrect(x0=d * 24, x1=d * 24 + 6, fillcolor="rgba(40,40,120,0.06)",
                          line_width=0)
        fig.update_yaxes(title="Active users")
        fig.update_xaxes(title="Hour index")
        st.plotly_chart(base_layout(fig), use_container_width=True)

# ============================ TAB 2 — MODES & POWER ==============================
with tab2:
    strat = st.radio(
        "Strategy", list(STRATEGY_LABELS.keys()),
        format_func=lambda k: STRATEGY_LABELS[k], horizontal=True, index=2,
    )
    col = f"mode_{strat}"

    fig = go.Figure()
    rank = df[col].map(MODE_RANK)
    fig.add_trace(go.Scatter(x=df["t"], y=rank, mode="lines", line_shape="hv",
                             line=dict(color="#999", width=1), showlegend=False))
    for m in ("ON", "IDLE", "SLEEP"):
        mask = df[col] == m
        fig.add_trace(go.Scatter(x=df.loc[mask, "t"], y=rank[mask], mode="markers",
                                 name=m, marker=dict(color=MODE_COLORS[m], size=7)))
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["SLEEP", "IDLE", "ON"], title="Mode")
    fig.update_xaxes(title="Hour index")
    st.plotly_chart(base_layout(fig, height=320), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        pw = go.Figure()
        pw.add_trace(go.Scatter(x=df["t"], y=df[f"power_{strat}"], line_shape="hv",
                                name="Power draw", line=dict(color="#8e44ad", width=1.6),
                                fill="tozeroy", fillcolor="rgba(142,68,173,0.15)"))
        pw.update_yaxes(title="Power (W)")
        pw.update_xaxes(title="Hour index")
        st.plotly_chart(base_layout(pw, height=300), use_container_width=True)

    with c2:
        mh = stats[strat]["mode_hours"]
        pie = go.Figure(go.Pie(
            labels=list(mh.keys()), values=list(mh.values()), hole=0.55,
            marker=dict(colors=[MODE_COLORS[k] for k in mh.keys()]),
        ))
        pie.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                          title="Hours spent per mode")
        st.plotly_chart(pie, use_container_width=True)

# ============================ TAB 3 — LIVE SIMULATION =============================
with tab3:
    st.subheader("🔴 Live Real-Time Simulation Engine")
    st.caption("A continuous ticking cellular controller simulating real-time hour-by-hour network traffic and proactive mode transitions.")
    if "sim_idx" not in st.session_state:
        st.session_state.sim_idx = 0
    if "sim_playing" not in st.session_state:
        st.session_state.sim_playing = False

    st.session_state.sim_idx = int(np.clip(st.session_state.sim_idx, 0, len(df) - 1))

    # Control bar
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.5, 1.1, 1.1, 1.8, 1.3])

    with ctrl1:
        if st.session_state.sim_playing:
            if st.button("⏸️ Pause Simulation", use_container_width=True, type="secondary"):
                st.session_state.sim_playing = False
                st.rerun()
        else:
            if st.button("▶️ Start Live Ticker", use_container_width=True, type="primary"):
                st.session_state.sim_playing = True
                st.rerun()

    with ctrl2:
        if st.button("⏭️ Step (+1h)", use_container_width=True):
            st.session_state.sim_playing = False
            st.session_state.sim_idx = (st.session_state.sim_idx + 1) % len(df)
            st.rerun()

    with ctrl3:
        if st.button("⏮️ Reset", use_container_width=True):
            st.session_state.sim_playing = False
            st.session_state.sim_idx = 0
            st.rerun()

    with ctrl4:
        sim_speed = st.select_slider(
            "Ticker speed",
            options=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            value=0.5,
            format_func=lambda s: f"{s}s / sim-hr",
        )

    with ctrl5:
        auto_loop = st.checkbox("Auto-loop", value=True)

    def on_scrub():
        st.session_state.sim_idx = st.session_state.scrubber_val

    st.slider(
        "Timeline scrubber (hour index)",
        min_value=0,
        max_value=len(df) - 1,
        value=st.session_state.sim_idx,
        key="scrubber_val",
        on_change=on_scrub,
    )

    def render_frame(i):
        r = df.iloc[i]
        mode = r["mode_ml"]
        is_outage = bool(r.get("outage_ml", False))

        a, b, c, d, e = st.columns(5)
        a.metric("Sim Clock", f"Day {int(r['day'])+1}, {int(r['hour']):02d}:00")
        b.metric("Demand", f"{r['actual']:.0f} users", f"Pred: {r['pred']:.0f}")
        c.metric("Power", f"{r['power_ml']:.0f} W", f"{r['power_ml'] - cfg['power']['ON']:+.0f} W vs ON")
        d.metric("Cumulative Energy", f"{r['cumenergy_ml']:,.0f} Wh")
        e.metric(
            "Outage Alert",
            "⚠️ EXCEEDED" if is_outage else "✅ NORMAL",
            delta_color="inverse" if is_outage else "normal",
        )

        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"<div style='text-align:center;padding:24px 10px;background:rgba(255,255,255,0.03);"
                f"border-radius:12px;margin-top:10px'>"
                f"<div style='font-size:.85rem;color:#888;margin-bottom:8px;letter-spacing:1px;font-weight:600'>"
                f"CURRENT CELL STATE</div>"
                f"{mode_badge(mode)}<br><br>"
                f"<div style='font-size:0.85rem;color:#aaa'>Capacity: <b>{cfg['cap'][mode]} users</b></div>"
                f"<div style='font-size:0.85rem;color:#aaa'>Power: <b>{cfg['power'][mode]:.0f} W</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(r["actual"]),
                number={"suffix": " users"},
                gauge={
                    "axis": {"range": [0, max(700, float(df['actual'].max()) * 1.1)]},
                    "bar": {"color": MODE_COLORS[mode]},
                    "steps": [
                        {"range": [0, idle_thr], "color": "rgba(46,204,113,.18)"},
                        {"range": [idle_thr, on_thr], "color": "rgba(241,196,15,.18)"},
                        {"range": [on_thr, max(700, float(df['actual'].max()) * 1.1)],
                         "color": "rgba(231,76,60,.18)"},
                    ],
                },
            ))
            gauge.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        # rolling window of the last 24h
        lo = max(0, i - 23)
        win = df.iloc[lo:i + 1]
        trail = go.Figure()
        trail.add_trace(go.Scatter(x=win["t"], y=win["actual"], name="Actual Demand",
                                   line=dict(color="#1f77b4", width=2)))
        trail.add_trace(go.Scatter(x=win["t"], y=win["pred"], name="Predicted Demand",
                                   line=dict(color="#ff7f0e", dash="dash", width=2)))
        trail.add_hline(y=on_thr, line_dash="dot", line_color=MODE_COLORS["ON"], annotation_text="ON threshold")
        trail.add_hline(y=idle_thr, line_dash="dot", line_color=MODE_COLORS["IDLE"], annotation_text="IDLE threshold")
        trail.add_vline(x=i, line_color="#fff", line_width=2, annotation_text="Current")
        trail.update_yaxes(title="Users", range=[0, max(700, float(df['actual'].max()) * 1.1)])
        trail.update_xaxes(title="Hour index")
        st.plotly_chart(base_layout(trail, height=260), use_container_width=True)

    render_frame(st.session_state.sim_idx)

    # Auto-advancing ticking loop
    if st.session_state.sim_playing:
        time.sleep(sim_speed)
        if st.session_state.sim_idx < len(df) - 1:
            st.session_state.sim_idx += 1
            st.rerun()
        elif auto_loop:
            st.session_state.sim_idx = 0
            st.rerun()
        else:
            st.session_state.sim_playing = False
            st.rerun()

# ============================ TAB 4 — ENERGY & KPIs ==============================
with tab4:
    c1, c2 = st.columns([3, 2])

    with c1:
        fig = go.Figure()
        for key, color in (("always_on", "#7f8c8d"), ("rule", "#3498db"), ("ml", "#e67e22")):
            fig.add_trace(go.Scatter(
                x=df["t"], y=df[f"cumenergy_{key}"], name=STRATEGY_LABELS[key],
                line=dict(color=color, width=2),
            ))
        fig.update_yaxes(title="Cumulative energy (Wh)")
        fig.update_xaxes(title="Hour index")
        st.plotly_chart(base_layout(fig, height=380), use_container_width=True)

    with c2:
        table = pd.DataFrame({
            "Strategy": [STRATEGY_LABELS[k] for k in stats],
            "Energy (Wh)": [stats[k]["total_energy"] for k in stats],
            "Avg power (W)": [stats[k]["avg_power"] for k in stats],
            "EE (bits/J)": [stats[k]["ee"] for k in stats],
            "Outage (%)": [stats[k]["outage_pct"] for k in stats],
            "Switches": [stats[k]["switches"] for k in stats],
        })
        st.dataframe(
            table.style.format({
                "Energy (Wh)": "{:,.0f}", "Avg power (W)": "{:.2f}",
                "EE (bits/J)": "{:,.0f}", "Outage (%)": "{:.2f}",
            }),
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "⬇️ Download raw results (CSV)",
            df.to_csv(index=False).encode(),
            file_name="5g_sleep_simulation.csv",
            mime="text/csv",
        )

    b1, b2 = st.columns(2)
    with b1:
        bar = go.Figure(go.Bar(
            x=[STRATEGY_LABELS[k] for k in stats],
            y=[stats[k]["total_energy"] for k in stats],
            marker_color=["#7f8c8d", "#3498db", "#e67e22"],
            text=[f"{stats[k]['total_energy']:,.0f} Wh" for k in stats],
            textposition="outside",
        ))
        bar.update_yaxes(title="Total energy (Wh)")
        st.plotly_chart(base_layout(bar, height=320), use_container_width=True)
    with b2:
        bar2 = go.Figure(go.Bar(
            x=[STRATEGY_LABELS[k] for k in stats],
            y=[stats[k]["outage_pct"] for k in stats],
            marker_color=["#7f8c8d", "#3498db", "#e67e22"],
            text=[f"{stats[k]['outage_pct']:.2f} %" for k in stats],
            textposition="outside",
        ))
        bar2.update_yaxes(title="Outage probability (%)")
        st.plotly_chart(base_layout(bar2, height=320), use_container_width=True)

# ============================ TAB 5 — DIAGNOSTICS ================================
with tab5:
    st.subheader("Safety margin trade-off sweep")
    st.caption(
        "Energy cost vs. outage risk as the safety margin varies. "
        "Hysteresis and thresholds are held at the sidebar values."
    )

    margins = list(range(0, 81, 5))
    sweep = []
    for m in margins:
        modes = _controller(df["pred"].to_numpy(), on_thr, idle_thr, m, hyst,
                            True, use_hyst)
        ev = _evaluate(df["actual"].to_numpy(), modes, cfg)
        sweep.append({"margin": m, "energy": ev["total_energy"],
                      "outage": ev["outage_pct"], "switches": ev["switches"]})
    sweep = pd.DataFrame(sweep)

    sw = go.Figure()
    sw.add_trace(go.Scatter(x=sweep["margin"], y=sweep["energy"], name="Energy (Wh)",
                            line=dict(color="#e67e22", width=2), yaxis="y1"))
    sw.add_trace(go.Scatter(x=sweep["margin"], y=sweep["outage"], name="Outage (%)",
                            line=dict(color="#e74c3c", width=2), yaxis="y2"))
    sw.add_vline(x=margin, line_dash="dash", line_color="#2c3e50")
    sw.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Safety margin (users)"),
        yaxis=dict(title="Energy (Wh)", color="#e67e22"),
        yaxis2=dict(title="Outage (%)", color="#e74c3c", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(sw, use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Where does the ML controller lose capacity?")
        out = df[df["outage_ml"]]
        if len(out):
            fig = go.Figure(go.Scatter(
                x=df["t"], y=df["actual"], name="Actual",
                line=dict(color="#1f77b4", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=out["t"], y=out["actual"], mode="markers", name="Outage interval",
                marker=dict(color="#e74c3c", size=6),
            ))
            fig.update_yaxes(title="Users")
            st.plotly_chart(base_layout(fig, height=320), use_container_width=True)
            st.caption(f"{len(out)} outage intervals out of {len(df)} "
                       f"({len(out)/len(df)*100:.2f} %).")
        else:
            st.success("No outages in this configuration 🎉")

    with c2:
        st.markdown("#### Mode decision matrix (ML)")
        bins = pd.cut(df["pred"], bins=[-1, idle_thr, on_thr, 1e9],
                      labels=["< IDLE thr", "IDLE band", "> ON thr"])
        matrix = pd.crosstab(bins, df["mode_ml"])
        st.dataframe(matrix, use_container_width=True)
        st.caption("Rows: predicted demand band — Columns: mode actually chosen "
                   "(difference = hysteresis + margin effect).")

    with st.expander("🔍 Full configuration used for this run"):
        st.json(cfg)
