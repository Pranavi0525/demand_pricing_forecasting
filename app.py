"""
Fintech Demand Forecasting — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pickle

from src.features import load_data, build_features, get_feature_columns, scale_features
from src.pricing import find_optimal_price, segment_pricing, scenario_analysis
from src.evaluate import build_leaderboard

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fintech Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "rf": "#2563EB",
    "xgb": "#16A34A",
    "lstm": "#DC2626",
    "prophet": "#9333EA",
    "actual": "#F59E0B",
    "bg": "#0F172A",
}

DATA_PATH = "data/Daily Demand Forecasting Orders.csv"
MODEL_DIR = "models"


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
@st.cache_data
def get_processed_data():
    raw = load_data(DATA_PATH)
    df = build_features(raw)
    return df


@st.cache_data
def get_predictions(df):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["demand"].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    preds = {"Actual": y_test}

    def try_load(name, X):
        path = f"{MODEL_DIR}/{name}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f).predict(X)
        return None

    rf = try_load("random_forest", X_test)
    if rf is not None:
        preds["Random Forest"] = rf

    xgb = try_load("xgboost", X_test)
    if xgb is not None:
        preds["XGBoost"] = xgb

    try:
        from src.model import predict_lstm
        lstm_preds, lstm_true = predict_lstm(X_test_s, y_test, window=14)
        if len(lstm_preds) > 0:
            preds["LSTM"] = np.concatenate([np.full(14, np.nan), lstm_preds])
    except:
        pass

    return preds, y_test, split


def load_leaderboard():
    path = f"{MODEL_DIR}/leaderboard.json"
    if os.path.exists(path):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return None


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("📊 Fintech Demand\nForecasting")
    st.caption("Production ML Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Forecast", "💰 Pricing Engine", "🏆 Model Leaderboard", "🔬 Scenario Analysis"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Built by Pranavi Devulapalli")
    st.caption("B.Tech Data Science | Sai University")


# ─────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error("⚠️ Data file not found. Please add `Daily Demand Forecasting Orders.csv` to the `data/` folder.")
    st.stop()

df = get_processed_data()
preds, y_test, split = get_predictions(df)

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Fintech Demand Forecasting & Dynamic Pricing")
    st.caption("A production-grade ML platform for predicting loan/product demand and optimizing pricing strategies")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}", help="Training data size")
    with col2:
        st.metric("Avg Daily Demand", f"{df['demand'].mean():.0f}", help="Mean demand across all days")
    with col3:
        st.metric("Peak Demand", f"{df['demand'].max():.0f}", help="Maximum single-day demand")
    with col4:
        lb = load_leaderboard()
        champion = lb.iloc[0]["Model"] if lb is not None else "—"
        st.metric("Champion Model", champion)

    st.divider()

    # Demand trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df["demand"], mode="lines",
        line=dict(color=COLORS["actual"], width=1.5),
        name="Daily Demand", fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"
    ))
    fig.add_vline(x=split, line_dash="dash", line_color="white",
                  annotation_text="Train/Test Split", annotation_position="top right")
    fig.update_layout(
        title="Historical Demand — Full Dataset",
        xaxis_title="Day", yaxis_title="Demand",
        template="plotly_dark", height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(df, x="demand", nbins=30, title="Demand Distribution",
                            template="plotly_dark", color_discrete_sequence=["#2563EB"])
        fig2.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        rolling = df["demand"].rolling(7).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=df["demand"], name="Raw", line=dict(color="#94A3B8", width=1)))
        fig3.add_trace(go.Scatter(y=rolling, name="7-Day MA", line=dict(color="#F59E0B", width=2)))
        fig3.update_layout(title="Trend Smoothing", template="plotly_dark",
                           height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────
# PAGE 2: FORECAST
# ─────────────────────────────────────────
elif page == "📈 Forecast":
    st.title("Model Predictions vs Actuals")

    model_colors = {
        "Random Forest": COLORS["rf"],
        "XGBoost": COLORS["xgb"],
        "LSTM": COLORS["lstm"],
    }

    selected_models = st.multiselect(
        "Select models to compare:",
        [m for m in preds.keys() if m != "Actual"],
        default=[m for m in preds.keys() if m != "Actual"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=preds["Actual"], mode="lines",
        name="Actual", line=dict(color=COLORS["actual"], width=2)
    ))
    for model in selected_models:
        if model in preds:
            fig.add_trace(go.Scatter(
                y=preds[model], mode="lines",
                name=model, line=dict(color=model_colors.get(model, "#fff"), width=1.5)
            ))

    fig.update_layout(
        title="Test Set: Actual vs Predicted Demand",
        xaxis_title="Day (Test Set)", yaxis_title="Demand",
        template="plotly_dark", height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    if selected_models:
        model = selected_models[0]
        if model in preds:
            p = preds[model]
            a = preds["Actual"]
            min_len = min(len(p), len(a))
            residuals = a[:min_len] - p[:min_len]
            fig_r = go.Figure(go.Bar(y=residuals, marker_color=np.where(residuals > 0, "#16A34A", "#DC2626")))
            fig_r.update_layout(
                title=f"Residuals — {model}",
                template="plotly_dark", height=260,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ─────────────────────────────────────────
# PAGE 3: PRICING ENGINE
# ─────────────────────────────────────────
elif page == "💰 Pricing Engine":
    st.title("Dynamic Pricing Engine")
    st.caption("Simulates how fintech lenders (KreditBee, Slice, LazyPay) optimize credit product pricing based on demand forecasts")

    col1, col2, col3 = st.columns(3)
    with col1:
        base_demand = st.slider("Base Demand", 100, 5000, 1000, step=50)
    with col2:
        base_price = st.slider("Base Price / Rate (₹)", 60, 150, 100)
    with col3:
        elasticity = st.slider("Price Elasticity", 0.1, 1.0, 0.4, step=0.05,
                               help="Higher = demand more sensitive to price changes")

    result = find_optimal_price(base_demand, base_price, elasticity=elasticity)

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimal Price", f"₹{result['optimal_price']}")
    m2.metric("Projected Demand", f"{result['optimal_demand']:.0f} units")
    m3.metric("Max Revenue", f"₹{result['optimal_revenue']:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["prices"], y=result["revenues"],
        mode="lines", name="Revenue",
        line=dict(color="#2563EB", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.1)"
    ))
    fig.add_vline(
        x=result["optimal_price"], line_dash="dash", line_color="#F59E0B",
        annotation_text=f"Optimal ₹{result['optimal_price']}",
        annotation_font_color="#F59E0B"
    )
    fig.update_layout(
        title="Revenue vs Price Curve",
        xaxis_title="Price (₹)", yaxis_title="Revenue (₹)",
        template="plotly_dark", height=380,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment-Based Pricing")
    st.caption("Risk-tiered pricing across borrower segments")
    seg_df = segment_pricing(base_demand)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
# PAGE 4: LEADERBOARD
# ─────────────────────────────────────────
elif page == "🏆 Model Leaderboard":
    st.title("Model Leaderboard")
    st.caption("Champion model selected by composite score (RMSE 40% + MAE 30% + R² 30%)")

    lb = load_leaderboard()
    if lb is None:
        st.warning("No leaderboard found. Run `python train.py` first.")
    else:
        champion = lb.iloc[0]["Model"]
        st.success(f"🏆 Champion Model: **{champion}**")

        st.dataframe(
            lb[["Rank", "Model", "RMSE ↓", "MAE ↓", "R² ↑", "MAPE (%) ↓", "Score"]],
            use_container_width=True, hide_index=True
        )

        fig = go.Figure()
        colors = [COLORS["rf"], COLORS["xgb"], COLORS["lstm"], COLORS["prophet"]]
        for i, row in lb.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"], x=["RMSE", "MAE"],
                y=[row["RMSE ↓"], row["MAE ↓"]],
                marker_color=colors[i % len(colors)]
            ))
        fig.update_layout(
            barmode="group", title="RMSE & MAE Comparison",
            template="plotly_dark", height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# PAGE 5: SCENARIO ANALYSIS
# ─────────────────────────────────────────
elif page == "🔬 Scenario Analysis":
    st.title("Scenario Analysis")
    st.caption("What-if analysis across different market conditions and demand elasticity assumptions")

    base_demand = st.slider("Base Demand", 100, 5000, 1000, step=100)
    base_price = st.slider("Base Price (₹)", 60, 200, 100)

    sc_df = scenario_analysis(base_demand, base_price)
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=sc_df["Scenario"],
        y=sc_df["Max Revenue (₹)"],
        marker_color=["#16A34A", "#2563EB", "#F59E0B", "#DC2626"],
        text=sc_df["Max Revenue (₹)"].apply(lambda x: f"₹{x:,.0f}"),
        textposition="outside"
    ))
    fig.update_layout(
        title="Max Revenue by Scenario",
        template="plotly_dark", height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Revenue (₹)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Sensitivity Curves")
    from src.pricing import find_optimal_price
    scenarios = {"Conservative": 0.2, "Baseline": 0.4, "Aggressive": 0.6}
    fig2 = go.Figure()
    cols = ["#16A34A", "#2563EB", "#DC2626"]
    for (label, e), color in zip(scenarios.items(), cols):
        r = find_optimal_price(base_demand, base_price, elasticity=e)
        fig2.add_trace(go.Scatter(
            x=r["prices"], y=r["revenues"],
            name=f"{label} (e={e})",
            line=dict(color=color, width=2)
        ))
    fig2.update_layout(
        title="Revenue Curves Across Elasticity Scenarios",
        xaxis_title="Price (₹)", yaxis_title="Revenue (₹)",
        template="plotly_dark", height=360,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)