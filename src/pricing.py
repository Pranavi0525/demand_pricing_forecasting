"""
Dynamic Pricing Engine
Simulates fintech-grade dynamic pricing: interest rate / credit limit optimization
based on demand forecasts and price elasticity models.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────
# Core Pricing Model
# ─────────────────────────────────────────
def simulate_demand_at_price(
    base_demand: float,
    base_price: float,
    price_range: np.ndarray,
    elasticity: float = 0.4
) -> np.ndarray:
    """
    Linearized price elasticity model:
    d(p) = d_base * (1 - e * (p - p_base) / p_base)
    
    In fintech context:
    - price = interest rate / fee / credit cost
    - demand = loan applications / credit uptake
    """
    return base_demand * (1 - elasticity * (price_range - base_price) / base_price)


def compute_revenue(prices: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Revenue = Price * Demand (simulate total portfolio value)"""
    return prices * demands


def find_optimal_price(
    base_demand: float,
    base_price: float = 100.0,
    price_min: float = 60.0,
    price_max: float = 140.0,
    elasticity: float = 0.4,
    n_points: int = 200
) -> dict:
    """
    Returns optimal price, max revenue, and full curve for visualization.
    """
    prices = np.linspace(price_min, price_max, n_points)
    demands = simulate_demand_at_price(base_demand, base_price, prices, elasticity)
    demands = np.clip(demands, 0, None)  # demand can't go negative
    revenues = compute_revenue(prices, demands)

    optimal_idx = np.argmax(revenues)
    return {
        "prices": prices,
        "demands": demands,
        "revenues": revenues,
        "optimal_price": round(prices[optimal_idx], 2),
        "optimal_revenue": round(revenues[optimal_idx], 2),
        "optimal_demand": round(demands[optimal_idx], 2),
    }


# ─────────────────────────────────────────
# Segment-Based Pricing (Fintech Extension)
# ─────────────────────────────────────────
RISK_SEGMENTS = {
    "Low Risk": {"elasticity": 0.25, "base_price": 85, "label": "Prime Borrowers"},
    "Medium Risk": {"elasticity": 0.40, "base_price": 100, "label": "Near-Prime Borrowers"},
    "High Risk": {"elasticity": 0.60, "base_price": 130, "label": "Subprime Borrowers"},
}


def segment_pricing(base_demand: float) -> pd.DataFrame:
    """
    Returns optimal pricing per borrower risk segment.
    Simulates how fintechs (KreditBee, Slice) price credit products.
    """
    rows = []
    segment_demand = base_demand / len(RISK_SEGMENTS)

    for segment, params in RISK_SEGMENTS.items():
        result = find_optimal_price(
            base_demand=segment_demand,
            base_price=params["base_price"],
            elasticity=params["elasticity"],
        )
        rows.append({
            "Segment": segment,
            "Borrower Type": params["label"],
            "Optimal Price (₹)": result["optimal_price"],
            "Expected Demand": round(result["optimal_demand"], 0),
            "Projected Revenue (₹)": round(result["optimal_revenue"], 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Scenario Analysis
# ─────────────────────────────────────────
def scenario_analysis(base_demand: float, base_price: float = 100.0) -> pd.DataFrame:
    """
    What-if analysis across different elasticity assumptions.
    """
    scenarios = {
        "Conservative (Low Sensitivity)": 0.2,
        "Baseline": 0.4,
        "Aggressive (High Sensitivity)": 0.6,
        "Crisis Mode": 0.8,
    }
    rows = []
    for label, e in scenarios.items():
        result = find_optimal_price(base_demand, base_price, elasticity=e)
        rows.append({
            "Scenario": label,
            "Elasticity": e,
            "Optimal Price (₹)": result["optimal_price"],
            "Max Revenue (₹)": result["optimal_revenue"],
        })
    return pd.DataFrame(rows)