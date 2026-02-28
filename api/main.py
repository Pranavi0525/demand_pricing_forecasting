"""
api/main.py — Fintech Demand Forecasting REST API

Endpoints:
  GET  /health              — check API + which models are ready
  GET  /leaderboard         — model comparison results
  POST /predict             — demand forecast + optimal price
  POST /pricing/optimize    — optimal price for any demand
  GET  /pricing/segments    — risk-tiered pricing table
  GET  /pricing/scenarios   — what-if scenario analysis
  GET  /logs                — prediction audit trail

Auth: every endpoint (except /health) requires header:
  X-API-Key: fintech-api-key-2025

Run locally:
  uvicorn api.main:app --reload --port 8000
  Then open: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pickle
import json
import os
from datetime import datetime

# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="Fintech Demand Forecasting API",
    description="""
## Secured ML API for Demand Forecasting & Dynamic Pricing

Built for fintech companies (KreditBee, Slice, LazyPay style) to:
- Predict tomorrow's loan/product demand
- Get the optimal price/interest rate to charge
- Analyze pricing across borrower risk segments

### Authentication
All endpoints except `/health` require this header:
```
X-API-Key: fintech-api-key-2025
```

### Quick Test
Use the **Authorize** button above, enter `fintech-api-key-2025`, then try `/predict`.
""",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
LOG_PATH  = "models/prediction_log.json"

# ── API Key Auth ────────────────────────────────────────────────
VALID_API_KEYS = {
    "fintech-api-key-2025": "default-user",
    "admin-key-9999":       "admin",
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Add header: X-API-Key: fintech-api-key-2025"
        )
    return VALID_API_KEYS[api_key]

# ── Prediction logging ──────────────────────────────────────────
def log_prediction(inputs: dict, prediction: dict):
    """Save every prediction to a JSON log — audit trail."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "inputs":     inputs,
        "prediction": prediction,
    }
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

def _count_logs():
    if not os.path.exists(LOG_PATH):
        return 0
    with open(LOG_PATH) as f:
        try:
            return len(json.load(f))
        except Exception:
            return 0

# ── Model loading ───────────────────────────────────────────────
def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' not found. Run: python train.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Request / Response schemas ──────────────────────────────────
class PredictRequest(BaseModel):
    # ── Original dataset columns (11 features) ──────────────────
    week_of_month:       int   = Field(..., example=2,    description="Week of month (1-5)")
    day_of_week:         float = Field(..., example=3.0,  description="Day of week Mon-Fri (1-5)")
    non_urgent_order:    float = Field(..., example=50.0, description="Non-urgent order count")
    urgent_order:        float = Field(..., example=30.0, description="Urgent order count")
    order_type_a:        float = Field(..., example=20.0, description="Order type A count")
    order_type_b:        float = Field(..., example=35.0, description="Order type B count")
    order_type_c:        float = Field(..., example=25.0, description="Order type C count")
    fiscal_sector:       float = Field(..., example=40.0, description="Fiscal sector orders")
    traffic_controller:  float = Field(..., example=15.0, description="Traffic controller sector orders")
    banking_1:           float = Field(..., example=60.0, description="Banking orders (1)")
    banking_2:           float = Field(..., example=45.0, description="Banking orders (2)")
    banking_3:           float = Field(..., example=55.0, description="Banking orders (3)")
    # ── Lag & rolling features (13 features) ────────────────────
    lag_1:               float = Field(..., example=312.5, description="Demand 1 day ago")
    lag_3:               float = Field(..., example=289.0, description="Demand 3 days ago")
    lag_7:               float = Field(..., example=301.2, description="Demand 7 days ago")
    lag_14:              float = Field(..., example=278.4, description="Demand 14 days ago")
    rolling_mean_7:      float = Field(..., example=295.3, description="7-day rolling average")
    rolling_std_7:       float = Field(..., example=18.2,  description="7-day rolling std dev")
    rolling_mean_14:     float = Field(..., example=291.0, description="14-day rolling average")
    rolling_max_7:       float = Field(..., example=320.1, description="7-day rolling max")
    rolling_min_7:       float = Field(..., example=270.5, description="7-day rolling min")
    momentum_3:          float = Field(..., example=23.5,  description="lag_1 - lag_3")
    momentum_7:          float = Field(..., example=11.3,  description="lag_1 - lag_7")
    is_week_start:       int   = Field(0,   example=0,     description="1 if first week of month")
    is_week_end:         int   = Field(0,   example=0,     description="1 if last week of month")
    model: Literal["random_forest", "xgboost"] = Field(
        "xgboost", description="ML model to use for prediction"
    )

class PredictResponse(BaseModel):
    predicted_demand:  float
    model_used:        str
    optimal_price:     float
    projected_revenue: float
    confidence:        str
    timestamp:         str

class PricingRequest(BaseModel):
    base_demand: float = Field(...,   example=1000.0, description="Forecasted demand")
    base_price:  float = Field(100.0, example=100.0,  description="Current base price (₹)")
    elasticity:  float = Field(0.4,   example=0.4,    description="Price sensitivity (0.1=low, 0.8=high)")
    price_min:   float = Field(60.0,  example=60.0,   description="Minimum price to consider")
    price_max:   float = Field(140.0, example=140.0,  description="Maximum price to consider")

# ── Routes ──────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """Public — no auth needed."""
    return {
        "service": "Fintech Demand Forecasting API",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
        "auth":    "Add header: X-API-Key: fintech-api-key-2025",
    }


@app.get("/health", tags=["Info"])
def health():
    """
    Health check — Docker uses this to verify the container is alive.
    Also shows which models are trained and ready.
    """
    models_ready = []
    if os.path.exists(MODEL_DIR):
        models_ready = [
            f.replace(".pkl", "")
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pkl")
        ]
    return {
        "status":              "healthy",
        "models_ready":        models_ready,
        "predictions_logged":  _count_logs(),
    }


@app.get("/leaderboard", tags=["Models"])
def get_leaderboard(user: str = Depends(verify_api_key)):
    """
    Returns the model comparison leaderboard.
    Shows RMSE, MAE, R², MAPE, and composite score for all trained models.
    Champion model is ranked #1.
    """
    path = os.path.join(MODEL_DIR, "leaderboard.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run python train.py first.")
    with open(path) as f:
        return {"leaderboard": json.load(f), "requested_by": user}


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(req: PredictRequest, user: str = Depends(verify_api_key)):
    """
    **Main prediction endpoint.**

    Send today's order breakdown + demand statistics → get tomorrow's demand forecast
    + the optimal price to charge to maximise revenue.

    Feature order matches training exactly:
    week_of_month, day_of_week, non_urgent_order, urgent_order,
    order_type_a, order_type_b, order_type_c, fiscal_sector,
    traffic_controller, banking_1, banking_2, banking_3,
    lag_1, lag_3, lag_7, lag_14, rolling_mean_7, rolling_std_7,
    rolling_mean_14, rolling_max_7, rolling_min_7,
    momentum_3, momentum_7, is_week_start, is_week_end  →  25 total
    """
    # ── Build feature vector in exact training order ─────────────
    features = np.array([[
        req.week_of_month,
        req.day_of_week,
        req.non_urgent_order,
        req.urgent_order,
        req.order_type_a,
        req.order_type_b,
        req.order_type_c,
        req.fiscal_sector,
        req.traffic_controller,
        req.banking_1,
        req.banking_2,
        req.banking_3,
        req.lag_1,
        req.lag_3,
        req.lag_7,
        req.lag_14,
        req.rolling_mean_7,
        req.rolling_std_7,
        req.rolling_mean_14,
        req.rolling_max_7,
        req.rolling_min_7,
        req.momentum_3,
        req.momentum_7,
        req.is_week_start,
        req.is_week_end,
    ]])  # shape: (1, 25)

    model = load_model(req.model)
    pred  = float(model.predict(features)[0])

    from src.pricing import find_optimal_price
    pricing = find_optimal_price(base_demand=pred)

    response_data = {
        "predicted_demand":  round(pred, 2),
        "model_used":        req.model,
        "optimal_price":     pricing["optimal_price"],
        "projected_revenue": pricing["optimal_revenue"],
        "confidence":        "high" if pred > 0 else "low",
        "timestamp":         datetime.now().isoformat(),
    }

    log_prediction(inputs=req.dict(), prediction=response_data)
    return PredictResponse(**response_data)


@app.post("/pricing/optimize", tags=["Pricing"])
def optimize_pricing(req: PricingRequest, user: str = Depends(verify_api_key)):
    """
    Given a demand forecast, returns the optimal price to maximise revenue.

    Uses the price elasticity model:
    `demand(price) = base_demand × (1 - elasticity × (price - base_price) / base_price)`

    Higher elasticity = customers are more price-sensitive.
    """
    from src.pricing import find_optimal_price
    result = find_optimal_price(
        base_demand=req.base_demand,
        base_price=req.base_price,
        elasticity=req.elasticity,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    return {
        "optimal_price":   result["optimal_price"],
        "optimal_revenue": result["optimal_revenue"],
        "optimal_demand":  result["optimal_demand"],
        "requested_by":    user,
    }


@app.get("/pricing/segments", tags=["Pricing"])
def get_segment_pricing(
    base_demand: float = 1000.0,
    user: str = Depends(verify_api_key)
):
    """
    Risk-tiered pricing for 3 borrower segments:
    - **Low Risk** (prime borrowers) — low elasticity, lower rate
    - **Medium Risk** (near-prime) — baseline pricing
    - **High Risk** (subprime) — high elasticity, higher rate
    """
    from src.pricing import segment_pricing
    df = segment_pricing(base_demand)
    return {"segments": df.to_dict(orient="records"), "requested_by": user}


@app.get("/pricing/scenarios", tags=["Pricing"])
def get_scenarios(
    base_demand: float = 1000.0,
    base_price:  float = 100.0,
    user: str = Depends(verify_api_key)
):
    """
    What-if scenario analysis across 4 market conditions:
    Conservative / Baseline / Aggressive / Crisis

    Use this to stress-test your pricing strategy before going live.
    """
    from src.pricing import scenario_analysis
    df = scenario_analysis(base_demand, base_price)
    return {"scenarios": df.to_dict(orient="records"), "requested_by": user}


@app.get("/logs", tags=["Monitoring"])
def get_logs(limit: int = 20, user: str = Depends(verify_api_key)):
    """
    Returns recent prediction logs — the full audit trail.

    Every call to /predict is saved here with:
    - timestamp
    - input features
    - predicted demand
    - recommended price
    - revenue projection

    In production this would go to PostgreSQL.
    """
    if not os.path.exists(LOG_PATH):
        return {"logs": [], "total": 0}
    with open(LOG_PATH) as f:
        try:
            logs = json.load(f)
        except Exception:
            logs = []
    return {
        "logs":              logs[-limit:],
        "total_predictions": len(logs),
        "requested_by":      user,
    }
