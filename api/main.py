"""
api/main.py — Secured FastAPI Production Serving Layer

WHY WE ADDED AUTHENTICATION:
Real fintech APIs never allow anyone to call them freely.
Every request must carry an API key. This is exactly how
Stripe, Razorpay, and Twilio work — you get an API key,
you include it in every request, otherwise you get blocked.

WHY WE LOG PREDICTIONS:
Every prediction made gets saved to a JSON log file.
In production, this goes to a database (PostgreSQL).
This is called an "audit trail" — companies need to know
what predictions were made, when, and what the inputs were.
If a model makes bad predictions, you can trace back why.

Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
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

app = FastAPI(
    title="Fintech Demand Forecasting API",
    description="""
    ## Secured ML API for Demand Forecasting & Dynamic Pricing
    
    ### Authentication
    All endpoints (except /health) require an API key.
    Include it in the request header: `X-API-Key: your-key-here`
    
    ### Default API Key (for development)
    `fintech-api-key-2025`
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

# ─────────────────────────────────────────
# API KEY AUTHENTICATION
#
# WHY: Without this, anyone can call your API.
# With this, only people with the key can use it.
# In production, you'd store keys in a database
# and give each customer their own unique key.
# ─────────────────────────────────────────

VALID_API_KEYS = {
    "fintech-api-key-2025": "default-user",
    "admin-key-9999": "admin",
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Checks if the API key in the request header is valid.
    If not, returns 403 Forbidden — same as what Stripe does.
    """
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Include 'X-API-Key: fintech-api-key-2025' in your request headers."
        )
    return VALID_API_KEYS[api_key]


# ─────────────────────────────────────────
# PREDICTION LOGGING
#
# WHY: We save every prediction to a log file.
# This tells us: what inputs came in, what the model predicted,
# what price was recommended, and when it happened.
# In production this would go to PostgreSQL.
# ─────────────────────────────────────────

def log_prediction(inputs: dict, prediction: dict):
    """Save prediction to a JSON log file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "inputs": inputs,
        "prediction": prediction,
    }
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(log_entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


# ─────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
#
# WHY PYDANTIC: Pydantic automatically validates
# every incoming request. If someone sends a string
# where a number is expected, it rejects it immediately
# with a clear error — before it even reaches your model.
# ─────────────────────────────────────────

class ForecastRequest(BaseModel):
    lag_1: float         = Field(..., description="Demand 1 day ago",        example=312.5)
    lag_3: float         = Field(..., description="Demand 3 days ago",       example=289.0)
    lag_7: float         = Field(..., description="Demand 7 days ago",       example=301.2)
    lag_14: float        = Field(..., description="Demand 14 days ago",      example=278.4)
    rolling_mean_7: float  = Field(..., description="7-day rolling average", example=295.3)
    rolling_std_7: float   = Field(..., description="7-day rolling std dev", example=18.2)
    rolling_mean_14: float = Field(..., description="14-day rolling average",example=291.0)
    rolling_max_7: float   = Field(..., description="7-day rolling max",     example=320.1)
    rolling_min_7: float   = Field(..., description="7-day rolling min",     example=270.5)
    momentum_3: float    = Field(..., description="3-day momentum (lag1-lag3)", example=23.5)
    momentum_7: float    = Field(..., description="7-day momentum (lag1-lag7)", example=11.3)
    week_of_month: int   = Field(..., ge=1, le=5,  description="Week of month (1-5)", example=2)
    is_week_start: int   = Field(0,   ge=0, le=1,  description="1 if first week",    example=0)
    is_week_end: int     = Field(0,   ge=0, le=1,  description="1 if last week",     example=0)
    model: Literal["random_forest", "xgboost"] = Field("xgboost", description="Model to use")


class ForecastResponse(BaseModel):
    predicted_demand: float
    model_used: str
    optimal_price: float
    projected_revenue: float
    confidence: str
    timestamp: str


class PricingRequest(BaseModel):
    base_demand: float  = Field(..., description="Forecasted demand", example=1000.0)
    base_price: float   = Field(100.0, description="Current base price")
    elasticity: float   = Field(0.4,   description="Price sensitivity (0.1=low, 0.8=high)")
    price_min: float    = Field(60.0,  description="Minimum price to consider")
    price_max: float    = Field(140.0, description="Maximum price to consider")


# ─────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────

def load_model_from_disk(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' not found. Run python train.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """Public endpoint — no auth needed. Shows API info."""
    return {
        "service": "Fintech Demand Forecasting API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "auth": "Include 'X-API-Key: fintech-api-key-2025' in headers"
    }


@app.get("/health", tags=["Info"])
def health():
    """
    Health check — used by Docker to verify the container is alive.
    Also shows which models are trained and ready.
    """
    models_available = []
    if os.path.exists(MODEL_DIR):
        models_available = [
            f.replace(".pkl", "")
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pkl")
        ]
    return {
        "status": "healthy",
        "models_ready": models_available,
        "predictions_logged": _count_logs()
    }


def _count_logs():
    if not os.path.exists(LOG_PATH):
        return 0
    with open(LOG_PATH) as f:
        try:
            return len(json.load(f))
        except Exception:
            return 0


@app.get("/leaderboard", tags=["Models"])
def get_leaderboard(user: str = Depends(verify_api_key)):
    """
    Returns the model comparison leaderboard.
    Shows RMSE, MAE, R², and composite score for all trained models.
    Requires API key.
    """
    path = os.path.join(MODEL_DIR, "leaderboard.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run python train.py first.")
    with open(path) as f:
        return {"leaderboard": json.load(f), "requested_by": user}


@app.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict(request: ForecastRequest, user: str = Depends(verify_api_key)):
    """
    Main prediction endpoint.

    Send yesterday's demand stats → get tomorrow's demand forecast + optimal price.

    WHY THIS IS USEFUL:
    A fintech company can call this every morning with the previous day's numbers.
    The API returns how much demand to expect + what price to charge today.
    No human needed — fully automated.
    """
    features = np.array([[
        request.lag_1, request.lag_3, request.lag_7, request.lag_14,
        request.rolling_mean_7, request.rolling_std_7, request.rolling_mean_14,
        request.rolling_max_7, request.rolling_min_7,
        request.momentum_3, request.momentum_7,
        request.week_of_month, request.is_week_start, request.is_week_end
    ]])

    model = load_model_from_disk(request.model)
    pred  = float(model.predict(features)[0])

    from src.pricing import find_optimal_price
    pricing = find_optimal_price(base_demand=pred)

    timestamp = datetime.now().isoformat()
    response_data = {
        "predicted_demand": round(pred, 2),
        "model_used": request.model,
        "optimal_price": pricing["optimal_price"],
        "projected_revenue": pricing["optimal_revenue"],
        "confidence": "high" if pred > 0 else "low",
        "timestamp": timestamp,
    }

    # Log the prediction
    log_prediction(inputs=request.dict(), prediction=response_data)

    return ForecastResponse(**response_data)


@app.post("/pricing/optimize", tags=["Pricing"])
def optimize_pricing(request: PricingRequest, user: str = Depends(verify_api_key)):
    """
    Given a demand forecast, returns the optimal price to maximize revenue.

    Uses price elasticity model:
    demand(price) = base_demand × (1 - elasticity × (price - base_price) / base_price)

    Higher elasticity = customers are more price-sensitive.
    """
    from src.pricing import find_optimal_price
    result = find_optimal_price(
        base_demand=request.base_demand,
        base_price=request.base_price,
        elasticity=request.elasticity,
        price_min=request.price_min,
        price_max=request.price_max,
    )
    return {
        "optimal_price": result["optimal_price"],
        "optimal_revenue": result["optimal_revenue"],
        "optimal_demand": result["optimal_demand"],
        "requested_by": user,
    }


@app.get("/pricing/segments", tags=["Pricing"])
def get_segment_pricing(base_demand: float = 1000.0, user: str = Depends(verify_api_key)):
    """
    Returns risk-tiered pricing for 3 borrower segments:
    Low Risk (prime), Medium Risk (near-prime), High Risk (subprime).
    This is how real fintech lenders price credit products differently
    for different customer segments.
    """
    from src.pricing import segment_pricing
    df = segment_pricing(base_demand)
    return {"segments": df.to_dict(orient="records"), "requested_by": user}


@app.get("/pricing/scenarios", tags=["Pricing"])
def get_scenarios(
    base_demand: float = 1000.0,
    base_price: float = 100.0,
    user: str = Depends(verify_api_key)
):
    """
    What-if analysis: how does revenue change under different market conditions?
    Conservative / Baseline / Aggressive / Crisis elasticity scenarios.
    """
    from src.pricing import scenario_analysis
    df = scenario_analysis(base_demand, base_price)
    return {"scenarios": df.to_dict(orient="records"), "requested_by": user}


@app.get("/logs", tags=["Monitoring"])
def get_prediction_logs(limit: int = 20, user: str = Depends(verify_api_key)):
    """
    Returns recent prediction logs.

    WHY THIS EXISTS:
    In production, you monitor every prediction your model makes.
    If predictions suddenly spike or drop, something is wrong.
    This endpoint lets you audit what the model has been predicting.
    """
    if not os.path.exists(LOG_PATH):
        return {"logs": [], "total": 0}
    with open(LOG_PATH) as f:
        try:
            logs = json.load(f)
        except Exception:
            logs = []
    return {
        "logs": logs[-limit:],
        "total_predictions": len(logs),
        "requested_by": user,
    }