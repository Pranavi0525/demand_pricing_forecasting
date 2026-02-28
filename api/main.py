"""
api/main.py — Fintech Demand Forecasting REST API  (v3.0 — PostgreSQL edition)
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pickle
import json
import os
import io
import csv
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(
    title="Fintech Demand Forecasting API",
    description="""
## Secured ML API for Demand Forecasting & Dynamic Pricing

### Authentication
All endpoints except `/health` require this header:
```
X-API-Key: fintech-api-key-2025
```
""",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://admin:secret@db:5432/fintech"
)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    for attempt in range(15):
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS predictions (
                            id                 SERIAL PRIMARY KEY,
                            timestamp          TIMESTAMPTZ DEFAULT NOW(),
                            model_used         VARCHAR(50),
                            predicted_demand   FLOAT,
                            optimal_price      FLOAT,
                            projected_revenue  FLOAT,
                            confidence         VARCHAR(20),
                            inputs             JSONB
                        )
                    """)
                conn.commit()
            print("PostgreSQL ready.")
            return
        except Exception as e:
            print(f"Waiting for DB ({attempt+1}/15): {e}")
            time.sleep(3)
    raise RuntimeError("PostgreSQL unavailable after 15 attempts.")

@app.on_event("startup")
def startup():
    init_db()

VALID_API_KEYS = {
    "fintech-api-key-2025": "default-user",
    "admin-key-9999":       "admin",
}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Header: X-API-Key: fintech-api-key-2025"
        )
    return VALID_API_KEYS[api_key]

def log_prediction(inputs: dict, prediction: dict):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions
                    (model_used, predicted_demand, optimal_price,
                     projected_revenue, confidence, inputs)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                str(prediction["model_used"]),
                float(prediction["predicted_demand"]),
                float(prediction["optimal_price"]),
                float(prediction["projected_revenue"]),
                str(prediction["confidence"]),
                json.dumps(inputs, default=float),
            ))
        conn.commit()

def _count_logs():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM predictions")
                return cur.fetchone()["count"]
    except Exception:
        return 0

def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' not found. Run: python train.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)

class PredictRequest(BaseModel):
    week_of_month:      int   = Field(..., example=2)
    day_of_week:        float = Field(..., example=3.0)
    non_urgent_order:   float = Field(..., example=50.0)
    urgent_order:       float = Field(..., example=30.0)
    order_type_a:       float = Field(..., example=20.0)
    order_type_b:       float = Field(..., example=35.0)
    order_type_c:       float = Field(..., example=25.0)
    fiscal_sector:      float = Field(..., example=40.0)
    traffic_controller: float = Field(..., example=15.0)
    banking_1:          float = Field(..., example=60.0)
    banking_2:          float = Field(..., example=45.0)
    banking_3:          float = Field(..., example=55.0)
    lag_1:              float = Field(..., example=312.5)
    lag_3:              float = Field(..., example=289.0)
    lag_7:              float = Field(..., example=301.2)
    lag_14:             float = Field(..., example=278.4)
    rolling_mean_7:     float = Field(..., example=295.3)
    rolling_std_7:      float = Field(..., example=18.2)
    rolling_mean_14:    float = Field(..., example=291.0)
    rolling_max_7:      float = Field(..., example=320.1)
    rolling_min_7:      float = Field(..., example=270.5)
    momentum_3:         float = Field(..., example=23.5)
    momentum_7:         float = Field(..., example=11.3)
    is_week_start:      int   = Field(0,   example=0)
    is_week_end:        int   = Field(0,   example=0)
    model: Literal["random_forest", "xgboost"] = Field("xgboost")

class PredictResponse(BaseModel):
    predicted_demand:  float
    model_used:        str
    optimal_price:     float
    projected_revenue: float
    confidence:        str
    timestamp:         str

class PricingRequest(BaseModel):
    base_demand: float = Field(...,   example=1000.0)
    base_price:  float = Field(100.0, example=100.0)
    elasticity:  float = Field(0.4,   example=0.4)
    price_min:   float = Field(60.0,  example=60.0)
    price_max:   float = Field(140.0, example=140.0)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Fintech Demand Forecasting API",
        "version": "3.0.0",
        "docs":    "/docs",
        "auth":    "X-API-Key: fintech-api-key-2025",
    }


@app.get("/health", tags=["Info"])
def health():
    models_ready = []
    if os.path.exists(MODEL_DIR):
        models_ready = [
            f.replace(".pkl", "")
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pkl")
        ]
    return {
        "status":             "healthy",
        "models_ready":       models_ready,
        "predictions_logged": _count_logs(),
    }


@app.get("/leaderboard", tags=["Models"])
def get_leaderboard(user: str = Depends(verify_api_key)):
    path = os.path.join(MODEL_DIR, "leaderboard.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run python train.py first.")
    with open(path) as f:
        return {"leaderboard": json.load(f), "requested_by": user}


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(req: PredictRequest, user: str = Depends(verify_api_key)):
    """Forecast tomorrow's demand + optimal price. Auto-logged to PostgreSQL."""
    features = np.array([[
        req.week_of_month, req.day_of_week, req.non_urgent_order,
        req.urgent_order, req.order_type_a, req.order_type_b,
        req.order_type_c, req.fiscal_sector, req.traffic_controller,
        req.banking_1, req.banking_2, req.banking_3,
        req.lag_1, req.lag_3, req.lag_7, req.lag_14,
        req.rolling_mean_7, req.rolling_std_7, req.rolling_mean_14,
        req.rolling_max_7, req.rolling_min_7,
        req.momentum_3, req.momentum_7,
        req.is_week_start, req.is_week_end,
    ]])  # shape: (1, 25)

    model = load_model(req.model)
    pred  = float(model.predict(features)[0])

    from src.pricing import find_optimal_price
    pricing = find_optimal_price(base_demand=pred)

    # ── CRITICAL: cast all numpy types to plain Python float ────
    response_data = {
        "predicted_demand":  round(float(pred), 2),
        "model_used":        str(req.model),
        "optimal_price":     float(pricing["optimal_price"]),
        "projected_revenue": float(pricing["optimal_revenue"]),
        "confidence":        "high" if pred > 0 else "low",
        "timestamp":         datetime.now().isoformat(),
    }

    log_prediction(inputs=req.dict(), prediction=response_data)
    return PredictResponse(**response_data)


@app.post("/pricing/optimize", tags=["Pricing"])
def optimize_pricing(req: PricingRequest, user: str = Depends(verify_api_key)):
    from src.pricing import find_optimal_price
    result = find_optimal_price(
        base_demand=req.base_demand,
        base_price=req.base_price,
        elasticity=req.elasticity,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    return {
        "optimal_price":   float(result["optimal_price"]),
        "optimal_revenue": float(result["optimal_revenue"]),
        "optimal_demand":  float(result["optimal_demand"]),
        "requested_by":    user,
    }


@app.get("/pricing/segments", tags=["Pricing"])
def get_segment_pricing(base_demand: float = 1000.0, user: str = Depends(verify_api_key)):
    from src.pricing import segment_pricing
    df = segment_pricing(base_demand)
    return {"segments": df.to_dict(orient="records"), "requested_by": user}


@app.get("/pricing/scenarios", tags=["Pricing"])
def get_scenarios(
    base_demand: float = 1000.0,
    base_price:  float = 100.0,
    user: str = Depends(verify_api_key)
):
    from src.pricing import scenario_analysis
    df = scenario_analysis(base_demand, base_price)
    return {"scenarios": df.to_dict(orient="records"), "requested_by": user}


@app.get("/logs", tags=["Monitoring"])
def get_logs(limit: int = 20, user: str = Depends(verify_api_key)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) AS count FROM predictions")
            total = cur.fetchone()["count"]
    return {
        "logs":              [dict(r) for r in rows],
        "total_predictions": total,
        "requested_by":      user,
    }


@app.get("/logs/export/csv", tags=["Monitoring"])
def export_logs_csv(user: str = Depends(verify_api_key)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No predictions logged yet.")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    output.seek(0)

    filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )