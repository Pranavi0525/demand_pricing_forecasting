# 📈 Fintech Demand Forecasting & Dynamic Pricing

> An end-to-end machine learning system that predicts daily loan/product demand and automatically optimizes pricing — built for fintech companies like KreditBee, Slice, and LazyPay.

---

## 🧠 What Problem Does This Solve?

Fintech companies face two daily problems:

1. **Demand uncertainty** — How many loan applications will come tomorrow? If unprepared, servers crash and customers leave.
2. **Pricing decisions** — What interest rate should we charge today? Too high = customers leave. Too low = lost revenue.

This system solves both automatically using machine learning.

---

## 🏗️ System Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────┐
│  Feature        │  ← Lag features, rolling averages,
│  Engineering    │    momentum, calendar features
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Model Training Pipeline     │
│  ┌──────────┐  ┌──────────────────┐ │
│  │  Random  │  │    XGBoost       │ │
│  │  Forest  │  │  (Champion ✓)    │ │
│  └──────────┘  └──────────────────┘ │
│  ┌──────────┐  ┌──────────────────┐ │
│  │   LSTM   │  │    Prophet       │ │
│  └──────────┘  └──────────────────┘ │
└────────────────────┬────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Pricing Engine │    │  Model Leaderboard│
│  (Elasticity    │    │  (Auto-selects    │
│   Model)        │    │   best model)     │
└────────┬────────┘    └──────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
FastAPI     Streamlit
  API       Dashboard
(port 8000) (port 8501)
    │
    ▼
Docker Compose
(runs both together)
```

---

## 📊 Results

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Score |
|-------|--------|-------|------|-------|
| 🥇 XGBoost | 25.99 | 18.86 | 0.759 | 100 |
| 🥈 Random Forest | 30.63 | 26.66 | 0.665 | 86.4 |
| 🥉 Prophet | 83.53 | 61.42 | -1.49 | 3.7 |

**Champion Model: XGBoost** — best at capturing non-linear demand patterns in structured tabular data.

---

## 🗂️ Project Structure

```
demand-pricing-forecasting/
├── data/
│   └── Daily Demand Forecasting Orders.csv   ← raw data
├── src/
│   ├── features.py     ← feature engineering pipeline
│   ├── model.py        ← RF + XGBoost + LSTM + Prophet
│   ├── pricing.py      ← dynamic pricing engine
│   └── evaluate.py     ← metrics + model leaderboard
├── api/
│   └── main.py         ← FastAPI REST endpoints
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── app.py              ← Streamlit dashboard (5 pages)
├── train.py            ← training pipeline entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ How to Run

### Option 1 — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/demand-pricing-forecasting.git
cd demand-pricing-forecasting

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train all models (do this first!)
python train.py

# 5. Launch dashboard
streamlit run app.py

# 6. Launch API (new terminal)
uvicorn api.main:app --reload --port 8000
```

### Option 2 — Run with Docker (Recommended)

```bash
# Runs both FastAPI + Streamlit together
docker-compose up --build
```

Then visit:
- Dashboard → http://localhost:8501
- API Docs → http://localhost:8000/docs

---

## 🔌 API Usage

### Predict Demand + Optimal Price

```bash
POST http://localhost:8000/predict
```

**Request:**
```json
{
  "lag_1": 312.5,
  "lag_3": 289.0,
  "lag_7": 301.2,
  "lag_14": 278.4,
  "rolling_mean_7": 295.3,
  "rolling_std_7": 18.2,
  "rolling_mean_14": 291.0,
  "rolling_max_7": 320.1,
  "rolling_min_7": 270.5,
  "momentum_3": 23.5,
  "momentum_7": 11.3,
  "week_of_month": 2,
  "is_week_start": 0,
  "is_week_end": 0,
  "model": "xgboost"
}
```

**Response:**
```json
{
  "predicted_demand": 334.7,
  "model_used": "xgboost",
  "optimal_price": 106.5,
  "projected_revenue": 35645.55,
  "confidence": "high"
}
```

### Other Endpoints

| Method | Endpoint | What it does |
|--------|----------|-------------|
| GET | `/health` | Check if API is running |
| GET | `/leaderboard` | Get model comparison results |
| POST | `/pricing/optimize` | Get optimal price for any demand |
| GET | `/pricing/segments` | Risk-based pricing table |
| GET | `/pricing/scenarios` | What-if scenario analysis |

---

## 💡 Key Technical Decisions & Why

### Why XGBoost won?
XGBoost is a gradient boosting algorithm — it builds trees sequentially, each one learning from the mistakes of the previous one. For structured tabular data like daily orders, it consistently outperforms other models because it handles non-linear relationships, missing values, and feature interactions natively.

### Why Dynamic Pricing?
We use a **price elasticity model**: `demand(price) = base_demand × (1 - elasticity × (price - base_price) / base_price)`. This is the same mathematical model used by airlines, ride-sharing apps, and fintech lenders to set prices in real time.

### Why Docker?
So the project runs identically on any machine — your laptop, a teammate's laptop, or a cloud server — with zero setup. `docker-compose up` starts both services automatically.

### Why FastAPI over Flask?
FastAPI is async, faster, and auto-generates interactive API documentation at `/docs`. It's the industry standard for ML model serving in 2024-25.

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost, TensorFlow/Keras, Prophet |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |

---

## 👤 Author

**Pranavi Devulapalli**
B.Tech Data Science — Sai University