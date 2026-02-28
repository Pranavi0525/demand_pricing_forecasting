# 📈 Fintech Demand Forecasting & Dynamic Pricing

> Predicts daily loan demand and automatically recommends optimal pricing — built for fintech companies like KreditBee and Slice.

---

## Problem

Fintech companies need to know two things every morning:
1. How many loan applications are coming today?
2. What interest rate maximizes revenue without losing customers?

This system answers both automatically.

---

## Results

| Model | RMSE ↓ | R² ↑ | Rank |
|-------|--------|------|------|
| 🥇 XGBoost | 25.53 | 0.767 | 1 |
| 🥈 Random Forest | 30.63 | 0.665 | 2 |
| 🥉 Prophet | 83.53 | -1.49 | 3 |

XGBoost selected as champion model automatically.

---

## How to Run

```bash
# Local
pip install -r requirements.txt
python train.py
streamlit run app.py

# Docker
docker-compose up --build
# Dashboard → http://localhost:8501
# API       → http://localhost:8000/docs
```

---

## API

All requests require: `X-API-Key: fintech-api-key-2025`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Demand forecast + optimal price |
| GET | `/leaderboard` | Model comparison results |
| GET | `/pricing/segments` | Risk-based pricing tiers |
| GET | `/health` | API status |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML Models | XGBoost, Random Forest, Prophet |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker, GitHub Actions |
| Testing | Pytest (10 unit tests) |

---

## Structure

```
├── src/           ← features, models, pricing engine
├── api/           ← FastAPI endpoints
├── tests/         ← unit tests
├── app.py         ← Streamlit dashboard (7 pages)
├── train.py       ← training pipeline
├── Dockerfile
└── docker-compose.yml
```

---

**Pranavi Devulapalli** — B.Tech Data Science, Sai University