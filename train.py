"""
Training Pipeline Entry Point
Run: python train.py
"""

import numpy as np
import pandas as pd
import json
import os
from src.features import (
    load_data, build_features, get_feature_columns,
    scale_features
)
from src.model import (
    train_random_forest, predict_random_forest, evaluate,
    train_xgboost, predict_xgboost,
    train_lstm, predict_lstm,
    train_predict_prophet
)
from src.evaluate import compute_metrics, build_leaderboard, get_champion_model

DATA_PATH = "data/Daily Demand Forecasting Orders.csv"
RESULTS_PATH = "models/leaderboard.json"
TEST_SIZE = 0.2


def main():
    print("=" * 60)
    print("  Fintech Demand Forecasting — Training Pipeline")
    print("=" * 60)

    # ── Load & Feature Engineering ──
    print("\n[1/5] Loading and engineering features...")
    raw = load_data(DATA_PATH)
    df = build_features(raw)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df["demand"].values

    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    results = []

    # ── Random Forest ──
    print("[2/5] Training Random Forest...")
    train_random_forest(X_train, y_train)
    rf_preds = predict_random_forest(X_test)
    results.append(compute_metrics(y_test, rf_preds, "Random Forest"))
    print(f"      RMSE: {results[-1]['RMSE ↓']}")

    # ── XGBoost ──
    print("[3/5] Training XGBoost...")
    train_xgboost(X_train, y_train)
    xgb_preds = predict_xgboost(X_test)
    results.append(compute_metrics(y_test, xgb_preds, "XGBoost"))
    print(f"      RMSE: {results[-1]['RMSE ↓']}")

    # ── LSTM ──
    print("[4/5] Training LSTM...")
    try:
        train_lstm(X_train_s, y_train, window=14)
        lstm_preds, lstm_true = predict_lstm(X_test_s, y_test, window=14)
        if len(lstm_preds) > 0:
            results.append(compute_metrics(lstm_true, lstm_preds, "LSTM"))
            print(f"      RMSE: {results[-1]['RMSE ↓']}")
    except Exception as e:
        print(f"      LSTM skipped: {e}")

    # ── Prophet ──
    print("[5/5] Training Prophet...")
    try:
        p_true, p_preds, _, _ = train_predict_prophet(df, TEST_SIZE)
        results.append(compute_metrics(p_true, p_preds, "Prophet"))
        print(f"      RMSE: {results[-1]['RMSE ↓']}")
    except Exception as e:
        print(f"      Prophet skipped: {e}")

    # ── Leaderboard ──
    leaderboard = build_leaderboard(results)
    champion = get_champion_model(leaderboard)

    print("\n" + "=" * 60)
    print("  MODEL LEADERBOARD")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    print(f"\n  🏆 Champion Model: {champion}")

    # Save results
    leaderboard.to_json(RESULTS_PATH, orient="records", indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()