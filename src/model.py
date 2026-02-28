"""
Model Training Module
Trains and compares: Random Forest, XGBoost, LSTM, Prophet
"""

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.features import prepare_lstm_sequences

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────
def evaluate(y_true, y_pred, model_name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return {
        "Model": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }


# ─────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    with open(f"{MODEL_DIR}/random_forest.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def predict_random_forest(X_test):
    with open(f"{MODEL_DIR}/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(X_test)


def get_feature_importance(feature_names: list) -> pd.DataFrame:
    with open(f"{MODEL_DIR}/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    importance = model.feature_importances_
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(10)


# ─────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────
def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    with open(f"{MODEL_DIR}/xgboost.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def predict_xgboost(X_test):
    with open(f"{MODEL_DIR}/xgboost.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(X_test)


# ─────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────
def train_lstm(X_train_scaled, y_train, window: int = 14):
    X_seq, y_seq = prepare_lstm_sequences(X_train_scaled, y_train, window)
    if len(X_seq) == 0:
        raise ValueError("Not enough data for LSTM sequences with given window.")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(X_seq, y_seq, epochs=80, batch_size=16, callbacks=[early_stop], verbose=0)
    model.save(f"{MODEL_DIR}/lstm_model.keras")
    return model


def predict_lstm(X_test_scaled, y_test, window: int = 14):
    model = load_model(f"{MODEL_DIR}/lstm_model.keras")
    X_seq, y_seq = prepare_lstm_sequences(X_test_scaled, y_test, window)
    if len(X_seq) == 0:
        return np.array([]), np.array([])
    preds = model.predict(X_seq, verbose=0).flatten()
    return preds, y_seq


# ─────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────
def train_predict_prophet(df_full: pd.DataFrame, test_size: int = 0.2):
    """
    Prophet needs a 'ds' (date) and 'y' (target) column.
    We simulate dates from row index since dataset has no real dates.
    """
    from prophet import Prophet

    n = len(df_full)
    split = int(n * (1 - test_size))

    prophet_df = pd.DataFrame({
        "ds": pd.date_range(start="2022-01-01", periods=n, freq="D"),
        "y": df_full["demand"].values
    })

    train_prophet = prophet_df.iloc[:split]
    test_prophet = prophet_df.iloc[split:]

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )
    model.fit(train_prophet)

    future = model.make_future_dataframe(periods=len(test_prophet))
    forecast = model.predict(future)

    y_pred = forecast["yhat"].iloc[split:].values
    y_true = test_prophet["y"].values

    with open(f"{MODEL_DIR}/prophet.pkl", "wb") as f:
        import pickle
        pickle.dump(model, f)

    return y_true, y_pred, forecast, prophet_df