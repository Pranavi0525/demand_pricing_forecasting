"""
Feature Engineering Pipeline
Transforms raw daily demand data into ML-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(path: str) -> pd.DataFrame:
    """Load the Daily Demand Forecasting Orders CSV."""
    df = pd.read_csv(path)  # simple comma separator, no tricks needed
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers lag, rolling, and calendar features.
    Target: 'Target (Total orders)'
    """
    df = df.copy()

    # --- Rename target column ---
    target_col = "Target (Total orders)"
    df = df.rename(columns={target_col: "demand"})

    # --- Rename week column ---
    week_col = "Week of the month (first week, second, third, fourth or fifth week"
    if week_col in df.columns:
        df = df.rename(columns={week_col: "week_of_month"})

    # --- Drop the unnamed index column if present ---
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # --- Ensure demand is float ---
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna(subset=["demand"]).reset_index(drop=True)

    # --- Lag Features ---
    df["lag_1"]  = df["demand"].shift(1)
    df["lag_3"]  = df["demand"].shift(3)
    df["lag_7"]  = df["demand"].shift(7)
    df["lag_14"] = df["demand"].shift(14)

    # --- Rolling Window Features ---
    df["rolling_mean_7"]  = df["demand"].shift(1).rolling(7).mean()
    df["rolling_std_7"]   = df["demand"].shift(1).rolling(7).std()
    df["rolling_mean_14"] = df["demand"].shift(1).rolling(14).mean()
    df["rolling_max_7"]   = df["demand"].shift(1).rolling(7).max()
    df["rolling_min_7"]   = df["demand"].shift(1).rolling(7).min()

    # --- Momentum Features ---
    df["momentum_3"] = df["lag_1"] - df["lag_3"]
    df["momentum_7"] = df["lag_1"] - df["lag_7"]

    # --- Calendar Features ---
    if "week_of_month" not in df.columns:
        df["week_of_month"] = (df.index % 4) + 1

    df["is_week_start"] = (df["week_of_month"] == 1).astype(int)
    df["is_week_end"]   = (df["week_of_month"] == 4).astype(int)

    # --- Drop NaN rows from lag/rolling ---
    df = df.dropna().reset_index(drop=True)

    print(f"      Features built successfully. Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all columns except the target."""
    exclude = {"demand"}
    return [c for c in df.columns if c not in exclude]


def prepare_lstm_sequences(X: np.ndarray, y: np.ndarray, window: int = 14):
    """Convert flat features into LSTM sequences."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def scale_features(X_train, X_test):
    """Scale features to 0-1 range."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler