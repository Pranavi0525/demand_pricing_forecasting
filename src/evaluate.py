"""
Model Evaluation & Comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    return {
        "Model": model_name,
        "RMSE ↓": round(rmse, 3),
        "MAE ↓": round(mae, 3),
        "R² ↑": round(r2, 3),
        "MAPE (%) ↓": round(mape, 2),
    }


def build_leaderboard(results: list[dict]) -> pd.DataFrame:
    """
    results: list of dicts from compute_metrics()
    Returns ranked leaderboard with best model highlighted.
    """
    df = pd.DataFrame(results)
    df["Score"] = (
        (1 / (df["RMSE ↓"] + 1e-9)) * 0.4 +
        (1 / (df["MAE ↓"] + 1e-9)) * 0.3 +
        df["R² ↑"].clip(0) * 0.3
    )
    df["Score"] = (df["Score"] / df["Score"].max() * 100).round(1)
    df["Rank"] = df["Score"].rank(ascending=False).astype(int)
    return df.sort_values("Rank").reset_index(drop=True)


def get_champion_model(leaderboard: pd.DataFrame) -> str:
    return leaderboard.iloc[0]["Model"]