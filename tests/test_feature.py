"""
tests/test_features.py

Unit Tests for Feature Engineering Pipeline
Run with: pytest tests/ -v

WHY WE WRITE TESTS:
In real companies, before any code goes to production, it must pass tests.
Tests make sure your functions work correctly even when data changes.
If someone changes features.py and breaks something, the test will catch it.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_features, get_feature_columns, scale_features, prepare_lstm_sequences


# ─────────────────────────────────────────
# FIXTURES — sample data to test with
# ─────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """
    Creates a small fake DataFrame that looks like our CSV.
    We use fake data so tests don't depend on the actual CSV file.
    """
    np.random.seed(42)
    n = 40  # need at least 30 rows for rolling features
    return pd.DataFrame({
        "Week of the month (first week, second, third, fourth or fifth week": np.tile([1, 2, 3, 4], 10),
        "Day of the week (Monday to Friday)": np.tile([2, 3, 4, 5, 6], 8),
        "Non-urgent order": np.random.uniform(100, 300, n),
        "Urgent order": np.random.uniform(80, 200, n),
        "Order type A": np.random.uniform(30, 80, n),
        "Order type B": np.random.uniform(50, 150, n),
        "Order type C": np.random.uniform(80, 200, n),
        "Fiscal sector orders": np.random.uniform(0, 100, n),
        "Orders from the traffic controller sector": np.random.randint(10000, 80000, n),
        "Banking orders (1)": np.random.randint(5000, 50000, n),
        "Banking orders (2)": np.random.randint(20000, 200000, n),
        "Banking orders (3)": np.random.randint(5000, 40000, n),
        "Target (Total orders)": np.random.uniform(150, 600, n),
    })


# ─────────────────────────────────────────
# TEST 1: load_data returns a DataFrame
# ─────────────────────────────────────────

def test_build_features_returns_dataframe(sample_raw_df):
    """
    WHY: build_features() must always return a DataFrame.
    If it returns None or crashes, nothing else works.
    """
    result = build_features(sample_raw_df)
    assert isinstance(result, pd.DataFrame), "build_features must return a DataFrame"


# ─────────────────────────────────────────
# TEST 2: demand column exists after build
# ─────────────────────────────────────────

def test_demand_column_exists(sample_raw_df):
    """
    WHY: The target column must be renamed to 'demand'.
    All downstream code (model.py, train.py) uses df['demand'].
    If this breaks, everything breaks.
    """
    result = build_features(sample_raw_df)
    assert "demand" in result.columns, "'demand' column must exist after build_features"


# ─────────────────────────────────────────
# TEST 3: lag features are created
# ─────────────────────────────────────────

def test_lag_features_created(sample_raw_df):
    """
    WHY: Lag features are the most important features for time series.
    They must exist for the model to learn from past demand.
    """
    result = build_features(sample_raw_df)
    for lag_col in ["lag_1", "lag_3", "lag_7", "lag_14"]:
        assert lag_col in result.columns, f"Missing lag feature: {lag_col}"


# ─────────────────────────────────────────
# TEST 4: rolling features are created
# ─────────────────────────────────────────

def test_rolling_features_created(sample_raw_df):
    """
    WHY: Rolling features capture recent trends.
    Without them, the model can't see if demand is trending up or down.
    """
    result = build_features(sample_raw_df)
    for col in ["rolling_mean_7", "rolling_std_7", "rolling_mean_14"]:
        assert col in result.columns, f"Missing rolling feature: {col}"


# ─────────────────────────────────────────
# TEST 5: no NaN values after build
# ─────────────────────────────────────────

def test_no_nulls_after_build(sample_raw_df):
    """
    WHY: ML models crash if they receive NaN values.
    build_features must drop all rows with missing values.
    """
    result = build_features(sample_raw_df)
    assert result.isnull().sum().sum() == 0, "No NaN values should remain after build_features"


# ─────────────────────────────────────────
# TEST 6: get_feature_columns excludes demand
# ─────────────────────────────────────────

def test_feature_columns_excludes_demand(sample_raw_df):
    """
    WHY: We never want 'demand' as an input feature — it's the target.
    If it accidentally gets included, the model cheats by seeing the answer.
    """
    result = build_features(sample_raw_df)
    feature_cols = get_feature_columns(result)
    assert "demand" not in feature_cols, "'demand' must not be in feature columns"


# ─────────────────────────────────────────
# TEST 7: scale_features returns correct shape
# ─────────────────────────────────────────

def test_scale_features_shape(sample_raw_df):
    """
    WHY: Scaling must not change the shape of data.
    It only changes the values (to 0-1 range), not the number of rows/columns.
    """
    result = build_features(sample_raw_df)
    feature_cols = get_feature_columns(result)
    X = result[feature_cols].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    assert X_train_s.shape == X_train.shape, "Scaled train shape must match original"
    assert X_test_s.shape == X_test.shape, "Scaled test shape must match original"


# ─────────────────────────────────────────
# TEST 8: scaled values are between 0 and 1
# ─────────────────────────────────────────

def test_scale_features_range(sample_raw_df):
    """
    WHY: MinMaxScaler is supposed to bring values between 0 and 1.
    LSTM needs this — it can't learn well with values like 65000 or 188411.
    """
    result = build_features(sample_raw_df)
    feature_cols = get_feature_columns(result)
    X = result[feature_cols].values
    split = int(len(X) * 0.8)

    X_train_s, X_test_s, _ = scale_features(X[:split], X[split:])

    assert X_train_s.min() >= -0.01, "Scaled values should be >= 0"
    assert X_train_s.max() <= 1.01, "Scaled values should be <= 1"


# ─────────────────────────────────────────
# TEST 9: LSTM sequences have correct shape
# ─────────────────────────────────────────

def test_lstm_sequences_shape(sample_raw_df):
    """
    WHY: LSTM needs 3D input: (samples, window, features).
    If the shape is wrong, TensorFlow will throw a cryptic error.
    Better to catch it here with a clear test.
    """
    result = build_features(sample_raw_df)
    feature_cols = get_feature_columns(result)
    X = result[feature_cols].values
    y = result["demand"].values
    window = 7

    X_train_s, _, _ = scale_features(X, X)
    X_seq, y_seq = prepare_lstm_sequences(X_train_s, y, window=window)

    assert X_seq.ndim == 3, "LSTM X must be 3D: (samples, window, features)"
    assert X_seq.shape[1] == window, f"Window dimension must be {window}"
    assert len(X_seq) == len(y_seq), "X and y sequences must have same length"


# ─────────────────────────────────────────
# TEST 10: momentum features make sense
# ─────────────────────────────────────────

def test_momentum_features(sample_raw_df):
    """
    WHY: Momentum = lag_1 - lag_3 (is demand going up or down recently?)
    We verify the math is correct — not just that the column exists.
    """
    result = build_features(sample_raw_df)
    # momentum_3 should equal lag_1 - lag_3
    diff = (result["lag_1"] - result["lag_3"] - result["momentum_3"]).abs()
    assert diff.max() < 1e-6, "momentum_3 must equal lag_1 - lag_3"