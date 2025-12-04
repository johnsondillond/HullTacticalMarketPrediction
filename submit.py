"""
Hull Tactical Market Prediction - Kaggle Submission Script
For Kaggle Code Competition with streaming inference API
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
# Use Kaggle's path format (no leading ./)
DATA_PATH = "/kaggle/input/hull-tactical-market-prediction/"
MIN_DATE_ID_FOR_TRAIN = 8000
RANDOM_STATE = 42
TARGET_CLIP_MIN = -4.0
TARGET_CLIP_MAX = 4.0
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
BEST_DAMPENING = 2.0

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================
def generate_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling volatility, momentum, and MACD features."""
    df = df.sort("date_id")
    exclude = ["date_id", "target", "is_scored", "forward_returns", "risk_free_rate"]
    feat_cols = [c for c in df.columns if c not in exclude]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in feat_cols])

    key_roots = ["E2", "S2", "P8", "I2", "M1"]
    valid_keys = [k for k in key_roots if k in df.columns]
    ops = []
    for feat in valid_keys:
        col = pl.col(feat)
        ops.append(col.rolling_std(20).alias(f"{feat}_vol_20"))
        ops.append((col - col.shift(5)).alias(f"{feat}_mom_5d"))
        ema_12 = col.ewm_mean(span=12, adjust=False)
        ema_26 = col.ewm_mean(span=26, adjust=False)
        ops.append((ema_12 - ema_26).alias(f"{feat}_macd"))
    if ops:
        df = df.with_columns(ops)

    all_numeric = [c for c in df.columns if c not in exclude]
    return df.with_columns([pl.col(c).forward_fill().fill_null(0.0) for c in all_numeric])


def generate_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Add risk-adjusted momentum and systemic volatility."""
    vol_cols = [c for c in df.columns if "_vol_20" in c]
    ops = []
    for vol_col in vol_cols:
        base_feat = vol_col.replace("_vol_20", "")
        mom_col = f"{base_feat}_mom_5d"
        if mom_col in df.columns:
            ops.append((pl.col(mom_col) / (pl.col(vol_col) + 1e-6)).alias(f"{base_feat}_risk_adj_mom"))
    if vol_cols:
        ops.append(pl.sum_horizontal(vol_cols).alias("systemic_volatility"))
    if ops:
        df = df.with_columns(ops)
    return df


def add_kmeans_features_single(df: pl.DataFrame, kmeans, imputer, scaler, regime_cols, n_clusters: int = 5):
    """Add K-Means features for a single dataframe using pre-fit transformers."""
    X = df.select(regime_cols).to_pandas()
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    clusters = kmeans.predict(X_scaled)
    
    df = df.with_columns(pl.Series("market_regime", clusters).cast(pl.Float64))
    for i in range(1, n_clusters):
        df = df.with_columns(
            pl.when(pl.col("market_regime") == i).then(1.0).otherwise(0.0).alias(f"regime_{i}")
        )
    return df


# =============================================================================
# GLOBAL MODEL TRAINING (runs once when script loads)
# =============================================================================
print("=" * 60)
print("TRAINING MODELS FOR KAGGLE INFERENCE")
print("=" * 60)

# Load training data
train_full = pl.read_csv(os.path.join(DATA_PATH, "train.csv"))

# Rename target column if needed
if "market_forward_excess_returns" in train_full.columns:
    train_full = train_full.rename({"market_forward_excess_returns": "target"})
if "lagged_forward_returns" in train_full.columns:
    train_full = train_full.rename({"lagged_forward_returns": "target"})

# Filter to modern era
train_filtered = train_full.filter(pl.col("date_id") > MIN_DATE_ID_FOR_TRAIN)
print(f"Training samples: {train_filtered.height:,}")

# Feature engineering on training data
train = generate_technical_features(train_filtered)
train = generate_interactions(train)

# Prepare K-Means (fit on training data)
N_CLUSTERS = 5
regime_cols = [c for c in train.columns if "_vol_" in c or "_mom_" in c]
if not regime_cols:
    regime_cols = [c for c in train.columns if c.startswith("V")]

X_regime = train.select(regime_cols).to_pandas()
KMEANS_IMPUTER = SimpleImputer(strategy="median")
X_regime_imputed = KMEANS_IMPUTER.fit_transform(X_regime)
KMEANS_SCALER = StandardScaler()
X_regime_scaled = KMEANS_SCALER.fit_transform(X_regime_imputed)
KMEANS_MODEL = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
train_clusters = KMEANS_MODEL.fit_predict(X_regime_scaled)

# Add K-Means features to training data
train = train.with_columns(pl.Series("market_regime", train_clusters).cast(pl.Float64))
for i in range(1, N_CLUSTERS):
    train = train.with_columns(
        pl.when(pl.col("market_regime") == i).then(1.0).otherwise(0.0).alias(f"regime_{i}")
    )

# Define features
exclude_cols = ["date_id", "target", "is_scored", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
FEATURES = [c for c in train.columns if c not in exclude_cols]
REGIME_COLS = regime_cols  # Save for inference
print(f"Total features: {len(FEATURES)}")

# Prepare X, y
X = train.select(FEATURES).to_pandas()
y = train.get_column("target").to_numpy()
y = np.clip(y, TARGET_CLIP_MIN, TARGET_CLIP_MAX)

# Train champion model
print("Training champion model...")
CHAMPION_MODEL = make_pipeline(
    SimpleImputer(strategy="median"),
    RobustScaler(),
    SelectKBest(f_regression, k=min(70, len(FEATURES))),
    HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.02,
        max_iter=200,
        max_depth=5,
        l2_regularization=25,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
)
CHAMPION_MODEL.fit(X, y)

train_pred = CHAMPION_MODEL.predict(X)
GLOBAL_MEAN = float(np.mean(train_pred))
GLOBAL_STD = float(np.std(train_pred))
print(f"Champion model trained. mean={GLOBAL_MEAN:.6f}, std={GLOBAL_STD:.6f}")

# Train volatility proxy model
print("Training volatility proxy model...")
y_vol = np.abs(y)
VOL_MODEL = make_pipeline(
    SimpleImputer(strategy="median"),
    RobustScaler(),
    SelectKBest(f_regression, k=min(30, len(FEATURES))),
    HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=3,
        learning_rate=0.01,
        random_state=RANDOM_STATE,
    ),
)
VOL_MODEL.fit(X, y_vol)
print("Volatility model trained.")

print("=" * 60)
print("MODELS READY FOR INFERENCE")
print("=" * 60)


# =============================================================================
# PREDICTION FUNCTION (called by Kaggle inference server)
# =============================================================================
def predict(test_df: pl.DataFrame) -> float:
    """
    Generate a single prediction for the Kaggle inference server.
    This function is called once per test row during evaluation.
    """
    # Apply same feature engineering as training
    test_processed = generate_technical_features(test_df)
    test_processed = generate_interactions(test_processed)
    
    # Add K-Means features using pre-fit transformers
    test_processed = add_kmeans_features_single(
        test_processed, KMEANS_MODEL, KMEANS_IMPUTER, KMEANS_SCALER, REGIME_COLS, N_CLUSTERS
    )
    
    # Prepare input features
    X_input = test_processed.select([pl.col(c) for c in FEATURES if c in test_processed.columns]).to_pandas()
    
    # Handle missing columns
    for missing_col in set(FEATURES) - set(X_input.columns):
        X_input[missing_col] = np.nan
    X_input = X_input[FEATURES]
    
    # Get prediction
    pred = CHAMPION_MODEL.predict(X_input)[0]
    risk = VOL_MODEL.predict(X_input)[0]
    
    # Apply risk factor
    risk_factor = 1.0
    if risk > 0.015:
        risk_factor = 0.5  # High volatility → dampen
    elif risk < 0.005:
        risk_factor = 1.2  # Low volatility → amplify
    
    # Convert to trading signal
    if GLOBAL_STD == 0:
        signal = 1.0
    else:
        z = (pred - GLOBAL_MEAN) / (GLOBAL_STD * BEST_DAMPENING)
        signal = norm.cdf(z * risk_factor) * 2.0
    
    return float(np.clip(signal, MIN_INVESTMENT, MAX_INVESTMENT))


# =============================================================================
# KAGGLE INFERENCE SERVER SETUP
# =============================================================================
try:
    import kaggle_evaluation.default_inference_server
    
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        print("Starting Kaggle inference server...")
        server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
        server.serve()
    else:
        print("Local mode - inference server not started.")
        
except ImportError:
    print("kaggle_evaluation package not found - running in local mode.")


# =============================================================================
# LOCAL TESTING / CSV GENERATION
# =============================================================================
if __name__ == "__main__" and not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("\n" + "=" * 60)
    print("GENERATING LOCAL SUBMISSION CSV")
    print("=" * 60)
    
    # Load test data
    test_raw = pl.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print(f"Test samples: {test_raw.height:,}")
    
    # Process test data
    test = generate_technical_features(test_raw)
    test = generate_interactions(test)
    test = add_kmeans_features_single(test, KMEANS_MODEL, KMEANS_IMPUTER, KMEANS_SCALER, REGIME_COLS, N_CLUSTERS)
    
    # Prepare test features
    X_test = test.select(FEATURES).to_pandas()
    for missing in (set(FEATURES) - set(X_test.columns)):
        X_test[missing] = np.nan
    X_test = X_test[FEATURES]
    
    # Generate predictions
    test_preds = CHAMPION_MODEL.predict(X_test)
    test_risk = VOL_MODEL.predict(X_test)
    
    final_signals = []
    for pred, risk in zip(test_preds, test_risk):
        risk_factor = 1.0
        if risk > 0.015:
            risk_factor = 0.5
        elif risk < 0.005:
            risk_factor = 1.2
        
        if GLOBAL_STD == 0:
            signal = 1.0
        else:
            z = (pred - GLOBAL_MEAN) / (GLOBAL_STD * BEST_DAMPENING)
            signal = norm.cdf(z * risk_factor) * 2.0
        
        final_signals.append(float(np.clip(signal, MIN_INVESTMENT, MAX_INVESTMENT)))
    
    # Create submission
    submission = pl.DataFrame({
        "date_id": test.get_column("date_id"),
        "prediction": final_signals
    })
    submission = submission.unique(subset=["date_id"], keep="last").sort("date_id")
    
    # Save
    submission.to_pandas().to_csv("submission.csv", index=False)
    print("Saved: submission.csv")
    print(f"Signal range: [{min(final_signals):.4f}, {max(final_signals):.4f}]")
    print(f"Signal mean: {np.mean(final_signals):.4f}")
