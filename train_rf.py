"""
Train Random Forest and XGBoost models for bearing RUL prediction
"""
import os
import numpy as np
from datetime import datetime
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloatTensorType
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False

from rf_config import RFConfig
from rf_data_loader import load_all_data, create_train_val_test_split

# Human-readable feature names (must match rf_features.py extract order)
# Time domain: 8 per channel × 2 channels = 16
# Freq domain: 8 per channel × 2 channels = 16
# Envelope: 4 per channel × 2 channels = 8
# Operating conditions: 3 (speed, load, temp)
FEATURE_NAMES = [
    # Horizontal time-domain (0-7)
    'Horiz_Mean', 'Horiz_Std', 'Horiz_RMS', 'Horiz_Peak',
    'Horiz_Kurtosis', 'Horiz_Skewness', 'Horiz_CrestFactor', 'Horiz_ShapeFactor',
    # Vertical time-domain (8-15)
    'Vert_Mean', 'Vert_Std', 'Vert_RMS', 'Vert_Peak',
    'Vert_Kurtosis', 'Vert_Skewness', 'Vert_CrestFactor', 'Vert_ShapeFactor',
    # Horizontal frequency-domain (16-23)
    'Horiz_FreqCentroid', 'Horiz_FreqStd', 'Horiz_DominantFreq',
    'Horiz_Band_10-50Hz', 'Horiz_Band_60-180Hz', 'Horiz_Band_100-300Hz',
    'Horiz_Band_300-1kHz', 'Horiz_Band_1k-3kHz',
    # Vertical frequency-domain (24-31)
    'Vert_FreqCentroid', 'Vert_FreqStd', 'Vert_DominantFreq',
    'Vert_Band_10-50Hz', 'Vert_Band_60-180Hz', 'Vert_Band_100-300Hz',
    'Vert_Band_300-1kHz', 'Vert_Band_1k-3kHz',
    # Horizontal envelope (32-35)
    'Horiz_EnvMean', 'Horiz_EnvStd', 'Horiz_EnvMax', 'Horiz_EnvKurtosis',
    # Vertical envelope (36-39)
    'Vert_EnvMean', 'Vert_EnvStd', 'Vert_EnvMax', 'Vert_EnvKurtosis',
    # Operating conditions (40-42)
    'Speed_RPM', 'Load_N', 'Temperature'
]


def get_feature_name(idx):
    """Get human-readable feature name by index"""
    if idx < len(FEATURE_NAMES):
        return FEATURE_NAMES[idx]
    return f'Feature_{idx}'


# ─── RUL Transform & Weighting Helpers ─────────────────────────────────────

def transform_rul(y, config):
    """Apply RUL capping and optional log-transform"""
    y_out = np.copy(y)
    # Cap RUL to reduce noise from very long-lived bearings
    if config.RUL_CAP is not None:
        y_out = np.clip(y_out, 0, config.RUL_CAP)
    # Log-transform for better resolution at low RUL
    if config.USE_LOG_RUL:
        y_out = np.log1p(y_out + config.LOG_OFFSET)
    return y_out


def inverse_transform_rul(y_pred, config):
    """Reverse the RUL transform back to original scale"""
    y_out = np.copy(y_pred)
    if config.USE_LOG_RUL:
        y_out = np.expm1(y_out) - config.LOG_OFFSET
    if config.CLIP_PREDICTIONS:
        y_out = np.clip(y_out, config.PRED_MIN, config.PRED_MAX)
    return y_out


def compute_sample_weights(y, config):
    """Compute sample weights that emphasize low-RUL (critical) samples"""
    if not config.USE_SAMPLE_WEIGHTS:
        return None
    
    weights = np.ones_like(y, dtype=np.float64)
    
    if config.WEIGHT_SCHEME == 'inverse_sqrt':
        # Smooth inverse weighting: low RUL → high weight
        weights = 1.0 / np.sqrt(y + 1.0)
    elif config.WEIGHT_SCHEME == 'inverse':
        weights = 1.0 / (y + 1.0)
    elif config.WEIGHT_SCHEME == 'exponential':
        # Exponential boost for samples below critical threshold
        weights = np.where(y < config.CRITICAL_RUL,
                           config.WEIGHT_MAX,
                           config.WEIGHT_MIN)
    
    # Normalize to [WEIGHT_MIN, WEIGHT_MAX]
    if weights.max() > weights.min():
        weights = config.WEIGHT_MIN + (weights - weights.min()) / (weights.max() - weights.min()) * (config.WEIGHT_MAX - config.WEIGHT_MIN)
    
    return weights


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics + accuracy bands"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE with epsilon for safety
    epsilon = 10.0
    mask = y_true > epsilon
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    # Accuracy bands: % of predictions within ±N cycles
    abs_errors = np.abs(y_true - y_pred)
    n = len(y_true)
    acc_50 = np.sum(abs_errors <= 50) / n * 100
    acc_100 = np.sum(abs_errors <= 100) / n * 100
    acc_200 = np.sum(abs_errors <= 200) / n * 100
    acc_500 = np.sum(abs_errors <= 500) / n * 100
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
        'acc_50': acc_50, 'acc_100': acc_100, 'acc_200': acc_200, 'acc_500': acc_500
    }


def print_metrics(metrics, label=""):
    """Pretty-print all metrics"""
    print(f"\n  {label} Metrics:")
    print(f"    MAE:    {metrics['mae']:.2f} cycles")
    print(f"    RMSE:   {metrics['rmse']:.2f} cycles")
    print(f"    R²:     {metrics['r2']:.4f}")
    print(f"    MAPE:   {metrics['mape']:.2f}%")
    print(f"    Accuracy:  ±50: {metrics['acc_50']:.1f}%  |  ±100: {metrics['acc_100']:.1f}%  |  ±200: {metrics['acc_200']:.1f}%  |  ±500: {metrics['acc_500']:.1f}%")


def plot_predictions(y_true, y_pred, title, save_path):
    """Plot predicted vs actual RUL"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel('True RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (cycles)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution\nMean: {np.mean(errors):.1f}, Std: {np.std(errors):.1f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved plot: {save_path}")


def train_random_forest(X_train, y_train, X_val, y_val, config):
    """Train Random Forest model with epoch-based iterative training"""
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ── RUL Transform (Fix #1: low-RUL underprediction) ──
    y_train_t = transform_rul(y_train, config)
    y_val_t = transform_rul(y_val, config)
    
    if config.USE_LOG_RUL:
        print(f"  RUL transform: log1p(clip(RUL, 0, {config.RUL_CAP}) + {config.LOG_OFFSET})")
        print(f"  Transformed range: {y_train_t.min():.2f} - {y_train_t.max():.2f}")
    elif config.RUL_CAP:
        print(f"  RUL capped at {config.RUL_CAP}")
    
    # ── Sample Weights (Fix #2: mid-range variance) ──
    sample_weights = compute_sample_weights(y_train, config)
