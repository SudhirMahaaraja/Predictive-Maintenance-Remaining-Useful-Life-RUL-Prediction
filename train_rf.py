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
    if sample_weights is not None:
        print(f"  Sample weights: {config.WEIGHT_SCHEME} (range {sample_weights.min():.2f}-{sample_weights.max():.2f})")
    
    epochs = config.EPOCHS
    trees_per_epoch = config.N_ESTIMATORS // epochs
    remainder_trees = config.N_ESTIMATORS % epochs
    
    print(f"\nTraining for {epochs} epochs ({trees_per_epoch} trees/epoch, {config.N_ESTIMATORS} total)...")
    t0 = time.time()
    
    # Open epoch log file
    log_path = os.path.join(config.OUTPUT_PATH, 'rf_epoch_log.txt')
    log_file = open(log_path, 'w')
    log_file.write(f"Random Forest Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total Epochs: {epochs} | Trees/Epoch: {trees_per_epoch} | Total Trees: {config.N_ESTIMATORS}\n")
    log_file.write(f"RUL Cap: {config.RUL_CAP} | Log RUL: {config.USE_LOG_RUL} | Sample Weights: {config.USE_SAMPLE_WEIGHTS} ({config.WEIGHT_SCHEME})\n")
    log_file.write("="*100 + "\n")
    log_file.write(f"{'Epoch':>6} | {'Trees':>6} | {'Train MAE':>10} | {'Train RMSE':>11} | {'Train R²':>9} | {'Val MAE':>10} | {'Val RMSE':>11} | {'Val R²':>9} | {'Val MAPE':>9} | {'Time(s)':>8}\n")
    log_file.write("-"*100 + "\n")
    log_file.flush()
    
    model = RandomForestRegressor(
        n_estimators=trees_per_epoch,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        max_features=config.MAX_FEATURES,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        warm_start=True,
        verbose=0
    )
    
    epoch_history = []
    best_val_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Add trees for this epoch
        if epoch == 1:
            current_trees = trees_per_epoch
        else:
            current_trees = model.n_estimators + trees_per_epoch
        # Add remainder trees on last epoch
        if epoch == epochs:
            current_trees += remainder_trees
        model.n_estimators = current_trees
        
        # Train on transformed RUL with sample weights
        model.fit(X_train_scaled, y_train_t, sample_weight=sample_weights)
        
        # Predictions → inverse transform back to original scale
        y_train_pred_t = model.predict(X_train_scaled)
        y_val_pred_t = model.predict(X_val_scaled)
        y_train_pred = inverse_transform_rul(y_train_pred_t, config)
        y_val_pred = inverse_transform_rul(y_val_pred_t, config)
        
        # Metrics on ORIGINAL uncapped scale (correct R²)
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        epoch_history.append({
            'epoch': epoch,
            'n_trees': current_trees,
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'val_mape': val_metrics['mape']
        })
        
        # Track best
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_epoch = epoch
        
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{epochs} | Trees: {current_trees:4d} | "
              f"Train MAE: {train_metrics['mae']:.2f} | Val MAE: {val_metrics['mae']:.2f} | "
              f"Val R²: {val_metrics['r2']:.4f} | Time: {elapsed:.1f}s")
        
        # Write to log file
        log_file.write(f"{epoch:>6} | {current_trees:>6} | {train_metrics['mae']:>10.2f} | {train_metrics['rmse']:>11.2f} | {train_metrics['r2']:>9.4f} | {val_metrics['mae']:>10.2f} | {val_metrics['rmse']:>11.2f} | {val_metrics['r2']:>9.4f} | {val_metrics['mape']:>8.2f}% | {elapsed:>8.1f}\n")
        log_file.flush()
    
    train_time = time.time() - t0
    print(f"\n✓ Training completed in {train_time:.1f}s")
    print(f"✓ Best Val MAE: {best_val_mae:.2f} at epoch {best_epoch}")
    
    # Write final summary to log
    log_file.write("="*100 + "\n")
    log_file.write(f"Training completed in {train_time:.1f}s\n")
    log_file.write(f"Best Val MAE: {best_val_mae:.2f} at epoch {best_epoch}\n")
    log_file.close()
    print(f"✓ Epoch log saved: {log_path}")
    
    # Final metrics on UNCAPPED original scale (correct R²)
    y_train_pred = inverse_transform_rul(model.predict(X_train_scaled), config)
    y_val_pred = inverse_transform_rul(model.predict(X_val_scaled), config)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    print_metrics(train_metrics, "Training")
    print_metrics(val_metrics, "Validation")
    
    # Feature importance with human-readable names
    feature_importance = model.feature_importances_
    top_15_idx = np.argsort(feature_importance)[-15:][::-1]
    print("\n  ── Factors Affecting RUL (Feature Importance) ──")
    for i, idx in enumerate(top_15_idx):
        bar = '█' * int(feature_importance[idx] * 200)
        print(f"    {i+1:2d}. {get_feature_name(idx):<25s} {feature_importance[idx]:.4f}  {bar}")
    
    # Plot epoch history
    _plot_epoch_history(epoch_history, config)
    
    return model, scaler, val_metrics


def _plot_epoch_history(epoch_history, config):
    """Plot training metrics across epochs"""
    epochs = [e['epoch'] for e in epoch_history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE plot
    axes[0].plot(epochs, [e['train_mae'] for e in epoch_history], 'b-o', markersize=4, label='Train MAE')
    axes[0].plot(epochs, [e['val_mae'] for e in epoch_history], 'r-o', markersize=4, label='Val MAE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE (cycles)')
    axes[0].set_title('MAE per Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² plot
    axes[1].plot(epochs, [e['train_r2'] for e in epoch_history], 'b-o', markersize=4, label='Train R²')
    axes[1].plot(epochs, [e['val_r2'] for e in epoch_history], 'r-o', markersize=4, label='Val R²')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('R² per Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, 'rf_epoch_history.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved epoch history plot: {save_path}")


def train_xgboost(X_train, y_train, X_val, y_val, config):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        return None, None, None
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST")
    print("="*80)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply same RUL transform
    y_train_t = transform_rul(y_train, config)
    y_val_t = transform_rul(y_val, config)
    sample_weights = compute_sample_weights(y_train, config)
    
    # Train model
    print(f"\nTraining with {config.XGB_N_ESTIMATORS} estimators...")
    t0 = time.time()
    
    model = XGBRegressor(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbosity=1
    )
    
    model.fit(
        X_train_scaled, y_train_t,
        eval_set=[(X_val_scaled, y_val_t)],
        sample_weight=sample_weights,
        verbose=True
    )
    
    train_time = time.time() - t0
    print(f"✓ Training completed in {train_time:.1f}s")
    
    # Predictions → inverse transform
    y_train_pred = inverse_transform_rul(model.predict(X_train_scaled), config)
    y_val_pred = inverse_transform_rul(model.predict(X_val_scaled), config)
    
    # Metrics on original uncapped scale
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    print_metrics(train_metrics, "XGB Training")
    print_metrics(val_metrics, "XGB Validation")
    
    return model, scaler, val_metrics


def save_onnx_models(model, scaler, n_features, config, model_type='rf'):
    """Export model + scaler as a single ONNX pipeline"""
    from sklearn.pipeline import Pipeline
    
    # Create pipeline: scaler → model
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    if model_type == 'rf':
        if not SKL2ONNX_AVAILABLE:
            print("⚠ skl2onnx not installed. Skipping RF ONNX export.")
            return
        try:
            onnx_model = convert_sklearn(pipeline, initial_types=initial_type,
                                         target_opset=12)
            onnx_path = os.path.join(config.MODEL_SAVE_PATH, 'random_forest_model.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"✓ Random Forest ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"⚠ RF ONNX export failed: {e}")
    
    elif model_type == 'xgb':
        if not ONNXMLTOOLS_AVAILABLE:
            print("⚠ onnxmltools not installed. Skipping XGBoost ONNX export.")
            return
        try:
            # Export XGBoost separately (pipeline conversion can be tricky)
            xgb_initial = [('float_input', XGBFloatTensorType([None, n_features]))]
            onnx_xgb = onnxmltools.convert_xgboost(model, initial_types=xgb_initial,
                                                     target_opset=12)
            onnx_path = os.path.join(config.MODEL_SAVE_PATH, 'xgboost_model.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_xgb.SerializeToString())
            print(f"✓ XGBoost ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")
            
            # Also save scaler separately for XGBoost ONNX inference
            scaler_onnx = convert_sklearn(scaler, initial_types=initial_type,
                                           target_opset=12)
            scaler_path = os.path.join(config.MODEL_SAVE_PATH, 'xgb_scaler.onnx')
            with open(scaler_path, 'wb') as f:
                f.write(scaler_onnx.SerializeToString())
            print(f"✓ XGBoost scaler ONNX saved: {scaler_path}")
        except Exception as e:
            print(f"⚠ XGBoost ONNX export failed: {e}")


def main():
    config = RFConfig()
    
    # Create output directories
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    print("="*80)
    print("RANDOM FOREST BEARING RUL PREDICTION")
    print("="*80)
    
    # Load data
    all_data = load_all_data(config)
    
    if len(all_data) == 0:
        print("❌ No data loaded. Check dataset paths.")
        return
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_split(all_data, config)
    
    print(f"\nFeature dimension: {X_train.shape[1]}")
    print(f"RUL range: {y_train.min():.0f} - {y_train.max():.0f} cycles")
    
    # Train Random Forest
    rf_model, rf_scaler, rf_val_metrics = train_random_forest(
        X_train, y_train, X_val, y_val, config
    )
    
    # Save Random Forest model
    rf_save_path = os.path.join(config.MODEL_SAVE_PATH, 'random_forest_model.pkl')
    with open(rf_save_path, 'wb') as f:
        pickle.dump({
            'model': rf_model,
            'scaler': rf_scaler,
            'config': config,
            'val_metrics': rf_val_metrics
        }, f)
    print(f"\n✓ Random Forest model saved: {rf_save_path}")
    
    # Plot predictions (inverse-transformed)
    X_val_scaled = rf_scaler.transform(X_val)
    y_val_pred = inverse_transform_rul(rf_model.predict(X_val_scaled), config)
    plot_predictions(
        y_val, y_val_pred,
        f"Random Forest - Val MAE: {rf_val_metrics['mae']:.1f}",
        os.path.join(config.OUTPUT_PATH, 'rf_predictions.png')
    )
    
    # Train XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_scaler, xgb_val_metrics = train_xgboost(
            X_train, y_train, X_val, y_val, config
        )
        
        if xgb_model is not None:
            # Save XGBoost model
            xgb_save_path = os.path.join(config.MODEL_SAVE_PATH, 'xgboost_model.pkl')
            with open(xgb_save_path, 'wb') as f:
                pickle.dump({
                    'model': xgb_model,
                    'scaler': xgb_scaler,
                    'config': config,
                    'val_metrics': xgb_val_metrics
                }, f)
            print(f"\n✓ XGBoost model saved: {xgb_save_path}")
            
            # Plot predictions
            X_val_scaled = xgb_scaler.transform(X_val)
            y_val_pred = inverse_transform_rul(xgb_model.predict(X_val_scaled), config)
            plot_predictions(
                y_val, y_val_pred,
                f"XGBoost - Val MAE: {xgb_val_metrics['mae']:.1f}",
                os.path.join(config.OUTPUT_PATH, 'xgb_predictions.png')
            )
    
    # Test set evaluation — use UNCAPPED y_test for R² (fix R² collapse)
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # RF test
    X_test_scaled = rf_scaler.transform(X_test)
    y_test_pred_rf = inverse_transform_rul(rf_model.predict(X_test_scaled), config)
    test_metrics_rf = calculate_metrics(y_test, y_test_pred_rf)
    
    print("\n── Random Forest ──")
    print_metrics(test_metrics_rf, "RF Test")
    
    plot_predictions(
        y_test, y_test_pred_rf,
        f"Random Forest Test - MAE: {test_metrics_rf['mae']:.1f}",
        os.path.join(config.OUTPUT_PATH, 'rf_test_predictions.png')
    )
    
    # Save test results
    results = {
        'random_forest': {
            'val_metrics': rf_val_metrics,
            'test_metrics': test_metrics_rf
        }
    }
    
    xgb_test_done = False
    if XGBOOST_AVAILABLE and xgb_model is not None:
        X_test_scaled = xgb_scaler.transform(X_test)
        y_test_pred_xgb = inverse_transform_rul(xgb_model.predict(X_test_scaled), config)
        test_metrics_xgb = calculate_metrics(y_test, y_test_pred_xgb)
        
        print("\n── XGBoost ──")
        print_metrics(test_metrics_xgb, "XGB Test")
        
        plot_predictions(
            y_test, y_test_pred_xgb,
            f"XGBoost Test - MAE: {test_metrics_xgb['mae']:.1f}",
            os.path.join(config.OUTPUT_PATH, 'xgb_test_predictions.png')
        )
        
        results['xgboost'] = {
            'val_metrics': xgb_val_metrics,
            'test_metrics': test_metrics_xgb
        }
        xgb_test_done = True
    
    # ── FINAL SUMMARY TABLE ──
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    print(f"{'Metric':<20} | {'RF Val':>12} | {'RF Test':>12}", end='')
    if xgb_test_done:
        print(f" | {'XGB Val':>12} | {'XGB Test':>12}", end='')
    print()
    print("-"*80)
    for key, label in [('mae','MAE (cycles)'), ('rmse','RMSE (cycles)'), ('r2','R²'), ('mape','MAPE (%)'),
                        ('acc_50','Acc ±50'), ('acc_100','Acc ±100'), ('acc_200','Acc ±200'), ('acc_500','Acc ±500')]:
        fmt = '.2f' if key != 'r2' else '.4f'
        suffix = '%' if 'acc' in key else ''
        rv = rf_val_metrics.get(key, 0)
        rt = test_metrics_rf.get(key, 0)
        print(f"{label:<20} | {rv:>11{fmt}}{suffix} | {rt:>11{fmt}}{suffix}", end='')
        if xgb_test_done:
            xv = xgb_val_metrics.get(key, 0)
            xt = test_metrics_xgb.get(key, 0)
            print(f" | {xv:>11{fmt}}{suffix} | {xt:>11{fmt}}{suffix}", end='')
        print()
    print("="*80)
    
    # Save results JSON
    results_path = os.path.join(config.OUTPUT_PATH, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Test results saved: {results_path}")
    
    # ── Export to ONNX ──
    n_features = X_train.shape[1]
    save_onnx_models(rf_model, rf_scaler, n_features, config, model_type='rf')
    if XGBOOST_AVAILABLE and xgb_model is not None:
        save_onnx_models(xgb_model, xgb_scaler, n_features, config, model_type='xgb')
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
