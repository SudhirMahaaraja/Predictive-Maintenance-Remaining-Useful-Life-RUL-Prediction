"""
Test/inference script for Random Forest bearing RUL prediction
Predicts RUL for a single CSV file or picks a random one from datasets
"""
import os
import random
import numpy as np
import pandas as pd
import pickle
import re
from scipy.signal import resample
from rf_features import extract_all_features
from rf_config import RFConfig


# ─── RUL inverse transform (must match train_rf.py) ──────────────────────

def inverse_transform_rul(y_pred, config):
    """Reverse the RUL transform back to original scale"""
    y_out = np.copy(y_pred)
    if config.USE_LOG_RUL:
        y_out = np.expm1(y_out) - config.LOG_OFFSET
    if config.CLIP_PREDICTIONS:
        y_out = np.clip(y_out, config.PRED_MIN, config.PRED_MAX)
    return y_out


# ─── Random file picker ──────────────────────────────────────────────────

def pick_random_csv(config):
    """
    Pick a random CSV file from the dataset folders.
    Returns (csv_path, bearing_id, condition_info, total_files, file_index)
    """
    candidates = []
    
    # Scan XJTU-SY datasets
    for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
        cond_path = os.path.join(config.XJTU_PATH, condition)
        if not os.path.exists(cond_path):
            continue
        cond_info = config.XJTU_CONDITIONS.get(condition, {'speed': 2100, 'load': 12000})
        for bearing in os.listdir(cond_path):
            bearing_path = os.path.join(cond_path, bearing)
            if not os.path.isdir(bearing_path):
                continue
            csv_files = sorted([f for f in os.listdir(bearing_path) if f.endswith('.csv')],
                               key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])
            if csv_files:
                candidates.append({
                    'dataset': 'XJTU-SY',
                    'condition': condition,
                    'bearing': bearing,
                    'bearing_path': bearing_path,
                    'csv_files': csv_files,
                    'speed': cond_info['speed'],
                    'load': cond_info['load'],
                    'temp': 30.0
                })
    
    # Scan FEMTO datasets
    for dataset_type in ['Learning_set', 'Test_set', 'Full_Test_Set']:
        data_path = os.path.join(config.FEMTO_PATH, dataset_type)
        if not os.path.exists(data_path):
            continue
        for bearing in os.listdir(data_path):
            bearing_path = os.path.join(data_path, bearing)
            if not os.path.isdir(bearing_path):
                continue
            csv_files = sorted([f for f in os.listdir(bearing_path) 
                               if f.startswith('acc') and f.endswith('.csv')],
                               key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])
            if csv_files:
                cond_info = config.FEMTO_CONDITIONS.get(bearing, {'speed': 1800, 'load': 4000})
                candidates.append({
                    'dataset': 'FEMTO',
                    'condition': dataset_type,
                    'bearing': bearing,
                    'bearing_path': bearing_path,
                    'csv_files': csv_files,
                    'speed': cond_info['speed'],
                    'load': cond_info['load'],
                    'temp': 35.0
                })
    
    if not candidates:
        print("❌ No dataset folders found.")
        return None
    
    # Pick random bearing, then random file
    chosen = random.choice(candidates)
    file_idx = random.randint(0, len(chosen['csv_files']) - 1)
    csv_file = chosen['csv_files'][file_idx]
    csv_path = os.path.join(chosen['bearing_path'], csv_file)
    total_files = len(chosen['csv_files'])
    true_rul = total_files - file_idx - 1
    
    return {
        'csv_path': csv_path,
        'dataset': chosen['dataset'],
        'condition': chosen['condition'],
        'bearing': chosen['bearing'],
        'speed': chosen['speed'],
        'load': chosen['load'],
        'temp': chosen['temp'],
        'file_index': file_idx,
        'total_files': total_files,
        'true_rul': true_rul
    }


def predict_rul(csv_path, model_path='rf_models/random_forest_model.pkl',
                speed=1800, load=4000, temp=35):
    """
    Predict RUL for a single bearing vibration CSV file
    
    Args:
        csv_path: Path to CSV file with vibration data
        model_path: Path to saved model pickle file
        speed: Operating speed (RPM)
        load: Operating load (N)
        temp: Operating temperature (°C)
    
    Returns:
        predicted_rul: Predicted RUL in cycles
    """
    config = RFConfig()
    
    print("="*80)
    print(f"PREDICTING RUL: {os.path.basename(csv_path)}")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model = checkpoint['model']
    scaler = checkpoint['scaler']
    saved_config = checkpoint.get('config', config)
    val_metrics = checkpoint.get('val_metrics', {})
    
    if val_metrics:
        print(f"Model validation MAE: {val_metrics.get('mae', 'N/A'):.2f} cycles")
    
    # Load CSV
    print(f"\nLoading vibration data...")
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns
    h_col = next((c for c in df.columns if 'horiz' in c.lower() or 'horizontal' in c.lower()), None)
    v_col = next((c for c in df.columns if 'vert' in c.lower() or 'vertical' in c.lower()), None)
    
    if h_col is None or v_col is None:
        if len(df.columns) >= 2:
            h_col, v_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("CSV must have at least two columns")
    
    print(f"Using columns: '{h_col}' and '{v_col}'")
    
    sig_h = df[h_col].values.astype(np.float32)
    sig_v = df[v_col].values.astype(np.float32)
    
    # Resample if needed
    if len(sig_h) != config.SAMPLE_RATE:
        sig_h = resample(sig_h, config.SAMPLE_RATE)
        sig_v = resample(sig_v, config.SAMPLE_RATE)
    
    # Extract window
    window_size = config.WINDOW_SIZE
    if len(sig_h) >= window_size:
        start = (len(sig_h) - window_size) // 2
        sig_h = sig_h[start:start+window_size]
        sig_v = sig_v[start:start+window_size]
    
    vibration = np.stack([sig_h, sig_v], axis=0)
    
    # Extract features
    print("Extracting features...")
    features = extract_all_features(vibration, fs=config.SAMPLE_RATE)
    
    # Add operating conditions
    if config.USE_OPERATING_CONDITIONS:
        norm_speed = speed / config.MAX_SPEED
        norm_load = load / config.MAX_LOAD
        norm_temp = temp / config.MAX_TEMP
        features = np.concatenate([features, [norm_speed, norm_load, norm_temp]])
    
    print(f"Feature vector: {len(features)} dimensions")
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict (raw output is in transformed space)
    print("Running inference...")
    raw_pred = model.predict(features_scaled)
    
    # Inverse transform if model was trained with log-RUL
    predicted_rul = inverse_transform_rul(raw_pred, saved_config)[0]
    
    # Clamp to reasonable range
    predicted_rul = max(0, min(predicted_rul, config.PRED_MAX))
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"  Predicted RUL:  {predicted_rul:.1f} cycles")
    print(f"  RUL (%):        {(predicted_rul/config.RUL_CAP)*100:.1f}%")
    
    if predicted_rul > 500:
        status = "HEALTHY"
    elif predicted_rul > 200:
        status = "DEGRADED"
    else:
        status = "CRITICAL"
    print(f"  Status:         {status}")
    
    if val_metrics:
        mae = val_metrics.get('mae', 0)
        print(f"\n  Expected error: ±{mae:.0f} cycles (based on validation)")
    
    print("="*80)
    
    return predicted_rul


def random_test(model_path='rf_models/random_forest_model.pkl'):
    """Pick a random CSV from the datasets and test the model"""
    config = RFConfig()
    
    print("="*80)
    print("RANDOM FILE TEST")
    print("="*80)
    
    info = pick_random_csv(config)
    if info is None:
        return
    
    print(f"\n  Dataset:   {info['dataset']}")
    print(f"  Condition: {info['condition']}")
    print(f"  Bearing:   {info['bearing']}")
    print(f"  File:      {os.path.basename(info['csv_path'])} (#{info['file_index']+1} of {info['total_files']})")
    print(f"  True RUL:  {info['true_rul']} cycles")
    print(f"  Speed:     {info['speed']} RPM | Load: {info['load']} N")
    print()
    
    predicted_rul = predict_rul(
        info['csv_path'], model_path,
        speed=info['speed'], load=info['load'], temp=info['temp']
    )
    
    # Compare
