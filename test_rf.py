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
    true_rul = info['true_rul']
    error = predicted_rul - true_rul
    pct_error = abs(error / (true_rul + 1)) * 100
    
    print("\n" + "="*80)
    print("ACCURACY CHECK")
    print("="*80)
    print(f"  True RUL:      {true_rul} cycles")
    print(f"  Predicted RUL: {predicted_rul:.1f} cycles")
    print(f"  Error:         {error:+.1f} cycles ({pct_error:.1f}%)")
    
    if abs(error) < 50:
        verdict = "🟢 EXCELLENT"
    elif abs(error) < 150:
        verdict = "🟡 GOOD"
    elif abs(error) < 350:
        verdict = "🟠 MODERATE"
    else:
        verdict = "🔴 POOR"
    print(f"  Verdict:       {verdict}")
    print("="*80)
    
    return predicted_rul, true_rul


def batch_predict(bearing_folder, model_path='rf_models/random_forest_model.pkl',
                  speed=1800, load=4000, temp=35):
    """
    Predict RUL for all CSV files in a bearing folder
    Useful for validating degradation trend
    """
    csv_files = sorted([f for f in os.listdir(bearing_folder) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {bearing_folder}")
        return
    
    print(f"Found {len(csv_files)} files in bearing folder")
    print("Predicting RUL for each file...\n")
    
    predictions = []
    true_ruls = []
    
    total_cycles = len(csv_files)
    
    for i, csv_file in enumerate(csv_files):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(csv_files)}")
        
        csv_path = os.path.join(bearing_folder, csv_file)
        try:
            pred_rul = predict_rul(csv_path, model_path, speed, load, temp)
            predictions.append(pred_rul)
            
            true_rul = total_cycles - i - 1
            true_ruls.append(true_rul)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            predictions.append(np.nan)
            true_ruls.append(total_cycles - i - 1)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    x = np.arange(len(predictions))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, true_ruls, 'b-', label='True RUL', linewidth=2)
    plt.plot(x, predictions, 'r--', label='Predicted RUL', linewidth=2, alpha=0.7)
    plt.xlabel('Cycle Index')
    plt.ylabel('RUL (cycles)')
    plt.title('RUL Prediction Over Bearing Lifetime')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    errors = np.array(predictions) - np.array(true_ruls)
    plt.scatter(true_ruls, errors, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True RUL (cycles)')
    plt.ylabel('Prediction Error (cycles)')
    plt.title('Prediction Error vs True RUL')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'rf_output/batch_prediction.png'
    os.makedirs('rf_output', exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Saved prediction plot: {output_path}")
    
    # Calculate metrics
    valid_mask = ~np.isnan(predictions)
    if np.sum(valid_mask) > 0:
        mae = np.mean(np.abs(np.array(predictions)[valid_mask] - np.array(true_ruls)[valid_mask]))
        print(f"\nOverall MAE: {mae:.1f} cycles")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict bearing RUL')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--folder', type=str, help='Folder containing multiple CSV files (batch mode)')
    parser.add_argument('--random', action='store_true', help='Pick a random CSV from datasets and test')
    parser.add_argument('--model', type=str, default='rf_models/random_forest_model.pkl',
                       help='Path to model file')
    parser.add_argument('--speed', type=float, default=1800, help='Operating speed (RPM)')
    parser.add_argument('--load', type=float, default=4000, help='Operating load (N)')
    parser.add_argument('--temp', type=float, default=35, help='Operating temperature (°C)')
    
    args = parser.parse_args()
    
    if args.random:
        # Random test mode
        random_test(args.model)
    elif args.folder:
        # Batch mode
        batch_predict(args.folder, args.model, args.speed, args.load, args.temp)
    elif args.csv:
        # Single file mode
        predict_rul(args.csv, args.model, args.speed, args.load, args.temp)
    else:
        # Default: random test
        print("No input specified. Running random test...\n")
        random_test(args.model)
