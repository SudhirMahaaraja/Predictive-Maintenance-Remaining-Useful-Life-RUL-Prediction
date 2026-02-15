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
