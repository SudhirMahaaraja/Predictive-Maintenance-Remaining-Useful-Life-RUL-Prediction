"""
Data loading and preprocessing for Random Forest bearing RUL prediction
Each bearing file becomes one sample (not sliding windows)
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import resample
import re
from rf_features import extract_all_features, aggregate_features


def natural_sort_key(s):
    """Sort strings with numbers correctly"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def load_femto_bearing(bearing_path, bearing_id, config):
    """
    Load one FEMTO bearing and extract features from all CSV files
    
    Returns:
        features: (n_files, feature_dim) array
        ruls: (n_files,) array of RUL values
        metadata: dict with bearing info
    """
    csv_files = sorted([f for f in os.listdir(bearing_path) 
                       if f.startswith('acc') and f.endswith('.csv')],
                       key=natural_sort_key)
    
    if len(csv_files) == 0:
        return None
    
    # Get operating conditions
    conditions = config.FEMTO_CONDITIONS.get(bearing_id, {'speed': 1800, 'load': 4000})
    speed = conditions['speed']
    load = conditions['load']
    
    total_cycles = len(csv_files)
    features_list = []
    ruls = []
    
    for idx, csv_file in enumerate(csv_files):
        file_path = os.path.join(bearing_path, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            
            # Extract vibration columns
            if 'horizontal_vibration' in df.columns:
                horiz = df['horizontal_vibration'].values
                vert = df['vertical_vibration'].values
            elif df.shape[1] >= 6:
                horiz = df.iloc[:, 4].values
                vert = df.iloc[:, 5].values
            else:
                horiz = df.iloc[:, 0].values
                vert = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
            
            # Resample to target rate
            if len(horiz) != config.SAMPLE_RATE:
                horiz = resample(horiz, config.SAMPLE_RATE)
                vert = resample(vert, config.SAMPLE_RATE)
            
            vibration = np.stack([horiz, vert], axis=0).astype(np.float32)
            
            # Extract features from a window
            window_size = config.WINDOW_SIZE
            if vibration.shape[1] >= window_size:
                start = (vibration.shape[1] - window_size) // 2
                vib_window = vibration[:, start:start+window_size]
            else:
                vib_window = vibration
            
            feats = extract_all_features(vib_window, fs=config.SAMPLE_RATE)
            
            # Add operating conditions if enabled
            if config.USE_OPERATING_CONDITIONS:
                norm_speed = speed / config.MAX_SPEED
                norm_load = load / config.MAX_LOAD
                temp = 35.0  # Estimated for FEMTO
                norm_temp = temp / config.MAX_TEMP
                feats = np.concatenate([feats, [norm_speed, norm_load, norm_temp]])
            
            features_list.append(feats)
            
            # Calculate RUL
            rul = total_cycles - idx - 1
            ruls.append(rul)
            
        except Exception as e:
            print(f"    Error loading {csv_file}: {e}")
            continue
    
    if len(features_list) == 0:
        return None
    
    features = np.array(features_list)
    ruls = np.array(ruls)
    
