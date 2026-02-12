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
    
    metadata = {
        'bearing_id': bearing_id,
        'dataset': 'FEMTO',
        'speed': speed,
        'load': load,
        'n_samples': len(features)
    }
    
    return features, ruls, metadata


def load_xjtu_bearing(bearing_path, bearing_id, condition, config):
    """Load one XJTU-SY bearing"""
    csv_files = sorted([f for f in os.listdir(bearing_path) if f.endswith('.csv')],
                      key=natural_sort_key)
    
    if len(csv_files) == 0:
        return None
    
    conditions = config.XJTU_CONDITIONS.get(condition, {'speed': 2100, 'load': 12000})
    speed = conditions['speed']
    load = conditions['load']
    
    total_cycles = len(csv_files)
    features_list = []
    ruls = []
    
    for idx, csv_file in enumerate(csv_files):
        file_path = os.path.join(bearing_path, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            
            if df.shape[1] >= 2:
                horiz = df.iloc[:, 0].values
                vert = df.iloc[:, 1].values
            else:
                horiz = df.iloc[:, 0].values
                vert = df.iloc[:, 0].values
            
            if len(horiz) != config.XJTU_SAMPLE_RATE:
                horiz = resample(horiz, config.XJTU_SAMPLE_RATE)
                vert = resample(vert, config.XJTU_SAMPLE_RATE)
            
            vibration = np.stack([horiz, vert], axis=0).astype(np.float32)
            
            window_size = config.WINDOW_SIZE
            if vibration.shape[1] >= window_size:
                start = (vibration.shape[1] - window_size) // 2
                vib_window = vibration[:, start:start+window_size]
            else:
                vib_window = vibration
            
            feats = extract_all_features(vib_window, fs=config.XJTU_SAMPLE_RATE)
            
            if config.USE_OPERATING_CONDITIONS:
                norm_speed = speed / config.MAX_SPEED
                norm_load = load / config.MAX_LOAD
                temp = 25.0 + (idx / total_cycles) * 15.0
                norm_temp = temp / config.MAX_TEMP
                feats = np.concatenate([feats, [norm_speed, norm_load, norm_temp]])
            
            features_list.append(feats)
            
            rul = total_cycles - idx - 1
            ruls.append(rul)
            
        except Exception as e:
            print(f"    Error loading {csv_file}: {e}")
            continue
    
    if len(features_list) == 0:
        return None
    
    features = np.array(features_list)
    ruls = np.array(ruls)
    
    metadata = {
        'bearing_id': f"{condition}_{bearing_id}",
        'dataset': 'XJTU',
        'speed': speed,
        'load': load,
        'n_samples': len(features)
    }
    
    return features, ruls, metadata


def load_all_data(config):
    """
    Load all bearing data from FEMTO and XJTU-SY datasets
    
    Returns:
        all_data: List of (features, ruls, metadata) tuples
    """
    all_data = []
    
    print("Loading datasets...")
    
    # Load FEMTO
    for dataset_type in ['Learning_set', 'Test_set', 'Full_Test_Set']:
        data_path = os.path.join(config.FEMTO_PATH, dataset_type)
        if not os.path.exists(data_path):
            continue
        
        bearing_folders = [f for f in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, f))]
        
        for bearing_id in bearing_folders:
            bearing_path = os.path.join(data_path, bearing_id)
            print(f"  Loading {dataset_type}/{bearing_id}...")
            
            result = load_femto_bearing(bearing_path, bearing_id, config)
            if result is not None:
                features, ruls, metadata = result
                metadata['bearing_id'] = f"{dataset_type}_{bearing_id}"
                all_data.append((features, ruls, metadata))
                print(f"    Loaded {len(features)} samples, RUL range: {ruls.min()}-{ruls.max()}")
    
    # Load XJTU
    for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
        data_path = os.path.join(config.XJTU_PATH, condition)
        if not os.path.exists(data_path):
            continue
        
        bearing_folders = [f for f in os.listdir(data_path)
                          if os.path.isdir(os.path.join(data_path, f))]
        
        for bearing_id in bearing_folders:
            bearing_path = os.path.join(data_path, bearing_id)
            print(f"  Loading {condition}/{bearing_id}...")
            
            result = load_xjtu_bearing(bearing_path, bearing_id, condition, config)
            if result is not None:
                features, ruls, metadata = result
                all_data.append((features, ruls, metadata))
                print(f"    Loaded {len(features)} samples, RUL range: {ruls.min()}-{ruls.max()}")
    
    print(f"\n✓ Loaded {len(all_data)} bearings total")
    
    return all_data


def create_train_val_test_split(all_data, config):
    """
    Split data by bearing (not by samples) with stratification by lifetime
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(config.SEED)
    
    # Calculate lifetime for each bearing
    bearing_info = []
    for features, ruls, metadata in all_data:
        lifetime = len(features)
        bearing_info.append({
            'data': (features, ruls, metadata),
            'lifetime': lifetime,
            'bearing_id': metadata['bearing_id']
        })
    
    # Sort by lifetime for stratification
    bearing_info.sort(key=lambda x: x['lifetime'])
    
    # Stratified split
    n_bearings = len(bearing_info)
    train_data, val_data, test_data = [], [], []
    
    # Round-robin assignment
    for i, info in enumerate(bearing_info):
        if i % 10 < 7:  # 70% train
            train_data.append(info['data'])
        elif i % 10 < 8.5:  # 15% val
            val_data.append(info['data'])
        else:  # 15% test
            test_data.append(info['data'])
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} bearings")
    print(f"  Val:   {len(val_data)} bearings")
    print(f"  Test:  {len(test_data)} bearings")
    
    # Flatten into (X, y) arrays
    def flatten_data(data_list):
        X_list, y_list = [], []
        for features, ruls, metadata in data_list:
            X_list.append(features)
            y_list.append(ruls)
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y
    
    X_train, y_train = flatten_data(train_data)
    X_val, y_val = flatten_data(val_data)
    X_test, y_test = flatten_data(test_data)
    
    print(f"\nSample counts:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
