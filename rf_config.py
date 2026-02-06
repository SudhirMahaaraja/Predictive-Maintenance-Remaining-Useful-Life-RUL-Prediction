"""
Configuration for Random Forest Bearing RUL Prediction
Standalone baseline - no deep learning dependencies
"""
import os

class RFConfig:
    # Paths
    FEMTO_PATH = "datasets/FEMTO datasets/"
    XJTU_PATH = "datasets/XJTU-SY_Bearing_Datasets/"
    OUTPUT_PATH = "./rf_output"
    MODEL_SAVE_PATH = "./rf_models"
    
    # Data Parameters
    SAMPLE_RATE = 25600
    XJTU_SAMPLE_RATE = 25600
    WINDOW_SIZE = 2560  # For feature extraction
    
    # Feature Extraction
    USE_TIME_DOMAIN = True
    USE_FREQ_DOMAIN = True
    USE_ENVELOPE = True
    USE_OPERATING_CONDITIONS = True
    
    # Normalization
    MAX_SPEED = 2400.0
    MAX_LOAD = 12000.0
    MAX_TEMP = 80.0
    RUL_MAX = 3000.0
    
    # Training
    EPOCHS = 20  # Number of training rounds
    
    # RUL Preprocessing (fixes low-RUL underprediction & extreme errors)
    RUL_CAP = 1500          # Cap max RUL to reduce noise from very high values
    USE_LOG_RUL = False     # Disabled: log compresses high-RUL too aggressively, kills R²
    LOG_OFFSET = 1.0        # Offset for log(RUL + offset) to avoid log(0)
    
    # Sample Weighting (gives more importance to critical low-RUL samples)
    USE_SAMPLE_WEIGHTS = True
    WEIGHT_SCHEME = 'inverse_sqrt'  # 'inverse', 'inverse_sqrt', 'exponential'
    WEIGHT_MIN = 1.0        # Minimum weight for high-RUL samples
    WEIGHT_MAX = 5.0        # Maximum weight for low-RUL samples
    CRITICAL_RUL = 200      # RUL threshold below which samples get max weight
    
    # Post-processing (reduces extreme errors)
    CLIP_PREDICTIONS = True
    PRED_MIN = 0.0
    PRED_MAX = 1500.0       # Match RUL_CAP
    
    # Random Forest Hyperparameters
    N_ESTIMATORS = 500  # Number of trees
    MAX_DEPTH = 30  # Maximum tree depth (None = unlimited)
    MIN_SAMPLES_SPLIT = 5  # Minimum samples to split a node
    MIN_SAMPLES_LEAF = 2  # Minimum samples in leaf node
    MAX_FEATURES = 'sqrt'  # Features per split: 'sqrt', 'log2', or int
    N_JOBS = -1  # Use all CPU cores
    RANDOM_STATE = 42
    
    # XGBoost Hyperparameters (alternative)
    XGB_N_ESTIMATORS = 300
    XGB_MAX_DEPTH = 10
    XGB_LEARNING_RATE = 0.05
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE_BYTREE = 0.8
    
    # Data Split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    SEED = 42
    
    # Feature Aggregation Strategy
    # Options: 'last', 'mean', 'std', 'trend', 'all'
    AGGREGATION = 'all'  # Use multiple aggregations
    
    # Operating Conditions
    FEMTO_CONDITIONS = {
        'Bearing1_1': {'speed': 1800, 'load': 4000}, 'Bearing1_2': {'speed': 1800, 'load': 4000},
        'Bearing1_3': {'speed': 1800, 'load': 4000}, 'Bearing1_4': {'speed': 1800, 'load': 4000},
        'Bearing1_5': {'speed': 1800, 'load': 4000}, 'Bearing1_6': {'speed': 1800, 'load': 4000},
        'Bearing1_7': {'speed': 1800, 'load': 4000},
        'Bearing2_1': {'speed': 1650, 'load': 4200}, 'Bearing2_2': {'speed': 1650, 'load': 4200},
        'Bearing2_3': {'speed': 1650, 'load': 4200}, 'Bearing2_4': {'speed': 1650, 'load': 4200},
        'Bearing2_5': {'speed': 1650, 'load': 4200}, 'Bearing2_6': {'speed': 1650, 'load': 4200},
        'Bearing2_7': {'speed': 1650, 'load': 4200},
        'Bearing3_1': {'speed': 1500, 'load': 5000}, 'Bearing3_2': {'speed': 1500, 'load': 5000},
        'Bearing3_3': {'speed': 1500, 'load': 5000}
    }
    
    XJTU_CONDITIONS = {
        '35Hz12kN':   {'speed': 2100, 'load': 12000},
        '37.5Hz11kN': {'speed': 2250, 'load': 11000},
        '40Hz10kN':   {'speed': 2400, 'load': 10000}
    }
