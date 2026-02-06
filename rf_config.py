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
