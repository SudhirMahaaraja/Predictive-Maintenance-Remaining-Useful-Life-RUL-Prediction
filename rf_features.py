"""
Feature extraction for Random Forest bearing RUL prediction
Extracts physics-based features from vibration signals
"""
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


def extract_time_features(vibration):
    """Extract time-domain features (16 features total)"""
    features = []
    
    for ch in range(2):  # Horizontal and vertical
        sig = vibration[ch]
        
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        rms = np.sqrt(np.mean(sig**2))
        peak = np.max(np.abs(sig))
        
        features.extend([
            mean_val,
            std_val,
            rms,
            peak,
            kurtosis(sig),
            skew(sig),
            peak / (rms + 1e-8),  # Crest factor
            rms / (np.mean(np.abs(sig)) + 1e-8)  # Shape factor
        ])
    
    return np.array(features, dtype=np.float32)


def extract_freq_features(vibration, fs=25600):
    """Extract frequency-domain features (16 features total)"""
    features = []
    f_nyquist = fs / 2.0
    
    for ch in range(2):
