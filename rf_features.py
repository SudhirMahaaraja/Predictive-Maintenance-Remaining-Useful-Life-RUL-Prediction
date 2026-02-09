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
        sig = vibration[ch]
        
        fft_vals = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1/fs)
        magnitude = np.abs(fft_vals)
        
        total_mag = np.sum(magnitude) + 1e-8
        mag_norm = magnitude / total_mag
        
        centroid = np.sum(freqs * mag_norm)
        variance = np.sum(((freqs - centroid)**2) * mag_norm)
        dominant_freq = freqs[np.argmax(magnitude)]
        
        features.extend([
            centroid / f_nyquist,
            np.sqrt(variance + 1e-8) / f_nyquist,
            dominant_freq / f_nyquist
        ])
        
        # Band powers (log-scaled)
        bands = [(10, 50), (60, 180), (100, 300), (300, 1000), (1000, 3000)]
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(magnitude[mask] ** 2)
            features.append(np.log1p(band_power))
    
    return np.array(features, dtype=np.float32)


def extract_envelope_features(vibration, fs=25600):
    """Extract envelope features (8 features total)"""
    features = []
    
    for ch in range(2):
        sig = vibration[ch]
        
        sos = signal.butter(4, 1000, 'hp', fs=fs, output='sos')
        sig_filtered = signal.sosfilt(sos, sig)
        
        analytic = signal.hilbert(sig_filtered)
        envelope = np.abs(analytic)
        
        features.extend([
            np.mean(envelope),
            np.std(envelope),
            np.max(envelope),
            kurtosis(envelope)
        ])
    
    return np.array(features, dtype=np.float32)


def extract_all_features(vibration, fs=25600):
    """
    Extract all features from a single vibration window
    Returns: 40 features (16 time + 16 freq + 8 envelope)
    """
    try:
        time_feat = extract_time_features(vibration)
        freq_feat = extract_freq_features(vibration, fs)
        env_feat = extract_envelope_features(vibration, fs)
        
        all_feat = np.concatenate([time_feat, freq_feat, env_feat])
        
        # Replace any NaN/Inf
        all_feat = np.nan_to_num(all_feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return all_feat
    except Exception as e:
        print(f"Warning: Feature extraction failed: {e}")
        return np.zeros(40, dtype=np.float32)


def aggregate_features(features_list, method='all'):
    """
    Aggregate features from multiple windows into a single feature vector
    
    Args:
        features_list: List of feature arrays from different time windows
        method: 'last', 'mean', 'std', 'trend', or 'all'
    
    Returns:
        Aggregated feature vector
    """
    if len(features_list) == 0:
        return np.zeros(40, dtype=np.float32)
