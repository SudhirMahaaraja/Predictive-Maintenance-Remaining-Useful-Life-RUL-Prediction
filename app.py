"""
Flask backend for Bearing RUL Prediction Dashboard
Serves EDA plots, model metrics, and handles CSV/manual prediction requests
Enhanced with: Confidence Intervals, SHAP, Ensemble, Comparison
"""
import os
import io
import json
import pickle
import base64
import traceback
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from scipy.signal import resample
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rf_config import RFConfig
from rf_features import extract_all_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
app.config['UPLOAD_FOLDER'] = './rf_output/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Globals ─────────────────────────────────────────────────────────────────
_eda_cache   = None
_eda_lock    = threading.Lock()
_eda_running = False

config = RFConfig()

# Cache for SHAP explainers (expensive to create)
_shap_cache = {}


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(model_type='rf'):
    path_map = {
        'rf':  './rf_models/random_forest_model.pkl',
        'xgb': './rf_models/xgboost_model.pkl',
    }
    path = path_map.get(model_type)
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def inverse_transform_rul(y_pred, cfg):
    y_out = np.copy(y_pred)
    if cfg.USE_LOG_RUL:
        y_out = np.expm1(y_out) - cfg.LOG_OFFSET
    if cfg.CLIP_PREDICTIONS:
        y_out = np.clip(y_out, cfg.PRED_MIN, cfg.PRED_MAX)
    return y_out


def rul_to_status(rul):
    if rul > 500:
        return 'HEALTHY', 'success'
    elif rul > 200:
        return 'DEGRADED', 'warning'
    else:
        return 'CRITICAL', 'danger'


# ─── Confidence Interval Helper ──────────────────────────────────────────────

def _get_confidence_interval(model, model_type, features_scaled, cfg):
    """Compute prediction confidence interval using individual tree predictions (RF)
       or bootstrap approach (XGBoost)."""
    try:
        if model_type == 'rf' and hasattr(model, 'estimators_'):
            # Use individual tree predictions for RF
            tree_preds = np.array([
                tree.predict(features_scaled)[0]
                for tree in model.estimators_
            ])
            tree_preds = inverse_transform_rul(tree_preds, cfg)
            tree_preds = np.clip(tree_preds, 0, cfg.PRED_MAX)

            mean_pred = float(np.mean(tree_preds))
            std_pred = float(np.std(tree_preds))
            ci_lower = float(max(0, mean_pred - 1.96 * std_pred))
            ci_upper = float(min(cfg.PRED_MAX, mean_pred + 1.96 * std_pred))

            return {
                'ci_lower': round(ci_lower, 1),
                'ci_upper': round(ci_upper, 1),
                'ci_std': round(std_pred, 1),
                'ci_method': 'RF Tree Variance',
                'n_estimators': len(model.estimators_),
            }
        elif model_type == 'xgb':
            # For XGBoost, use the model's validation MAE as proxy uncertainty
            ckpt = load_model('xgb')
            val_metrics = ckpt.get('val_metrics', {}) if ckpt else {}
            mae = val_metrics.get('mae', 150)
            raw_pred = model.predict(features_scaled)
            pred_val = float(inverse_transform_rul(raw_pred, cfg)[0])
            pred_val = max(0, min(pred_val, cfg.PRED_MAX))

            return {
                'ci_lower': round(max(0, pred_val - mae), 1),
                'ci_upper': round(min(cfg.PRED_MAX, pred_val + mae), 1),
                'ci_std': round(mae, 1),
                'ci_method': 'Validation MAE Proxy',
                'n_estimators': int(getattr(model, 'n_estimators', 0)),
            }
    except Exception as e:
        print(f"CI computation error: {e}")

    return None


# ─── SHAP Helper ─────────────────────────────────────────────────────────────

def _get_shap_explanation(model, model_type, features_scaled, feature_names=None):
    """Compute SHAP values for a single prediction."""
    try:
        import shap

        cache_key = model_type
        if cache_key not in _shap_cache:
            if model_type == 'rf':
                _shap_cache[cache_key] = shap.TreeExplainer(model)
            elif model_type == 'xgb':
                _shap_cache[cache_key] = shap.TreeExplainer(model)

        explainer = _shap_cache.get(cache_key)
        if explainer is None:
            return None

        shap_values = explainer.shap_values(features_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        # Handle expected_value being scalar or array
        ev = explainer.expected_value
        if hasattr(ev, '__len__') and len(ev) > 0:
            base_value = float(ev[0])
        else:
            base_value = float(ev)

        # Build top features list
        n_features = len(sv)
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(sv))[::-1]
        top_n = min(10, len(indices))
        top_features = []
        for i in range(top_n):
            idx = indices[i]
            top_features.append({
                'name': feature_names[idx] if idx < len(feature_names) else f'F{idx}',
                'shap_value': round(float(sv[idx]), 4),
                'feature_value': round(float(features_scaled[0, idx]), 4),
            })

        # Generate SHAP bar plot as base64
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='none')
        names = [f['name'] for f in top_features[:8]][::-1]
        vals = [f['shap_value'] for f in top_features[:8]][::-1]
        colors = ['#6F8F72' if v >= 0 else '#e85050' for v in vals]

        ax.barh(names, vals, color=colors, height=0.6, edgecolor='none')
        ax.set_facecolor('none')
        ax.tick_params(colors='#8a9490', labelsize=8)
        ax.set_xlabel('SHAP Value (impact on RUL)', fontsize=9, color='#8a9490')
        for spine in ax.spines.values():
