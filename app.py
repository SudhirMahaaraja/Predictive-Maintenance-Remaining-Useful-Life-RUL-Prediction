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
            spine.set_color('#8a9490')
            spine.set_alpha(0.3)
        ax.axvline(x=0, color='#8a9490', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=120)
        plt.close()
        buf.seek(0)
        shap_plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            'base_value': round(base_value, 2),
            'top_features': top_features,
            'shap_plot_b64': shap_plot_b64,
        }

    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None
    except Exception as e:
        print(f"SHAP computation error: {e}\n{traceback.format_exc()}")
        return None


# ─── Feature names for SHAP ──────────────────────────────────────────────────

FEATURE_NAMES = [
    'Horiz_Mean', 'Horiz_Std', 'Horiz_RMS', 'Horiz_Peak',
    'Horiz_Kurtosis', 'Horiz_Skewness', 'Horiz_CrestFactor', 'Horiz_ShapeFactor',
    'Vert_Mean', 'Vert_Std', 'Vert_RMS', 'Vert_Peak',
    'Vert_Kurtosis', 'Vert_Skewness', 'Vert_CrestFactor', 'Vert_ShapeFactor',
    'Horiz_FreqCentroid', 'Horiz_FreqSpread', 'Horiz_DomFreq',
    'Horiz_Band_FTF', 'Horiz_Band_BSF', 'Horiz_Band_BPFO', 'Horiz_Band_Mid', 'Horiz_Band_High',
    'Vert_FreqCentroid', 'Vert_FreqSpread', 'Vert_DomFreq',
    'Vert_Band_FTF', 'Vert_Band_BSF', 'Vert_Band_BPFO', 'Vert_Band_Mid', 'Vert_Band_High',
    'Horiz_EnvMean', 'Horiz_EnvStd', 'Horiz_EnvMax', 'Horiz_EnvKurtosis',
    'Vert_EnvMean', 'Vert_EnvStd', 'Vert_EnvMax', 'Vert_EnvKurtosis',
    'Speed_RPM', 'Load_N', 'Temperature'
]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/eda', methods=['GET'])
def api_eda():
    """Return EDA results (run if not cached)"""
    global _eda_cache, _eda_running

    force = request.args.get('force', 'false').lower() == 'true'

    if force:
        _eda_cache = None

    if _eda_cache is not None:
        return jsonify({'status': 'ready', 'data': _eda_cache})

    if _eda_running:
        return jsonify({'status': 'running'})

    # Try loading from disk cache
    cache_path = './rf_output/eda_cache.json'
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                _eda_cache = json.load(f)
            return jsonify({'status': 'ready', 'data': _eda_cache})
        except Exception:
            pass

    # Launch EDA in background thread
    _eda_running = True

    def run_eda():
        global _eda_cache, _eda_running
        try:
            from eda import run_full_eda
            result = run_full_eda()
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            with _eda_lock:
                _eda_cache = result
        except Exception as e:
            print(f"EDA error: {e}\n{traceback.format_exc()}")
        finally:
            _eda_running = False

    t = threading.Thread(target=run_eda, daemon=True)
    t.start()

    return jsonify({'status': 'started'})


@app.route('/api/eda/status', methods=['GET'])
def api_eda_status():
    """Poll EDA progress"""
    global _eda_cache, _eda_running
    if _eda_cache is not None:
        return jsonify({'status': 'ready'})
    if _eda_running:
        return jsonify({'status': 'running'})
    return jsonify({'status': 'idle'})


@app.route('/api/models', methods=['GET'])
def api_models():
    """Return list of available models and their metrics"""
    models_info = {}
    for mtype, fname in [('rf', 'random_forest_model.pkl'), ('xgb', 'xgboost_model.pkl')]:
        path = f'./rf_models/{fname}'
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            ckpt = load_model(mtype)
            metrics = ckpt.get('val_metrics', {}) if ckpt else {}
            models_info[mtype] = {
                'available': True,
                'size_mb': round(size_mb, 1),
                'val_metrics': metrics
            }
        else:
            models_info[mtype] = {'available': False}

    # Also load test results
    results_path = './rf_output/test_results.json'
    test_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            test_results = json.load(f)

    return jsonify({'models': models_info, 'test_results': test_results})


@app.route('/api/predict/csv', methods=['POST'])
def predict_csv():
    """Predict RUL from uploaded CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    model_type = request.form.get('model', 'rf')
    speed = float(request.form.get('speed', 1800))
    load  = float(request.form.get('load', 4000))
    temp  = float(request.form.get('temp', 35))

    try:
        fname = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(save_path)

        result = _predict_from_csv(save_path, model_type, speed, load, temp)
        os.remove(save_path)
        
        # Add EDA plot
        global _eda_cache
        if _eda_cache and 'plots' in _eda_cache:
            result['eda_plot_b64'] = _eda_cache['plots'].get('feature_importance')
            
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/predict/manual', methods=['POST'])
def predict_manual():
    """Predict RUL from manually entered feature values"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    model_type = data.get('model', 'rf')
    speed = float(data.get('speed', 1800))
    load  = float(data.get('load', 4000))
