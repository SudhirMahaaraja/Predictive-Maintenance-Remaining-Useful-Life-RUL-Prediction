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
    temp  = float(data.get('temp', 35))

    # Build a synthetic vibration signal from the provided parameters
    rms       = float(data.get('rms', 0.5))
    kurt      = float(data.get('kurtosis', 3.0))
    dom_freq  = float(data.get('dominant_freq', 120.0))
    env_rms   = float(data.get('envelope_rms', 0.3))

    try:
        features, sig_h, sig_v = _build_features_from_manual(
            rms, kurt, dom_freq, env_rms, speed, load, temp
        )

        plot_b64 = _generate_signal_plot(sig_h, sig_v)
        result = _predict_features(features, model_type, speed, load, temp)
        result['plot_b64'] = plot_b64
        
        # Add EDA plot
        global _eda_cache
        if _eda_cache and 'plots' in _eda_cache:
            result['eda_plot_b64'] = _eda_cache['plots'].get('feature_importance')
            
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ─── NEW: Side-by-Side Comparison Endpoint ────────────────────────────────────

@app.route('/api/predict/compare', methods=['POST'])
def predict_compare():
    """Run both RF and XGBoost on the same input, return side-by-side results."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    speed = float(data.get('speed', 1800))
    load  = float(data.get('load', 4000))
    temp  = float(data.get('temp', 35))
    mode  = data.get('mode', 'manual')

    try:
        if mode == 'manual':
            rms      = float(data.get('rms', 0.5))
            kurt     = float(data.get('kurtosis', 3.0))
            dom_freq = float(data.get('dominant_freq', 120.0))
            env_rms  = float(data.get('envelope_rms', 0.3))

            features, sig_h, sig_v = _build_features_from_manual(
                rms, kurt, dom_freq, env_rms, speed, load, temp
            )
            plot_b64 = _generate_signal_plot(sig_h, sig_v)
        else:
            return jsonify({'error': 'Compare mode only supports manual input'}), 400

        results = {}
        for mt in ['rf', 'xgb']:
            try:
                r = _predict_features(features.copy(), mt, speed, load, temp)
                r['plot_b64'] = plot_b64
                results[mt] = r
            except Exception as e:
                results[mt] = {'error': str(e)}

        # Ensemble: average of the two
        if 'error' not in results.get('rf', {}) and 'error' not in results.get('xgb', {}):
            rf_rul = results['rf']['predicted_rul']
            xgb_rul = results['xgb']['predicted_rul']

            # Weighted average by validation R² if available
            rf_ckpt = load_model('rf')
            xgb_ckpt = load_model('xgb')
            rf_r2 = rf_ckpt.get('val_metrics', {}).get('r2', 0.5) if rf_ckpt else 0.5
            xgb_r2 = xgb_ckpt.get('val_metrics', {}).get('r2', 0.5) if xgb_ckpt else 0.5
            total_r2 = rf_r2 + xgb_r2
            w_rf = rf_r2 / total_r2 if total_r2 > 0 else 0.5
            w_xgb = xgb_r2 / total_r2 if total_r2 > 0 else 0.5

            ens_rul = round(rf_rul * w_rf + xgb_rul * w_xgb, 1)
            ens_status, ens_badge = rul_to_status(ens_rul)
            ens_pct = round((ens_rul / config.RUL_CAP) * 100, 1) if config.RUL_CAP else 0

            results['ensemble'] = {
                'predicted_rul': ens_rul,
                'rul_pct': ens_pct,
                'status': ens_status,
                'status_badge': ens_badge,
                'model_used': 'ENSEMBLE',
                'speed': speed,
                'load': load,
                'temp': temp,
                'n_features': results['rf'].get('n_features', 0),
                'weights': {'rf': round(w_rf, 3), 'xgb': round(w_xgb, 3)},
                'plot_b64': plot_b64,
            }

            # Ensemble confidence if both have CI
            rf_ci = results['rf'].get('confidence_interval')
            xgb_ci = results['xgb'].get('confidence_interval')
            if rf_ci and xgb_ci:
                results['ensemble']['confidence_interval'] = {
                    'ci_lower': round(min(rf_ci['ci_lower'], xgb_ci['ci_lower']), 1),
                    'ci_upper': round(max(rf_ci['ci_upper'], xgb_ci['ci_upper']), 1),
                    'ci_std': round((rf_ci['ci_std'] + xgb_ci['ci_std']) / 2, 1),
                    'ci_method': 'Ensemble Range',
                }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ─── NEW: Ensemble Prediction Endpoint ────────────────────────────────────────

@app.route('/api/predict/ensemble', methods=['POST'])
def predict_ensemble():
    """Average RF + XGBoost predictions, weighted by validation R²."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    speed = float(data.get('speed', 1800))
    load  = float(data.get('load', 4000))
    temp  = float(data.get('temp', 35))
    rms      = float(data.get('rms', 0.5))
    kurt     = float(data.get('kurtosis', 3.0))
    dom_freq = float(data.get('dominant_freq', 120.0))
    env_rms  = float(data.get('envelope_rms', 0.3))

    try:
        features, sig_h, sig_v = _build_features_from_manual(
            rms, kurt, dom_freq, env_rms, speed, load, temp
        )
        plot_b64 = _generate_signal_plot(sig_h, sig_v)

        rf_result = _predict_features(features.copy(), 'rf', speed, load, temp)
        xgb_result = _predict_features(features.copy(), 'xgb', speed, load, temp)

        # Weighted average
        rf_ckpt = load_model('rf')
        xgb_ckpt = load_model('xgb')
        rf_r2 = rf_ckpt.get('val_metrics', {}).get('r2', 0.5) if rf_ckpt else 0.5
        xgb_r2 = xgb_ckpt.get('val_metrics', {}).get('r2', 0.5) if xgb_ckpt else 0.5
        total = rf_r2 + xgb_r2
        w_rf = rf_r2 / total if total > 0 else 0.5
        w_xgb = xgb_r2 / total if total > 0 else 0.5

        ens_rul = round(rf_result['predicted_rul'] * w_rf + xgb_result['predicted_rul'] * w_xgb, 1)
        ens_status, ens_badge = rul_to_status(ens_rul)

        result = {
            'predicted_rul': ens_rul,
            'rul_pct': round((ens_rul / config.RUL_CAP) * 100, 1),
            'status': ens_status,
            'status_badge': ens_badge,
            'model_used': 'ENSEMBLE',
            'speed': speed, 'load': load, 'temp': temp,
            'n_features': rf_result.get('n_features', 0),
            'plot_b64': plot_b64,
            'weights': {'rf': round(w_rf, 3), 'xgb': round(w_xgb, 3)},
            'rf_rul': rf_result['predicted_rul'],
            'xgb_rul': xgb_result['predicted_rul'],
            'confidence_interval': rf_result.get('confidence_interval'),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ─── Helper: Build features from manual inputs ───────────────────────────────

def _build_features_from_manual(rms, kurt, dom_freq, env_rms, speed, load, temp):
    """Synthesize vibration signal and extract features."""
    fs = config.SAMPLE_RATE
    t  = np.linspace(0, 1, fs)
    noise = np.random.randn(fs) * rms
    sinusoidal = rms * np.sin(2 * np.pi * dom_freq * t)
    sig_h = noise + sinusoidal
    sig_v = noise * 0.9 + sinusoidal * 0.85

    sig_h = sig_h / (np.std(sig_h) + 1e-8) * rms
    sig_v = sig_v / (np.std(sig_v) + 1e-8) * rms * 0.9

    window_size = config.WINDOW_SIZE
    sig_h = sig_h[:window_size]
    sig_v = sig_v[:window_size]

    vibration = np.stack([sig_h, sig_v], axis=0)
    features  = extract_all_features(vibration, fs=fs)

    if config.USE_OPERATING_CONDITIONS:
        features = np.concatenate([features, [
            speed / config.MAX_SPEED,
            load  / config.MAX_LOAD,
            temp  / config.MAX_TEMP
        ]])

    return features, sig_h, sig_v


# ─── Helper prediction functions ─────────────────────────────────────────────

def _predict_from_csv(csv_path, model_type, speed, load, temp):
    ckpt = load_model(model_type)
    if ckpt is None:
        raise ValueError(f"Model '{model_type}' not found. Train the model first.")

    model  = ckpt['model']
    scaler = ckpt['scaler']
    cfg    = ckpt.get('config', config)

    df = pd.read_csv(csv_path)

    # Auto-detect columns
    h_col = next((c for c in df.columns if 'horiz' in c.lower() or 'horizontal' in c.lower()), None)
    v_col = next((c for c in df.columns if 'vert'  in c.lower() or 'vertical'   in c.lower()), None)
    if h_col is None or v_col is None:
        if len(df.columns) >= 2:
            h_col, v_col = df.columns[0], df.columns[1]
        elif len(df.columns) == 1:
            h_col = v_col = df.columns[0]
        else:
            raise ValueError("CSV must have at least one column")

    sig_h = df[h_col].dropna().values.astype(np.float32)
    sig_v = df[v_col].dropna().values.astype(np.float32) if h_col != v_col else sig_h.copy()

    if len(sig_h) != config.SAMPLE_RATE:
        sig_h = resample(sig_h, config.SAMPLE_RATE)
        sig_v = resample(sig_v, config.SAMPLE_RATE)

    window_size = config.WINDOW_SIZE
    if len(sig_h) >= window_size:
        start = (len(sig_h) - window_size) // 2
        sig_h = sig_h[start:start + window_size]
        sig_v = sig_v[start:start + window_size]

    vibration = np.stack([sig_h, sig_v], axis=0)
    features  = extract_all_features(vibration, fs=config.SAMPLE_RATE)

    if config.USE_OPERATING_CONDITIONS:
        features = np.concatenate([features, [
            speed / config.MAX_SPEED,
            load  / config.MAX_LOAD,
            temp  / config.MAX_TEMP
        ]])

    plot_b64 = _generate_signal_plot(sig_h, sig_v)
    result = _predict_features(features, model_type, speed, load, temp, model, scaler, cfg)
    result['plot_b64'] = plot_b64
    return result


def _generate_signal_plot(sig_h, sig_v):
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='none')
    ax.plot(sig_h, label='Horiz', alpha=0.9, color='#6F8F72', linewidth=1)
    ax.plot(sig_v, label='Vert', alpha=0.9, color='#F2A65A', linewidth=1)
    ax.set_facecolor('none')
    ax.tick_params(colors='#8a9490', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#8a9490')
        spine.set_alpha(0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.2)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _predict_features(features, model_type, speed, load, temp,
                       model=None, scaler=None, cfg=None):
    ckpt = None
    if model is None:
        ckpt = load_model(model_type)
        if ckpt is None:
            raise ValueError(f"Model '{model_type}' not found.")
        model  = ckpt['model']
        scaler = ckpt['scaler']
        cfg    = ckpt.get('config', config)

    features_scaled = scaler.transform(features.reshape(1, -1))
    raw_pred        = model.predict(features_scaled)
    predicted_rul   = float(inverse_transform_rul(raw_pred, cfg)[0])
    predicted_rul   = max(0.0, min(predicted_rul, float(cfg.PRED_MAX)))

    status, badge = rul_to_status(predicted_rul)
    pct = (predicted_rul / cfg.RUL_CAP) * 100 if cfg.RUL_CAP else 0

    result = {
        'predicted_rul': round(predicted_rul, 1),
        'rul_pct': round(pct, 1),
        'status': status,
        'status_badge': badge,
        'model_used': model_type.upper(),
        'speed': speed,
        'load': load,
        'temp': temp,
        'n_features': int(len(features)),
    }

    # ── Confidence Interval ──
    ci = _get_confidence_interval(model, model_type, features_scaled, cfg)
    if ci:
        result['confidence_interval'] = ci

    # ── SHAP Explanation ──
    shap_result = _get_shap_explanation(model, model_type, features_scaled, FEATURE_NAMES)
    if shap_result:
        result['shap'] = shap_result

    return result


# ─── NEW: Batch CSV Prediction ────────────────────────────────────────────────

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict RUL for multiple uploaded CSV files."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'Empty file list'}), 400

    model_type = request.form.get('model', 'rf')
    speed = float(request.form.get('speed', 1800))
    load  = float(request.form.get('load', 4000))
    temp  = float(request.form.get('temp', 35))

    results = []
    for file in files:
        fname = secure_filename(file.filename)
        if not fname:
            continue
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        try:
            file.save(save_path)
            r = _predict_from_csv(save_path, model_type, speed, load, temp)
            # Don't include heavy base64 plots for batch results
            r.pop('plot_b64', None)
            r.pop('shap', None)
            r['filename'] = fname
            results.append(r)
        except Exception as e:
            results.append({'filename': fname, 'error': str(e)})
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    return jsonify({'results': results, 'count': len(results)})


if __name__ == '__main__':
    print("Starting RUL Dashboard on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
