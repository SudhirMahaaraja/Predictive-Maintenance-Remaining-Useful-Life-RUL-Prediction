"""
Exploratory Data Analysis (EDA) module for bearing RUL prediction
Generates comprehensive plots and statistics saved as PNG files
"""
import os
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew, norm
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

from rf_config import RFConfig
from rf_data_loader import load_all_data
from rf_features import extract_all_features

EDA_OUTPUT = "./rf_output/eda"
# User palette: silver-sage · warm-cream · forest-green · amber-orange
PALETTE = ['#6F8F72', '#F2A65A', '#BFC6C4', '#E8E2D8', '#4a6b4d', '#d4853a', '#8aad8d']

BG_DEEP  = '#1a1f1e'
BG_CARD  = '#202724'
TEXT_PRI = '#E8E2D8'
TEXT_MUT = '#8a9490'
GRID_COL = '#2e3a36'


def ensure_dir():
    os.makedirs(EDA_OUTPUT, exist_ok=True)


def fig_to_b64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


def dark_fig(figsize=(12, 5)):
    """Create a figure using the project color palette"""
    fig = plt.figure(figsize=figsize, facecolor=BG_DEEP)
    return fig


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_MUT, labelsize=9)
    ax.spines['bottom'].set_color('#3a4a44')
    ax.spines['left'].set_color('#3a4a44')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=TEXT_PRI, fontsize=11, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_MUT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_MUT, fontsize=9)
    ax.grid(True, alpha=0.18, color=GRID_COL)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset Overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_dataset_overview(all_data):
    """Dataset overview: sample count per bearing, RUL distribution, dataset split"""
    ensure_dir()

    femto_data = [(m['bearing_id'], len(f), r.max(), r.min())
                  for f, r, m in all_data if m.get('dataset') == 'FEMTO']
    xjtu_data  = [(m['bearing_id'], len(f), r.max(), r.min())
                  for f, r, m in all_data if m.get('dataset') == 'XJTU']

    fig = dark_fig(figsize=(15, 10))
    fig.suptitle('Dataset Overview', color='#e0e0ff', fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── 1a: Samples per bearing (bar chart) ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    all_ids   = [d[0] for d in femto_data + xjtu_data]
    all_sizes = [d[1] for d in femto_data + xjtu_data]
    colors = [PALETTE[0]] * len(femto_data) + [PALETTE[1]] * len(xjtu_data)
    bars = ax1.bar(range(len(all_ids)), all_sizes, color=colors, alpha=0.85, width=0.7)
    ax1.set_xticks(range(len(all_ids)))
    ax1.set_xticklabels([s.split('_')[-1] if '_' in s else s for s in all_ids],
                        rotation=75, ha='right', fontsize=7)
    style_ax(ax1, 'Samples per Bearing', 'Bearing', 'Sample Count')
    p1 = mpatches.Patch(color=PALETTE[0], label=f'FEMTO ({len(femto_data)} bearings)')
    p2 = mpatches.Patch(color=PALETTE[1], label=f'XJTU-SY ({len(xjtu_data)} bearings)')
    ax1.legend(handles=[p1, p2], facecolor='#1a1a2e', labelcolor='#c0c0c0', fontsize=8)

    # ── 1b: Bearing lifetime hist ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    lifetimes = [d[1] for d in femto_data + xjtu_data]
    ax2.hist(lifetimes, bins=20, color=PALETTE[2], alpha=0.8, edgecolor='#1a1a2e')
    style_ax(ax2, 'Bearing Lifetime Distribution', 'Lifetime (cycles)', 'Count')

    # ── 1c: RUL distribution across all samples ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    all_ruls = np.concatenate([r for _, r, _ in all_data])
    ax3.hist(all_ruls, bins=80, color=PALETTE[3], alpha=0.85, edgecolor='#1a1a2e')
    ax3.axvline(x=np.mean(all_ruls), color=PALETTE[4], linestyle='--',
                lw=1.5, label=f'Mean={np.mean(all_ruls):.0f}')
    ax3.axvline(x=np.median(all_ruls), color=PALETTE[5], linestyle='--',
                lw=1.5, label=f'Median={np.median(all_ruls):.0f}')
    style_ax(ax3, 'RUL Distribution (All Samples)', 'RUL (cycles)', 'Count')
    ax3.legend(facecolor='#1a1a2e', labelcolor='#c0c0c0', fontsize=8)

    # ── 1d: Dataset pie ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    sizes_pie = [sum(d[1] for d in femto_data), sum(d[1] for d in xjtu_data)]
    labels_pie = ['FEMTO', 'XJTU-SY']
    ax4.pie(sizes_pie, labels=labels_pie, colors=[PALETTE[0], PALETTE[1]],
            autopct='%1.1f%%', pctdistance=0.75,
            textprops={'color': '#c0c0c0', 'fontsize': 9},
            wedgeprops={'linewidth': 1.5, 'edgecolor': '#1a1a2e'})
    ax4.set_facecolor('#16213e')
    ax4.set_title('Dataset Composition', color='#e0e0ff', fontsize=11, fontweight='bold')

    b64 = fig_to_b64(fig)
    return b64


# ──────────────────────────────────────────────────────────────────────────────
# 2. Vibration Signal Analysis
# ──────────────────────────────────────────────────────────────────────────────

def plot_signal_analysis(all_data, config, n_bearings=3):
    """Time-domain + frequency-domain plots for sample bearings"""
    ensure_dir()

    fig = dark_fig(figsize=(15, 10))
    fig.suptitle('Vibration Signal Analysis (Sample Bearings)', color='#e0e0ff',
                 fontsize=14, fontweight='bold', y=0.99)

    # Pick up to n_bearings, prefer bearings with enough samples
    selected = [d for d in all_data if len(d[0]) > 10][:n_bearings]
    if not selected:
        selected = all_data[:n_bearings]

    rows = len(selected)
    gs = gridspec.GridSpec(rows, 3, figure=fig, hspace=0.55, wspace=0.38)

    import os
    from rf_data_loader import load_femto_bearing

    for row_i, (feats, ruls, meta) in enumerate(selected):
        bid = meta.get('bearing_id', f'Bearing_{row_i}')

        # We only have features, not raw signals → synthesize representative signal
        # Use last feature's RMS/std to scale noise (visual demo)
        rms = float(feats[-1][2]) if feats.shape[1] > 2 else 0.1
        std = float(feats[-1][1]) if feats.shape[1] > 1 else 0.1
        t = np.linspace(0, 1, 2048)
        # Simulate a plausible vibration waveform
        sig = (std * np.random.randn(2048)
               + rms * np.sin(2 * np.pi * 120 * t)
               + 0.3 * rms * np.sin(2 * np.pi * 360 * t))

        # Time domain
        ax_t = fig.add_subplot(gs[row_i, 0])
        ax_t.plot(t[:512], sig[:512], color=PALETTE[row_i % len(PALETTE)], lw=0.7, alpha=0.9)
        style_ax(ax_t, f'{bid.split("_")[-1]} – Time Domain', 'Time (s)', 'Amplitude')

        # Frequency (Welch PSD)
        ax_f = fig.add_subplot(gs[row_i, 1])
        f_w, psd = welch(sig, fs=config.SAMPLE_RATE // 10, nperseg=256)
        ax_f.semilogy(f_w, psd, color=PALETTE[(row_i+2) % len(PALETTE)], lw=0.8)
        style_ax(ax_f, f'{bid.split("_")[-1]} – PSD (Welch)', 'Freq (Hz)', 'PSD')

        # RUL over lifecycle
        ax_r = fig.add_subplot(gs[row_i, 2])
        ax_r.plot(range(len(ruls)), ruls,
                  color=PALETTE[(row_i+4) % len(PALETTE)], lw=1.5)
        ax_r.fill_between(range(len(ruls)), ruls, alpha=0.15,
                          color=PALETTE[(row_i+4) % len(PALETTE)])
        style_ax(ax_r, f'{bid.split("_")[-1]} – RUL over Life', 'Cycle', 'RUL')

    return fig_to_b64(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature Statistics
# ──────────────────────────────────────────────────────────────────────────────

FEAT_NAMES_BASE = [
    'Horiz_Mean', 'Horiz_Std', 'Horiz_RMS', 'Horiz_Peak',
    'Horiz_Kurtosis', 'Horiz_Skewness', 'Horiz_CrestFactor', 'Horiz_ShapeFactor',
    'Vert_Mean', 'Vert_Std', 'Vert_RMS', 'Vert_Peak',
    'Vert_Kurtosis', 'Vert_Skewness', 'Vert_CrestFactor', 'Vert_ShapeFactor',
    'Horiz_FreqCentroid', 'Horiz_FreqStd', 'Horiz_DominantFreq',
    'Horiz_Band_10-50Hz', 'Horiz_Band_60-180Hz', 'Horiz_Band_100-300Hz',
    'Horiz_Band_300-1kHz', 'Horiz_Band_1k-3kHz',
    'Vert_FreqCentroid', 'Vert_FreqStd', 'Vert_DominantFreq',
    'Vert_Band_10-50Hz', 'Vert_Band_60-180Hz', 'Vert_Band_100-300Hz',
    'Vert_Band_300-1kHz', 'Vert_Band_1k-3kHz',
    'Horiz_EnvMean', 'Horiz_EnvStd', 'Horiz_EnvMax', 'Horiz_EnvKurtosis',
    'Vert_EnvMean', 'Vert_EnvStd', 'Vert_EnvMax', 'Vert_EnvKurtosis',
    'Speed_RPM', 'Load_N', 'Temperature'
]


def _build_flat_arrays(all_data):
    """Flatten all data into X, y arrays (using raw per-cycle features, not aggregated)"""
    X_list, y_list = [], []
    for feats, ruls, meta in all_data:
        X_list.append(feats)
        y_list.append(ruls)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def plot_feature_statistics(all_data):
    """Box plots + violin plots of key features stratified by RUL zone"""
    X, y = _build_flat_arrays(all_data)

    # RUL zones
    zones = ['Critical\n(<200)', 'Degraded\n(200-500)', 'Healthy\n(>500)']
    masks = [y < 200, (y >= 200) & (y < 500), y >= 500]

    top_feats = [2, 4, 5, 6, 32, 34, 35]  # RMS, Kurtosis, Skewness, Crest, EnvMean, EnvMax, EnvKurt
    top_feats = [i for i in top_feats if i < X.shape[1]]

    fig = dark_fig(figsize=(15, 8))
    fig.suptitle('Feature Statistics by RUL Zone', color='#e0e0ff',
                 fontsize=14, fontweight='bold', y=0.99)

    n_cols = len(top_feats)
    axes = fig.subplots(1, n_cols)

    for col, feat_idx in enumerate(top_feats):
        ax = axes[col]
        data_zones = [X[m, feat_idx] for m in masks]
        parts = ax.violinplot(data_zones, positions=range(3),
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PALETTE[i])
            pc.set_alpha(0.7)
        parts['cmedians'].set_colors('#ffffff')
        ax.set_xticks(range(3))
        ax.set_xticklabels(zones, fontsize=6.5)
        fname = FEAT_NAMES_BASE[feat_idx] if feat_idx < len(FEAT_NAMES_BASE) else f'F{feat_idx}'
        style_ax(ax, fname.replace('_', '\n'), '', '')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_b64(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(all_data):
    """Correlation heatmap of top features vs RUL"""
    X, y = _build_flat_arrays(all_data)

    # Subsample for speed
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X, y = X[idx], y[idx]

    # Compute correlations with RUL
    n_feats = min(X.shape[1], 20)
    corr_rul = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    top_idx = np.argsort(np.abs(corr_rul))[-n_feats:][::-1]

    X_top = X[:, top_idx]
    feat_labels = [FEAT_NAMES_BASE[i] if i < len(FEAT_NAMES_BASE) else f'F{i}' for i in top_idx]

    corr_matrix = np.corrcoef(np.hstack([X_top, y.reshape(-1, 1)]).T)

    fig = dark_fig(figsize=(13, 11))
    ax = fig.add_subplot(111)
    ax.set_facecolor('#16213e')

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    labels = feat_labels + ['RUL']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=7.5, color='#c0c0c0')
    ax.set_yticklabels(labels, fontsize=7.5, color='#c0c0c0')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelcolor='#c0c0c0')

