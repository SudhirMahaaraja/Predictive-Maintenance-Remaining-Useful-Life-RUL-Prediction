/* ─────────────────────────────────────────────────────────────────────────────
   Bearing RUL Dashboard — Main JavaScript
   Enhanced: Confidence Intervals, Prediction History, Side-by-Side Comparison,
             Ensemble Voting, SHAP Explanations
───────────────────────────────────────────────────────────────────────────── */

// ─── State ────────────────────────────────────────────────────────────────
const state = {
  edaData: null,
  modelsData: null,
  activeSection: 'overview',
  activeEdaPlot: 'dataset_overview',
  activeMode: 'csv',
  edaStatus: 'idle',   // idle | started | running | ready
  lastFileName: '',    // tracks last processed file name for exports
  edaTimer: null,
  edaElapsed: 0,
  pollInterval: null,
  healthChart: null,
};

const PLOT_TITLES = {
  dataset_overview: 'Dataset Overview',
  signal_analysis: 'Vibration Signal Analysis',
  feature_stats: 'Feature Statistics by RUL Zone',
  correlation: 'Feature Correlation Heatmap',
  degradation: 'RUL Degradation Trends',
  operating_conditions: 'Operating Conditions Analysis',
  feature_importance: 'Feature Importance (Random Forest)',
  outlier_analysis: 'Outlier / Anomaly Analysis',
};

const PLOT_DESCRIPTIONS = {
  dataset_overview: 'A quick look at what\'s in the training data — how many bearings from each dataset, how the samples are spread, and the range of remaining life values.\nThis helps you see if the data is balanced before the models learn from it.',
  signal_analysis: 'Real vibration recordings at different points in a bearing\'s life — early on, halfway through, and close to failure.\nYou\'ll notice the shaking gets wilder as the bearing wears out. That\'s exactly the pattern our models learn to recognize!',
  feature_stats: 'How do key measurements (like RMS, kurtosis, etc.) change depending on the bearing\'s health?\nIf a measurement looks very different between "Healthy" and "Critical" zones, it\'s a strong clue the model can use.',
  correlation: 'Which measurements move together? This heatmap shows relationships between all the features.\nStrongly linked features (bright squares) might be saying the same thing — the ones linked to RUL are extra useful for predictions.',
  degradation: 'Watch how vibration energy climbs over a bearing\'s lifetime.\nA steady upward trend means the feature reliably tracks wearing out — exactly what we need for good predictions.',
  operating_conditions: 'Speed, load, and temperature readings color-coded by health status.\nIf the colors overlap a lot, it means we can\'t judge health from operating conditions alone — that\'s why vibration features matter so much.',
  feature_importance: 'Which features does the Random Forest model rely on the most?\nThe taller the bar, the more that feature helps predict remaining life. Top features are keepers; short bars could be dropped without losing much.',
  outlier_analysis: 'Spots unusual data points that don\'t fit the normal pattern.\nThese could be sensor glitches or real anomalies — worth checking before deciding whether to keep them in training.',
};

const PLOT_ABBREVIATIONS = {
  dataset_overview: [
    ['RUL', 'Remaining Useful Life — cycles left before bearing failure'],
    ['FEMTO', 'FEMTO-ST bearing dataset (IEEE PHM 2012, Besançon, France)'],
    ['XJTU-SY', 'Xi\'an Jiaotong University – Shenyang bearing dataset'],
  ],
  signal_analysis: [
    ['RUL', 'Remaining Useful Life'],
    ['H-vib', 'Horizontal Vibration — lateral accelerometer channel'],
    ['V-vib', 'Vertical Vibration — vertical accelerometer channel'],
    ['g', 'Unit of acceleration (1 g ≈ 9.81 m/s²)'],
  ],
  feature_stats: [
    ['RMS', 'Root Mean Square — overall vibration energy level'],
    ['Kurtosis', 'Fourth statistical moment — sensitivity to sharp impulse peaks (fault indicator)'],
    ['Crest Factor', 'Peak ÷ RMS — how spiky the signal is relative to average energy'],
    ['IQR', 'Interquartile Range — spread of the middle 50% of values'],
    ['RUL', 'Remaining Useful Life'],
  ],
  correlation: [
    ['r', 'Pearson Correlation Coefficient — linear relationship (−1 to +1)'],
    ['RUL', 'Remaining Useful Life — the prediction target variable'],
    ['RMS', 'Root Mean Square'],
    ['FFT', 'Fast Fourier Transform — converts signal to frequency domain'],
  ],
  degradation: [
    ['RUL', 'Remaining Useful Life'],
    ['RMS', 'Root Mean Square — used as the health indicator over time'],
    ['FEMTO', 'FEMTO-ST bearing dataset'],
    ['XJTU-SY', 'Xi\'an Jiaotong University – Shenyang bearing dataset'],
  ],
  operating_conditions: [
    ['RPM', 'Revolutions Per Minute — rotational shaft speed'],
    ['N', 'Newton — SI unit of radial load on the bearing'],
    ['°C', 'Degrees Celsius — bearing operating temperature'],
    ['RUL', 'Remaining Useful Life — colour-codes each data point by health zone'],
  ],
  feature_importance: [
    ['MDI', 'Mean Decrease in Impurity — how much a feature reduces Gini impurity across RF trees'],
    ['RF', 'Random Forest — ensemble of decision trees'],
    ['RMS', 'Root Mean Square'],
    ['FFT', 'Fast Fourier Transform'],
    ['RUL', 'Remaining Useful Life'],
  ],
  outlier_analysis: [
    ['IQR', 'Interquartile Range — values beyond 1.5×IQR flagged as outliers'],
    ['IF', 'Isolation Forest — unsupervised algorithm isolating anomalous samples'],
    ['RUL', 'Remaining Useful Life'],
    ['g', 'Unit of acceleration (1 g ≈ 9.81 m/s²)'],
  ],
};

// ─── DOM helpers ─────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const setText = (id, val) => { const el = $(id); if (el) el.textContent = val; };
const setHTML = (id, val) => { const el = $(id); if (el) el.innerHTML = val; };

// ─── Navigation ──────────────────────────────────────────────────────────
function showSection(name) {
  document.querySelectorAll('.dash-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.sidebar-nav .nav-link').forEach(a => a.classList.remove('active'));

  const section = document.getElementById(`section-${name}`);
  if (section) section.classList.add('active');

  const navLink = document.getElementById(`nav-${name}`);
  if (navLink) navLink.classList.add('active');

  state.activeSection = name;

  const headings = {
    overview: ['Overview', 'Your Bearing Health Dashboard — at a glance'],
    eda: ['EDA & Analysis', 'Explore your data with interactive charts'],
    models: ['Model Performance', 'How well are RF & XGBoost doing?'],
    predict: ['Predict RUL', 'Upload data or punch in values to get a prediction'],
    history: ['Prediction History', 'All your past predictions in one place'],
  };
  const [h, sub] = headings[name] || [name, ''];
  setText('page-heading', h);
  setText('page-sub', sub);

  // Load section data on demand
  if (name === 'models' && !state.modelsData) loadModels();
  if (name === 'eda' && state.edaData) renderEdaPlot(state.activeEdaPlot);
  if (name === 'history') renderHistory();
}

// Sidebar nav clicks
document.querySelectorAll('.sidebar-nav .nav-link[data-section]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    showSection(link.dataset.section);
  });
});

// Sidebar toggle
$('sidebar-toggle').addEventListener('click', () => {
  const sidebar = document.querySelector('.sidebar');
  const main = document.querySelector('.main-content');
  if (window.innerWidth <= 768) {
    sidebar.classList.toggle('mobile-open');
  } else {
    sidebar.classList.toggle('collapsed');
    main.classList.toggle('sidebar-hidden');
  }
});

// ─── EDA ───────────────────────────────────────────────────────────────
function startEda(force = false) {
  return new Promise(async (resolve) => {
    state.edaReadyResolve = resolve;   // called by onEdaReady when data arrives

    const url = '/api/eda' + (force ? '?force=true' : '');
    const resp = await fetch(url);
    const json = await resp.json();

    if (json.status === 'ready' && json.data) {
      onEdaReady(json.data);
      return;
    }

    if (json.status === 'started' || json.status === 'running') {
      setEdaRunning();
    }

    if (state.pollInterval) clearInterval(state.pollInterval);
    state.pollInterval = setInterval(pollEda, 4000);
  });
}

async function pollEda() {
  try {
    const resp = await fetch('/api/eda/status');
    const json = await resp.json();

    if (json.status === 'ready') {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
      const r2 = await fetch('/api/eda');
      const j2 = await r2.json();
      if (j2.status === 'ready' && j2.data) onEdaReady(j2.data);
    }
  } catch (e) { console.warn('Poll error', e); }
}


function setEdaRunning() {
  state.edaStatus = 'running';
  $('eda-loading-banner').classList.remove('d-none');
  $('eda-spinner').classList.remove('d-none');

  // Timer
  state.edaElapsed = 0;
  if (state.edaTimer) clearInterval(state.edaTimer);
  state.edaTimer = setInterval(() => {
    state.edaElapsed++;
    const m = Math.floor(state.edaElapsed / 60);
    const s = String(state.edaElapsed % 60).padStart(2, '0');
    setText('eda-timer', `(${m}:${s} elapsed)`);
  }, 1000);

  setText('system-status', '⚙ EDA Running…');
  document.querySelector('.status-pill').style.borderColor = 'rgba(249,199,79,0.4)';
  document.querySelector('.status-pill').style.color = '#F9C74F';
}

function onEdaReady(data) {
  state.edaData = data;
  state.edaStatus = 'ready';

  // Clear loading indicators
  $('eda-loading-banner').classList.add('d-none');
  $('eda-spinner').classList.add('d-none');
  if (state.edaTimer) { clearInterval(state.edaTimer); state.edaTimer = null; }

  // Status pill
  document.querySelector('.status-pill').innerHTML =
    '<span class="pulse-dot"></span> System Ready';
  document.querySelector('.status-pill').style.color = '';
  document.querySelector('.status-pill').style.borderColor = '';

  populateOverview(data);
  state.edaPlots = data.plots ?? {};
  if (state.activeSection === 'eda') renderEdaPlot(state.activeEdaPlot);

  // Resolve any pending Promise from startEda (used by Refresh button)
  if (state.edaReadyResolve) { state.edaReadyResolve(); state.edaReadyResolve = null; }
}

// ─── Overview KPIs ────────────────────────────────────────────────────────
function populateOverview(d) {
  setText('kv-bearings', d.n_bearings?.toLocaleString() ?? '—');
  setText('kv-samples', formatNum(d.n_samples));
  setText('kv-features', d.n_features ?? '—');
  setText('kv-rulmean', d.rul_mean ? Math.round(d.rul_mean).toLocaleString() : '—');

  // Health donut
  const pctC = d.pct_critical ?? 0;
  const pctD = d.pct_degraded ?? 0;
  const pctH = d.pct_healthy ?? 0;
  renderHealthDonut(pctH, pctD, pctC);

  // Summary table
  const rows = [
    ['FEMTO Bearings', d.n_femto ?? '—'],
    ['XJTU-SY Bearings', d.n_xjtu ?? '—'],
    ['Total Samples', (d.n_samples ?? '—').toLocaleString?.() ?? d.n_samples],
    ['Feature Dimensions', d.n_features ?? '—'],
    ['RUL Min (cycles)', d.rul_min ? Math.round(d.rul_min) : '—'],
    ['RUL Max (cycles)', d.rul_max ? Math.round(d.rul_max) : '—'],
    ['RUL Mean (cycles)', d.rul_mean ? Math.round(d.rul_mean) : '—'],
    ['RUL Median (cycles)', d.rul_median ? Math.round(d.rul_median) : '—'],
    ['RUL Std Dev', d.rul_std ? Math.round(d.rul_std) : '—'],
    ['Critical (<200 cyc)', d.pct_critical ? d.pct_critical.toFixed(1) + '%' : '—'],
    ['Degraded (200–500)', d.pct_degraded ? d.pct_degraded.toFixed(1) + '%' : '—'],
    ['Healthy (>500 cyc)', d.pct_healthy ? d.pct_healthy.toFixed(1) + '%' : '—'],
  ];

  setHTML('summary-tbody', rows.map(([k, v], i) =>
    `<tr style="border-color:rgba(111,143,114,0.12)">
       <td style="width:58%;color:#BFC6C4;font-size:13px;padding:9px 12px;border-color:rgba(111,143,114,0.12)">${k}</td>
       <td style="color:#E8E2D8;font-weight:600;font-size:13px;font-family:'JetBrains Mono',monospace;padding:9px 12px;border-color:rgba(111,143,114,0.12)">${v}</td>
     </tr>`
  ).join(''));
}

function renderHealthDonut(h, d, c) {
  const ctx = document.getElementById('health-donut').getContext('2d');
  if (state.healthChart) state.healthChart.destroy();
  state.healthChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Healthy', 'Degraded', 'Critical'],
      datasets: [{
        data: [h, d, c],
        backgroundColor: ['rgba(111,143,114,0.85)', 'rgba(242,166,90,0.85)', 'rgba(232,80,80,0.85)'],
        borderColor: ['#6F8F72', '#F2A65A', '#e85050'],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%` }
        }
      },
      animation: { animateRotate: true, duration: 900 }
    }
  });

  // Legend
  const colors = ['#6F8F72', '#F2A65A', '#e85050'];
  const labels = ['Healthy', 'Degraded', 'Critical'];
  const vals = [h, d, c];
  setHTML('health-legend', labels.map((l, i) =>
    `<div class="health-leg-item">
       <div class="health-leg-dot" style="background:${colors[i]}"></div>
       <span>${l} <b style="color:${colors[i]}">${vals[i].toFixed(1)}%</b></span>
     </div>`
  ).join(''));
}

// ─── EDA Plots ────────────────────────────────────────────────────────────
document.querySelectorAll('.eda-tabs .nav-link').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.eda-tabs .nav-link').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.activeEdaPlot = btn.dataset.plot;
    renderEdaPlot(btn.dataset.plot);
    setText('eda-plot-title', PLOT_TITLES[btn.dataset.plot] || btn.dataset.plot);
    setEdaDescription(btn.dataset.plot);
  });
});

function setEdaDescription(plotKey) {
  const descEl = $('eda-plot-description');
  const desc2El = $('eda-plot-desc-line2');
  const abbrEl = $('eda-plot-abbr');
  if (!descEl) return;

  const desc = PLOT_DESCRIPTIONS[plotKey] || '';
  const parts = desc.split('\n');
  descEl.textContent = '\u2139\ufe0f ' + (parts[0] || desc);
  if (desc2El) desc2El.textContent = parts[1] || '';

  if (!abbrEl) return;
  const abbrs = PLOT_ABBREVIATIONS[plotKey] || [];
  if (abbrs.length === 0) { abbrEl.innerHTML = ''; return; }
  abbrEl.innerHTML = abbrs.map(([abbr, meaning]) =>
    `<span class="plot-abbr-pill" title="${meaning}"><b>${abbr}</b> = ${meaning}</span>`
  ).join('');
}

function renderEdaPlot(plotKey) {
  const container = $('eda-plot-container');
  setEdaDescription(plotKey);
  if (!state.edaData) {
    container.innerHTML = '<p class="text-muted py-5">EDA not loaded. Click <b>Refresh EDA</b>.</p>';
    return;
  }
  const b64 = state.edaData.plots?.[plotKey];
  if (!b64) {
    container.innerHTML = `<p class="text-muted py-5">Plot "${plotKey}" not available.</p>`;
    return;
  }
  container.innerHTML = `<img src="data:image/png;base64,${b64}" class="eda-plot-img" alt="${plotKey}"/>`;
}


// ─── Models ───────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    state.modelsData = await resp.json();
    renderModels(state.modelsData);
  } catch (e) {
    setHTML('rf-metrics-body', '<p class="text-danger">Failed to load model info.</p>');
  }
}

function renderModels(data) {
  const { models, test_results } = data;

  renderModelCard('rf', models.rf, test_results?.random_forest);
  renderModelCard('xgb', models.xgb, test_results?.xgboost);

  // Epoch plot
  if (state.edaData?.plots?.epoch_history) {
    $('epoch-plot-container').innerHTML =
      `<img src="data:image/png;base64,${state.edaData.plots.epoch_history}" class="eda-plot-img" alt="epoch"/>`;
  } else {
    $('epoch-plot-container').innerHTML =
      '<p class="text-muted">Epoch history not available. Run training first.</p>';
  }

  // Prediction plots from EDA
  const plots = state.edaData?.plots ?? {};

  for (const [domId, key] of [
    ['rf-val-plot', 'rf_val'],
    ['xgb-val-plot', 'xgb_val'],
    ['rf-test-plot', 'rf_test'],
    ['xgb-test-plot', 'xgb_test'],
  ]) {
    const b64 = plots[key];
    if (b64) {
      $(domId).innerHTML = `<img src="data:image/png;base64,${b64}" class="eda-plot-img" alt="${key}"/>`;
    }
  }
}

function renderModelCard(type, info, testRes) {
  const badgeId = `${type}-size-badge`;
  const bodyId = `${type}-metrics-body`;
  const color = type === 'rf' ? '#6F8F72' : '#F2A65A';

  if (!info?.available) {
    setText(badgeId, 'Not found');
    setHTML(bodyId, '<p class="text-muted small">Model not trained yet. Run <code>train_rf.py</code>.</p>');
    return;
  }

  setText(badgeId, `${info.size_mb} MB`);

  const vm = info.val_metrics || {};
  const tm = testRes?.test_metrics || {};

  const metricsHtml = `
    <div class="metrics-grid">
      ${metCell('MAE', vm.mae ? vm.mae.toFixed(1) : '—', 'cycles', color)}
      ${metCell('RMSE', vm.rmse ? vm.rmse.toFixed(1) : '—', 'cycles', color)}
      ${metCell('R²', vm.r2 ? vm.r2.toFixed(4) : '—', '', color)}
      ${metCell('MAPE', vm.mape ? vm.mape.toFixed(2) + '%' : '—', '', color)}
    </div>
    <div class="mb-3">
      <div class="small text-muted mb-2 fw-semibold">Accuracy Bands (Validation)</div>
      ${accBar('±50', vm.acc_50, color)}
      ${accBar('±100', vm.acc_100, color)}
      ${accBar('±200', vm.acc_200, color)}
      ${accBar('±500', vm.acc_500, color)}
      <div class="mt-2 p-2 rounded mg-acc-note">
        <strong class="mg-warn">&#9432; Acc±N</strong> = % of predictions within ±N cycles of the true RUL (Remaining Useful Life).
        Each band is a progressively looser tolerance — <strong class="mg-highlight">Acc±200 of ~30–55%</strong> is the industry norm for FEMTO / XJTU-SY datasets.
        A high <strong class="mg-good">Acc±500</strong> and low <strong class="mg-good">MAE (Mean Absolute Error)</strong> matter most for real maintenance scheduling.
      </div>
    </div>
    ${tm.mae ? `
    <div class="small text-muted mb-2 fw-semibold">Test Set Results</div>
    <div class="d-flex gap-2 flex-wrap">
      ${pill('MAE', tm.mae.toFixed(1) + ' cyc')}
      ${pill('RMSE', tm.rmse.toFixed(1) + ' cyc')}
      ${pill('R²', tm.r2.toFixed(4))}
      ${pill('Acc±200', tm.acc_200.toFixed(1) + '%')}
    </div>` : ''}
  `;

  setHTML(bodyId, metricsHtml);
}

function metCell(label, val, unit, color) {
  return `<div class="metric-cell">
    <div class="metric-cell-val" style="color:${color}">${val}${unit ? '<span style="font-size:11px;color:#9090b0"> ' + unit + '</span>' : ''}</div>
    <div class="metric-cell-label">${label}</div>
  </div>`;
}

function accBar(label, pct, color) {
  const val = pct ?? 0;
  return `<div class="acc-bar-row">
    <span class="acc-bar-label">${label}</span>
    <div class="acc-bar-track">
      <div class="acc-bar-fill" style="width:${val}%;background:${color}"></div>
    </div>
    <span class="acc-bar-pct">${val.toFixed(1)}%</span>
  </div>`;
}

function pill(label, val) {
  return `<span class="badge" style="background:rgba(111,143,114,0.15);color:#BFC6C4;border:1px solid rgba(111,143,114,0.25);font-family:'JetBrains Mono',monospace;font-size:11px">
    ${label}: <b>${val}</b>
  </span>`;
}

// ─── Prediction Mode Tabs ─────────────────────────────────────────────────
document.querySelectorAll('.predict-tabs .nav-link').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.predict-tabs .nav-link').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.activeMode = btn.dataset.mode;
    $('mode-csv').classList.toggle('d-none', btn.dataset.mode !== 'csv');
    $('mode-manual').classList.toggle('d-none', btn.dataset.mode !== 'manual');
  });
});

// ─── CSV Upload ───────────────────────────────────────────────────────────
const uploadZone = $('upload-zone');
const fileInput = $('csv-file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
  fileInput._selectedFile = file;
  $('file-name').innerHTML = `📄 ${file.name} <button type="button" class="btn btn-sm ms-2" style="font-size:10px;padding:0 6px;color:#e85050;border:1px solid rgba(232,80,80,0.3);border-radius:4px" id="btn-clear-file">✕</button>`;
  $('file-name').classList.remove('d-none');
  $('btn-clear-file').addEventListener('click', (ev) => {
    ev.stopPropagation();
    fileInput._selectedFile = null;
    fileInput.value = '';
    $('file-name').classList.add('d-none');
    $('file-name').innerHTML = '';
  });
}

$('csv-form').addEventListener('submit', async e => {
  e.preventDefault();
  const file = fileInput._selectedFile || (fileInput.files[0]);
  if (!file) { showError('Please select a CSV file first.'); return; }

  const fd = new FormData();
  fd.append('file', file);
  fd.append('speed', $('csv-speed').value);
  fd.append('load', $('csv-load').value);
  fd.append('temp', $('csv-temp').value);
  fd.append('model', $('csv-model').value);

  setPredicting(true);
  try {
    const resp = await fetch('/api/predict/csv', { method: 'POST', body: fd });
    const json = await resp.json();
    if (json.error) throw new Error(json.error);
    state.lastFileName = file.name.replace(/\.csv$/i, '');
    showResult(json);
    saveToHistory(json);
  } catch (err) {
    showError(err.message);
  } finally {
    setPredicting(false);
  }
});

// ─── Manual Prediction ────────────────────────────────────────────────────
const rmsSlider = $('rms-slider');
rmsSlider.addEventListener('input', () => {
  $('m-rms').value = parseFloat(rmsSlider.value).toFixed(2);
  setText('rms-slider-val', parseFloat(rmsSlider.value).toFixed(2));
});
$('m-rms').addEventListener('input', () => {
  rmsSlider.value = $('m-rms').value;
  setText('rms-slider-val', parseFloat($('m-rms').value).toFixed(2));
});

$('manual-form').addEventListener('submit', async e => {
  e.preventDefault();
  const payload = {
    model: $('m-model').value,
    speed: +$('m-speed').value,
    load: +$('m-load').value,
    temp: +$('m-temp').value,
    rms: +$('m-rms').value,
    kurtosis: +$('m-kurtosis').value,
    dominant_freq: +$('m-domfreq').value,
    envelope_rms: +$('m-envrms').value,
  };

  setPredicting(false, 'manual');
  try {
    const endpoint = payload.model === 'ensemble' ? '/api/predict/ensemble' : '/api/predict/manual';
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await resp.json();
    if (json.error) throw new Error(json.error);
    showResult(json);
    saveToHistory(json);
  } catch (err) {
    showError(err.message);
  } finally {
    const btn = $('btn-predict-manual');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-lightning-charge-fill"></i> Predict RUL';
  }
});


// ─── Compare Both Models ──────────────────────────────────────────────────

$('btn-compare-models')?.addEventListener('click', async () => {
  const btn = $('btn-compare-models');
  const origHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Comparing…';

  const payload = {
    mode: 'manual',
    speed: +$('m-speed').value,
    load: +$('m-load').value,
    temp: +$('m-temp').value,
    rms: +$('m-rms').value,
    kurtosis: +$('m-kurtosis').value,
    dominant_freq: +$('m-domfreq').value,
    envelope_rms: +$('m-envrms').value,
  };

  try {
    const resp = await fetch('/api/predict/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await resp.json();
    if (json.error) throw new Error(json.error);
    showComparisonResult(json);
    // Save ensemble to history
    if (json.ensemble) saveToHistory(json.ensemble);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origHTML;
  }
});


// ─── Result Rendering ───────────────────────────────────────────────
function showResult(r) {
  const rul = r.predicted_rul;
  const status = r.status?.toLowerCase() ?? 'healthy';
  const pct = Math.min(r.rul_pct ?? 0, 100);

  const barColor = status === 'healthy' ? '#6F8F72' :
    status === 'degraded' ? '#F2A65A' : '#e85050';
  const textColor = barColor;

  let html = `
    <div class="text-center w-100">
      <div class="rul-gauge ${status} mx-auto mb-3">
        <div>
          <div class="rul-val" style="color:${textColor}">${rul.toLocaleString()}</div>
          <div class="rul-unit">cycles remaining</div>
        </div>
      </div>

      <div class="status-badge ${status} mx-auto d-inline-block mb-3">${r.status}</div>

      <div class="px-3 mb-3">
        <div class="d-flex justify-content-between small text-muted mb-1">
          <span>Remaining Life</span>
          <span>${pct.toFixed(1)}%</span>
        </div>
        <div class="rul-progress-track">
          <div class="rul-progress-fill" style="width:${pct}%;background:${barColor}"></div>
        </div>
      </div>

      ${renderConfidenceInterval(r)}

      <div class="row g-2 px-3 mt-1">
        ${infoTile('Model', r.model_used)}
        ${infoTile('Speed', r.speed + ' RPM')}
        ${infoTile('Load', r.load + ' N')}
        ${infoTile('Temp', r.temp + ' °C')}
        ${infoTile('Features', r.n_features)}
        ${infoTile('RUL %', pct.toFixed(1) + '%')}
      </div>

      <div class="mt-3 px-3">
        ${statusAdvice(status)}
      </div>

      <div class="mt-3 px-3 d-flex gap-2 justify-content-center">
        <button class="btn btn-export btn-sm" onclick="exportResultPNG()">
          <i class="bi bi-image"></i> Export PNG
        </button>
        <button class="btn btn-export btn-sm" onclick="exportResultPDF()">
          <i class="bi bi-file-earmark-pdf"></i> Export PDF
        </button>
      </div>

      ${renderShapSection(r)}

      ${r.plot_b64 ? `
      <div class="mt-4 px-3 text-start">
        <div class="text-muted small fw-semibold mb-2"><i class="bi bi-graph-up text-info"></i> Analyzed Signal (Horizontal & Vertical)</div>
        <div class="p-2 rounded mb-2" style="background: var(--bg-card2); border: 1px solid var(--border)">
            <img src="data:image/png;base64,${r.plot_b64}" class="img-fluid w-100" alt="Signal Plot"/>
        </div>
        <div class="p-3 rounded mt-2" style="background: rgba(111,143,114,0.08); border: 1px solid var(--border); line-height: 1.5">
           <div class="small mb-2" style="color: var(--text-primary)">
               <strong>What you're seeing:</strong> This chart shows the vibration shaking we captured (or simulated) — both sideways and up-down directions.<br>
               Bigger and messier waves usually mean the bearing is getting worn. Think of it like listening to a machine — a healthy one hums smoothly, a worn one rattles.
           </div>
           <div class="small fw-semibold mt-2" style="color: #6F8F72">
               <i class="bi bi-check-circle"></i> What to do next:<br>
               <span style="font-weight: 400; color: var(--text-primary)">If peak vibrations keep climbing above the healthy baseline, it's time to take a closer look at the bearing or plan a replacement.</span>
           </div>
        </div>
      </div>` : ''}

      ${r.eda_plot_b64 ? `
      <div class="mt-4 px-3 text-start">
        <div class="text-muted small fw-semibold mb-2"><i class="bi bi-trophy-fill text-warning"></i> Predictive Feature Importance</div>
        <div class="p-2 rounded mb-2" style="background: var(--bg-card2); border: 1px solid var(--border)">
            <img src="data:image/png;base64,${r.eda_plot_b64}" class="img-fluid w-100" alt="Feature Importance"/>
        </div>
        <div class="p-3 rounded mt-2" style="background: rgba(111,143,114,0.08); border: 1px solid var(--border); line-height: 1.5">
           <div class="small mb-2" style="color: var(--text-primary)">
               <strong>What this means:</strong> The model looks at several measurements from your vibration data. The ones at the top of this chart matter the most.<br>
               Think of them as the biggest clues the model uses to figure out how much life is left.<br>
               Your data was checked against these top clues to come up with the RUL number above.
           </div>
           <div class="small fw-semibold mt-2" style="color: #F2A65A">
               <i class="bi bi-arrow-right-circle"></i> What to do next:<br>
               <span style="font-weight: 400; color: var(--text-primary)">Keep an eye on those top features over time. If they start spiking consistently, that's a sign the bearing needs attention soon.</span>
           </div>
        </div>
      </div>` : ''}
    </div>
  `;

  $('result-panel').innerHTML = html;

  // Animate progress bar
  setTimeout(() => {
    const fill = $('result-panel').querySelector('.rul-progress-fill');
    if (fill) fill.style.width = pct + '%';
  }, 50);
}


// ─── Confidence Interval Rendering ────────────────────────────────────────
function renderConfidenceInterval(r) {
  const ci = r.confidence_interval;
  if (!ci) return '';

  return `
    <div class="ci-card mx-3 mb-3">
      <div class="ci-header">
        <i class="bi bi-shield-check"></i> 95% Confidence Interval
        <span class="ci-method">${ci.ci_method || ''}</span>
      </div>
      <div class="ci-body">
        <div class="ci-range">
          <span class="ci-val ci-lower">${ci.ci_lower.toLocaleString()}</span>
          <div class="ci-bar-wrap">
            <div class="ci-bar-track">
              <div class="ci-bar-fill" style="left:${ciBarPct(ci.ci_lower)}%;width:${ciBarPct(ci.ci_upper) - ciBarPct(ci.ci_lower)}%"></div>
              <div class="ci-bar-marker" style="left:${ciBarPct(r.predicted_rul)}%"></div>
            </div>
            <div class="ci-bar-labels">
              <span>0</span>
              <span>${(1500).toLocaleString()}</span>
            </div>
          </div>
          <span class="ci-val ci-upper">${ci.ci_upper.toLocaleString()}</span>
        </div>
        <div class="ci-detail">
          <span>σ = ${ci.ci_std.toLocaleString()} cycles</span>
          ${ci.n_estimators ? `<span>${ci.n_estimators} estimators</span>` : ''}
        </div>
      </div>
    </div>`;
}

function ciBarPct(val) {
  return Math.max(0, Math.min(100, (val / 1500) * 100));
}


// ─── SHAP Explanation Rendering ───────────────────────────────────────────
function renderShapSection(r) {
  if (!r.shap) return '';

  const s = r.shap;
  let featRows = '';
  const topFeats = s.top_features?.slice(0, 8) || [];
  topFeats.forEach(f => {
    const dir = f.shap_value >= 0 ? 'positive' : 'negative';
    const icon = f.shap_value >= 0
      ? '<i class="bi bi-arrow-up-short" style="color:#6F8F72"></i>'
      : '<i class="bi bi-arrow-down-short" style="color:#e85050"></i>';
    featRows += `
      <div class="shap-feat-row ${dir}">
        <span class="shap-feat-name">${f.name}</span>
        <span class="shap-feat-val">${icon} ${f.shap_value.toFixed(3)}</span>
      </div>`;
  });

  return `
    <div class="mt-4 px-3 text-start">
      <div class="text-muted small fw-semibold mb-2">
        <i class="bi bi-lightbulb-fill" style="color:#F2A65A"></i>
        SHAP Explanation — Why This Prediction?
      </div>
      <div class="shap-card">
        ${s.shap_plot_b64 ? `
        <div class="p-2 rounded mb-3" style="background: var(--bg-card2); border: 1px solid var(--border)">
          <img src="data:image/png;base64,${s.shap_plot_b64}" class="img-fluid w-100" alt="SHAP Plot"/>
        </div>` : ''}
        <div class="shap-feat-grid">
          ${featRows}
        </div>
        <div class="shap-base-note mt-2">
          Base value (avg prediction): <strong>${s.base_value?.toLocaleString() ?? '—'}</strong> cycles
        </div>
        <div class="p-3 rounded mt-2" style="background: rgba(111,143,114,0.08); border: 1px solid var(--border); line-height: 1.5">
          <div class="small" style="color: var(--text-primary)">
            <strong>How to read this:</strong>
            <span style="color:#6F8F72">Green (positive)</span> features are pushing the prediction <strong>higher</strong> (healthier bearing).
            <span style="color:#e85050">Red (negative)</span> features pull it <strong>lower</strong> (more worn out).
            The ones at the top have the biggest impact on this specific result.
          </div>
        </div>
      </div>
    </div>`;
}


// ─── Side-by-Side Comparison Rendering ────────────────────────────────────
function showComparisonResult(data) {
  const rf = data.rf;
  const xgb = data.xgb;
  const ens = data.ensemble;

  function miniGauge(r, label, color) {
    if (r?.error) return `<div class="compare-mini text-center"><p class="text-danger small">${label}: ${r.error}</p></div>`;
    const rul = r.predicted_rul;
    const st = r.status?.toLowerCase() || 'healthy';
    const pct = Math.min(r.rul_pct ?? 0, 100);

    return `
      <div class="compare-mini">
        <div class="compare-model-label" style="color:${color}">${label}</div>
        <div class="compare-gauge ${st}">
          <div class="compare-rul-val" style="color:${color}">${rul.toLocaleString()}</div>
          <div class="compare-rul-unit">cycles</div>
        </div>
        <div class="status-badge ${st} mx-auto d-inline-block mb-2" style="font-size:11px;padding:3px 12px">${r.status}</div>
        ${r.confidence_interval ? `
        <div class="compare-ci small text-muted">
          CI: ${r.confidence_interval.ci_lower.toLocaleString()} – ${r.confidence_interval.ci_upper.toLocaleString()}
        </div>` : ''}
        <div class="compare-progress-track mt-2">
          <div class="rul-progress-fill" style="width:${pct}%;background:${color}"></div>
        </div>
      </div>`;
  }

  const html = `
    <div class="text-center w-100">
      <div class="compare-header mb-3">
        <i class="bi bi-layout-split"></i> Side-by-Side Model Comparison
      </div>
      <div class="compare-grid">
        ${miniGauge(rf, 'Random Forest', '#6F8F72')}
        ${miniGauge(xgb, 'XGBoost', '#F2A65A')}
        ${ens ? miniGauge(ens, 'Ensemble (Weighted)', '#BFC6C4') : ''}
      </div>

      ${ens ? `
      <div class="ensemble-detail mt-3 mx-3">
        <div class="small text-muted mb-1 fw-semibold">
          <i class="bi bi-layers-fill" style="color:#BFC6C4"></i> Ensemble Weights
        </div>
        <div class="d-flex justify-content-center gap-3">
          <span class="badge" style="background:rgba(111,143,114,0.15);color:#6F8F72;border:1px solid rgba(111,143,114,0.3)">
            RF: <b>${(ens.weights?.rf * 100).toFixed(1)}%</b>
          </span>
          <span class="badge" style="background:rgba(242,166,90,0.15);color:#F2A65A;border:1px solid rgba(242,166,90,0.3)">
            XGB: <b>${(ens.weights?.xgb * 100).toFixed(1)}%</b>
          </span>
        </div>
      </div>` : ''}

      ${rf?.plot_b64 ? `
      <div class="mt-4 px-3 text-start">
        <div class="text-muted small fw-semibold mb-2"><i class="bi bi-graph-up text-info"></i> Analyzed Signal</div>
        <div class="p-2 rounded" style="background: var(--bg-card2); border: 1px solid var(--border)">
            <img src="data:image/png;base64,${rf.plot_b64}" class="img-fluid w-100" alt="Signal Plot"/>
        </div>
      </div>` : ''}

      ${rf?.shap?.shap_plot_b64 ? `
      <div class="mt-3 px-3 text-start">
        <div class="text-muted small fw-semibold mb-2"><i class="bi bi-lightbulb-fill" style="color:#F2A65A"></i> SHAP — RF Explanation</div>
        <div class="p-2 rounded" style="background: var(--bg-card2); border: 1px solid var(--border)">
            <img src="data:image/png;base64,${rf.shap.shap_plot_b64}" class="img-fluid w-100" alt="SHAP RF"/>
        </div>
      </div>` : ''}

      <div class="mt-3 px-3">
        ${statusAdvice(ens?.status?.toLowerCase() || rf?.status?.toLowerCase() || 'healthy')}
      </div>
    </div>`;

  $('result-panel').innerHTML = html;
}


// ─── Prediction History (localStorage) ────────────────────────────────────

const HISTORY_KEY = 'rul-prediction-history';
const MAX_HISTORY = 50;

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch { return []; }
}

function saveToHistory(r) {
  const history = getHistory();
  history.unshift({
    timestamp: new Date().toISOString(),
    predicted_rul: r.predicted_rul,
    status: r.status,
    model_used: r.model_used,
    speed: r.speed,
    load: r.load,
    temp: r.temp,
    rul_pct: r.rul_pct,
    n_features: r.n_features,
    ci_lower: r.confidence_interval?.ci_lower,
    ci_upper: r.confidence_interval?.ci_upper,
  });
  // Keep only last MAX_HISTORY entries
  if (history.length > MAX_HISTORY) history.length = MAX_HISTORY;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  updateHistoryBadge();
}

function updateHistoryBadge() {
  const badge = $('history-count-badge');
  if (badge) {
    const count = getHistory().length;
    badge.textContent = count;
    badge.classList.toggle('d-none', count === 0);
  }
}

function renderHistory() {
  const container = $('history-container');
  if (!container) return;

  const history = getHistory();
  if (history.length === 0) {
    container.innerHTML = `
      <div class="text-center text-muted py-5">
        <i class="bi bi-clock-history fs-1 d-block mb-3" style="opacity:0.3"></i>
        <p>Nothing here yet! Run a prediction and it'll show up automatically.</p>
      </div>`;
    return;
  }

  let tableRows = history.map((h, i) => {
    const st = h.status?.toLowerCase() || 'healthy';
    const stColor = st === 'healthy' ? '#6F8F72' : st === 'degraded' ? '#F2A65A' : '#e85050';
    const ts = new Date(h.timestamp);
    const timeStr = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dateStr = ts.toLocaleDateString([], { month: 'short', day: 'numeric' });

    return `
      <tr class="history-row">
        <td class="hist-cell-time">
          <div class="hist-date">${dateStr}</div>
          <div class="hist-time">${timeStr}</div>
        </td>
        <td>
          <span class="hist-model-badge">${h.model_used || '—'}</span>
        </td>
        <td>
          <span class="hist-rul-val" style="color:${stColor}">${h.predicted_rul?.toLocaleString() ?? '—'}</span>
          <span class="hist-rul-unit">cyc</span>
        </td>
        <td>
          <span class="hist-status-dot" style="background:${stColor}"></span>
          ${h.status || '—'}
        </td>
        <td class="hist-cell-ci">
          ${h.ci_lower != null ? `${h.ci_lower.toLocaleString()} – ${h.ci_upper.toLocaleString()}` : '—'}
        </td>
        <td class="hist-cell-conditions">
          ${h.speed ?? '—'} RPM / ${h.load ?? '—'} N / ${h.temp ?? '—'} °C
        </td>
      </tr>`;
  }).join('');

  container.innerHTML = `
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div class="small text-muted">${history.length} prediction${history.length > 1 ? 's' : ''} recorded</div>
      <button class="btn btn-sm btn-outline-danger" id="btn-clear-history" style="font-size:11px">
        <i class="bi bi-trash3"></i> Clear History
      </button>
    </div>
    <div class="table-responsive">
      <table class="table table-dark table-sm table-hover history-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Model</th>
            <th>RUL</th>
            <th>Status</th>
            <th>95% CI</th>
            <th>Conditions</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>`;

  $('btn-clear-history')?.addEventListener('click', () => {
    const modal = new bootstrap.Modal($('confirm-clear-modal'));
    modal.show();
    $('btn-confirm-clear').onclick = () => {
      localStorage.removeItem(HISTORY_KEY);
      renderHistory();
      updateHistoryBadge();
      modal.hide();
    };
  });
}


function infoTile(label, val) {
  return `<div class="col-4">
    <div class="metric-cell py-2">
      <div style="font-size:13px;font-weight:700;color:#E8E2D8;font-family:'JetBrains Mono',monospace">${val}</div>
      <div style="font-size:10px;color:#8a9490;text-transform:uppercase;letter-spacing:0.3px">${label}</div>
    </div>
  </div>`;
}

function statusAdvice(status) {
  const advice = {
    healthy: { icon: '🟢', text: 'Looking good! Your bearing is healthy. Keep up the regular check-ups 👍', cls: 'success' },
    degraded: { icon: '🟡', text: 'Heads up — the bearing is showing some wear. Check it more often from now on.', cls: 'warning' },
    critical: { icon: '🔴', text: 'Attention needed! This bearing is close to failure. Schedule maintenance ASAP ⚠️', cls: 'danger' },
  };
  const a = advice[status] || advice.healthy;
  return `<div class="alert alert-${a.cls} py-2 px-3 text-start small mb-0">
    <span class="me-1">${a.icon}</span> ${a.text}
  </div>`;
}

// ─── Helpers ──────────────────────────────────────────────────────────────
function setPredicting(loading, mode = 'csv') {
  const btn = mode === 'csv' ? $('btn-predict-csv') : $('btn-predict-manual');
  if (!btn) return;
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Predicting…';
  } else {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-lightning-charge-fill"></i> Predict RUL';
  }
}

function showError(msg) {
  $('error-msg').textContent = msg;
  new bootstrap.Modal($('error-modal')).show();
}

function formatNum(n) {
  if (!n && n !== 0) return '—';
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return n.toLocaleString();
}

// ─── Refresh button ───────────────────────────────────────────────────────
$('btn-refresh-eda').addEventListener('click', () => {
  // Clear all cached data
  state.edaData = null;
  state.edaStatus = 'idle';

  // Clear all visible plot containers so stale images are gone
  $('eda-plot-container').innerHTML =
    '<p class="text-muted py-5"><span class="spinner-border spinner-border-sm me-2"></span> Re-running EDA analysis — this may take a minute…</p>';
  $('eda-plot-description').textContent = 'ℹ️ Regenerating plots…';
  const d2 = $('eda-plot-desc-line2');
  if (d2) d2.textContent = '';
  const ab = $('eda-plot-abbr');
  if (ab) ab.innerHTML = '';

  // Button loading state
  const btn = $('btn-refresh-eda');
  const origHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Running…';

  // Navigate to EDA section so user sees progress
  showSection('eda');

  // Force a full EDA re-run, then restore button and re-render active tab
  startEda(true).finally(() => {
    btn.disabled = false;
    btn.innerHTML = origHTML;
    // Re-render the currently active tab with fresh data
    if (state.edaData) renderEdaPlot(state.activeEdaPlot);
  });
});


// ─── Theme Toggle ─────────────────────────────────────────────────────────
function applyTheme(isLight) {
  const body = document.body;
  const icon = $('theme-icon');
  if (isLight) {
    body.classList.add('light-mode');
    icon.className = 'bi bi-moon-stars-fill';
    $('btn-theme-toggle').title = 'Switch to Dark Mode';
  } else {
    body.classList.remove('light-mode');
    icon.className = 'bi bi-sun-fill';
    $('btn-theme-toggle').title = 'Switch to Light Mode';
  }
}

$('btn-theme-toggle').addEventListener('click', () => {
  const isNowLight = !document.body.classList.contains('light-mode');
  applyTheme(isNowLight);
  localStorage.setItem('rul-theme', isNowLight ? 'light' : 'dark');
});

// ─── Export Report ─────────────────────────────────────────────────────────
function _loadScript(url) {
  return new Promise((resolve, reject) => {
    const id = 's_' + url.split('/').pop().replace(/[^a-z0-9]/gi, '');
    if (document.getElementById(id)) { resolve(); return; }
    const s = document.createElement('script');
    s.id = id;
    s.src = url;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

function _exportBaseName() {
  if (state.lastFileName) return state.lastFileName;
  return 'RUL_Report_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-');
}

function exportResultPNG() {
  const panel = $('result-panel');
  if (!panel || !panel.innerText.trim()) return;
  _loadScript('https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js')
    .then(() => html2canvas(panel, { backgroundColor: '#1a1f1e', scale: 2, useCORS: true, allowTaint: true }))
    .then(canvas => {
      const link = document.createElement('a');
      link.download = `${_exportBaseName()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    })
    .catch(err => showError('Export PNG failed: ' + err.message));
}

function exportResultPDF() {
  const panel = $('result-panel');
  if (!panel || !panel.innerText.trim()) return;
  Promise.all([
    _loadScript('https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js'),
    _loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js'),
  ])
    .then(() => html2canvas(panel, { backgroundColor: '#1a1f1e', scale: 2, useCORS: true, allowTaint: true }))
    .then(canvas => {
      const { jsPDF } = window.jspdf;
      const imgData = canvas.toDataURL('image/png');
      // Page = exact image size, no margins, no header
      const pxToMm = 0.264583;
      const w = canvas.width * pxToMm / 2;
      const h = canvas.height * pxToMm / 2;
      const pdf = new jsPDF({ unit: 'mm', format: [w, h] });
      pdf.addImage(imgData, 'PNG', 0, 0, w, h);
      pdf.save(`${_exportBaseName()}.pdf`);
    })
    .catch(err => showError('Export PDF failed: ' + err.message));
}

// ─── Batch CSV Upload ─────────────────────────────────────────────────────
const batchInput = $('batch-file-input');
const batchZone = $('batch-upload-zone');

if (batchZone && batchInput) {
  batchZone.addEventListener('click', () => batchInput.click());
  batchZone.addEventListener('dragover', e => { e.preventDefault(); batchZone.classList.add('drag-over'); });
  batchZone.addEventListener('dragleave', () => batchZone.classList.remove('drag-over'));
  batchZone.addEventListener('drop', e => {
    e.preventDefault();
    batchZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleBatchFiles(e.dataTransfer.files);
  });
  batchInput.addEventListener('change', () => {
    if (batchInput.files.length) handleBatchFiles(batchInput.files);
  });
}

function handleBatchFiles(fileList) {
  const files = Array.from(fileList).filter(f => f.name.endsWith('.csv'));
  if (files.length === 0) { showError('No CSV files found.'); return; }

  const batchInfo = $('batch-file-info');
  if (batchInfo) {
    batchInfo.innerHTML = `📁 ${files.length} CSV file(s) selected <button type="button" class="btn btn-sm ms-2" style="font-size:10px;padding:0 6px;color:#e85050;border:1px solid rgba(232,80,80,0.3);border-radius:4px" id="btn-clear-batch">✕ Clear</button>`;
    $('btn-clear-batch')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      batchInput.value = '';
      batchInfo.innerHTML = '';
      $('btn-batch-predict')?.classList.add('d-none');
    });
  }

  const btn = $('btn-batch-predict');
  if (btn) {
    btn.classList.remove('d-none');
    btn.onclick = () => runBatchPredict(files);
  }
}

async function runBatchPredict(files) {
  const btn = $('btn-batch-predict');
  const origHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Processing…';

  const fd = new FormData();
  files.forEach(f => fd.append('files[]', f));
  fd.append('model', $('csv-model')?.value || 'rf');
  fd.append('speed', $('csv-speed')?.value || '1800');
  fd.append('load', $('csv-load')?.value || '4000');
  fd.append('temp', $('csv-temp')?.value || '35');

  try {
    const resp = await fetch('/api/predict/batch', { method: 'POST', body: fd });
    const json = await resp.json();
    if (json.error) throw new Error(json.error);
    state.lastFileName = `Batch_Results_${files.length}_files`;
    showBatchResults(json.results);
    // Save all to history
    json.results.forEach(r => { if (!r.error) saveToHistory(r); });
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origHTML;
  }
}

function showBatchResults(results) {
  const valid = results.filter(r => !r.error);
  const errCount = results.length - valid.length;
  const avgRul = valid.length ? Math.round(valid.reduce((s, r) => s + r.predicted_rul, 0) / valid.length) : 0;
  const minRul = valid.length ? Math.min(...valid.map(r => r.predicted_rul)) : 0;
  const maxRul = valid.length ? Math.max(...valid.map(r => r.predicted_rul)) : 0;
  const healCount = valid.filter(r => r.status?.toLowerCase() === 'healthy').length;
  const degCount = valid.filter(r => r.status?.toLowerCase() === 'degraded').length;
  const critCount = valid.filter(r => r.status?.toLowerCase() === 'critical').length;

  let rows = results.map(r => {
    if (r.error) {
      return `<div class="batch-item"><div class="batch-fname">${r.filename}</div><span class="text-danger small">❌ ${r.error}</span></div>`;
    }
    const st = r.status?.toLowerCase() || 'healthy';
    const color = st === 'healthy' ? '#6F8F72' : st === 'degraded' ? '#F2A65A' : '#e85050';
    const pct = Math.min(r.rul_pct ?? 0, 100);
    return `<div class="batch-item">
      <div class="batch-status-dot" style="background:${color}"></div>
      <div class="batch-fname" title="${r.filename}">${r.filename}</div>
      <div style="flex:1;max-width:120px"><div class="compare-progress-track"><div class="rul-progress-fill" style="width:${pct}%;background:${color}"></div></div></div>
      <div class="batch-rul" style="color:${color}">${r.predicted_rul?.toLocaleString()}</div>
      <span class="small" style="color:${color};min-width:65px">${r.status}</span>
    </div>`;
  }).join('');

  $('result-panel').innerHTML = `
    <div class="w-100">
      <div class="text-center mb-3">
        <i class="bi bi-collection-fill fs-3" style="color:#6F8F72"></i>
        <div class="fw-bold mt-1">📦 Batch Results — ${results.length} files analyzed</div>
      </div>
      <div class="row g-2 mb-3 px-2">
        <div class="col-4 col-md-2"><div class="metric-cell py-2"><div class="metric-cell-val" style="font-size:18px;color:#BFC6C4">${avgRul.toLocaleString()}</div><div class="metric-cell-label">Avg RUL</div></div></div>
        <div class="col-4 col-md-2"><div class="metric-cell py-2"><div class="metric-cell-val" style="font-size:18px;color:#e85050">${minRul.toLocaleString()}</div><div class="metric-cell-label">Min RUL</div></div></div>
        <div class="col-4 col-md-2"><div class="metric-cell py-2"><div class="metric-cell-val" style="font-size:18px;color:#6F8F72">${maxRul.toLocaleString()}</div><div class="metric-cell-label">Max RUL</div></div></div>
        <div class="col-4 col-md-2"><div class="metric-cell py-2"><div class="metric-cell-val" style="font-size:18px;color:#6F8F72">${healCount}</div><div class="metric-cell-label">🟢 Healthy</div></div></div>
        <div class="col-4 col-md-2"><div class="metric-cell py-2"><div class="metric-cell-val" style="font-size:18px;color:#F2A65A">${degCount}</div><div class="metric-cell-label">🟡 Degraded</div></div></div>
        <div class="col-4 col-md-2"><div class="metric-cell py-2"><div class="metric-cell-val" style="font-size:18px;color:#e85050">${critCount}</div><div class="metric-cell-label">🔴 Critical</div></div></div>
      </div>
      ${errCount ? `<div class="alert alert-danger py-1 px-3 small mx-2 mb-2">${errCount} file(s) had errors</div>` : ''}
      <div class="batch-results rounded" style="background:var(--bg-card2);border:1px solid var(--border)">${rows}</div>
      <div class="mt-3 d-flex gap-2 justify-content-center">
        <button class="btn btn-export btn-sm" onclick="exportResultPNG()"><i class="bi bi-image"></i> Export PNG</button>
        <button class="btn btn-export btn-sm" onclick="exportResultPDF()"><i class="bi bi-file-earmark-pdf"></i> Export PDF</button>
      </div>
    </div>`;
}

// ─── Init ─────────────────────────────────────────────────────────────────
(async () => {
  // Restore saved theme (default: dark)
  const savedTheme = localStorage.getItem('rul-theme');
  applyTheme(savedTheme === 'light');

  updateHistoryBadge();
  showSection('overview');
  await startEda(false);
  // Load models in background
  await loadModels();
})();
