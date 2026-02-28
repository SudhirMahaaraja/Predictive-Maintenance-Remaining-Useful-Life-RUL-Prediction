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

