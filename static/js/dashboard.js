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

