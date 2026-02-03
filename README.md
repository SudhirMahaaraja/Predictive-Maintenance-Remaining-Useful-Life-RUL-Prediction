# Predictive Maintenance: Remaining Useful Life (RUL) Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-orange)
![Web Dashboard](https://img.shields.io/badge/Frontend-Flask%20%7C%20JS%20%7C%20CSS-brightgreen)

A comprehensive, end-to-end Machine Learning project for predicting the Remaining Useful Life (RUL) of industrial bearings using classical machine learning algorithms (Random Forest and XGBoost). This project features a complete data pipeline, exploratory data analysis tools, and a beautiful web-based dashboard for real-time inference and batch processing.

---

## 🌟 Project Overview

Industrial machinery maintenance often relies on replacing parts *before* they break to avoid catastrophic failure, leading to wasted resources. This project leverages vibration signals from bearings to predict exactly how many operational cycles remain before failure (RUL), transitioning from **preventative maintenance** to **predictive maintenance**.

Unlike computationally expensive Deep Learning models, this project utilizes meticulously engineered physics-based features coupled with powerful ensemble methods to achieve high accuracy with minimal computational overhead.

### ✨ Key Features
- **Robust Feature Engineering:** Extracts 43 meaningful time-domain, frequency-domain, and envelope features from raw vibration data.
- **Ensemble ML Models:** Implements both Random Forest and XGBoost for robust, generalizable predictions.
- **Interactive Web Dashboard:** A sleek, user-friendly Flask-based web interface to run predictions, view history, and export reports (PDF/PNG).
- **Batch Processing:** Ability to upload multiple CSV files at once to visualize the degradation trend of a bearing over its complete lifecycle.
- **Exploratory Data Analysis (EDA):** Built-in tools (`eda.py`) to analyze feature distributions, correlations, and signal degradation over time.

---

## 📁 Project Structure

```text
rul-prediction-rf-xgb/
├── app.py                # Main Flask application for the web dashboard
├── rf_config.py          # Configuration file (Hyperparameters, Data Paths)
├── rf_data_loader.py     # Utilities for loading FEMTO and XJTU-SY datasets
├── rf_features.py        # Logic for extracting 43 physics-based features
├── train_rf.py           # Script to train RF and XGBoost models
├── test_rf.py            # Script for CLI-based inference and evaluation
├── eda.py                # Exploratory Data Analysis toolset
├── clear.py              # Utility to clear cached outputs/predictions
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation (this file)
├── templates/
│   └── index.html        # Web dashboard layout
├── static/
│   ├── css/dashboard.css # Dashboard styling
│   ├── js/dashboard.js   # Dashboard interactive logic
│   └── favicon.png       # App icon
└── rf_models/            # Directory containing pickled trained models
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rul-prediction-rf-xgb.git
cd rul-prediction-rf-xgb
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

This project natively supports two widely used bearing datasets:
1. **PRONOSTIA (FEMTO) Bearing Dataset**
2. **XJTU-SY Bearing Dataset**

To train the models from scratch, download these datasets and update the paths in `rf_config.py`:
```python
