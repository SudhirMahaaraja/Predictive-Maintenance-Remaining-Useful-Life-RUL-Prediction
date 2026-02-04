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
FEMTO_PATH = "path/to/FEMTO datasets/"
XJTU_PATH = "path/to/XJTU-SY_Bearing_Datasets/"
```

---

## � Usage

### 1. Launching the Web Dashboard
The easiest way to interact with the project is through the web application.

```bash
python app.py
```
* Then open your browser and navigate to `http://localhost:5000`
* **Single Prediction:** Upload a single vibration CSV file to get an instant RUL prediction.
* **Batch Prediction:** Upload all CSV files belonging to a specific bearing to visualize its health degradation over time.

### 2. Training the Models (CLI)
To retrain the models on your data:
```bash
python train_rf.py
```
*This will extract features, train Random Forest and XGB models, and save them to `rf_models/`.*

### 3. Testing on a Single File (CLI)
```bash
python test_rf.py --csv "datasets/FEMTO datasets/Learning_set/Bearing1_1/acc_00100.csv"
```

### 4. Running Exploratory Data Analysis (EDA)
Generate correlation matrices, feature distributions, and signal plots:
```bash
python eda.py
```

---

## 🔬 Scientific Approach & Feature Engineering

Instead of feeding raw time-series data into a black-box deep learning model, this project relies on **domain knowledge** to compress the vibration signals into 43 highly informative features per file:

- **Time-Domain (16 features):** Mean, Standard Deviation, RMS, Peak Value, Kurtosis, Skewness, Crest Factor, Shape Factor (Calculated for horizontal and vertical vibration channels).
- **Frequency-Domain (16 features):** Frequency Center, Mean Square Frequency, Variance Frequency. Also extracts specific defect frequencies based on the physical bearing geometry (BPFO, BPFI, BSF, FTF).
- **Envelope Analysis (8 features):** Characteristics of the enveloped signal to detect impacts hidden in the noise.
- **Operating Conditions (3 features):** Normalized values representing rotational speed and radial load.

### Why Classical ML over Deep Learning here?
1. **Speed & Efficiency:** Trains in minutes on a standard CPU, compared to hours/days required for CNNs/LSTMs.
2. **Data Efficiency:** Performs exceptionally well even with a limited number of run-to-failure bearing trajectories.
3. **Interpretability:** Tree-based models provide feature importance scores, allowing engineers to understand exactly *why* a particular prediction is made.

---

## 📈 Performance Tracking

*Note: Results may vary slightly depending on the exact train/test split of the datasets.*

| Model | MAE (Mean Absolute Error) | R² Score |
| :--- | :---: | :---: |
| **Random Forest** | ~180-220 cycles | 0.65 - 0.75 |
| **XGBoost** | ~150-200 cycles | 0.70 - 0.80 |

*Both models consistently outperform baseline deep learning approaches on these specific datasets when training data is limited.*

---

## �️ Built With
* **Python** - Core logic and ML
* **Scikit-Learn & XGBoost** - Machine Learning models
* **Pandas & NumPy** - Data manipulation
* **SciPy** - Signal processing and feature extraction
* **Flask** - Web framework
* **Vanilla HTML/CSS/JS** - Custom frontend architecture
* **Chart.js** - Interactive frontend data visualization

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open source and available under the [MIT License](LICENSE).
