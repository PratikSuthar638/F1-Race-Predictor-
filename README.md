# 🏎️ F1 Race Predictor — Machine Learning Based Podium Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An authentic, end-to-end Machine Learning college engineering project that predicts whether a Formula 1 driver is likely to finish on the **Podium (Top 3)** in a Grand Prix. The system utilizes real historical data (1950–2024 Ergast Kaggle dataset), enforces strict pre-race feature engineering to prevent data leakage, trains and evaluates 5 classification models using chronological validation, and deploys predictions via a modern F1-themed Flask dashboard.

---

## 📌 Problem Statement & Objective

Formula 1 is one of the most technologically advanced and data-rich sports in the world. However, predicting race outcomes is challenging due to stochastic elements (safety cars, weather, mechanical failures) and shifting technical regulations. 

Rather than predicting exact numerical positions (which introduces high regression error), this project formulates a **supervised binary classification problem**:

$$\text{Podium} = \begin{cases} 1 & \text{if } \text{positionOrder} \in \{1, 2, 3\} \quad (\text{Podium Finish}) \\ 0 & \text{if } \text{positionOrder} \ge 4 \text{ or DNF} \quad (\text{No Podium}) \end{cases}$$

The objective is to output:
* **Prediction Outcome:** `PODIUM` / `NO PODIUM`
* **Podium Probability:** Calibrated probability percentage ($0.0\% - 100.0\%$)
* **Model Confidence:** Statistical certainty metric
* **Feature Impact Breakdown:** Transparent, pre-race factors driving the prediction

---

## 🛡️ Strict Pre-Race Data Leakage Prevention

A primary requirement of this project is to simulate an authentic **PRE-RACE** prediction before the starting lights go out.

| Prohibited Post-Race Features (Dropped) | Pre-Race Alternatives (Used) |
| :--- | :--- |
| Final race position & points | Starting grid & qualifying position |
| Race duration & fastest lap speed | Historical driver recent form score (last 5 finishes) |
| In-race pit stops & lap times | Constructor recent performance rating |
| Current race championship standings | Prior championship standings (lagged to round $k-1$) |
| Current race status / DNF status | Driver & team historical DNF rates |

All rolling statistics are calculated with a strict cumulative historical window:
$$\text{Historical Stats}(t) = f(\{\text{Race } k \mid \text{date}(k) < \text{date}(t)\})$$

---

## ⏱️ Chronological Validation Strategy

Formula 1 results are temporally correlated and subject to technical regulation changes. Therefore, random train/test splitting causes future data leakage. We enforce **Chronological Splitting**:

* **Training Set (2010–2021):** 5,077 driver starts across the V8 and Hybrid V6 turbo eras.
* **Validation Set (2022):** 440 driver starts during the major ground-effect aerodynamic regulation shift.
* **Testing Set (2023–2024):** 919 driver starts used as the unseen evaluation benchmark.

---

## 📊 Model Comparison & Benchmark Results

All 5 classification models were trained and benchmarked on the unseen 2023–2024 test dataset (calibrated threshold $\tau = 0.35$ for the ~15% podium class base rate):

| Rank | Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest Classifier** | **85.96%** | **0.5198** | **0.8551** | **0.6466** | **0.9237** |
| 🥈 | **Gradient Boosting** | **87.92%** | **0.5808** | **0.7029** | **0.6361** | **0.9202** |
| 🥉 | **XGBoost Classifier** | **84.87** | **0.4979** | **0.8406** | **0.6253** | **0.9211** |
| 4 | **Decision Tree** | 81.28% | 0.4366 | 0.8478 | 0.5764 | 0.8622 |
| 5 | **Logistic Regression** | 79.11% | 0.4140 | 0.9420 | 0.5752 | 0.9195 |

> **Best Model:** **Random Forest** achieved the highest overall **F1-Score (0.6466)** and **ROC-AUC (0.9237)** with balanced precision and high recall (85.51%).

---

## 📈 Visualizations & Figures

Generated visual analytics in `reports/figures/` and displayed on the web dashboard:
1. `chart1_podiums_by_driver.png`: Top 10 all-time drivers by total podium finishes.
2. `chart2_podiums_by_constructor.png`: Top 10 constructors by podium finishes.
3. `chart3_grid_vs_podium.png`: Starting grid slot vs historical podium probability.
4. `chart4_driver_recent_form.png`: Driver recent form score vs actual race finish.
5. `chart5_constructor_performance.png`: Team performance rating vs podium rate.
6. `chart6_model_comparison.png`: Multi-metric comparison bar chart across all 5 models.
7. `chart7_feature_importance.png`: Top 12 engineered pre-race feature importances.
8. `chart8_confusion_matrix.png`: Test set confusion matrix heatmap.
9. `chart9_roc_curves.png`: Multi-model ROC curves with AUC scores.

---

## 🛠️ Project Structure

```
F1-Race-Predictor/
│
├── data/
│   ├── raw/                  # 14 original Ergast Kaggle CSV files
│   └── processed/            # 26,759 rows engineered pre-race features dataset
│
├── notebooks/
│   └── f1_analysis.ipynb     # Interactive Jupyter notebook for EDA and modeling
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Loading and cleaning raw Ergast tables
│   ├── preprocessing.py      # Imputation, scalers, and chronological splits
│   ├── feature_engineering.py# Strict leak-free rolling stats computation
│   ├── train.py              # Training pipeline for all 5 models & serialization
│   ├── evaluate.py           # Metric calculation and figure generation
│   └── prediction.py         # Real-time inference engine and factor explanations
│
├── models/
│   ├── best_model.pkl        # Serialized Random Forest model
│   ├── preprocessing.pkl     # Fitted scalers and imputer pipeline
│   └── model_metadata.pkl    # Metadata & evaluation benchmarks
│
├── reports/
│   ├── figures/              # 9 publication-quality analytical charts
│   └── model_results.csv     # Model comparison benchmark table
│
├── templates/
│   ├── base.html             # Common navbar, styling, and footer
│   ├── index.html            # Landing / Hero page
│   ├── predict.html          # Interactive 2-column F1 prediction console
│   ├── dashboard.html        # Analytics & ML performance dashboard
│   └── about.html            # Documentation, architecture, and methodology
│
├── static/
│   ├── css/style.css         # Modern dark carbon F1 styling & glassmorphism
│   ├── js/script.js          # Dynamic slider bindings & AJAX predict handler
│   └── images/               # Generated charts and diagrams
│
├── app.py                    # Flask web application entrypoint
├── requirements.txt          # Python dependencies
├── README.md                 # Complete project documentation
├── PROJECT_REPORT.md         # 23-section College Engineering Report
├── VIVA_QUESTIONS.md         # 45+ College Viva examination Q&As
└── .gitignore
```

---

## 🚀 How to Run the Project

### 1. Prerequisites & Virtual Environment
Make sure Python 3.10+ is installed on your machine.

```bash
# Clone or navigate to the project directory
cd "F1 Race predictor"

# Install all required dependencies
pip install -r requirements.txt
```

### 2. Run the Feature Engineering & Training Pipeline
```bash
# Generate leak-free features and train all 5 models
python src/train.py
```

### 3. Start the Flask Web Application
```bash
# Run the local Flask server
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🎓 College Viva & Documentation Links
* [PROJECT_REPORT.md](file:///PROJECT_REPORT.md): Comprehensive 23-section academic project report.
* [VIVA_QUESTIONS.md](file:///VIVA_QUESTIONS.md): 45+ detailed Questions & Answers for external viva examination.
* [f1_analysis.ipynb](file:///notebooks/f1_analysis.ipynb): Jupyter notebook walkthrough.
