# F1 RACE PREDICTOR — MACHINE LEARNING BASED PODIUM PREDICTION
## College Project Report | Department of Computer Science & Artificial Intelligence Engineering

---

## 1. Title
**F1 Race Predictor: A Leakage-Free Pre-Race Machine Learning Classification Framework for Formula 1 Grand Prix Podium Prediction**

---

## 2. Abstract
Predicting race outcomes in Formula 1 is a complex task governed by non-linear aerodynamics, team developmental cycles, tire degradation, and dynamic qualifying sessions. Existing sports analytics systems often suffer from severe post-race data leakage and unrealistic random data splitting. This project presents an end-to-end Machine Learning system designed to predict whether a driver will achieve a Podium finish (Top 3) using strictly pre-race historical features. 

Using 75 seasons of official FIA Formula 1 data (1950–2024, 1,125 Grand Prix events, 26,759 driver entries), we engineered 26 pre-race features capturing driver recent form (exponential decay over previous starts), constructor baseline performance, track-specific podium conversion rates, and qualifying differentials. We established a chronological evaluation strategy: training on 2010–2021 (5,077 starts), validating on 2022 (440 starts), and evaluating on unseen modern seasons 2023–2024 (919 starts). Five classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost) were systematically benchmarked. The Random Forest Classifier demonstrated superior generalization with an **ROC-AUC of 0.9237**, **F1-Score of 0.6466**, and **Accuracy of 85.96%**. The model is deployed via a full-stack Flask web application featuring a modern F1 telemetry UI and transparent feature-impact explanations.

---

## 3. Introduction
Formula 1 represents the pinnacle of motorsport engineering, combining cutting-edge mechanical technology with elite human athleticism. Modern Formula 1 cars generate hundreds of megabytes of telemetry per lap; however, for fans, teams, and predictive analysts, the central question before every Grand Prix remains: *"Which drivers will step onto the podium?"*

Traditional predictive models in motorsport frequently rely on basic heuristics or naive averages. Furthermore, academic machine learning literature in motorsport prediction frequently commits methodological errors, such as training on random splits or including in-race data (such as fastest lap times or pit stop strategies) that cannot be known prior to race start. This project bridges these gaps by constructing a rigorous, leakage-free machine learning pipeline paired with an interactive, responsive analytics console.

---

## 4. Problem Statement
Given pre-race parameters available prior to race start for race $R$ held at circuit $C$ with driver $D$ starting from grid position $G$ for constructor $T$:
Predict the binary outcome variable $y \in \{0, 1\}$ defined as:
$$y = \begin{cases} 1 & \text{if driver finishes in positions 1, 2, or 3 (Podium)} \\ 0 & \text{if driver finishes in position 4 or lower, or suffers a DNF} \end{cases}$$
along with a calibrated probability $P(y=1 \mid X)$ and a confidence score.

---

## 5. Motivation
1. **High Inherent Complexity:** Formula 1 is subject to aerodynamic regulation changes (e.g., 2014 Hybrid transition, 2022 ground-effect shift), making historical modeling non-trivial.
2. **Realistic Pre-Race Modeling:** Most existing sports predictors operate in hindsight; developing a strictly pre-race model mirrors genuine industry forecasting.
3. **Addressing Class Imbalance:** In a standard 20-car grid, only 3 drivers (15%) reach the podium. Proper metric selection (F1-score, ROC-AUC) and probability calibration are critical.

---

## 6. Objectives
1. Clean and integrate 14 relational tables from the Kaggle Ergast Formula 1 dataset.
2. Formulate a strictly leakage-free feature engineering pipeline with cumulative historical rolling windows.
3. Enforce chronological dataset partitioning (2010–2021 Train, 2022 Val, 2023–2024 Test).
4. Train, tune, and benchmark 5 diverse classification algorithms.
5. Identify and rank key pre-race predictive drivers using feature importance.
6. Deploy the best model via an interactive, responsive Flask web application.

---

## 7. Existing System vs. Proposed System

| Dimension | Existing / Naive Approaches | Proposed System (F1 Race Predictor) |
| :--- | :--- | :--- |
| **Target Formulation** | Regression on exact finish position (high noise) | Calibrated Binary Podium Classification |
| **Data Splitting** | Random $K$-Fold / Random Train-Test (Data Leakage) | Strict Chronological Splitting (2010-21 / 2022 / 2023-24) |
| **Feature Validity** | Uses in-race telemetry (pit stops, lap times) | 100% Pre-Race Features (Grid, Form, Track History) |
| **Class Imbalance** | Ignored (evaluates on misleading accuracy) | Balanced class weights, calibrated $\tau=0.35$ threshold |
| **Deployment** | Static scripts or simple CLI | Modern responsive F1 telemetry web dashboard |

---

## 8. Proposed System Architecture
The system architecture follows a modular, decoupled pipeline:
```
Raw CSV Tables (14 Datasets)
       ↓
Data Loader & Relational Merging
       ↓
Chronological Rolling Feature Engine (Strictly Pre-Race)
       ↓
Preprocessing & Standardization Pipeline
       ↓
Chronological Train / Val / Test Split
       ↓
Multi-Model Training & Benchmark (5 Classifiers)
       ↓
Best Model Serialization (Random Forest: 0.9237 ROC-AUC)
       ↓
Flask Web Application & REST APIs
       ↓
Interactive Frontend (Predictor Console & Analytics Dashboard)
```

---

## 9. Dataset Description
The dataset was sourced from the official Ergast Formula 1 World Championship database (1950–2024):
* **`races.csv`** (1,125 rows): Chronological calendar, circuit mappings, dates.
* **`results.csv`** (26,759 rows): Final finishing ranks, points, grid slots.
* **`drivers.csv`** (861 rows): Driver biographical data and unique references.
* **`constructors.csv`** (212 rows): Constructor histories and identities.
* **`circuits.csv`** (77 rows): Geographical coordinates, locations, and altitudes.
* **`qualifying.csv`** (10,494 rows): Qualifying positions across modern eras.
* **`driver_standings.csv` & `constructor_standings.csv`**: Historical championship point progressions.
* **`status.csv`** (139 rows): Detailed mechanical/accident classification codes.

---

## 10. Data Preprocessing
1. **Null Representation:** Handled `\N` string tokens by mapping them to `np.nan` and applying median imputation.
2. **Missing Grid Slots:** Replaced grid 0 values (pit lane starts) with position 20.
3. **Qualifying Alignment:** Imputed missing historical qualifying sessions using official starting grid positions.
4. **Standardization:** Applied `StandardScaler` to ensure zero mean and unit variance across continuous feature inputs.

---

## 11. Feature Engineering
We engineered 26 pre-race features categorized into four distinct domains:

### 11.1 Driver Features
* `driver_recent_form_score`: Exponentially weighted score over last 5 starts ($w = [0.35, 0.25, 0.20, 0.12, 0.08]$):
  $$\text{Score} = \text{clip}\left(10 \times \left(1 - \frac{\sum w_j \cdot \text{pos}_j - 1}{19}\right), 0, 10\right)$$
* `driver_career_podium_rate`: Total prior podiums divided by total career starts.
* `driver_recent_dnf_rate_last_5`: Rate of mechanical/accident retirements in last 5 starts.

### 11.2 Constructor Features
* `constructor_recent_performance_score`: Weighted recent form across team entries.
* `constructor_career_podium_rate`: Historical constructor podium conversion percentage.
* `constructor_recent_dnf_rate_last_5`: Reliability failure rate over last 10 entries.

### 11.3 Grid & Qualifying Features
* `grid_position`: Starting slot on the race grid.
* `qualifying_diff_team`: Driver qualifying position relative to constructor average (driver pace delta).

### 11.4 Circuit & Track Context Features
* `driver_circuit_podium_rate`: Driver's historical podium rate at the specific track.
* `circuit_altitude`: Elevation above sea level (influences turbo vs cooling efficiency).

---

## 12. Exploratory Data Analysis (EDA)
* **Grid vs. Podium Probability:** Starting P1 yields a ~72% historical podium conversion rate, dropping sharply to ~14% by P6 and <3% beyond P10.
* **Class Imbalance:** Across 26,759 entries, only 3,397 (12.69%) are podium finishes.
* **Era Evolution:** Average car reliability improved from a 48% DNF rate in the 1960s to <12% in the modern Hybrid era (2014–2024).

---

## 13. Machine Learning Algorithms
Five distinct classification paradigms were evaluated:
1. **Logistic Regression:** Linear probabilistic baseline with L2 regularization.
2. **Decision Tree Classifier:** Non-parametric hierarchical tree with cost-complexity pruning.
3. **Random Forest Classifier:** Bagged ensemble of 250 decision trees using sub-sampled feature spaces.
4. **Gradient Boosting Classifier:** Boosted sequential decision trees minimizing log-loss.
5. **XGBoost Classifier:** Extreme gradient boosting utilizing regularized objective functions.

---

## 14. Model Training & Validation
* **Training Window (2010–2021):** 5,077 samples.
* **Validation Window (2022):** 440 samples (used for probability calibration).
* **Testing Window (2023–2024):** 919 samples (unseen benchmark).

---

## 15. Model Evaluation
Models were evaluated using **F1-Score**, **ROC-AUC**, **Precision**, **Recall**, and **Accuracy** on the 2023–2024 test seasons.

---

## 16. Results & Discussion

### Model Benchmark Table (Test Set 2023–2024):
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **85.96%** | **0.5198** | **0.8551** | **0.6466** | **0.9237** |
| **Gradient Boosting** | **87.92%** | **0.5808** | **0.7029** | **0.6361** | **0.9202** |
| **XGBoost** | **84.87%** | **0.4979** | **0.8406** | **0.6253** | **0.9211** |
| **Decision Tree** | 81.28% | 0.4366 | 0.8478 | 0.5764 | 0.8622 |
| **Logistic Regression** | 79.11% | 0.4140 | 0.9420 | 0.5752 | 0.9195 |

Random Forest proved most resilient to regulation shifts, capturing high-order interactions between grid position and team performance.

---

## 17. System Architecture & Modules
* `src/data_loader.py`: Relational ETL pipeline.
* `src/feature_engineering.py`: Time-series rolling feature generation.
* `src/preprocessing.py`: Imputation, scaling, and chronological split.
* `src/train.py`: Model training, comparison, and serialization.
* `src/evaluate.py`: Statistical validation and chart generation.
* `src/prediction.py`: Low-latency inference engine with explanatory factors.
* `app.py`: Flask web application routes.

---

## 18. Web Application
* **Prediction Console (`/predict`):** 2-column layout matching F1 telemetry interfaces.
* **Analytics Dashboard (`/dashboard`):** 8 visual charts and ML benchmark tables.
* **Documentation (`/about`):** Architecture and academic project methodology.

---

## 19. Advantages
1. Zero data leakage ensures realistic pre-race utility.
2. Chronological validation prevents temporal bias.
3. Transparent feature explanations build user trust.
4. Robust fallback handling for novel drivers or teams.

---

## 20. Limitations
1. Does not currently integrate live in-session weather sensors.
2. Mid-season aerodynamic upgrade packages are inferred from results rather than CAD/wind-tunnel data.

---

## 21. Future Scope
1. Real-time telemetry ingestion via FastF1 API.
2. Simulation of multi-car race outcomes using Dirichlet / Bradley-Terry ranking models.
3. Neural network architectures (LSTM / Transformers) for time-series driver trajectory analysis.

---

## 22. Conclusion
The F1 Race Predictor successfully implements an authentic, leakage-free machine learning framework for Grand Prix podium prediction. Achieving an **ROC-AUC of 0.9237** on the 2023–2024 test seasons, the project demonstrates that pre-race grid position, exponentially decayed recent form, and constructor ratings provide a powerful signal for motorsport outcome prediction.

---

## 23. References
1. Ergast Developer API & Historical Formula 1 World Championship Dataset (1950–2024).
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
