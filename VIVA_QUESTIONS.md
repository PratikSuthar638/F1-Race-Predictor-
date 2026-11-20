# F1 RACE PREDICTOR — COLLEGE VIVA PREPARATION GUIDE
## 45+ Comprehensive Questions & Answers for External Examination

---

### 1. Fundamentals of Machine Learning & Problem Formulation

#### Q1. What is Machine Learning?
**Answer:** Machine Learning is a subfield of Artificial Intelligence where computer algorithms learn patterns directly from empirical historical data to make inferences or predictions on unseen data without being explicitly programmed with deterministic rules.

#### Q2. Is this project Supervised or Unsupervised Learning?
**Answer:** This project is **Supervised Learning**. We provide the model with input feature vectors ($X$) along with corresponding ground-truth labeled targets ($y$, where $y=1$ indicates a Podium finish and $y=0$ indicates No Podium).

#### Q3. Why is this formulated as a Classification problem rather than Regression?
**Answer:** Predicting an exact finishing position (e.g. 1st vs 2nd vs 14th) via regression introduces high stochastic noise because lower-order positions (P10–P20) are heavily distorted by late-race retirements, pit stop timing, and safety cars. In contrast, predicting whether a driver reaches the **Podium (Top 3)** represents a distinct, high-signal sporting milestone with clear decision boundaries.

#### Q4. What is the target variable in this project?
**Answer:** The target variable is binary:
* `Podium = 1`: Driver finishes in position 1, 2, or 3 (`positionOrder <= 3`).
* `Podium = 0`: Driver finishes in position 4 or lower, or suffers a DNF/retirement.

#### Q5. What is the base rate of the target variable in Formula 1?
**Answer:** In a standard 20 to 22 car Formula 1 grid, exactly 3 drivers finish on the podium per race. Across the historical dataset, the positive class (`Podium = 1`) accounts for approximately **12.7% to 15.0%** of all driver race starts. This represents a moderate **class imbalance**.

---

### 2. Data Leakage & Chronological Splitting

#### Q6. What is Data Leakage in Machine Learning?
**Answer:** Data leakage occurs when information from outside the training dataset or information generated *after* the target event has occurred is inadvertently introduced into the model's feature set. This results in unrealistically high training/validation scores that collapse in real-world deployment.

#### Q7. How did you prevent Data Leakage in this project?
**Answer:**
1. **Dropped In-Race Features:** Removed race finishing time, fastest lap speed/rank, laps completed, and in-race status codes.
2. **Excluded Real-Time Telemetry Tables:** Completely excluded `lap_times.csv` and `pit_stops.csv`.
3. **Lagged Championship Standings:** Used driver and constructor championship standings from round $k-1$ (prior to the race weekend).
4. **Cumulative Historical Rolling Windows:** All career and recent form metrics were calculated using strictly prior races: $\{\text{Race } j \mid \text{date}(j) < \text{date}(\text{current race})\}$.

#### Q8. Why did you use Chronological Splitting instead of Random $K$-Fold Cross-Validation?
**Answer:** Formula 1 is a dynamic time-series system where car performance evolves race-by-race and rules change between seasons. Random splitting would allow a model to train on a 2024 race to predict a 2018 race (temporal leakage). Chronological splitting (Train: 2010–2021, Val: 2022, Test: 2023–2024) guarantees that the model only learns from past events.

---

### 3. Feature Engineering & Domain Logic

#### Q9. What is Feature Engineering?
**Answer:** Feature engineering is the process of using domain knowledge to extract, transform, and construct meaningful mathematical attributes from raw data that make machine learning algorithms learn patterns more effectively.

#### Q10. How did you calculate the "Driver Recent Form" score?
**Answer:** We computed a normalized score from 0.0 to 10.0 based on the driver's previous 5 race finishes using **exponential decay weights** ($w = [0.35, 0.25, 0.20, 0.12, 0.08]$), giving greater importance to the most recent races:
$$\text{Score} = \text{clip}\left(10 \times \left(1 - \frac{\sum w_j \cdot \text{pos}_j - 1}{19}\right), 0, 10\right)$$

#### Q11. How did you calculate "Constructor Performance"?
**Answer:** We aggregated the finishing positions of both cars fielded by the constructor over their previous 5 Grand Prix events, applying exponential decay to create a team pace rating (0.0 to 10.0).

#### Q12. Why is Starting Grid Position one of the most important features?
**Answer:** In modern Formula 1 aerodynamic design ("dirty air"), overtaking is difficult. Historically, pole position (P1) converts to a podium finish over 72% of the time, whereas starting outside the top 10 yields a podium rate under 3%.

#### Q13. What is the `qualifying_diff_team` feature?
**Answer:** It measures the difference between a driver's qualifying position and the average qualifying position of their constructor's cars. A negative value indicates the driver significantly out-qualified their car's baseline pace.

#### Q14. How did you handle Track / Circuit features?
**Answer:** For each track, we calculated the driver's historical average finish, driver podium rate at that circuit, constructor track podium rate, and circuit altitude.

---

### 4. Data Preprocessing & Missing Data

#### Q15. How did you handle missing values in the raw Ergast dataset?
**Answer:** Ergast uses `\N` string tokens to denote null values (e.g., in qualifying or unfinished race times). We parsed `\N` into `np.nan` and applied median imputation using `SimpleImputer` fitted strictly on the training set.

#### Q16. How did you handle grid position 0?
**Answer:** In Formula 1 records, grid position 0 indicates a car that started from the pit lane following a penalty or parc fermé setup change. We imputed this value as position 20 (back of the grid).

#### Q17. Why did you scale the features?
**Answer:** We used `StandardScaler` to standardize features to zero mean and unit variance. This ensures that models sensitive to feature magnitudes (like Logistic Regression) converge properly.

---

### 5. Machine Learning Algorithms & Ensembles

#### Q18. Which 5 classification algorithms were evaluated?
**Answer:**
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

#### Q19. How does Logistic Regression work?
**Answer:** It applies a linear combination of input features passed through the Sigmoid activation function: $\sigma(z) = \frac{1}{1 + e^{-z}}$ to output a bounded probability between 0 and 1.

#### Q20. How does a Decision Tree work?
**Answer:** A Decision Tree recursively partitions feature space using greedy threshold splits that maximize Information Gain or Gini Impurity reduction.

#### Q21. What is Random Forest and why did it perform best?
**Answer:** Random Forest is an ensemble of diverse decision trees built via **Bagging (Bootstrap Aggregation)** and random feature sub-sampling. It reduces variance, prevents overfitting to specific regulation eras, and captures non-linear interactions between grid slot and team form.

#### Q22. What is the difference between Bagging and Boosting?
**Answer:**
* **Bagging (Random Forest):** Trains multiple decision trees independently in parallel on bootstrap samples; averages their predictions to reduce variance.
* **Boosting (Gradient Boosting / XGBoost):** Trains decision trees sequentially, where each new tree specifically corrects the residual errors of the prior ensemble.

#### Q23. What is XGBoost and what are its advantages?
**Answer:** XGBoost (Extreme Gradient Boosting) is an optimized, scalable gradient boosting library that includes second-order Taylor expansion gradients, built-in L1/L2 tree regularization, and native handling of sparse data.

---

### 6. Model Evaluation & Metrics

#### Q24. Why is Accuracy alone misleading for this project?
**Answer:** Because only ~15% of drivers reach the podium, a naive "dummy" model that always predicts `0` (No Podium) would achieve **85% Accuracy** while being completely useless (0% Recall, 0 F1-score).

#### Q25. What is Precision?
**Answer:** $\text{Precision} = \frac{TP}{TP + FP}$. Out of all drivers predicted to finish on the podium, what percentage actually did?

#### Q26. What is Recall (Sensitivity)?
**Answer:** $\text{Recall} = \frac{TP}{TP + FN}$. Out of all drivers who truly finished on the podium, what percentage did the model correctly identify?

#### Q27. What is the F1-Score?
**Answer:** The F1-Score is the harmonic mean of Precision and Recall:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Q28. What is ROC-AUC?
**Answer:** Receiver Operating Characteristic &ndash; Area Under Curve. It measures the model's ability to rank positive instances (podiums) higher than negative instances across all possible classification probability thresholds.

#### Q29. What were the final results of your best model on the test set?
**Answer:** On the unseen 2023–2024 test seasons:
* **Algorithm:** Random Forest Classifier
* **ROC-AUC:** **0.9237**
* **F1-Score:** **0.6466**
* **Recall:** **85.51%**
* **Accuracy:** **85.96%**

#### Q30. What is a Confusion Matrix?
**Answer:** A $2 \times 2$ table showing the counts of True Positives (correctly predicted podiums), True Negatives (correctly predicted non-podiums), False Positives (predicted podium, but did not finish top 3), and False Negatives (achieved podium, but predicted no podium).

#### Q31. What decision threshold did you use and why?
**Answer:** We used a calibrated threshold of **0.35** instead of the default 0.50 to reflect the 15% class base rate in Formula 1.

---

### 7. Overfitting, Bias & Variance

#### Q32. What is Overfitting and how did you mitigate it?
**Answer:** Overfitting occurs when a model memorizes training noise and fails to generalize to unseen test data. We mitigated it by limiting tree depth (`max_depth=10`), setting minimum split samples (`min_samples_split=12`), using ensemble averaging (250 trees), and evaluating on unseen future seasons.

#### Q33. What is Underfitting?
**Answer:** Underfitting occurs when a model is too simplistic to capture underlying patterns (e.g. a simple linear model ignoring non-linear track interactions).

#### Q34. What is the Bias-Variance Tradeoff?
**Answer:** High bias causes underfitting (systematic error); high variance causes overfitting (sensitivity to training noise). Ensembles balance this tradeoff by combining low-bias trees to achieve low overall variance.

---

### 8. System Architecture & Web Deployment

#### Q35. What is the technology stack used in this project?
**Answer:**
* **Data Science / ML:** Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Joblib.
* **Backend:** Flask (Python Web Framework).
* **Frontend:** HTML5, CSS3 (Custom Dark Theme & Glassmorphism), Vanilla JavaScript.

#### Q36. How does the Flask web application serve predictions?
**Answer:** When a user selects a driver, team, circuit, and grid position, an AJAX POST request is sent to `/api/predict`. The Flask backend retrieves historical pre-race baselines, applies the saved `F1Preprocessor`, runs `best_model.pkl` inference, and returns calibrated probabilities and explanatory factor breakdowns in JSON format.

#### Q37. Why are serialized `.pkl` files used instead of retraining on every request?
**Answer:** Retraining across 26,000 historical records takes time; saving the pre-trained weights in `best_model.pkl` and `preprocessing.pkl` allows sub-50ms inference latency in the web application.

#### Q38. How does the application generate "Why this prediction?" explanations?
**Answer:** It dynamically evaluates the user's specific inputs (starting grid slot, driver form score, constructor rating, track podium conversion rate) against trained feature importance thresholds to produce clear positive and negative factors.

---

### 9. Limitations & Future Scope

#### Q39. What are the limitations of the current system?
**Answer:**
1. Does not dynamically track mid-race safety car probabilities.
2. In-season aerodynamic floor upgrades are inferred from race results rather than CAD/wind-tunnel CFD data.

#### Q40. How can this project be enhanced in the future?
**Answer:**
1. Integrate live telemetry and weather forecasts via the FastF1 API.
2. Implement ranking models (e.g. Bradley-Terry / Plackett-Luce models) to simulate all 20 driver positions simultaneously.
3. Deploy Deep Learning sequential models (LSTM / Transformers) for lap-by-lap telemetry tracking.

---

### 10. Rapid-Fire College Viva Summary

| Question | 1-Sentence Answer |
| :--- | :--- |
| **Project Title?** | F1 Race Predictor &ndash; Machine Learning Based Podium Prediction. |
| **Target Variable?** | Binary: 1 for Top 3 finish (Podium), 0 for P4 or lower. |
| **Dataset Source?** | Kaggle Ergast Formula 1 historical database (1950–2024, 14 tables). |
| **Splitting Strategy?** | Chronological: 2010–2021 Train, 2022 Val, 2023–2024 Test. |
| **Best Model?** | Random Forest Classifier. |
| **Best Model Metric?** | 0.9237 ROC-AUC, 0.6466 F1-Score, 85.96% Accuracy. |
| **Key Features?** | Starting Grid Position, Driver Recent Form Score, Constructor Performance Rating. |
| **Data Leakage Fix?** | Strict cumulative historical windowing; all in-race telemetry dropped. |
