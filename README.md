# Bike Sharing: High-Demand Hour Classification (Assignment 2, ML - AIML)

## a. Problem Statement
Predict whether an hour is a high-demand hour for bike rentals (hourly rental count above the dataset median) using weather, season, and calendar features. Five classifiers are trained on the same data and compared.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository, Bike Sharing Dataset (Capital Bikeshare, Washington D.C., 2011-2012), file `hour.csv`.
- **Instances:** 17,379 hourly records.
- **Features used (12):** season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed.
- **Removed:** `instant` and `dteday` (identifier and date string), and `casual`, `registered`, `cnt` (`casual + registered = cnt`, so they leak the target).
- **Target (derived):** `high_demand` = 1 if `cnt` > median `cnt` (142), else 0. Classes are roughly balanced.

## c. GitHub Repository Link
<!-- Paste your repo URL here -->

## d. Models Used
Evaluation on the 20% stratified held-out test split (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |

**Overall winner for this dataset:**

## How to Run Locally
```bash
pip install -r requirements.txt
python train_models.py    # creates model/*.joblib and test_data.csv
streamlit run app.py
```
