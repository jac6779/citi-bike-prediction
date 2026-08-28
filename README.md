# 🚲 Citi Bike Availability Prediction Pipeline

End-to-end machine learning pipeline that predicts whether a currently healthy Citi Bike station will fall to **10% or less dock availability within the next 60 minutes**, with scheduled ingestion, automated retraining, and cloud deployment on AWS.

---

# Project Overview

This project began as a supervised learning workflow for predicting low dock availability and was later expanded into a lightweight MLOps pipeline. The system collects fresh Citi Bike data, rebuilds training data, retrains the model, and serves predictions through a FastAPI endpoint on AWS.

The system is designed to:

- collect fresh Citi Bike station snapshots on a schedule
- store raw data in Amazon S3
- rebuild training data from recent snapshots
- retrain the selected model automatically
- serve predictions through a FastAPI endpoint deployed on AWS

---

# Business Problem

Bike-share systems can become operationally inefficient when stations run low on available docks. The goal is to identify stations that still have adequate dock availability now but are at risk of becoming constrained within the next hour, giving operators time to rebalance stations before availability becomes critical.

---

# Target

Binary classification target:

- `1` = station is currently above 10% dock availability but falls to **10% or less within 60 minutes**
- `0` = station remains above the low-dock threshold within the prediction window

The modeling population is limited to stations that are currently above the 10% threshold so the model acts as an early-warning system rather than simply identifying stations that are already low on docks.

---

# Data Sources

This project uses Citi Bike GBFS feeds, including:

- `station_information.json`
- `station_status.json`

These feeds provide station metadata and real-time operational availability data, which are combined into a station-level time series.

Model features include:

- station capacity
- station latitude and longitude
- distance to nearest MTA station
- current dock availability percentage
- hour of day
- day of week

### Data Preparation Highlights

Raw Citi Bike station metadata and status feeds are merged into a unified dataset.

During preprocessing:

- New Jersey stations are excluded so the model focuses on NYC operations.
- Stations with invalid capacity or availability values are removed.
- Timestamps are standardized and station observations are sorted chronologically.
- Distance to the nearest MTA station is calculated from station coordinates.
- Future dock availability is created by shifting each station's time series forward approximately 60 minutes.
- Only future observations separated by 55–65 minutes are retained.
- The final modeling population is restricted to stations currently above 10% dock availability.

---

# Notebook Workflow

The notebooks document the end-to-end analytical workflow.

## 1. Preprocessing

`01_citi_bike_prediction_preprocessing.ipynb`

- combines raw station snapshots into a structured dataset
- merges station status with station metadata
- cleans timestamps and invalid observations
- calculates current and future dock availability
- creates the 60-minute binary prediction target
- filters the modeling population to currently healthy stations

---

## 2. Exploratory Data Analysis

`02_citi_bike_prediction_exploratory_analysis.ipynb`

- examines station availability patterns
- explores hourly and weekday behavior
- reviews class balance and high-risk stations
- looks at geographic differences across the Citi Bike network

### Key Findings

- Low-dock events represent only about 4% of observations, making this a strongly imbalanced classification problem.
- Risk changes throughout the day, with noticeably higher low-dock rates during busier periods.
- Low-dock risk is concentrated among a smaller group of stations rather than being evenly distributed across the network.
- These patterns support the use of temporal, geographic, station-capacity, and current-availability features.

---

## 3. Feature Engineering

`03_citi_bike_prediction_feature_engineering.ipynb`

- sorts the data chronologically
- uses an 80/20 time-based train/test split
- standardizes continuous variables
- converts hour and weekday into cyclical features
- fits preprocessing only on the training period to avoid test-set leakage
- saves the fitted preprocessor for later inference

### Feature Engineering Decisions

- Current dock availability is retained because it is known at prediction time and is an important operational signal.
- Hour of day and day of week are represented cyclically to preserve their repeating structure.
- Distance to the nearest MTA station is included as a measure of transit proximity.
- Station identifiers and future-looking fields are excluded from model training.

---

## 4. Modeling

`04_citi_bike_prediction_modeling.ipynb`

- compares Logistic Regression, Random Forest, and XGBoost
- evaluates each model using ROC-AUC, PR-AUC, precision, recall, and F1-score
- compares ML performance against a simple current-dock-availability baseline
- tests Random Forest threshold sensitivity
- selects **Random Forest** as the final model based on PR-AUC

---

# Modeling Approach

The project compares several model types to balance interpretability, nonlinear performance, and deployment practicality.

- **Logistic Regression** provides a linear benchmark.
- **Random Forest** captures nonlinear relationships across station, time, and availability features.
- **XGBoost** provides a boosted-tree comparison.
- **Current Dock % Baseline** tests whether the ML models add value beyond simply ranking stations by their current availability.

Evaluation focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Because low-dock events are rare, **PR-AUC is the primary model-selection metric**.

---

# Model Performance

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|-------|--------:|-------:|----------:|-------:|---------:|
| **Random Forest** | **0.927** | **0.363** | 0.257 | 0.793 | 0.388 |
| XGBoost | **0.927** | 0.359 | 0.257 | **0.797** | **0.389** |
| Logistic Regression | 0.895 | 0.267 | 0.216 | 0.751 | 0.336 |
| Current Dock % Baseline | 0.887 | 0.265 | — | — | — |

*Precision, recall, and F1-score are shown at the common 0.70 classification threshold used for the initial model comparison.*

### Results Analysis

Random Forest and XGBoost clearly outperform Logistic Regression and the simple current-availability baseline on PR-AUC. This is important because current dock availability alone is already a strong predictor, so the tree-based models need to improve on that benchmark to justify the added complexity.

**Random Forest is selected as the final model** because it produces the highest PR-AUC at **0.363**, compared with **0.359** for XGBoost and **0.265** for the current-dock baseline.

### Threshold Sensitivity

Random Forest was also evaluated across thresholds from 0.30 to 0.90. A threshold around **0.80** provides a useful operating balance, with approximately **30.6% precision, 66.3% recall, and a 0.419 F1-score** while preserving nearly the best observed F1-score.

---

# AWS / MLOps Architecture

This project includes a cloud-based data collection, retraining, and deployment workflow using AWS services.

- EventBridge schedules ingestion and retraining
- Lambda pulls Citi Bike GBFS data
- Amazon S3 stores raw snapshots
- CodeBuild performs scheduled retraining
- Amazon ECR stores Docker images
- AWS App Runner hosts the FastAPI prediction API

---

# Retraining Workflow

1. Citi Bike snapshot data is collected automatically.
2. Raw JSON files are stored in Amazon S3.
3. Scheduled jobs rebuild the training dataset.
4. The selected model is retrained on updated data.
5. Updated artifacts are generated.
6. The prediction service can be redeployed with the latest model.

This workflow allows the model to be refreshed as station usage patterns change over time.

---

# API Deployment

The prediction service is packaged as a FastAPI application inside a Docker container and deployed on AWS App Runner.

**Live FastAPI Docs**

https://er8i8uv8hc.us-east-1.awsapprunner.com/docs#/default/predict_predict_post

Deployment includes:

- Docker image build
- versioned image tagging
- Amazon ECR push
- AWS App Runner deployment

The API accepts current station conditions and returns the probability that a currently healthy station will fall to low dock availability within the next **60 minutes**.

---

# Repository Structure

```text
citi-bike-prediction/
├── app/
│   ├── main.py
│   └── feature_builder.py
├── training/
│   ├── build_training_data.py
│   └── train_model.py
├── models/
├── notebooks/
│   ├── 01_citi_bike_prediction_preprocessing.ipynb
│   ├── 02_citi_bike_prediction_exploratory_analysis.ipynb
│   ├── 03_citi_bike_prediction_feature_engineering.ipynb
│   └── 04_citi_bike_prediction_modeling.ipynb
├── Dockerfile
├── buildspec.yml
└── README.md
```

---

# Key Skills Demonstrated

- machine learning for imbalanced tabular classification
- time-based train/test splitting
- leakage-aware target construction
- baseline benchmarking and model comparison
- threshold sensitivity analysis
- FastAPI model serving
- Docker containerization
- AWS scheduling, storage, retraining, and deployment
- lightweight MLOps workflow design

---

# Future Improvements

Potential future enhancements include:

- incorporating weather data
- additional temporal lag features
- automated drift monitoring
- model version tracking
- CI/CD for fully automated deployment

---

# Related Projects

- NYC 311 Complaint Resolution Prediction(https://github.com/jac6779/nyc-311-ml-api)
- Brooklyn Home Price Prediction(https://github.com/jac6779/brooklyn-home-sales-llm)

---

# Author

**Justin Cox**

GitHub:  
https://github.com/jac6779

LinkedIn:  
https://www.linkedin.com/in/justincox1
