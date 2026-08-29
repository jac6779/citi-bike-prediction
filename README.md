# 🚲 Citi Bike Availability Prediction Pipeline

End-to-end machine learning pipeline that predicts whether a Citi Bike station will have **10% or less dock availability 30 minutes ahead**, with scheduled ingestion, rolling retraining, performance monitoring, and cloud deployment on AWS.

---

# Project Overview

This project began as a supervised learning workflow for predicting low dock availability and was later expanded into a lightweight MLOps pipeline. The system collects fresh Citi Bike data, rebuilds a rolling training dataset, retrains the model weekly, publishes evaluation metrics to Amazon CloudWatch, and serves predictions through a FastAPI endpoint on AWS.

The system is designed to:

- collect fresh Citi Bike station snapshots on a schedule
- store raw data in Amazon S3
- rebuild training data from a rolling 30-day window
- retrain the selected model automatically each week
- publish model performance metrics to CloudWatch
- serve predictions through a FastAPI endpoint deployed on AWS

---

# Business Problem

Bike-share systems can become operationally inefficient when stations run low on available docks. The goal is to forecast which stations are likely to have critically low dock availability 30 minutes ahead, giving operators a short-term signal that can support station rebalancing and operational planning.

The production collection window focuses on active daytime hours, approximately **7:00 AM to 6:30 PM Eastern**, rather than overnight periods with lower operational value.

---

# Target

Binary classification target:

- `1` = station has **10% or less dock availability 30 minutes ahead**
- `0` = station remains above the low-dock threshold 30 minutes ahead

The target is created from each station's next snapshot, with only observations approximately **25–35 minutes apart** retained to keep the prediction horizon consistent.

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
- hour of day
- day of week

### Data Preparation Highlights

Raw Citi Bike station metadata and status feeds are merged into a unified dataset.

During preprocessing:

- New Jersey stations are excluded so the model focuses on NYC operations.
- Stations with invalid capacity or availability values are removed.
- Timestamps are standardized and station observations are sorted chronologically.
- Distance to the nearest MTA station is calculated from station coordinates.
- Future dock availability is created by shifting each station's time series forward one snapshot.
- Only future observations separated by approximately 25–35 minutes are retained.
- The binary target is defined as future dock availability at or below 10%.

---

# Notebook Workflow

The notebooks document the end-to-end analytical workflow.

## 1. Preprocessing

`01_citi_bike_prediction_preprocessing.ipynb`

- combines raw station snapshots into a structured dataset
- merges station status with station metadata
- cleans timestamps and invalid observations
- calculates future dock availability
- creates the 30-minute binary prediction target
- retains only observations with a consistent 25–35 minute future interval

---

## 2. Exploratory Data Analysis

`02_citi_bike_prediction_exploratory_analysis.ipynb`

- examines station availability patterns
- reviews class balance and low-dock behavior
- identifies stations with higher concentrations of low-dock events
- looks at geographic differences across the Citi Bike network

### Key Findings

- Low-dock events make up roughly **22% of observations** in the current 30-day dataset, so the positive class remains the minority class.
- Low-dock risk is concentrated among a smaller group of stations rather than being evenly distributed across the network.
- Geographic and station-level differences support the use of location, capacity, transit proximity, and temporal features.
- The production modeling window is intentionally limited to active daytime hours rather than overnight periods.

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

- Current dock availability is excluded from the final model to avoid relying on a strong short-term persistence signal.
- Hour of day and day of week are represented cyclically to preserve their repeating structure.
- Distance to the nearest MTA station is included as a measure of transit proximity.
- Station identifiers and future-looking fields are excluded from model training.

---

## 4. Modeling

`04_citi_bike_prediction_modeling.ipynb`

- compares Logistic Regression, Random Forest, and XGBoost
- evaluates each model using ROC-AUC, PR-AUC, precision, recall, and F1-score
- uses consistent default classification thresholds for the initial comparison
- selects **XGBoost** as the final model based primarily on PR-AUC

---

# Modeling Approach

The project compares several model types to balance interpretability, nonlinear performance, and deployment practicality.

- **Logistic Regression** provides a linear benchmark.
- **Random Forest** captures nonlinear relationships across station, geographic, and temporal features.
- **XGBoost** provides a boosted-tree model that can capture more complex interactions.

Evaluation focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Because low-dock events are the minority class, **PR-AUC is the primary model-selection metric**.

---

# Model Performance

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|-------|--------:|-------:|----------:|-------:|---------:|
| **XGBoost** | **0.814** | **0.563** | 0.455 | **0.712** | **0.555** |
| Random Forest | 0.797 | 0.545 | **0.457** | 0.663 | 0.542 |
| Logistic Regression | 0.587 | 0.291 | 0.263 | 0.649 | 0.375 |

### Results Analysis

ROC-AUC and PR-AUC both show the tree-based models outperforming Logistic Regression. **XGBoost performs best overall**, with a PR-AUC of **0.563** and ROC-AUC of **0.814**, slightly ahead of Random Forest.

The gap between Logistic Regression and the tree-based models suggests that low-dock risk is driven by nonlinear relationships and interactions that are not captured well by a simple linear decision boundary.

---

# AWS / MLOps Architecture

This project includes a cloud-based data collection, retraining, monitoring, and deployment workflow using AWS services.

- EventBridge schedules Citi Bike snapshot collection and weekly retraining
- Lambda pulls Citi Bike GBFS data
- Amazon S3 stores raw snapshots
- CodeBuild rebuilds the rolling training dataset and retrains the model
- Amazon CloudWatch tracks weekly model performance
- Amazon ECR stores Docker images
- AWS App Runner hosts the FastAPI prediction API

---

# Retraining Workflow

1. Citi Bike snapshot data is collected automatically during the active daytime window.
2. Raw JSON files are stored in Amazon S3.
3. The weekly training job rebuilds the dataset using the most recent **30 days** of snapshots.
4. XGBoost is retrained on the updated data.
5. ROC-AUC, PR-AUC, precision, recall, and F1 are published to CloudWatch.
6. Updated model artifacts are generated for the prediction service.

The rolling 30-day window helps the model stay responsive to changing station usage patterns rather than depending on a fixed historical training period.

---

# Model Monitoring

Weekly retraining performance is tracked in Amazon CloudWatch using custom metrics under the production XGBoost model.

Recent retraining runs have produced PR-AUC values generally in the low-to-mid **0.50s**, providing a simple way to confirm that scheduled retraining is completing successfully and to monitor performance over time.

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

The API returns the probability that a Citi Bike station will have low dock availability within the next **30 minutes**.

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
- nonlinear model comparison
- cyclical temporal feature engineering
- FastAPI model serving
- Docker containerization
- AWS scheduling, storage, retraining, monitoring, and deployment
- rolling-window model retraining
- lightweight MLOps workflow design

---

# Future Improvements

Potential future enhancements include:

- incorporating weather data
- additional temporal lag features
- automated performance alerts and drift thresholds
- model version tracking
- CI/CD for fully automated deployment

---

# Related Projects

- [NYC 311 Complaint Resolution Prediction](https://github.com/jac6779/nyc-311-ml-api)
- [Brooklyn Home Price Prediction](https://github.com/jac6779/brooklyn-home-sales-llm)

---

# Author

**Justin Cox**

GitHub:  
https://github.com/jac6779

LinkedIn:  
https://www.linkedin.com/in/justincox1
