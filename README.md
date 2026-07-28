# 🚲 Citi Bike Availability Prediction Pipeline

End-to-end machine learning pipeline that predicts **low dock availability** at Citi Bike stations using real-time station data, with **scheduled ingestion, weekly retraining, and cloud deployment on AWS**.

---

# Project Overview

This project began as a supervised learning workflow for predicting whether a Citi Bike station would experience **low dock availability** in the near future. It was later expanded into a lightweight **MLOps pipeline** that automates data collection, retraining, and model serving.

The system is designed to:

- collect fresh Citi Bike station snapshots on a schedule
- store raw data in AWS S3
- rebuild training data from recent snapshots
- retrain the model automatically each week
- serve predictions through a FastAPI endpoint deployed on AWS

---

# Business Problem

Bike-share systems can become operationally inefficient when stations run low on available docks. Predicting these shortages in advance can help support station rebalancing, improve rider experience, and reduce friction during peak usage periods.

Rather than predicting current station conditions, the objective is to provide advance notice of potential dock shortages so operational teams can rebalance stations before they become unavailable.

---

# Target

Binary classification target:

- `1` = station is projected to have **low dock availability**
- `0` = station is **not** projected to have low dock availability

In this project, low dock availability is defined as **10% or less of total station capacity** remaining as open docks.

---

# Data Sources

This project uses Citi Bike GBFS feeds, including:

- `station_information.json`
- `station_status.json`

These feeds provide station metadata and real-time operational availability data, which are combined into a modeling dataset.

Additional engineered features include:

- station capacity
- station latitude and longitude
- distance to nearest MTA station
- current dock availability percentage
- hour of day
- day of week

### Data Preparation Highlights

Raw Citi Bike GBFS station metadata and status feeds were merged into a unified station-level dataset.

During preprocessing:

- New Jersey stations were excluded so the model focused exclusively on NYC operations.
- Invalid observations (such as stations with zero capacity or impossible dock availability values) were removed.
- Station snapshots were standardized to consistent timestamps before downstream feature engineering.
- Geographic station metadata was merged with real-time operational data to create a complete modeling dataset.

---

# Notebook Workflow

The notebooks document the end-to-end analytical workflow.

## 1. Preprocessing

`01_citi_bike_prediction_preprocessing.ipynb`

- combines raw station snapshots into a structured dataset
- merges station status with station metadata
- cleans columns and standardizes timestamps
- prepares the base table used in analysis

---

## 2. Exploratory Data Analysis

`02_citi_bike_prediction_exploratory_analysis.ipynb`

- examines station availability patterns
- explores hourly and weekday behavior
- reviews low-availability frequency and class balance
- identifies trends that inform feature engineering

### Key Findings

Exploratory analysis revealed several patterns that informed feature engineering and model selection.

- Low dock availability events represented a relatively small percentage of observations, confirming that this is an imbalanced classification problem.
- Dock availability followed clear hourly patterns consistent with commuter demand.
- Weekday behavior differed from weekends, demonstrating that temporal features were likely to improve predictive performance.
- Stations near major transit corridors exhibited greater variability in dock availability than neighborhood stations.
- These findings motivated the inclusion of time-based and location-based engineered features.

---

## 3. Feature Engineering

`03_citi_bike_prediction_feature_engineering.ipynb`

- sorts station snapshots chronologically
- creates future-looking targets using grouped time shifting
- engineers time-based and station-level features
- removes or avoids identifiers that could weaken generalization

### Feature Engineering Decisions

Several engineered features were created to improve predictive performance while minimizing target leakage.

- Future dock availability was generated using grouped time shifting to create a 30-minute prediction target.
- Hour of day and day of week features captured cyclical commuting patterns.
- Distance to the nearest MTA station was incorporated as a proxy for transit accessibility.
- Station identifiers were excluded from model training to encourage better generalization.
- Features containing future information were removed to prevent target leakage.

---

## 4. Modeling

`04_citi_bike_prediction_modeling.ipynb`

- compares Logistic Regression, Random Forest, XGBoost, and TensorFlow
- evaluates model performance using classification metrics
- emphasizes performance under class imbalance
- selects **XGBoost** as the final deployment model based on overall holdout performance

---

# Modeling Approach

The project compares several model types to balance interpretability, predictive performance, and deployment practicality.

- **Logistic Regression** for a clean baseline
- **Random Forest** for nonlinear tabular modeling
- **XGBoost** for boosted tree performance
- **TensorFlow Neural Network** as an experimental deep learning benchmark

Evaluation focused on:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Because this is an imbalanced classification problem, **PR-AUC** was prioritized during model selection over accuracy.

---

# Model Performance

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|-------|--------:|-------:|----------:|-------:|---------:|
| **XGBoost** | **0.816** | **0.466** | 0.397 | **0.750** | **0.519** |
| TensorFlow | 0.795 | 0.438 | **0.507** | 0.313 | 0.387 |
| Random Forest | 0.795 | 0.433 | 0.376 | 0.725 | 0.495 |
| Logistic Regression | 0.625 | 0.273 | 0.237 | 0.621 | 0.343 |

### Results Analysis

Logistic Regression established a strong baseline but was limited by its linear decision boundary, resulting in substantially lower PR-AUC than the tree-based models.

Random Forest and XGBoost both captured nonlinear relationships between station characteristics, geographic location, and temporal demand, producing significant performance improvements across every evaluation metric.

Although the TensorFlow neural network achieved the highest precision, it sacrificed recall by identifying considerably fewer low dock availability events.

Because the operational objective is to proactively identify stations at risk of running out of docks, recall and PR-AUC were prioritized over precision alone.

**XGBoost was selected as the final deployment model** because it achieved:

**Live FastAPI Docs:** [Citi Bike Prediction API](https://hjpkidba7c.us-east-1.awsapprunner.com/docs#/)
- the highest PR-AUC
- the highest recall
- the highest F1-score
- the strongest overall balance across evaluation metrics


These results indicate that gradient-boosted decision trees provide the most effective solution for predicting future low dock availability while maintaining competitive precision.

---

# AWS / MLOps Architecture

This project includes a cloud-based retraining and deployment workflow using AWS services.

- EventBridge schedules ingestion and retraining
- Lambda pulls Citi Bike GBFS data
- Amazon S3 stores raw snapshots
- CodeBuild performs weekly retraining
- Amazon ECR stores Docker images
- AWS App Runner hosts the FastAPI prediction API

---

# Weekly Retraining Workflow

1. Citi Bike snapshot data is collected automatically.
2. Raw JSON files are stored in Amazon S3.
3. Weekly jobs rebuild the training dataset.
4. XGBoost is retrained.
5. Updated artifacts are generated.
6. The API is redeployed with the latest model.

Weekly retraining enables the model to adapt to changing station usage patterns over time while reducing the impact of concept drift.

---

# API Deployment

The prediction service is packaged as a FastAPI application inside a Docker container and deployed on AWS App Runner.

**Live FastAPI Docs**

https://er8i8uv8hc.us-east-1.awsapprunner.com/docs#/default/predict_predict_post

Deployment includes:

- Docker image build
- Versioned image tagging
- Amazon ECR push
- AWS App Runner deployment

The API accepts current station conditions and returns the probability that a station will experience low dock availability within the next 30 minutes.

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

- Machine learning for tabular classification
- Time-based feature engineering
- Leakage-aware target construction
- Model comparison and evaluation
- FastAPI model serving
- Docker containerization
- AWS scheduling, storage, and deployment
- Lightweight MLOps workflow design

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

- NYC 311 ML API
- Brooklyn Home Price API

---

# Author

**Justin Cox**

GitHub:
https://github.com/jac6779

LinkedIn:
https://www.linkedin.com/in/justincox1