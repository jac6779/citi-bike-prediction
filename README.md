# 🚲 Citi Bike Availability Prediction Pipeline

End-to-end machine learning pipeline that predicts **low dock availability** at Citi Bike stations using real-time station data, with **scheduled ingestion, weekly retraining, and cloud deployment on AWS**.

## Project Overview

This project began as a supervised learning workflow for predicting whether a Citi Bike station would experience **low dock availability** in the near future. It was later expanded into a lightweight **MLOps pipeline** that automates data collection, retraining, and model serving.

The system is designed to:

- collect fresh Citi Bike station snapshots on a schedule
- store raw data in AWS S3
- rebuild training data from recent snapshots
- retrain the model automatically each week
- serve predictions through a FastAPI endpoint deployed on AWS

## Business Problem

Bike-share systems can become operationally inefficient when stations run low on available docks. Predicting these shortages in advance can help support station rebalancing, improve rider experience, and reduce friction during peak usage periods.

## Target

Binary classification target:

- `1` = station is projected to have **low dock availability**
- `0` = station is **not** projected to have low dock availability

In this project, low dock availability is defined as **10% or less of total station capacity** remaining as open docks.

## Data Sources

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

## Notebook Workflow

The notebooks document the end-to-end analytical workflow:

### 1. Preprocessing
`01_citi_bike_prediction_preprocessing.ipynb`

- combines raw station snapshots into a structured dataset
- merges station status with station metadata
- cleans columns and standardizes timestamps
- prepares the base table used in analysis

### 2. Exploratory Data Analysis
`02_citi_bike_prediction_exploratory_analysis.ipynb`

- examines station availability patterns
- explores hourly and weekday behavior
- reviews low-availability frequency and class balance
- identifies trends that inform feature engineering

### 3. Feature Engineering
`03_citi_bike_prediction_feature_engineering.ipynb`

- sorts station snapshots chronologically
- creates future-looking targets using grouped time shifting
- engineers time-based and station-level features
- removes or avoids identifiers that could weaken generalization

### 4. Modeling
`04_citi_bike_prediction_modeling.ipynb`

- compares Logistic Regression, Random Forest, XGBoost, and TensorFlow
- evaluates model performance using classification metrics
- emphasizes performance under class imbalance
- selects the best candidate for deployment

## Modeling Approach

The project compares several model types to balance interpretability, predictive performance, and deployment practicality:

- **Logistic Regression** for a clean baseline
- **Random Forest** for nonlinear tabular modeling
- **XGBoost** for boosted tree performance
- **TensorFlow Neural Network** as an experimental deep learning benchmark

Evaluation focused on standard classification metrics such as:

- precision
- recall
- F1 score
- ROC-AUC
- PR-AUC

Because this is an imbalanced classification problem, special attention was given to how well models identify the positive class rather than relying only on accuracy.

## AWS / MLOps Architecture

This project now includes a cloud-based retraining and deployment workflow using AWS services:

- **EventBridge** schedules ingestion and retraining jobs
- **Lambda** pulls Citi Bike snapshot data from the GBFS API
- **S3** stores raw snapshots and training inputs
- **CodeBuild** runs the weekly retraining workflow
- **ECR** stores the Docker image for the inference API
- **App Runner** hosts the FastAPI prediction service

## Weekly Retraining Workflow

The retraining workflow is designed to keep the model refreshed with recent operating patterns:

1. Citi Bike snapshot data is collected on a recurring schedule
2. Raw JSON snapshots are written to S3
3. A weekly training job rebuilds the training dataset from recent files
4. The model is retrained automatically
5. Updated artifacts are prepared for deployment
6. The containerized API can be redeployed with the refreshed model

This shifts the project from a one-time modeling exercise to an **automated ML system**.

## API Deployment

The prediction service is packaged as a **FastAPI application** inside a **Docker container** and deployed through AWS.

**Live FastAPI Docs:** [Citi Bike Prediction API](https://er8i8uv8hc.us-east-1.awsapprunner.com/docs#/default/predict_predict_post)

The deployment workflow includes:

- building the image locally
- tagging with a versioned ECR image tag
- pushing the image to Amazon ECR
- updating the App Runner service to use the latest model-serving image

## Repository Structure

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

## Key Skills Demonstrated

- machine learning for tabular classification
- time-based feature engineering
- leakage-aware target construction
- model comparison and evaluation
- FastAPI model serving
- Docker containerization
- AWS scheduling, storage, and deployment
- lightweight MLOps workflow design

## Future Improvements

Potential next steps include:

- more frequent retraining schedules
- automated model evaluation reporting
- model version tracking
- drift monitoring
- CI/CD improvements for fully automated redeployment

## Related Projects

- [NYC 311 ML API](https://github.com/jac6779/nyc-311-ml-api)
- [Brooklyn Home Price API](https://github.com/jac6779/brooklyn-home-price-api)

## Author

**Justin Cox**

- GitHub: [https://github.com/jac6779](https://github.com/jac6779)
- LinkedIn: [https://www.linkedin.com/in/justincox1](https://www.linkedin.com/in/justincox1)
