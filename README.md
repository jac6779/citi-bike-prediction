# 🚲 Citi Bike Dock Availability Prediction

An end-to-end machine learning project that predicts whether a Citi Bike station will experience **low dock availability in the next time interval**, enabling better operational awareness and future real-time decisioning.

This project follows a structured **CRISP-DM workflow** and is designed with future productionization in mind (AWS, scheduled retraining, API deployment).

---

## 📌 Project Overview

Citi Bike systems frequently experience imbalances where stations become either:

* **Full (no docks available)** or
* **Empty (no bikes available)**

This project focuses on predicting **low dock availability (≤10% capacity)** at a given station in the near future.

### 🎯 Target Variable

Binary classification:

* `1` → Low dock availability (≤10% of station capacity)
* `0` → Otherwise

---

## 📊 Data Sources

* Citi Bike GBFS snapshots:

  * `station_status` (real-time availability)
  * `station_information` (station metadata)

* Additional engineered features:

  * Temporal features (hour, weekday)
  * Station-level characteristics (capacity, location)
  * Distance to nearest MTA station

---

## 🧱 Project Structure (CRISP-DM)

### 1️⃣ Preprocessing (`01_citi_bike_prediction_preprocessing.ipynb`)

* Aggregates raw JSON snapshots into a structured dataset
* Cleans and filters NYC stations
* Merges station status with station metadata
* Handles missing and invalid values
* Converts timestamps and prepares time-based features

---

### 2️⃣ Exploratory Data Analysis (`02_citi_bike_prediction_exploratory_analysis.ipynb`)

* Distribution analysis of bike/dock availability
* Temporal patterns (hourly, weekday trends)
* Station-level variability
* Early signal identification for low dock conditions

---

### 3️⃣ Feature Engineering (`03_citi_bike_prediction_feature_engineering.ipynb`)

* Creates target variable using **future availability via groupby + shift**
* Engineers key features:

  * `current_dock_pct`
  * `future_dock_pct`
  * Time-based variables (hour, weekday)
* Handles feature selection decisions:

  * Removes identifiers (e.g., station_id)
  * Evaluates categorical vs continuous variables
* Sorts data chronologically for proper train/test splitting

---

### 4️⃣ Modeling (`04_citi_bike_prediction_modeling.ipynb`)

* Implements and compares multiple models:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * Neural Network (TensorFlow, experimental)

* Evaluation metrics:

  * Precision
  * Recall
  * F1 Score
  * ROC-AUC

* Key focus:

  * Handling class imbalance
  * Comparing model performance across approaches
  * Selecting best model for future deployment

---

## 🤖 Modeling Approach

This project intentionally compares:

* **Interpretable models** → Logistic Regression
* **Tree-based models** → Random Forest, XGBoost
* **Deep learning (optional)** → Neural Network

This mirrors real-world workflows where simpler models often outperform or complement deep learning depending on the problem.

---

## ⚙️ Key Techniques

* Time-series aware feature engineering using `.groupby().shift()`
* Chronological train/test split (no leakage)
* Feature scaling where appropriate
* Model comparison framework across multiple algorithms

---

## 🚀 Future Improvements (Planned)

* Automated data ingestion via AWS (Lambda + S3)
* Scheduled model retraining (monthly pipeline)
* API deployment (FastAPI + Docker)
* Real-time predictions
* Dashboard integration (Tableau / Athena)

---

## 📈 Why This Project Matters

This project demonstrates:

* End-to-end ML workflow
* Real-world data challenges (time dependency, imbalance)
* Feature engineering for temporal prediction
* Model comparison and evaluation
* Readiness for production deployment

---

## 🔗 Related Projects

* NYC 311 Complaint Resolution Prediction
* Brooklyn Home Price Prediction API

These projects complement this work by showcasing:

* Tabular ML modeling
* API deployment
* Production-oriented workflows

---

## 👤 Author

Justin Cox

GitHub: https://github.com/jac6779
LinkedIn: https://linkedin.com/in/justincox1

---
