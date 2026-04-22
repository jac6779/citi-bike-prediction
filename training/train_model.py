# Core libraries
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from feature_engine.creation import CyclicalFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import json
import joblib
import boto3
import os

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean_data"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load and prepare data
print("Loading preprocessed data...")
data = pd.read_parquet(CLEAN_DATA_DIR / "01_citi_bike_prediction_preprocessing.parquet")
model_data = data.sort_values(by="snapshot_time", ascending=True).reset_index(drop=True).copy()

continuous_features = ["citi_bike_lat", "citi_bike_lon", "capacity", "meters_to_nearest_mta_station"]
cyclical_features = ["snapshot_hr", "snapshot_weekday"]

# Train/test split
train_size = int(len(model_data) * 0.8)
train_data = model_data.iloc[:train_size]
test_data = model_data.iloc[train_size:]

X_train = train_data[continuous_features + cyclical_features]
y_train = train_data["lda_30min"]
X_test = test_data[continuous_features + cyclical_features]
y_test = test_data["lda_30min"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cyclical", CyclicalFeatures(variables=cyclical_features, drop_original=True), cyclical_features),
        ("scaler", StandardScaler(), continuous_features)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

# Fit model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_df, y_train)

# Evaluate
y_pred = model.predict(X_test_df)
y_proba = model.predict_proba(X_test_df)[:, 1]

metrics = {
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "pr_auc": float(average_precision_score(y_test, y_proba))
}
print(f"Model Metrics: {metrics}")

# Save locally
model_path = MODELS_DIR / "model.joblib"
preprocessor_path = MODELS_DIR / "preprocessor.joblib"

print(f"Saving artifacts locally to {MODELS_DIR}...")
joblib.dump(model, model_path)
joblib.dump(preprocessor, preprocessor_path)

# AWS Integrations
def upload_artifacts_to_s3():
    s3 = boto3.client('s3')
    bucket = "jac6779-citibike-snapshots-2026"
    
    artifacts = {
        model_path: "citibike_snapshots/models/model.joblib",
        preprocessor_path: "citibike_snapshots/models/preprocessor.joblib"
    }

    print(f"Starting upload to S3 bucket: {bucket}")
    for local_file, s3_key in artifacts.items():
        try:
            s3.upload_file(str(local_file), bucket, s3_key)
            print(f"Successfully uploaded {local_file.name} to {s3_key}")
        except Exception as e:
            print(f"Upload failed for {local_file.name}: {e}")

def restart_app_runner():
    client = boto3.client('apprunner', region_name='us-east-1')
    SERVICE_ARN = "arn:aws:apprunner:us-east-1:528757830050:service/citi-bike-mlops-service/08e669d43e4a486b87e1ca77a7e93b97" 
    
    print("Notifying App Runner to redeploy and load the new model...")
    try:
        response = client.start_deployment(ServiceArn=SERVICE_ARN)
        operation_id = response.get('OperationId', 'Unknown ID')
        print(f"Deployment triggered successfully! Operation ID: {operation_id}")
    except Exception as e:
        print(f"Failed to trigger deployment: {e}")
        
        
def log_metrics_to_cloudwatch(metrics):
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    namespace = "CitiBike/MLOps/WeeklyTraining"
    
    metric_data = []
    for name, value in metrics.items():
        metric_data.append({
            'MetricName': name.upper(),
            'Dimensions': [
                {'Name': 'ModelName', 'Value': 'CitiBike-Prediction-RF'},
                {'Name': 'Environment', 'Value': 'Production'}
            ],
            'Timestamp': datetime.utcnow(),
            'Value': value,
            'Unit': 'None'
        })

    print(f"Publishing metrics to CloudWatch...")
    try:
        cloudwatch.put_metric_data(Namespace=namespace, MetricData=metric_data)
    except Exception as e:
        print(f"Failed to publish metrics: {e}")

# Single execution block
if __name__ == "__main__":
    log_metrics_to_cloudwatch(metrics)
    upload_artifacts_to_s3()
    restart_app_runner()
    print("Training, upload, and redeployment trigger complete.")