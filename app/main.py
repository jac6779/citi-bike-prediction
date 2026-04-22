from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import boto3
import os
from app.feature_builder import build_features

# Configuration
BUCKET_NAME = "jac6779-citibike-snapshots-2026" 
MODEL_KEY = "citibike_snapshots/models/model.joblib"
PREPROCESSOR_KEY = "citibike_snapshots/models/preprocessor.joblib"

LOCAL_MODEL_PATH = "/tmp/model.joblib"
LOCAL_PREPROCESSOR_PATH = "/tmp/preprocessor.joblib"

app = FastAPI(
    title="Citi Bike Low Dock Availability Predictor",
    description="Predict low Citi Bike dock availability in the next 30 minutes.",
    version="1.0.0"
)

def load_artifacts_from_s3():
    global model_s3_key

    s3 = boto3.client("s3")
    try:
        print(f"Downloading model from s3://{BUCKET_NAME}/{MODEL_KEY}")
        s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
        s3.download_file(BUCKET_NAME, PREPROCESSOR_KEY, LOCAL_PREPROCESSOR_PATH)

        model_s3_key = MODEL_KEY

        loaded_model = joblib.load(LOCAL_MODEL_PATH)
        loaded_preprocessor = joblib.load(LOCAL_PREPROCESSOR_PATH)

        return loaded_model, loaded_preprocessor

    except Exception as e:
        print(f"CRITICAL ERROR loading artifacts from S3: {e}")
        raise RuntimeError("Model artifacts could not be loaded from S3")

# Initialize both globally
model = None
preprocessor = None
model_s3_key = MODEL_KEY

@app.on_event("startup")
def startup_event():
    global model, preprocessor
    model, preprocessor = load_artifacts_from_s3()


class PredictionRequest(BaseModel):
    citi_bike_lat: float = Field(..., description="Latitude of Citi Bike station")
    citi_bike_lon: float = Field(..., description="Longitude of Citi Bike station")
    capacity: float = Field(..., description="Total number of docks at station")
    meters_to_nearest_mta_station: float = Field(..., description="Distance in meters to nearest MTA station")
    snapshot_hr: int = Field(..., description="Hour of day when the station snapshot was recorded")
    snapshot_weekday: int = Field(..., description="Weekday number when the station snapshot was recorded")

    model_config = {
        "json_schema_extra": {
            "example": {
                "citi_bike_lat": 40.764,
                "citi_bike_lon": -73.910,
                "capacity": 18,
                "meters_to_nearest_mta_station": 893,
                "snapshot_hr": 9,
                "snapshot_weekday": 6
            }
        }
    }


@app.get("/health")
def health():
    return {
    "status": "ok",
    "model_loaded": model is not None,
    "model_s3_key": model_s3_key
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        input_dict = payload.model_dump()   # Pydantic v2 style
        processed_features = build_features(input_dict, preprocessor)

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(processed_features)[0, 1])
        else:
            prob = float(model.predict(processed_features)[0])

        return {
            "Probability of lock dock availability within next 30 minutes": round(prob, 3),
            "model_s3_key": model_s3_key
        }
    except Exception as e:
        import traceback
        print("PREDICT ERROR:", str(e))
        traceback.print_exc()
        raise