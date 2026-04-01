from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from app.feature_builder import build_features

app = FastAPI(
    title="NYC Citi Bike Availability Predictor",
    description="Predict low Citi Bike dock availability in the next 30 minutes.",
    version="1.0.0"
)

model = joblib.load("models/citi_bike_random_forest_model.joblib")


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
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    X_processed = build_features(payload.model_dump())

    probability = model.predict_proba(X_processed)[0][1]

    return {
        "lda_30min_probability": round(float(probability), 3)
    }