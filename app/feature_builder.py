import joblib
import pandas as pd

MODEL_FEATURES = [
    "citi_bike_lat",
    "citi_bike_lon",
    "capacity",
    "meters_to_nearest_mta_station",
    "snapshot_hr",
    "snapshot_weekday"
]

preprocessor = joblib.load("models/preprocessor.joblib")


def build_features(input_data: dict):
    df = pd.DataFrame([input_data])

    # enforce the same raw feature set and order used in training
    df = df[MODEL_FEATURES]

    # apply the fitted preprocessor from training
    X_processed = preprocessor.transform(df)

    return X_processed