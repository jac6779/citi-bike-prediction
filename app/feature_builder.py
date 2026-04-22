import pandas as pd

MODEL_FEATURES = [
    "citi_bike_lat",
    "citi_bike_lon",
    "capacity",
    "meters_to_nearest_mta_station",
    "snapshot_hr",
    "snapshot_weekday"
]


def build_features(input_data: dict, preprocessor):
    df = pd.DataFrame([input_data])
    df = df[MODEL_FEATURES]

    # apply the fitted preprocessor from training
    X_processed = preprocessor.transform(df)
    return X_processed