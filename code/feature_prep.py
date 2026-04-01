import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "latitude",
    "longitude",
    "agency",
    "complaint_type",
    "borough",
    "location_type",
    "complaint_hr",
    "complaint_day",
]

def prepare_model_input(df: pd.DataFrame, zip_lookup: pd.DataFrame) -> pd.DataFrame:

    X = df.copy()

    if "incident_zip" in X.columns:
        X["incident_zip"] = X["incident_zip"].astype(str).str.strip()

        X = X.merge(zip_lookup, on="incident_zip", how="left", suffixes=("", "_zip"))

        if "latitude_zip" in X.columns:
            X["latitude"] = X["latitude"].fillna(X["latitude_zip"])
            X = X.drop(columns=["latitude_zip"])

        if "longitude_zip" in X.columns:
            X["longitude"] = X["longitude"].fillna(X["longitude_zip"])
            X = X.drop(columns=["longitude_zip"])

    if "created_date" in X.columns:
        created_dt = pd.to_datetime(X["created_date"], errors="coerce")

        if "complaint_hr" not in X.columns:
            X["complaint_hr"] = created_dt.dt.hour

        if "complaint_day" not in X.columns:
            X["complaint_day"] = created_dt.dt.dayofweek

    for col in ["agency", "complaint_type", "borough", "location_type"]:
        if col not in X.columns:
            X[col] = "unknown"

        X[col] = X[col].fillna("unknown").astype(str).str.strip()
        X.loc[X[col].eq("") | X[col].eq("nan"), col] = "unknown"

    for col in ["latitude", "longitude"]:
        if col not in X.columns:
            X[col] = np.nan

        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in ["complaint_hr", "complaint_day"]:
        if col not in X.columns:
            X[col] = -1

        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype(int)

    return X[MODEL_FEATURES].copy()