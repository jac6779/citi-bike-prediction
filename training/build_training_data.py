import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path
import requests
import json
import boto3
from io import BytesIO
from datetime import datetime, timedelta, timezone


# File paths - Set to work relative to the script location
PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean_data"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_NAME = "jac6779-citibike-snapshots-2026"

def download_static_files():
    """Downloads the MTA data from your new S3 archive path."""
    s3 = boto3.client('s3')
    # Updated to match your new path
    mta_s3_key = "citibike_snapshots/archive/00_mta_subway_stations.parquet"
    local_mta_path = RAW_DATA_DIR / "00_mta_subway_stations.parquet"
    
    print(f"Downloading MTA station data from s3://{BUCKET_NAME}/{mta_s3_key}...")
    s3.download_file(BUCKET_NAME, mta_s3_key, str(local_mta_path))
    return local_mta_path

def collect_snapshots_from_s3():
    """Sweeps the S3 bucket for JSON snapshots from the last 5 days."""
    s3 = boto3.client('s3')
    prefix = "citibike_snapshots/2026/"
    print(f"Scanning s3://{BUCKET_NAME}/{prefix} for snapshots from the last 5 days...")
    
    # Calculate the cutoff time (5 days ago from now)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=5)
    
    snapshots_dfs = []
    paginator = s3.get_paginator('list_objects_v2')
    
    files_found = 0
    files_skipped = 0

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get('Contents', []):
            # Check if file is JSON AND if it was modified within the last 5 days
            if obj['Key'].endswith('.json'):
                if obj['LastModified'] >= cutoff_date:
                    response = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                    snapshot = json.loads(response['Body'].read().decode('utf-8'))
                    
                    df = pd.DataFrame(snapshot["station_status"]["data"]["stations"])
                    df["snapshot_time_utc"] = snapshot["snapshot_time_utc"]
                    snapshots_dfs.append(df)
                    files_found += 1
                else:
                    files_skipped += 1
    
    print(f"Process complete. Loaded {files_found} recent files. Skipped {files_skipped} older files.")
    
    if not snapshots_dfs:
        raise ValueError(f"No snapshots from the last 5 days found in s3://{BUCKET_NAME}/{prefix}")
        
    return pd.concat(snapshots_dfs, ignore_index=True)


def upload_clean_parquet_to_s3(local_path):
    """Uploads the final processed dataset back to S3 for the training script."""
    s3 = boto3.client('s3')
    s3_key = "citibike_snapshots/data/clean/01_citi_bike_prediction_preprocessing.parquet"
    print(f"Uploading cleaned training data to s3://{BUCKET_NAME}/{s3_key}...")
    s3.upload_file(str(local_path), BUCKET_NAME, s3_key)

# --- EXECUTION FLOW ---

if __name__ == "__main__":
    # 1. Get Static MTA Data
    mta_path = download_static_files()
    mta = pd.read_parquet(mta_path)

    # 2. Get Live Station Information (for lat/lon/capacity)
    print("Fetching live station information from Lyft GBFS API...")
    station_info_url = "https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_information.json"
    response = requests.get(station_info_url, timeout=10)
    station_info = pd.DataFrame(response.json()["data"]["stations"])
    
    # Filter for NYC (Region 71) and clean columns
    station_info = station_info[station_info["region_id"] == "71"].copy()
    station_info = station_info.rename(columns={
        "name": "citi_bike_station_name",
        "lat": "citi_bike_lat",
        "lon": "citi_bike_lon"
    })

    # 3. Collect Snapshots from S3
    station_status_full_df = collect_snapshots_from_s3()

    # 4. Feature Engineering & Spatial Join
    print("Processing features and calculating MTA proximity...")
    station_status = station_status_full_df[["station_id", "num_bikes_available", "num_docks_available", "snapshot_time_utc"]].copy()
    station_status["snapshot_time"] = pd.to_datetime(station_status["snapshot_time_utc"], utc=True).dt.floor("s").dt.tz_localize(None)
    
    # Haversine distance to nearest MTA station
    citi_bike_radians = np.radians(station_info[["citi_bike_lat", "citi_bike_lon"]])
    mta_reduced = mta.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))[["stop_name", "gtfs_latitude", "gtfs_longitude"]]
    mta_radians = np.radians(mta_reduced[["gtfs_latitude", "gtfs_longitude"]])
    mta_tree = BallTree(mta_radians, metric="haversine")
    mta_distances, _ = mta_tree.query(citi_bike_radians, k=1)
    station_info["meters_to_nearest_mta_station"] = mta_distances.flatten() * 6371000

    # Merge status with info
    merged = station_status.merge(station_info, on="station_id", how="left")
    merged = merged.dropna(subset=["capacity"]).sort_values(by=["station_id", "snapshot_time"])

    # Create target: Low availability (<=10%) in ~30 minutes
    merged["snapshot_hr"] = merged["snapshot_time"].dt.hour
    merged["snapshot_weekday"] = merged["snapshot_time"].dt.dayofweek
    merged["future_snapshot_time"] = merged.groupby("station_id")["snapshot_time"].shift(-1)
    merged["future_dock_pct"] = (merged.groupby("station_id")["num_docks_available"].shift(-1) / merged["capacity"])
    merged["minutes_to_future"] = (merged["future_snapshot_time"] - merged["snapshot_time"]).dt.total_seconds() / 60

    final_df = merged[(merged["minutes_to_future"] >= 25) & (merged["minutes_to_future"] <= 35)].copy()
    final_df["lda_30min"] = (final_df["future_dock_pct"] <= 0.10).astype(int)

    # 5. Export and Ship
    output_path = CLEAN_DATA_DIR / "01_citi_bike_prediction_preprocessing.parquet"
    final_df.to_parquet(output_path, index=False)
    
    upload_clean_parquet_to_s3(output_path)
    print("Weekly data build complete.")