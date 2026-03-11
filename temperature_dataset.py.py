import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
CITY = "Delhi"
LATITUDE = 28.6139
LONGITUDE = 77.2090
YEARS_BACK = 3
OUTPUT_FILE = "temperature_dataset.json"

# =========================
# DATE WINDOW (Sliding 3 Years)
# =========================
yesterday = datetime.utcnow() - timedelta(days=1)
start_date = yesterday - timedelta(days=365 * YEARS_BACK)

START_DATE = start_date.strftime("%Y-%m-%d")
END_DATE = yesterday.strftime("%Y-%m-%d")

# =========================
# FETCH DATA
# =========================
def fetch_temperature_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"]
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# =========================
# FEATURE ENGINEERING
# =========================
def add_time_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)

    return df

# =========================
# LAG FEATURES
# =========================
def add_lag_features(df):
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    for lag in lags:
        df[f"lag_{lag}"] = df["temperature"].shift(lag)
    return df

# =========================
# ROLLING FEATURES
# =========================
def add_rolling_features(df):
    windows = [6, 12, 24, 72]
    for w in windows:
        df[f"rolling_mean_{w}"] = df["temperature"].rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df["temperature"].rolling(window=w).std()
    return df

# =========================
# VECTOR TARGETS (Next 24 Hours)
# =========================
def create_vector_targets(df):
    for i in range(1, 25):
        df[f"target_t+{i}"] = df["temperature"].shift(-i)
    return df

# =========================
# CONVERT TO JSON STRUCTURE
# =========================
def dataframe_to_json(df):
    dataset = {
        "metadata": {
            "city": CITY,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "years_back": YEARS_BACK,
            "granularity": "hourly",
            "parameter": "temperature"
        },
        "data": []
    }

    feature_columns = [
        col for col in df.columns
        if not col.startswith("target_") and col != "temperature"
    ]

    target_columns = [
        col for col in df.columns if col.startswith("target_")
    ]

    for _, row in df.iterrows():
        entry = {
            "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "features": {col: row[col] for col in feature_columns if col != "datetime"},
            "targets": {col: row[col] for col in target_columns}
        }
        dataset["data"].append(entry)

    return dataset

# =========================
# MAIN PIPELINE
# =========================
def build_dataset():
    print("Fetching data...")
    df = fetch_temperature_data(LATITUDE, LONGITUDE, START_DATE, END_DATE)

    print("Adding features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = create_vector_targets(df)

    print("Cleaning dataset...")
    df = df.dropna().reset_index(drop=True)

    print("Converting to JSON...")
    dataset_json = dataframe_to_json(df)

    print(f"Writing to {OUTPUT_FILE} (overwriting if exists)...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print("Dataset build complete ✅")
    print("Total samples:", len(dataset_json["data"]))

if __name__ == "__main__":
    build_dataset()