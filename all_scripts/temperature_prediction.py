import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Blockchain pipeline import
from blockchain_pipeline import run_blockchain_pipeline

# ===============================
# CONFIG
# ===============================

LATITUDE = 28.6139
LONGITUDE = 77.2090
LOCATION_NAME = "delhi"

ROLLING_YEARS = 3
LAG_HOURS = 24
FORECAST_HOURS = 24

DATA_FILE = "temperature_dataset.json"
MODEL_FILE = f"{LOCATION_NAME}_model.pkl"
PREDICTION_FILE = "temperature_prediction.json"

# ===============================
# DATE WINDOW (Last 3 Years)
# ===============================

end_date = datetime.utcnow().date() - timedelta(days=1)
start_date = end_date - timedelta(days=ROLLING_YEARS * 365)

print(f"Fetching data from {start_date} to {end_date}")

# ===============================
# FETCH DATA FROM OPEN-METEO
# ===============================

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m&timezone=UTC"
)

response = requests.get(url)
data = response.json()

df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]),
    "temperature": data["hourly"]["temperature_2m"]
})

df.set_index("datetime", inplace=True)

# ===============================
# FEATURE ENGINEERING
# ===============================

for lag in range(1, LAG_HOURS + 1):
    df[f"lag_{lag}"] = df["temperature"].shift(lag)

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

df["season"] = df["month"].apply(get_season)

# ===============================
# VECTOR TARGET (Next 24 Hours)
# ===============================

for i in range(1, FORECAST_HOURS + 1):
    df[f"target_t+{i}"] = df["temperature"].shift(-i)

df.dropna(inplace=True)

# ===============================
# SAVE DATASET
# ===============================

df.to_json(DATA_FILE, orient="records")
print("Dataset saved.")

# ===============================
# TRAIN / VALIDATION SPLIT
# ===============================

feature_cols = [col for col in df.columns if "lag_" in col or col in ["hour", "day_of_week", "month", "season"]]
target_cols = [col for col in df.columns if "target_" in col]

X = df[feature_cols]
y = df[target_cols]

split_index = int(len(df) * 0.8)

X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ===============================
# MODEL TRAINING
# ===============================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# VALIDATION
# ===============================

val_predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, val_predictions)
r2 = r2_score(y_val, val_predictions)

print(f"Validation MAE: {mae:.3f}")
print(f"Validation R2 Score: {r2:.3f}")

# ===============================
# SAVE MODEL
# ===============================

joblib.dump(model, MODEL_FILE)
print("Model saved.")

# ===============================
# FORECAST NEXT 24 HOURS
# ===============================

latest_row = df.iloc[-1][feature_cols].values.reshape(1, -1)

next_day_prediction = model.predict(latest_row)[0]

forecast_times = [
    (df.index[-1] + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
    for i in range(1, FORECAST_HOURS + 1)
]

forecast_output = [
    {"datetime": forecast_times[i], "predicted_temperature": float(next_day_prediction[i])}
    for i in range(FORECAST_HOURS)
]

with open(PREDICTION_FILE, "w") as f:
    json.dump(forecast_output, f, indent=4)

print("Next day hourly forecast saved as JSON.")

# ===============================
# BLOCKCHAIN PIPELINE CALL
# ===============================

run_blockchain_pipeline(
    parameter="temperature",
    mae=float(mae),
    r2=float(r2)
)