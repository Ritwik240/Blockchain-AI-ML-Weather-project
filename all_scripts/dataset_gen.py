import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# ===============================
# CONFIG
# ===============================

LATITUDE = 28.6139
LONGITUDE = 77.2090
LOCATION_NAME = "delhi"

ROLLING_YEARS = 3
FORECAST_HOURS = 24

OUTPUT_FILE = "rainfall_dataset.json"

# ===============================
# DATE WINDOW
# ===============================

end_date = datetime.utcnow().date() - timedelta(days=1)
start_date = end_date - timedelta(days=ROLLING_YEARS * 365)

print(f"Fetching rainfall data from {start_date} to {end_date}")

# ===============================
# FETCH DATA
# ===============================

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=precipitation"
    "&timezone=UTC"
)

response = requests.get(url)
data = response.json()

df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]),
    "rainfall": data["hourly"]["precipitation"]
})

df.set_index("datetime", inplace=True)

print("Rows fetched:", len(df))

# ===============================
# CLEAN DATA
# ===============================

df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
df["rainfall"].fillna(0, inplace=True)

# rainfall cannot be negative
df["rainfall"] = df["rainfall"].clip(lower=0)

# ===============================
# FEATURE ENGINEERING
# ===============================

# ---- Lag features
for lag in range(1, 7):
    df[f"lag_{lag}"] = df["rainfall"].shift(lag)

df["lag_24"] = df["rainfall"].shift(24)

# ---- Rolling features
df["rolling_6h"] = df["rainfall"].rolling(6).mean()
df["rolling_24h"] = df["rainfall"].rolling(24).mean()

# ===============================
# TIME FEATURES
# ===============================

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["day_of_year"] = df.index.dayofyear

# ---- season encoding
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

# ---- cyclical hour encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ===============================
# VECTOR TARGET (Next 24 Hours)
# ===============================

for i in range(1, FORECAST_HOURS + 1):
    df[f"target_t+{i}"] = df["rainfall"].shift(-i)

# ===============================
# FINAL CLEANING
# ===============================

df.dropna(inplace=True)

print("Rows after feature engineering:", len(df))

# ===============================
# SAVE DATASET
# ===============================

df.reset_index(inplace=True)

df.to_json(OUTPUT_FILE, orient="records", indent=4)

print(f"Rainfall dataset saved as {OUTPUT_FILE}")