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
LAG_HOURS = 24
FORECAST_HOURS = 24

OUTPUT_FILE = "humidity_dataset.json"

# ===============================
# DATE WINDOW (Last 3 Years)
# ===============================

end_date = datetime.utcnow().date() - timedelta(days=1)
start_date = end_date - timedelta(days=ROLLING_YEARS * 365)

print(f"Fetching humidity data from {start_date} to {end_date}")

# ===============================
# FETCH DATA FROM OPEN-METEO
# ===============================

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=relativehumidity_2m&timezone=UTC"
)

response = requests.get(url)
data = response.json()

df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]),
    "humidity": data["hourly"]["relativehumidity_2m"]
})

df.set_index("datetime", inplace=True)

# ===============================
# FEATURE ENGINEERING
# ===============================

# Lag features
for lag in range(1, LAG_HOURS + 1):
    df[f"lag_{lag}"] = df["humidity"].shift(lag)

# Time features
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn

df["season"] = df["month"].apply(get_season)

# ===============================
# VECTOR TARGET (Next 24 Hours)
# ===============================

for i in range(1, FORECAST_HOURS + 1):
    df[f"target_t+{i}"] = df["humidity"].shift(-i)

df.dropna(inplace=True)

# ===============================
# SAVE DATASET (OVERWRITE EACH RUN)
# ===============================

df.to_json(OUTPUT_FILE, orient="records", indent=4)

print(f"Humidity dataset saved as {OUTPUT_FILE}")