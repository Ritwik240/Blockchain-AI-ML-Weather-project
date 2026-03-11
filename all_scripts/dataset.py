import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import math
import sys
import json

# ===============================
# CONFIG
# ===============================
LATITUDE = 28.6139
LONGITUDE = 77.2090
LOCATION_NAME = "delhi"

ROLLING_YEARS = 3
FORECAST_HOURS = 24

OUTPUT_FILE = "windspeed_dataset.json"

# ===============================
# DATE WINDOW
# ===============================
end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
start_date = end_date - timedelta(days=ROLLING_YEARS * 365)

print(f"Fetching windspeed data from {start_date} to {end_date}")

# ===============================
# FETCH DATA FROM OPEN-METEO
# ===============================
url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=windspeed_10m&timezone=UTC"
)

response = requests.get(url)
if response.status_code != 200:
    print("Open-Meteo API request failed")
    sys.exit()

data = response.json()

# Check data availability
if "hourly" not in data or "windspeed_10m" not in data["hourly"]:
    print("Windspeed data not available for this period")
    sys.exit()

df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"]),
    "windspeed": pd.to_numeric(data["hourly"]["windspeed_10m"], errors="coerce")
})

df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

# Fill missing values with 0 (or interpolate if desired)
df["windspeed"] = df["windspeed"].fillna(0)

print("Rows fetched:", len(df))

# ===============================
# FEATURE ENGINEERING
# ===============================

# ---- Lag features
for lag in range(1, 7):
    df[f"lag_{lag}"] = df["windspeed"].shift(lag)

df["lag_24"] = df["windspeed"].shift(24)

# ---- Time features
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["day_of_year"] = df.index.dayofyear

# ---- Season encoding
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # winter
    elif month in [3, 4, 5]:
        return 1  # spring
    elif month in [6, 7, 8]:
        return 2  # summer
    else:
        return 3  # autumn

df["season"] = df["month"].apply(get_season)

# ---- Cyclical hour encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ===============================
# SOLAR ANGLE CALCULATION (Optional)
# ===============================
lat_rad = math.radians(LATITUDE)

def solar_features(row):
    day = row["day_of_year"]
    hour = row["hour"]
    decl = 23.44 * math.cos(math.radians((360 / 365) * (day - 81)))
    decl_rad = math.radians(decl)
    hour_angle = math.radians((hour - 12) * 15)
    elevation = math.asin(
        math.sin(lat_rad) * math.sin(decl_rad) +
        math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle)
    )
    elevation_deg = math.degrees(elevation)
    zenith_deg = 90 - elevation_deg
    return pd.Series([max(elevation_deg, 0), zenith_deg])

df[["solar_elevation", "solar_zenith"]] = df.apply(solar_features, axis=1)

# ===============================
# VECTOR TARGET (Next 24 Hours)
# ===============================
for i in range(1, FORECAST_HOURS + 1):
    df[f"target_t+{i}"] = df["windspeed"].shift(-i)

df.dropna(inplace=True)

print("Rows after feature engineering:", len(df))

if len(df) == 0:
    print("Dataset became empty after feature engineering")
    sys.exit()

# ===============================
# SAVE DATASET
# ===============================
df.reset_index(inplace=True)
df.to_json(OUTPUT_FILE, orient="records", indent=4)

print(f"Windspeed dataset saved as {OUTPUT_FILE}")