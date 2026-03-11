import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import math
import sys

# ===============================
# CONFIG
# ===============================

LATITUDE = 28.6139
LONGITUDE = 77.2090
LOCATION_NAME = "delhi"

ROLLING_YEARS = 3
FORECAST_HOURS = 24

OUTPUT_FILE = "uv_dataset.json"

# ===============================
# DATE WINDOW (Last 3 Years From Yesterday)
# ===============================

end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
start_date = end_date - timedelta(days=ROLLING_YEARS * 365)

# NASA POWER requires YYYYMMDD format
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

print(f"Fetching NASA POWER radiation data from {start_date} to {end_date}")

# ===============================
# FETCH DATA FROM NASA POWER
# ===============================

url = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point?"
    f"parameters=ALLSKY_SFC_SW_DWN"
    f"&community=RE"
    f"&longitude={LONGITUDE}"
    f"&latitude={LATITUDE}"
    f"&start={start_str}"
    f"&end={end_str}"
    f"&format=JSON"
)

response = requests.get(url)

if response.status_code != 200:
    print("NASA POWER API request failed")
    sys.exit()

data = response.json()

try:
    radiation_data = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
except KeyError:
    print("Radiation data not found in NASA response")
    sys.exit()

# ===============================
# CREATE DATAFRAME
# ===============================

df = pd.DataFrame(
    radiation_data.items(),
    columns=["datetime_str", "shortwave_radiation"]
)

df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d%H")
df["shortwave_radiation"] = pd.to_numeric(df["shortwave_radiation"], errors="coerce")

df.set_index("datetime", inplace=True)
df = df.sort_index()

# Fill missing radiation with 0
df["shortwave_radiation"] = df["shortwave_radiation"].fillna(0)

print("Rows fetched:", len(df))

# ===============================
# UV PROXY CREATION
# ===============================

# Empirical conversion factor
df["uv_index"] = df["shortwave_radiation"] * 0.0025

# Clip to realistic UV scale
df["uv_index"] = df["uv_index"].clip(0, 12)

# ===============================
# FEATURE ENGINEERING
# ===============================

# ---- Lag features
for lag in range(1, 7):
    df[f"lag_{lag}"] = df["uv_index"].shift(lag)

df["lag_24"] = df["uv_index"].shift(24)

# ---- Time features
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["day_of_year"] = df.index.dayofyear

# ---- Season encoding
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

# ---- Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ===============================
# SOLAR ANGLE CALCULATION (Vectorized)
# ===============================

lat_rad = np.radians(LATITUDE)
day = df["day_of_year"].values
hour = df["hour"].values

decl = 23.44 * np.cos(np.radians((360 / 365) * (day - 81)))
decl_rad = np.radians(decl)

hour_angle = np.radians((hour - 12) * 15)

elevation = np.arcsin(
    np.sin(lat_rad) * np.sin(decl_rad) +
    np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle)
)

elevation_deg = np.degrees(elevation)
zenith_deg = 90 - elevation_deg

df["solar_elevation"] = np.clip(elevation_deg, 0, None)
df["solar_zenith"] = zenith_deg

# ===============================
# VECTOR TARGET (Next 24 Hours)
# ===============================

for i in range(1, FORECAST_HOURS + 1):
    df[f"target_t+{i}"] = df["uv_index"].shift(-i)

df.dropna(inplace=True)

print("Rows after feature engineering:", len(df))

if len(df) == 0:
    print("Dataset became empty after feature engineering")
    sys.exit()

# ===============================
# SAVE JSON
# ===============================

df.reset_index(inplace=True)
df.to_json(OUTPUT_FILE, orient="records", indent=4)

print(f"UV dataset saved as {OUTPUT_FILE}")
print("Final dataset rows:", len(df))