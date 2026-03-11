import pandas as pd
import numpy as np
import json
from datetime import timedelta
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ===============================
# CONFIG
# ===============================
LOCATION_NAME = "delhi"
DATA_FILE = "windspeed_dataset.json"
OUTPUT_FILE = "windspeed_prediction.json"

FORECAST_HOURS = 24
TRAIN_SPLIT = 0.8

# Conversion: m/s → km/h
MPS_TO_KMH = 3.6

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_json(DATA_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
df.sort_values("datetime", inplace=True)

# Convert windspeed to km/h
df["windspeed"] = df["windspeed"] * MPS_TO_KMH

# ===============================
# DEFINE FEATURES + TARGETS
# ===============================
target_cols = [f"target_t+{i}" for i in range(1, FORECAST_HOURS + 1)]
feature_cols = [col for col in df.columns if col not in target_cols and col not in ["datetime"]]

X = df[feature_cols]
y = df[target_cols] * MPS_TO_KMH  # also convert targets to km/h

# ===============================
# TRAIN / VALIDATION SPLIT (80/20 Time-Based)
# ===============================
split_index = int(len(df) * TRAIN_SPLIT)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

# ===============================
# MULTI-OUTPUT XGBOOST MODEL
# ===============================
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
)

model.fit(X_train, y_train)

# ===============================
# VALIDATION METRICS
# ===============================
y_pred_val = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred_val)
print(f"Validation MAE: {mae:.4f} km/h")

# ===============================
# FORECAST NEXT 24 HOURS
# ===============================
latest_row = X.iloc[[-1]]  # last known hour
forecast_vector = model.predict(latest_row)[0]

# Clip negative values
forecast_vector = np.clip(forecast_vector, 0, None)

# ===============================
# BUILD FORECAST JSON
# ===============================
last_datetime = df["datetime"].iloc[-1]
forecast_output = []

for i in range(FORECAST_HOURS):
    forecast_time = last_datetime + timedelta(hours=i+1)
    forecast_output.append({
        "datetime": forecast_time.isoformat(),
        "windspeed": round(float(forecast_vector[i]), 2),
        "unit": "km/h"
    })

# ===============================
# SAVE JSON (Rewrite Every Run)
# ===============================
with open(OUTPUT_FILE, "w") as f:
    json.dump(forecast_output, f, indent=4)

print(f"Windspeed forecast saved as {OUTPUT_FILE}")
print("Next 24-hour windspeed forecast generated successfully (km/h).")