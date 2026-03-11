import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, timezone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ===============================
# CONFIG
# ===============================

LOCATION_NAME = "delhi"
DATA_FILE = "uv_dataset.json"
OUTPUT_FILE = "uv_prediction.json"

FORECAST_HOURS = 24
TRAIN_SPLIT = 0.8

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_json(DATA_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])
df.sort_values("datetime", inplace=True)

# ===============================
# DEFINE FEATURES + TARGETS
# ===============================

target_cols = [f"target_t+{i}" for i in range(1, FORECAST_HOURS + 1)]

feature_cols = [
    col for col in df.columns
    if col not in target_cols and col not in ["datetime"]
]

X = df[feature_cols]
y = df[target_cols]

# ===============================
# TRAIN / VALIDATION SPLIT (80/20 Time-Based)
# ===============================

split_index = int(len(df) * TRAIN_SPLIT)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

# ===============================
# MODEL
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

val_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE: {mae:.4f}")

# ===============================
# FORECAST NEXT 24 HOURS
# ===============================

latest_row = X.iloc[[-1]]

forecast_vector = model.predict(latest_row)[0]

# Clip negative UV values
forecast_vector = np.clip(forecast_vector, 0, 12)

# ===============================
# BUILD FORECAST OUTPUT
# ===============================

last_datetime = df["datetime"].iloc[-1]

forecast_output = []

for i in range(FORECAST_HOURS):
    forecast_time = last_datetime + timedelta(hours=i+1)
    forecast_output.append({
        "datetime": forecast_time.isoformat(),
        "uv_index": round(float(forecast_vector[i]), 3)
    })

# ===============================
# SAVE JSON (Rewrite Every Run)
# ===============================

with open(OUTPUT_FILE, "w") as f:
    json.dump(forecast_output, f, indent=4)

print(f"UV forecast saved as {OUTPUT_FILE}")
print("Next 24-hour UV forecast generated successfully.")