import pandas as pd
import numpy as np
import json
from datetime import timedelta
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

# ===============================
# CONFIG
# ===============================

LOCATION_NAME = "delhi"

DATA_FILE = "rainfall_dataset.json"
OUTPUT_FILE = "rainfall_prediction.json"

FORECAST_HOURS = 24
TRAIN_SPLIT = 0.8

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_json(DATA_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])
df.sort_values("datetime", inplace=True)

# ===============================
# TARGETS
# ===============================

target_cols = [f"target_t+{i}" for i in range(1, FORECAST_HOURS + 1)]

feature_cols = [
    c for c in df.columns
    if c not in target_cols and c != "datetime"
]

X = df[feature_cols]
y_reg = df[target_cols]

# ===============================
# BUILD RAIN OCCURRENCE TARGET
# ===============================

y_class = (y_reg > 0).astype(int)

# ===============================
# TRAIN / VALIDATION SPLIT
# ===============================

split_index = int(len(df) * TRAIN_SPLIT)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_reg_train = y_reg.iloc[:split_index]
y_reg_val = y_reg.iloc[split_index:]

y_class_train = y_class.iloc[:split_index]
y_class_val = y_class.iloc[split_index:]

# ===============================
# MODEL 1 : RAIN OCCURRENCE
# ===============================

rain_classifier = MultiOutputRegressor(
    XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
)

rain_classifier.fit(X_train, y_class_train)

# ===============================
# MODEL 2 : RAIN INTENSITY
# ===============================

rain_regressor = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
)

rain_regressor.fit(X_train, y_reg_train)

# ===============================
# VALIDATION
# ===============================

y_pred_val = rain_regressor.predict(X_val)

mae = mean_absolute_error(y_reg_val, y_pred_val)

print(f"Validation MAE: {mae:.4f}")

# ===============================
# FORECAST NEXT 24 HOURS
# ===============================

latest_row = X.iloc[[-1]]

rain_prob = rain_classifier.predict(latest_row)[0]
rain_amount = rain_regressor.predict(latest_row)[0]

forecast = []

for i in range(FORECAST_HOURS):

    if rain_prob[i] == 0:
        rainfall = 0
    else:
        rainfall = max(0, rain_amount[i])

    forecast.append(rainfall)

# ===============================
# BUILD FORECAST JSON
# ===============================

last_datetime = df["datetime"].iloc[-1]

forecast_output = []

for i in range(FORECAST_HOURS):

    forecast_time = last_datetime + timedelta(hours=i+1)

    forecast_output.append({
        "datetime": forecast_time.isoformat(),
        "rainfall": round(float(forecast[i]), 2),
        "unit": "mm"
    })

# ===============================
# SAVE FORECAST
# ===============================

with open(OUTPUT_FILE, "w") as f:
    json.dump(forecast_output, f, indent=4)

print(f"Rainfall forecast saved as {OUTPUT_FILE}")
print("Next 24-hour rainfall forecast generated successfully.")