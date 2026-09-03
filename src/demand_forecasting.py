"""
Demand Forecasting Pipeline
E-Commerce Delivery Operations Intelligence

Forecasts daily order demand using time-series features and
machine learning.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "amazon_delivery_final.csv"
)

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DEMAND FORECASTING PIPELINE")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    parse_dates=["Order_Date"],
)

print(f"\nLoaded orders: {len(df):,}")


# ============================================================
# CREATE DAILY DEMAND DATASET
# ============================================================

daily = (
    df.groupby("Order_Date")
    .size()
    .reset_index(name="Demand")
    .sort_values("Order_Date")
    .reset_index(drop=True)
)

print(f"Daily observations: {len(daily)}")


# ============================================================
# TIME FEATURES
# ============================================================

daily["Year"] = daily["Order_Date"].dt.year
daily["Month"] = daily["Order_Date"].dt.month
daily["Day"] = daily["Order_Date"].dt.day
daily["DayOfWeek"] = daily["Order_Date"].dt.dayofweek
daily["Quarter"] = daily["Order_Date"].dt.quarter

daily["Is_Weekend"] = (
    daily["DayOfWeek"] >= 5
).astype(int)


# ============================================================
# LAG FEATURES
# ============================================================

daily["Lag_1"] = daily["Demand"].shift(1)
daily["Lag_7"] = daily["Demand"].shift(7)

daily["Rolling_Mean_7"] = (
    daily["Demand"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily["Rolling_Mean_30"] = (
    daily["Demand"]
    .shift(1)
    .rolling(30)
    .mean()
)

daily = daily.dropna().reset_index(drop=True)

print(f"Observations after lag creation: {len(daily)}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "Is_Weekend",
    "Lag_1",
    "Lag_7",
    "Rolling_Mean_7",
    "Rolling_Mean_30",
]

TARGET = "Demand"

X = daily[FEATURES]
y = daily[TARGET]


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(daily) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = daily["Order_Date"].iloc[split_index:]

print(f"\nTraining observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions,
) ** 0.5

r2 = r2_score(
    y_test,
    predictions,
)

mape = (
    (
        abs(
            (y_test - predictions)
            / y_test
        )
    ).mean()
    * 100
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.2f} orders")
print(f"RMSE : {rmse:.2f} orders")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = (
    pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        }
    )
    .sort_values(
        "Importance",
        ascending=False,
    )
)

print("\nFeature Importance:")
print(importance.to_string(index=False))


# ============================================================
# SAVE FORECAST RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "Order_Date": test_dates.values,
        "Actual_Demand": y_test.values,
        "Predicted_Demand": predictions.round().astype(int),
    }
)

results.to_csv(
    MODEL_OUTPUT_DIR / "demand_forecast_results.csv",
    index=False,
)

print("\nSaved forecast results:")
print(MODEL_OUTPUT_DIR / "demand_forecast_results.csv")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("DEMAND FORECASTING COMPLETED")
print("=" * 60)