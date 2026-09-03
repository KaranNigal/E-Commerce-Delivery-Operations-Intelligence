"""
Delivery Capacity Planning Engine

Uses demand forecasting results to estimate the operational
capacity required for future delivery demand.

This is designed for the synthetic demonstration dataset generated
for the E-Commerce Delivery Operations Intelligence project.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = BASE_DIR / "models" / "demand_forecast_results.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_OUTPUT_FILE = OUTPUT_DIR / "capacity_plan.csv"


# ============================================================
# CAPACITY ASSUMPTIONS
# ============================================================

# Average number of deliveries one delivery agent can complete
# during one operational day.
DELIVERIES_PER_AGENT_PER_DAY = 25

# Buffer added to handle uncertainty, demand spikes,
# absenteeism, traffic disruptions, etc.
SAFETY_BUFFER = 0.15

# Minimum capacity utilization target.
# We avoid planning exactly at 100% utilization.
TARGET_UTILIZATION = 0.85


# ============================================================
# HELPER FUNCTION: DETECT COLUMN
# ============================================================

def find_column(columns, possible_names):
    """
    Finds a column from a list of possible names.
    """

    lower_map = {
        column.lower().replace("_", "").replace(" ", ""): column
        for column in columns
    }

    for name in possible_names:
        normalized = name.lower().replace("_", "").replace(" ", "")

        if normalized in lower_map:
            return lower_map[normalized]

    return None


# ============================================================
# LOAD FORECAST RESULTS
# ============================================================

print("=" * 60)
print("DELIVERY CAPACITY PLANNING ENGINE")
print("=" * 60)

print("\nLoading demand forecast...")

if not FORECAST_FILE.exists():
    raise FileNotFoundError(
        f"\nForecast file not found:\n{FORECAST_FILE}\n\n"
        "Run the demand forecasting pipeline first."
    )

forecast = pd.read_csv(FORECAST_FILE)

print(f"Loaded forecast records: {len(forecast):,}")

print("\nAvailable columns:")
print(list(forecast.columns))


# ============================================================
# DETECT DATE COLUMN
# ============================================================

DATE_COLUMN = find_column(
    forecast.columns,
    [
        "Date",
        "Order_Date",
        "OrderDate",
        "date",
    ]
)

if DATE_COLUMN is None:

    # Try finding a datetime-like column automatically
    for column in forecast.columns:

        try:
            converted = pd.to_datetime(
                forecast[column],
                errors="coerce"
            )

            if converted.notna().sum() > len(forecast) * 0.8:
                DATE_COLUMN = column
                break

        except Exception:
            pass


if DATE_COLUMN is None:
    raise ValueError(
        "\nCould not detect the date column in the forecast file.\n"
        f"Available columns: {list(forecast.columns)}"
    )


print(f"\nDetected date column: {DATE_COLUMN}")

forecast[DATE_COLUMN] = pd.to_datetime(
    forecast[DATE_COLUMN],
    errors="coerce"
)

forecast = forecast.dropna(subset=[DATE_COLUMN])


# ============================================================
# DETECT ACTUAL DEMAND COLUMN
# ============================================================

ACTUAL_COLUMN = find_column(
    forecast.columns,
    [
        "Actual",
        "Actual_Demand",
        "ActualDemand",
        "Demand",
        "True_Demand",
        "y_test",
    ]
)


# ============================================================
# DETECT FORECAST COLUMN
# ============================================================

FORECAST_COLUMN = find_column(
    forecast.columns,
    [
        "Forecast",
        "Predicted",
        "Prediction",
        "Predicted_Demand",
        "Forecast_Demand",
        "PredictedDemand",
        "y_pred",
    ]
)


if FORECAST_COLUMN is None:

    # Find numeric columns that might represent predictions
    numeric_columns = forecast.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("\nNumeric columns found:")
    print(numeric_columns)

    # Try selecting a likely forecast column
    possible_forecast_columns = [
        column
        for column in numeric_columns
        if any(
            keyword in column.lower()
            for keyword in [
                "forecast",
                "predict",
                "prediction",
                "pred"
            ]
        )
    ]

    if possible_forecast_columns:
        FORECAST_COLUMN = possible_forecast_columns[0]


if FORECAST_COLUMN is None:
    raise ValueError(
        "\nCould not detect the forecast column.\n"
        f"Available columns: {list(forecast.columns)}"
    )


print(f"Detected forecast column: {FORECAST_COLUMN}")

if ACTUAL_COLUMN is not None:
    print(f"Detected actual demand column: {ACTUAL_COLUMN}")
else:
    print("Actual demand column not detected.")


# ============================================================
# PREPARE CAPACITY DATA
# ============================================================

capacity = pd.DataFrame()

capacity["Date"] = forecast[DATE_COLUMN]

if ACTUAL_COLUMN is not None:
    capacity["Actual_Demand"] = pd.to_numeric(
        forecast[ACTUAL_COLUMN],
        errors="coerce"
    )

capacity["Forecast_Demand"] = pd.to_numeric(
    forecast[FORECAST_COLUMN],
    errors="coerce"
)

capacity = capacity.dropna(
    subset=["Forecast_Demand"]
)

capacity = capacity.sort_values("Date").reset_index(drop=True)


# ============================================================
# CLEAN FORECAST VALUES
# ============================================================

capacity["Forecast_Demand"] = (
    capacity["Forecast_Demand"]
    .clip(lower=0)
    .round()
    .astype(int)
)


# ============================================================
# CALCULATE REQUIRED DELIVERY CAPACITY
# ============================================================

print("\nCalculating capacity requirements...")

# Base agents required
capacity["Base_Agents_Required"] = np.ceil(
    capacity["Forecast_Demand"]
    / DELIVERIES_PER_AGENT_PER_DAY
).astype(int)


# Add safety buffer
capacity["Agents_With_Safety_Buffer"] = np.ceil(
    capacity["Base_Agents_Required"]
    * (1 + SAFETY_BUFFER)
).astype(int)


# Adjust for target utilization
capacity["Recommended_Agents"] = np.ceil(
    capacity["Forecast_Demand"]
    / (
        DELIVERIES_PER_AGENT_PER_DAY
        * TARGET_UTILIZATION
    )
).astype(int)


# Recommended delivery capacity
capacity["Planned_Delivery_Capacity"] = (
    capacity["Recommended_Agents"]
    * DELIVERIES_PER_AGENT_PER_DAY
)


# Capacity surplus
capacity["Capacity_Surplus"] = (
    capacity["Planned_Delivery_Capacity"]
    - capacity["Forecast_Demand"]
)


# ============================================================
# DEMAND RISK CLASSIFICATION
# ============================================================

forecast_mean = capacity["Forecast_Demand"].mean()
forecast_std = capacity["Forecast_Demand"].std()

high_threshold = forecast_mean + forecast_std
medium_threshold = forecast_mean


def classify_demand_risk(demand):

    if demand >= high_threshold:
        return "High"

    elif demand >= medium_threshold:
        return "Medium"

    return "Normal"


capacity["Demand_Risk"] = capacity[
    "Forecast_Demand"
].apply(classify_demand_risk)


# ============================================================
# CAPACITY UTILIZATION
# ============================================================

capacity["Expected_Utilization_Percent"] = (
    capacity["Forecast_Demand"]
    / capacity["Planned_Delivery_Capacity"]
    * 100
).round(2)


# ============================================================
# IF ACTUAL DEMAND EXISTS
# ============================================================

if "Actual_Demand" in capacity.columns:

    capacity["Actual_Demand"] = (
        capacity["Actual_Demand"]
        .round()
    )

    capacity["Actual_vs_Forecast"] = (
        capacity["Actual_Demand"]
        - capacity["Forecast_Demand"]
    )

    capacity["Actual_Capacity_Gap"] = (
        capacity["Actual_Demand"]
        - capacity["Planned_Delivery_Capacity"]
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CAPACITY PLANNING SUMMARY")
print("=" * 60)

print(f"\nPlanning days: {len(capacity):,}")

print(
    f"Date range: "
    f"{capacity['Date'].min().date()} "
    f"to "
    f"{capacity['Date'].max().date()}"
)

print("\n--- Forecast Demand ---")

print(
    capacity["Forecast_Demand"]
    .agg(["mean", "std", "min", "max"])
    .round(2)
)

print("\n--- Recommended Agents ---")

print(
    capacity["Recommended_Agents"]
    .agg(["mean", "min", "max"])
    .round(2)
)

print("\n--- Planned Delivery Capacity ---")

print(
    capacity["Planned_Delivery_Capacity"]
    .agg(["mean", "min", "max"])
    .round(2)
)

print("\n--- Expected Utilization ---")

print(
    capacity["Expected_Utilization_Percent"]
    .agg(["mean", "min", "max"])
    .round(2)
)

print("\n--- Demand Risk Distribution ---")

print(
    capacity["Demand_Risk"]
    .value_counts()
)


# ============================================================
# HIGH DEMAND DAYS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 HIGH-DEMAND DAYS")
print("=" * 60)

display_columns = [
    "Date",
    "Forecast_Demand",
    "Recommended_Agents",
    "Planned_Delivery_Capacity",
    "Expected_Utilization_Percent",
    "Demand_Risk",
]

available_columns = [
    column
    for column in display_columns
    if column in capacity.columns
]

print(
    capacity
    .nlargest(10, "Forecast_Demand")[available_columns]
    .to_string(index=False)
)


# ============================================================
# MONTHLY CAPACITY SUMMARY
# ============================================================

capacity["Month"] = (
    capacity["Date"]
    .dt.to_period("M")
)

monthly_summary = (
    capacity
    .groupby("Month")
    .agg(
        Forecast_Demand=(
            "Forecast_Demand",
            "sum"
        ),
        Average_Agents=(
            "Recommended_Agents",
            "mean"
        ),
        Peak_Agents=(
            "Recommended_Agents",
            "max"
        ),
        Average_Utilization=(
            "Expected_Utilization_Percent",
            "mean"
        )
    )
    .round(2)
)

print("\n" + "=" * 60)
print("MONTHLY CAPACITY SUMMARY")
print("=" * 60)

print(monthly_summary)


# ============================================================
# SAVE RESULTS
# ============================================================

capacity.to_csv(
    CAPACITY_OUTPUT_FILE,
    index=False
)

monthly_output_file = (
    OUTPUT_DIR
    / "monthly_capacity_summary.csv"
)

monthly_summary.to_csv(
    monthly_output_file
)


print("\n" + "=" * 60)
print("CAPACITY PLANNING COMPLETED")
print("=" * 60)

print("\nSaved capacity plan:")
print(CAPACITY_OUTPUT_FILE)

print("\nSaved monthly capacity summary:")
print(monthly_output_file)

print("\nOperational assumptions:")
print(
    f"Deliveries per agent/day: "
    f"{DELIVERIES_PER_AGENT_PER_DAY}"
)

print(
    f"Safety buffer: "
    f"{SAFETY_BUFFER * 100:.0f}%"
)

print(
    f"Target utilization: "
    f"{TARGET_UTILIZATION * 100:.0f}%"
)