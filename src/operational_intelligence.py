import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ORDERS_FILE = BASE_DIR / "data" / "synthetic" / "amazon_delivery_synthetic.csv"

FORECAST_FILE = (
    BASE_DIR / "models" / "demand_forecast_results.csv"
)

CAPACITY_FILE = (
    BASE_DIR / "reports" / "tables" / "capacity_plan.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_mode(series, default="None"):
    """
    Safely return the most common value in a Series.

    Prevents errors when a group contains only missing values.
    """
    series = series.dropna()

    if series.empty:
        return default

    mode = series.mode()

    if mode.empty:
        return default

    return mode.iloc[0]


def normalize_festival(value):
    """
    Normalize festival values.
    """
    if pd.isna(value):
        return "None"

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return "None"

    return value


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("OPERATIONAL INTELLIGENCE ENGINE")
print("=" * 60)

print("\nLoading datasets...")

orders = pd.read_csv(
    ORDERS_FILE,
    low_memory=False,
    keep_default_na=False,
    parse_dates=["Order_Date"]
)

forecast = pd.read_csv(
    FORECAST_FILE,
    low_memory=False
)

capacity = pd.read_csv(
    CAPACITY_FILE,
    low_memory=False
)


print(f"Orders loaded: {len(orders):,}")
print(f"Forecast records: {len(forecast):,}")
print(f"Capacity records: {len(capacity):,}")


# ============================================================
# NORMALIZE ORDER DATA
# ============================================================

print("\nPreparing operational data...")

orders["Festival"] = orders["Festival"].apply(normalize_festival)

orders["Is_Monsoon"] = (
    orders["Is_Monsoon"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

orders["Order_Date"] = pd.to_datetime(
    orders["Order_Date"],
    errors="coerce"
)

orders = orders.dropna(subset=["Order_Date"])


# ============================================================
# DETECT DATE COLUMN IN FORECAST
# ============================================================

forecast_date_candidates = [
    "Order_Date",
    "Date",
    "date"
]

forecast_date_column = None

for column in forecast_date_candidates:
    if column in forecast.columns:
        forecast_date_column = column
        break

if forecast_date_column is None:
    raise ValueError(
        f"No date column found in forecast file. "
        f"Available columns: {list(forecast.columns)}"
    )

forecast[forecast_date_column] = pd.to_datetime(
    forecast[forecast_date_column],
    errors="coerce"
)

forecast = forecast.dropna(
    subset=[forecast_date_column]
)

forecast = forecast.rename(
    columns={forecast_date_column: "Date"}
)


# ============================================================
# DETECT FORECAST COLUMN
# ============================================================

forecast_candidates = [
    "Predicted_Demand",
    "Forecast_Demand",
    "Prediction"
]

forecast_column = None

for column in forecast_candidates:
    if column in forecast.columns:
        forecast_column = column
        break

if forecast_column is None:
    raise ValueError(
        f"No forecast column found. "
        f"Available columns: {list(forecast.columns)}"
    )

forecast = forecast.rename(
    columns={forecast_column: "Forecast_Demand"}
)


# ============================================================
# DETECT ACTUAL DEMAND COLUMN
# ============================================================

actual_candidates = [
    "Actual_Demand",
    "Demand",
    "Actual"
]

actual_column = None

for column in actual_candidates:
    if column in forecast.columns:
        actual_column = column
        break

if actual_column is not None:
    forecast = forecast.rename(
        columns={actual_column: "Actual_Demand"}
    )
else:
    forecast["Actual_Demand"] = np.nan


# ============================================================
# DETECT DATE COLUMN IN CAPACITY FILE
# ============================================================

capacity_date_candidates = [
    "Date",
    "Order_Date",
    "date"
]

capacity_date_column = None

for column in capacity_date_candidates:
    if column in capacity.columns:
        capacity_date_column = column
        break

if capacity_date_column is None:
    raise ValueError(
        f"No date column found in capacity file. "
        f"Available columns: {list(capacity.columns)}"
    )

capacity[capacity_date_column] = pd.to_datetime(
    capacity[capacity_date_column],
    errors="coerce"
)

capacity = capacity.dropna(
    subset=[capacity_date_column]
)

capacity = capacity.rename(
    columns={capacity_date_column: "Date"}
)


# ============================================================
# CALCULATE DAILY OPERATIONAL METRICS
# ============================================================

print("\nCalculating daily operational metrics...")


daily_metrics = (
    orders
    .groupby("Order_Date")
    .agg(
        Total_Orders=("Order_ID", "count"),

        Average_Delivery_Time=(
            "Delivery_Time",
            "mean"
        ),

        Median_Delivery_Time=(
            "Delivery_Time",
            "median"
        ),

        Average_Distance_KM=(
            "Distance_KM",
            "mean"
        ),

        Average_Agent_Rating=(
            "Agent_Rating",
            "mean"
        ),

        Average_Pickup_Delay=(
            "Pickup_Delay_Minutes",
            "mean"
        ),

        Festival=(
            "Festival",
            lambda x: safe_mode(x, "None")
        ),

        Is_Monsoon=(
            "Is_Monsoon",
            lambda x: bool(x.mode().iloc[0])
            if not x.mode().empty
            else False
        ),

        Dominant_Traffic=(
            "Traffic",
            lambda x: safe_mode(x, "Unknown")
        ),

        Dominant_Weather=(
            "Weather",
            lambda x: safe_mode(x, "Unknown")
        ),

        Average_Demand_Pressure=(
            "Demand_Pressure",
            "mean"
        )
    )
    .reset_index()
)


daily_metrics = daily_metrics.rename(
    columns={"Order_Date": "Date"}
)


print(
    f"Daily operational records: "
    f"{len(daily_metrics):,}"
)


# ============================================================
# MERGE FORECAST DATA
# ============================================================

print("\nMerging demand forecasts...")

operational = daily_metrics.merge(
    forecast[
        [
            "Date",
            "Actual_Demand",
            "Forecast_Demand"
        ]
    ],
    on="Date",
    how="left"
)


# ============================================================
# MERGE CAPACITY DATA
# ============================================================

print("Merging capacity planning data...")

operational = operational.merge(
    capacity,
    on="Date",
    how="left",
    suffixes=("", "_Capacity")
)


# ============================================================
# CREATE OPERATIONAL KPIs
# ============================================================

print("Generating operational intelligence metrics...")


# ------------------------------------------------------------
# DELIVERY PERFORMANCE STATUS
# ------------------------------------------------------------

operational["Delivery_Performance_Status"] = pd.cut(
    operational["Average_Delivery_Time"],
    bins=[
        -np.inf,
        65,
        80,
        95,
        np.inf
    ],
    labels=[
        "Excellent",
        "Good",
        "Warning",
        "Critical"
    ]
)


# ------------------------------------------------------------
# DEMAND FORECAST ERROR
# ------------------------------------------------------------

operational["Forecast_Error"] = (
    operational["Actual_Demand"]
    - operational["Forecast_Demand"]
)


operational["Forecast_Absolute_Error"] = (
    operational["Forecast_Error"]
    .abs()
)


operational["Forecast_Error_Percent"] = np.where(
    operational["Actual_Demand"] > 0,

    (
        operational["Forecast_Absolute_Error"]
        / operational["Actual_Demand"]
    )
    * 100,

    np.nan
)


# ------------------------------------------------------------
# CAPACITY GAP
# ------------------------------------------------------------

if "Planned_Delivery_Capacity" in operational.columns:

    operational["Capacity_Gap"] = (
        operational["Planned_Delivery_Capacity"]
        - operational["Actual_Demand"]
    )

else:

    operational["Capacity_Gap"] = np.nan


# ------------------------------------------------------------
# UTILIZATION STATUS
# ------------------------------------------------------------

if "Expected_Utilization_Percent" in operational.columns:

    operational["Utilization_Status"] = pd.cut(
        operational["Expected_Utilization_Percent"],
        bins=[
            -np.inf,
            70,
            85,
            95,
            np.inf
        ],
        labels=[
            "Underutilized",
            "Optimal",
            "High",
            "Critical"
        ]
    )

else:

    operational["Utilization_Status"] = "Unknown"


# ============================================================
# OPERATIONAL RISK SCORE
# ============================================================

print("Calculating operational risk scores...")


risk_score = np.zeros(len(operational))


# Delivery delay risk
risk_score += np.where(
    operational["Average_Delivery_Time"] > 90,
    3,
    np.where(
        operational["Average_Delivery_Time"] > 75,
        2,
        1
    )
)


# Traffic risk
risk_score += np.where(
    operational["Dominant_Traffic"] == "Jam",
    3,
    np.where(
        operational["Dominant_Traffic"] == "High",
        2,
        1
    )
)


# Weather risk
risk_score += np.where(
    operational["Dominant_Weather"].isin(
        ["Fog", "Cloudy", "Stormy"]
    ),
    2,
    1
)


# Monsoon risk
risk_score += np.where(
    operational["Is_Monsoon"],
    1,
    0
)


# Festival risk
risk_score += np.where(
    operational["Festival"] != "None",
    1,
    0
)


# Capacity risk
if "Capacity_Gap" in operational.columns:

    risk_score += np.where(
        operational["Capacity_Gap"] < 0,
        3,
        np.where(
            operational["Capacity_Gap"] < 100,
            1,
            0
        )
    )


operational["Operational_Risk_Score"] = risk_score


# ============================================================
# RISK LEVEL
# ============================================================

operational["Operational_Risk_Level"] = pd.cut(
    operational["Operational_Risk_Score"],
    bins=[
        -np.inf,
        3,
        6,
        9,
        np.inf
    ],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendation(row):

    recommendations = []

    if row["Dominant_Traffic"] == "Jam":
        recommendations.append(
            "Increase route optimization and dispatch monitoring"
        )

    if row["Average_Delivery_Time"] > 85:
        recommendations.append(
            "Investigate delivery delays and agent allocation"
        )

    if row["Is_Monsoon"]:
        recommendations.append(
            "Prepare monsoon contingency capacity"
        )

    if row["Festival"] != "None":
        recommendations.append(
            f"Increase staffing for {row['Festival']} demand"
        )

    if pd.notna(row["Capacity_Gap"]):

        if row["Capacity_Gap"] < 0:
            recommendations.append(
                "Add delivery capacity immediately"
            )

    if not recommendations:
        recommendations.append(
            "Operations performing within expected range"
        )

    return " | ".join(recommendations)


operational["Recommended_Action"] = operational.apply(
    generate_recommendation,
    axis=1
)


# ============================================================
# SAVE OPERATIONAL DATASET
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "operational_intelligence_dashboard_data.csv"
)


operational.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CREATE RISK SUMMARY
# ============================================================

risk_summary = (
    operational
    .groupby(
        "Operational_Risk_Level",
        observed=False
    )
    .agg(
        Days=("Date", "count"),

        Average_Delivery_Time=(
            "Average_Delivery_Time",
            "mean"
        ),

        Average_Orders=(
            "Total_Orders",
            "mean"
        )
    )
    .round(2)
)


risk_summary_file = (
    OUTPUT_DIR
    / "operational_risk_summary.csv"
)


risk_summary.to_csv(
    risk_summary_file
)


# ============================================================
# HIGH-RISK DAYS
# ============================================================

high_risk_days = (
    operational[
        operational["Operational_Risk_Level"]
        .isin(["High", "Critical"])
    ]
    .sort_values(
        "Operational_Risk_Score",
        ascending=False
    )
)


high_risk_file = (
    OUTPUT_DIR
    / "high_risk_operational_days.csv"
)


high_risk_days.to_csv(
    high_risk_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("OPERATIONAL INTELLIGENCE SUMMARY")
print("=" * 60)

print()

print("--- Analysis Period ---")

print(
    operational["Date"].min().date(),
    "to",
    operational["Date"].max().date()
)


print()

print("--- Average Daily Performance ---")

print(
    operational[
        [
            "Total_Orders",
            "Average_Delivery_Time",
            "Average_Distance_KM",
            "Average_Pickup_Delay"
        ]
    ]
    .mean()
    .round(2)
)


print()

print("--- Operational Risk Distribution ---")

print(
    operational[
        "Operational_Risk_Level"
    ]
    .value_counts()
)


print()

print("--- Delivery Performance Status ---")

print(
    operational[
        "Delivery_Performance_Status"
    ]
    .value_counts()
)


print()

print("--- Top High-Risk Days ---")

display_columns = [
    "Date",
    "Total_Orders",
    "Average_Delivery_Time",
    "Dominant_Traffic",
    "Dominant_Weather",
    "Festival",
    "Operational_Risk_Score",
    "Operational_Risk_Level"
]

print(
    high_risk_days[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)


print()

print("=" * 60)
print("OPERATIONAL INTELLIGENCE COMPLETED")
print("=" * 60)


print()
print("Saved dashboard dataset:")
print(OUTPUT_FILE)

print()
print("Saved risk summary:")
print(risk_summary_file)

print()
print("Saved high-risk operational days:")
print(high_risk_file)