"""
Feature Engineering Pipeline
E-Commerce Delivery Operations Intelligence

Creates machine-learning and forecasting features from the
synthetic delivery operations dataset.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "amazon_delivery_synthetic.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "amazon_delivery_final.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 60)

print("\nLoading synthetic dataset...")
print(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    keep_default_na=False,
    parse_dates=["Order_Date"],
)

print(f"\nRows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# DATE FEATURES
# ============================================================

print("\nCreating date features...")

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Day"] = df["Order_Date"].dt.day
df["DayOfWeek"] = df["Order_Date"].dt.dayofweek
df["Quarter"] = df["Order_Date"].dt.quarter

df["Is_Weekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)


# ============================================================
# SEASON FEATURES
# ============================================================

print("Creating season features...")


def get_season(month):
    """Map Indian calendar months to operational seasons."""

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post_Monsoon"


df["Season"] = df["Month"].apply(get_season)


# ============================================================
# TIME FEATURES
# ============================================================

print("Creating time features...")

order_time = pd.to_datetime(
    df["Order_Time"],
    format="%H:%M:%S",
)

df["Order_Hour"] = order_time.dt.hour
df["Order_Minute"] = order_time.dt.minute


def get_time_period(hour):
    """Classify operational demand periods."""

    if 6 <= hour < 11:
        return "Morning"

    if 11 <= hour < 16:
        return "Afternoon"

    if 16 <= hour < 21:
        return "Evening"

    return "Night"


df["Time_Period"] = df["Order_Hour"].apply(
    get_time_period
)


df["Is_Peak_Hour"] = (
    df["Order_Hour"].isin([17, 18, 19, 20, 21, 22])
).astype(int)


# ============================================================
# FESTIVAL FEATURES
# ============================================================

print("Creating festival features...")

df["Is_Festival"] = (
    df["Festival"] != "None"
).astype(int)


# ============================================================
# DISTANCE FEATURES
# ============================================================

print("Creating distance features...")

df["Distance_Bucket"] = pd.cut(
    df["Distance_KM"],
    bins=[0, 2, 5, 10, 20, float("inf")],
    labels=[
        "Very_Short",
        "Short",
        "Medium",
        "Long",
        "Very_Long",
    ],
    include_lowest=True,
)


# ============================================================
# DELIVERY PERFORMANCE FEATURES
# ============================================================

print("Creating delivery performance features...")

df["Delivery_Speed_KM_Per_Min"] = (
    df["Distance_KM"]
    / df["Delivery_Time"]
)

df["Delivery_Speed_KM_Per_Hour"] = (
    df["Delivery_Speed_KM_Per_Min"]
    * 60
)


# ============================================================
# DATA VALIDATION
# ============================================================

print("\nRunning validation...")

assert len(df) == 750_000, (
    "Unexpected row count."
)

assert df["Order_ID"].duplicated().sum() == 0, (
    "Duplicate Order_ID values found."
)

assert df["Is_Festival"].isin([0, 1]).all(), (
    "Invalid Is_Festival values."
)

assert df["Is_Weekend"].isin([0, 1]).all(), (
    "Invalid Is_Weekend values."
)

assert df["Is_Peak_Hour"].isin([0, 1]).all(), (
    "Invalid Is_Peak_Hour values."
)

assert (df["Distance_KM"] > 0).all(), (
    "Invalid distance values."
)

assert (df["Delivery_Time"] > 0).all(), (
    "Invalid delivery time values."
)

assert (
    df["Delivery_Speed_KM_Per_Hour"] > 0
).all(), (
    "Invalid delivery speed."
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nFinal rows: {len(df):,}")
print(f"Final columns: {len(df.columns)}")

print("\nNew features created:")

new_features = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "Is_Weekend",
    "Season",
    "Order_Hour",
    "Order_Minute",
    "Time_Period",
    "Is_Peak_Hour",
    "Is_Festival",
    "Distance_Bucket",
    "Delivery_Speed_KM_Per_Min",
    "Delivery_Speed_KM_Per_Hour",
]

for feature in new_features:
    print(f"  + {feature}")

print(f"\nSaved final dataset:")
print(OUTPUT_FILE)