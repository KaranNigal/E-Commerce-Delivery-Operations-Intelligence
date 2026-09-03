"""
Exploratory Data Analysis (EDA)
E-Commerce Delivery Operations Intelligence

Analyzes the synthetic demonstration dataset and generates
business-focused visualizations for demand, delivery performance,
weather, traffic, seasonality, festivals, and regional operations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "amazon_delivery_synthetic.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    keep_default_na=False,
    parse_dates=["Order_Date"],
)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(
    f"Date range: {df['Order_Date'].min().date()} "
    f"to {df['Order_Date'].max().date()}"
)


# ============================================================
# FEATURE PREPARATION
# ============================================================

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Day_Name"] = df["Order_Date"].dt.day_name()
df["DayOfWeek"] = df["Order_Date"].dt.dayofweek

df["Hour"] = (
    pd.to_datetime(df["Order_Time"], format="%H:%M:%S")
    .dt.hour
)

df["Is_Weekend"] = df["DayOfWeek"] >= 5


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):
    """Save the current matplotlib figure."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path.name}")


# ============================================================
# 1. DAILY DEMAND TREND
# ============================================================

print("\nGenerating visualizations...")

daily_demand = df.groupby("Order_Date").size()

plt.figure(figsize=(14, 6))
plt.plot(daily_demand.index, daily_demand.values)
plt.title("Daily Order Demand Trend")
plt.xlabel("Date")
plt.ylabel("Orders")

save_plot("01_daily_demand_trend.png")


# ============================================================
# 2. MONTHLY DEMAND
# ============================================================

monthly_demand = (
    df.groupby(df["Order_Date"].dt.to_period("M"))
    .size()
)

plt.figure(figsize=(14, 6))
plt.plot(
    monthly_demand.index.astype(str),
    monthly_demand.values,
    marker="o",
)

plt.title("Monthly Order Demand")
plt.xlabel("Month")
plt.ylabel("Orders")
plt.xticks(rotation=45)

save_plot("02_monthly_demand.png")


# ============================================================
# 3. MONTHLY SEASONALITY
# ============================================================

monthly_seasonality = (
    df.groupby("Month_Name")
    .size()
    .reindex(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    )
)

plt.figure(figsize=(14, 6))
plt.bar(
    monthly_seasonality.index,
    monthly_seasonality.values,
)

plt.title("Seasonality: Orders by Month")
plt.xlabel("Month")
plt.ylabel("Total Orders")
plt.xticks(rotation=45)

save_plot("03_monthly_seasonality.png")


# ============================================================
# 4. HOURLY DEMAND
# ============================================================

hourly_demand = df.groupby("Hour").size()

plt.figure(figsize=(12, 6))
plt.bar(hourly_demand.index, hourly_demand.values)

plt.title("Hourly Order Demand Pattern")
plt.xlabel("Hour of Day")
plt.ylabel("Orders")
plt.xticks(range(24))

save_plot("04_hourly_demand.png")


# ============================================================
# 5. DAY OF WEEK DEMAND
# ============================================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

weekday_demand = (
    df.groupby("Day_Name")
    .size()
    .reindex(day_order)
)

plt.figure(figsize=(12, 6))
plt.bar(
    weekday_demand.index,
    weekday_demand.values,
)

plt.title("Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Orders")
plt.xticks(rotation=30)

save_plot("05_weekday_demand.png")


# ============================================================
# 6. FESTIVAL DEMAND
# ============================================================

festival_demand = (
    df.groupby("Festival")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
plt.bar(
    festival_demand.index,
    festival_demand.values,
)

plt.title("Festival vs Normal Order Demand")
plt.xlabel("Festival")
plt.ylabel("Orders")
plt.xticks(rotation=30)

save_plot("06_festival_demand.png")


# ============================================================
# 7. MONSOON EFFECT
# ============================================================

monsoon_stats = (
    df.groupby("Is_Monsoon")
    .agg(
        Orders=("Order_ID", "count"),
        Average_Delivery_Time=("Delivery_Time", "mean"),
    )
)

print("\n--- MONSOON ANALYSIS ---")
print(monsoon_stats.round(2))

plt.figure(figsize=(8, 6))
plt.bar(
    monsoon_stats.index.astype(str),
    monsoon_stats["Average_Delivery_Time"],
)

plt.title("Average Delivery Time: Monsoon vs Non-Monsoon")
plt.xlabel("Is Monsoon")
plt.ylabel("Average Delivery Time (Minutes)")

save_plot("07_monsoon_delivery_impact.png")


# ============================================================
# 8. CITY DEMAND
# ============================================================

city_demand = (
    df.groupby("City")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(14, 6))
plt.bar(
    city_demand.index,
    city_demand.values,
)

plt.title("Order Demand by City")
plt.xlabel("City")
plt.ylabel("Orders")
plt.xticks(rotation=45)

save_plot("08_city_demand.png")


# ============================================================
# 9. REGION DEMAND
# ============================================================

region_demand = (
    df.groupby("Region")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
plt.bar(
    region_demand.index,
    region_demand.values,
)

plt.title("Order Demand by Region")
plt.xlabel("Region")
plt.ylabel("Orders")

save_plot("09_region_demand.png")


# ============================================================
# 10. TRAFFIC IMPACT
# ============================================================

traffic_delivery = (
    df.groupby("Traffic")["Delivery_Time"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- TRAFFIC VS DELIVERY TIME ---")
print(traffic_delivery.round(2))

plt.figure(figsize=(10, 6))
plt.bar(
    traffic_delivery.index,
    traffic_delivery.values,
)

plt.title("Traffic Impact on Delivery Time")
plt.xlabel("Traffic")
plt.ylabel("Average Delivery Time (Minutes)")

save_plot("10_traffic_delivery_impact.png")


# ============================================================
# 11. WEATHER IMPACT
# ============================================================

weather_delivery = (
    df.groupby("Weather")["Delivery_Time"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- WEATHER VS DELIVERY TIME ---")
print(weather_delivery.round(2))

plt.figure(figsize=(12, 6))
plt.bar(
    weather_delivery.index,
    weather_delivery.values,
)

plt.title("Weather Impact on Delivery Time")
plt.xlabel("Weather")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=30)

save_plot("11_weather_delivery_impact.png")


# ============================================================
# 12. DISTANCE VS DELIVERY TIME
# ============================================================

sample = df.sample(
    n=min(30000, len(df)),
    random_state=42,
)

plt.figure(figsize=(10, 7))
plt.scatter(
    sample["Distance_KM"],
    sample["Delivery_Time"],
    alpha=0.25,
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (KM)")
plt.ylabel("Delivery Time (Minutes)")

save_plot("12_distance_vs_delivery_time.png")


# ============================================================
# 13. DELIVERY TIME BY AREA
# ============================================================

area_delivery = (
    df.groupby("Area")["Delivery_Time"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
plt.bar(
    area_delivery.index,
    area_delivery.values,
)

plt.title("Average Delivery Time by Area")
plt.xlabel("Area")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=30)

save_plot("13_area_delivery_time.png")


# ============================================================
# 14. DELIVERY TIME BY VEHICLE
# ============================================================

vehicle_delivery = (
    df.groupby("Vehicle")["Delivery_Time"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
plt.bar(
    vehicle_delivery.index,
    vehicle_delivery.values,
)

plt.title("Average Delivery Time by Vehicle")
plt.xlabel("Vehicle")
plt.ylabel("Average Delivery Time (Minutes)")

save_plot("14_vehicle_delivery_time.png")


# ============================================================
# 15. NUMERIC CORRELATION
# ============================================================

numeric_columns = [
    "Agent_Age",
    "Agent_Rating",
    "Store_Latitude",
    "Store_Longitude",
    "Drop_Latitude",
    "Drop_Longitude",
    "Delivery_Time",
    "Distance_KM",
    "Pickup_Delay_Minutes",
    "Demand_Pressure",
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(12, 10))
plt.imshow(correlation, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index,
)

plt.title("Correlation Matrix")

save_plot("15_correlation_matrix.png")


# ============================================================
# BUSINESS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)

print(
    f"\nPeak demand city: "
    f"{city_demand.idxmax()} "
    f"({city_demand.max():,} orders)"
)

print(
    f"Highest delivery traffic condition: "
    f"{traffic_delivery.idxmax()} "
    f"({traffic_delivery.max():.2f} minutes)"
)

print(
    f"Worst weather condition: "
    f"{weather_delivery.idxmax()} "
    f"({weather_delivery.max():.2f} minutes)"
)

print(
    f"Distance vs Delivery Time correlation: "
    f"{df['Distance_KM'].corr(df['Delivery_Time']):.3f}"
)

print(
    f"Monsoon average delivery time: "
    f"{df[df['Is_Monsoon']]['Delivery_Time'].mean():.2f} minutes"
)

print(
    f"Non-monsoon average delivery time: "
    f"{df[~df['Is_Monsoon']]['Delivery_Time'].mean():.2f} minutes"
)


print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nFigures saved in:")
print(OUTPUT_DIR)