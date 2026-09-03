from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "synthetic"
    / "amazon_delivery_synthetic.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DELIVERY PERFORMANCE ANALYSIS")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    parse_dates=["Order_Date"]
)

print(f"Loaded orders: {len(df):,}")


# ============================================================
# OVERALL DELIVERY PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("OVERALL DELIVERY PERFORMANCE")
print("=" * 60)

overall = df["Delivery_Time"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).round(2)

print(overall)


# ============================================================
# DELIVERY TIME BY TRAFFIC
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY TRAFFIC")
print("=" * 60)

traffic_performance = (
    df.groupby("Traffic")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(traffic_performance)


# ============================================================
# DELIVERY TIME BY WEATHER
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY WEATHER")
print("=" * 60)

weather_performance = (
    df.groupby("Weather")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(weather_performance)


# ============================================================
# DELIVERY TIME BY CITY
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY CITY")
print("=" * 60)

city_performance = (
    df.groupby("City")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(city_performance)


# ============================================================
# DELIVERY TIME BY REGION
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY REGION")
print("=" * 60)

region_performance = (
    df.groupby("Region")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(region_performance)


# ============================================================
# DELIVERY TIME BY AREA
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY AREA")
print("=" * 60)

area_performance = (
    df.groupby("Area")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(area_performance)


# ============================================================
# DELIVERY TIME BY VEHICLE
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY VEHICLE")
print("=" * 60)

vehicle_performance = (
    df.groupby("Vehicle")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(vehicle_performance)


# ============================================================
# DISTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE BY DISTANCE")
print("=" * 60)

distance_bins = [0, 2, 5, 10, 20, 35]
distance_labels = [
    "0-2 km",
    "2-5 km",
    "5-10 km",
    "10-20 km",
    "20-35 km"
]

df["Distance_Bucket"] = pd.cut(
    df["Distance_KM"],
    bins=distance_bins,
    labels=distance_labels,
    include_lowest=True
)

distance_performance = (
    df.groupby("Distance_Bucket", observed=True)["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print(distance_performance)


distance_correlation = df["Distance_KM"].corr(df["Delivery_Time"])

print(
    f"\nDistance vs Delivery Time correlation: "
    f"{distance_correlation:.4f}"
)


# ============================================================
# MONSOON IMPACT
# ============================================================

print("\n" + "=" * 60)
print("MONSOON IMPACT")
print("=" * 60)

monsoon_performance = (
    df.groupby("Is_Monsoon")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print(monsoon_performance)


# ============================================================
# FESTIVAL IMPACT
# ============================================================

print("\n" + "=" * 60)
print("FESTIVAL IMPACT")
print("=" * 60)

festival_performance = (
    df.groupby("Festival")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(festival_performance)


# ============================================================
# DEMAND PRESSURE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DEMAND PRESSURE IMPACT")
print("=" * 60)

pressure_performance = (
    df.groupby("Demand_Pressure")["Delivery_Time"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

print(pressure_performance)


# ============================================================
# TOP OPERATIONAL BOTTLENECKS
# ============================================================

print("\n" + "=" * 60)
print("TOP OPERATIONAL BOTTLENECKS")
print("=" * 60)

print("\nSlowest traffic condition:")
print(traffic_performance.head(1))

print("\nSlowest weather condition:")
print(weather_performance.head(1))

print("\nSlowest city:")
print(city_performance.head(1))

print("\nSlowest region:")
print(region_performance.head(1))

print("\nSlowest area:")
print(area_performance.head(1))

print("\nSlowest distance bucket:")
print(
    distance_performance
    .sort_values("mean", ascending=False)
    .head(1)
)


# ============================================================
# SAVE ANALYSIS TABLES
# ============================================================

OUTPUT_TABLES = BASE_DIR / "reports" / "tables"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

traffic_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_traffic.csv"
)

weather_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_weather.csv"
)

city_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_city.csv"
)

region_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_region.csv"
)

area_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_area.csv"
)

vehicle_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_vehicle.csv"
)

distance_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_distance.csv"
)

monsoon_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_monsoon.csv"
)

festival_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_festival.csv"
)

pressure_performance.to_csv(
    OUTPUT_TABLES / "delivery_by_demand_pressure.csv"
)


# ============================================================
# VISUALIZATION 1 — TRAFFIC
# ============================================================

plt.figure(figsize=(9, 6))

traffic_performance["mean"].sort_values().plot(kind="bar")

plt.title("Average Delivery Time by Traffic Condition")
plt.xlabel("Traffic")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "delivery_time_by_traffic.png",
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 2 — WEATHER
# ============================================================

plt.figure(figsize=(10, 6))

weather_performance["mean"].sort_values().plot(kind="bar")

plt.title("Average Delivery Time by Weather Condition")
plt.xlabel("Weather")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "delivery_time_by_weather.png",
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 3 — CITY
# ============================================================

plt.figure(figsize=(12, 6))

city_performance["mean"].sort_values().plot(kind="bar")

plt.title("Average Delivery Time by City")
plt.xlabel("City")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "delivery_time_by_city.png",
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 4 — DISTANCE
# ============================================================

plt.figure(figsize=(9, 6))

distance_performance["mean"].plot(kind="bar")

plt.title("Average Delivery Time by Distance")
plt.xlabel("Distance")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "delivery_time_by_distance.png",
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 5 — MONSOON
# ============================================================

plt.figure(figsize=(7, 6))

monsoon_chart = monsoon_performance.copy()
monsoon_chart.index = monsoon_chart.index.map(
    {False: "Non-Monsoon", True: "Monsoon"}
)

monsoon_chart["mean"].plot(kind="bar")

plt.title("Monsoon Impact on Delivery Time")
plt.xlabel("")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monsoon_delivery_impact.png",
    dpi=150
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE ANALYSIS COMPLETED")
print("=" * 60)

print("\nSaved analysis tables:")
print(OUTPUT_TABLES)

print("\nSaved visualizations:")
print(OUTPUT_DIR)