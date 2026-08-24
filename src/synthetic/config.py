from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_delivery_clean.csv"
)

SYNTHETIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
)

SYNTHETIC_DATA_PATH = (
    SYNTHETIC_DIR
    / "amazon_delivery_synthetic.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# SYNTHETIC DATA PERIOD
# ============================================================

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"


# ============================================================
# TARGET SCALE
# ============================================================

TARGET_ORDERS = 750_000


# ============================================================
# BASELINE SOURCE DATA
# ============================================================

# Approximate daily order volume observed in the cleaned
# Kaggle reference dataset.
BASE_DAILY_ORDERS = 994


# ============================================================
# REGIONS AND CITY ANCHORS
#
# These are synthetic modeling anchors. They are NOT Amazon
# operational locations and do not represent proprietary data.
# ============================================================

CITY_ANCHORS = {
    "Delhi": {
        "region": "North",
        "latitude": 28.6139,
        "longitude": 77.2090,
    },
    "Jaipur": {
        "region": "North",
        "latitude": 26.9124,
        "longitude": 75.7873,
    },
    "Lucknow": {
        "region": "North",
        "latitude": 26.8467,
        "longitude": 80.9462,
    },
    "Mumbai": {
        "region": "West",
        "latitude": 19.0760,
        "longitude": 72.8777,
    },
    "Pune": {
        "region": "West",
        "latitude": 18.5204,
        "longitude": 73.8567,
    },
    "Ahmedabad": {
        "region": "West",
        "latitude": 23.0225,
        "longitude": 72.5714,
    },
    "Bengaluru": {
        "region": "South",
        "latitude": 12.9716,
        "longitude": 77.5946,
    },
    "Hyderabad": {
        "region": "South",
        "latitude": 17.3850,
        "longitude": 78.4867,
    },
    "Chennai": {
        "region": "South",
        "latitude": 13.0827,
        "longitude": 80.2707,
    },
    "Kolkata": {
        "region": "East",
        "latitude": 22.5726,
        "longitude": 88.3639,
    },
    "Bhubaneswar": {
        "region": "East",
        "latitude": 20.2961,
        "longitude": 85.8245,
    },
    "Bhopal": {
        "region": "Central",
        "latitude": 23.2599,
        "longitude": 77.4126,
    },
    "Indore": {
        "region": "Central",
        "latitude": 22.7196,
        "longitude": 75.8577,
    },
}


# ============================================================
# REGION MIX
#
# This is a synthetic augmentation assumption informed by the
# operational distribution of the source dataset.
# ============================================================

REGION_WEIGHTS = {
    "North": 0.20,
    "West": 0.25,
    "South": 0.25,
    "East": 0.15,
    "Central": 0.15,
}


# ============================================================
# AREA MIX
#
# Preserves the approximate cleaned Kaggle distribution.
# ============================================================

AREA_WEIGHTS = {
    "Metropolitian": 0.7476,
    "Urban": 0.2229,
    "Other": 0.0260,
    "Semi-Urban": 0.0035,
}


# ============================================================
# VEHICLE MIX
#
# Preserves the approximate cleaned Kaggle distribution.
# ============================================================

VEHICLE_WEIGHTS = {
    "motorcycle": 0.5836,
    "scooter": 0.3347,
    "van": 0.0813,
    "bicycle": 0.0003,
}


# ============================================================
# PRODUCT CATEGORY MIX
#
# The source dataset is approximately balanced across its
# sixteen categories. These weights intentionally preserve
# that characteristic.
# ============================================================

CATEGORY_WEIGHTS = {
    "Electronics": 0.0651,
    "Books": 0.0646,
    "Jewelry": 0.0641,
    "Toys": 0.0636,
    "Skincare": 0.0634,
    "Snacks": 0.0633,
    "Outdoors": 0.0628,
    "Apparel": 0.0623,
    "Sports": 0.0622,
    "Grocery": 0.0615,
    "Pet Supplies": 0.0615,
    "Home": 0.0614,
    "Cosmetics": 0.0612,
    "Kitchen": 0.0611,
    "Clothing": 0.0610,
    "Shoes": 0.0610,
}


# ============================================================
# WEATHER
#
# Source-informed baseline distribution.
# Monsoon and regional effects will modify this later.
# ============================================================

WEATHER_WEIGHTS = {
    "Fog": 0.1701,
    "Stormy": 0.1686,
    "Cloudy": 0.1666,
    "Sandstorms": 0.1656,
    "Windy": 0.1651,
    "Sunny": 0.1618,
}


# ============================================================
# TRAFFIC
#
# Source-informed baseline distribution.
# Hour, demand and weather will modify this later.
# ============================================================

TRAFFIC_WEIGHTS = {
    "Low": 0.3429,
    "Jam": 0.3138,
    "Medium": 0.2430,
    "High": 0.0982,
}


# ============================================================
# VEHICLE DELIVERY-TIME EFFECT
#
# Relative operational adjustments used by the synthetic
# generator. These are modeling assumptions, not source facts.
# ============================================================

VEHICLE_TIME_EFFECT = {
    "motorcycle": 1.00,
    "scooter": 0.91,
    "van": 0.90,
    "bicycle": 1.05,
}


# ============================================================
# TRAFFIC DELIVERY-TIME EFFECT
# ============================================================

TRAFFIC_TIME_EFFECT = {
    "Low": 0.81,
    "Medium": 1.00,
    "High": 1.08,
    "Jam": 1.18,
}


# ============================================================
# WEATHER DELIVERY-TIME EFFECT
# ============================================================

WEATHER_TIME_EFFECT = {
    "Sunny": 0.83,
    "Windy": 0.99,
    "Sandstorms": 0.99,
    "Stormy": 0.99,
    "Fog": 1.09,
    "Cloudy": 1.10,
}


# ============================================================
# MONSOON
# ============================================================

MONSOON_MONTHS = {
    6,
    7,
    8,
    9,
}


# ============================================================
# FESTIVAL CALENDAR
#
# These dates are synthetic modeling inputs used to create
# demonstration seasonality. They are NOT Amazon demand data.
# ============================================================

FESTIVAL_CALENDAR = {
    2024: {
        "Holi": ("2024-03-20", "2024-03-26"),
        "Eid": ("2024-04-08", "2024-04-12"),
        "Raksha Bandhan": ("2024-08-17", "2024-08-20"),
        "Dussehra": ("2024-10-09", "2024-10-13"),
        "Diwali": ("2024-10-28", "2024-11-03"),
        "Christmas": ("2024-12-20", "2024-12-26"),
    },
    2025: {
        "Holi": ("2025-03-10", "2025-03-16"),
        "Eid": ("2025-03-28", "2025-04-02"),
        "Raksha Bandhan": ("2025-08-08", "2025-08-10"),
        "Dussehra": ("2025-09-28", "2025-10-03"),
        "Diwali": ("2025-10-15", "2025-10-22"),
        "Christmas": ("2025-12-20", "2025-12-26"),
    },
}


# ============================================================
# FESTIVAL DEMAND MULTIPLIERS
#
# Synthetic assumptions for demonstration forecasting.
# ============================================================

FESTIVAL_DEMAND_EFFECT = {
    "Holi": 1.12,
    "Eid": 1.10,
    "Raksha Bandhan": 1.15,
    "Dussehra": 1.18,
    "Diwali": 1.35,
    "Christmas": 1.12,
}


# ============================================================
# WEEKDAY DEMAND EFFECT
#
# Normalized around approximately 1.0.
# ============================================================

WEEKDAY_DEMAND_EFFECT = {
    0: 0.98,  # Monday
    1: 0.99,  # Tuesday
    2: 1.05,  # Wednesday
    3: 0.99,  # Thursday
    4: 1.05,  # Friday
    5: 1.01,  # Saturday
    6: 1.00,  # Sunday
}


# ============================================================
# PEAK-HOUR DEMAND EFFECT
# ============================================================

HOUR_DEMAND_EFFECT = {
    0: 0.40,
    1: 0.40,
    2: 0.40,
    3: 0.40,
    4: 0.40,
    5: 0.40,
    6: 0.40,
    7: 0.70,
    8: 0.85,
    9: 0.90,
    10: 0.95,
    11: 0.90,
    12: 0.55,
    13: 0.50,
    14: 0.52,
    15: 0.55,
    16: 0.48,
    17: 1.35,
    18: 1.42,
    19: 1.45,
    20: 1.43,
    21: 1.47,
    22: 1.44,
    23: 1.40,
}


# ============================================================
# CAPACITY BASELINES
# ============================================================

BASE_AGENTS_BY_AREA = {
    "Metropolitian": 180,
    "Urban": 90,
    "Other": 45,
    "Semi-Urban": 20,
}

BASE_VEHICLES_BY_AREA = {
    "Metropolitian": 210,
    "Urban": 105,
    "Other": 55,
    "Semi-Urban": 25,
}


# ============================================================
# SYNTHETIC NOISE
# ============================================================

DEMAND_NOISE_STD = 0.08
DELIVERY_TIME_NOISE_STD = 12.0
COORDINATE_NOISE_STD = 0.08