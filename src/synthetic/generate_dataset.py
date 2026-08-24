from pathlib import Path

import numpy as np
import pandas as pd

from src.synthetic.config import (
    AREA_WEIGHTS,
    BASE_DAILY_ORDERS,
    CATEGORY_WEIGHTS,
    CITY_ANCHORS,
    COORDINATE_NOISE_STD,
    DELIVERY_TIME_NOISE_STD,
    MONSOON_MONTHS,
    RANDOM_SEED,
    REGION_WEIGHTS,
    SYNTHETIC_DATA_PATH,
    SYNTHETIC_DIR,
    TRAFFIC_TIME_EFFECT,
    TRAFFIC_WEIGHTS,
    VEHICLE_TIME_EFFECT,
    VEHICLE_WEIGHTS,
    WEATHER_TIME_EFFECT,
    WEATHER_WEIGHTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DAILY_DATA_PATH = (
    SYNTHETIC_DIR
    / "synthetic_daily_operations.csv"
)


# ============================================================
# SOURCE-INFORMED DISTRIBUTIONS
# ============================================================

SOURCE_HOUR_WEIGHTS = {
    0: 0.0098,
    8: 0.0415,
    9: 0.0443,
    10: 0.0454,
    11: 0.0447,
    12: 0.0203,
    13: 0.0178,
    14: 0.0180,
    15: 0.0200,
    16: 0.0162,
    17: 0.0973,
    18: 0.1023,
    19: 0.1049,
    20: 0.1032,
    21: 0.1068,
    22: 0.1044,
    23: 0.1031,
}

AGENT_AGE_MEAN = 29.57
AGENT_AGE_STD = 5.82


# ============================================================
# HELPERS
# ============================================================

def normalized_weights(weights):
    """Return probabilities normalized to sum to one."""

    values = np.array(
        list(weights.values()),
        dtype=float,
    )

    values = values / values.sum()

    return list(weights.keys()), values


def sample_from_weights(rng, weights, size):
    """Sample categorical values using a probability dictionary."""

    choices, probabilities = normalized_weights(weights)

    return rng.choice(
        choices,
        size=size,
        p=probabilities,
    )


def build_city_table():
    """Create city metadata and normalized city weights."""

    rows = []

    for city, info in CITY_ANCHORS.items():
        rows.append(
            {
                "City": city,
                "Region": info["region"],
                "Latitude": info["latitude"],
                "Longitude": info["longitude"],
            }
        )

    cities = pd.DataFrame(rows)

    city_region_counts = (
        cities["Region"]
        .value_counts()
        .to_dict()
    )

    cities["Region_Weight"] = cities["Region"].map(
        REGION_WEIGHTS
    )

    cities["City_Weight"] = (
        cities["Region_Weight"]
        / cities["Region"].map(city_region_counts)
    )

    cities["City_Weight"] = (
        cities["City_Weight"]
        / cities["City_Weight"].sum()
    )

    return cities


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Calculate great-circle distance in kilometers."""

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    return (
        2
        * earth_radius_km
        * np.arcsin(
            np.sqrt(a)
        )
    )


def generate_coordinates(
    rng,
    cities,
    size,
):
    """Generate valid store and customer coordinates."""

    city_indices = rng.choice(
        len(cities),
        size=size,
        p=cities["City_Weight"].to_numpy(),
    )

    selected = cities.iloc[
        city_indices
    ].reset_index(drop=True)

    store_lat = (
        selected["Latitude"].to_numpy()
        + rng.normal(
            0,
            COORDINATE_NOISE_STD,
            size,
        )
    )

    store_lon = (
        selected["Longitude"].to_numpy()
        + rng.normal(
            0,
            COORDINATE_NOISE_STD,
            size,
        )
    )

    drop_lat = (
        store_lat
        + rng.normal(
            0,
            COORDINATE_NOISE_STD * 1.35,
            size,
        )
    )

    drop_lon = (
        store_lon
        + rng.normal(
            0,
            COORDINATE_NOISE_STD * 1.35,
            size,
        )
    )

    distance_km = haversine_km(
        store_lat,
        store_lon,
        drop_lat,
        drop_lon,
    )

    distance_km = np.clip(
        distance_km,
        0.5,
        35.0,
    )

    return (
        selected,
        store_lat,
        store_lon,
        drop_lat,
        drop_lon,
        distance_km,
    )


def generate_order_times(
    rng,
    size,
):
    """Generate order times using the empirical source profile."""

    hours = np.array(
        list(SOURCE_HOUR_WEIGHTS.keys())
    )

    probabilities = np.array(
        list(SOURCE_HOUR_WEIGHTS.values()),
        dtype=float,
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    selected_hours = rng.choice(
        hours,
        size=size,
        p=probabilities,
    )

    minutes = rng.integers(
        0,
        60,
        size=size,
    )

    seconds = rng.integers(
        0,
        60,
        size=size,
    )

    return (
        selected_hours * 3600
        + minutes * 60
        + seconds
    )


def generate_weather(
    rng,
    months,
    regions,
    size,
):
    """
    Generate weather with moderate seasonal and regional effects.

    The source distribution remains the dominant baseline.
    Monsoon effects are intentionally moderate so the synthetic
    distribution does not drift too far from the source.
    """

    weather = np.empty(
        size,
        dtype=object,
    )

    base_choices, base_probs = normalized_weights(
        WEATHER_WEIGHTS
    )

    for i in range(size):
        probabilities = base_probs.copy()

        month = months[i]
        region = regions[i]

        # Moderate monsoon adjustment.
        if month in MONSOON_MONTHS:
            for index, choice in enumerate(
                base_choices
            ):
                if choice in {
                    "Cloudy",
                    "Stormy",
                }:
                    probabilities[index] *= 1.18

                elif choice == "Sunny":
                    probabilities[index] *= 0.88

        # Moderate northern winter fog effect.
        if (
            region == "North"
            and month in {11, 12, 1, 2}
        ):
            for index, choice in enumerate(
                base_choices
            ):
                if choice == "Fog":
                    probabilities[index] *= 1.25

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        weather[i] = rng.choice(
            base_choices,
            p=probabilities,
        )

    return weather


def generate_traffic(
    rng,
    hours,
    weather,
    demand_pressure,
    size,
):
    """
    Generate traffic using the source distribution as the
    dominant baseline with moderate operational adjustments.
    """

    base_choices, base_probs = normalized_weights(
        TRAFFIC_WEIGHTS
    )

    traffic = np.empty(
        size,
        dtype=object,
    )

    for i in range(size):
        probabilities = base_probs.copy()

        hour = hours[i]

        # Moderate evening congestion effect.
        if hour in {
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        }:
            for index, choice in enumerate(
                base_choices
            ):
                if choice == "Jam":
                    probabilities[index] *= 1.16

                elif choice == "High":
                    probabilities[index] *= 1.08

                elif choice == "Low":
                    probabilities[index] *= 0.90

        # Moderate demand-pressure effect.
        pressure = demand_pressure[i]

        if pressure > 1.10:
            for index, choice in enumerate(
                base_choices
            ):
                if choice in {
                    "Jam",
                    "High",
                }:
                    probabilities[index] *= 1.08

        elif pressure < 0.90:
            for index, choice in enumerate(
                base_choices
            ):
                if choice == "Low":
                    probabilities[index] *= 1.08

        # Moderate poor-weather congestion effect.
        if weather[i] in {
            "Stormy",
            "Fog",
            "Cloudy",
        }:
            for index, choice in enumerate(
                base_choices
            ):
                if choice in {
                    "Jam",
                    "High",
                }:
                    probabilities[index] *= 1.06

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        traffic[i] = rng.choice(
            base_choices,
            p=probabilities,
        )

    return traffic


def generate_agent_attributes(
    rng,
    size,
):
    """Generate realistic agent age and rating."""

    ages = np.rint(
        rng.normal(
            AGENT_AGE_MEAN,
            AGENT_AGE_STD,
            size,
        )
    )

    ages = np.clip(
        ages,
        18,
        50,
    ).astype(int)

    ratings = rng.normal(
        4.63,
        0.25,
        size,
    )

    ratings = np.clip(
        ratings,
        3.0,
        5.0,
    )

    ratings = np.round(
        ratings,
        1,
    )

    return ages, ratings


def generate_delivery_time(
    rng,
    distance_km,
    traffic,
    weather,
    vehicle,
    area,
    agent_rating,
    demand_pressure,
    is_monsoon,
):
    """
    Generate delivery time from interacting operational factors.

    Monsoon contributes a modest additional operational penalty
    while weather and traffic retain their own effects.
    """

    size = len(distance_km)

    base_time = 55.0

    distance_effect = (
        distance_km * 2.4
    )

    traffic_effect = np.array(
        [
            TRAFFIC_TIME_EFFECT[x]
            for x in traffic
        ]
    )

    weather_effect = np.array(
        [
            WEATHER_TIME_EFFECT[x]
            for x in weather
        ]
    )

    vehicle_effect = np.array(
        [
            VEHICLE_TIME_EFFECT[x]
            for x in vehicle
        ]
    )

    area_effect = np.ones(size)

    area_effect[
        area == "Metropolitian"
    ] = 1.05

    area_effect[
        area == "Semi-Urban"
    ] = 1.45

    area_effect[
        area == "Urban"
    ] = 0.90

    area_effect[
        area == "Other"
    ] = 0.88

    capacity_effect = (
        1.0
        + np.clip(
            demand_pressure - 1.0,
            -0.15,
            0.30,
        )
        * 0.25
    )

    rating_effect = (
        1.0
        - (agent_rating - 4.0)
        * 0.035
    )

    # Modest monsoon operational effect.
    monsoon_effect = np.where(
        is_monsoon,
        1.06,
        1.0,
    )

    raw_time = (
        base_time
        + distance_effect
    )

    raw_time = (
        raw_time
        * traffic_effect
        * weather_effect
        * vehicle_effect
        * area_effect
        * capacity_effect
        * rating_effect
        * monsoon_effect
    )

    noise = rng.normal(
        0,
        DELIVERY_TIME_NOISE_STD,
        size,
    )

    delivery_time = (
        raw_time
        + noise
    )

    return np.clip(
        np.rint(delivery_time),
        10,
        270,
    ).astype(int)


def generate_orders_for_day(
    daily_row,
    rng,
    cities,
):
    """Generate all orders belonging to one calendar day."""

    size = int(
        daily_row["Demand"]
    )

    date = pd.Timestamp(
        daily_row["Date"]
    )

    demand_pressure = (
        size
        / BASE_DAILY_ORDERS
    )

    (
        city_data,
        store_lat,
        store_lon,
        drop_lat,
        drop_lon,
        distance_km,
    ) = generate_coordinates(
        rng,
        cities,
        size,
    )

    city = city_data[
        "City"
    ].to_numpy()

    region = city_data[
        "Region"
    ].to_numpy()

    area = sample_from_weights(
        rng,
        AREA_WEIGHTS,
        size,
    )

    vehicle = sample_from_weights(
        rng,
        VEHICLE_WEIGHTS,
        size,
    )

    category = sample_from_weights(
        rng,
        CATEGORY_WEIGHTS,
        size,
    )

    seconds_since_midnight = (
        generate_order_times(
            rng,
            size,
        )
    )

    hours = (
        seconds_since_midnight
        // 3600
    )

    order_time = (
        pd.to_datetime(
            seconds_since_midnight,
            unit="s",
        )
        .strftime("%H:%M:%S")
    )

    is_monsoon = np.full(
        size,
        bool(
            daily_row[
                "Is_Monsoon"
            ]
        ),
    )

    weather = generate_weather(
        rng,
        np.full(
            size,
            date.month,
        ),
        region,
        size,
    )

    traffic = generate_traffic(
        rng,
        hours,
        weather,
        np.full(
            size,
            demand_pressure,
        ),
        size,
    )

    agent_age, agent_rating = (
        generate_agent_attributes(
            rng,
            size,
        )
    )

    pickup_delay = (
        8
        + (
            traffic == "Jam"
        ) * 8
        + (
            traffic == "High"
        ) * 4
        + (
            weather == "Stormy"
        ) * 3
        + is_monsoon * 2
        + rng.normal(
            0,
            3,
            size,
        )
    )

    pickup_delay = np.clip(
        np.rint(pickup_delay),
        3,
        40,
    ).astype(int)

    delivery_time = (
        generate_delivery_time(
            rng,
            distance_km,
            traffic,
            weather,
            vehicle,
            area,
            agent_rating,
            np.full(
                size,
                demand_pressure,
            ),
            is_monsoon,
        )
    )

    order_timestamp = (
        pd.to_datetime(
            date.strftime("%Y-%m-%d")
            + " "
            + order_time
        )
    )

    pickup_timestamp = (
        order_timestamp
        + pd.to_timedelta(
            pickup_delay,
            unit="m",
        )
    )

    order_ids = [
        f"SYN{date.strftime('%Y%m%d')}{i:05d}"
        for i in range(size)
    ]

    # IMPORTANT:
    # Explicitly use "None" for normal days.
    # This prevents NaN/mixed-type Festival values.
    festival = str(
        daily_row["Festival"]
    ).strip()

    if not festival or festival.lower() == "nan":
        festival = "None"

    result = pd.DataFrame(
        {
            "Order_ID": order_ids,
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Store_Latitude": np.round(
                store_lat,
                6,
            ),
            "Store_Longitude": np.round(
                store_lon,
                6,
            ),
            "Drop_Latitude": np.round(
                drop_lat,
                6,
            ),
            "Drop_Longitude": np.round(
                drop_lon,
                6,
            ),
            "Order_Date": date.strftime(
                "%Y-%m-%d"
            ),
            "Order_Time": order_timestamp.strftime(
                "%H:%M:%S"
            ),
            "Pickup_Time": pickup_timestamp.strftime(
                "%H:%M:%S"
            ),
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Delivery_Time": delivery_time,
            "Category": category,
            "City": city,
            "Region": region,
            "Distance_KM": np.round(
                distance_km,
                2,
            ),
            "Pickup_Delay_Minutes": pickup_delay,
            "Demand_Pressure": np.round(
                demand_pressure,
                3,
            ),
            "Festival": festival,
            "Is_Monsoon": is_monsoon,
        }
    )

    return result


def main():
    print("=" * 60)
    print("SYNTHETIC INDIVIDUAL ORDER GENERATOR")
    print("=" * 60)

    print("\nLoading daily operational layer:")
    print(DAILY_DATA_PATH)

    daily = pd.read_csv(
        DAILY_DATA_PATH,
        parse_dates=["Date"],
    )

    # ========================================================
    # FESTIVAL NORMALIZATION
    # ========================================================
    # The daily generator may represent non-festival days as
    # blank/NaN values. Convert them explicitly to "None".
    daily["Festival"] = (
        daily["Festival"]
        .fillna("None")
        .astype(str)
        .str.strip()
    )

    daily.loc[
        daily["Festival"] == "",
        "Festival",
    ] = "None"

    daily.loc[
        daily["Festival"].str.lower() == "nan",
        "Festival",
    ] = "None"

    print(
        f"\nDaily records: {len(daily):,}"
    )

    print(
        f"Expected orders: "
        f"{daily['Demand'].sum():,}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    cities = build_city_table()

    print(
        f"\nSynthetic cities: "
        f"{len(cities)}"
    )

    all_orders = []

    for counter, (_, daily_row) in enumerate(
        daily.iterrows(),
        start=1,
    ):
        day_orders = generate_orders_for_day(
            daily_row,
            rng,
            cities,
        )

        all_orders.append(
            day_orders
        )

        if counter % 100 == 0:
            print(
                f"Generated "
                f"{counter:,}/{len(daily):,} days"
            )

    result = pd.concat(
        all_orders,
        ignore_index=True,
    )

    print(
        "\nGenerated orders:"
    )
    print(
        f"{len(result):,}"
    )

    print(
        "\nExpected orders:"
    )
    print(
        f"{daily['Demand'].sum():,}"
    )

    # ========================================================
    # FINAL GENERATION VALIDATION
    # ========================================================

    if len(result) != daily["Demand"].sum():
        raise ValueError(
            "Generated order count does not "
            "match daily demand."
        )

    if result["Order_ID"].duplicated().any():
        raise ValueError(
            "Duplicate Order_ID values detected."
        )

    if result["Festival"].isna().any():
        raise ValueError(
            "Festival contains missing values."
        )

    if (
        result[
            [
                "Store_Latitude",
                "Store_Longitude",
                "Drop_Latitude",
                "Drop_Longitude",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing coordinates detected."
        )

    if (
        result["Distance_KM"] <= 0
    ).any():
        raise ValueError(
            "Invalid distance detected."
        )

    if (
        result["Delivery_Time"] < 10
    ).any() or (
        result["Delivery_Time"] > 270
    ).any():
        raise ValueError(
            "Invalid delivery time detected."
        )

    if (
        result["Agent_Rating"] < 1
    ).any() or (
        result["Agent_Rating"] > 5
    ).any():
        raise ValueError(
            "Invalid agent rating detected."
        )

    if (
        result["Agent_Age"] < 18
    ).any() or (
        result["Agent_Age"] > 50
    ).any():
        raise ValueError(
            "Invalid agent age detected."
        )

    SYNTHETIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        SYNTHETIC_DATA_PATH,
        index=False,
    )

    print(
        "\nSaved synthetic order dataset:"
    )
    print(
        SYNTHETIC_DATA_PATH
    )

    print(
        "\nFinal shape:"
    )
    print(
        result.shape
    )

    print(
        "\nFestival values:"
    )
    print(
        result["Festival"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal generation validation:"
    )
    print(
        "PASS: Order count matches daily demand."
    )
    print(
        "PASS: Order_ID values are unique."
    )
    print(
        "PASS: Festival contains no missing values."
    )
    print(
        "PASS: Coordinates are complete."
    )
    print(
        "PASS: Distances are positive."
    )
    print(
        "PASS: Delivery times are within bounds."
    )
    print(
        "PASS: Agent attributes are within bounds."
    )


if __name__ == "__main__":
    main()