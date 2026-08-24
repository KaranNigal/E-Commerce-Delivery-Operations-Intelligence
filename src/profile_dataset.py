from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_delivery_clean.csv"
)


def main():
    print("=" * 60)
    print("CLEAN DATASET BASELINE PROFILE")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- Numeric distributions ---")

    numeric_columns = [
        "Agent_Age",
        "Agent_Rating",
        "Delivery_Time",
    ]

    print(df[numeric_columns].describe().T)

    print("\n--- Categorical distributions ---")

    categorical_columns = [
        "Weather",
        "Traffic",
        "Vehicle",
        "Area",
        "Category",
    ]

    for column in categorical_columns:
        print(f"\n{column}:")
        print(
            df[column]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    print("\n--- Order-time distribution ---")

    order_time = pd.to_datetime(
        df["Order_Time"],
        format="%H:%M:%S",
        errors="coerce",
    )

    print(
        order_time.dt.hour
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\n--- Day-of-week distribution ---")

    order_date = pd.to_datetime(df["Order_Date"])

    print(
        order_date.dt.day_name()
        .value_counts(normalize=True)
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
        .mul(100)
        .round(2)
    )

    print("\n--- Daily order volume ---")

    daily_orders = (
        order_date
        .value_counts()
        .sort_index()
    )

    print(daily_orders.describe())

    print("\n--- Correlations with Delivery_Time ---")

    correlation_columns = [
        "Agent_Age",
        "Agent_Rating",
        "Delivery_Time",
    ]

    print(
        df[correlation_columns]
        .corr()["Delivery_Time"]
        .sort_values()
    )

    print("\n--- Delivery time by operational category ---")

    for column in [
        "Weather",
        "Traffic",
        "Vehicle",
        "Area",
    ]:
        print(f"\n{column}:")
        print(
            df.groupby(column)["Delivery_Time"]
            .agg(["count", "mean", "median"])
            .round(2)
            .sort_values("mean", ascending=False)
        )


if __name__ == "__main__":
    main()