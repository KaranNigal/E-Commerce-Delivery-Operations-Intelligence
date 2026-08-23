from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_delivery.csv"


def main():
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset path:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\n" + "=" * 60)
    print("DATA QUALITY PROFILE")
    print("=" * 60)

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\nUnique values:")
    print(df.nunique().sort_values())

    print("\nNumerical summary:")
    print(df.describe().T)

    print("\nOrder date range:")
    print(f"Minimum: {df['Order_Date'].min()}")
    print(f"Maximum: {df['Order_Date'].max()}")

    print("\nCategorical value counts:")

    for column in ["Weather", "Traffic", "Vehicle", "Area", "Category"]:
        print(f"\n--- {column} ---")
        print(df[column].value_counts(dropna=False))

        print("\n" + "=" * 60)
    print("ANOMALY INVESTIGATION")
    print("=" * 60)

    print("\nAgent ratings outside 1-5:")
    print(
        df.loc[
            (df["Agent_Rating"] < 1) | (df["Agent_Rating"] > 5),
            ["Order_ID", "Agent_Rating"]
        ].to_string(index=False)
    )

    print("\nPotentially suspicious store coordinates:")
    print(
        df.loc[
            (df["Store_Latitude"] < 0) |
            (df["Store_Longitude"] < 0),
            [
                "Order_ID",
                "Store_Latitude",
                "Store_Longitude"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nPotentially suspicious drop coordinates:")
    print(
        df.loc[
            (df["Drop_Latitude"] <= 0) |
            (df["Drop_Longitude"] <= 0),
            [
                "Order_ID",
                "Drop_Latitude",
                "Drop_Longitude"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nOrder ID uniqueness:")
    print(f"Unique Order_IDs: {df['Order_ID'].nunique():,}")
    print(f"Total rows: {len(df):,}")

    print("\nOrder date distribution:")
    print(
        df["Order_Date"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTime samples:")
    print(
        df[
            ["Order_Date", "Order_Time", "Pickup_Time", "Delivery_Time"]
        ].head(20).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("ANOMALY COUNTS")
    print("=" * 60)

    # Invalid agent ratings
    invalid_ratings = (
        df["Agent_Rating"].notna()
        & ((df["Agent_Rating"] < 1) | (df["Agent_Rating"] > 5))
    )

    print(f"\nInvalid agent ratings: {invalid_ratings.sum():,}")

    # Missing weather / traffic overlap
    missing_weather = df["Weather"].isna()
    missing_traffic = df["Traffic"].isna()

    print(f"Missing weather: {missing_weather.sum():,}")
    print(f"Missing traffic: {missing_traffic.sum():,}")
    print(
        "Missing both weather and traffic: "
        f"{(missing_weather & missing_traffic).sum():,}"
    )

    print(
        "Missing weather but traffic present: "
        f"{(missing_weather & ~missing_traffic).sum():,}"
    )

    print(
        "Missing traffic but weather present: "
        f"{(missing_traffic & ~missing_weather).sum():,}"
    )

    # Store coordinate anomalies
    invalid_store_coordinates = (
        (df["Store_Latitude"] <= 0)
        | (df["Store_Longitude"] <= 0)
    )

    print(
        f"\nRows with non-positive store coordinates: "
        f"{invalid_store_coordinates.sum():,}"
    )

    # Drop coordinate anomalies
    invalid_drop_coordinates = (
        (df["Drop_Latitude"] <= 0)
        | (df["Drop_Longitude"] <= 0)
    )

    print(
        f"Rows with non-positive drop coordinates: "
        f"{invalid_drop_coordinates.sum():,}"
    )

    # Date gaps
    dates = pd.to_datetime(df["Order_Date"]).dt.normalize()
    unique_dates = pd.Series(dates.unique()).sort_values()

    full_date_range = pd.date_range(
        start=unique_dates.min(),
        end=unique_dates.max(),
        freq="D"
    )

    missing_dates = full_date_range.difference(unique_dates)

    print(f"\nObserved calendar dates: {len(unique_dates):,}")
    print(f"Missing calendar dates: {len(missing_dates):,}")

    if len(missing_dates) > 0:
        print("Missing dates:")
        for date in missing_dates:
            print(f"  - {date.date()}")

        print("\n" + "=" * 60)
    print("STORE COORDINATE INVESTIGATION")
    print("=" * 60)

    invalid_store = (
        (df["Store_Latitude"] <= 0)
        | (df["Store_Longitude"] <= 0)
    )

    print("\nInvalid store-coordinate rows by Area:")
    print(
        df.loc[invalid_store, "Area"]
        .value_counts(dropna=False)
    )

    print("\nInvalid store-coordinate rows by Vehicle:")
    print(
        df.loc[invalid_store, "Vehicle"]
        .value_counts(dropna=False)
    )

    print("\nInvalid store-coordinate rows by Weather:")
    print(
        df.loc[invalid_store, "Weather"]
        .value_counts(dropna=False)
    )

    print("\nInvalid store-coordinate rows by Traffic:")
    print(
        df.loc[invalid_store, "Traffic"]
        .value_counts(dropna=False)
    )

    print("\nInvalid store-coordinate rows by Category:")
    print(
        df.loc[invalid_store, "Category"]
        .value_counts(dropna=False)
    )

    print("\nSample invalid store coordinates with other fields:")
    print(
        df.loc[
            invalid_store,
            [
                "Order_ID",
                "Store_Latitude",
                "Store_Longitude",
                "Drop_Latitude",
                "Drop_Longitude",
                "Area",
                "Vehicle",
                "Traffic",
                "Delivery_Time",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("GEOGRAPHIC DATA INVESTIGATION")
    print("=" * 60)

    zero_store = (
        (df["Store_Latitude"] == 0)
        & (df["Store_Longitude"] == 0)
    )

    negative_store = (
        (df["Store_Latitude"] < 0)
        | (df["Store_Longitude"] < 0)
    )

    print(f"\nStore coordinates exactly (0,0): {zero_store.sum():,}")
    print(f"Store coordinates with negative value: {negative_store.sum():,}")

    print("\nDrop-coordinate range for (0,0) stores:")
    print(
        df.loc[
            zero_store,
            ["Drop_Latitude", "Drop_Longitude"]
        ].describe()
    )

    print("\nExamples of negative store coordinates:")
    print(
        df.loc[
            negative_store,
            [
                "Store_Latitude",
                "Store_Longitude",
                "Drop_Latitude",
                "Drop_Longitude"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()