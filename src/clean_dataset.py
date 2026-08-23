from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_delivery.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "amazon_delivery_clean.csv"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning rules to the raw delivery dataset."""

    cleaned = df.copy()

    # ---------------------------------------------------------
    # 1. Normalize whitespace in text columns
    # ---------------------------------------------------------
    for column in cleaned.select_dtypes(include=["object", "str"]).columns:
        cleaned[column] = cleaned[column].str.strip()

    # ---------------------------------------------------------
    # 2. Normalize invalid agent ratings
    # ---------------------------------------------------------
    invalid_rating = (
        (cleaned["Agent_Rating"] < 1)
        | (cleaned["Agent_Rating"] > 5)
    )

    cleaned.loc[invalid_rating, "Agent_Rating"] = pd.NA

    # Fill missing/invalid ratings using the dataset median.
    rating_median = cleaned["Agent_Rating"].median()

    cleaned["Agent_Rating"] = cleaned["Agent_Rating"].fillna(
        rating_median
    )

    # ---------------------------------------------------------
    # 3. Handle missing weather and traffic
    # ---------------------------------------------------------
    cleaned["Weather"] = cleaned["Weather"].fillna("Unknown")
    cleaned["Traffic"] = cleaned["Traffic"].fillna("Unknown")

    # ---------------------------------------------------------
    # 4. Handle invalid store coordinates
    #
    # Invalid if either latitude or longitude is <= 0.
    # We preserve the order but mark the unavailable
    # geographic information as missing.
    # ---------------------------------------------------------
    invalid_store_coordinates = (
        (cleaned["Store_Latitude"] <= 0)
        | (cleaned["Store_Longitude"] <= 0)
    )

    cleaned.loc[
        invalid_store_coordinates,
        ["Store_Latitude", "Store_Longitude"]
    ] = pd.NA

    # ---------------------------------------------------------
    # 5. Convert date and time fields
    # ---------------------------------------------------------
    cleaned["Order_Date"] = pd.to_datetime(
        cleaned["Order_Date"],
        errors="coerce"
    )

    cleaned["Order_Time"] = pd.to_datetime(
        cleaned["Order_Time"],
        format="%H:%M:%S",
        errors="coerce"
    ).dt.time

    cleaned["Pickup_Time"] = pd.to_datetime(
        cleaned["Pickup_Time"],
        format="%H:%M:%S",
        errors="coerce"
    ).dt.time

    return cleaned


def main():
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    print("\nReading raw dataset:")
    print(RAW_DATA_PATH)

    # ---------------------------------------------------------
    # Read raw dataset
    #
    # The source CSV uses values such as "NaN" and "NaN "
    # to represent missing data.
    # ---------------------------------------------------------
    df = pd.read_csv(
        RAW_DATA_PATH,
        na_values=["NaN", "NaN "],
        keep_default_na=True,
    )

    # Normalize whitespace before cleaning.
    for column in df.select_dtypes(include=["object", "str"]).columns:
        df[column] = df[column].str.strip()

    print(f"\nRaw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Apply cleaning
    # ---------------------------------------------------------
    cleaned_df = clean_dataset(df)

    # ---------------------------------------------------------
    # Save processed dataset
    # ---------------------------------------------------------
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Validation summary
    # ---------------------------------------------------------
    print(f"\nCleaned rows: {len(cleaned_df):,}")
    print(f"Cleaned columns: {len(cleaned_df.columns)}")

    print("\nSaved cleaned dataset:")
    print(PROCESSED_DATA_PATH)

    print("\nRemaining missing values:")
    print(cleaned_df.isna().sum())

    print("\nRow-count validation:")
    if len(df) == len(cleaned_df):
        print("PASS: No rows were removed during cleaning.")
    else:
        print("FAIL: Row count changed during cleaning.")

    print("\nOrder_ID uniqueness validation:")
    if cleaned_df["Order_ID"].nunique() == len(cleaned_df):
        print("PASS: Order_ID remains unique.")
    else:
        print("FAIL: Duplicate Order_ID values detected.")


if __name__ == "__main__":
    main()