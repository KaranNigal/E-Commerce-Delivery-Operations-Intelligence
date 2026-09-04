import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_FILE = (
    BASE_DIR
    / "data"
    / "synthetic"
    / "amazon_delivery_synthetic.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "dashboard"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dashboard_orders.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 50000
RANDOM_STATE = 42


# ============================================================
# CREATE DASHBOARD DATA
# ============================================================

print("=" * 60)
print("CREATING DASHBOARD DEPLOYMENT DATA")
print("=" * 60)

print("\nLoading source dataset...")

df = pd.read_csv(
    SOURCE_FILE,
    low_memory=False,
    keep_default_na=False,
)

print(f"Source rows: {len(df):,}")


# ------------------------------------------------------------
# RANDOM SAMPLE
# ------------------------------------------------------------

print(f"\nCreating representative sample of {SAMPLE_SIZE:,} orders...")

dashboard_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE,
).copy()


# ------------------------------------------------------------
# SORT BY DATE
# ------------------------------------------------------------

dashboard_df["Order_Date"] = pd.to_datetime(
    dashboard_df["Order_Date"]
)

dashboard_df = dashboard_df.sort_values(
    "Order_Date"
).reset_index(drop=True)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

dashboard_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nDashboard dataset created successfully.")

print(f"\nRows: {len(dashboard_df):,}")
print(f"Columns: {len(dashboard_df.columns)}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nFile size:")

size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

print(f"{size_mb:.2f} MB")

print("\n" + "=" * 60)
print("DASHBOARD DATA CREATION COMPLETED")
print("=" * 60)