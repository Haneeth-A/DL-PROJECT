from pathlib import Path
import pandas as pd
import hashlib

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel.parquet"
RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD PANEL
# ==========================================================

print("=" * 75)
print("TIER 1 FINAL VERIFICATION")
print("=" * 75)

panel = pd.read_parquet(PANEL_PATH)

print("\nPANEL")
print(f"Shape: {panel.shape}")
print(f"Columns: {len(panel.columns)}")

# ==========================================================
# EXACT COLUMN INVENTORY
# ==========================================================

print("\nALL COLUMN NAMES")

for col in panel.columns:
    print(col)

column_inventory = pd.DataFrame({
    "column": panel.columns,
    "dtype": [str(panel[col].dtype) for col in panel.columns]
})

column_inventory.to_csv(
    OUTPUT_DIR / "final_column_inventory.csv",
    index=False
)

# ==========================================================
# WEATHER AND FLOW INVESTIGATION
# ==========================================================

weather_keywords = [
    "tmin", "tmax", "tmean", "temp",
    "rain", "humid", "cdd", "hdd"
]

flow_keywords = [
    "flow", "interchange", "tie_line",
    "transfer", "import", "export", "exchange"
]

weather_cols = [
    col for col in panel.columns
    if any(k in col.lower() for k in weather_keywords)
]

flow_cols = [
    col for col in panel.columns
    if any(k in col.lower() for k in flow_keywords)
]

print("\nWEATHER COLUMNS")

for col in weather_cols:
    print(col)

print("\nFLOW COLUMNS")

if flow_cols:
    for col in flow_cols:
        print(col)
else:
    print("No columns matching flow-related keywords.")

# ==========================================================
# EXACT ENTITY LIST
# ==========================================================

entities = sorted(panel["entity"].dropna().unique())

print(f"\nENTITY COUNT: {len(entities)}")

print("\nENTITY LIST")

for entity in entities:
    print(entity)

# ==========================================================
# DATE COVERAGE
# ==========================================================

panel["date"] = pd.to_datetime(panel["date"])

print("\nDATE COVERAGE")
print(f"Start: {panel['date'].min()}")
print(f"End:   {panel['date'].max()}")

expected_dates = pd.date_range(
    "2020-01-01",
    "2025-12-31",
    freq="D"
)

coverage_records = []

for entity, group in panel.groupby("entity"):

    dates = pd.DatetimeIndex(group["date"].unique())

    missing_dates = expected_dates.difference(dates)

    coverage_records.append({
        "entity": entity,
        "rows": len(group),
        "unique_dates": len(dates),
        "expected_dates": len(expected_dates),
        "missing_dates": len(missing_dates),
        "first_date": dates.min(),
        "last_date": dates.max()
    })

coverage = pd.DataFrame(coverage_records)

coverage.to_csv(
    OUTPUT_DIR / "final_calendar_verification.csv",
    index=False
)

print("\nCALENDAR VERIFICATION")
print(coverage.to_string(index=False))

# ==========================================================
# TARGET
# ==========================================================

TARGET = "dem_energy_met_gwh"

print("\nTARGET VERIFICATION")
print(f"Target: {TARGET}")
print(f"Missing target values: {panel[TARGET].isna().sum()}")
print(f"Target missing %: {panel[TARGET].isna().mean() * 100:.4f}%")

# ==========================================================
# RAW ZIP HASH
# ==========================================================

print("\nRAW ZIP HASH INVESTIGATION")

zip_files = list(RAW_DIR.rglob("*.zip"))

if zip_files:

    for zip_file in zip_files:

        sha256 = hashlib.sha256()

        with open(zip_file, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)

        digest = sha256.hexdigest()

        print(f"\nZIP: {zip_file}")
        print(f"SHA256: {digest}")

else:

    print("No ZIP file found inside Data/Raw.")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 75)
print("FINAL VERIFICATION SUMMARY")
print("=" * 75)

print(f"Panel shape: {panel.shape}")
print(f"Entities: {len(entities)}")
print(f"Weather columns: {len(weather_cols)}")
print(f"Flow columns: {len(flow_cols)}")
print(f"Expected days/entity: {len(expected_dates)}")

print("\nVerification files saved to:")
print(OUTPUT_DIR)

print("\nVERIFICATION COMPLETE.")