from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATA = (
    PROJECT_ROOT
    / "Data"
    / "Raw"
    / "RE-FUSED_India_Datasets_v1.0"
    / "RE-FUSED"
    / "India_Grid_Study"
    / "data"
    / "processed"
    / "refused_state_day.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "dem_energy_met_gwh"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# ============================================================
# HELPERS
# ============================================================

def sha256_file(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def classify_column(name):
    name_lower = name.lower()

    if name == TARGET_COLUMN:
        return "Target: daily energy demand"

    if "wx_tmin" in name_lower:
        return "Weather: minimum temperature"

    if "wx_tmax" in name_lower:
        return "Weather: maximum temperature"

    if "wx_tmean" in name_lower:
        return "Weather: mean temperature"

    if "wx_rain" in name_lower:
        return "Weather: rainfall"

    if "wx_humidity" in name_lower or "humid" in name_lower:
        return "Weather: humidity"

    if "wx_cdd" in name_lower:
        return "Weather: cooling degree days"

    if "wx_hdd" in name_lower:
        return "Weather: heating degree days"

    if any(x in name_lower for x in ["flow", "interchange", "tie_line"]):
        return "Potential inter-state flow"

    if any(x in name_lower for x in ["date", "day"]):
        return "Temporal identifier"

    if any(x in name_lower for x in ["state", "entity", "region"]):
        return "Entity identifier"

    if "dem_" in name_lower:
        return "Demand-related variable"

    if "gen_" in name_lower:
        return "Generation-related variable"

    if "price" in name_lower:
        return "Price-related variable"

    if "dev" in name_lower:
        return "Deviation-related variable"

    if "re_" in name_lower or "renew" in name_lower:
        return "Renewable energy variable"

    return "Other / requires investigation"


def assign_unit(name):
    name_lower = name.lower()

    # Units explicitly inferable from column naming conventions
    if name == TARGET_COLUMN:
        return "GWh (dataset naming; verify source documentation)"

    if "wx_tmin" in name_lower or "wx_tmax" in name_lower or "wx_tmean" in name_lower:
        return "°C (dataset naming; verify source documentation)"

    if "wx_rain" in name_lower:
        return "mm (dataset naming; verify source documentation)"

    if "wx_cdd" in name_lower or "wx_hdd" in name_lower:
        return "Degree-day metric (definition requires verification)"

    if "dem_max_demand_mw" in name_lower:
        return "MW (dataset naming)"

    if "mw" in name_lower:
        return "MW (inferred from column name)"

    if "gwh" in name_lower:
        return "GWh (inferred from column name)"

    if "mu" in name_lower:
        return "MU (inferred from column name; verification required)"

    if "price" in name_lower:
        return "Requires source verification"

    if any(x in name_lower for x in ["flow", "interchange", "tie_line"]):
        return "Requires source verification"

    return "Unknown / source documentation required"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING MODELING PANEL")
print("=" * 70)

if not SOURCE_DATA.exists():
    raise FileNotFoundError(f"Source dataset not found:\n{SOURCE_DATA}")

print("\nLoading source dataset...")
df = pd.read_parquet(SOURCE_DATA)

print(f"Dataset shape: {df.shape}")

# ============================================================
# IDENTIFY KEY COLUMNS
# ============================================================

print("\nIdentifying key columns...")

date_candidates = ["date", "Date", "day", "datetime"]
entity_candidates = ["entity", "state", "state_code", "region", "control_area"]

DATE_COLUMN = next(
    (col for col in date_candidates if col in df.columns),
    None
)

ENTITY_COLUMN = next(
    (col for col in entity_candidates if col in df.columns),
    None
)

if DATE_COLUMN is None:
    raise ValueError("Could not identify date column.")

if ENTITY_COLUMN is None:
    raise ValueError("Could not identify entity column.")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column missing: {TARGET_COLUMN}")

print(f"Date column: {DATE_COLUMN}")
print(f"Entity column: {ENTITY_COLUMN}")
print(f"Target column: {TARGET_COLUMN}")

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

# ============================================================
# RAW DATA INTEGRITY
# ============================================================

print("\nChecking duplicate entity-date rows...")

duplicate_count = df.duplicated(
    subset=[ENTITY_COLUMN, DATE_COLUMN]
).sum()

print(f"Duplicate entity-date rows: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate entity-date records detected. "
        "Panel construction halted."
    )

# ============================================================
# EXACT ENTITY LIST
# ============================================================

entities = sorted(df[ENTITY_COLUMN].dropna().unique())

entity_df = pd.DataFrame({
    "entity_id": entities
})

entity_df["entity_count"] = len(entities)

entity_df.to_csv(
    OUTPUT_DIR / "entity_list.csv",
    index=False
)

print(f"\nEntities identified: {len(entities)}")

# ============================================================
# DATE RANGE AND PANEL EXTRACTION
# ============================================================

print("\nExtracting 2020–2025 panel...")

panel = df[
    (df[DATE_COLUMN] >= START_DATE)
    & (df[DATE_COLUMN] <= END_DATE)
].copy()

panel = panel.sort_values(
    [ENTITY_COLUMN, DATE_COLUMN]
).reset_index(drop=True)

print(f"Panel shape: {panel.shape}")
print(f"Date range: {panel[DATE_COLUMN].min()} to {panel[DATE_COLUMN].max()}")

# ============================================================
# CALENDAR COVERAGE
# ============================================================

print("\nChecking calendar coverage...")

coverage_records = []

for entity, group in panel.groupby(ENTITY_COLUMN):

    dates = pd.to_datetime(group[DATE_COLUMN])

    expected_dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(dates.unique())

    missing_dates = expected_dates.difference(actual_dates)

    coverage_records.append({
        "entity": entity,
        "rows": len(group),
        "unique_dates": dates.nunique(),
        "expected_dates": len(expected_dates),
        "missing_calendar_dates": len(missing_dates),
        "first_date": dates.min(),
        "last_date": dates.max()
    })

coverage_df = pd.DataFrame(coverage_records)

coverage_df.to_csv(
    OUTPUT_DIR / "calendar_coverage.csv",
    index=False
)

# ============================================================
# COLUMN MAPPING
# ============================================================

print("\nGenerating column mapping...")

column_mapping = pd.DataFrame({
    "column": panel.columns,
    "category": [
        classify_column(col)
        for col in panel.columns
    ]
})

column_mapping.to_csv(
    OUTPUT_DIR / "column_mapping.csv",
    index=False
)

# ============================================================
# UNIT MAPPING
# ============================================================

print("\nGenerating unit mapping...")

units_mapping = pd.DataFrame({
    "column": panel.columns,
    "unit": [
        assign_unit(col)
        for col in panel.columns
    ]
})

units_mapping.to_csv(
    OUTPUT_DIR / "units_mapping.csv",
    index=False
)

# ============================================================
# FLOW VARIABLE INVESTIGATION
# ============================================================

print("\nInvestigating possible flow variables...")

flow_keywords = [
    "flow",
    "interchange",
    "tie_line",
    "transfer",
    "exchange",
    "import",
    "export"
]

flow_columns = [
    col for col in panel.columns
    if any(keyword in col.lower() for keyword in flow_keywords)
]

flow_df = pd.DataFrame({
    "possible_flow_column": flow_columns
})

flow_df.to_csv(
    OUTPUT_DIR / "flow_variable_investigation.csv",
    index=False
)

if flow_columns:
    print("Potential flow columns found:")
    for col in flow_columns:
        print(f"  - {col}")
else:
    print("No flow columns identified through name-based search.")

# ============================================================
# MISSINGNESS
# ============================================================

print("\nGenerating missingness report...")

missingness_records = []

for col in panel.columns:

    missing_count = panel[col].isna().sum()

    missingness_records.append({
        "column": col,
        "missing_count": missing_count,
        "total_count": len(panel),
        "missing_percentage": (
            missing_count / len(panel) * 100
            if len(panel) > 0 else 0
        )
    })

missingness_df = pd.DataFrame(missingness_records)

missingness_df.to_csv(
    OUTPUT_DIR / "missingness_report.csv",
    index=False
)

# ============================================================
# TARGET MISSINGNESS BY ENTITY
# ============================================================

print("\nGenerating target missingness by entity...")

target_missingness = []

for entity, group in panel.groupby(ENTITY_COLUMN):

    missing = group[TARGET_COLUMN].isna().sum()

    target_missingness.append({
        "entity": entity,
        "target_missing": missing,
        "total_rows": len(group),
        "missing_percentage": (
            missing / len(group) * 100
            if len(group) > 0 else 0
        )
    })

target_missingness_df = pd.DataFrame(target_missingness)

target_missingness_df.to_csv(
    OUTPUT_DIR / "target_missingness_by_entity.csv",
    index=False
)

# ============================================================
# INTERPOLATION POLICY
# ============================================================

interpolation_policy = {
    "demand": "No interpolation performed",
    "weather": "No interpolation performed",
    "other_variables": "No interpolation performed",
    "missing_values": "Preserved as missing",
    "future_modeling_policy": (
        "Imputation or missing-window handling must be implemented "
        "inside the training pipeline using leakage-safe rules."
    )
}

with open(
    OUTPUT_DIR / "interpolation_policy.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        interpolation_policy,
        f,
        indent=4
    )

# ============================================================
# BUILD FINAL PANEL
# ============================================================

print("\nSaving panel...")

panel.to_parquet(
    OUTPUT_DIR / "panel.parquet",
    index=False
)

# ============================================================
# SHA256 OF SOURCE DATA
# ============================================================

print("\nCalculating SHA256 of source dataset...")

source_hash = sha256_file(SOURCE_DATA)

with open(
    OUTPUT_DIR / "source_sha256.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(f"File: {SOURCE_DATA}\n")
    f.write(f"SHA256: {source_hash}\n")

# ============================================================
# METADATA
# ============================================================

metadata = {
    "project": "DL Project",
    "dataset_source": str(SOURCE_DATA),
    "panel_file": str(OUTPUT_DIR / "panel.parquet"),
    "created_at": datetime.now().isoformat(),
    "date_column": DATE_COLUMN,
    "entity_column": ENTITY_COLUMN,
    "target_column": TARGET_COLUMN,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "entity_count": len(entities),
    "panel_rows": len(panel),
    "panel_columns": len(panel.columns),
    "source_sha256": source_hash,
    "interpolation": "None performed",
    "flow_columns": flow_columns
}

with open(
    OUTPUT_DIR / "panel_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODELING PANEL BUILD COMPLETE")
print("=" * 70)

print(f"Panel rows: {len(panel)}")
print(f"Panel columns: {len(panel.columns)}")
print(f"Entity count: {len(entities)}")
print(f"Target: {TARGET_COLUMN}")
print(f"Date range: {START_DATE} to {END_DATE}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {file.name}")

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print("""
No missing values have been interpolated.

Column units are inferred from column naming where possible,
but must be verified against the dataset documentation.

Potential flow columns are identified using column-name keywords.
This is not proof that actual inter-state flow data exist.

Panel construction is complete.
""")