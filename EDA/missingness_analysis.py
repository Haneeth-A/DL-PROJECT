import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
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

OUTPUT = PROJECT_ROOT / "EDA" / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

# Research period
df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

target = "dem_energy_met_gwh"

# ==========================================
# 1. MISSINGNESS BY ENTITY
# ==========================================

entity_missing = df.groupby("entity").agg(
    total_records=(target, "size"),
    missing_demand=(target, lambda x: x.isna().sum()),
    available_demand=(target, lambda x: x.notna().sum())
)

entity_missing["missing_percentage"] = (
    entity_missing["missing_demand"] /
    entity_missing["total_records"] * 100
)

entity_missing = entity_missing.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nMISSINGNESS BY ENTITY")
print(entity_missing)

entity_missing.to_csv(
    OUTPUT / "demand_missingness_by_entity.csv"
)

# ==========================================
# 2. MISSINGNESS BY YEAR
# ==========================================

df["year"] = df["date"].dt.year

year_missing = df.groupby("year").agg(
    total_records=(target, "size"),
    missing_demand=(target, lambda x: x.isna().sum()),
    available_demand=(target, lambda x: x.notna().sum())
)

year_missing["missing_percentage"] = (
    year_missing["missing_demand"] /
    year_missing["total_records"] * 100
)

print("\nMISSINGNESS BY YEAR")
print(year_missing)

year_missing.to_csv(
    OUTPUT / "demand_missingness_by_year.csv"
)

# ==========================================
# 3. MISSINGNESS BY DATE
# ==========================================

date_missing = df.groupby("date").agg(
    entities_total=(target, "size"),
    entities_missing=(target, lambda x: x.isna().sum()),
    entities_available=(target, lambda x: x.notna().sum())
)

date_missing["missing_percentage"] = (
    date_missing["entities_missing"] /
    date_missing["entities_total"] * 100
)

date_missing = date_missing.sort_values(
    "entities_missing",
    ascending=False
)

print("\nDATES WITH MOST MISSING DEMAND")
print(date_missing.head(30))

date_missing.to_csv(
    OUTPUT / "demand_missingness_by_date.csv"
)

# ==========================================
# 4. SIMULTANEOUS MISSINGNESS
# ==========================================

simultaneous = (
    df.groupby("date")[target]
    .apply(lambda x: x.isna().sum())
)

print("\nSIMULTANEOUS MISSINGNESS")
print(simultaneous.value_counts().sort_index())

# ==========================================
# 5. MISSINGNESS BY SPLIT
# ==========================================

split_missing = df.groupby("split").agg(
    total_records=(target, "size"),
    missing_demand=(target, lambda x: x.isna().sum()),
    available_demand=(target, lambda x: x.notna().sum())
)

split_missing["missing_percentage"] = (
    split_missing["missing_demand"] /
    split_missing["total_records"] * 100
)

print("\nMISSINGNESS BY SPLIT")
print(split_missing)

split_missing.to_csv(
    OUTPUT / "demand_missingness_by_split.csv"
)

# ==========================================
# 6. MISSINGNESS RUNS
# ==========================================

missing_rows = df[df[target].isna()].copy()

print("\nTOTAL MISSING TARGET OBSERVATIONS:")
print(len(missing_rows))

print("\nMISSING OBSERVATIONS BY ENTITY:")
print(missing_rows["entity"].value_counts())

print("\nMISSINGNESS ANALYSIS COMPLETED")