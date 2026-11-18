import pandas as pd
import numpy as np
from pathlib import Path

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

print("Dataset path:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())

OUTPUT = Path("EDA/outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

print("=" * 60)
print("RE-FUSED DATASET AUDIT")
print("=" * 60)

# ==========================================
# 2. BASIC STRUCTURE
# ==========================================

print("\n--- DATASET STRUCTURE ---")

print("Shape:", df.shape)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nUnique entities:", df["entity"].nunique())
print("Entities:", sorted(df["entity"].unique()))

print("\nUnique regions:")
print(df["region"].unique())

# ==========================================
# 3. DUPLICATE CHECK
# ==========================================

print("\n--- DUPLICATE CHECK ---")

duplicates = df.duplicated(
    subset=["entity", "date"]
).sum()

print("Duplicate entity-date records:", duplicates)

# ==========================================
# 4. TEMPORAL COVERAGE
# ==========================================

print("\n--- TEMPORAL COVERAGE ---")

coverage = df.groupby("entity").agg(
    first_date=("date", "min"),
    last_date=("date", "max"),
    observations=("date", "count"),
    unique_days=("date", "nunique")
)

coverage["expected_days"] = (
    coverage["last_date"] - coverage["first_date"]
).dt.days + 1

coverage["missing_calendar_days"] = (
    coverage["expected_days"] - coverage["unique_days"]
)

print(coverage)

coverage.to_csv(
    OUTPUT / "entity_coverage.csv"
)

# ==========================================
# 5. SPLIT DISTRIBUTION
# ==========================================

print("\n--- SPLIT DISTRIBUTION ---")

print(df["split"].value_counts())

split_dates = df.groupby("split").agg(
    start_date=("date", "min"),
    end_date=("date", "max"),
    observations=("date", "count")
)

print(split_dates)

split_dates.to_csv(
    OUTPUT / "split_distribution.csv"
)

# ==========================================
# 6. MISSING VALUE ANALYSIS
# ==========================================

print("\n--- MISSING VALUE ANALYSIS ---")

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": df.isna().mean() * 100
})

missing = missing.sort_values(
    "missing_percentage",
    ascending=False
)

print(missing.to_string())

missing.to_csv(
    OUTPUT / "missing_values.csv"
)

# ==========================================
# 7. DEMAND STATISTICS
# ==========================================

print("\n--- DEMAND STATISTICS ---")

target = "dem_energy_met_gwh"

print(df[target].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50,
                 0.75, 0.95, 0.99]
))

state_stats = df.groupby("entity")[target].agg(
    ["count", "mean", "std", "min", "max", "median"]
)

state_stats["coefficient_of_variation"] = (
    state_stats["std"] / state_stats["mean"]
)

print("\nState-wise demand statistics:")
print(state_stats)

state_stats.to_csv(
    OUTPUT / "state_demand_statistics.csv"
)

# ==========================================
# 8. NEGATIVE VALUES
# ==========================================

print("\n--- NEGATIVE VALUE CHECK ---")

numeric = df.select_dtypes(include="number")

negative_counts = (
    numeric.lt(0).sum()
)

negative_counts = negative_counts[
    negative_counts > 0
]

print(negative_counts)

negative_counts.to_csv(
    OUTPUT / "negative_values.csv"
)

# ==========================================
# 9. PERIOD OF INTEREST
# ==========================================

print("\n--- RESEARCH PERIOD: 2020-2025 ---")

research = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

print("Research period shape:", research.shape)

print("\nObservations by split:")
print(research["split"].value_counts())

print("\nObservations by year:")
print(research["date"].dt.year.value_counts().sort_index())

# ==========================================
# 10. SAVE RESEARCH SUBSET
# ==========================================

research.to_parquet(
    OUTPUT / "research_panel_2020_2025.parquet",
    index=False
)

print("\nAudit completed.")
print("Results saved in:", OUTPUT)