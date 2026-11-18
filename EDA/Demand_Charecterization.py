import pandas as pd
import matplotlib.pyplot as plt
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

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Research period
df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

TARGET = "dem_energy_met_gwh"

# Remove missing target observations for demand analysis
df = df.dropna(subset=[TARGET])

# ============================================================
# 1. NATIONAL AGGREGATE DAILY DEMAND
# ============================================================

daily_total = df.groupby("date")[TARGET].sum()

plt.figure(figsize=(15, 6))
plt.plot(daily_total.index, daily_total.values)
plt.title("Aggregate Daily Energy Met Across Entities (2020–2025)")
plt.xlabel("Date")
plt.ylabel("Energy Met (GWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "aggregate_demand_timeseries.png", dpi=300)
plt.close()

# ============================================================
# 2. ANNUAL AGGREGATE DEMAND
# ============================================================

annual = daily_total.groupby(daily_total.index.year).agg(
    ["mean", "median", "std", "min", "max"]
)

annual.to_csv(OUTPUT_DIR / "annual_aggregate_demand.csv")

print("\nANNUAL AGGREGATE DEMAND:")
print(annual)

# ============================================================
# 3. ENTITY-LEVEL STATISTICS
# ============================================================

entity_stats = df.groupby("entity")[TARGET].agg(
    mean="mean",
    median="median",
    std="std",
    minimum="min",
    maximum="max"
)

entity_stats["coefficient_of_variation"] = (
    entity_stats["std"] / entity_stats["mean"]
)

entity_stats = entity_stats.sort_values("mean", ascending=False)

entity_stats.to_csv(OUTPUT_DIR / "entity_demand_statistics.csv")

print("\nENTITY DEMAND STATISTICS:")
print(entity_stats)

# ============================================================
# 4. ENTITY MEAN DEMAND
# ============================================================

plt.figure(figsize=(14, 8))
entity_stats["mean"].sort_values().plot(kind="barh")
plt.title("Average Daily Energy Met by Entity (2020–2025)")
plt.xlabel("Mean Daily Energy Met (GWh)")
plt.ylabel("Entity")
plt.tight_layout()
plt.savefig(PLOT_DIR / "entity_mean_demand.png", dpi=300)
plt.close()

# ============================================================
# 5. ENTITY DEMAND VARIABILITY
# ============================================================

plt.figure(figsize=(14, 8))
entity_stats["coefficient_of_variation"].sort_values().plot(kind="barh")
plt.title("Demand Variability Across Entities")
plt.xlabel("Coefficient of Variation")
plt.ylabel("Entity")
plt.tight_layout()
plt.savefig(PLOT_DIR / "entity_demand_variability.png", dpi=300)
plt.close()

# ============================================================
# 6. MONTHLY SEASONALITY
# ============================================================

df["month"] = df["date"].dt.month

monthly = df.groupby("month")[TARGET].mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly.values, marker="o")
plt.xticks(range(1, 13))
plt.title("Average Daily Energy Met by Month")
plt.xlabel("Month")
plt.ylabel("Mean Daily Energy Met (GWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "monthly_seasonality.png", dpi=300)
plt.close()

monthly.to_csv(OUTPUT_DIR / "monthly_demand_profile.csv")

print("\nMONTHLY DEMAND PROFILE:")
print(monthly)

# ============================================================
# 7. DAY-OF-WEEK PATTERN
# ============================================================

df["day_of_week"] = df["date"].dt.dayofweek

dow = df.groupby("day_of_week")[TARGET].mean()

plt.figure(figsize=(10, 6))
plt.plot(dow.index, dow.values, marker="o")
plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.title("Average Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Mean Daily Energy Met (GWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "day_of_week_pattern.png", dpi=300)
plt.close()

dow.to_csv(OUTPUT_DIR / "day_of_week_demand.csv")

# ============================================================
# 8. ENTITY NORMALIZED MONTHLY PROFILE
# ============================================================

monthly_entity = df.groupby(["entity", "month"])[TARGET].mean().reset_index()

monthly_entity.to_csv(
    OUTPUT_DIR / "entity_monthly_demand.csv",
    index=False
)

# ============================================================
# 9. ENTITY NORMALIZED DAILY PROFILE
# ============================================================

df["day_of_year"] = df["date"].dt.dayofyear

entity_daily_normalized = df.copy()

entity_daily_normalized["normalized_demand"] = (
    entity_daily_normalized.groupby("entity")[TARGET]
    .transform(lambda x: x / x.mean())
)

plt.figure(figsize=(15, 6))

for entity, group in entity_daily_normalized.groupby("entity"):
    yearly_profile = group.groupby("day_of_year")["normalized_demand"].mean()
    plt.plot(yearly_profile.index, yearly_profile.values, alpha=0.5)

plt.title("Normalized Annual Demand Profiles Across Entities")
plt.xlabel("Day of Year")
plt.ylabel("Demand / Entity Mean")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "normalized_annual_profiles.png", dpi=300)
plt.close()

# ============================================================
# 10. SAVE CLEAN ANALYSIS DATA
# ============================================================

daily_total.to_csv(OUTPUT_DIR / "aggregate_daily_demand.csv")

print("\nDEMAND CHARACTERIZATION COMPLETED")
print(f"Outputs saved to: {OUTPUT_DIR}")