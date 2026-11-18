from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "dem_energy_met_gwh"

TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["entity", "date"]).copy()

# =========================================================
# GENERATE PERSISTENCE FORECAST
# =========================================================

df["forecast"] = df.groupby("entity")[TARGET].shift(1)

df["error"] = df[TARGET] - df["forecast"]

df["absolute_error"] = df["error"].abs()

df["squared_error"] = df["error"] ** 2

df["percentage_error"] = (
    df["absolute_error"] / df[TARGET].abs()
) * 100

df["smape"] = (
    2 * df["absolute_error"] /
    (df[TARGET].abs() + df["forecast"].abs())
) * 100

# =========================================================
# TEST PERIOD
# =========================================================

test_df = df[
    (df["date"] >= TEST_START) &
    (df["date"] <= TEST_END)
].copy()

test_df = test_df[
    test_df[TARGET].notna() &
    test_df["forecast"].notna()
].copy()

# =========================================================
# CALENDAR FEATURES
# =========================================================

test_df["month"] = test_df["date"].dt.month

test_df["month_name"] = test_df["date"].dt.strftime("%B")

test_df["day_of_week"] = test_df["date"].dt.dayofweek

test_df["day_name"] = test_df["date"].dt.day_name()

test_df["quarter"] = test_df["date"].dt.quarter

# =========================================================
# DEMAND CHANGE FEATURES
# =========================================================

test_df["daily_change"] = (
    test_df[TARGET] - test_df["forecast"]
)

test_df["relative_change"] = (
    test_df["daily_change"] / test_df["forecast"].abs()
) * 100

# =========================================================
# 1. MONTHLY ERROR ANALYSIS
# =========================================================

monthly_results = test_df.groupby("month").agg(
    mean_absolute_error=("absolute_error", "mean"),
    mean_rmse=("squared_error", lambda x: np.sqrt(x.mean())),
    mean_smape=("smape", "mean"),
    mean_demand=("dem_energy_met_gwh", "mean"),
    sample_count=("absolute_error", "count")
)

monthly_results.to_csv(
    OUTPUT_DIR / "baseline_monthly_error.csv"
)

print("\nMONTHLY ERROR ANALYSIS:")
print(monthly_results)

# =========================================================
# 2. DAY-OF-WEEK ERROR ANALYSIS
# =========================================================

weekday_results = test_df.groupby("day_name").agg(
    mean_absolute_error=("absolute_error", "mean"),
    mean_smape=("smape", "mean"),
    mean_demand=("dem_energy_met_gwh", "mean"),
    sample_count=("absolute_error", "count")
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_results = weekday_results.reindex(weekday_order)

weekday_results.to_csv(
    OUTPUT_DIR / "baseline_weekday_error.csv"
)

print("\nDAY-OF-WEEK ERROR ANALYSIS:")
print(weekday_results)

# =========================================================
# 3. ENTITY ERROR ANALYSIS
# =========================================================

entity_results = test_df.groupby("entity").agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    mean_sMAPE=("smape", "mean"),
    mean_demand=("dem_energy_met_gwh", "mean"),
    mean_daily_change=("daily_change", "mean"),
    mean_absolute_change=("absolute_error", "mean"),
    sample_count=("absolute_error", "count")
)

entity_results = entity_results.sort_values(
    "mean_sMAPE",
    ascending=False
)

entity_results.to_csv(
    OUTPUT_DIR / "baseline_entity_error_analysis.csv"
)

print("\nENTITY ERROR ANALYSIS:")
print(entity_results)

# =========================================================
# 4. EXTREME ERROR DAYS
# =========================================================

worst_days = test_df.sort_values(
    "absolute_error",
    ascending=False
).head(100)

worst_days.to_csv(
    OUTPUT_DIR / "baseline_worst_100_forecasts.csv",
    index=False
)

print("\nTOP 20 WORST FORECASTS:")
print(
    worst_days[
        [
            "date",
            "entity",
            TARGET,
            "forecast",
            "error",
            "absolute_error",
            "smape"
        ]
    ].head(20)
)

# =========================================================
# 5. SUDDEN DEMAND CHANGES
# =========================================================

change_analysis = test_df.groupby("entity").agg(
    mean_change=("daily_change", "mean"),
    std_change=("daily_change", "std"),
    mean_absolute_change=("absolute_error", "mean"),
    mean_sMAPE=("smape", "mean")
)

change_analysis.to_csv(
    OUTPUT_DIR / "baseline_change_analysis.csv"
)

# =========================================================
# 6. CORRELATION BETWEEN CHANGE AND FORECAST ERROR
# =========================================================

correlation = test_df[
    ["absolute_error", "smape", "daily_change", "relative_change"]
].corr()

correlation.to_csv(
    OUTPUT_DIR / "baseline_error_change_correlation.csv"
)

print("\nERROR / CHANGE CORRELATION:")
print(correlation)

# =========================================================
# 7. LARGE DEMAND CHANGE EVENTS
# =========================================================

large_changes = test_df.copy()

large_changes["absolute_relative_change"] = (
    large_changes["relative_change"].abs()
)

large_changes = large_changes.sort_values(
    "absolute_relative_change",
    ascending=False
)

large_changes.head(100).to_csv(
    OUTPUT_DIR / "baseline_large_demand_changes.csv",
    index=False
)

print("\nTOP 20 LARGE RELATIVE DEMAND CHANGES:")
print(
    large_changes[
        [
            "date",
            "entity",
            TARGET,
            "forecast",
            "daily_change",
            "relative_change"
        ]
    ].head(20)
)

# =========================================================
# 8. MONTHLY MAE VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_results.index,
    monthly_results["mean_absolute_error"],
    marker="o"
)

plt.xticks(range(1, 13))

plt.xlabel("Month")
plt.ylabel("Mean Absolute Error (GWh)")

plt.title("Persistence Forecast - Monthly MAE (2025)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_monthly_mae.png",
    dpi=300
)

plt.close()

# =========================================================
# 9. MONTHLY sMAPE VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_results.index,
    monthly_results["mean_smape"],
    marker="o"
)

plt.xticks(range(1, 13))

plt.xlabel("Month")
plt.ylabel("sMAPE (%)")

plt.title("Persistence Forecast - Monthly sMAPE (2025)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_monthly_smape.png",
    dpi=300
)

plt.close()

# =========================================================
# 10. ERROR VS DEMAND CHANGE
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    test_df["relative_change"],
    test_df["smape"],
    alpha=0.5
)

plt.xlabel("Relative Daily Demand Change (%)")

plt.ylabel("Forecast sMAPE (%)")

plt.title("Demand Change vs Persistence Forecast Error")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_change_vs_error.png",
    dpi=300
)

plt.close()

# =========================================================
# 11. ABSOLUTE ERROR DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    test_df["absolute_error"],
    bins=50
)

plt.xlabel("Absolute Error (GWh)")

plt.ylabel("Frequency")

plt.title("Distribution of Persistence Forecast Errors")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_error_distribution.png",
    dpi=300
)

plt.close()

# =========================================================
# FINAL
# =========================================================

print("\nBASELINE ERROR ANALYSIS COMPLETED")

print(f"Outputs saved to: {OUTPUT_DIR}")