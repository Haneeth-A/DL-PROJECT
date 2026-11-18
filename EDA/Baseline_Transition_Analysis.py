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
# LOAD AND STRUCTURE
# =========================================================

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["entity", "date"]).copy()

# Enforce calendar-day continuity per entity
# Missing dates become NaN rather than being silently skipped.

structured = []

for entity, group in df.groupby("entity"):

    group = group.set_index("date").sort_index()

    full_dates = pd.date_range(
        group.index.min(),
        group.index.max(),
        freq="D"
    )

    group = group.reindex(full_dates)

    group["entity"] = entity

    group.index.name = "date"

    structured.append(group)

df = pd.concat(structured).reset_index()

# =========================================================
# GENERATE PERSISTENCE FORECAST
# =========================================================

df["forecast"] = df.groupby("entity")[TARGET].shift(1)

df["error"] = df[TARGET] - df["forecast"]

df["absolute_error"] = df["error"].abs()

df["squared_error"] = df["error"] ** 2

denominator = df[TARGET].abs() + df["forecast"].abs()

df["smape"] = np.where(
    denominator == 0,
    0,
    2 * df["absolute_error"] / denominator * 100
)

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
# TRANSITION FEATURES
# =========================================================

test_df["target_day"] = test_df["date"].dt.day_name()

test_df["previous_day"] = (
    test_df["date"] - pd.Timedelta(days=1)
).dt.day_name()

test_df["transition"] = (
    test_df["previous_day"] + " -> " +
    test_df["target_day"]
)

test_df["target_weekday"] = test_df["date"].dt.day_of_week

test_df["absolute_relative_change"] = (
    test_df["absolute_error"] /
    test_df["forecast"].abs()
) * 100

# =========================================================
# 1. TRANSITION ANALYSIS
# =========================================================

transition_results = test_df.groupby("transition").agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    sMAPE=("smape", "mean"),
    mean_absolute_change=("absolute_error", "mean"),
    sample_count=("absolute_error", "count")
)

transition_results = transition_results.sort_values(
    "sMAPE",
    ascending=False
)

transition_results.to_csv(
    OUTPUT_DIR / "baseline_transition_results.csv"
)

print("\nTRANSITION PERFORMANCE:")
print(transition_results)

# =========================================================
# 2. WEEKEND VS WEEKDAY
# =========================================================

test_df["day_category"] = np.where(
    test_df["target_weekday"] >= 5,
    "Weekend",
    "Weekday"
)

weekend_results = test_df.groupby("day_category").agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    sMAPE=("smape", "mean"),
    sample_count=("absolute_error", "count")
)

weekend_results.to_csv(
    OUTPUT_DIR / "baseline_weekend_weekday_results.csv"
)

print("\nWEEKEND VS WEEKDAY:")
print(weekend_results)

# =========================================================
# 3. SUNDAY AND MONDAY ANALYSIS
# =========================================================

sunday_monday = test_df[
    test_df["target_day"].isin(["Sunday", "Monday"])
]

sm_results = sunday_monday.groupby("target_day").agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    sMAPE=("smape", "mean"),
    sample_count=("absolute_error", "count")
)

sm_results.to_csv(
    OUTPUT_DIR / "baseline_sunday_monday_results.csv"
)

print("\nSUNDAY AND MONDAY:")
print(sm_results)

# =========================================================
# 4. LARGE ABSOLUTE CHANGES
# =========================================================

# Determine unusual changes within each entity using its own
# distribution of absolute changes in the 2025 evaluation data.

test_df["large_change_threshold"] = (
    test_df.groupby("entity")["absolute_error"]
    .transform(lambda x: x.quantile(0.90))
)

test_df["change_category"] = np.where(
    test_df["absolute_error"] >=
    test_df["large_change_threshold"],
    "Large change",
    "Ordinary change"
)

change_category_results = test_df.groupby(
    "change_category"
).agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    sMAPE=("smape", "mean"),
    sample_count=("absolute_error", "count")
)

change_category_results.to_csv(
    OUTPUT_DIR / "baseline_large_change_comparison.csv"
)

print("\nLARGE CHANGE VS ORDINARY CHANGE:")
print(change_category_results)

# =========================================================
# 5. ENTITY SIZE GROUPING
# =========================================================

# Entity size is determined using its average 2025 demand.

entity_sizes = (
    test_df.groupby("entity")[TARGET]
    .mean()
    .sort_values()
)

size_groups = pd.qcut(
    entity_sizes,
    q=4,
    labels=["Small", "Medium-Small", "Medium-Large", "Large"]
)

size_map = size_groups.to_dict()

test_df["size_group"] = test_df["entity"].map(size_map)

size_results = test_df.groupby("size_group", observed=True).agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    sMAPE=("smape", "mean"),
    sample_count=("absolute_error", "count")
)

size_results.to_csv(
    OUTPUT_DIR / "baseline_size_group_results.csv"
)

print("\nENTITY SIZE GROUP PERFORMANCE:")
print(size_results)

# =========================================================
# 6. LARGE CHANGE PERFORMANCE BY SIZE GROUP
# =========================================================

size_change_results = test_df.groupby(
    ["size_group", "change_category"],
    observed=True
).agg(
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    sMAPE=("smape", "mean"),
    sample_count=("absolute_error", "count")
)

size_change_results.to_csv(
    OUTPUT_DIR / "baseline_size_change_results.csv"
)

print("\nSIZE GROUP × CHANGE CATEGORY:")
print(size_change_results)

# =========================================================
# 7. TRANSITION VISUALIZATION
# =========================================================

plot_data = transition_results.sort_values("sMAPE")

plt.figure(figsize=(12, 7))

plt.barh(
    plot_data.index,
    plot_data["sMAPE"]
)

plt.xlabel("sMAPE (%)")

plt.ylabel("Day-to-Day Transition")

plt.title("Persistence Forecast Error by Day Transition")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_transition_smape.png",
    dpi=300
)

plt.close()

# =========================================================
# 8. CHANGE CATEGORY VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    change_category_results.index,
    change_category_results["sMAPE"]
)

plt.ylabel("sMAPE (%)")

plt.title("Forecast Error: Large vs Ordinary Changes")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_change_category_smape.png",
    dpi=300
)

plt.close()

# =========================================================
# 9. SIZE GROUP VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    size_results.index.astype(str),
    size_results["sMAPE"]
)

plt.ylabel("sMAPE (%)")

plt.title("Persistence Forecast Error by Entity Size")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_size_group_smape.png",
    dpi=300
)

plt.close()

# =========================================================
# FINAL
# =========================================================

print("\nTRANSITION ANALYSIS COMPLETED")

print(f"Outputs saved to: {OUTPUT_DIR}")