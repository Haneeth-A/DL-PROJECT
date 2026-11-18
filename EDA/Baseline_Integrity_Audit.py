import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT = Path(r"C:\IIIT Raichur\Projects\DL Project")

DATA_PATH = PROJECT / "Data" / "Raw" / "RE-FUSED_India_Datasets_v1.0" / "RE-FUSED" / "India_Grid_Study" / "data" / "processed" / "refused_state_day.parquet"

OUTPUT = PROJECT / "EDA" / "outputs" / "baseline_integrity_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET = "dem_energy_met_gwh"
ENTITY = "entity"
DATE = "date"

TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_parquet(DATA_PATH)

df[DATE] = pd.to_datetime(df[DATE])
df = df.sort_values([ENTITY, DATE]).copy()

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 1. CALENDAR INTEGRITY
# --------------------------------------------------

print("\nChecking calendar integrity...")

calendar_results = []

for entity, group in df.groupby(ENTITY):

    dates = group[DATE].sort_values()

    calendar_results.append({
        "entity": entity,
        "rows": len(group),
        "unique_dates": dates.nunique(),
        "duplicate_dates": dates.duplicated().sum(),
        "missing_calendar_days": (
            dates.max() - dates.min()
        ).days + 1 - dates.nunique()
    })

calendar = pd.DataFrame(calendar_results)
calendar.to_csv(OUTPUT / "calendar_integrity.csv", index=False)

# --------------------------------------------------
# 2. BUILD STRICT ONE-DAY-AHEAD FORECASTS
# --------------------------------------------------

print("\nBuilding forecasts...")

records = []

for entity, group in df.groupby(ENTITY):

    group = group.sort_values(DATE).copy()
    group = group.set_index(DATE)

    # Actual target
    group["actual"] = group[TARGET]

    # Strict calendar-day lag
    group["persistence"] = group["actual"].shift(1)
    group["weekly"] = group["actual"].shift(7)

    group["ma7"] = (
        group["actual"].shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    group["ma14"] = (
        group["actual"].shift(1)
        .rolling(14, min_periods=14)
        .mean()
    )

    group = group.reset_index()

    group = group[
        (group[DATE] >= TEST_START) &
        (group[DATE] <= TEST_END)
    ]

    for _, row in group.iterrows():

        records.append({
            "entity": entity,
            "date": row[DATE],
            "actual": row["actual"],
            "persistence": row["persistence"],
            "weekly": row["weekly"],
            "ma7": row["ma7"],
            "ma14": row["ma14"]
        })

pred = pd.DataFrame(records)

# --------------------------------------------------
# 3. COMMON EVALUATION SAMPLE
# --------------------------------------------------

print("\nChecking missing observations and sample alignment...")

methods = ["persistence", "weekly", "ma7", "ma14"]

pred["common_valid"] = pred[
    ["actual"] + methods
].notna().all(axis=1)

common = pred[pred["common_valid"]].copy()

sample_counts = []

for method in methods:

    sample_counts.append({
        "method": method,
        "available_samples": pred[
            ["actual", method]
        ].notna().all(axis=1).sum(),
        "common_samples": len(common)
    })

pd.DataFrame(sample_counts).to_csv(
    OUTPUT / "sample_alignment.csv", index=False
)

common.to_csv(
    OUTPUT / "common_evaluation_predictions.csv",
    index=False
)

# --------------------------------------------------
# 4. METRIC CALCULATION
# --------------------------------------------------

print("\nCalculating metrics...")

def calculate_metrics(actual, forecast):

    mae = mean_absolute_error(actual, forecast)

    rmse = np.sqrt(
        mean_squared_error(actual, forecast)
    )

    smape = np.mean(
        200 * np.abs(actual - forecast) /
        (np.abs(actual) + np.abs(forecast) + 1e-8)
    )

    return mae, rmse, smape


# Macro metrics: equal weight per entity
macro_results = []

for method in methods:

    entity_metrics = []

    for entity, group in common.groupby("entity"):

        mae, rmse, smape = calculate_metrics(
            group["actual"],
            group[method]
        )

        entity_metrics.append({
            "entity": entity,
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smape
        })

    entity_df = pd.DataFrame(entity_metrics)

    macro_results.append({
        "method": method,
        "macro_MAE": entity_df["MAE"].mean(),
        "macro_RMSE": entity_df["RMSE"].mean(),
        "macro_sMAPE": entity_df["sMAPE"].mean(),
        "entities": len(entity_df),
        "samples": len(common)
    })


# Pooled metrics: all observations together
pooled_results = []

for method in methods:

    mae, rmse, smape = calculate_metrics(
        common["actual"],
        common[method]
    )

    pooled_results.append({
        "method": method,
        "pooled_MAE": mae,
        "pooled_RMSE": rmse,
        "pooled_sMAPE": smape,
        "samples": len(common)
    })


pd.DataFrame(macro_results).to_csv(
    OUTPUT / "macro_metrics.csv", index=False
)

pd.DataFrame(pooled_results).to_csv(
    OUTPUT / "pooled_metrics.csv", index=False
)

# --------------------------------------------------
# 5. DATE COVERAGE
# --------------------------------------------------

date_coverage = common.groupby("date").agg(
    entities=("entity", "nunique"),
    observations=("actual", "count")
).reset_index()

date_coverage.to_csv(
    OUTPUT / "date_coverage.csv", index=False
)

# --------------------------------------------------
# 6. ENTITY-LEVEL METRICS
# --------------------------------------------------

entity_results = []

for entity, group in common.groupby("entity"):

    for method in methods:

        mae, rmse, smape = calculate_metrics(
            group["actual"],
            group[method]
        )

        entity_results.append({
            "entity": entity,
            "method": method,
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smape,
            "samples": len(group)
        })

pd.DataFrame(entity_results).to_csv(
    OUTPUT / "entity_metrics.csv", index=False
)

# --------------------------------------------------
# 7. PRINT SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE INTEGRITY AUDIT — RESULTS")
print("=" * 60)

print("\nCALENDAR INTEGRITY")
print(calendar.to_string(index=False))

print("\nSAMPLE ALIGNMENT")
print(pd.DataFrame(sample_counts).to_string(index=False))

print("\nMACRO METRICS")
print(pd.DataFrame(macro_results).round(4).to_string(index=False))

print("\nPOOLED METRICS")
print(pd.DataFrame(pooled_results).round(4).to_string(index=False))

print("\nDATE COVERAGE")
print("Evaluation dates:", common[DATE].nunique())
print("First date:", common[DATE].min())
print("Last date:", common[DATE].max())
print("Entities:", common[ENTITY].nunique())
print("Total common observations:", len(common))

print("\nAudit outputs saved to:")
print(OUTPUT)

print("\nAUDIT COMPLETED.")