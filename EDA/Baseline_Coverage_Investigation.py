import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:\IIIT Raichur\Projects\DL Project")

DATA_PATH = (
    PROJECT / "Data" / "Raw" /
    "RE-FUSED_India_Datasets_v1.0" /
    "RE-FUSED" / "India_Grid_Study" /
    "data" / "processed" /
    "refused_state_day.parquet"
)

OUTPUT = PROJECT / "EDA" / "outputs" / "baseline_integrity_audit"

TARGET = "dem_energy_met_gwh"
ENTITY = "entity"
DATE = "date"

df = pd.read_parquet(DATA_PATH)

df[DATE] = pd.to_datetime(df[DATE])
df = df.sort_values([ENTITY, DATE])

results = []

for entity, group in df.groupby(ENTITY):

    group = group.copy().set_index(DATE)

    group["actual"] = group[TARGET]

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

    test = group.loc["2025-01-01":"2025-12-31"]

    for date, row in test.iterrows():

        results.append({
            "entity": entity,
            "date": date,
            "actual": row["actual"],
            "persistence": row["persistence"],
            "weekly": row["weekly"],
            "ma7": row["ma7"],
            "ma14": row["ma14"]
        })

df = pd.DataFrame(results)

methods = ["persistence", "weekly", "ma7", "ma14"]

# Daily coverage
coverage = df.groupby("date").agg(
    total_entities=("entity", "nunique"),
    actual_available=("actual", "count"),
    persistence_available=("persistence", "count"),
    weekly_available=("weekly", "count"),
    ma7_available=("ma7", "count"),
    ma14_available=("ma14", "count")
).reset_index()

coverage.to_csv(
    OUTPUT / "daily_method_coverage.csv",
    index=False
)

# Missing dates in common evaluation
df["common_valid"] = df[
    ["actual"] + methods
].notna().all(axis=1)

common_dates = set(
    df.loc[df["common_valid"], "date"].unique()
)

all_dates = pd.date_range(
    "2025-01-01",
    "2025-12-31",
    freq="D"
)

missing_dates = sorted(set(all_dates) - common_dates)

print("\nCOMMON EVALUATION COVERAGE")
print("Expected calendar days:", len(all_dates))
print("Available common dates:", len(common_dates))
print("Missing common dates:", len(missing_dates))

print("\nMISSING COMMON DATES")

for date in missing_dates:
    print(date.date())

print("\nDAILY COVERAGE SUMMARY")
print(coverage.describe().round(2).to_string())

print("\nFIRST 20 DAYS")
print(coverage.head(20).to_string(index=False))

print("\nLAST 10 DAYS")
print(coverage.tail(10).to_string(index=False))

print("\nSaved daily coverage to:", OUTPUT)