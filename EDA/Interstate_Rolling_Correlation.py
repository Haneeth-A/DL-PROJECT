from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "Data" / "Raw"
    / "RE-FUSED_India_Datasets_v1.0"
    / "RE-FUSED" / "India_Grid_Study"
    / "data" / "processed"
    / "refused_state_day.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "outputs"

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
]

panel = df.pivot(
    index="date",
    columns="entity",
    values="dem_energy_met_gwh"
).sort_index()

pairs = [
    ("CH", "DL"),
    ("DL", "HR"),
    ("HR", "PB"),
    ("DL", "UP"),
    ("PY", "TN"),
    ("BR", "WB"),
]

results = []

for a, b in pairs:

    pair = panel[[a, b]].dropna()

    rolling = pair[a].rolling(
        window=180,
        min_periods=150
    ).corr(pair[b])

    for date, correlation in rolling.items():

        results.append({
            "date": date,
            "entity_1": a,
            "entity_2": b,
            "rolling_correlation": correlation
        })

result = pd.DataFrame(results)

result.to_csv(
    OUTPUT_DIR / "interstate_rolling_correlations.csv",
    index=False
)

summary = result.groupby(
    ["entity_1", "entity_2"]
)["rolling_correlation"].agg(
    ["mean", "std", "min", "max"]
)

print("\nROLLING CORRELATION SUMMARY")
print(summary.round(3))

print("\nROLLING CORRELATION ANALYSIS COMPLETED")