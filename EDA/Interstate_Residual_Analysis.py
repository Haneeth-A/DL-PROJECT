from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "Data" / "Raw"
    / "RE-FUSED_India_Datasets_v1.0"
    / "RE-FUSED" / "India_Grid_Study"
    / "data" / "processed"
    / "refused_state_day.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

# Aggregate daily demand into entity columns
panel = df.pivot(
    index="date",
    columns="entity",
    values="dem_energy_met_gwh"
).sort_index()

# Remove missing observations for each entity
# First remove linear trend, then calendar-month effects
residuals = pd.DataFrame(index=panel.index)

for entity in panel.columns:

    series = panel[entity]

    temp = pd.DataFrame({
        "demand": series,
        "time": np.arange(len(series)),
        "month": panel.index.month
    }).dropna()

    if len(temp) < 100:
        continue

    # Linear trend + month fixed effects
    month_dummies = pd.get_dummies(
        temp["month"], drop_first=True, dtype=float
    )

    X = pd.concat([
        temp[["time"]].astype(float),
        month_dummies
    ], axis=1)

    X.insert(0, "intercept", 1.0)

    coefficients = np.linalg.lstsq(
        X.values,
        temp["demand"].values,
        rcond=None
    )[0]

    fitted = X.values @ coefficients

    residuals.loc[temp.index, entity] = (
        temp["demand"].values - fitted
    )

# Pairwise residual correlations
results = []

for a, b in combinations(residuals.columns, 2):

    pair = residuals[[a, b]].dropna()

    if len(pair) < 100:
        continue

    results.append({
        "entity_1": a,
        "entity_2": b,
        "residual_correlation": pair[a].corr(pair[b]),
        "sample_size": len(pair)
    })

result = pd.DataFrame(results)

result = result.sort_values(
    "residual_correlation",
    ascending=False
)

print("\nTOP RESIDUAL CORRELATIONS")
print(result.head(20).to_string(index=False))

print("\nLOWEST RESIDUAL CORRELATIONS")
print(result.tail(20).to_string(index=False))

result.to_csv(
    OUTPUT_DIR / "interstate_residual_correlations.csv",
    index=False
)

residuals.to_csv(
    OUTPUT_DIR / "interstate_demand_residuals.csv"
)

print("\nINTERSTATE RESIDUAL ANALYSIS COMPLETED")