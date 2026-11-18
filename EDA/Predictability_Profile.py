from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "Data" / "Raw"
    / "RE-FUSED_India_Datasets_v1.0"
    / "RE-FUSED" / "India_Grid_Study"
    / "data" / "processed"
    / "refused_state_day.parquet"
)

OUT = PROJECT_ROOT / "EDA" / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

target = "dem_energy_met_gwh"

results = []

for entity, group in df.groupby("entity"):

    group = group.sort_values("date").copy()
    series = group[target].dropna()

    if len(series) < 100:
        continue

    # CV
    mean_demand = series.mean()
    std_demand = series.std()

    cv = std_demand / mean_demand if mean_demand else np.nan

    # Autocorrelation
    lag1 = series.autocorr(lag=1)
    lag7 = series.autocorr(lag=7)
    lag30 = series.autocorr(lag=30)

    # Day-to-day changes
    daily_change = series.diff().dropna()

    normalized_daily_change = (
        daily_change.abs().mean() / mean_demand
    )

    daily_change_std = (
        daily_change.std() / mean_demand
    )

    # Weekly seasonal profile
    group["dow"] = group["date"].dt.day_of_week

    weekday_means = group.groupby("dow")[target].mean()

    weekday_seasonality = (
        weekday_means.std() / group[target].mean()
    )

    # Monthly seasonal profile
    group["month"] = group["date"].dt.month

    month_means = group.groupby("month")[target].mean()

    monthly_seasonality = (
        month_means.std() / group[target].mean()
    )

    results.append({
        "entity": entity,
        "mean_demand": mean_demand,
        "CV": cv,
        "autocorrelation_lag1": lag1,
        "autocorrelation_lag7": lag7,
        "autocorrelation_lag30": lag30,
        "normalized_mean_absolute_daily_change": normalized_daily_change,
        "normalized_daily_change_std": daily_change_std,
        "weekday_seasonality_ratio": weekday_seasonality,
        "monthly_seasonality_ratio": monthly_seasonality
    })

profile = pd.DataFrame(results)

profile = profile.sort_values(
    "normalized_mean_absolute_daily_change"
)

profile.to_csv(
    OUT / "entity_predictability_profile.csv",
    index=False
)

print("\nENTITY PREDICTABILITY PROFILE")
print(profile.to_string(index=False))

print("\nPREDICTABILITY PROFILE COMPLETED")