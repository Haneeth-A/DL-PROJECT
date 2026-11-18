import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================

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

# =========================
# LOAD DATA
# =========================

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

target = "dem_energy_met_gwh"

weather_vars = [
    "wx_tmax_c",
    "wx_tmean_c",
    "wx_tmin_c",
    "wx_cdd24",
    "wx_hdd18",
    "wx_rain_mm"
]

# Keep only required columns
df = df[["entity", "date", target] + weather_vars]

# =========================
# METHOD 1:
# RAW CORRELATIONS
# =========================

raw_results = []

for entity, group in df.groupby("entity"):

    for weather in weather_vars:

        subset = group[[target, weather]].dropna()

        if len(subset) >= 30:

            corr = subset[target].corr(subset[weather])

            raw_results.append({
                "entity": entity,
                "weather_variable": weather,
                "raw_correlation": corr,
                "sample_size": len(subset)
            })

raw_corr = pd.DataFrame(raw_results)

# =========================
# METHOD 2:
# REMOVE MONTHLY SEASONALITY
# =========================

df["month"] = df["date"].dt.month

monthly_results = []

for entity, group in df.groupby("entity"):

    group = group.copy()

    # Remove each entity's month-specific mean
    group["demand_monthly_residual"] = (
        group[target] -
        group.groupby("month")[target].transform("mean")
    )

    for weather in weather_vars:

        group[f"{weather}_monthly_residual"] = (
            group[weather] -
            group.groupby("month")[weather].transform("mean")
        )

        subset = group[
            ["demand_monthly_residual",
             f"{weather}_monthly_residual"]
        ].dropna()

        if len(subset) >= 30:

            corr = subset["demand_monthly_residual"].corr(
                subset[f"{weather}_monthly_residual"]
            )

            monthly_results.append({
                "entity": entity,
                "weather_variable": weather,
                "monthly_adjusted_correlation": corr,
                "sample_size": len(subset)
            })

monthly_corr = pd.DataFrame(monthly_results)

# =========================
# MERGE RESULTS
# =========================

comparison = raw_corr.merge(
    monthly_corr,
    on=["entity", "weather_variable"],
    how="inner"
)

comparison["correlation_change"] = (
    comparison["monthly_adjusted_correlation"]
    - comparison["raw_correlation"]
)

# =========================
# SAVE RESULTS
# =========================

raw_corr.to_csv(
    OUTPUT_DIR / "weather_raw_correlations.csv",
    index=False
)

monthly_corr.to_csv(
    OUTPUT_DIR / "weather_monthly_adjusted_correlations.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "weather_correlation_comparison.csv",
    index=False
)

# =========================
# PRINT RESULTS
# =========================

print("\nRAW CORRELATIONS:")
print(
    raw_corr.sort_values(
        "raw_correlation",
        ascending=False
    ).head(20)
)

print("\nMONTHLY ADJUSTED CORRELATIONS:")
print(
    monthly_corr.sort_values(
        "monthly_adjusted_correlation",
        ascending=False
    ).head(20)
)

print("\nCORRELATION COMPARISON:")
print(comparison.head(20))

print("\nWEATHER-SEASONALITY CONTROL COMPLETED")
print(f"Outputs saved to: {OUTPUT_DIR}")