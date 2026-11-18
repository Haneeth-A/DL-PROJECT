from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Remove rows with missing target
df = df.dropna(subset=["dem_energy_met_gwh"])

weather_cols = [
    "wx_tmean_c",
    "wx_tmax_c",
    "wx_tmin_c",
    "wx_rain_mm",
    "wx_cdd24",
    "wx_hdd18"
]

weather_cols = [c for c in weather_cols if c in df.columns]

# --------------------------------------------------
# 1. OVERALL CORRELATIONS
# --------------------------------------------------

corr = df[
    ["dem_energy_met_gwh"] + weather_cols
].corr(method="pearson")

print("\nOVERALL WEATHER-DEMAND CORRELATION:")
print(corr["dem_energy_met_gwh"].sort_values(ascending=False))

corr.to_csv(OUT / "weather_overall_correlation.csv")

# --------------------------------------------------
# 2. ENTITY-WISE CORRELATION
# --------------------------------------------------

records = []

for entity, group in df.groupby("entity"):

    for weather in weather_cols:

        valid = group[
            ["dem_energy_met_gwh", weather]
        ].dropna()

        if len(valid) < 100:
            continue

        corr_value = valid["dem_energy_met_gwh"].corr(
            valid[weather],
            method="pearson"
        )

        records.append({
            "entity": entity,
            "weather_variable": weather,
            "correlation": corr_value,
            "sample_size": len(valid)
        })

entity_corr = pd.DataFrame(records)

entity_corr.to_csv(
    OUT / "weather_entity_correlations.csv",
    index=False
)

# --------------------------------------------------
# 3. TEMPERATURE CORRELATION RANKING
# --------------------------------------------------

temperature_variables = [
    c for c in ["wx_tmean_c", "wx_cdd24", "wx_hdd18"]
    if c in weather_cols
]

for variable in temperature_variables:

    ranking = entity_corr[
        entity_corr["weather_variable"] == variable
    ].sort_values("correlation")

    print(f"\nENTITY CORRELATION RANKING: {variable}")
    print(ranking[["entity", "correlation"]])

# --------------------------------------------------
# 4. VISUALIZATION
# --------------------------------------------------

if not entity_corr.empty:

    pivot = entity_corr.pivot(
        index="entity",
        columns="weather_variable",
        values="correlation"
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Entity-Level Weather-Demand Correlations")
    plt.tight_layout()

    plt.savefig(
        OUT / "weather_entity_correlation_heatmap.png",
        dpi=300
    )

    plt.show()

print("\nWEATHER SENSITIVITY ANALYSIS COMPLETED")