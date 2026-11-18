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

weather_vars = [
    "wx_tmax_c", "wx_tmean_c", "wx_tmin_c",
    "wx_cdd24", "wx_hdd18", "wx_rain_mm"
]

df["month"] = df["date"].dt.month.astype("category")
df["dow"] = df["date"].dt.day_of_week.astype("category")
df["time_index"] = (df["date"] - df["date"].min()).dt.days

results = []

for entity, group in df.groupby("entity"):

    group = group.copy()

    for weather in weather_vars:

        subset = group[
            [target, weather, "month", "dow", "time_index"]
        ].dropna().copy()

        if len(subset) < 100:
            continue

        # Control variables: month, weekday and linear trend
        controls = pd.get_dummies(
            subset[["month", "dow"]],
            drop_first=True,
            dtype=float
        )

        controls["time_index"] = subset["time_index"].astype(float)

        controls.insert(0, "intercept", 1.0)

        # Residualize demand
        y = subset[target].astype(float).values

        beta_y = np.linalg.lstsq(
            controls.values, y, rcond=None
        )[0]

        demand_residual = y - controls.values @ beta_y

        # Residualize weather using the same controls
        w = subset[weather].astype(float).values

        beta_w = np.linalg.lstsq(
            controls.values, w, rcond=None
        )[0]

        weather_residual = w - controls.values @ beta_w

        # Residual correlation
        corr = np.corrcoef(
            demand_residual, weather_residual
        )[0, 1]

        results.append({
            "entity": entity,
            "weather_variable": weather,
            "adjusted_correlation": corr,
            "sample_size": len(subset)
        })

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUT / "weather_trend_calendar_adjusted.csv",
    index=False
)

print("\nTREND + CALENDAR ADJUSTED WEATHER CORRELATIONS")

for variable in ["wx_cdd24", "wx_tmean_c", "wx_tmax_c", "wx_hdd18"]:

    ranking = result_df[
        result_df["weather_variable"] == variable
    ].sort_values("adjusted_correlation", ascending=False)

    print(f"\n{variable}")
    print(ranking[["entity", "adjusted_correlation"]].to_string(index=False))

print("\nWEATHER RESIDUAL ANALYSIS COMPLETED")