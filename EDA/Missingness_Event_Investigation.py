import pandas as pd
from pathlib import Path

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

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
].copy()

# Identify dates where demand is entirely missing across entities
daily_missing = (
    df.groupby("date")["dem_energy_met_gwh"]
    .apply(lambda x: x.isna().sum())
)

fully_missing_dates = daily_missing[daily_missing == 34].index

print("\nFULLY MISSING DATES:")
print(fully_missing_dates)

# Check other variable groups on these dates
columns_to_check = [
    "dem_energy_met_gwh",
    "dem_max_demand_mw",
    "dev_od_ud_gwh",
    "gen_conv_total_gwh",
    "gen_re_total_gwh",
    "price_weighted_avg",
    "wx_tmean_c",
    "wx_tmax_c",
    "wx_tmin_c",
    "wx_rain_mm"
]

available_columns = [c for c in columns_to_check if c in df.columns]

event_df = df[df["date"].isin(fully_missing_dates)]

print("\nMISSINGNESS OF OTHER VARIABLES ON FULLY MISSING DEMAND DATES:")
print(event_df[available_columns].isna().sum())

print("\nMONTH DISTRIBUTION OF FULLY MISSING DATES:")
print(pd.Series(fully_missing_dates.month).value_counts().sort_index())

# Consecutive missing dates analysis
missing_date_list = sorted(fully_missing_dates)

clusters = []
cluster = []

for date in missing_date_list:
    if not cluster:
        cluster = [date]
    elif (date - cluster[-1]).days == 1:
        cluster.append(date)
    else:
        clusters.append(cluster)
        cluster = [date]

if cluster:
    clusters.append(cluster)

print("\nMISSING DATE CLUSTERS:")
for i, c in enumerate(clusters, 1):
    print(
        f"Cluster {i}: {c[0].date()} to {c[-1].date()} "
        f"({len(c)} day(s))"
    )

# Save results
output_dir = PROJECT_ROOT / "EDA" / "outputs"
output_dir.mkdir(exist_ok=True)

pd.DataFrame({
    "date": fully_missing_dates
}).to_csv(output_dir / "fully_missing_dates.csv", index=False)

print("\nINVESTIGATION COMPLETED")