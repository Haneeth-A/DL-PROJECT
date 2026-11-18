from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
]

# Aggregate only available demand values
daily = df.groupby("date")["dem_energy_met_gwh"].sum(min_count=1)

# Smooth daily noise
rolling = daily.rolling(30, min_periods=20).mean()

# Plot
plt.figure(figsize=(16, 6))
plt.plot(daily.index, daily.values, alpha=0.35, label="Daily aggregate")
plt.plot(rolling.index, rolling.values, linewidth=2, label="30-day rolling mean")

plt.title("Aggregate Demand with 30-Day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Energy Met (GWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUT / "regime_rolling_mean.png", dpi=300)
plt.show()

# Annual statistics
annual = daily.groupby(daily.index.year).agg(
    mean="mean",
    std="std",
    min="min",
    max="max"
)

print("\nANNUAL STATISTICS:")
print(annual)

# Quarterly statistics
quarterly = daily.groupby(daily.index.to_period("Q")).agg(
    mean="mean",
    std="std",
    min="min",
    max="max"
)

print("\nQUARTERLY STATISTICS:")
print(quarterly)

annual.to_csv(OUT / "regime_annual_validation.csv")
quarterly.to_csv(OUT / "regime_quarterly_validation.csv")

print("\nREGIME VALIDATION COMPLETED")