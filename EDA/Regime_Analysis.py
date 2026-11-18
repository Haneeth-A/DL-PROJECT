import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "Data" / "Raw"
    / "RE-FUSED_India_Datasets_v1.0"
    / "RE-FUSED" / "India_Grid_Study"
    / "data" / "processed"
    / "refused_state_day.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] < "2026-01-01")
]

# Aggregate demand
daily = df.groupby("date")["dem_energy_met_gwh"].agg(
    lambda x: x.sum() if x.notna().all() else np.nan
).dropna()

# Smooth short-term fluctuations using a 30-day rolling mean
rolling = daily.rolling(30, center=True).mean().dropna()

# Change-point detection
signal = rolling.values.reshape(-1, 1)

algo = rpt.Pelt(model="rbf", min_size=60, jump=1).fit(signal)

# Penalty controls number of detected changes
breakpoints = algo.predict(pen=10)

change_dates = [
    rolling.index[index - 1]
    for index in breakpoints[:-1]
]

print("\nDETECTED CHANGE POINTS:")
for date in change_dates:
    print(date.date())

# Plot
plt.figure(figsize=(16, 7))

plt.plot(daily.index, daily.values, alpha=0.35,
         label="Daily aggregate demand")

plt.plot(rolling.index, rolling.values,
         linewidth=2, label="30-day rolling mean")

for i, date in enumerate(change_dates):
    plt.axvline(
        date,
        linestyle="--",
        label="Detected change point" if i == 0 else None
    )

plt.title("Structural Change Detection in Aggregate Demand")
plt.xlabel("Date")
plt.ylabel("Energy Met (GWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "aggregate_regime_detection.png",
    dpi=300
)

plt.close()

# Regime statistics
regime_dates = [daily.index.min()] + change_dates + [daily.index.max()]

regimes = []

for i in range(len(regime_dates) - 1):
    start = regime_dates[i]
    end = regime_dates[i + 1]

    segment = daily.loc[start:end]

    regimes.append({
        "regime": i + 1,
        "start": start,
        "end": end,
        "days": len(segment),
        "mean": segment.mean(),
        "std": segment.std(),
        "minimum": segment.min(),
        "maximum": segment.max()
    })

regime_df = pd.DataFrame(regimes)

regime_df.to_csv(
    OUTPUT_DIR / "aggregate_regime_statistics.csv",
    index=False
)

print("\nREGIME STATISTICS:")
print(regime_df)

print("\nREGIME ANALYSIS COMPLETED")