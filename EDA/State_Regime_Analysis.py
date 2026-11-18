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
].copy()

df["year"] = df["date"].dt.year

# --------------------------------------------------
# 1. ENTITY-YEAR DEMAND
# --------------------------------------------------

entity_year = df.groupby(["entity", "year"])["dem_energy_met_gwh"].mean().unstack()

entity_year.to_csv(OUT / "entity_year_demand.csv")

# --------------------------------------------------
# 2. GROWTH: 2020 TO 2025
# --------------------------------------------------

growth = pd.DataFrame(index=entity_year.index)

growth["2020_mean"] = entity_year[2020]
growth["2025_mean"] = entity_year[2025]

growth["absolute_change"] = growth["2025_mean"] - growth["2020_mean"]

growth["percentage_change"] = (
    (growth["2025_mean"] - growth["2020_mean"])
    / growth["2020_mean"]
) * 100

growth = growth.sort_values("percentage_change", ascending=False)

print("\n2020-2025 ENTITY DEMAND GROWTH:")
print(growth)

growth.to_csv(OUT / "entity_growth_2020_2025.csv")

# --------------------------------------------------
# 3. COVID PERIOD COMPARISON
# --------------------------------------------------

periods = {
    "Pre_disruption": ("2020-01-01", "2020-03-24"),
    "Disruption": ("2020-03-25", "2020-06-30"),
    "Recovery": ("2020-07-01", "2020-12-31")
}

period_means = {}

for name, (start, end) in periods.items():
    subset = df[
        (df["date"] >= start) &
        (df["date"] <= end)
    ]

    period_means[name] = (
        subset.groupby("entity")["dem_energy_met_gwh"].mean()
    )

covid = pd.DataFrame(period_means)

covid["disruption_change_pct"] = (
    (covid["Disruption"] - covid["Pre_disruption"])
    / covid["Pre_disruption"]
) * 100

covid["recovery_change_pct"] = (
    (covid["Recovery"] - covid["Disruption"])
    / covid["Disruption"]
) * 100

covid = covid.sort_values("disruption_change_pct")

print("\nCOVID PERIOD COMPARISON:")
print(covid)

covid.to_csv(OUT / "entity_covid_comparison.csv")

# --------------------------------------------------
# 4. ENTITY VARIABILITY
# --------------------------------------------------

variability = df.groupby("entity")["dem_energy_met_gwh"].agg(
    mean="mean",
    std="std",
    median="median",
    minimum="min",
    maximum="max"
)

variability["CV"] = variability["std"] / variability["mean"]

variability = variability.sort_values("CV", ascending=False)

print("\nENTITY VARIABILITY:")
print(variability)

variability.to_csv(OUT / "entity_variability.csv")

# --------------------------------------------------
# 5. VISUALIZATION
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

growth.sort_values("percentage_change").plot(
    kind="barh",
    y="percentage_change",
    ax=axes[0],
    legend=False
)

axes[0].set_title("Entity Demand Growth (2020-2025)")
axes[0].set_xlabel("Percentage Change (%)")
axes[0].set_ylabel("Entity")

variability.sort_values("CV").plot(
    kind="barh",
    y="CV",
    ax=axes[1],
    legend=False
)

axes[1].set_title("Demand Variability Across Entities")
axes[1].set_xlabel("Coefficient of Variation")
axes[1].set_ylabel("Entity")

plt.tight_layout()

plt.savefig(OUT / "entity_growth_variability.png", dpi=300)
plt.show()

print("\nSTATE REGIME ANALYSIS COMPLETED")