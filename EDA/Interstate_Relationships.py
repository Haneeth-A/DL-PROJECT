from pathlib import Path
import pandas as pd
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

target = "dem_energy_met_gwh"

# Date x entity matrix
panel = df.pivot(
    index="date",
    columns="entity",
    values=target
)

# Correlation matrix
corr = panel.corr(method="pearson")

corr.to_csv(
    OUT / "interstate_demand_correlation_matrix.csv"
)

# Strongest pairs
pairs = corr.where(
    ~pd.DataFrame(
        True,
        index=corr.index,
        columns=corr.columns
    ).values == False
)

# Extract unique pairs
upper = corr.where(
    pd.DataFrame(
        True,
        index=corr.index,
        columns=corr.columns
    ).values
)

pair_records = []

for i, entity1 in enumerate(corr.columns):

    for entity2 in corr.columns[i + 1:]:

        value = corr.loc[entity1, entity2]

        if pd.notna(value):

            pair_records.append({
                "entity_1": entity1,
                "entity_2": entity2,
                "correlation": value
            })

pairs_df = pd.DataFrame(pair_records)

pairs_df = pairs_df.sort_values(
    "correlation",
    ascending=False
)

pairs_df.to_csv(
    OUT / "interstate_correlation_pairs.csv",
    index=False
)

print("\nTOP CORRELATED ENTITY PAIRS")
print(pairs_df.head(30).to_string(index=False))

print("\nLOWEST CORRELATED ENTITY PAIRS")
print(pairs_df.tail(20).to_string(index=False))

# Heatmap
plt.figure(figsize=(16, 12))

plt.imshow(corr, aspect="auto", interpolation="nearest")
plt.colorbar(label="Pearson Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.index)),
    corr.index
)

plt.title("Inter-Entity Demand Correlation (2020–2025)")
plt.tight_layout()

plt.savefig(
    OUT / "interstate_correlation_heatmap.png",
    dpi=300
)

plt.show()

print("\nINTERSTATE RELATIONSHIP ANALYSIS COMPLETED")