from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# PATHS
# =========================================================

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

TARGET = "dem_energy_met_gwh"

TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["entity", "date"]).copy()

# =========================================================
# FORECAST METHODS
# =========================================================

def persistence(series):
    return series.shift(1)


def weekly_persistence(series):
    return series.shift(7)


def moving_average_7(series):
    return series.shift(1).rolling(7).mean()


def moving_average_14(series):
    return series.shift(1).rolling(14).mean()


methods = {
    "Persistence": persistence,
    "Weekly_Persistence": weekly_persistence,
    "Moving_Average_7": moving_average_7,
    "Moving_Average_14": moving_average_14,
}

# =========================================================
# METRICS
# =========================================================

def calculate_metrics(actual, predicted):

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    denominator = np.abs(actual) + np.abs(predicted)

    smape_values = np.where(
        denominator == 0,
        0,
        2 * np.abs(errors) / denominator
    )

    smape = np.mean(smape_values) * 100

    return mae, rmse, smape


# =========================================================
# ENTITY-LEVEL EVALUATION
# =========================================================

entity_results = []

panel_records = []

for entity, group in df.groupby("entity"):

    group = group.sort_values("date").copy()

    series = group[TARGET]

    group["date"] = pd.to_datetime(group["date"])

    test_mask = (
        (group["date"] >= TEST_START) &
        (group["date"] <= TEST_END)
    )

    actual = group.loc[test_mask, TARGET]

    for method_name, method_function in methods.items():

        forecasts = method_function(series)

        predicted = forecasts.loc[test_mask]

        valid = actual.notna() & predicted.notna()

        actual_valid = actual[valid]
        predicted_valid = predicted[valid]

        if len(actual_valid) == 0:
            continue

        mae, rmse, smape = calculate_metrics(
            actual_valid.to_numpy(),
            predicted_valid.to_numpy()
        )

        entity_results.append({
            "entity": entity,
            "method": method_name,
            "test_samples": len(actual_valid),
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smape
        })

        panel_records.extend([
            {
                "entity": entity,
                "method": method_name,
                "date": date,
                "actual": a,
                "predicted": p
            }
            for date, a, p in zip(
                actual_valid.index,
                actual_valid.to_numpy(),
                predicted_valid.to_numpy()
            )
        ])

# =========================================================
# SAVE ENTITY RESULTS
# =========================================================

entity_results_df = pd.DataFrame(entity_results)

entity_results_df.to_csv(
    OUTPUT_DIR / "refined_baseline_entity_results.csv",
    index=False
)

# =========================================================
# MACRO AVERAGE
# Equal weight for each entity
# =========================================================

macro_summary = entity_results_df.groupby("method").agg(
    macro_MAE=("MAE", "mean"),
    macro_RMSE=("RMSE", "mean"),
    macro_sMAPE=("sMAPE", "mean")
)

macro_summary.to_csv(
    OUTPUT_DIR / "refined_baseline_macro_summary.csv"
)

# =========================================================
# PANEL-LEVEL METRICS
# All observations pooled together
# =========================================================

panel_df = pd.DataFrame(panel_records)

panel_results = []

for method_name, group in panel_df.groupby("method"):

    mae, rmse, smape = calculate_metrics(
        group["actual"].to_numpy(),
        group["predicted"].to_numpy()
    )

    panel_results.append({
        "method": method_name,
        "panel_MAE": mae,
        "panel_RMSE": rmse,
        "panel_sMAPE": smape,
        "total_test_samples": len(group)
    })

panel_summary = pd.DataFrame(panel_results)

panel_summary.to_csv(
    OUTPUT_DIR / "refined_baseline_panel_summary.csv",
    index=False
)

# =========================================================
# ENTITY DIFFICULTY RANKING
# =========================================================

difficulty = (
    entity_results_df[
        entity_results_df["method"] == "Persistence"
    ]
    .sort_values("sMAPE", ascending=False)
)

difficulty.to_csv(
    OUTPUT_DIR / "baseline_entity_difficulty.csv",
    index=False
)

# =========================================================
# OUTPUT
# =========================================================

print("\nMACRO-AVERAGE RESULTS:")
print(macro_summary)

print("\nPANEL-LEVEL RESULTS:")
print(panel_summary)

print("\nENTITY DIFFICULTY (PERSISTENCE sMAPE):")
print(difficulty[
    ["entity", "MAE", "RMSE", "sMAPE"]
])

print("\nREFINED BASELINE ANALYSIS COMPLETED")
print(f"Outputs saved to: {OUTPUT_DIR}")