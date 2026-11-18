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

# =========================================================
# LOAD AND FILTER DATA
# =========================================================

df = pd.read_parquet(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2020-01-01") &
    (df["date"] <= "2025-12-31")
].copy()

df = df.sort_values(["entity", "date"])

# =========================================================
# FORECASTING FUNCTIONS
# =========================================================

def persistence(series):
    return series.shift(1)


def weekly_persistence(series):
    return series.shift(7)


def moving_average_7(series):
    return series.shift(1).rolling(7).mean()


def moving_average_14(series):
    return series.shift(1).rolling(14).mean()


# =========================================================
# EVALUATION
# =========================================================

results = []

for entity, group in df.groupby("entity"):

    group = group.sort_values("date").copy()

    series = group[TARGET]

    # Chronological split: last 20% used for testing
    split_idx = int(len(group) * 0.8)

    test_dates = group.iloc[split_idx:]["date"]

    if len(test_dates) == 0:
        continue

    forecasts = {
        "Persistence": persistence(series),
        "Weekly_Persistence": weekly_persistence(series),
        "Moving_Average_7": moving_average_7(series),
        "Moving_Average_14": moving_average_14(series),
    }

    actual = series.iloc[split_idx:]

    for method_name, forecast_series in forecasts.items():

        predicted = forecast_series.iloc[split_idx:]

        valid = actual.notna() & predicted.notna()

        actual_valid = actual[valid]
        predicted_valid = predicted[valid]

        if len(actual_valid) == 0:
            continue

        errors = actual_valid - predicted_valid

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # MAPE can be unstable for small entities
        mape = np.mean(
            np.abs(errors / actual_valid)
        ) * 100

        smape = np.mean(
            2 * np.abs(actual_valid - predicted_valid)
            / (np.abs(actual_valid) + np.abs(predicted_valid))
        ) * 100

        results.append({
            "entity": entity,
            "method": method_name,
            "test_samples": len(actual_valid),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape
        })


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "baseline_forecast_results.csv",
    index=False
)

summary = results_df.groupby("method").agg(
    mean_MAE=("MAE", "mean"),
    mean_RMSE=("RMSE", "mean"),
    mean_MAPE=("MAPE", "mean"),
    mean_sMAPE=("sMAPE", "mean")
)

summary.to_csv(
    OUTPUT_DIR / "baseline_forecast_summary.csv"
)

# Entity-wise best method
best_methods = results_df.loc[
    results_df.groupby("entity")["MAE"].idxmin()
].sort_values("MAE")

best_methods.to_csv(
    OUTPUT_DIR / "best_baseline_by_entity.csv",
    index=False
)

print("\nBASELINE FORECAST SUMMARY:")
print(summary)

print("\nBEST METHOD BY ENTITY:")
print(best_methods[
    ["entity", "method", "MAE", "RMSE", "sMAPE"]
])

print("\nBASELINE FORECASTABILITY ANALYSIS COMPLETED")
print(f"Outputs saved to: {OUTPUT_DIR}")