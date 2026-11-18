from pathlib import Path
import pandas as pd

PROJECT = Path(r"C:\IIIT Raichur\Projects\DL Project")

OUTPUT = PROJECT / "EDA" / "outputs"

FILES = {

    "Dataset": [
        "entity_coverage.csv",
        "missing_values.csv",
        "negative_values.csv",
        "split_distribution.csv",
        "demand_missingness_by_entity.csv",
        "demand_missingness_by_year.csv",
        "demand_missingness_by_split.csv"
    ],

    "Demand": [
        "annual_aggregate_demand.csv",
        "monthly_demand_profile.csv",
        "entity_demand_statistics.csv",
        "entity_year_demand.csv",
        "entity_growth_2020_2025.csv",
        "day_of_week_demand.csv"
    ],

    "Regime": [
        "aggregate_regime_statistics.csv",
        "regime_annual_validation.csv",
        "regime_quarterly_validation.csv"
    ],

    "Predictability": [
        "entity_predictability_profile.csv",
        "entity_variability.csv",
        "baseline_entity_difficulty.csv"
    ],

    "Weather": [
        "weather_raw_correlations.csv",
        "weather_monthly_adjusted_correlations.csv",
        "weather_trend_calendar_adjusted.csv",
        "weather_entity_correlations.csv"
    ],

    "Interstate": [
        "interstate_correlation_pairs.csv",
        "interstate_residual_correlations.csv",
        "interstate_rolling_correlations.csv"
    ],

    "Baseline": [
        "baseline_monthly_error.csv",
        "baseline_weekday_error.csv",
        "baseline_transition_results.csv",
        "baseline_size_group_results.csv",
        "baseline_large_change_comparison.csv",
        "baseline_worst_100_forecasts.csv"
    ]
}

manifest = []

print("=" * 70)
print("ROLE A — CONSOLIDATION MANIFEST")
print("=" * 70)

for category, files in FILES.items():

    print(f"\n[{category}]")

    for filename in files:

        path = OUTPUT / filename

        if path.exists():

            try:
                df = pd.read_csv(path)

                status = "AVAILABLE"

                rows = len(df)
                columns = len(df.columns)

            except Exception as e:

                status = f"READ ERROR: {e}"
                rows = None
                columns = None

        else:

            status = "MISSING"
            rows = None
            columns = None

        manifest.append({
            "category": category,
            "file": filename,
            "status": status,
            "rows": rows,
            "columns": columns
        })

        print(
            f"{filename:45} "
            f"{status:15} "
            f"rows={rows} "
            f"columns={columns}"
        )

manifest_df = pd.DataFrame(manifest)

manifest_df.to_csv(
    OUTPUT / "role_a_consolidation_manifest.csv",
    index=False
)

print("\nManifest saved.")