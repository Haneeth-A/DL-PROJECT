# DL Project --- EDA and Role A Evidence Guide

This README documents the scripts, datasets, tables, plots, and audit
outputs currently visible in the project's `EDA` directory. It is an
inventory guide: descriptions of analyses are based on filenames, the
documented workflow, and results previously produced during this
project. A filename does not imply that a result is definitive or
causal.

## Project context

**Research working title:** *Information, Regimes, and Architecture:
Forecasting Indian State-Level Electricity Demand, 2020--2025.*

**Research question:** What limits forecast accuracy for Indian
state-level electricity demand---the forecasting architecture, or the
information available to the model?

Role A focuses on exploratory data analysis, integrity checks, demand
characterization, regime exploration, weather and inter-state
relationships, predictability profiling, and baseline forecasting/error
analysis.

## Folder structure

``` text
DL Project/
├── Data/
│   └── Raw/
│       └── RE-FUSED_India_Datasets_v1.0/
│           └── RE-FUSED/
│               └── India_Grid_Study/
│                   └── data/processed/
│                       └── refused_state_day.parquet
└── EDA/
    ├── EDA/
    │   └── outputs/                 # dataset-audit/panel outputs
    ├── outputs/                     # analysis outputs, tables and plots
    │   ├── baseline_integrity_audit/
    │   └── plots/
    └── *.py                         # analysis scripts
```

Some outputs are stored in the nested `EDA/EDA/outputs` directory, while
most analysis results are in `EDA/outputs`. This is intentional in the
current project state and explains why some inventory scripts initially
looked in the wrong directory.

------------------------------------------------------------------------

# 1. Python scripts

## Dataset and missingness

### `Dataset_Audit.py`

Initial audit of the source panel: dimensions, entity/date coverage,
duplicate records, calendar completeness, variable overview, and basic
integrity checks. Its results establish what is in the dataset before
analysis.

### `missingness_analysis.py`

Examines missing values in the dataset and/or target panel, providing an
overview of where and how much information is absent.

### `Missingness_Event_Investigation.py`

Investigates missingness in more detail, including missing-data patterns
or events over time. Intended to help distinguish isolated missing
observations from systematic gaps.

## Demand characterization

### `Demand_Charecterization.py`

Explores demand distributions and temporal patterns, including aggregate
demand, annual and monthly behavior, entity-level demand, and associated
plots/tables. (The filename uses the spelling `Charecterization`.)

## Regime exploration

### `Regime_Analysis.py`

Detects candidate change points or segments in the aggregate demand
series and summarizes the resulting candidate regimes. These are
exploratory detected segments, not automatically validated structural
regimes.

### `Regime_Validation.py`

Checks detected-regime results against annual and quarterly summaries
and visualizations. This is a validation-oriented descriptive check;
further sensitivity analysis may still be needed before strong regime
claims.

## State-level analysis

### `State_Regime_Analysis.py`

Examines state/entity-level changes and comparisons over time, including
growth and variation between entities and periods.

## Predictability profiling

### `Predictability_Profile.py`

Builds descriptive profiles of each entity using characteristics such as
demand scale, coefficient of variation, autocorrelation, normalized
day-to-day changes, and seasonality-related measures. These features
describe series behavior; they are not themselves model forecast scores.

## Weather analysis

### `Weather_Sensitivity.py`

Explores associations between demand and weather variables at an
aggregate and/or entity level using correlation-based analysis.

### `Weather_Seasonality_Control.py`

Examines weather--demand relationships after accounting for monthly
seasonal effects, to reduce the influence of recurring month-of-year
patterns on simple correlations.

### `Weather_Residual_Analysis.py`

Analyzes weather--demand relationships after trend and calendar-related
adjustments/residualization. These are still descriptive associations
and do not prove causal effects or forecast utility.

## Interstate relationships

### `Interstate_Relationships.py`

Computes and examines pairwise relationships between entity demand
series, including correlation matrices and pair rankings.

### `Interstate_Residual_Analysis.py`

Examines relationships between entity demand after
residualization/adjustment, aiming to reduce shared trend/calendar
effects.

### `Interstate_Rolling_Correlation.py`

Computes rolling pairwise correlations to explore whether relationships
between entities change over time. Rolling correlations are descriptive
and can be sensitive to window choice.

## Baseline forecasting

### `Baseline_Forecastability.py`

Initial baseline forecasting experiment and metrics. This was an earlier
preliminary benchmark.

### `Baseline_Forecastability_refined.py`

Refined baseline experiment using a calendar-defined 2025 evaluation
period and rolling one-day-ahead predictions. Produces per-entity and
summary results.

### `Baseline_Error_Analysis.py`

Analyzes baseline forecast errors across months, weekdays, entities,
demand changes, and notable forecast errors. Outputs tables and
visualizations.

### `Baseline_Transition_Analysis.py`

Analyzes errors associated with weekday transitions, weekend/weekday
differences, large changes, and entity-size groups.

### `Baseline_Integrity_Audit.py`

Audits baseline evaluation alignment and computes baseline metrics on a
common sample. Checks calendar integrity, forecast availability, common
sample counts, and macro/pooled/entity metrics.

### `Baseline_Coverage_Investigation.py`

Investigates why the common baseline sample covers fewer dates than the
full 2025 calendar. Checks method-specific daily availability and common
evaluation dates.

### `RoleA_Evidence.py`

Lists project files and outputs to provide an evidence inventory of Role
A work.

### `RoleA_Consolidation.py`

Creates a manifest of selected evidence files, checking their existence
and basic dimensions. The initial manifest used `EDA/outputs` for every
item, so dataset-audit files stored in `EDA/EDA/outputs` were marked
missing despite existing.

------------------------------------------------------------------------

# 2. Dataset and panel outputs

These files are currently located under `EDA/EDA/outputs/`.

### `entity_coverage.csv`

Entity-level coverage summary. Useful for checking the number of records
and coverage characteristics for each entity.

### `missing_values.csv`

Missingness summary for dataset variables.

### `negative_values.csv`

Audit of negative-valued fields. Negative values are not automatically
errors: variables representing signed deviations or balances can
legitimately be negative.

### `research_panel_2020_2025.parquet`

A prepared panel covering the research period used in exploratory work.
This is a working analytical dataset, not the original raw source.

### `split_distribution.csv`

Summary of records across dataset split categories.

### `state_demand_statistics.csv`

State/entity-level demand summary statistics.

------------------------------------------------------------------------

# 3. Demand and temporal-analysis outputs

These are generally located in `EDA/outputs/`.

### `aggregate_daily_demand.csv`

Daily aggregate demand series used in aggregate time-series exploration.

### `annual_aggregate_demand.csv`

Year-level summary of aggregate demand for the analyzed period.

### `monthly_demand_profile.csv`

Month-of-year demand profile used to examine seasonality.

### `day_of_week_demand.csv`

Day-of-week demand profile.

### `entity_demand_statistics.csv`

Entity-level descriptive demand statistics, including scale and
distribution characteristics.

### `entity_year_demand.csv`

Year-wise demand summary for each entity.

### `entity_monthly_demand.csv`

Monthly demand summaries at the entity level.

### `entity_growth_2020_2025.csv`

Entity-level growth comparison across the 2020--2025 period.

### `entity_covid_comparison.csv`

Comparison of entity demand across COVID-era/reference periods, as
defined by the analysis script.

### `entity_growth_variability.png`

Visualization of entity growth and/or variation.

### `entity_mean_demand.png`

Plot comparing average demand across entities.

### `entity_demand_variability.png`

Visualization of demand variability across entities.

### `monthly_seasonality.png`

Visualization of monthly demand patterns.

### `day_of_week_pattern.png`

Visualization of weekday-related demand behavior.

### `normalized_annual_profiles.png`

Normalized annual demand profiles intended to compare shapes/patterns
while reducing differences in absolute demand scale.

### `aggregate_demand_timeseries.png`

Aggregate demand time-series visualization.

------------------------------------------------------------------------

# 4. Regime-analysis outputs

### `aggregate_regime_statistics.csv`

Summary statistics for candidate segments identified by the change-point
analysis, including segment-level characteristics.

### `regime_annual_validation.csv`

Annual summaries used to compare or contextualize candidate regime
patterns.

### `regime_quarterly_validation.csv`

Quarterly summaries used to examine demand variation within years.

### `aggregate_regime_detection.png`

Plot of aggregate demand with candidate detected change points/segments.

### `regime_rolling_mean.png`

Rolling-mean visualization used to contextualize the time series and its
evolving level.

### `Regime_Validation.png`

Visualization associated with the regime-validation analysis.

**Interpretation caution:** Change points are algorithm-dependent. The
current exploratory detection identified many candidate segments; their
number and timing should not be treated as confirmed events without
sensitivity testing and external/contextual validation.

------------------------------------------------------------------------

# 5. Predictability-profile outputs

### `entity_predictability_profile.csv`

Entity-level descriptive predictability profile, including temporal
characteristics such as autocorrelation and normalized changes. This
helps characterize series, but forecast difficulty must ultimately be
measured on a consistent out-of-sample evaluation.

### `entity_variability.csv`

Entity-level variability statistics.

### `baseline_entity_difficulty.csv`

Entity-level difficulty indicators derived from baseline forecast
results. These are benchmark-relative and dependent on the evaluation
protocol.

------------------------------------------------------------------------

# 6. Weather-analysis outputs

### `weather_raw_correlations.csv`

Raw weather--demand correlations.

### `weather_overall_correlation.csv`

Aggregate weather correlation summary.

### `weather_entity_correlations.csv`

Entity-level weather correlation results.

### `weather_monthly_adjusted_correlations.csv`

Weather--demand correlations after accounting for month-of-year effects.

### `weather_trend_calendar_adjusted.csv`

Weather--demand correlations using trend/calendar-adjusted or
residualized series.

### `weather_correlation_comparison.csv`

Comparison across weather-correlation approaches.

### `weather_entity_correlation_heatmap.png`

Visual summary of entity-level weather relationships.

**Interpretation caution:** Correlations, including adjusted
correlations, are exploratory. They do not establish causality, data
availability at forecast time, or that weather features improve
predictive performance.

------------------------------------------------------------------------

# 7. Interstate relationship outputs

### `interstate_demand_correlation_matrix.csv`

Pairwise raw demand correlation matrix.

### `interstate_correlation_pairs.csv`

Pairwise correlation values organized as entity pairs.

### `interstate_demand_residuals.csv`

Residualized/adjusted demand series used for the residual relationship
analysis.

### `interstate_residual_correlations.csv`

Pairwise correlations calculated from residualized series.

### `interstate_rolling_correlations.csv`

Rolling correlation observations for selected or evaluated entity pairs.

### `interstate_correlation_heatmap.png`

Heatmap of pairwise demand relationships.

**Interpretation caution:** Strong correlation does not establish direct
transmission of demand influence between entities. Shared seasonality,
national trends, weather, and other common factors may contribute.

------------------------------------------------------------------------

# 8. Baseline forecasting outputs

## Earlier benchmark files

### `baseline_forecast_results.csv`

Forecast-level results from the earlier baseline experiment.

### `baseline_forecast_summary.csv`

Summary metrics from the earlier baseline experiment.

### `best_baseline_by_entity.csv`

Best baseline method by entity under the earlier experiment's selection
criterion.

These earlier outputs should not be confused with the refined 2025
common-sample audit.

## Refined baseline files

### `refined_baseline_entity_results.csv`

Entity-level baseline results from the refined 2025 evaluation.

### `refined_baseline_macro_summary.csv`

Macro summary of refined baseline metrics.

### `refined_baseline_panel_summary.csv`

Pooled/panel summary of refined baseline metrics.

### `baseline_entity_difficulty.csv`

Entity difficulty ranking or descriptive summary based on baseline
performance.

## Integrity-audit directory: `outputs/baseline_integrity_audit/`

### `calendar_integrity.csv`

Per-entity calendar coverage and duplicate/missing calendar-day checks.

### `common_evaluation_predictions.csv`

Actual values and predictions for all baseline methods on the common
evaluation sample.

### `daily_method_coverage.csv`

Daily counts of available actuals and forecasts for each method.

### `date_coverage.csv`

Coverage of common evaluation observations by date.

### `entity_metrics.csv`

Per-entity metrics for each baseline method, using the audited common
sample.

### `macro_metrics.csv`

Macro metrics: averages across entities, giving each entity equal
weight.

### `pooled_metrics.csv`

Metrics computed across all observations pooled together, weighting
entities according to their number of included observations.

### `sample_alignment.csv`

Method-specific sample availability and common-sample counts.

## Error-analysis outputs

### `baseline_monthly_error.csv`

Baseline errors summarized by calendar month.

### `baseline_weekday_error.csv`

Baseline errors summarized by weekday.

### `baseline_transition_results.csv`

Error summaries associated with transitions between weekdays.

### `baseline_sunday_monday_results.csv`

Specific Sunday/Monday error comparison.

### `baseline_weekend_weekday_results.csv`

Comparison between weekend and weekday performance.

### `baseline_size_group_results.csv`

Performance grouped by entity demand-size categories.

### `baseline_size_change_results.csv`

Performance organized by entity size and demand-change categories.

### `baseline_large_change_comparison.csv`

Comparison of forecast errors for large-change versus ordinary-change
observations.

### `baseline_large_demand_changes.csv`

Large demand-change events identified by the analysis.

### `baseline_change_analysis.csv`

Demand-change/error analysis data.

### `baseline_error_change_correlation.csv`

Correlation analysis between demand changes and forecast error. For
persistence forecasts, change and forecast error can be mathematically
linked; this relationship should not be interpreted as independent
evidence.

### `baseline_entity_error_analysis.csv`

Entity-level error-analysis results.

### `baseline_worst_100_forecasts.csv`

Records of the 100 worst forecast cases according to the script's
ranking criterion.

### `baseline_error_distribution.png`

Visualization of the baseline error distribution.

### `baseline_monthly_mae.png`

Monthly MAE visualization.

### `baseline_monthly_smape.png`

Monthly sMAPE visualization.

### `baseline_transition_smape.png`

Visualization of transition-related sMAPE.

### `baseline_size_group_smape.png`

Visualization of performance across entity-size groups.

### `baseline_change_category_smape.png`

Visualization comparing error across demand-change categories.

### `baseline_change_vs_error.png`

Visualization of demand changes versus forecast errors.

------------------------------------------------------------------------

# 9. Preliminary baseline findings recorded so far

The audited common-sample evaluation uses 34 entities and 11,934
observations across 351 dates, from 15 January through 31 December 2025.
The first 14 days were excluded from the common sample because the
14-day moving-average method lacked its full initialization window; this
is not due to missing actual demand values.

Audited common-sample metrics:

  Method                 Macro MAE   Macro RMSE   Macro sMAPE
  -------------------- ----------- ------------ -------------
  Persistence               4.7092       6.6468        4.6626
  Moving Average 7          7.0575       9.7459        5.7487
  Moving Average 14         8.4519      11.3608        6.4633
  Weekly Persistence       10.2416      14.2106        7.8920

Persistence is the strongest of these four simple methods on this
audited common sample. This is a benchmark result, not a conclusion
about deep-learning architectures.

Previously observed error-analysis patterns include: - Errors vary by
month. - Sunday and Monday were among the higher-error weekdays in the
exploratory benchmark. - Weekend errors were higher than weekday errors
in the preliminary comparison. - Large-change observations were
substantially harder for persistence forecasting. - Error behavior
differs by entity scale, and relative metrics can behave differently for
very small entities.

These are descriptive results and should be interpreted according to the
specific evaluation protocol and data subsets used.

------------------------------------------------------------------------

# 10. Role A completion status

Role A's exploratory and baseline analytical responsibilities are
considered **complete**:

-   Dataset integrity exploration
-   Missingness exploration
-   Demand characterization
-   Candidate regime exploration and descriptive validation
-   Entity growth and variability analysis
-   Weather correlation and adjustment explorations
-   Inter-state raw, residual, and rolling correlation analysis
-   Predictability profiling
-   Baseline forecasting
-   Baseline error analysis
-   Baseline integrity and coverage investigation
-   Evidence inventory

Report consolidation is documentation work and does not require further
exploratory scripts unless a specific evidence gap is identified.

## 11. Important limitations

-   Exploratory correlations do not establish causality.
-   Adjusted correlations do not automatically prove forecast utility.
-   Candidate change points need method sensitivity and validation.
-   The refined baseline is a rolling one-day-ahead benchmark, not a
    fixed-origin multi-day forecast.
-   Common evaluation samples are used to ensure comparability across
    methods.
-   Forecast features must be checked for realistic availability at
    prediction time.
-   Missing target values should not be silently imputed.
-   Small entities may have low absolute errors but comparatively high
    percentage errors.
-   Results from preliminary and refined baseline scripts should be
    distinguished.
-   Descriptive entity profiles should not be mistaken for
    forecast-performance measurements.

------------------------------------------------------------------------

## How to use this directory

1.  **Raw data:** Keep source data in `Data/Raw/`; avoid overwriting it.
2.  **Scripts:** Run analysis scripts from the project environment.
3.  **Outputs:** CSVs and figures in `EDA/outputs/` are generated
    evidence. Treat them as outputs rather than manually edited data.
4.  **Audit results:** Prefer the audited common-sample baseline metrics
    when comparing baseline methods.
5.  **Modeling phase:** Use Role A findings to design controlled
    experiments testing historical demand, weather, neighboring-state
    information, and model architectures.
