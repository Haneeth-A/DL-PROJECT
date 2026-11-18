# Data dictionary

Source: RE-FUSED v1.0 (`Data/Raw/RE-FUSED_India_Datasets_v1.0/`), 74,528 rows × 107 columns,
2020-01-01 … 2025-12-31, 34 entities. Units follow the source registry unless stated.

## Modelling view (`Data/processed/model_panel.parquet`)

| Column | Unit | Role | Missing (all rows) |
|---|---|---|---|
| `entity` | id | key | 0 |
| `date` | date | key | 0 |
| `y` (`dem_energy_met_gwh`) | GWh/day | target | 3.11 % |
| `wx_tmin_c` | °C | I1 | TBD |
| `wx_tmean_c` | °C | I1 | TBD |
| `wx_tmax_c` | °C | I1 | TBD |
| `wx_rain_mm` | mm | I1 | TBD |
| `wx_cdd24` | degree-days | I1 | TBD |
| `wx_hdd18` | degree-days | I1 | TBD |
| `cal_dow` | 0–6 | calendar | 0 |
| `cal_month` | 1–12 | calendar | 0 |
| `cal_doy` | 1–366 | calendar | 0 |
| `cal_fy` | year | calendar | 0 |
| `regime` | label | evaluation strata | 0 |
| `split` | label | D16 split | 0 |
| `source_split` | label | RE-FUSED's own split; audit only | 0 |

Not available in v1.0: humidity, holiday flag, inter-state flows. No holiday flag is proxied; all
thresholds that would need one are stated in the methodology instead.

## Source blocks

| Block | Examples | Unit | Used? |
|---|---|---|---|
| demand | `dem_energy_met_gwh`, `dem_max_demand_mw`, `dem_energy_shortage_gwh` | GWh/day, MW | target only |
| deviation | `dev_*` | GWh, MW | no (post-hoc operational) |
| regional/national | `reg_*`, `nat_*` | GWh, MW | no (aggregates include the entity) |
| system | `sys_fvi`, `sys_pct_*` | Hz, % | no |
| price | `prc_dam_acp`, `prc_rtm_acp`, `prc_dsm_*` | INR/kWh | no (not known at origin) |
| generation | `gen_cap_mw_*`, `gen_act_gwh_*`, `gen_outage_*` | MW, GWh | no |
| renewable | `re_all_*`, `re_ctrl_*` | GWh | no |
| emissions | `co2_*` | t, t/MWh | no |
| weather | `wx_*` | °C, mm, degree-days | yes (6) |
| calendar/keys | `entity`, `date`, `region`, `bid_area`, `pilot18`, `avail_*`, `qc_*`, `split` | — | keys + split audit |

## Dropped or excluded columns

| Column(s) | Reason |
|---|---|
| all `prc_*` | day-ahead price is not observed at forecast origin; would leak |
| all `dev_*` | computed after delivery; not available at origin |
| `reg_*`, `nat_*`, `sys_*` | contain the entity's own demand → target leakage |
| `gen_*`, `re_*`, `co2_*` | realised generation; contemporaneous with the target |
| `qc_*`, `avail_*` | source quality flags, not predictors |

Every exclusion is recorded in `Data/processed/variables_dropped.csv` (A). Any change here is a
`DECISIONS.md` entry.

## Known defects (fix before freeze)

- `Data/processed/entity_list.csv` has a constant `entity_count = 34` (should be rows per entity).
- `panel_metadata.json` and `source_sha256.txt` contain absolute Windows paths.
- `column_mapping.csv` classifies `cal_*` as "Other".
- `EDA/EDA/` is a duplicate of `EDA/outputs/` produced by a nested run; ignored in git.
