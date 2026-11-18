# Data provenance

## Source

**RE-FUSED** — state-day panel of India's power system.

- Kumar, R. (2026). *RE-FUSED: State-day panel of India's power system from official data…* (v1.0.0)
  [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23301897
- Record: https://zenodo.org/records/23685972 (the record now serves v1.1; **this repo uses v1.0**,
  downloaded 18 Nov 2026).
- Licence: data and documentation **CC BY 4.0**; code MIT. No endorsement implied.

## What is in git

| Path | Committed? | Notes |
|---|---|---|
| `Data/Raw/` | no | original zip + extracted parquet, 39 MB, gitignored |
| `Data/processed/panel.parquet` | yes | 74,528 × 107, 2020-01-01 … 2025-12-31, 34 entities |
| `Data/processed/*.csv`, `*.json` | yes | metadata, inventory, missingness, checksums |

Source hash (SHA256 of `refused_state_day.parquet`):
`0b36fa40160128c83555240319a9a1f219b5ef18ce1fa894d53cb4bd2a38e747`.

## Changes we made

- Subset to 2020-01-01 … 2025-12-31.
- Added calendar features (`cal_*`), D16 split labels, and ex-ante regime labels.
- No imputation: missing targets are preserved as NaN and handled inside the training pipeline.
- Added a 90-day lookback modelling view (`model_panel.parquet`).

## What we do not redistribute

Raw Grid-India / CEA / IMD source documents are not openly licensed and are not included. Use the
Zenodo archive to rebuild `Data/Raw/`.
