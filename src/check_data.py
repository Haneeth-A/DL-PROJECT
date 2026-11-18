"""Verify the committed data artefacts before any run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
SOURCE_PARQUET = (
    ROOT
    / "Data"
    / "Raw"
    / "RE-FUSED_India_Datasets_v1.0"
    / "RE-FUSED"
    / "India_Grid_Study"
    / "data"
    / "processed"
    / "refused_state_day.parquet"
)
EXPECTED_SHA256 = "0b36fa40160128c83555240319a9a1f219b5ef18ce1fa894d53cb4bd2a38e747"
EXPECTED_ROWS = 74528
EXPECTED_COLS = 107
EXPECTED_ENTITIES = 34


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_source_hash() -> bool:
    if not SOURCE_PARQUET.exists():
        print("skip: raw parquet absent (gitignored); cannot verify source hash")
        return True
    actual = sha256(SOURCE_PARQUET)
    if actual != EXPECTED_SHA256:
        print(f"FAIL source hash {actual} != {EXPECTED_SHA256}")
        return False
    print("ok: source SHA256")
    return True


def check_panel() -> bool:
    path = PROCESSED / "panel.parquet"
    if not path.exists():
        print(f"FAIL missing {path}")
        return False
    df = pd.read_parquet(path)
    ok = True
    for label, actual, expected in [
        ("rows", len(df), EXPECTED_ROWS),
        ("columns", df.shape[1], EXPECTED_COLS),
        ("entities", df["entity"].nunique(), EXPECTED_ENTITIES),
    ]:
        if actual != expected:
            print(f"FAIL {label}: {actual} != {expected}")
            ok = False
    for col in ("entity", "date", "dem_energy_met_gwh"):
        if col not in df.columns:
            print(f"FAIL missing column {col}")
            ok = False
    meta = json.loads((PROCESSED / "panel_metadata.json").read_text())
    if meta.get("flow_columns"):
        print(f"note: metadata lists flow columns {meta['flow_columns']}")
    if ok:
        print(f"ok: panel {df.shape}, {df['date'].min().date()} … {df['date'].max().date()}")
    return ok


def main() -> int:
    return 0 if (check_source_hash() & check_panel()) else 1


if __name__ == "__main__":
    sys.exit(main())
