from pathlib import Path

PROJECT = Path(r"C:\IIIT Raichur\Projects\DL Project")

EDA = PROJECT / "EDA"

print("=" * 70)
print("ROLE A — EVIDENCE INVENTORY")
print("=" * 70)

for folder in sorted(EDA.rglob("*")):

    if folder.is_file():

        if any(
            excluded in folder.parts
            for excluded in [".venv", "__pycache__"]
        ):
            continue

        print(folder.relative_to(PROJECT))