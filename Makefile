PY ?= python

.PHONY: help setup check data train eval tables figures all clean

help:
	@echo "setup    install dependencies"
	@echo "check    verify data files and hashes"
	@echo "data     build model panel, splits, regimes, neighbours"
	@echo "train    train all cells x seeds"
	@echo "eval     metrics + statistical tests"
	@echo "tables   generate tables"
	@echo "figures  generate figures"
	@echo "all      data train eval tables figures"

setup:
	$(PY) -m pip install -r requirements.txt

check:
	$(PY) -m src.check_data

data:
	$(PY) -m src.data

train:
	$(PY) -m src.train

eval:
	$(PY) -m src.evaluate

tables:
	$(PY) -m src.tables

figures:
	$(PY) -m src.figures

all: data train eval tables figures

clean:
	rm -rf __pycache__ src/__pycache__ src/models/__pycache__ .pytest_cache
