# Makefile — commandes opérationnelles du projet CIF Credit Intelligence
.PHONY: install lint typecheck test generate train evaluate decision api stack up down clean pre-commit

VENV := .venv
PY := $(VENV)/Scripts/python.exe

install:
	python -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

test:
	pytest -q

generate:
	$(PY) -m cif_credit.cli.generate

train:
	$(PY) -m cif_credit.cli.train

evaluate:
	$(PY) -m cif_credit.cli.evaluate

decision:
	$(PY) -m cif_credit.cli.decision

api:
	$(PY) -m uvicorn cif_credit.serving.api:create_app --factory --host 0.0.0.0 --port 8000

dagster:
	dagster dev -f pipelines/definitions.py -h 0.0.0.0 -p 3000

pre-commit:
	pre-commit install
	pre-commit run --all-files

stack:
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache mlruns mlruns.db data/raw data/processed data/artifacts