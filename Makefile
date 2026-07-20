# DRIVYX make targets (CLAUDE.md section 15).

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup stage test test-device test-all gui lint fmt clean

help:
	@echo "DRIVYX targets:"
	@echo "  setup        Bootstrap the Orin environment (venv, CUDA torch, PyQt6)."
	@echo "  stage        Extract the IDD archives into the data root."
	@echo "  test         Run the CPU test suite (default; excludes device tests)."
	@echo "  test-device  Run the Orin-only tests (CUDA, real data, TensorRT)."
	@echo "  test-all     Run every test."
	@echo "  gui          Launch the desktop application."
	@echo "  lint         ruff check + format check."
	@echo "  fmt          Apply ruff formatting and autofixes."

setup:
	bash scripts/setup_orin.sh

stage:
	bash scripts/stage_data.sh

test:
	$(PY) -m pytest

test-device:
	$(PY) -m pytest -m device

test-all:
	$(PY) -m pytest -m ""

gui:
	$(VENV)/bin/drivyx-gui

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
