.PHONY: install test lint format docs clean

# Python environment
VENV = .venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

# Development setup
install:
	python -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements.txt

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ --cov=micromouse --cov-report=term-missing

# Linting and formatting
lint:
	$(PYTHON) -m flake8 micromouse/ tests/
	$(PYTHON) -m mypy micromouse/ tests/

format:
	$(PYTHON) -m black micromouse/ tests/
	$(PYTHON) -m isort micromouse/ tests/

# Documentation
docs:
	$(PYTHON) -m sphinx -b html docs/ docs/_build/html

# Cleanup
clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf docs/_build
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development server
run:
	$(PYTHON) micromouse/main.py

# Java simulator build
build-simulator:
	cd simulator && ./gradlew build

# Performance benchmarks
benchmark:
	$(PYTHON) -m micromouse.benchmark

# Type checking
typecheck:
	$(PYTHON) -m mypy micromouse/ tests/

# Security checks
security:
	$(PIP) install safety
	$(PYTHON) -m safety check 