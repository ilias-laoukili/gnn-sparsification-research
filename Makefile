.PHONY: help setup install test clean train lint format docs

help:
	@echo "GNN Sparsification Research - Available Commands"
	@echo "================================================"
	@echo "setup      - Create venv and install all dependencies"
	@echo "install    - Install package in editable mode"
	@echo "test       - Run all tests with coverage"
	@echo "test-fast  - Run tests without coverage"
	@echo "lint       - Run linting checks (flake8)"
	@echo "format     - Format code with black and isort"
	@echo "clean      - Remove cache and generated files"
	@echo "train      - Run training with default config"
	@echo "docs       - Open documentation"
	@echo ""

setup:
	@echo "Running setup script..."
	@bash setup.sh

install:
	@echo "Installing package in editable mode..."
	pip install -e .

test:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term tests/
	@echo "Coverage report generated in htmlcov/"

test-fast:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Running linting checks..."
	flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "Formatting code..."
	black src/ scripts/ tests/ --line-length=100
	isort src/ scripts/ tests/ --profile black

clean:
	@echo "Cleaning cache and generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf outputs/
	@echo "Cleanup complete!"

train:
	@echo "Running training with default configuration..."
	python scripts/train.py

docs:
	@echo "Opening documentation..."
	@cat docs/QUICKSTART.md

# Specific experiment shortcuts
train-baseline:
	python scripts/train.py experiment=baseline

train-jaccard:
	python scripts/train.py sparsifier=jaccard

train-spectral:
	python scripts/train.py sparsifier=spectral

# Dataset-specific training
train-cora:
	python scripts/train.py dataset=cora

train-pubmed:
	python scripts/train.py dataset=pubmed

# Model-specific training
train-gcn:
	python scripts/train.py model=gcn

train-gat:
	python scripts/train.py model=gat
