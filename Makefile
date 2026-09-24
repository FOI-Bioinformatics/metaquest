.PHONY: env env-dev test test-network lint format check build clean pipeline install help dev-install

help:
	@echo "Available commands:"
	@echo "  env         - Create or update the 'metaquest' conda environment (Python 3.12)"
	@echo "  env-dev     - Install the package with dev extras into the 'metaquest' conda environment"
	@echo "  test        - Run tests with coverage"
	@echo "  lint        - Run flake8 linting"
	@echo "  format      - Format code with black"
	@echo "  check       - Run all quality checks (format, lint, type check)"
	@echo "  build       - Build distribution packages"
	@echo "  clean       - Clean build artifacts and cache"
	@echo "  pipeline    - Run full integration test pipeline"
	@echo "  install     - Install package for development"
	@echo "  dev-install - Install with development dependencies"

env:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

env-dev:
	conda run -n metaquest pip install -e ".[dev]"

dev-install:
	python -m pip install -e ".[dev]"

install:
	python -m pip install -e .

test:
	python -m pytest tests/ --cov=metaquest

test-network:
	python -m pytest tests/test_network_smoke.py -m network -x -v

lint:
	python -m flake8 metaquest tests

format:
	python -m black metaquest tests

check:
	@echo "Running under $$(python --version)..."
	@echo "Running format check..."
	python -m black --check --diff metaquest tests
	@echo "Running linting..."
	python -m flake8 metaquest tests
	@echo "Running type check..."
	python -m mypy metaquest
	@echo "Guarding against the frozen plotly-latest CDN alias..."
	@if grep -rn "cdn.plot.ly/plotly-latest" metaquest --include='*.py'; then \
		echo "ERROR: use metaquest.utils.html.plotly_cdn_script() instead of the plotly-latest alias"; \
		exit 1; \
	fi
	@echo "Checking cyclomatic complexity ceiling (fail on rank D or worse)..."
	@output=$$(python -m radon cc metaquest -n D -s); \
	if [ -n "$$output" ]; then \
		echo "$$output"; \
		echo "ERROR: functions at complexity rank D or worse; refactor before merging"; \
		exit 1; \
	fi
	@echo "All quality checks passed!"

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete

pipeline:
	bash local_test.sh
