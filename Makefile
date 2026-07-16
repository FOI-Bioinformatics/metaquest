.PHONY: test lint format check build clean pipeline install help dev-install

help:
	@echo "Available commands:"
	@echo "  test        - Run tests with coverage"
	@echo "  lint        - Run flake8 linting"
	@echo "  format      - Format code with black"
	@echo "  check       - Run all quality checks (format, lint, type check)"
	@echo "  build       - Build distribution packages"
	@echo "  clean       - Clean build artifacts and cache"
	@echo "  pipeline    - Run full integration test pipeline"
	@echo "  install     - Install package for development"
	@echo "  dev-install - Install with development dependencies"

dev-install:
	pip install -e ".[dev]"

install:
	pip install -e .

test:
	pytest tests/ --cov=metaquest

lint:
	flake8 metaquest tests

format:
	black metaquest tests

check:
	@echo "Running format check..."
	black --check --diff metaquest tests
	@echo "Running linting..."
	flake8 metaquest tests
	@echo "Running type check..."
	mypy metaquest
	@echo "Guarding against the frozen plotly-latest CDN alias..."
	@if grep -rn "cdn.plot.ly/plotly-latest" metaquest --include='*.py'; then \
		echo "ERROR: use metaquest.utils.html.plotly_cdn_script() instead of the plotly-latest alias"; \
		exit 1; \
	fi
	@echo "Checking cyclomatic complexity ceiling (fail on rank D or worse)..."
	@output=$$(radon cc metaquest -n D -s); \
	if [ -n "$$output" ]; then \
		echo "$$output"; \
		echo "ERROR: functions at complexity rank D or worse; refactor before merging"; \
		exit 1; \
	fi
	@echo "All quality checks passed!"

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/ test_data/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete

pipeline:
	bash local_test.sh
