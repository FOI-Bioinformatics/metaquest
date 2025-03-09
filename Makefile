.PHONY: test lint format check build clean pipeline

test:
    pytest tests/ --cov=metaquest

lint:
    flake8 metaquest

format:
    black metaquest

check:
    black --check --diff metaquest
    flake8 metaquest
    mypy --ignore-missing-imports metaquest

build:
    python -m build
    check-wheel-contents dist/*.whl
    twine check dist/*

clean:
    rm -rf build/ dist/ *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

pipeline:
    bash local_test.sh
