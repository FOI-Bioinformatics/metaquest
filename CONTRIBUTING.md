# Contributing to MetaQuest

Thank you for your interest in contributing to MetaQuest! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FOI-Bioinformatics/MetaQuest.git
   cd MetaQuest
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run the test suite**:
   ```bash
   make test
   ```

## Development Workflow

1. **Code Quality**: Before submitting changes, ensure your code passes all quality checks:
   ```bash
   make check
   ```
   This runs:
   - `black` for code formatting
   - `flake8` for linting
   - `mypy` for type checking

2. **Auto-format code**:
   ```bash
   make format
   ```

3. **Run tests**:
   ```bash
   make test        # Run tests with coverage
   make pipeline    # Run full integration test
   ```

## Architecture Guidelines

### CLI Commands
- All commands should inherit from base classes in `cli/base.py`
- Register new commands in `cli/main.py`
- Follow existing argument naming conventions (use dashes, not underscores)

### Plugin Development
- Format plugins inherit from `Plugin` class in `plugins/base.py`
- Register with appropriate registries (`format_registry`, `visualizer_registry`)
- Implement required methods: `validate_header`, `parse_file`, `extract_metadata`

### Data Models
- Use dataclasses in `core/models.py` for data structures
- Follow the existing patterns for `Containment`, `SRAMetadata`, etc.

### Error Handling
- Use custom exceptions from `core/exceptions.py`
- Follow the hierarchy starting with `MetaQuestError`

## Testing

- Write tests for new functionality in the `tests/` directory
- Ensure both unit tests and integration tests pass
- The `local_test.sh` script provides end-to-end testing

## Submitting Changes

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes** following the guidelines above
3. **Run quality checks**: `make check`
4. **Run tests**: `make test`
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push and create a pull request**

## Code Style

- Follow PEP 8 Python style guidelines
- Use `black` for automatic code formatting (line length: 88)
- Use type hints where appropriate
- Write clear docstrings for functions and classes

## Pull Request Guidelines

- Provide a clear description of the changes
- Include tests for new functionality
- Ensure all tests pass
- Update documentation if necessary
- Reference any related issues

## Questions?

If you have questions about contributing, please open an issue or start a discussion in the repository.