# Changelog

All notable changes to MetaQuest will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2024-08-26

### Added
- Modular CLI architecture with individual command classes
- Comprehensive security layer for subprocess operations
- Centralized constants module for configuration management
- Enhanced documentation with architecture guide
- Code quality improvements with Black formatting and flake8 compliance
- Improved error handling and logging throughout the application

### Changed
- **BREAKING**: Refactored CLI from monolithic parser to modular command system
- Enhanced security with input validation and subprocess sanitization
- Improved plugin system with better error handling
- Updated GitHub Actions to use current versions (v4/v5)
- Standardized code formatting across entire codebase

### Security
- Added comprehensive input validation for SRA accessions
- Implemented secure subprocess wrapper to prevent command injection
- Added path traversal protection for file operations
- Environment variable sanitization for subprocess calls

### Fixed
- Reduced cyclomatic complexity in visualization functions
- Fixed Black formatting conflicts with flake8 E203
- Corrected import issues and unused variable warnings
- Improved memory management for large dataset processing

### Developer Experience
- Added comprehensive test coverage framework
- Implemented make commands for development workflow
- Enhanced documentation with contributing guidelines
- Added type hints and improved code organization

## [0.3.0] - Previous Release

### Added
- Basic Branchwater file processing
- SRA metadata downloading from NCBI
- Containment analysis and parsing
- Visualization capabilities for plots and metadata
- Plugin system for format handling

### Features
- Command-line interface for all operations
- Support for multiple file formats (Branchwater, Mastiff)
- Parallel processing for SRA downloads
- Metadata extraction and analysis tools

## [Unreleased]

### Planned
- Web API interface for remote access
- Enhanced visualization options
- Caching system for expensive operations  
- Async processing for I/O operations
- Extended plugin system for custom analyzers

---

## Migration Guide

### Upgrading from 0.3.0 to 0.3.1

The core functionality remains the same, but the CLI architecture has been significantly improved. All existing command-line usage patterns continue to work without changes.

#### What's Compatible
- All CLI commands and their arguments remain unchanged
- Plugin interfaces are backward compatible
- Configuration files and data formats are unchanged

#### What's New
- Enhanced security with input validation
- Improved error messages and logging
- Better performance for large datasets
- More robust handling of edge cases

#### Development Changes
If you're contributing to the codebase:
- CLI commands are now in separate modules under `metaquest/cli/commands/`
- Security utilities are available in `metaquest/utils/security.py`
- Constants are centralized in `metaquest/core/constants.py`
- Follow new code quality standards (Black formatting, flake8 compliance)

For detailed technical changes, see the [Architecture Guide](ARCHITECTURE.md).