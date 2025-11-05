# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive API documentation in `docs/API_REFERENCE.md`
- Detailed testing guide in `docs/TESTING_GUIDE.md`
- New test cases for LLM services
- Rate limiting middleware for API endpoints

### Changed
- Upgraded to Pydantic v2 with new model patterns
- Replaced deprecated `min_items`/`max_items` with `min_length`/`max_length`
- Updated FastAPI event handlers to use lifespan context manager
- Improved error handling and validation
- Enhanced test coverage and reliability

### Fixed
- Fixed event loop issues in async tests
- Resolved Pydantic deprecation warnings
- Fixed Redis connection handling during shutdown
- Addressed race conditions in test execution

## [0.1.0] - 2023-11-05

### Added
- Initial project setup
- Basic API endpoints for text generation and embeddings
- Integration with Together.AI services
- Redis caching layer
- Unit tests for core functionality
