# Changelog

## [2025-11-05] - Pydantic v2 Migration & Docker Setup

### Added
- Pydantic v2 support with updated model configurations
- New environment variables for API authentication control
- Enhanced error handling for database operations
- Comprehensive test suite for core functionality

### Changed
- Migrated all Pydantic models to v2 syntax
- Updated configuration management with `SettingsConfigDict`
- Improved database connection handling in Docker
- Optimized Docker build process

### Fixed
- Resolved SQLAlchemy async engine initialization
- Fixed Redis serialization issues with datetime objects
- Corrected API key authentication flow
- Addressed test suite configuration issues

## [2025-11-04] - TimescaleDB & API Endpoint Updates

### Added
- New API endpoints for real-time market data ingestion
- Health check endpoint for monitoring service status
- Detailed error responses for API endpoints
- Database migration scripts for TimescaleDB schema updates

### Changed
- Updated database connection handling for better reliability
- Improved error messages for database connection issues
- Enhanced API documentation with examples
- Optimized query performance for time-series data

### Fixed
- Resolved TimescaleDB connection issues in Docker environment
- Fixed SQL query execution in DataIngestor.initialize()
- Addressed race condition in Redis client initialization
- Corrected container networking in docker-compose.yml

## [2025-11-03] - Neo4j Connection Fixes

### Fixed

1. **Neo4j Connection Issues**
   - Changed connection protocol from `neo4j://` to `bolt://` for direct connection to a single Neo4j instance
   - Updated `NEO4J_URI` in `src/utils/config.py` to use `bolt://localhost:7687`
   - Resolved "Unable to retrieve routing information" error by using the correct protocol

2. **Synchronous Driver Implementation**
   - Made `GraphitiClient` methods synchronous to match the Neo4j Python driver's behavior
   - Removed `async/await` from connection handling methods
   - Added proper error handling and logging for connection attempts

3. **Redis Client Initialization**
   - Added missing Redis client import in `src/api/main.py`
   - Initialized Redis client at module level
   - Fixed async/await usage in the FastAPI startup event

4. **Code Cleanup**
   - Removed duplicate exception handling
   - Improved error messages and logging
   - Fixed potential resource leaks in connection handling

### Configuration Changes

1. **config.py**
   ```python
   # Before
   NEO4J_URI: str = "neo4j://localhost:7687"
   
   # After
   NEO4J_URI: str = "bolt://localhost:7687"  # Using bolt:// for direct connection
   ```

2. **schema.py**
   - Converted async methods to synchronous:
     - `connect()`
     - `close()`
     - `initialize_schema()`
   - Added proper connection testing with `session.run("RETURN 1")`

3. **main.py**
   - Added Redis client import and initialization
   - Updated startup event to handle both sync and async operations correctly
   - Improved error handling and logging

### Verification Steps

1. **Neo4j Connection**
   - Run the application: `poetry run uvicorn src.api.main:app --reload`
   - Check logs for "Successfully connected to Neo4j" message
   - Verify connection at `http://localhost:7474` (Neo4j Browser)

2. **API Endpoints**
   - Access Swagger UI at `http://localhost:8000/docs`
   - Test health check endpoint: `GET /health`

### Notes
- The application now uses synchronous Neo4j driver for better stability
- All database connections are properly closed on application shutdown
- Error messages are more descriptive for easier debugging

---
*This document will be updated with future changes. Please refer to git history for detailed commit information.*
