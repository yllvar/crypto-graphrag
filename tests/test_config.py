"""Test configuration settings."""
import os
from unittest.mock import MagicMock

# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/1"
TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_NEO4J_USER = "neo4j"
TEST_NEO4J_PASSWORD = "testpassword"

# Mock environment variables for testing
os.environ["TIMESCALEDB_URI"] = TEST_DATABASE_URL
os.environ["REDIS_URL"] = TEST_REDIS_URL
os.environ["NEO4J_URI"] = TEST_NEO4J_URI
os.environ["NEO4J_USER"] = TEST_NEO4J_USER
os.environ["NEO4J_PASSWORD"] = TEST_NEO4J_PASSWORD

# Import settings after setting environment variables
from src.utils.config import settings

# Create a mock settings object for testing
class MockSettings:
    def __init__(self):
        self.TIMESCALEDB_URI = TEST_DATABASE_URL
        self.REDIS_URL = TEST_REDIS_URL
        self.NEO4J_URI = TEST_NEO4J_URI
        self.NEO4J_USER = TEST_NEO4J_USER
        self.NEO4J_PASSWORD = TEST_NEO4J_PASSWORD
        self.TESTING = True
        self.DEBUG = True

# Create a mock settings instance
mock_settings = MockSettings()

# Function to apply test settings
def apply_test_settings():
    """Apply test settings to the application."""
    settings.TIMESCALEDB_URI = TEST_DATABASE_URL
    settings.REDIS_URL = TEST_REDIS_URL
    settings.NEO4J_URI = TEST_NEO4J_URI
    settings.NEO4J_USER = TEST_NEO4J_USER
    settings.NEO4J_PASSWORD = TEST_NEO4J_PASSWORD

# Apply test settings on import
apply_test_settings()
