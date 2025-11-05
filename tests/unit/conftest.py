"""Configuration and fixtures for unit tests."""
import pytest
from unittest.mock import MagicMock, AsyncMock
import sys

# Set up our mocks before importing the module under test
mock_ccxt_async_support = MagicMock()
mock_exchange_class = MagicMock()
mock_exchange_instance = AsyncMock()
mock_exchange_instance.load_markets = AsyncMock()
mock_exchange_instance.close = AsyncMock()
mock_exchange_instance.__class__.__name__ = 'binance'
mock_exchange_class.return_value = mock_exchange_instance
mock_ccxt_async_support.binance = mock_exchange_class

# Patch the module before any tests run
sys.modules['ccxt.async_support'] = mock_ccxt_async_support

# Now import the module under test
import src.data.ingestion  # noqa: E402

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    mock_ccxt_async_support.reset_mock()
    mock_exchange_class.reset_mock()
    mock_exchange_instance.reset_mock()
    mock_exchange_instance.load_markets.reset_mock()
    mock_exchange_instance.close.reset_mock()
    
    # Re-set the return value in case it was modified
    mock_exchange_class.return_value = mock_exchange_instance
    
    # Re-set the exchange class in the mock module
    mock_ccxt_async_support.binance = mock_exchange_class

@pytest.fixture
def mock_ccxt_module():
    """Fixture to provide access to the mocks used for ccxt."""
    return mock_ccxt_async_support, mock_exchange_class, mock_exchange_instance
