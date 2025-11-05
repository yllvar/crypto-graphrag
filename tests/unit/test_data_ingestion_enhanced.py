"""Enhanced unit tests for the data ingestion module."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
from src.data.ingestion import DataIngestor

class TestDataIngestionEnhanced:
    """Enhanced test suite for the DataIngestor class."""

    @pytest.fixture
    def mock_exchange(self):
        """Fixture to create a mock exchange with OHLCV data."""
        mock = AsyncMock()
        mock.load_markets = AsyncMock()
        mock.close = AsyncMock()
        
        # Mock OHLCV data
        mock.ohlcv = AsyncMock(return_value=[
            [int(datetime.now().timestamp()) * 1000, 50000, 51000, 49000, 50500, 1000],
            [int((datetime.now() - timedelta(minutes=5)).timestamp()) * 1000, 49500, 50500, 48500, 50000, 800]
        ])
        
        return mock

    @pytest.fixture
    def mock_db_connection(self):
        """Fixture to create a mock database connection."""
        mock_conn = AsyncMock()
        
        async def mock_execute(query, *args, **kwargs):
            if 'INSERT' in query:
                return []
            return [{'count': 1}]
            
        mock_conn.execute.side_effect = mock_execute
        return mock_conn

    @pytest.mark.asyncio
    async def test_stream_real_time_data(self, mock_exchange, mock_db_connection):
        """Test real-time data streaming and processing."""
        with patch('src.data.ingestion.create_async_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            
            ingestor = DataIngestor("binance", ["BTC/USDT"], "1m")
            ingestor.exchange = mock_exchange
            
            # Test streaming for a short duration
            task = asyncio.create_task(ingestor.start_ingestion(seconds=1))
            await asyncio.sleep(0.1)  # Allow the task to start
            await ingestor.stop_ingestion()
            
            # Verify data was processed
            assert mock_exchange.ohlcv.called
            assert mock_db_connection.execute.called

    @pytest.mark.asyncio
    async def test_handle_market_closed(self, mock_exchange, mock_db_connection):
        """Test behavior when market is closed."""
        # Mock market closed scenario (empty OHLCV data)
        mock_exchange.ohlcv = AsyncMock(return_value=[])
        
        with patch('src.data.ingestion.create_async_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            
            ingestor = DataIngestor("binance", ["BTC/USDT"], "1m")
            ingestor.exchange = mock_exchange
            
            # This should not raise an exception
            await ingestor._fetch_ohlcv("BTC/USDT", "1m")
            
            # Verify appropriate logging or handling
            assert mock_exchange.ohlcv.called

    @pytest.mark.asyncio
    async def test_network_recovery(self, mock_exchange, mock_db_connection):
        """Test reconnection after network failure."""
        # First call fails, second succeeds
        mock_exchange.ohlcv.side_effect = [
            ConnectionError("Network error"),
            [[int(datetime.now().timestamp()) * 1000, 50000, 51000, 49000, 50500, 1000]]
        ]
        
        with patch('src.data.ingestion.create_async_engine') as mock_engine, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_engine.return_value = MagicMock()
            
            ingestor = DataIngestor("binance", ["BTC/USDT"], "1m")
            ingestor.exchange = mock_exchange
            
            # Should retry on failure
            result = await ingestor._fetch_ohlcv("BTC/USDT", "1m")
            
            # Verify retry logic
            assert mock_sleep.called
            assert result is not None
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_data_validation(self, mock_exchange, mock_db_connection):
        """Test validation of incoming market data."""
        # Invalid data (missing fields)
        invalid_data = [
            [int(datetime.now().timestamp()) * 1000, None, 51000, 49000, 50500, 1000],
            [int(datetime.now().timestamp()) * 1000, 50000, 0, 49000, 50500, 1000]  # Invalid high price
        ]
        
        mock_exchange.ohlcv = AsyncMock(return_value=invalid_data)
        
        with patch('src.data.ingestion.create_async_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            
            ingestor = DataIngestor("binance", ["BTC/USDT"], "1m")
            ingestor.exchange = mock_exchange
            
            # Should handle invalid data gracefully
            result = await ingestor._fetch_ohlcv("BTC/USDT", "1m")
            
            # Verify invalid data was filtered out or handled
            assert result is not None
            assert len(result) == 0  # Or assert specific validation behavior
