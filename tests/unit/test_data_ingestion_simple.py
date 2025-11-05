"""Simplified unit tests for the data ingestion module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.ingestion import DataIngestor


class TestDataIngestorSimple:
    """Simplified test suite for the DataIngestor class."""

    @pytest.fixture
    def mock_exchange_class(self):
        """Fixture to create a mock exchange class."""
        mock_exchange_class = MagicMock()
        mock_exchange_class.__name__ = "binance"
        return mock_exchange_class

    @pytest.fixture
    def mock_exchange_instance(self):
        """Fixture to create a mock exchange instance."""
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.close = AsyncMock()
        return mock_exchange

    @pytest.fixture
    def mock_db_connection(self):
        """Fixture to create a mock database connection."""
        mock_conn = AsyncMock()

        # Mock the execute method
        async def mock_execute(*args, **kwargs):
            mock_result = AsyncMock()
            mock_result.scalar.return_value = True
            return mock_result

        mock_conn.execute = mock_execute

        # Mock the run_sync method to handle the create_all call
        async def mock_run_sync(func, *args, **kwargs):
            # If this is a call to create_all, call it with the bind parameter
            if func.__name__ == "create_all":
                if "bind" in kwargs:
                    return func(bind=kwargs["bind"])
                return func(*args, **kwargs)
            return func(*args, **kwargs)

        mock_conn.run_sync = mock_run_sync
        return mock_conn

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Fixture to create a mock SQLAlchemy engine."""
        mock_engine = MagicMock()

        # Create a mock async context manager for the connection
        class AsyncContextManager:
            def __init__(self, conn):
                self.conn = conn

            async def __aenter__(self):
                return self.conn

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        # Create a function that returns our async context manager
        def mock_begin():
            return AsyncContextManager(mock_db_connection)

        mock_engine.begin = mock_begin
        return mock_engine

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        mock_exchange_class,
        mock_exchange_instance,
        mock_engine,
        mock_db_connection,
    ):
        """Test successful initialization of DataIngestor."""
        # Configure the mock exchange class to return our mock instance
        mock_exchange_class.return_value = mock_exchange_instance

        # Mock the metadata create_all method
        mock_metadata = MagicMock()
        mock_metadata.create_all = MagicMock()

        # Patch the necessary components
        with (
            patch("ccxt.binance", mock_exchange_class),  # Mock exchange
            patch(
                "src.data.ingestion.engine", mock_engine
            ),  # Mock the SQLAlchemy engine
            patch(
                "src.data.ingestion.redis_client.connect",
                AsyncMock(return_value=True)
            ),  # Mock Redis connection
            # Mock metadata
            patch("src.data.ingestion.Base.metadata", mock_metadata),
        ):
            # Create a mock for the logger
            with patch("src.data.ingestion.logger") as mock_logger:
                # Create the ingestor and initialize it
                ingestor = DataIngestor(exchange_id="binance")
                await ingestor.initialize()

                # Verify the logger was called with the expected message
                expected_message = "Initialized binance exchange"
                mock_logger.info.assert_called_once_with(expected_message)

        # Verify the exchange was initialized correctly
        mock_exchange_instance.load_markets.assert_awaited_once()

        # Verify the database was initialized
        mock_db_connection.run_sync.assert_called_once()
        mock_metadata.create_all.assert_called_once()

        # Verify Redis was connected
        assert mock_engine.begin.called

    @pytest.mark.asyncio
    async def test_initialize_with_exception(
        self, mock_exchange_class, mock_exchange_instance, mock_engine
    ):
        """Test that exceptions during initialization are properly handled."""
        # Configure the mock exchange to raise an exception
        error_message = "Connection error"
        mock_exchange_instance.load_markets = AsyncMock(
            side_effect=Exception(error_message)
        )
        mock_exchange_class.return_value = mock_exchange_instance

        # Patch the necessary components
        with (
            patch("ccxt.binance", mock_exchange_class),
            patch("src.data.ingestion.engine", mock_engine),
            patch(
                "src.data.ingestion.redis_client.connect",
                AsyncMock(return_value=True)
            ),
            patch("src.data.ingestion.logger") as mock_logger,
        ):
            # Create the ingestor and initialize it
            ingestor = DataIngestor(exchange_id="binance")

            # Verify that the exception is raised
            with pytest.raises(Exception) as exc_info:
                await ingestor.initialize()

            # Verify the error message is correct
            assert error_message in str(exc_info.value)

            # Verify the error was logged
            mock_logger.error.assert_called_once()
            error_message = mock_logger.error.call_args[0][0]
            assert "Failed to initialize DataIngestor" in error_message
