"""Unit tests for the graph schema module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.schema import GraphitiClient


class TestGraphitiClient:
    """Test suite for the GraphitiClient class."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Fixture to create a mock Neo4j driver."""
        # Create a mock result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"id": "test_id"}

        # Create a mock session with async run method
        mock_session = AsyncMock()
        mock_session.run.return_value = mock_result

        # Create a mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = False

        # Create a mock driver
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session_context

        return mock_driver, mock_session

    def test_initialize_schema(self, mock_neo4j_driver):
        """Test schema initialization."""
        mock_driver, mock_session = mock_neo4j_driver

        # For the sync session.run calls in initialize_schema
        mock_sync_session = MagicMock()
        mock_sync_session.run.return_value = MagicMock()
        mock_sync_session_context = MagicMock()
        mock_sync_session_context.__enter__.return_value = mock_sync_session
        mock_sync_session_context.__exit__.return_value = False
        mock_driver.session.return_value = mock_sync_session_context

        client = GraphitiClient()
        client.driver = mock_driver

        # Call the method (not async)
        client.initialize_schema()

        # Verify session.run was called with expected constraints and indexes
        # Expect one call for constraints and one for indexes
        expected_calls = 2  # For node/relationship constraints and indexes
        assert mock_driver.execute_query.await_count == expected_calls

    @pytest.mark.asyncio
    async def test_create_cryptocurrency_node(self, mock_neo4j_driver):
        """Test cryptocurrency node creation."""
        mock_driver, mock_session = mock_neo4j_driver

        client = GraphitiClient()
        client.driver = mock_driver

        # Call the method
        symbol = "BTC"
        name = "Bitcoin"
        market_cap = 1000000
        await client.create_cryptocurrency_node(
            symbol, name, market_cap=market_cap
        )

        # Verify the correct Cypher query was executed
        mock_session.run.assert_awaited_once()
        query = mock_session.run.await_args[0][0]
        assert "MERGE (c:Cryptocurrency {id: $symbol})" in query

    @pytest.mark.asyncio
    async def test_create_price_point(self, mock_neo4j_driver):
        """Test price point creation."""
        mock_driver, mock_session = mock_neo4j_driver

        client = GraphitiClient()
        client.driver = mock_driver

        price_data = {
            "open": 50000,
            "high": 51000,
            "low": 49000,
            "close": 50500,
            "volume": 1000,
            "timestamp": datetime.now(timezone.utc),
            "exchange": "binance",  # Added required field
        }

        # Call the method
        await client.create_price_point("BTC", price_data)

        # Verify the correct Cypher query was executed
        mock_session.run.assert_awaited_once()
        query = mock_session.run.await_args[0][0]
        assert "MATCH (c:Cryptocurrency {id: $symbol})" in query
        assert "CREATE (p:PricePoint" in query

    @pytest.mark.asyncio
    async def test_create_technical_indicator(self, mock_neo4j_driver):
        """Test technical indicator creation."""
        mock_driver, mock_session = mock_neo4j_driver

        client = GraphitiClient()
        client.driver = mock_driver

        values = {"rsi": 65.5, "macd": 12.3, "signal": 10.1, "histogram": 2.2}

        # Call the method
        await client.create_technical_indicator(
            symbol="BTC",
            indicator_type="RSI",
            values=values,
            timestamp=datetime.now(timezone.utc),
        )

        # Verify the correct Cypher query was executed
        mock_session.run.assert_awaited_once()
        query = mock_session.run.await_args[0][0]
        assert "MATCH (c:Cryptocurrency {id: $symbol})" in query
        assert "CREATE (i:TechnicalIndicator" in query

    @pytest.mark.asyncio
    async def test_create_correlation(self, mock_neo4j_driver):
        """Test correlation creation."""
        mock_driver, mock_session = mock_neo4j_driver

        client = GraphitiClient()
        client.driver = mock_driver

        # Call the method
        await client.create_correlation(
            symbol1="BTC",
            symbol2="ETH",
            correlation=0.85,
            window="24h",
            timestamp=datetime.now(timezone.utc),
        )

        # Verify the correct Cypher query was executed
        mock_session.run.assert_awaited_once()
        query = mock_session.run.await_args[0][0]
# The query uses a single MATCH with a comma
        # to match both nodes
        assert (
            "MATCH (c1:Cryptocurrency {id: $symbol1}), "
            "(c2:Cryptocurrency {id: $symbol2})" in query
        )
        assert "MERGE (c1)-[r:CORRELATED_WITH" in query
