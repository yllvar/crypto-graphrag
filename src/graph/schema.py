from typing import Optional, List, Dict, Any
from datetime import datetime
from neo4j import GraphDatabase, basic_auth
from neo4j.time import DateTime
from ..utils.config import settings
from ..utils.logger import logger

class GraphitiClient:
    """Client for interacting with Graphiti (Neo4j) knowledge graph"""
    
    def __init__(self):
        self.driver = None
        self._is_connected = False
    
    def connect(self):
        """Initialize connection to Neo4j"""
        if not self._is_connected:
            try:
                # Create a new driver instance with direct connection
                logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI}...")
                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    connection_timeout=5,  # 5 second timeout
                    connection_acquisition_timeout=5,
                    max_connection_lifetime=1000,
                    max_connection_pool_size=10,
                    max_transaction_retry_time=5
                )
                
                # Test connection with a simple query
                with self.driver.session() as session:
                    result = session.run("RETURN 1")
                    record = result.single()
                    if record is None or record[0] != 1:
                        raise ValueError("Unexpected result from Neo4j connection test")
                
                logger.info("Successfully connected to Neo4j")
                self._is_connected = True
                
                # Initialize schema
                self.initialize_schema()
                logger.info("Graph schema initialized")
                
            except Exception as e:
                self._is_connected = False
                if self.driver:
                    self.driver.close()
                    self.driver = None
                logger.error(f"Failed to connect to Neo4j: {str(e)}")
                raise
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Closed Neo4j driver connection")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")
            finally:
                self.driver = None
                self._is_connected = False
                logger.info("Disconnected from Neo4j")
    
    def initialize_schema(self):
        """Initialize the graph schema with constraints and indexes"""
        if not self.driver:
            logger.error("Cannot initialize schema: No database connection")
            return
            
        constraints = [
            "CREATE CONSTRAINT crypto_id IF NOT EXISTS FOR (c:Cryptocurrency) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT exchange_id IF NOT EXISTS FOR (e:Exchange) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT price_id IF NOT EXISTS FOR (p:PricePoint) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT indicator_id IF NOT EXISTS FOR (i:TechnicalIndicator) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:MarketEvent) REQUIRE e.id IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX crypto_symbol IF NOT EXISTS FOR (c:Cryptocurrency) ON (c.symbol)",
            "CREATE INDEX price_timestamp IF NOT EXISTS FOR (p:PricePoint) ON (p.timestamp)",
            "CREATE INDEX indicator_type IF NOT EXISTS FOR (i:TechnicalIndicator) ON (i.type)",
            "CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:MarketEvent) ON (e.timestamp)"
        ]
        
        try:
            with self.driver.session() as session:
                # Create constraints
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Could not create constraint {constraint[:50]}...: {e}")
                
                # Create indexes
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.warning(f"Could not create index {index[:50]}...: {e}")
                
                logger.info("Graph schema initialization completed")
                
        except Exception as e:
            logger.error(f"Error initializing graph schema: {e}")
            raise
    
    async def create_cryptocurrency_node(self, symbol: str, name: str, **properties) -> str:
        """Create or update a Cryptocurrency node"""
        query = """
        MERGE (c:Cryptocurrency {id: $symbol})
        ON CREATE SET 
            c.name = $name,
            c.created_at = datetime(),
            c.updated_at = datetime()
        ON MATCH SET
            c.name = $name,
            c.updated_at = datetime()
        RETURN c.id as id
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    symbol=symbol,
                    name=name,
                    **properties
                )
                record = await result.single()
                return record["id"]
        except Exception as e:
            logger.error(f"Error creating cryptocurrency node: {e}")
            raise
    
    async def create_exchange_node(self, exchange_id: str, name: str, **properties) -> str:
        """Create or update an Exchange node"""
        query = """
        MERGE (e:Exchange {id: $exchange_id})
        ON CREATE SET 
            e.name = $name,
            e.created_at = datetime(),
            e.updated_at = datetime()
        ON MATCH SET
            e.name = $name,
            e.updated_at = datetime()
        RETURN e.id as id
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    exchange_id=exchange_id,
                    name=name,
                    **properties
                )
                record = await result.single()
                return record["id"]
        except Exception as e:
            logger.error(f"Error creating exchange node: {e}")
            raise
    
    async def create_price_point(self, symbol: str, price_data: Dict[str, Any]) -> str:
        """Create a PricePoint node and connect it to the Cryptocurrency"""
        query = """
        MATCH (c:Cryptocurrency {id: $symbol})
        CREATE (p:PricePoint {
            id: $price_id,
            symbol: $symbol,
            open: $open,
            high: $high,
            low: $low,
            close: $close,
            volume: $volume,
            timestamp: datetime($timestamp),
            exchange: $exchange,
            created_at: datetime()
        })
        CREATE (c)-[:HAS_PRICE]->(p)
        RETURN p.id as id
        """
        
        price_id = f"{symbol}_{price_data['exchange']}_{price_data['timestamp'].isoformat()}"
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    symbol=symbol,
                    price_id=price_id,
                    open=float(price_data['open']),
                    high=float(price_data['high']),
                    low=float(price_data['low']),
                    close=float(price_data['close']),
                    volume=float(price_data['volume']),
                    timestamp=price_data['timestamp'].isoformat(),
                    exchange=price_data['exchange']
                )
                record = await result.single()
                return record["id"]
        except Exception as e:
            logger.error(f"Error creating price point: {e}")
            raise
    
    async def create_technical_indicator(
        self, 
        symbol: str, 
        indicator_type: str, 
        values: Dict[str, Any],
        timestamp: datetime
    ) -> str:
        """Create a TechnicalIndicator node"""
        query = """
        MATCH (c:Cryptocurrency {id: $symbol})
        CREATE (i:TechnicalIndicator {
            id: $indicator_id,
            symbol: $symbol,
            type: $indicator_type,
            timestamp: datetime($timestamp),
            values: $values,
            created_at: datetime()
        })
        CREATE (c)-[:HAS_INDICATOR]->(i)
        RETURN i.id as id
        """
        
        indicator_id = f"{symbol}_{indicator_type}_{timestamp.isoformat()}"
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    symbol=symbol,
                    indicator_id=indicator_id,
                    indicator_type=indicator_type,
                    timestamp=timestamp.isoformat(),
                    values=values
                )
                record = await result.single()
                return record["id"]
        except Exception as e:
            logger.error(f"Error creating technical indicator: {e}")
            raise
    
    async def create_correlation(
        self, 
        symbol1: str, 
        symbol2: str, 
        correlation: float,
        window: str,
        timestamp: datetime
    ) -> None:
        """Create a correlation relationship between two cryptocurrencies"""
        query = """
        MATCH (c1:Cryptocurrency {id: $symbol1}), (c2:Cryptocurrency {id: $symbol2})
        MERGE (c1)-[r:CORRELATED_WITH {
            window: $window,
            timestamp: datetime($timestamp)
        }]->(c2)
        SET r.correlation = $correlation,
            r.updated_at = datetime()
        """
        
        try:
            async with self.driver.session() as session:
                await session.run(
                    query,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    correlation=float(correlation),
                    window=window,
                    timestamp=timestamp.isoformat()
                )
        except Exception as e:
            logger.error(f"Error creating correlation: {e}")
            raise

# Global Graphiti client instance
graphiti = GraphitiClient()

# Initialize Graphiti connection on import
async def init_graphiti():
    await graphiti.connect()

# Clean up on exit
async def close_graphiti():
    await graphiti.close()
