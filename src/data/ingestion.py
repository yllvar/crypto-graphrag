import asyncio
import socket
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from ..utils.config import settings
from ..utils.redis_client import redis_client
from .models import Price, Base, engine, async_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, insert, update
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataIngestor:
    """Handles data ingestion from cryptocurrency exchanges"""
    
    def __init__(self, exchange_id: str = None):
        self.exchange_id = exchange_id or settings.CCXT_EXCHANGE
        self.exchange = getattr(ccxt, self.exchange_id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # or 'spot'
            }
        })
        self.symbols = ['BTC/USDT', 'ETH/USDT']  # Default symbols to track
        self.running = False
        
    async def initialize(self):
        """Initialize the exchange and database"""
        logger.debug("=== Starting DataIngestor.initialize() ===")
        try:
            logger.debug("Loading exchange markets...")
            await self.exchange.load_markets()
            logger.info(f"Initialized {self.exchange_id} exchange")
            
            logger.debug("Initializing database...")
            logger.debug(f"Engine: {engine}")
            
            async with engine.begin() as conn:
                logger.debug("Database connection established")
                
                # Create tables
                logger.debug("Creating database tables...")
                await conn.run_sync(Base.metadata.create_all)
                logger.debug("Database tables created")
                
                # Check if the prices table exists before creating hypertable
                logger.debug("Checking if prices table exists...")
                table_exists = await conn.execute(
                    text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'prices'
                    )
                    """)
                )
                
                logger.debug(f"Table exists check result: {table_exists}")
                
                # Get the result and check if the table exists
                table_exists_result = table_exists.scalar()
                if table_exists_result:
                    logger.debug("Prices table exists, creating hypertable...")
                    # Create hypertable for time-series data
                    try:
                        await conn.execute(
                            text("""
                            SELECT create_hypertable(
                                'prices', 
                                'timestamp', 
                                if_not_exists => TRUE
                            )
                            """)
                        )
                        logger.info("Successfully created TimescaleDB hypertable")
                    except Exception as e:
                        logger.warning(f"Could not create hypertable (this might be expected if it already exists): {e}")
                else:
                    logger.debug("Prices table does not exist, skipping hypertable creation")
            
            logger.debug("Connecting to Redis...")
            await redis_client.connect()
            logger.debug("Successfully connected to Redis")
            
            logger.debug("=== DataIngestor.initialize() completed successfully ===")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataIngestor: {e}", exc_info=True)
            raise
    
    async def close(self):
        """Close exchange and database connections"""
        try:
            await self.exchange.close()
            await redis_client.disconnect()
            logger.info("Closed all connections")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1000) -> List[Dict]:
        """Fetch OHLCV data for a symbol"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [{
                'timestamp': datetime.utcfromtimestamp(candle[0] / 1000),
                'symbol': symbol,
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5],
                'exchange': self.exchange_id
            } for candle in ohlcv]
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []
    
    async def store_prices(self, prices: List[Dict], session: AsyncSession) -> int:
        """Store price data in the database"""
        if not prices:
            return 0
            
        try:
            # Prepare batch insert
            stmt = insert(Price).values(prices).on_conflict_do_update(
                index_elements=['timestamp', 'symbol', 'exchange_id'],
                set_={
                    'open': Price.open,
                    'high': Price.high,
                    'low': Price.low,
                    'close': Price.close,
                    'volume': Price.volume
                }
            )
            
            result = await session.execute(stmt)
            await session.commit()
            
            # Publish to Redis stream
            for price in prices[-10:]:  # Only publish last 10 prices to avoid flooding
                await redis_client.add_to_stream(
                    f"prices:{price['symbol'].lower().replace('/', '')}",
                    price
                )
            
            return len(prices)
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error storing prices: {e}")
            return 0
    
    async def fetch_and_store_historical(self, symbol: str, days: int = 30, timeframe: str = '1h'):
        """Fetch and store historical data"""
        logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        # Calculate since timestamp
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        
        # Get timeframe in milliseconds
        tf_millis = self.exchange.parse_timeframe(timeframe) * 1000
        
        all_ohlcv = []
        
        try:
            while since < end_time.timestamp() * 1000:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                # Convert to our format
                for candle in ohlcv:
                    all_ohlcv.append({
                        'timestamp': datetime.utcfromtimestamp(candle[0] / 1000),
                        'symbol': symbol,
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5],
                        'exchange_id': self.exchange_id
                    })
                
                # Move window forward
                since = ohlcv[-1][0] + tf_millis
                
                # Rate limiting
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            
            # Store in database
            async with async_session() as session:
                async with session.begin():
                    count = await self.store_prices(all_ohlcv, session)
                    logger.info(f"Stored {count} price records for {symbol}")
                
        except Exception as e:
            logger.error(f"Error in fetch_and_store_historical: {e}")
    
    async def start_realtime_feed(self, symbols: List[str] = None, interval: int = 60):
        """Start real-time data feed"""
        if symbols:
            self.symbols = symbols
            
        self.running = True
        logger.info(f"Starting real-time feed for {', '.join(self.symbols)}")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    try:
                        # Fetch latest OHLCV
                        ohlcv = await self.fetch_ohlcv(symbol, '1m', limit=1)
                        if ohlcv:
                            async with async_session() as session:
                                async with session.begin():
                                    await self.store_prices(ohlcv, session)
                            
                            # Publish to Redis
                            await redis_client.publish(
                                f"price:{symbol}", 
                                ohlcv[0]
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Real-time feed cancelled")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in real-time feed: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop_realtime_feed(self):
        """Stop the real-time feed"""
        self.running = False
        logger.info("Stopped real-time feed")
