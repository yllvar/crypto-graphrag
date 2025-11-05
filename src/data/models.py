from sqlalchemy import Column, String, Float, DateTime, BigInteger, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from ..utils.config import settings
import logging

# Create async engine
engine = create_async_engine(str(settings.TIMESCALEDB_URI))
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# Database session dependency
@contextmanager
async def get_db():
    db = async_session()
    try:
        yield db
    except Exception as e:
        await db.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        await db.close()

# SQLAlchemy Models
class Cryptocurrency(Base):
    __tablename__ = 'cryptocurrencies'
    
    id = Column(String, primary_key=True)  # e.g., 'BTC'
    name = Column(String, nullable=False)  # e.g., 'Bitcoin'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Exchange(Base):
    __tablename__ = 'exchanges'
    
    id = Column(String, primary_key=True)  # e.g., 'binance'
    name = Column(String, nullable=False)  # e.g., 'Binance'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Price(Base):
    __tablename__ = 'prices'
    __table_args__ = {
        'postgresql_partition_by': 'RANGE (timestamp)'
    }
    
    timestamp = Column(DateTime, primary_key=True)
    symbol = Column(String, primary_key=True)
    exchange_id = Column(String, ForeignKey('exchanges.id'), primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class TechnicalIndicator(Base):
    __tablename__ = 'technical_indicators'
    
    id = Column(String, primary_key=True)  # Format: {symbol}_{indicator}_{timestamp}
    symbol = Column(String, nullable=False)
    indicator_type = Column(String, nullable=False)  # e.g., 'RSI', 'MACD'
    value = Column(JSONB, nullable=False)  # Store indicator values as JSON
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models for API
class PriceCreate(BaseModel):
    symbol: str
    exchange_id: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

class TechnicalIndicatorCreate(BaseModel):
    symbol: str
    indicator_type: str
    value: dict
    timestamp: datetime

# Initialize database
def init_db():
    # This would be called during application startup
    # In a real app, you'd use Alembic for migrations
    async def _init_db():
        async with engine.begin() as conn:
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Create hypertable for time-series data
            await conn.execute(
                """
                SELECT create_hypertable('prices', 'timestamp', 
                                       if_not_exists => TRUE, 
                                       migrate_data => TRUE);
                """
            )
            
            # Add compression
            await conn.execute(
                """
                ALTER TABLE prices SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol, exchange_id'
                );
                """
            )
            
            # Add compression policy
            await conn.execute(
                """
                SELECT add_compression_policy('prices', INTERVAL '7 days');
                """
            )
    
    # Run the initialization
    import asyncio
    asyncio.run(_init_db())
