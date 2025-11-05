from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Any, Optional, List, Union, Annotated

import uvicorn
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Depends, status, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pydantic_extra_types.coordinate import Latitude, Longitude

# Application imports
from src.utils.redis_client import RedisClient
from src.utils.logger import logger as app_logger
from src.utils.config import settings
from src.data.ingestion import DataIngestor
from src.agents.technical import technical_agent
from src.graph.schema import graphiti

# Initialize logger
logger = app_logger

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Dict[str, Any], None]:
    """Manage application startup and shutdown events.
    
    This context manager handles the application's lifecycle events, including:
    - Startup: Initialize connections and services
    - Shutdown: Clean up resources
    """
    global redis_client
    
    # Startup logic
    try:
        logger.info("Starting application initialization...")
        
        # 1. Initialize Graphiti connection (synchronous)
        logger.info("Initializing Neo4j connection...")
        graphiti.connect()
        logger.info("Neo4j connection established")
        
        # 2. Initialize Redis client
        logger.info("Initializing Redis connection...")
        redis_client = RedisClient()
        await redis_client.connect()
        logger.info("Redis connection established")
        
        # 3. Start with default symbols
        logger.info("Starting real-time feed...")
        await start_realtime_feed()
        
        logger.info("Application startup completed successfully")
        
        # Pass any state to the application
        yield {
            "redis": redis_client,
            "started_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise
        
    finally:
        # Shutdown logic
        logger.info("Initiating application shutdown...")
        
        # Close Redis connection if it exists
        if redis_client is not None:
            logger.info("Closing Redis connection...")
            try:
                await redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
            finally:
                redis_client = None
        
        # Close Neo4j connection
        if hasattr(graphiti, 'close'):
            try:
                logger.info("Closing Neo4j connection...")
                graphiti.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {str(e)}")
        
        logger.info("Application shutdown completed")
from ..utils.redis_client import RedisClient
from ..utils.config import settings
from ..utils.logger import setup_logging, logger

# Import routers
from .v1.endpoints.llm import router as llm_router

# Initialize Redis client as None, will be initialized in startup_event
redis_client = None

# Setup logging
setup_logging()

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Crypto RAG API",
    description="API for Crypto RAG application",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware - configure according to your requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key authentication middleware
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)) -> None:
    """Verify API key for protected endpoints."""
    if not settings.API_AUTH_REQUIRED:
        return
        
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API-Key"},
        )

# Include routers
app.include_router(
    llm_router,
    prefix="/api/v1/llm",
    tags=["LLM Services"],
    dependencies=[Depends(verify_api_key)] if settings.API_AUTH_REQUIRED else []
)

# Request/Response Models
class PriceData(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

class TechnicalIndicatorRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    
    symbol: str
    indicator: str
    params: Optional[Dict[str, Any]] = {}

class TechnicalIndicatorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    
    symbol: str
    indicator: str
    value: Optional[Any] = None
    values: Optional[List[Any]] = None
    metadata: Dict[str, Any] = {}

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Crypto RAG API",
        "version": "0.1.0",
        "status": "running",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connections
        await graphiti.connect()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "graphiti": "connected",
                "redis": "connected" if await redis_client.connect() else "disconnected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

@app.post("/ingest/test")
async def test_ingestion():
    """Test data ingestion endpoint"""
    try:
        ingestor = DataIngestor()
        await ingestor.initialize()
        
        # Test with BTC/USDT and ETH/USDT
        symbols = ["BTC/USDT", "ETH/USDT"]
        
        # Fetch and store historical data
        for symbol in symbols:
            await ingestor.fetch_and_store_historical(symbol, days=7, timeframe='1h')
            
            # Run technical analysis
            analysis = await technical_agent.analyze(symbol)
            logger.info(f"Technical analysis for {symbol}: {analysis}")
        
        await ingestor.close()
        
        return {
            "status": "success",
            "message": f"Successfully ingested and analyzed data for {', '.join(symbols)}"
        }
        
    except Exception as e:
        logger.error(f"Ingestion test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion test failed: {str(e)}"
        )

@app.post("/ingest/historical")
async def ingest_historical(
    symbol: str,
    days: int = 30,
    timeframe: str = '1h',
    exchange: str = 'binance'
):
    """Ingest historical price data"""
    try:
        ingestor = DataIngestor(exchange_id=exchange)
        await ingestor.initialize()
        
        await ingestor.fetch_and_store_historical(symbol, days, timeframe)
        
        await ingestor.close()
        
        return {
            "status": "success",
            "message": f"Successfully ingested {days} days of {timeframe} data for {symbol}"
        }
    except Exception as e:
        logger.error(f"Historical ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Historical ingestion failed: {str(e)}"
        )

@app.post("/ingest/start")
async def start_realtime_feed(
    symbols: List[str] = ["BTC/USDT", "ETH/USDT"],
    interval: int = 60
):
    """Start real-time data feed"""
    try:
        global ingestor_task
        
        # Stop existing feed if running
        if 'ingestor_task' in globals() and not ingestor_task.done():
            ingestor_task.cancel()
            
        ingestor = DataIngestor()
        await ingestor.initialize()
        
        # Start in background
        ingestor_task = asyncio.create_task(
            ingestor.start_realtime_feed(symbols, interval)
        )
        
        return {
            "status": "success",
            "message": f"Started real-time feed for {', '.join(symbols)} with {interval}s interval"
        }
    except Exception as e:
        logger.error(f"Failed to start real-time feed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start real-time feed: {str(e)}"
        )

@app.post("/ingest/stop")
async def stop_realtime_feed():
    """Stop real-time data feed"""
    try:
        if 'ingestor_task' in globals() and not ingestor_task.done():
            ingestor_task.cancel()
            return {"status": "success", "message": "Stopped real-time feed"}
        else:
            return {"status": "success", "message": "No active feed to stop"}
    except Exception as e:
        logger.error(f"Failed to stop real-time feed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop real-time feed: {str(e)}"
        )

@app.post("/analyze/technical", response_model=TechnicalIndicatorResponse)
async def analyze_technical(
    request: TechnicalIndicatorRequest
) -> TechnicalIndicatorResponse:
    """Perform technical analysis"""
    try:
        # Get indicator function
        indicator_fn = getattr(technical_agent, f"calculate_{request.indicator.lower()}", None)
        if not indicator_fn:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported indicator: {request.indicator}"
            )
        
        # Get price data
        prices = await technical_agent._get_price_data(
            symbol=request.symbol,
            timeframe=request.params.get('timeframe', '1h'),
            limit=request.params.get('limit', 1000)
        )
        
        if len(prices) < 14:  # Minimum data points
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient data points for analysis"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': p.timestamp,
            'open': float(p.open),
            'high': float(p.high),
            'low': float(p.low),
            'close': float(p.close),
            'volume': float(p.volume)
        } for p in prices])
        
        # Calculate indicator
        result = indicator_fn(df, **request.params)
        
        # Format response
        response = {
            'symbol': request.symbol,
            'indicator': request.indicator,
            'metadata': {
                'period': len(prices),
                'timeframe': request.params.get('timeframe', '1h'),
                'last_updated': datetime.utcnow().isoformat()
            }
        }
        
        if isinstance(result, dict):
            response.update(result)
        else:
            response['value'] = result
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Technical analysis failed: {str(e)}"
        )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
