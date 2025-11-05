import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ..graph.schema import graphiti
from ..data.models import async_session, Price
from sqlalchemy.future import select
from sqlalchemy import func

logger = logging.getLogger(__name__)

class TechnicalAnalysisAgent:
    """Technical Analysis Agent for calculating indicators and patterns"""
    
    def __init__(self):
        self.indicators = {
            'rsi': self.calculate_rsi,
            'macd': self.calculate_macd,
            'bollinger_bands': self.calculate_bollinger_bands,
            'sma': self.calculate_sma,
            'ema': self.calculate_ema
        }
    
    async def analyze(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> Dict:
        """
        Perform technical analysis on the given symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
            limit: Number of data points to analyze
            
        Returns:
            Dict containing technical indicators and patterns
        """
        try:
            # Get price data from database
            prices = await self._get_price_data(symbol, timeframe, limit)
            if len(prices) < 14:  # Minimum data points required for most indicators
                logger.warning(f"Insufficient data points for {symbol}")
                return {}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame([{
                'timestamp': p.timestamp,
                'open': float(p.open),
                'high': float(p.high),
                'low': float(p.low),
                'close': float(p.close),
                'volume': float(p.volume)
            } for p in prices])
            
            # Calculate all indicators
            results = {}
            for indicator_name, indicator_fn in self.indicators.items():
                try:
                    indicator_result = indicator_fn(df)
                    if indicator_result is not None:
                        results[indicator_name] = indicator_result
                        
                        # Store in graph database if we have a current value
                        if isinstance(indicator_result, dict) and 'value' in indicator_result:
                            await self._store_indicator(
                                symbol=symbol,
                                indicator_type=indicator_name,
                                values=indicator_result,
                                timestamp=datetime.utcnow()
                            )
                except Exception as e:
                    logger.error(f"Error calculating {indicator_name}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            raise
    
    async def _get_price_data(self, symbol: str, timeframe: str, limit: int):
        """Fetch price data from the database"""
        async with async_session() as session:
            result = await session.execute(
                select(Price)
                .where(Price.symbol == symbol)
                .order_by(Price.timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def _store_indicator(
        self, 
        symbol: str, 
        indicator_type: str, 
        values: Dict, 
        timestamp: datetime
    ) -> None:
        """Store indicator in the graph database"""
        try:
            await graphiti.create_technical_indicator(
                symbol=symbol,
                indicator_type=indicator_type,
                values=values,
                timestamp=timestamp
            )
            logger.debug(f"Stored {indicator_type} for {symbol}")
        except Exception as e:
            logger.error(f"Error storing {indicator_type} for {symbol}: {e}")
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Relative Strength Index (RSI)"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1] if not rsi.empty else None
        
        return {
            'value': float(current_rsi) if current_rsi is not None else None,
            'values': rsi.dropna().tolist(),
            'period': period,
            'overbought': 70,
            'oversold': 30
        }
    
    def calculate_macd(
        self, 
        df: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict:
        """Calculate Moving Average Convergence Divergence (MACD)"""
        close_prices = df['close']
        
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not macd_line.empty else None,
            'signal': float(signal_line.iloc[-1]) if not signal_line.empty else None,
            'histogram': float(histogram.iloc[-1]) if not histogram.empty else None,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        num_std: int = 2
    ) -> Dict:
        """Calculate Bollinger Bands"""
        close_prices = df['close']
        
        # Calculate SMA and standard deviation
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        # Calculate bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': float(upper_band.iloc[-1]) if not upper_band.empty else None,
            'middle': float(sma.iloc[-1]) if not sma.empty else None,
            'lower': float(lower_band.iloc[-1]) if not lower_band.empty else None,
            'period': period,
            'num_std': num_std,
            'bandwidth': float((upper_band - lower_band).iloc[-1] / sma.iloc[-1] * 100) 
                        if not (sma.empty or upper_band.empty or lower_band.empty) else None
        }
    
    def calculate_sma(self, df: pd.DataFrame, period: int = 50) -> Dict:
        """Calculate Simple Moving Average"""
        sma = df['close'].rolling(window=period).mean()
        return {
            'value': float(sma.iloc[-1]) if not sma.empty else None,
            'period': period
        }
    
    def calculate_ema(self, df: pd.DataFrame, period: int = 21) -> Dict:
        """Calculate Exponential Moving Average"""
        ema = df['close'].ewm(span=period, adjust=False).mean()
        return {
            'value': float(ema.iloc[-1]) if not ema.empty else None,
            'period': period
        }
    
    def detect_support_resistance(
        self, 
        df: pd.DataFrame, 
        window: int = 20, 
        tolerance: float = 0.02
    ) -> Dict:
        """
        Detect support and resistance levels using local minima and maxima
        
        Args:
            df: DataFrame with price data
            window: Number of periods to consider for local extrema
            tolerance: Price tolerance for considering levels as equal
            
        Returns:
            Dict with support and resistance levels
        """
        high = df['high']
        low = df['low']
        
        # Find local maxima (resistance) and minima (support)
        local_max = high[(high.shift(1) < high) & (high > high.shift(-1))]
        local_min = low[(low.shift(1) > low) & (low < low.shift(-1))]
        
        # Cluster similar price levels
        def cluster_levels(levels, tolerance):
            if levels.empty:
                return []
                
            levels = sorted(levels, reverse=True)
            clusters = []
            
            for level in levels:
                if not clusters:
                    clusters.append([level])
                else:
                    found = False
                    for cluster in clusters:
                        if abs(level - sum(cluster)/len(cluster)) <= tolerance * sum(cluster)/len(cluster):
                            cluster.append(level)
                            found = True
                            break
                    if not found:
                        clusters.append([level])
            
            # Return average of each cluster
            return [sum(cluster)/len(cluster) for cluster in clusters]
        
        support_levels = cluster_levels(local_min, tolerance)
        resistance_levels = cluster_levels(local_max, tolerance)
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True),
            'window': window,
            'tolerance': tolerance
        }

# Global agent instance
technical_agent = TechnicalAnalysisAgent()
