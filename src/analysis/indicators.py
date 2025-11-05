"""Technical analysis indicators for market data."""
from typing import Tuple, Union
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def calculate_rsi(prices: Series, period: int = 14) -> Series:
    """Calculate the Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices
        period: Number of periods to use for RSI calculation
        
    Returns:
        Series containing RSI values
    """
    if len(prices) < period + 1:
        raise ValueError(f"Not enough data points. Need at least {period + 1} points, got {len(prices)}")
    
    # Calculate price changes
    delta = prices.diff(1)
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Calculate average gains and losses using the first 'period' data points
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and handle edge cases
    with np.errstate(divide='ignore', invalid='ignore'):
        # For pure uptrend (no losses), RSI should be 100
        pure_uptrend = (avg_loss == 0) & (avg_gain > 0)
        # For pure downtrend (no gains), RSI should be 0
        pure_downtrend = (avg_gain == 0) & (avg_loss > 0)
        # For neutral (no price changes), RSI should be 50
        neutral = (avg_gain == 0) & (avg_loss == 0)
        
        # Calculate RS for normal cases
        rs = avg_gain / avg_loss
        
        # Calculate RSI for all cases
        rsi = pd.Series(index=prices.index, dtype=float)
        rsi[pure_uptrend] = 100.0
        rsi[pure_downtrend] = 0.0
        rsi[neutral] = 50.0
        
        # For mixed cases, use standard RSI formula
        mixed = ~(pure_uptrend | pure_downtrend | neutral)
        rsi[mixed] = 100 - (100 / (1 + rs[mixed]))
    
    return rsi

def calculate_macd(prices: Series, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[Series, Series, Series]:
    """Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Series of closing prices
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def bollinger_bands(prices: Series, window: int = 20, num_std: float = 2.0) -> Tuple[Series, Series, Series]:
    """Calculate Bollinger Bands.
    
    Args:
        prices: Series of closing prices
        window: Number of periods for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_sma(prices: Series, window: int) -> Series:
    """Calculate Simple Moving Average (SMA)."""
    return prices.rolling(window=window).mean()

def calculate_ema(prices: Series, span: int) -> Series:
    """Calculate Exponential Moving Average (EMA)."""
    return prices.ewm(span=span, adjust=False).mean()

def calculate_volume_profile(high: Series, low: Series, volume: Series, 
                           bins: int = 20) -> dict:
    """Calculate volume profile.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        volume: Series of volume data
        bins: Number of price bins
        
    Returns:
        Dictionary with volume profile data
    """
    price_range = np.linspace(low.min(), high.max(), bins)
    vol_profile = {}
    
    for i in range(len(price_range) - 1):
        mask = (low <= price_range[i+1]) & (high >= price_range[i])
        vol_in_range = volume[mask].sum()
        price_level = (price_range[i] + price_range[i+1]) / 2
        vol_profile[price_level] = vol_in_range
    
    # Find Point of Control (POC)
    if vol_profile:
        poc_price = max(vol_profile.items(), key=lambda x: x[1])[0]
        
        # Calculate Value Area (70% of total volume)
        sorted_vol = sorted(vol_profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(vol for _, vol in sorted_vol)
        target_volume = total_volume * 0.7
        
        value_area_prices = []
        cumulative_volume = 0
        
        for price, vol in sorted_vol:
            if cumulative_volume < target_volume:
                value_area_prices.append(price)
                cumulative_volume += vol
            else:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        return {
            'poc_price': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'volume_profile': vol_profile
        }
    
    return {}
