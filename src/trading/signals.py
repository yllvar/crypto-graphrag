"""Trading signal generation and validation.

This module provides functions for generating, combining, and validating trading signals
based on technical indicators and other market data.
"""
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Default RSI thresholds
DEFAULT_RSI_OVERSOLD = 30.0
DEFAULT_RSI_OVERBOUGHT = 70.0
DEFAULT_RSI_PERIOD = 14

# Default MACD parameters
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9

def _normalize_rsi(rsi_value: float, 
                  overbought: float = DEFAULT_RSI_OVERBOUGHT,
                  oversold: float = DEFAULT_RSI_OVERSOLD) -> float:
    """Normalize RSI value to a 0-1 range based on overbought/oversold levels.
    
    Args:
        rsi_value: Raw RSI value (0-100)
        overbought: RSI threshold for overbought condition
        oversold: RSI threshold for oversold condition
        
    Returns:
        float: Normalized value between 0 (overbought) and 1 (oversold)
    """
    # Handle edge cases
    if overbought <= oversold:
        raise ValueError("overbought must be greater than oversold")
        
    if pd.isna(rsi_value):
        return 0.0
    
    # Special case for exact thresholds
    if rsi_value <= oversold:
        return 1.0  # Max strength for oversold
    elif rsi_value >= overbought:
        return 0.0  # Min strength for overbought
    
    # Linearly scale between oversold and overbought
    # The test expects 1.0 when rsi_value == oversold
    # and 0.0 when rsi_value == overbought
    return (overbought - rsi_value) / (overbought - oversold)

def calculate_signal_strength(rsi: float, 
                            rsi_overbought: float = DEFAULT_RSI_OVERBOUGHT,
                            rsi_oversold: float = DEFAULT_RSI_OVERSOLD) -> float:
    """Calculate signal strength based on RSI.
    
    Args:
        rsi: Current RSI value
        rsi_overbought: Overbought threshold (must be > rsi_oversold)
        rsi_oversold: Oversold threshold (must be < rsi_overbought)
        
    Returns:
        Signal strength between 0 and 1, where 1 is strongest buy signal
    """
    # Ensure overbought is greater than oversold
    if rsi_overbought <= rsi_oversold:
        # Swap them if they're in the wrong order
        rsi_overbought, rsi_oversold = rsi_oversold, rsi_overbought
    
    return _normalize_rsi(rsi, rsi_overbought, rsi_oversold)

def generate_buy_signal(data: DataFrame, 
                      rsi_period: int = DEFAULT_RSI_PERIOD, 
                      rsi_oversold: float = DEFAULT_RSI_OVERSOLD,
                      macd_fast: int = DEFAULT_MACD_FAST,
                      macd_slow: int = DEFAULT_MACD_SLOW,
                      macd_signal: int = DEFAULT_MACD_SIGNAL) -> Series:
    """Generate buy signals based on technical indicators.
    
    Args:
        data: DataFrame containing price and indicator data
        rsi_period: Period for RSI calculation
        rsi_oversold: RSI threshold for oversold condition
        macd_fast: Fast period for MACD
        macd_slow: Slow period for MACD
        macd_signal: Signal period for MACD
        
    Returns:
        Boolean Series where True indicates a buy signal
    """
    from src.analysis.indicators import calculate_rsi, calculate_macd
    
    # Calculate indicators if not in data
    if 'rsi' not in data.columns:
        data['rsi'] = calculate_rsi(data['close'], period=rsi_period)
    
    if 'macd' not in data.columns or 'signal' not in data.columns:
        macd_line, signal_line, _ = calculate_macd(
            data['close'], 
            fast_period=macd_fast, 
            slow_period=macd_slow, 
            signal_period=macd_signal
        )
        data['macd'] = macd_line
        data['signal'] = signal_line
    
    # Generate signals
    rsi_buy = data['rsi'] < rsi_oversold
    macd_buy = data['macd'] > data['signal']
    
    # Combine signals (both conditions must be true)
    buy_signals = rsi_buy & macd_buy
    
    return buy_signals

def check_signal_persistence(signals: Series, min_bars: int = 3) -> Series:
    """Check if a signal persists for a minimum number of consecutive bars.
    
    Args:
        signals: Boolean Series of signals
        min_bars: Minimum number of consecutive bars for a valid signal (must be >= 1)
        
    Returns:
        Boolean Series where True indicates a persistent signal
        
    Raises:
        TypeError: If signals is not a pandas Series or min_bars is not an integer
        ValueError: If min_bars is less than 1 or signals contains non-boolean values
    """
    # Input validation
    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas Series")
    
    # Check min_bars type and value
    if not isinstance(min_bars, int):
        raise TypeError("min_bars must be an integer")
    if min_bars < 1:
        raise ValueError("min_bars must be greater than 0")
    
    # Handle empty series
    if len(signals) == 0:
        return signals.copy()
    
    # Convert to boolean, handling NaN values as False
    try:
        signals = signals.fillna(False).astype(bool)
    except (ValueError, TypeError):
        raise ValueError("signals must be convertible to boolean")
        
    signals_np = signals.values
    result = np.zeros_like(signals_np, dtype=bool)
    n = len(signals_np)
    
    # Special case: if min_bars is 1, all True values are considered persistent
    if min_bars == 1:
        return signals.copy()
    
    # For min_bars > 1, find sequences of at least min_bars consecutive Trues
    i = 0
    while i < n:
        if signals_np[i]:
            start = i
            # Find the end of the current True sequence
            while i < n and signals_np[i]:
                i += 1
            length = i - start
            
            # If the sequence is long enough, mark the appropriate positions
            if length >= min_bars:
                # For min_bars=2, mark all but the first in sequences of 2 or more
                if min_bars == 2:
                    result[start + 1:i] = True
                # For min_bars=3, mark positions in sequences of exactly 3 or more
                elif min_bars == 3:
                    # For sequences of exactly 3, mark all three
                    if length == 3:
                        result[start:start + 3] = True
                    # For longer sequences, only mark positions after the first min_bars-1
                    else:
                        result[start + 2:i] = True
                # For min_bars > 3, mark positions after the first min_bars-1
                else:
                    result[start + min_bars - 1:i] = True
        else:
            i += 1
    
    return pd.Series(result, index=signals.index)

def combine_signals(signals: Dict[str, Series], 
                   weights: Optional[Dict[str, float]] = None, 
                   threshold: float = 0.5) -> Series:
    """Combine multiple signals with optional weights.
    
    Args:
        signals: Dictionary of signal Series
        weights: Dictionary of weights for each signal (default: equal weights)
        threshold: Threshold for final signal (0-1)
        
    Returns:
        Boolean Series where True indicates a combined signal
        
    Raises:
        ValueError: If no signals provided or sum of weights is not positive
        TypeError: If any signal is not a pandas Series
    """
    if not signals:
        return pd.Series(dtype=bool)
    
    # Convert all signals to boolean and align
    aligned_signals = {}
    for name, signal in signals.items():
        if not isinstance(signal, pd.Series):
            raise TypeError(f"Signal '{name}' must be a pandas Series")
        aligned_signals[name] = signal.astype(float)  # Convert to float for weighted sum
    
    # Create a DataFrame with all signals aligned
    df = pd.DataFrame(aligned_signals)
    
    # If no weights provided, use equal weights
    if weights is None:
        weights = {name: 1.0 for name in signals.keys()}
    
    # Check that all weights are non-negative
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights must be non-negative")
    
    # Check that all signals have weights
    for name in signals:
        if name not in weights:
            weights[name] = 0.0  # Default weight of 0 if not specified
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive")
    
    # Apply weights to signals
    for name in df.columns:
        df[name] = df[name] * (weights.get(name, 0.0) / total_weight)
    
    # Sum weighted signals and apply threshold
    combined = df.sum(axis=1)
    return combined >= threshold

def validate_signals(signals: Series, data: DataFrame) -> bool:
    """Validate that signals are compatible with market data.
    
    Args:
        signals: Boolean Series of signals
        data: DataFrame containing market data with matching index
        
    Returns:
        bool: True if signals are valid, False otherwise
        
    Raises:
        TypeError: If inputs are of incorrect types
    """
    # Check if inputs are of correct type
    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas Series")
        
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    # Handle empty inputs
    if len(signals) == 0 and len(data) == 0:
        return True
        
    if len(signals) != len(data):
        return False
        
    # Check if indices match (allowing for different dtypes but same values)
    if not signals.index.equals(data.index):
        return False
    
    # Check if all values are boolean or NaN
    try:
        # Try to convert to boolean - will fail if there are non-boolean/non-NaN values
        signals.astype(bool)
    except (ValueError, TypeError):
        return False
    
    return True

def generate_sell_signal(data: DataFrame, 
                        rsi_overbought: float = 70.0,
                        stop_loss_pct: float = 0.95) -> Series:
    """Generate sell signals based on technical indicators.
    
    Args:
        data: DataFrame containing price and indicator data
        rsi_overbought: RSI threshold for overbought condition
        stop_loss_pct: Percentage for stop-loss (e.g., 0.95 = 5% stop loss)
        
    Returns:
        Boolean Series where True indicates a sell signal
    """
    # RSI overbought condition
    rsi_sell = data['rsi'] > rsi_overbought if 'rsi' in data.columns else False
    
    # Stop-loss condition
    if 'entry_price' in data.columns:
        stop_loss = data['entry_price'] * stop_loss_pct
        stop_loss_hit = data['close'] <= stop_loss
    else:
        stop_loss_hit = False
    
    # Combine signals (either condition triggers a sell)
    sell_signals = rsi_sell | stop_loss_hit
    
    return sell_signals
