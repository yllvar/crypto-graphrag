"""Unit tests for technical analysis indicators."""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Set test environment variables
os.environ["TIMESCALEDB_URI"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "testpassword"

# Import the module to test
try:
    from src.analysis.indicators import calculate_rsi, calculate_macd, bollinger_bands, calculate_sma, calculate_ema, calculate_volume_profile
except ImportError as e:
    print(f"Error importing indicators: {e}")
    raise

class TestTechnicalIndicators:
    """Test suite for technical analysis indicators."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing."""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
        }, index=pd.date_range('2023-01-01', periods=10, freq='D'))

    @pytest.mark.parametrize("prices,period,expected_rsi,test_name", [
        # Test case 1: Simple uptrend
        (list(range(100, 116)), 14, 100.0, "uptrend"),
        # Test case 2: Simple downtrend
        (list(range(116, 100, -1)), 14, 0.0, "downtrend"),
        # Test case 3: Neutral market
        ([100] * 20, 14, 50.0, "neutral"),
    ])
    def test_calculate_rsi(self, prices, period, expected_rsi, test_name):
        """Test RSI calculation with various price patterns."""
        print(f"\nTesting RSI - {test_name}")
        print(f"Prices: {prices}")
        
        # Arrange
        prices_series = pd.Series(prices)
        
        # Act
        result = calculate_rsi(prices_series, period)
        print(f"RSI result: {result.tolist()}")
        
        # Basic assertions
        assert isinstance(result, pd.Series), "Result should be a pandas Series"
        assert len(result) == len(prices_series), "Result length should match input length"
        
        # For the test cases where we know the expected RSI values
        if expected_rsi in [0.0, 50.0, 100.0]:
            # Get non-nan values
            non_nan_values = result[~result.isna()]
            if len(non_nan_values) > 0:
                # Check the last non-nan value
                last_value = non_nan_values.iloc[-1]
                print(f"Last non-nan RSI value: {last_value} (expected ~{expected_rsi})")
                
                # For the neutral case, allow some tolerance due to floating point precision
                tolerance = 1e-10 if expected_rsi != 50.0 else 1e-8
                assert abs(last_value - expected_rsi) < tolerance, \
                    f"Expected RSI ~{expected_rsi}, got {last_value}"
            else:
                # If all values are NaN, that's a problem
                assert False, "All RSI values are NaN"
        
    def test_calculate_macd(self, sample_ohlcv_data):
        """Test MACD calculation with sample data."""
        # Arrange
        close_prices = sample_ohlcv_data['close']
        
        # Act
        macd, signal, hist = calculate_macd(close_prices)
        
        # Assert
        assert len(macd) == len(close_prices), "MACD line length should match input length"
        assert len(signal) == len(close_prices), "Signal line length should match input length"
        assert len(hist) == len(close_prices), "Histogram length should match input length"
        
    @pytest.mark.parametrize("prices,window,num_std,expected_upper", [
        (
            [100, 101, 102, 101, 100, 101, 102, 103, 104, 103],  # prices
            2,  # window
            2.0,  # multiplier
            104.0,  # expected_upper
        )
    ])
    def test_bollinger_bands(self, prices, window, num_std, expected_upper):
        """Test Bollinger Bands calculation."""
        # Arrange
        prices_series = pd.Series(prices)
        
        # Act
        upper, middle, lower = bollinger_bands(prices_series, window, num_std)
        
        # Assert
        assert len(upper) == len(prices_series), "Upper band length should match input length"
        assert len(middle) == len(prices_series), "Middle band length should match input length"
        assert len(lower) == len(prices_series), "Lower band length should match input length"
        
        # Check that the middle band is the SMA
        expected_middle = prices_series.rolling(window=window).mean()
        pd.testing.assert_series_equal(middle, expected_middle, check_names=False)
        
        # Check that the bands are the correct distance from the middle
        std = prices_series.rolling(window=window).std()
        expected_upper_band = middle + (std * num_std)
        expected_lower_band = middle - (std * num_std)
        
        pd.testing.assert_series_equal(upper, expected_upper_band, check_names=False)
        pd.testing.assert_series_equal(lower, expected_lower_band, check_names=False)

    def test_moving_averages(self, sample_ohlcv_data):
        """Test simple and exponential moving averages."""
        # Arrange
        close_prices = sample_ohlcv_data['close']
        
        # Test SMA
        window = 5
        sma = calculate_sma(close_prices, window=window)
        
        # Assert SMA
        assert len(sma) == len(close_prices), "SMA length should match input length"
        assert sma.iloc[window-1] == close_prices.iloc[:window].mean(), "SMA should be the mean of the window"
        
        # Test EMA
        span = 5
        ema = calculate_ema(close_prices, span=span)
        
        # Assert EMA
        assert len(ema) == len(close_prices), "EMA length should match input length"
        # EMA is more complex to test directly, but we can check basic properties
        assert not ema.isna().all(), "EMA should not be all NaN"
        assert ema.iloc[-1] is not np.nan, "Last EMA value should not be NaN"

    def test_volume_profile(self, sample_ohlcv_data):
        """Test volume profile analysis."""
        # Arrange
        high = sample_ohlcv_data['high']
        low = sample_ohlcv_data['low']
        volume = sample_ohlcv_data['volume']
        
        # Act
        profile = calculate_volume_profile(high, low, volume, bins=10)
        
        # Assert
        assert 'poc_price' in profile, "Profile should contain point of control price"
        assert 'value_area_high' in profile, "Profile should contain value area high"
        assert 'value_area_low' in profile, "Profile should contain value area low"
        assert 'volume_profile' in profile, "Profile should contain volume profile dictionary"
        
        # Check types
        assert isinstance(profile['poc_price'], (int, float)), "poc_price should be a number"
        assert isinstance(profile['value_area_high'], (int, float)), "value_area_high should be a number"
        assert isinstance(profile['value_area_low'], (int, float)), "value_area_low should be a number"
        assert isinstance(profile['volume_profile'], dict), "volume_profile should be a dictionary"
        
        # Check that POC price is within the high-low range
        assert low.min() <= profile['poc_price'] <= high.max(), "POC price should be within high-low range"
        
        # Check that value area is within the high-low range
        assert low.min() <= profile['value_area_low'] <= profile['value_area_high'] <= high.max(), \
            "Value area should be within high-low range and properly ordered"
            
        # Check that volumes are non-negative
        assert all(v >= 0 for v in profile['volume_profile'].values()), "All volumes should be non-negative"
