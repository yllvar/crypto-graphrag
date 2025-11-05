"""Unit tests for risk management rules and calculations with enhanced logging."""
import pytest
import logging
import pandas as pd
from unittest.mock import patch, MagicMock
from decimal import Decimal, getcontext

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set decimal precision for consistent test results
getcontext().prec = 8

class TestRiskManagement:
    """Test suite for risk management functionality with detailed logging."""
    
    @pytest.fixture
    def position_data(self):
        """Generate sample position data for testing with logging."""
        data = {
            'entry_price': Decimal('100.00'),
            'stop_loss': Decimal('95.00'),
            'take_profit': Decimal('110.00'),
            'position_size': Decimal('1.0'),
            'leverage': 10
        }
        logger.info(f"Generated position data: {data}")
        return data

    @pytest.fixture
    def account_balance(self):
        """Generate sample account balance data with logging."""
        balance = {
            'total_balance': Decimal('10000.00'),
            'available_balance': Decimal('8000.00'),
            'margin_used': Decimal('2000.00'),
            'unrealized_pnl': Decimal('150.00')
        }
        logger.info(f"Generated account balance: {balance}")
        return balance

    @pytest.mark.parametrize("balance,risk_percent,entry_price,stop_loss,leverage,expected_size,test_case", [
        # Test case 1: Basic long position, no leverage
        (Decimal('10000.00'), 1.0, Decimal('100.00'), Decimal('99.00'), 1, Decimal('100.0'), "Basic long position"),
        # Test case 2: With 2x leverage
        (Decimal('10000.00'), 1.0, Decimal('100.00'), Decimal('99.00'), 2, Decimal('200.0'), "With 2x leverage"),
        # Test case 3: Different prices, smaller position
        (Decimal('5000.00'), 2.0, Decimal('50.00'), Decimal('49.00'), 1, Decimal('100.0'), "Different prices"),
    ])
    def test_calculate_position_size(self, balance, risk_percent, entry_price, stop_loss, leverage, expected_size, test_case):
        """Test position size calculation with various scenarios."""
        # Arrange
        from src.risk.management import calculate_position_size
        
        logger.info(f"\nTest Case: {test_case}")
        logger.info(f"Balance: {balance}, Risk: {risk_percent}%, Leverage: {leverage}x")
        logger.info(f"Entry: {entry_price}, Stop Loss: {stop_loss}")
        
        # Act
        position_size = calculate_position_size(
            account_balance=balance,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
            leverage=leverage
        )
        
        # Log the actual calculation
        logger.info(f"Calculated position size: {position_size}")
        
        # Assert basic properties
        assert position_size > 0, "Position size should be positive"
        
        # Verify the position size is properly quantized
        quantized_size = position_size.quantize(Decimal('0.00000001'), rounding='ROUND_DOWN')
        assert position_size == quantized_size, \
            f"Position size {position_size} is not properly quantized to 8 decimal places"
        
        # Calculate expected value using a more robust method
        try:
            risk_amount = balance * (Decimal(str(risk_percent)) / Decimal('100'))
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                expected = Decimal('0')
            else:
                expected = (risk_amount / risk_per_unit) * Decimal(leverage)
                # Round to 8 decimal places to match the function's output
                expected = expected.quantize(Decimal('0.00000001'), rounding='ROUND_DOWN')
            
            logger.info(f"Expected position size: {expected}, Got: {position_size}")
            
            # Use almost equal comparison to handle floating point imprecision
            assert abs(position_size - expected) < Decimal('0.00000001'), \
                f"Expected position size ~{expected}, got {position_size}"
                
        except Exception as e:
            logger.error(f"Error in test assertion: {str(e)}")
            raise

    @pytest.mark.parametrize("entry,stop_loss,current,expected,reason", [
        # Long positions (entry > stop_loss, current > stop_loss)
        (Decimal('100'), Decimal('95'), Decimal('98'), True, "Valid long position"),
        # Short positions (entry < stop_loss, current < stop_loss)
        (Decimal('100'), Decimal('105'), Decimal('102'), True, "Valid short position"),
        # Edge cases
        (Decimal('100'), Decimal('100'), Decimal('100'), False, "Stop equals entry"),
        (Decimal('100'), Decimal('0'), Decimal('50'), False, "Zero stop loss"),
        (Decimal('100'), Decimal('99.5'), Decimal('100'), False, "Stop too close to entry"),
    ])
    def test_validate_stop_loss(self, entry, stop_loss, current, expected, reason):
        """Test stop-loss validation with various scenarios."""
        # Arrange
        from src.risk.management import validate_stop_loss
        min_stop_distance = 0.01  # 1%
        
        logger.info(f"\nTesting stop loss validation: {reason}")
        logger.info(f"Entry: {entry}, Stop: {stop_loss}, Current: {current}")
        
        # Act
        is_valid = validate_stop_loss(
            entry_price=entry,
            stop_loss=stop_loss,
            current_price=current,
            min_stop_distance=min_stop_distance
        )
        
        # Log the validation result
        logger.info(f"Validation result: {'VALID' if is_valid else 'INVALID'}")
        
        # Assert
        assert is_valid == expected, f"Stop loss validation failed: {reason}"

    @pytest.mark.parametrize("entry,stop_loss,take_profit,expected_ratio,description", [
        # Long positions
        (Decimal('100'), Decimal('95'), Decimal('110'), 2.0, "Standard long position"),  # (110-100)/(100-95) = 2.0
        # Short positions
        (Decimal('100'), Decimal('105'), Decimal('90'), 2.0, "Standard short position"),  # (100-90)/(105-100) = 2.0
        # Edge cases
        (Decimal('100'), Decimal('99'), Decimal('103'), 3.0, "Tight spread"),
    ])
    def test_calculate_risk_reward_ratio(self, entry, stop_loss, take_profit, expected_ratio, description):
        """Test risk/reward ratio calculation."""
        # Arrange
        from src.risk.management import calculate_risk_reward_ratio
        
        logger.info(f"\nTesting R:R ratio: {description}")
        logger.info(f"Entry: {entry}, Stop: {stop_loss}, Take Profit: {take_profit}")
        
        # Act
        ratio = calculate_risk_reward_ratio(
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Log the result
        logger.info(f"Calculated ratio: {ratio}")
        
        # Assert
        assert ratio > 0, "Ratio should be positive"
        assert ratio == expected_ratio, \
            f"Expected ratio {expected_ratio}, got {ratio}"

    @pytest.mark.parametrize("current_balance,initial_balance,max_drawdown,expected", [
        (Decimal('9000.00'), Decimal('10000.00'), 10.0, True),   # 10% drawdown, at limit (returns True as it's within limit)
        (Decimal('9500.00'), Decimal('10000.00'), 10.0, True),   # 5% drawdown, under limit
        (Decimal('8900.00'), Decimal('10000.00'), 10.0, False),  # 11% drawdown, over limit (returns False)
        (Decimal('10000.00'), Decimal('10000.00'), 10.0, True),  # No drawdown
        (Decimal('11000.00'), Decimal('10000.00'), 10.0, True),  # Profit
    ])
    def test_check_drawdown(self, current_balance, initial_balance, max_drawdown, expected):
        """Test drawdown protection logic."""
        # Arrange
        from src.risk.management import check_drawdown
        
        logger.info(f"\nTesting drawdown check:")
        logger.info(f"Current: {current_balance}, Initial: {initial_balance}, Max DD: {max_drawdown}%")
        
        # Act
        should_stop = check_drawdown(
            account_balance=current_balance,
            initial_balance=initial_balance,
            max_drawdown_percent=Decimal(str(max_drawdown))
        )
        
        # Log the result
        logger.info(f"Should stop trading: {should_stop}")
        
        # Assert
        assert should_stop == expected, \
            f"Expected {expected} for {current_balance}/{initial_balance} with {max_drawdown}% max drawdown"

    @pytest.mark.parametrize("balance,risk,entry,stop,leverage,expected_size,test_case", [
        # No leverage
        (Decimal('10000.00'), 1.0, Decimal('100.00'), Decimal('99.00'), 1, Decimal('100.00000000'), "Basic no leverage"),
        # With 2x leverage
        (Decimal('10000.00'), 1.0, Decimal('100.00'), Decimal('99.00'), 2, Decimal('200.00000000'), "With 2x leverage"),
        # With 5x leverage, smaller risk
        (Decimal('10000.00'), 0.5, Decimal('100.00'), Decimal('99.00'), 5, Decimal('250.00000000'), "With 5x leverage, 0.5% risk"),
    ])
    def test_calculate_leveraged_position(self, balance, risk, entry, stop, leverage, expected_size, test_case):
        """Test position size calculation with leverage."""
        # Arrange
        from src.risk.management import calculate_leveraged_position
        
        logger.info(f"\nTest Case: {test_case}")
        logger.info(f"Balance: {balance}, Risk: {risk}%, Leverage: {leverage}x")
        logger.info(f"Entry: {entry}, Stop: {stop}")
        
        # Act
        position_size = calculate_leveraged_position(
            account_balance=balance,
            risk_percent=risk,
            entry_price=entry,
            stop_loss=stop,
            leverage=leverage
        )
        
        # Log the result
        logger.info(f"Calculated position size: {position_size}")
        
        # Assert
        assert position_size > 0, "Position size should be positive"
        
        # Calculate expected value using string formatting to avoid precision issues
        risk_amount = balance * (Decimal(str(risk)) / Decimal('100'))
        risk_per_unit = abs(entry - stop)
        expected = (risk_amount / risk_per_unit) * Decimal(leverage)
        expected_str = f"{float(expected):.8f}"
        expected = Decimal(expected_str)
        
        assert position_size == expected, \
            f"Expected size {expected}, got {position_size}"
            
        # Also verify against the provided expected size if it was provided
        if expected_size is not None:
            assert position_size == expected_size, \
                f"Expected size {expected_size}, got {position_size}"

    @pytest.mark.parametrize("position_size,entry_price,available_balance,leverage,expected", [
        # Valid positions
        (Decimal('1.0'), Decimal('100.00'), Decimal('1000.00'), 1, True),  # 100 * 1 = 100 <= 1000
        (Decimal('0.5'), Decimal('200.00'), Decimal('1000.00'), 1, True),  # 200 * 0.5 = 100 <= 1000
        # Invalid positions (not enough balance)
        (Decimal('11.0'), Decimal('100.00'), Decimal('1000.00'), 1, False),  # 100 * 11 = 1100 > 1000
        # With leverage
        (Decimal('5.0'), Decimal('200.00'), Decimal('1000.00'), 5, True),    # 200 * 5 / 5 = 200 <= 1000
        (Decimal('6.0'), Decimal('200.00'), Decimal('1000.00'), 5, True),    # 200 * 6 / 5 = 240 <= 1000 (this is actually valid)
        (Decimal('50.1'), Decimal('200.00'), Decimal('2000.00'), 5, False),  # 200 * 50.1 / 5 = 2004 > 2000
    ])
    def test_validate_position_size(self, position_size, entry_price, available_balance, leverage, expected):
        """Test position validation against account balance."""
        # Arrange
        from src.risk.management import validate_position_size
        
        logger.info(f"\nTesting position validation:")
        logger.info(f"Size: {position_size}, Entry: {entry_price}, Balance: {available_balance}, Leverage: {leverage}x")
        
        # Act
        is_valid = validate_position_size(
            position_size=position_size,
            entry_price=entry_price,
            available_balance=available_balance,
            leverage=leverage
        )
        
        # Log the result
        logger.info(f"Position valid: {is_valid}")
        
        # Assert
        assert is_valid == expected, \
            f"Expected {expected} for position size {position_size} with {leverage}x leverage"

    @pytest.mark.parametrize("position_size,entry_price,stop_loss,expected_risk", [
        # Long position
        (Decimal('1.0'), Decimal('100.00'), Decimal('95.00'), Decimal('5.00')),  # 1 * (100-95)
        # Larger position
        (Decimal('2.0'), Decimal('100.00'), Decimal('95.00'), Decimal('10.00')), # 2 * (100-95)
        # Short position
        (Decimal('1.0'), Decimal('100.00'), Decimal('105.00'), Decimal('5.00')), # 1 * (105-100)
    ])
    def test_calculate_position_risk(self, position_size, entry_price, stop_loss, expected_risk):
        """Test calculation of position risk in account currency."""
        # Arrange
        from src.risk.management import calculate_position_risk
        
        logger.info(f"\nTesting position risk calculation:")
        logger.info(f"Size: {position_size}, Entry: {entry_price}, Stop: {stop_loss}")
        
        # Act
        risk_amount = calculate_position_risk(
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Log the result
        logger.info(f"Calculated risk: {risk_amount}")
        
        # Assert
        assert risk_amount > 0, "Risk amount should be positive"
        assert risk_amount == expected_risk, \
            f"Expected risk {expected_risk}, got {risk_amount}"
