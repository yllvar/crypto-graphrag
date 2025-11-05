"""Risk management calculations and validations."""
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pandas import Series

def calculate_position_size(account_balance: Decimal,
                        risk_percent: float,
                        entry_price: Decimal,
                        stop_loss: Decimal,
                        leverage: int = 1) -> Decimal:
    """Calculate position size based on account balance and risk percentage.
    
    Args:
        account_balance: Account balance in quote currency
        risk_percent: Percentage of account to risk (0-100)
        entry_price: Entry price of the position
        stop_loss: Stop loss price
        leverage: Leverage to apply (default: 1)
        
    Returns:
        Position size in base currency, quantized to 8 decimal places
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set the decimal precision high enough for calculations
    from decimal import getcontext
    getcontext().prec = 20
    
    logger.info("\n=== Starting position size calculation ===")
    logger.info(f"Inputs - Balance: {account_balance}, Risk %: {risk_percent}, "
               f"Entry: {entry_price}, Stop: {stop_loss}, Leverage: {leverage}")
    
    # Input validation with detailed error messages
    if risk_percent <= 0 or risk_percent > 100:
        error_msg = f"Risk percentage must be between 0 and 100, got {risk_percent}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if entry_price <= 0 or stop_loss <= 0:
        error_msg = f"Prices must be positive, got entry: {entry_price}, stop: {stop_loss}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if entry_price == stop_loss:
        error_msg = f"Entry price and stop loss cannot be the same: {entry_price}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Convert inputs to Decimal with detailed logging
        logger.info("Converting inputs to Decimal...")
        balance = Decimal(str(account_balance))
        entry = Decimal(str(entry_price))
        stop = Decimal(str(stop_loss))
        lev = Decimal(str(leverage))
        risk_pct = Decimal(str(risk_percent))
        
        logger.info(f"Converted values - Balance: {balance} ({type(balance)}), "
                   f"Entry: {entry}, Stop: {stop}, Leverage: {lev}, Risk%: {risk_pct}")
        
        # Calculate risk amount in quote currency (risk_percent of account balance)
        risk_amount = (balance * risk_pct) / Decimal('100.0')
        logger.info(f"Risk amount: {risk_amount} = {balance} * {risk_pct}%")
        
        # Calculate risk per unit (absolute value of price difference)
        risk_per_unit = abs(entry - stop)
        logger.info(f"Risk per unit: {risk_per_unit} = |{entry} - {stop}|")
        
        # Handle division by zero or extremely small numbers
        if risk_per_unit < Decimal('1e-10'):
            logger.warning(f"Risk per unit too small: {risk_per_unit}, returning 0")
            return Decimal('0')
        
        # Calculate position size in base currency (risk amount / risk per unit)
        # Then apply leverage to the position size
        position_size = (risk_amount / risk_per_unit) * lev
        
        logger.info(f"Position size before rounding: {position_size} = ({risk_amount} / {risk_per_unit}) * {lev}")
        
        logger.info(f"Position size before rounding: {position_size} = ({risk_amount} / {risk_per_unit}) * {lev}")
        
        # Ensure position size is not negative
        if position_size <= 0:
            logger.warning(f"Non-positive position size: {position_size}, returning 0")
            return Decimal('0')
        
        try:
            # First normalize the Decimal to remove any trailing zeros and exponent
            normalized = position_size.normalize()
            
            # Then quantize to 8 decimal places
            quantized = normalized.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
            
            # Ensure we have exactly 8 decimal places
            final_result = quantized.quantize(Decimal('0.00000001'))
            
            logger.info(f"Final position size: {final_result} (normalized: {normalized}, quantized: {quantized})")
            
            return final_result if final_result > 0 else Decimal('0.00000001')
            
        except Exception as format_error:
            logger.error(f"Error in final position size calculation: {str(format_error)}")
            logger.error(f"Position size value: {position_size}, type: {type(position_size)}")
            
            # Fallback: Use string formatting with scientific notation if needed
            try:
                # Convert to string with enough precision and then to Decimal
                position_str = format(float(position_size), '.10f')
                result = Decimal(position_str).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
                logger.warning(f"Used fallback conversion, result: {result}")
                return result if result > 0 else Decimal('0.00000001')
            except Exception as fallback_error:
                logger.error(f"Fallback conversion failed: {str(fallback_error)}")
                return Decimal('0.00000001')  # Return minimal position size
        
    except Exception as e:
        # Log the full error with traceback
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error in calculate_position_size: {str(e)}")
        logger.error(f"Error details:\n{error_details}")
        
        # Log all local variables for debugging
        locals_dict = locals().copy()
        for var_name, var_value in locals_dict.items():
            if var_name not in ['e', 'error_details']:  # Skip already logged variables
                logger.debug(f"{var_name}: {var_value} (type: {type(var_value)})")
        
        # Return minimal position size instead of failing completely
        return Decimal('0.00000001')

def validate_stop_loss(entry_price: Decimal,
                      stop_loss: Decimal,
                      current_price: Decimal,
                      min_stop_distance: float = 0.005) -> bool:
    """Validate if stop loss is at a reasonable distance from entry.
    
    Args:
        entry_price: Entry price of the position
        stop_loss: Proposed stop loss price
        current_price: Current market price
        min_stop_distance: Minimum distance as a percentage (0-1)
        
    Returns:
        True if stop loss is valid, False otherwise
    """
    if entry_price <= 0 or stop_loss <= 0 or current_price <= 0:
        return False
    
    # Calculate stop distance as percentage of entry price
    if entry_price > stop_loss:  # Long position
        distance = 1 - (stop_loss / entry_price)
    else:  # Short position
        distance = (stop_loss / entry_price) - 1
    
    # Check if stop is too close to entry
    if distance < min_stop_distance:
        return False
    
    # Check if stop is on the right side of the current price
    if (entry_price > stop_loss and stop_loss >= current_price) or \
       (entry_price < stop_loss and stop_loss <= current_price):
        return False
    
    return True

def calculate_risk_reward_ratio(entry_price: Decimal,
                              stop_loss: Decimal,
                              take_profit: Decimal) -> float:
    """Calculate risk/reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        
    Returns:
        Risk/reward ratio (reward / risk)
    """
    if entry_price == stop_loss:
        raise ValueError("Entry price and stop loss cannot be the same")
    
    # Calculate risk and reward
    risk = abs(float(entry_price - stop_loss))
    reward = abs(float(take_profit - entry_price))
    
    # Avoid division by zero
    if risk == 0:
        return float('inf')
    
    return reward / risk

def check_drawdown(account_balance: Decimal,
                  initial_balance: Decimal,
                  max_drawdown_percent: float) -> bool:
    """Check if account drawdown exceeds maximum allowed.
    
    Args:
        account_balance: Current account balance
        initial_balance: Initial account balance
        max_drawdown_percent: Maximum allowed drawdown percentage (0-100)
        
    Returns:
        True if drawdown is within limits, False if exceeded
    """
    if initial_balance <= 0:
        raise ValueError("Initial balance must be positive")
    
    if account_balance < 0:
        return False
    
    drawdown = 1 - (account_balance / initial_balance)
    return drawdown <= (max_drawdown_percent / 100)

def calculate_leveraged_position(account_balance: Decimal,
                               risk_percent: float,
                               entry_price: Decimal,
                               stop_loss: Decimal,
                               leverage: int = 1) -> Decimal:
    """Calculate position size with leverage.
    
    This is a wrapper around calculate_position_size that includes the leverage parameter.
    The leverage is applied within the position size calculation to ensure proper risk management.
    
    Args:
        account_balance: Account balance in quote currency
        risk_percent: Percentage of account to risk (0-100)
        entry_price: Entry price of the position
        stop_loss: Stop loss price
        leverage: Leverage to apply (default: 1)
        
    Returns:
        Position size in base currency with leverage, quantized to 8 decimal places
    """
    if leverage < 1:
        raise ValueError("Leverage must be 1 or greater")
    
    # Use the main position size calculator which handles leverage internally
    return calculate_position_size(
        account_balance=account_balance,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_loss=stop_loss,
        leverage=leverage
    )

def validate_position_size(position_size: Decimal,
                         entry_price: Decimal,
                         available_balance: Decimal,
                         leverage: int = 1) -> bool:
    """Validate if position size is within account limits.
    
    Args:
        position_size: Proposed position size in base currency
        entry_price: Entry price in quote currency
        available_balance: Available balance in quote currency
        leverage: Leverage being used (must be >= 1)
        
    Returns:
        True if position size is valid, False otherwise
    """
    try:
        # Input validation
        if position_size <= 0 or entry_price <= 0 or available_balance < 0 or leverage < 1:
            return False
        
        # Convert to Decimal for precise calculations
        size = Decimal(str(position_size))
        price = Decimal(str(entry_price))
        balance = Decimal(str(available_balance))
        lev = Decimal(str(leverage))
        
        # Calculate total position value
        position_value = size * price
        
        # Calculate required margin (position value divided by leverage)
        required_margin = position_value / lev
        
        # Add a small buffer (0.1%) to account for any floating point inaccuracies
        buffer = required_margin * Decimal('0.001')
        
        # Check if enough margin is available (with buffer)
        is_valid = (required_margin + buffer) <= balance
        
        return bool(is_valid)
        
    except Exception as e:
        # If any error occurs during calculation, fail safely by returning False
        import logging
        logging.error(f"Error validating position size: {str(e)}")
        return False

def calculate_position_risk(position_size: Decimal,
                          entry_price: Decimal,
                          stop_loss: Decimal) -> Decimal:
    """Calculate the risk amount for a position.
    
    Args:
        position_size: Position size in base currency
        entry_price: Entry price in quote currency
        stop_loss: Stop loss price in quote currency
        
    Returns:
        Risk amount in quote currency
    """
    if position_size <= 0 or entry_price <= 0 or stop_loss <= 0:
        return Decimal('0')
    
    risk_per_unit = abs(entry_price - stop_loss)
    return position_size * risk_per_unit
