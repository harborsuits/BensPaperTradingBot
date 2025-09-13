"""
Pre-trade risk check functionality for the Trading Bot system.

This module provides a standardized way to perform pre-execution risk checks
for all trades, ensuring that risk management rules are enforced consistently
across the system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

# Import existing risk manager
from trading_bot.risk.risk_manager import RiskManager, RiskLevel

# Import typed settings if available
try:
    from trading_bot.config.typed_settings import RiskSettings, load_config
    TYPED_SETTINGS_AVAILABLE = True
except ImportError:
    TYPED_SETTINGS_AVAILABLE = False

logger = logging.getLogger("RiskCheck")

class RiskCheckError(Exception):
    """Exception raised when a trade fails risk checks"""
    pass

def check_trade(risk_manager: RiskManager, order: Dict[str, Any], settings: Optional[RiskSettings] = None) -> Dict[str, Any]:
    """
    Perform comprehensive pre-trade risk checks for a potential order.
    
    This function evaluates an order against risk management rules to determine
    if it should be allowed to proceed. It enforces position sizing limits,
    portfolio concentration rules, drawdown protections, and more.
    
    Args:
        risk_manager: The RiskManager instance containing risk state
        order: Dictionary with order details including:
            - symbol: Trading symbol
            - side: Order side ('buy' or 'sell')
            - quantity: Number of shares/contracts
            - price: Entry price
            - dollar_amount: Total dollar value of the order
            - strategy: Strategy name (optional)
        settings: Optional RiskSettings from typed_settings system
            
    Returns:
        Dictionary with:
            - approved: Boolean indicating if the trade passes risk checks
            - reason: Explanation for rejection (if not approved)
            - warnings: List of non-blocking concerns
    """
    # Default response format
    result = {
        "approved": True,
        "reason": "",
        "warnings": []
    }
    
    # Track if we've performed a fresh risk evaluation
    risk_evaluated = False
    
    # Extract order details
    symbol = order.get("symbol")
    side = order.get("side", "")
    quantity = order.get("quantity", 0)
    price = order.get("price", 0)
    stop_price = order.get("stop_price")
    dollar_amount = order.get("dollar_amount", quantity * price)
    
    # Convert side to direction (1 for buy, -1 for sell)
    direction = 1 if side.lower() == "buy" else -1
    
    # 1. Basic validation
    if not symbol or not side or quantity <= 0 or price <= 0:
        result["approved"] = False
        result["reason"] = f"Invalid order parameters: symbol={symbol}, side={side}, quantity={quantity}, price={price}"
        return result
    
    # 2. Check if we're in a critical risk state
    if risk_manager.risk_level == RiskLevel.CRITICAL:
        result["approved"] = False
        result["reason"] = "System is in CRITICAL risk state - new trades are blocked"
        return result
    
    # 3. Check if this would exceed maximum position size
    # Try to get max_position_pct from settings or fall back to risk_manager.config
    max_position_pct = None
    if settings and TYPED_SETTINGS_AVAILABLE:
        max_position_pct = settings.max_position_pct
    else:
        max_position_pct = risk_manager.config.get("max_position_pct", 0.05)
    
    max_position_dollars = risk_manager.portfolio_value * max_position_pct
    if dollar_amount > max_position_dollars:
        result["approved"] = False
        result["reason"] = f"Position size (${dollar_amount:,.2f}) exceeds maximum allowed (${max_position_dollars:,.2f})"
        return result
    
    # 4. Check portfolio-wide risk exposure 
    # Only evaluate if we haven't yet to avoid redundant calculations
    if not risk_evaluated:
        should_reduce, reasons = risk_manager.check_risk_limits()
        risk_evaluated = True
        
        if should_reduce:
            # If we're already at risk limits, prevent new positions in same direction
            if direction == 1:  # Only block buys, allow sells to reduce exposure
                result["approved"] = False
                result["reason"] = f"Portfolio risk limits exceeded: {', '.join(reasons)}"
                return result
            else:
                # If selling, this might reduce risk, so add warning but allow
                result["warnings"].append(f"Portfolio at risk limits: {', '.join(reasons)}")
    
    # 5. Check if adequate stop-loss is provided for risk control
    if direction == 1 and not stop_price:  # Long position with no stop
        result["warnings"].append("No stop-loss specified for trade - using default risk parameters")
    
    # 6. Check symbol correlation with existing portfolio
    # This would require the risk_manager to track correlations
    # Simplified check for demonstration
    existing_positions = risk_manager.positions
    
    # Warn about overexposure to a single symbol
    if symbol in existing_positions:
        current_exposure = existing_positions[symbol]["size"] * existing_positions[symbol]["current_price"]
        new_exposure = current_exposure + dollar_amount
        
        max_symbol_exposure = risk_manager.portfolio_value * risk_manager.max_position_size_pct * 1.5
        
        if new_exposure > max_symbol_exposure:
            result["approved"] = False
            result["reason"] = f"Adding to {symbol} would exceed maximum symbol concentration (${new_exposure:,.2f} > ${max_symbol_exposure:,.2f})"
            return result
    
    # 7. Check maximum number of open positions
    # Get max_open_trades from settings or fall back to risk_manager config
    max_open_trades = None
    if settings and TYPED_SETTINGS_AVAILABLE:
        max_open_trades = settings.max_open_trades
    else:
        max_open_trades = risk_manager.config.get("max_open_trades", 10)
        
    if len(existing_positions) >= max_open_trades and symbol not in existing_positions:
        if direction == 1:  # Only block new buys, allow sells
            result["approved"] = False
            result["reason"] = f"Maximum number of positions ({max_open_trades}) already reached"
            return result
    
    # Log outcome
    if result["approved"]:
        log_msg = f"Risk check PASSED for {side} {quantity} {symbol} at {price}"
        if result["warnings"]:
            log_msg += f" with warnings: {', '.join(result['warnings'])}"
        logger.info(log_msg)
    else:
        logger.warning(f"Risk check FAILED for {side} {quantity} {symbol}: {result['reason']}")
    
    return result


def add_check_trade_to_risk_manager(risk_manager: RiskManager, settings: Optional[RiskSettings] = None) -> None:
    """
    Add the check_trade method to an existing RiskManager instance.
    
    This function adds a check_trade method to an existing RiskManager
    instance to provide pre-trade risk validation without modifying
    the original RiskManager class definition.
    
    Args:
        risk_manager: RiskManager instance to augment
        settings: Optional RiskSettings from typed_settings system
    """
    def _check_trade(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # Pass along the settings to check_trade
        return check_trade(self, order, settings)
    
    # Add the method to the instance
    risk_manager.check_trade = _check_trade.__get__(risk_manager)
    
    # Log which settings approach we're using
    if settings and TYPED_SETTINGS_AVAILABLE:
        logger.info("Added check_trade method to RiskManager instance with typed settings")
    else:
        logger.info("Added check_trade method to RiskManager instance with legacy settings")
