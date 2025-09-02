"""Fix for position tracking in strategies to ensure proper trade execution."""

from typing import Dict, Any
import logging


def fix_moving_average_strategy(strategy):
    """
    Apply the position tracking fix to a MovingAverageCrossover strategy instance.
    
    This ensures the strategy correctly tracks positions using OrderSide enums
    instead of string comparisons.
    
    Args:
        strategy: The strategy instance to modify
    """
    from ..core.trading_bot import OrderSide
    
    # Replace the on_order_filled method with the fixed implementation
    def fixed_on_order_filled(self, order_data: Dict[str, Any]) -> None:
        """Update strategy state when an order is filled."""
        order = order_data.get("order")
        if not order:
            return
            
        symbol = order.symbol
        side = order.side
        fill_price = order_data.get("fill_price", 0)
        pnl = order_data.get("pnl", 0)
        
        # Log the order fill for debugging
        self.logger.info(f"[STRATEGY] Order filled for {symbol}: {side} at {fill_price}, PnL: {pnl}")
        
        # Track positions using proper OrderSide enum comparison
        if side == OrderSide.BUY:
            self.current_positions[symbol] = {
                "entry_price": fill_price,
                "quantity": order.quantity
            }
            self.logger.debug(f"[STRATEGY] Added position for {symbol} at {fill_price}")
            
        elif side == OrderSide.SELL and symbol in self.current_positions:
            # Position closed
            entry_price = self.current_positions[symbol].get("entry_price", 0)
            profit_loss = (fill_price - entry_price) * order.quantity
            self.logger.info(f"[STRATEGY] Closed position for {symbol}. P&L: ${profit_loss:.2f}")
            del self.current_positions[symbol]
    
    # Bind the new method to the strategy instance
    strategy.on_order_filled = fixed_on_order_filled.__get__(strategy)
    strategy.logger.info("[STRATEGY] Applied position tracking fix to MovingAverageCrossover strategy")
    return strategy


def fix_rsi_strategy(strategy):
    """
    Apply the position tracking fix to an RSIStrategy strategy instance.
    
    Args:
        strategy: The strategy instance to modify
    """
    from ..core.trading_bot import OrderSide
    
    # Replace the on_order_filled method with the fixed implementation
    def fixed_on_order_filled(self, order_data: Dict[str, Any]) -> None:
        """Update strategy state when an order is filled."""
        order = order_data.get("order")
        if not order:
            return
            
        symbol = order.symbol
        side = order.side
        fill_price = order_data.get("fill_price", 0)
        
        # Track positions using proper OrderSide enum comparison
        if side == OrderSide.BUY:
            self.current_positions[symbol] = {
                "entry_price": fill_price,
                "quantity": order.quantity
            }
            self.logger.debug(f"[STRATEGY] Added RSI position for {symbol} at {fill_price}")
            
        elif side == OrderSide.SELL and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
    
    # Bind the new method to the strategy instance
    strategy.on_order_filled = fixed_on_order_filled.__get__(strategy)
    strategy.logger.info("[STRATEGY] Applied position tracking fix to RSI strategy")
    return strategy
