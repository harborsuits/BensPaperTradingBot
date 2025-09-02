"""
Enhanced debug logging for trading bot components.
This module provides debugging utilities to trace the complete trading process.
"""

from typing import Dict, Any
import logging


def enhance_bot_logging(bot):
    """
    Enhance the bot with detailed logging for each step of the trading process.
    
    Args:
        bot: ChallengeBot instance to enhance
    """
    # Store original methods to preserve functionality
    original_on_data = bot.on_data
    original_create_order = bot._create_order_from_signal
    original_place_order = bot.place_order
    original_simulate_executions = bot._simulate_executions
    
    # Inspect the original method signature to match it exactly
    import inspect
    sig = inspect.signature(original_on_data)
    params = list(sig.parameters.keys())
    
    # Enhance on_data method with verbose logging, but preserve exact signature
    def enhanced_on_data(*args, **kwargs):
        # First arg is self
        self = args[0]
        # Second arg is market_data
        market_data = args[1] if len(args) > 1 else kwargs.get('market_data', {})
        
        self.logger.info(f"[DEBUG] on_data called with {len(market_data)} symbols")
        
        # Log sample of market data
        if market_data:
            sample_symbol = list(market_data.keys())[0]
            self.logger.info(f"[DEBUG] Sample market data keys for {sample_symbol}: {list(market_data[sample_symbol].keys())}")
            
            if 'price' in market_data[sample_symbol]:
                self.logger.info(f"[DEBUG] Sample price: {market_data[sample_symbol]['price']}")
                
            if 'history' in market_data[sample_symbol]:
                history_keys = list(market_data[sample_symbol]['history'].keys())
                self.logger.info(f"[DEBUG] History keys: {history_keys}")
                
                # Log price history length
                if 'prices' in market_data[sample_symbol]['history']:
                    prices = market_data[sample_symbol]['history']['prices']
                    self.logger.info(f"[DEBUG] History length: {len(prices)} points")
                    self.logger.info(f"[DEBUG] Recent price movement: {prices[-5:] if len(prices) >= 5 else prices}")
        
        # Call original method and capture results
        result = original_on_data(*args, **kwargs)
        self.logger.info(f"[DEBUG] on_data returned {len(result) if result else 0} orders")
        return result
    
    # Enhance _create_order_from_signal method with detailed logging
    def enhanced_create_order(*args, **kwargs):
        # First arg is self
        self = args[0]
        # Extract signal and market_data from args or kwargs
        signal = args[1] if len(args) > 1 else kwargs.get('signal')
        market_data = args[2] if len(args) > 2 else kwargs.get('market_data', {})
        
        if not signal:
            self.logger.warning("[DEBUG] No signal provided to _create_order_from_signal")
            return original_create_order(*args, **kwargs)
        
        self.logger.info(f"[DEBUG] Creating order from signal: {signal.symbol} {signal.signal_type} confidence={signal.confidence}")
        
        # Get current price for signal symbol
        current_price = market_data.get(signal.symbol, {}).get('price', 0)
        self.logger.info(f"[DEBUG] Current price for {signal.symbol}: {current_price}")
        
        # Log balance and position info
        self.logger.info(f"[DEBUG] Current balance: ${self.balance:.2f}")
        position_info = "No position" if signal.symbol not in self.positions else f"Quantity: {self.positions[signal.symbol].quantity}"
        self.logger.info(f"[DEBUG] Current position for {signal.symbol}: {position_info}")
        
        # Call original method and log result
        result = original_create_order(*args, **kwargs)
        if result:
            self.logger.info(f"[DEBUG] Created order: {result.symbol} {result.side} {result.quantity} @ {result.price}")
        else:
            self.logger.info(f"[DEBUG] Order creation failed for signal: {signal.symbol} {signal.signal_type}")
        return result
    
    # Enhance place_order with logging
    def enhanced_place_order(*args, **kwargs):
        # First arg is self
        self = args[0]
        # Extract order from args or kwargs
        order = args[1] if len(args) > 1 else kwargs.get('order')
        
        if not order:
            self.logger.warning("[DEBUG] No order provided to place_order")
            return original_place_order(*args, **kwargs)
        
        self.logger.info(f"[DEBUG] Placing order: {order.symbol} {order.side} {order.quantity} @ {order.price}")
        result = original_place_order(*args, **kwargs)
        if result:
            self.logger.info(f"[DEBUG] Order placed with ID: {result.id}")
        else:
            self.logger.info(f"[DEBUG] Order placement failed")
        return result
    
    # Enhance _simulate_executions with logging
    def enhanced_simulate_executions(*args, **kwargs):
        # First arg is self
        self = args[0]
        # Extract market_data from args or kwargs
        market_data = args[1] if len(args) > 1 else kwargs.get('market_data', {})
        
        self.logger.info(f"[DEBUG] Simulating executions with {len(self.open_orders)} open orders")
        
        # Verify market data contains price information
        for order_id, order in self.open_orders.items():
            if order.symbol in market_data:
                price = market_data[order.symbol].get('price')
                self.logger.info(f"[DEBUG] Current price for {order.symbol}: {price}")
        
        # Call original method
        original_simulate_executions(*args, **kwargs)
        
        # Log remaining open orders after simulation
        self.logger.info(f"[DEBUG] After simulation: {len(self.open_orders)} open orders remain")
    
    # Replace the methods with enhanced versions
    bot.on_data = enhanced_on_data.__get__(bot)
    bot._create_order_from_signal = enhanced_create_order.__get__(bot)
    bot.place_order = enhanced_place_order.__get__(bot)
    bot._simulate_executions = enhanced_simulate_executions.__get__(bot)
    
    bot.logger.info(f"[DEBUG] Enhanced logging applied to bot {bot.bot_id}")
    return bot
