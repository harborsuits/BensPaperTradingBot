#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-Enhanced Trading Strategy

This strategy integrates the ML-Enhanced Alpha Discovery & RL Position Sizing system
with the IndicatorStrategy framework for more intelligent trading decisions.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.ml.ml_enhanced_trading import MLEnhancedTradingSystem, create_ml_enhanced_trading_system
from trading_bot.core.strategy import Strategy, AccountAwareMixin
from trading_bot.event_system.event import Event

logger = logging.getLogger(__name__)

class MLEnhancedStrategy(Strategy, AccountAwareMixin):
    """
    ML-Enhanced Trading Strategy that uses machine learning for signal generation
    and reinforcement learning for position sizing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ML-Enhanced Strategy
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._set_default_config()
        
        # Initialize parent class
        Strategy.__init__(self, self.config)
        AccountAwareMixin.__init__(self)
        
        # Initialize ML-Enhanced Trading System
        ml_config = self.config.get("ml_system", {})
        self.ml_system = create_ml_enhanced_trading_system(ml_config)
        
        # Track signals and positions
        self.current_signals = {}
        self.current_positions = {}
        
        # Status tracking
        self.last_update_time = None
        self.market_data_cache = {}
        
        logger.info("ML-Enhanced Strategy initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Strategy parameters
        self.config.setdefault("name", "ml_enhanced_strategy")
        self.config.setdefault("description", "ML-Enhanced Trading Strategy using Alpha Discovery and RL Position Sizing")
        self.config.setdefault("version", "1.0.0")
        self.config.setdefault("author", "Trading Bot")
        
        # Trading parameters
        self.config.setdefault("symbols", ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN"])
        self.config.setdefault("timeframe", "1d")
        self.config.setdefault("min_data_points", 100)
        
        # Signal parameters
        self.config.setdefault("min_confidence", 0.60)
        self.config.setdefault("signal_expiry_bars", 3)
        
        # Risk management
        self.config.setdefault("max_position_size", 0.20)
        self.config.setdefault("max_portfolio_allocation", 0.8)
        self.config.setdefault("trailing_stop_pct", 0.05)
        self.config.setdefault("take_profit_pct", 0.10)
        
        # ML system config (will be passed to ML system)
        self.config.setdefault("ml_system", {
            "use_ml_signals": True,
            "use_regime_detection": True,
            "use_rl_sizing": True,
            "position_constraints": {
                "max_total_allocation": 0.8,
                "max_correlated_allocation": 0.4
            }
        })
    
    def initialize(self) -> bool:
        """
        Initialize the strategy
        
        Returns:
            Success flag
        """
        logger.info("Initializing ML-Enhanced Strategy")
        
        # Initialize and load the ML system
        if not self.ml_system.initialized:
            self.ml_system.initialize()
        
        # Load models
        if not self.ml_system.models_loaded:
            self.ml_system.load_models()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("ML-Enhanced Strategy initialization complete")
        return True
    
    def _register_event_handlers(self):
        """Register event handlers with the event bus"""
        if not hasattr(self, 'event_bus') or self.event_bus is None:
            logger.warning("Event bus not available, event handlers not registered")
            return
        
        # Register for market data events
        self.event_bus.register_handler('market_data', self._handle_market_data)
        self.event_bus.register_handler('bar_data', self._handle_bar_data)
        
        # Register for account events
        self.event_bus.register_handler('account_update', self._handle_account_update)
        
        # Register for order events
        self.event_bus.register_handler('order_filled', self._handle_order_filled)
        self.event_bus.register_handler('order_cancelled', self._handle_order_cancelled)
        
        logger.info("Event handlers registered")
    
    def _handle_market_data(self, event: Event):
        """
        Handle market data events
        
        Args:
            event: Market data event
        """
        # Extract market data
        symbol = event.data.get('symbol')
        data = event.data.get('data')
        
        if symbol is None or data is None:
            return
        
        # Cache market data
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = []
        
        self.market_data_cache[symbol].append(data)
        
        # Limit cache size
        max_cache = self.config.get("min_data_points", 100)
        if len(self.market_data_cache[symbol]) > max_cache:
            self.market_data_cache[symbol] = self.market_data_cache[symbol][-max_cache:]
        
        # Update ML system if we have enough data
        if len(self.market_data_cache[symbol]) >= max_cache:
            self._update_ml_system()
    
    def _handle_bar_data(self, event: Event):
        """
        Handle bar data events
        
        Args:
            event: Bar data event
        """
        # Extract bar data
        symbol = event.data.get('symbol')
        bar = event.data.get('bar')
        
        if symbol is None or bar is None:
            return
        
        # Convert bar to market data format and update cache
        market_data = {
            'timestamp': bar.get('timestamp'),
            'open': bar.get('open'),
            'high': bar.get('high'),
            'low': bar.get('low'),
            'close': bar.get('close'),
            'volume': bar.get('volume')
        }
        
        # Add to market data cache
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = []
        
        self.market_data_cache[symbol].append(market_data)
        
        # Limit cache size
        max_cache = self.config.get("min_data_points", 100)
        if len(self.market_data_cache[symbol]) > max_cache:
            self.market_data_cache[symbol] = self.market_data_cache[symbol][-max_cache:]
        
        # Update ML system if we have enough data
        if len(self.market_data_cache[symbol]) >= max_cache:
            self._update_ml_system()
            
        # Generate trading signals
        self._generate_signals()
    
    def _handle_account_update(self, event: Event):
        """
        Handle account update events
        
        Args:
            event: Account update event
        """
        # Extract account data
        account = event.data.get('account')
        
        if account is None:
            return
        
        # Update account state
        self.account_state = account
        
        # Adjust position sizing based on account state
        self._adjust_positions()
    
    def _handle_order_filled(self, event: Event):
        """
        Handle order filled events
        
        Args:
            event: Order filled event
        """
        # Extract order data
        order = event.data.get('order')
        
        if order is None:
            return
        
        # Update positions
        symbol = order.get('symbol')
        if symbol and symbol in self.current_positions:
            # Recalculate position
            self._calculate_positions()
    
    def _handle_order_cancelled(self, event: Event):
        """
        Handle order cancelled events
        
        Args:
            event: Order cancelled event
        """
        # Extract order data
        order = event.data.get('order')
        
        if order is None:
            return
        
        # Handle cancelled order
        symbol = order.get('symbol')
        if symbol and symbol in self.current_positions:
            # Recalculate position
            self._calculate_positions()
    
    def _update_ml_system(self):
        """Update the ML system with the latest market data"""
        # Convert market data cache to the format expected by ML system
        market_data = {}
        for symbol, data_list in self.market_data_cache.items():
            if len(data_list) < self.config.get("min_data_points", 100):
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for {symbol}")
                continue
            
            # Set timestamp as index if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            market_data[symbol] = df
        
        # Update ML system
        if market_data:
            self.ml_system.update(market_data)
            self.last_update_time = datetime.now()
            logger.info(f"ML system updated with data for {len(market_data)} symbols")
    
    def _generate_signals(self):
        """Generate trading signals using the ML system"""
        # Check if ML system is initialized and models are loaded
        if not self.ml_system.initialized or not self.ml_system.models_loaded:
            logger.warning("ML system not initialized or models not loaded")
            return
        
        # Get account state for position sizing
        account_state = self.get_account_state()
        
        # Get signals for each symbol
        signals = {}
        for symbol in self.config["symbols"]:
            # Get ML signal
            signal = self.ml_system.get_trade_signal(symbol)
            
            # Check if signal is available
            if not signal.get("available", False):
                continue
            
            # Check signal confidence
            confidence = signal.get("confidence", 0)
            min_confidence = self.config["min_confidence"]
            
            if confidence < min_confidence:
                continue
            
            # Get position size
            position = self.ml_system.get_position_size(symbol, account_state)
            
            # Store signal and position
            signals[symbol] = {
                "symbol": symbol,
                "signal": signal.get("signal", "neutral"),
                "confidence": confidence,
                "position_size": position.get("position_size_pct", 0),
                "position_value": position.get("position_size_currency", 0),
                "regime": position.get("regime", 0),
                "regime_name": position.get("regime_name", "unknown"),
                "timestamp": datetime.now()
            }
        
        # Update current signals
        self.current_signals = signals
        
        # Calculate positions
        self._calculate_positions()
        
        # Execute trades
        self._execute_trades()
        
        logger.info(f"Generated {len(signals)} trading signals")
    
    def _calculate_positions(self):
        """Calculate position sizes based on signals and portfolio constraints"""
        # Get account state
        account_state = self.get_account_state()
        
        if not account_state:
            logger.warning("Account state not available")
            return
        
        # Get portfolio value
        portfolio_value = account_state.get("equity", 0)
        
        if portfolio_value <= 0:
            logger.warning("Invalid portfolio value")
            return
        
        # Calculate positions based on signals
        positions = {}
        total_allocation = 0
        
        for symbol, signal in self.current_signals.items():
            # Skip non-actionable signals
            if signal["signal"] == "neutral":
                continue
            
            # Get raw position size
            position_size = signal["position_size"]
            
            # Apply max position constraint
            position_size = min(position_size, self.config["max_position_size"])
            
            # Calculate position value
            position_value = portfolio_value * position_size
            
            # Add to positions
            positions[symbol] = {
                "symbol": symbol,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "position_size": position_size,
                "position_value": position_value,
                "regime": signal["regime"],
                "regime_name": signal["regime_name"],
                "timestamp": signal["timestamp"]
            }
            
            # Update total allocation
            total_allocation += position_size
        
        # Apply max portfolio allocation constraint
        max_allocation = self.config["max_portfolio_allocation"]
        
        if total_allocation > max_allocation and total_allocation > 0:
            # Scale down positions
            scale_factor = max_allocation / total_allocation
            
            for symbol in positions:
                positions[symbol]["position_size"] *= scale_factor
                positions[symbol]["position_value"] = portfolio_value * positions[symbol]["position_size"]
                positions[symbol]["scaled"] = True
                positions[symbol]["scale_factor"] = scale_factor
        
        # Update current positions
        self.current_positions = positions
        
        logger.info(f"Calculated {len(positions)} positions with total allocation {total_allocation:.2%}")
    
    def _execute_trades(self):
        """Execute trades based on calculated positions"""
        # Check if trade execution is enabled
        if not self.config.get("execute_trades", True):
            logger.info("Trade execution disabled")
            return
        
        # Get current portfolio
        current_portfolio = self.get_portfolio()
        
        if current_portfolio is None:
            logger.warning("Portfolio data not available")
            return
        
        # Prepare orders
        orders = []
        
        # Close positions that are no longer in our target portfolio
        for symbol, position in current_portfolio.items():
            if symbol not in self.current_positions:
                # Close position
                order = self._create_close_order(symbol, position)
                if order:
                    orders.append(order)
        
        # Open or adjust positions in our target portfolio
        for symbol, target in self.current_positions.items():
            current_size = 0
            if symbol in current_portfolio:
                current_size = current_portfolio[symbol].get("position_size", 0)
            
            # Calculate difference
            size_diff = target["position_size"] - current_size
            
            # Skip small adjustments
            min_adjustment = self.config.get("min_position_adjustment", 0.005)
            if abs(size_diff) < min_adjustment:
                continue
            
            # Create order
            order = self._create_position_order(symbol, size_diff, target)
            if order:
                orders.append(order)
        
        # Submit orders
        if hasattr(self, 'send_orders') and callable(self.send_orders):
            self.send_orders(orders)
            logger.info(f"Submitted {len(orders)} orders")
        else:
            logger.warning("Order submission not available")
    
    def _create_close_order(self, symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an order to close a position
        
        Args:
            symbol: Symbol to close
            position: Current position data
            
        Returns:
            Order dictionary
        """
        # Get position details
        size = position.get("position_size", 0)
        
        if size == 0:
            return None
        
        # Determine direction
        if size > 0:
            side = "sell"
        else:
            side = "buy"
        
        # Create order
        order = {
            "symbol": symbol,
            "quantity": abs(size),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "reason": "ml_signal_exit"
        }
        
        return order
    
    def _create_position_order(self, symbol: str, size_diff: float, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an order to open or adjust a position
        
        Args:
            symbol: Symbol to trade
            size_diff: Size difference to execute
            target: Target position data
            
        Returns:
            Order dictionary
        """
        if size_diff == 0:
            return None
        
        # Determine direction
        if size_diff > 0:
            side = "buy"
        else:
            side = "sell"
        
        # Create order
        order = {
            "symbol": symbol,
            "quantity": abs(size_diff),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "reason": "ml_signal_entry" if side == "buy" else "ml_signal_exit"
        }
        
        # Add optional stop loss and take profit
        if side == "buy":
            # Add stop loss
            stop_pct = self.config.get("trailing_stop_pct", 0.05)
            order["stop_loss"] = {
                "type": "trailing_percent",
                "percent": stop_pct * 100  # Convert to percentage
            }
            
            # Add take profit
            take_profit_pct = self.config.get("take_profit_pct", 0.10)
            order["take_profit"] = {
                "type": "percent",
                "percent": take_profit_pct * 100  # Convert to percentage
            }
        
        return order
    
    def _adjust_positions(self):
        """Adjust positions based on account state changes"""
        # Recalculate positions
        self._calculate_positions()
        
        # Execute trades
        self._execute_trades()
    
    def get_account_state(self) -> Dict[str, Any]:
        """
        Get current account state
        
        Returns:
            Account state dictionary
        """
        # If we have AccountAwareMixin, use its method
        if hasattr(self, 'get_account_info') and callable(self.get_account_info):
            return self.get_account_info()
        
        # Otherwise, return default state or stored state
        return getattr(self, 'account_state', {
            "equity": 100000.0,
            "balance": 100000.0,
            "margin_used": 0.0,
            "margin_level": 100.0
        })
    
    def get_portfolio(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current portfolio
        
        Returns:
            Portfolio dictionary
        """
        # If we have AccountAwareMixin, use its method
        if hasattr(self, 'get_positions') and callable(self.get_positions):
            return self.get_positions()
        
        # Otherwise, return stored portfolio
        return getattr(self, 'portfolio', {})
    
    def on_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        Handle tick data
        
        Args:
            tick_data: Tick data dictionary
        """
        # Add tick data to market data cache
        symbol = tick_data.get('symbol')
        
        if symbol is None:
            return
        
        # Update last price and potentially generate signals
        self._handle_market_data(Event("market_data", {
            "symbol": symbol,
            "data": tick_data
        }))
    
    def on_bar(self, bar_data: Dict[str, Any]) -> None:
        """
        Handle bar data
        
        Args:
            bar_data: Bar data dictionary
        """
        # Add bar data to market data cache and generate signals
        symbol = bar_data.get('symbol')
        
        if symbol is None:
            return
        
        # Generate signals on new bar
        self._handle_bar_data(Event("bar_data", {
            "symbol": symbol,
            "bar": bar_data
        }))
    
    def on_start(self) -> None:
        """Called when the strategy is started"""
        # Initialize the strategy
        self.initialize()
        
        logger.info("ML-Enhanced Strategy started")
    
    def on_stop(self) -> None:
        """Called when the strategy is stopped"""
        # Save state for later restoration
        if self.ml_system and self.ml_system.initialized:
            self.ml_system.save_state()
        
        logger.info("ML-Enhanced Strategy stopped")
    
    def on_error(self, error: Exception) -> None:
        """
        Handle error
        
        Args:
            error: Exception that occurred
        """
        logger.error(f"Strategy error: {error}")


# Factory function to create the strategy
def create_ml_enhanced_strategy(config: Dict[str, Any] = None) -> MLEnhancedStrategy:
    """
    Create a ML-Enhanced Strategy instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MLEnhancedStrategy instance
    """
    return MLEnhancedStrategy(config)
