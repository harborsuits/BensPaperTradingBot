#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Trading Mode - Traditional implementation of trading logic 
that separates signal generation from order execution.

Based on best practices from FreqTrade and OctoBot.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

from trading_bot.trading_modes.base_trading_mode import BaseTradingMode, Order, OrderType
from trading_bot.strategies.base_strategy import Strategy, SignalType
from trading_bot.risk.risk_manager import RiskManager

# Setup logging
logger = logging.getLogger("StandardTradingMode")

class StandardTradingMode(BaseTradingMode):
    """
    Standard trading mode that implements a traditional approach to 
    signal processing and execution.
    
    This mode:
    1. Processes signals from multiple strategies with optional weighting
    2. Applies configurable risk management rules
    3. Manages positions with customizable exit conditions
    4. Supports multiple order types and execution modes
    
    It provides a clean separation between signal generation (strategies),
    trading logic (this mode), and risk management.
    """
    
    def __init__(
        self,
        strategies: Dict[str, Strategy],
        risk_manager: RiskManager,
        symbols: List[str],
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the standard trading mode
        
        Args:
            strategies: Dictionary of strategy name -> strategy instance
            risk_manager: Risk management system
            symbols: Symbols to trade
            parameters: Trading mode parameters
        """
        # Default parameters
        default_params = {
            "strategy_weights": {},  # Equal weight by default
            "signal_threshold": 0.5,  # Minimum consensus for entry
            "max_trades_per_day": 10,
            "max_positions": 5,
            "order_type": "market",
            "enable_trailing_stop": True,
            "enable_take_profit": True,
            "partial_exits": False,
            "regime_aware": False
        }
        
        # Merge defaults with provided parameters
        if parameters:
            params = {**default_params, **parameters}
        else:
            params = default_params
            
        # Initialize base class
        super().__init__("Standard", strategies, risk_manager, symbols, params)
        
        # Trading state
        self.daily_trade_count = {}
        self.last_signals = {}
        
        logger.info(f"StandardTradingMode initialized with {len(strategies)} strategies")
    
    def process_signals(
        self, 
        strategy_signals: Dict[str, Dict[str, SignalType]], 
        market_data: Dict[str, Dict[str, Any]],
        current_time: pd.Timestamp,
        account_balance: float
    ) -> List[Order]:
        """
        Process signals from strategies and generate orders
        
        This method:
        1. Aggregates signals from multiple strategies
        2. Applies weights if configured
        3. Checks against position limits and daily trade limits
        4. Applies risk management rules
        5. Generates orders with appropriate risk parameters
        
        Args:
            strategy_signals: Dictionary of strategy name -> {symbol -> signal}
            market_data: Market data for each symbol
            current_time: Current timestamp
            account_balance: Current account balance
            
        Returns:
            List of orders to execute
        """
        # Reset today's trade count if it's a new day
        current_date = current_time.date()
        if current_date not in self.daily_trade_count:
            self.daily_trade_count = {current_date: 0}
            logger.info(f"New trading day: {current_date}")
        
        # Check if daily trade limit reached
        if self.daily_trade_count[current_date] >= self.parameters["max_trades_per_day"]:
            logger.info("Daily trade limit reached, no new entries")
            return []
        
        # Check if max positions reached
        if len(self.positions) >= self.parameters["max_positions"]:
            logger.info("Maximum positions reached, no new entries")
            return []
        
        # Aggregate signals with weighting
        aggregated_signals = self._aggregate_signals(strategy_signals)
        
        # Generate orders
        orders = []
        for symbol, signal_info in aggregated_signals.items():
            # Skip if already have position
            if symbol in self.positions:
                continue
            
            # Skip if signal strength below threshold
            if abs(signal_info["strength"]) < self.parameters["signal_threshold"]:
                continue
            
            # Determine direction
            if signal_info["strength"] > 0:
                direction = SignalType.LONG
            elif signal_info["strength"] < 0:
                direction = SignalType.SHORT
            else:
                continue  # Skip neutral signals
            
            # Get market data for symbol
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}, skipping signal")
                continue
            
            symbol_data = market_data[symbol]
            current_price = symbol_data.get("price", symbol_data.get("close", 0))
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}, skipping signal")
                continue
            
            # Calculate stop loss (via risk manager)
            stop_loss = self.risk_manager.calculate_stop_loss(
                symbol, current_price, direction.value, symbol_data
            )
            
            # Calculate position size
            atr = self.risk_manager._calculate_atr(symbol_data)
            size = self.risk_manager.calculate_position_size(
                symbol, current_price, stop_loss, symbol_data, account_balance
            )
            
            if size <= 0:
                logger.warning(f"Invalid position size for {symbol}: {size}, skipping signal")
                continue
            
            # Calculate take profit based on risk:reward
            risk_reward_ratio = self.parameters.get("risk_reward_ratio", 2.0)
            if direction == SignalType.LONG:
                risk = current_price - stop_loss
                take_profit = current_price + (risk * risk_reward_ratio)
            else:
                risk = stop_loss - current_price
                take_profit = current_price - (risk * risk_reward_ratio)
            
            # Create order
            order_type = OrderType(self.parameters["order_type"])
            order = Order(
                symbol=symbol,
                order_type=order_type,
                side=direction,
                quantity=size,
                price=current_price if order_type == OrderType.MARKET else None,
                stop_price=None,
                take_profit=take_profit if self.parameters["enable_take_profit"] else None,
                stop_loss=stop_loss,
                trailing_stop=self.parameters.get("trailing_stop", 0.1) if self.parameters["enable_trailing_stop"] else None,
                reduce_only=False,
                metadata={
                    "signal_strength": signal_info["strength"],
                    "supporting_strategies": signal_info["strategies"],
                    "entry_reason": ", ".join(signal_info["strategies"])
                }
            )
            
            orders.append(order)
            
            # Update daily trade count
            self.daily_trade_count[current_date] += 1
            
            # Store last signal for reference
            self.last_signals[symbol] = signal_info
            
            # Log order details
            logger.info(f"Generated {direction.name} order for {symbol} at {current_price}, "
                       f"size: {size}, stop: {stop_loss}, take profit: {take_profit}, "
                       f"signal strength: {signal_info['strength']:.2f}")
        
        return orders
    
    def manage_positions(
        self,
        market_data: Dict[str, Dict[str, Any]],
        current_time: pd.Timestamp
    ) -> List[Order]:
        """
        Manage existing positions and generate exit orders if needed
        
        This method:
        1. Checks positions against stop loss and take profit levels
        2. Updates trailing stops if enabled
        3. Checks time-based exit conditions
        4. Evaluates exit signals from strategies
        
        Args:
            market_data: Market data for each symbol
            current_time: Current timestamp
            
        Returns:
            List of orders to execute
        """
        exit_orders = []
        
        # Process each position
        for symbol, position in list(self.positions.items()):
            # Skip if no market data
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}, skipping position management")
                continue
            
            symbol_data = market_data[symbol]
            current_price = symbol_data.get("price", symbol_data.get("close", 0))
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}, skipping position management")
                continue
            
            # Check stop loss
            if position["stop_loss"] is not None:
                if (position["side"] == SignalType.LONG and current_price <= position["stop_loss"]) or \
                   (position["side"] == SignalType.SHORT and current_price >= position["stop_loss"]):
                    # Create exit order
                    exit_order = Order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=SignalType.SHORT if position["side"] == SignalType.LONG else SignalType.LONG,
                        quantity=position["quantity"],
                        price=current_price,
                        reduce_only=True,
                        metadata={"exit_reason": "stop_loss"}
                    )
                    exit_orders.append(exit_order)
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}, "
                              f"stop level: {position['stop_loss']}")
                    continue  # Skip further checks for this position
            
            # Check take profit
            if position["take_profit"] is not None:
                if (position["side"] == SignalType.LONG and current_price >= position["take_profit"]) or \
                   (position["side"] == SignalType.SHORT and current_price <= position["take_profit"]):
                    # Create exit order
                    exit_order = Order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=SignalType.SHORT if position["side"] == SignalType.LONG else SignalType.LONG,
                        quantity=position["quantity"],
                        price=current_price,
                        reduce_only=True,
                        metadata={"exit_reason": "take_profit"}
                    )
                    exit_orders.append(exit_order)
                    logger.info(f"Take profit triggered for {symbol} at {current_price}, "
                              f"take profit level: {position['take_profit']}")
                    continue  # Skip further checks for this position
            
            # Update trailing stop if enabled
            if position["trailing_stop"] is not None:
                # Existing implementation assumes position stored highest/lowest seen price
                # We'll need to store this or compute from market data
                
                # Simple implementation for now
                if "highest_price" not in position:
                    position["highest_price"] = current_price
                    position["lowest_price"] = current_price
                
                # Update high/low water marks
                if position["side"] == SignalType.LONG:
                    position["highest_price"] = max(position["highest_price"], current_price)
                    trail_price = position["highest_price"] * (1 - position["trailing_stop"])
                    
                    # If price drops below trail level
                    if current_price <= trail_price:
                        exit_order = Order(
                            symbol=symbol,
                            order_type=OrderType.MARKET,
                            side=SignalType.SHORT,
                            quantity=position["quantity"],
                            price=current_price,
                            reduce_only=True,
                            metadata={"exit_reason": "trailing_stop"}
                        )
                        exit_orders.append(exit_order)
                        logger.info(f"Trailing stop triggered for {symbol} at {current_price}, "
                                   f"from high: {position['highest_price']}")
                        continue
                else:  # SHORT position
                    position["lowest_price"] = min(position["lowest_price"], current_price)
                    trail_price = position["lowest_price"] * (1 + position["trailing_stop"])
                    
                    # If price rises above trail level
                    if current_price >= trail_price:
                        exit_order = Order(
                            symbol=symbol,
                            order_type=OrderType.MARKET,
                            side=SignalType.LONG,
                            quantity=position["quantity"],
                            price=current_price,
                            reduce_only=True,
                            metadata={"exit_reason": "trailing_stop"}
                        )
                        exit_orders.append(exit_order)
                        logger.info(f"Trailing stop triggered for {symbol} at {current_price}, "
                                   f"from low: {position['lowest_price']}")
                        continue
            
            # Check exit signals from strategies (reversal)
            counter_signal = self._check_exit_signals(symbol, position["side"], market_data, current_time)
            if counter_signal:
                exit_order = Order(
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=SignalType.SHORT if position["side"] == SignalType.LONG else SignalType.LONG,
                    quantity=position["quantity"],
                    price=current_price,
                    reduce_only=True,
                    metadata={"exit_reason": "strategy_signal"}
                )
                exit_orders.append(exit_order)
                logger.info(f"Exit signal triggered for {symbol} at {current_price}")
                continue
        
        return exit_orders
    
    def _aggregate_signals(self, strategy_signals: Dict[str, Dict[str, SignalType]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate signals from multiple strategies with optional weighting
        
        Args:
            strategy_signals: Dictionary of strategy name -> {symbol -> signal}
            
        Returns:
            Dictionary of aggregated signals by symbol
        """
        aggregated = {}
        
        # Get strategy weights (default to equal weighting)
        weights = self.parameters.get("strategy_weights", {})
        
        # Process signals from each strategy
        for strategy_name, signals in strategy_signals.items():
            # Get weight for this strategy (default to 1.0)
            weight = weights.get(strategy_name, 1.0)
            
            # Process signals for each symbol
            for symbol, signal in signals.items():
                if symbol not in aggregated:
                    aggregated[symbol] = {
                        "strength": 0.0,
                        "count": 0,
                        "strategies": []
                    }
                
                # Convert signal to numeric value and apply weight
                if signal == SignalType.LONG:
                    signal_value = 1.0 * weight
                elif signal == SignalType.SHORT:
                    signal_value = -1.0 * weight
                else:
                    signal_value = 0.0
                
                # Only count non-zero signals
                if signal_value != 0:
                    aggregated[symbol]["strength"] += signal_value
                    aggregated[symbol]["count"] += 1
                    aggregated[symbol]["strategies"].append(strategy_name)
        
        # Normalize signals by count
        for symbol, data in aggregated.items():
            if data["count"] > 0:
                data["strength"] = data["strength"] / data["count"]
        
        return aggregated
    
    def _check_exit_signals(
        self,
        symbol: str,
        position_side: SignalType,
        market_data: Dict[str, Dict[str, Any]],
        current_time: pd.Timestamp
    ) -> bool:
        """
        Check if strategies generate exit signals for an existing position
        
        Args:
            symbol: Symbol to check
            position_side: Current position direction
            market_data: Market data
            current_time: Current timestamp
            
        Returns:
            True if exit signal detected, False otherwise
        """
        opposite_signals = 0
        total_signals = 0
        
        # Check signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            signals = strategy.generate_signals(market_data, current_time)
            if symbol in signals:
                signal = signals[symbol]
                total_signals += 1
                
                # Check if signal is opposite to position
                if (position_side == SignalType.LONG and signal == SignalType.SHORT) or \
                   (position_side == SignalType.SHORT and signal == SignalType.LONG):
                    opposite_signals += 1
        
        # Exit if majority of strategies signal opposite direction
        if total_signals > 0 and opposite_signals / total_signals > 0.5:
            return True
            
        return False
