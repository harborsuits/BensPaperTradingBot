#!/usr/bin/env python3
"""
Enhanced Strategy Manager Implementation

This module implements the core strategy management system that handles:
- Strategy loading and initialization
- Event-driven execution
- Signal processing and order generation
- Portfolio constraints enforcement
- Performance monitoring and dynamic strategy adjustment

It integrates with the broker management system for secure order execution
across multiple brokers and asset classes.
"""

import importlib
import inspect
import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Type, DefaultDict

from trading_bot.core.constants import EventType, AssetType
from trading_bot.core.event_bus import Event, get_global_event_bus, EventBus
from trading_bot.core.strategy_base import Strategy, StrategyState, StrategyType
from trading_bot.core.enhanced_strategy_manager import Signal, SignalAction, StrategyEnsemble
from trading_bot.core.strategy_manager import StrategyPerformanceManager, StrategyStatus, StrategyAction
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.models.order import Order, OrderSide, OrderType, TimeInForce
from trading_bot.models.position import Position
from trading_bot.models.portfolio import Portfolio

logger = logging.getLogger(__name__)


class EnhancedStrategyManager:
    """
    Manages the loading, execution, and monitoring of trading strategies.
    Provides an event-driven system for strategy execution and signal processing.
    
    Key responsibilities:
    1. Loading and initializing strategies from configuration
    2. Managing strategy lifecycles (start, pause, stop)
    3. Processing market data and routing to appropriate strategies
    4. Collecting and processing trading signals
    5. Converting signals to orders with portfolio constraints
    6. Monitoring strategy performance and dynamically adjusting
    """
    
    def __init__(
        self,
        broker_manager: MultiBrokerManager = None,
        performance_manager: StrategyPerformanceManager = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the enhanced strategy manager.
        
        Args:
            broker_manager: Manager for broker interactions and order execution
            performance_manager: Manager for strategy performance evaluation
            config: Configuration dictionary for the strategy manager
        """
        # Core components
        self.broker_manager = broker_manager
        self.performance_manager = performance_manager or StrategyPerformanceManager()
        self.event_bus = get_global_event_bus()
        self.config = config or {}
        
        # Strategy collections
        self.strategies: Dict[str, Strategy] = {}  # All loaded strategies
        self.active_strategies: Dict[str, Strategy] = {}  # Currently active strategies
        self.ensembles: Dict[str, StrategyEnsemble] = {}  # Strategy ensembles
        
        # Signal and order tracking
        self.signals: List[Signal] = []  # Recent signals
        self.pending_signals: Dict[str, Signal] = {}  # Signals being processed
        self.signal_history: List[Signal] = []  # Historical signals
        
        # Portfolio constraints
        self.portfolio = Portfolio()
        self.risk_limits = self.config.get("risk_limits", {
            "max_position_per_symbol": 0.05,  # Max 5% allocation per symbol
            "max_allocation_per_strategy": 0.20,  # Max 20% allocation per strategy
            "max_allocation_per_asset_type": 0.50,  # Max 50% allocation per asset type
            "max_total_allocation": 0.80,  # Max 80% of portfolio allocated
            "max_drawdown": 0.10,  # Max 10% drawdown before intervention
            "correlation_threshold": 0.70  # Max correlation between strategies
        })
        
        # Runtime state
        self.is_running = False
        self.monitoring_thread = None
        self.last_performance_check = datetime.now()
        self.performance_check_interval = timedelta(hours=1)
        
        # Register for events
        self._register_event_handlers()
        
        logger.info("Enhanced Strategy Manager initialized")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events on the event bus."""
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATE, self.on_market_data)
        self.event_bus.subscribe(EventType.QUOTE_UPDATE, self.on_quote)
        self.event_bus.subscribe(EventType.TRADE_UPDATE, self.on_trade)
        self.event_bus.subscribe(EventType.BAR_UPDATE, self.on_bar)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self.on_order_filled)
        self.event_bus.subscribe(EventType.ORDER_CANCELED, self.on_order_canceled)
        self.event_bus.subscribe(EventType.ORDER_REJECTED, self.on_order_rejected)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self.on_position_update)
        self.event_bus.subscribe(EventType.ACCOUNT_UPDATE, self.on_account_update)
        
        logger.debug("Registered event handlers")
    
    def load_strategies(self, strategy_configs: List[Dict[str, Any]]) -> None:
        """
        Load and initialize strategies from configuration.
        
        Args:
            strategy_configs: List of strategy configuration dictionaries
        """
        if not strategy_configs:
            logger.warning("No strategy configurations provided")
            return
        
        for config in strategy_configs:
            strategy_id = config.get("strategy_id")
            strategy_type = config.get("type")
            class_path = config.get("class_path")
            parameters = config.get("parameters", {})
            enabled = config.get("enabled", True)
            
            if not strategy_id or not class_path:
                logger.error(f"Invalid strategy configuration: {config}")
                continue
            
            try:
                # Import the strategy class
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                
                # Check if it's a valid strategy class
                if not inspect.isclass(strategy_class) or not issubclass(strategy_class, Strategy):
                    logger.error(f"Invalid strategy class: {class_path}")
                    continue
                
                # Initialize the strategy
                strategy = strategy_class(
                    strategy_id=strategy_id,
                    name=config.get("name", class_name),
                    description=config.get("description", ""),
                    symbols=config.get("symbols", []),
                    asset_type=config.get("asset_type"),
                    timeframe=config.get("timeframe", "1m"),
                    parameters=parameters,
                    risk_limits=config.get("risk_limits", {}),
                    broker_id=config.get("broker_id"),
                    enabled=enabled
                )
                
                # Add to strategies collection
                self.strategies[strategy_id] = strategy
                
                # Register with performance manager
                self.performance_manager.register_strategy(strategy_id, strategy)
                
                if enabled:
                    self.active_strategies[strategy_id] = strategy
                
                logger.info(f"Loaded strategy: {strategy.name} ({strategy_id})")
            
            except Exception as e:
                logger.error(f"Error loading strategy {strategy_id}: {str(e)}")
        
        logger.info(f"Loaded {len(self.strategies)} strategies, {len(self.active_strategies)} active")
    
    def create_ensembles(self, ensemble_configs: List[Dict[str, Any]]) -> None:
        """
        Create strategy ensembles from configuration.
        
        Args:
            ensemble_configs: List of ensemble configuration dictionaries
        """
        if not ensemble_configs:
            logger.warning("No ensemble configurations provided")
            return
        
        for config in ensemble_configs:
            ensemble_id = config.get("ensemble_id")
            name = config.get("name")
            strategy_weights = config.get("strategies", {})
            
            if not ensemble_id or not name or not strategy_weights:
                logger.error(f"Invalid ensemble configuration: {config}")
                continue
            
            try:
                # Create the ensemble
                ensemble = StrategyEnsemble(
                    ensemble_id=ensemble_id,
                    name=name,
                    combination_method=config.get("combination_method", "weighted"),
                    min_consensus=config.get("min_consensus", 0.5),
                    auto_adjust_weights=config.get("auto_adjust_weights", True),
                    description=config.get("description", "")
                )
                
                # Add strategies to the ensemble
                for strategy_id, weight in strategy_weights.items():
                    if strategy_id in self.strategies:
                        ensemble.add_strategy(self.strategies[strategy_id], weight)
                    else:
                        logger.warning(f"Strategy {strategy_id} not found for ensemble {ensemble_id}")
                
                # Add to ensembles collection
                self.ensembles[ensemble_id] = ensemble
                
                logger.info(f"Created ensemble: {name} ({ensemble_id}) with {len(ensemble.strategies)} strategies")
            
            except Exception as e:
                logger.error(f"Error creating ensemble {ensemble_id}: {str(e)}")
        
        logger.info(f"Created {len(self.ensembles)} ensembles")
    
    def start_strategies(self) -> None:
        """
        Start all enabled strategies and ensembles.
        Begin monitoring and processing market data.
        """
        if self.is_running:
            logger.warning("Strategy manager is already running")
            return
        
        # Start strategies
        for strategy_id, strategy in self.active_strategies.items():
            try:
                strategy.start()
                logger.info(f"Started strategy: {strategy.name} ({strategy_id})")
            except Exception as e:
                logger.error(f"Error starting strategy {strategy_id}: {str(e)}")
        
        # Start ensembles
        for ensemble_id, ensemble in self.ensembles.items():
            try:
                ensemble.start()
                logger.info(f"Started ensemble: {ensemble.name} ({ensemble_id})")
            except Exception as e:
                logger.error(f"Error starting ensemble {ensemble_id}: {str(e)}")
        
        # Start monitoring thread
        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_strategies, daemon=True, name="StrategyMonitor"
        )
        self.monitoring_thread.start()
        
        logger.info("Strategy manager started")
    
    def stop_strategies(self) -> None:
        """
        Stop all running strategies and ensembles.
        Clean up resources and save state.
        """
        if not self.is_running:
            logger.warning("Strategy manager is not running")
            return
        
        # Stop strategies
        for strategy_id, strategy in list(self.active_strategies.items()):
            try:
                strategy.stop()
                logger.info(f"Stopped strategy: {strategy.name} ({strategy_id})")
            except Exception as e:
                logger.error(f"Error stopping strategy {strategy_id}: {str(e)}")
        
        # Stop ensembles
        for ensemble_id, ensemble in list(self.ensembles.items()):
            try:
                ensemble.stop()
                logger.info(f"Stopped ensemble: {ensemble.name} ({ensemble_id})")
            except Exception as e:
                logger.error(f"Error stopping ensemble {ensemble_id}: {str(e)}")
        
        # Stop monitoring thread
        self.is_running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Save state
        if self.performance_manager:
            self.performance_manager.save_state()
        
        logger.info("Strategy manager stopped")
    
    def process_signals(self, signals: List[Signal]) -> List[Order]:
        """
        Process trading signals and convert them to executable orders.
        Apply portfolio constraints and risk management rules.
        
        Args:
            signals: List of signals to process
            
        Returns:
            List of orders created from the signals
        """
        if not signals:
            return []
        
        if not self.broker_manager:
            logger.error("Cannot process signals: broker manager not initialized")
            return []
        
        # Track signals
        for signal in signals:
            self.signals.append(signal)
            self.pending_signals[signal.id] = signal
        
        # Apply portfolio constraints
        filtered_signals = self._apply_portfolio_constraints(signals)
        
        # Create orders from signals
        orders = []
        for signal in filtered_signals:
            try:
                order = self._create_order_from_signal(signal)
                if order:
                    orders.append(order)
                    # Mark signal as processed
                    signal.processed = True
                    signal.result = {"order_id": order.order_id, "status": "created"}
                    
                    # Publish order creation event
                    self.event_bus.publish(Event(
                        event_type=EventType.ORDER_CREATED,
                        data=order.to_dict(),
                        source="strategy_manager"
                    ))
            except Exception as e:
                logger.error(f"Error creating order from signal {signal.id}: {str(e)}")
                signal.processed = True
                signal.result = {"error": str(e), "status": "failed"}
        
        # Move processed signals from pending to history
        for signal_id in list(self.pending_signals.keys()):
            signal = self.pending_signals[signal_id]
            if signal.processed:
                self.signal_history.append(signal)
                del self.pending_signals[signal_id]
        
        # Limit history size
        max_history = self.config.get("max_signal_history", 1000)
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
        
        logger.info(f"Processed {len(signals)} signals, created {len(orders)} orders")
        return orders
    
    def _apply_portfolio_constraints(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply portfolio constraints to filter and adjust signals.
        
        Args:
            signals: List of signals to filter
            
        Returns:
            Filtered list of signals
        """
        # Get current portfolio allocation
        if not self.broker_manager:
            return signals
        
        # Group signals by strategy for allocation checks
        strategy_signals = defaultdict(list)
        for signal in signals:
            strategy_signals[signal.strategy_id].append(signal)
        
        # Check overall portfolio constraints
        filtered_signals = []
        
        # Get current allocations
        current_allocations = self._get_current_allocations()
        
        # Process signals by strategy
        for strategy_id, strategy_sigs in strategy_signals.items():
            # Check strategy allocation limit
            strategy_allocation = current_allocations.get("strategy", {}).get(strategy_id, 0.0)
            max_strategy_allocation = self.risk_limits.get("max_allocation_per_strategy", 0.20)
            
            if strategy_allocation >= max_strategy_allocation:
                logger.warning(f"Strategy {strategy_id} exceeds allocation limit ({strategy_allocation:.2%}), "
                             f"skipping {len(strategy_sigs)} signals")
                continue
            
            # Process individual signals
            for signal in strategy_sigs:
                # Check if symbol has max allocation
                symbol = signal.symbol
                symbol_allocation = current_allocations.get("symbol", {}).get(symbol, 0.0)
                max_symbol_allocation = self.risk_limits.get("max_position_per_symbol", 0.05)
                
                if symbol_allocation >= max_symbol_allocation and signal.action != SignalAction.CLOSE:
                    logger.warning(f"Symbol {symbol} exceeds allocation limit ({symbol_allocation:.2%}), "
                                 f"skipping signal {signal.id}")
                    continue
                
                # All checks passed
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _get_current_allocations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate current allocations by strategy, symbol, and asset type.
        
        Returns:
            Dictionary of allocations
        """
        # Initialize result structure
        allocations = {
            "strategy": {},
            "symbol": {},
            "asset_type": {}
        }
        
        # Get all positions
        if not self.broker_manager:
            return allocations
        
        try:
            positions = self.broker_manager.get_all_positions()
            
            # Calculate total portfolio value
            total_value = sum(position.market_value for position in positions.values() if position.market_value)
            if total_value <= 0:
                return allocations
            
            # Calculate allocations by symbol
            for symbol, position in positions.items():
                if position.market_value:
                    symbol_allocation = position.market_value / total_value
                    allocations["symbol"][symbol] = symbol_allocation
                    
                    # Track by strategy
                    strategy_id = position.metadata.get("strategy_id")
                    if strategy_id:
                        allocations["strategy"][strategy_id] = allocations["strategy"].get(strategy_id, 0.0) + symbol_allocation
                    
                    # Track by asset type
                    asset_type = position.metadata.get("asset_type")
                    if asset_type:
                        allocations["asset_type"][asset_type] = allocations["asset_type"].get(asset_type, 0.0) + symbol_allocation
            
        except Exception as e:
            logger.error(f"Error calculating allocations: {str(e)}")
        
        return allocations
    
    def _create_order_from_signal(self, signal: Signal) -> Optional[Order]:
        """
        Convert a trading signal to an executable order.
        
        Args:
            signal: Trading signal to convert
            
        Returns:
            Order object or None if conversion failed
        """
        if not self.broker_manager:
            return None
        
        # Determine order side
        side = None
        if signal.action == SignalAction.BUY:
            side = OrderSide.BUY
        elif signal.action == SignalAction.SELL:
            side = OrderSide.SELL
        elif signal.action == SignalAction.CLOSE:
            # Determine side based on current position
            try:
                position = self.broker_manager.get_position(signal.symbol)
                if position and position.quantity != 0:
                    side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                else:
                    logger.warning(f"Cannot close non-existent position for {signal.symbol}")
                    return None
            except Exception as e:
                logger.error(f"Error checking position for {signal.symbol}: {str(e)}")
                return None
        else:
            logger.warning(f"Signal action {signal.action} cannot be converted to an order")
            return None
        
        # Determine order quantity
        quantity = None
        if signal.quantity is not None:
            quantity = signal.quantity
        elif signal.allocation is not None:
            # Calculate quantity based on allocation percentage
            try:
                # Get account balance
                accounts = self.broker_manager.get_all_accounts()
                if accounts:
                    # Use the first account's buying power for simplicity
                    # In practice, you might want to use the account for the specific broker
                    first_account = next(iter(accounts.values()))
                    buying_power = first_account.buying_power
                    
                    # Get current price
                    price = self._get_current_price(signal.symbol)
                    if price and price > 0:
                        # Calculate quantity based on allocation
                        quantity = (buying_power * signal.allocation) / price
                        # Round down to appropriate precision
                        quantity = round(quantity, 2)  # Adjust precision as needed
            except Exception as e:
                logger.error(f"Error calculating quantity for {signal.symbol}: {str(e)}")
                return None
        else:
            logger.warning(f"Signal for {signal.symbol} has no quantity or allocation")
            return None
        
        if not quantity or quantity <= 0:
            logger.warning(f"Invalid quantity {quantity} for {signal.symbol}")
            return None
        
        # Determine order type
        order_type = OrderType.MARKET
        if signal.target_price is not None:
            order_type = OrderType.LIMIT
        
        # Create the order
        strategy = self.strategies.get(signal.strategy_id)
        broker_id = strategy.broker_id if strategy else None
        
        # Get broker from asset routing if not specified
        if not broker_id and self.broker_manager and signal.symbol:
            try:
                # Determine asset type from symbol or metadata
                asset_type = None
                if strategy and strategy.asset_type:
                    asset_type = strategy.asset_type
                elif signal.metadata and signal.metadata.get("asset_type"):
                    asset_type = signal.metadata.get("asset_type")
                
                # Get broker from routing
                if asset_type:
                    broker_id = self.broker_manager.get_broker_for_asset_type(asset_type)
            except Exception as e:
                logger.error(f"Error determining broker for {signal.symbol}: {str(e)}")
        
        # Create the order object
        order = Order(
            symbol=signal.symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            price=signal.target_price,
            stop_price=signal.stop_loss,
            time_in_force=TimeInForce.DAY,
            status="created",
            broker_id=broker_id,
            strategy_id=signal.strategy_id,
            metadata={
                "signal_id": signal.id,
                "confidence": signal.confidence,
                "strength": signal.strength.value,
                **signal.metadata
            }
        )
        
        return order
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if not available
        """
        if not self.broker_manager:
            return None
        
        try:
            # Try to get quote from broker manager
            quote = self.broker_manager.get_quote(symbol)
            if quote and quote.last:
                return quote.last
            
            # If no quote, try to get last price from positions
            position = self.broker_manager.get_position(symbol)
            if position and position.current_price:
                return position.current_price
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
        
        return None
    
    # Event handlers
    def on_market_data(self, event: Event) -> None:
        """
        Handle market data updates and route to appropriate strategies.
        
        Args:
            event: Market data event
        """
        if not self.is_running:
            return
        
        data = event.data
        if not data or not isinstance(data, dict):
            return
        
        symbol = data.get("symbol")
        if not symbol:
            return
        
        # Process signals from strategies
        signals_to_process = []
        
        # Route to individual strategies
        for strategy_id, strategy in self.active_strategies.items():
            if symbol in strategy.symbols and strategy.is_running():
                try:
                    # Let strategy process the data
                    signal = strategy.generate_signal(data)
                    if signal:
                        signals_to_process.append(signal)
                except Exception as e:
                    logger.error(f"Error in strategy {strategy_id} processing data for {symbol}: {str(e)}")
        
        # Process signals from ensembles
        for ensemble_id, ensemble in self.ensembles.items():
            if symbol in ensemble.symbols and ensemble.is_running():
                # Get signals from all strategies in the ensemble for this symbol
                ensemble_signals = []
                for strategy_id, (strategy, _) in ensemble.strategies.items():
                    if symbol in strategy.symbols and strategy.is_running():
                        try:
                            signal = strategy.generate_signal(data)
                            if signal:
                                ensemble_signals.append(signal)
                        except Exception as e:
                            logger.error(f"Error in ensemble strategy {strategy_id} for {symbol}: {str(e)}")
                
                # Generate combined signal if any component signals were generated
                if ensemble_signals:
                    combined_signal = ensemble.generate_combined_signal(symbol, ensemble_signals)
                    if combined_signal:
                        signals_to_process.append(combined_signal)
        
        # Process all generated signals
        if signals_to_process:
            self.process_signals(signals_to_process)
    
    def on_quote(self, event: Event) -> None:
        """Handle quote updates."""
        # Similar to on_market_data but for quote-specific events
        self.on_market_data(event)  # Reuse the same logic for now
    
    def on_trade(self, event: Event) -> None:
        """Handle trade updates."""
        # Similar to on_market_data but for trade-specific events
        self.on_market_data(event)  # Reuse the same logic for now
    
    def on_bar(self, event: Event) -> None:
        """Handle bar/candle updates."""
        # Similar to on_market_data but for bar-specific events
        self.on_market_data(event)  # Reuse the same logic for now
    
    def on_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        if not self.is_running:
            return
        
        order_data = event.data
        if not order_data or not isinstance(order_data, dict):
            return
        
        # Update signal result if this order was created from a signal
        order_id = order_data.get("order_id")
        strategy_id = order_data.get("strategy_id")
        metadata = order_data.get("metadata", {})
        signal_id = metadata.get("signal_id")
        
        if signal_id and signal_id in self.pending_signals:
            signal = self.pending_signals[signal_id]
            signal.processed = True
            signal.result = {
                "order_id": order_id,
                "status": "filled",
                "fill_price": order_data.get("fill_price"),
                "fill_quantity": order_data.get("fill_quantity"),
                "fill_time": order_data.get("fill_time")
            }
            
            # Move to history
            self.signal_history.append(signal)
            del self.pending_signals[signal_id]
        
        # Route to appropriate strategy
        if strategy_id in self.strategies:
            try:
                strategy = self.strategies[strategy_id]
                # Strategy's own event handling will take care of this
                # but we can add additional processing here if needed
            except Exception as e:
                logger.error(f"Error handling order filled event in strategy {strategy_id}: {str(e)}")
    
    def on_order_canceled(self, event: Event) -> None:
        """Handle order canceled events."""
        # Similar to on_order_filled but for canceled orders
        if not self.is_running:
            return
        
        order_data = event.data
        if not order_data or not isinstance(order_data, dict):
            return
        
        # Update signal result if this order was created from a signal
        metadata = order_data.get("metadata", {})
        signal_id = metadata.get("signal_id")
        
        if signal_id and signal_id in self.pending_signals:
            signal = self.pending_signals[signal_id]
            signal.processed = True
            signal.result = {
                "order_id": order_data.get("order_id"),
                "status": "canceled",
                "reason": order_data.get("cancel_reason")
            }
            
            # Move to history
            self.signal_history.append(signal)
            del self.pending_signals[signal_id]
    
    def on_order_rejected(self, event: Event) -> None:
        """Handle order rejected events."""
        # Similar to on_order_canceled but for rejected orders
        if not self.is_running:
            return
        
        order_data = event.data
        if not order_data or not isinstance(order_data, dict):
            return
        
        # Update signal result if this order was created from a signal
        metadata = order_data.get("metadata", {})
        signal_id = metadata.get("signal_id")
        
        if signal_id and signal_id in self.pending_signals:
            signal = self.pending_signals[signal_id]
            signal.processed = True
            signal.result = {
                "order_id": order_data.get("order_id"),
                "status": "rejected",
                "reason": order_data.get("reject_reason")
            }
            
            # Move to history
            self.signal_history.append(signal)
            del self.pending_signals[signal_id]
    
    def on_position_update(self, event: Event) -> None:
        """Handle position update events."""
        # Update portfolio tracking
        if not self.is_running:
            return
        
        position_data = event.data
        if not position_data or not isinstance(position_data, dict):
            return
        
        # Update portfolio
        symbol = position_data.get("symbol")
        if symbol:
            # Portfolio tracking logic would go here
            pass
    
    def on_account_update(self, event: Event) -> None:
        """Handle account update events."""
        # Update portfolio tracking
        if not self.is_running:
            return
        
        account_data = event.data
        if not account_data or not isinstance(account_data, dict):
            return
        
        # Update portfolio with account information
        # Portfolio tracking logic would go here
    
    def _monitor_strategies(self) -> None:
        """
        Background thread to monitor strategy health and performance.
        Performs periodic evaluation and dynamic adjustment of strategies.
        """
        logger.info("Strategy monitoring thread started")
        
        while self.is_running:
            try:
                # Check if it's time for a performance evaluation
                now = datetime.now()
                time_since_last_check = now - self.last_performance_check
                
                if time_since_last_check >= self.performance_check_interval:
                    # Evaluate strategy performance
                    self.evaluate_performance()
                    self.last_performance_check = now
                
                # Check for any strategy errors or exceptional conditions
                self._check_strategy_health()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in strategy monitoring thread: {str(e)}")
                time.sleep(60)  # Extra sleep on error
        
        logger.info("Strategy monitoring thread stopped")
    
    def _check_strategy_health(self) -> None:
        """
        Check the health of all active strategies and ensembles.
        Restart any crashed strategies if necessary.
        """
        # Check individual strategies
        for strategy_id, strategy in list(self.active_strategies.items()):
            try:
                # Check if strategy is in an error state
                if strategy.state == StrategyState.ERROR:
                    logger.warning(f"Strategy {strategy_id} is in error state, attempting to restart")
                    
                    # Attempt to restart
                    strategy.reset()
                    strategy.start()
                    
                    # Log the recovery attempt
                    self.event_bus.publish(Event(
                        event_type=EventType.STRATEGY_RESTARTED,
                        data={
                            "strategy_id": strategy_id,
                            "name": strategy.name,
                            "time": datetime.now().isoformat()
                        },
                        source="strategy_manager"
                    ))
                
                # Check if strategy has been stuck (no signals for a long time)
                # Implementation would depend on how you track strategy activity
            
            except Exception as e:
                logger.error(f"Error checking health of strategy {strategy_id}: {str(e)}")
        
        # Check ensembles
        for ensemble_id, ensemble in list(self.ensembles.items()):
            try:
                # Ensembles don't have state themselves, but we can check their component strategies
                all_healthy = True
                for strategy_id, (strategy, _) in ensemble.strategies.items():
                    if strategy.state == StrategyState.ERROR:
                        all_healthy = False
                        break
                
                if not all_healthy:
                    logger.warning(f"Ensemble {ensemble_id} has unhealthy strategies")
                    # Could implement automatic remediation here
            
            except Exception as e:
                logger.error(f"Error checking health of ensemble {ensemble_id}: {str(e)}")
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate the performance of all strategies and ensembles.
        Apply dynamic activation/deactivation based on performance metrics.
        
        Returns:
            Dictionary of performance results
        """
        logger.info("Evaluating strategy performance")
        
        results = {
            "strategies": {},
            "ensembles": {},
            "actions_taken": []
        }
        
        # Use performance manager to evaluate strategies
        if self.performance_manager:
            try:
                # Evaluate all strategies
                all_evaluations = self.performance_manager.evaluate_all()
                
                # Process each evaluation
                for strategy_id, evaluation in all_evaluations.items():
                    strategy = self.strategies.get(strategy_id)
                    if not strategy:
                        continue
                    
                    # Store performance metrics
                    results["strategies"][strategy_id] = {
                        "name": strategy.name,
                        "status": evaluation.get("status"),
                        "metrics": evaluation.get("metrics", {}),
                        "current_state": strategy.state.value
                    }
                    
                    # Check for necessary actions based on performance
                    action = evaluation.get("action")
                    if action:
                        action_taken = self._apply_strategy_action(strategy_id, action, evaluation)
                        if action_taken:
                            results["actions_taken"].append(action_taken)
            
            except Exception as e:
                logger.error(f"Error evaluating strategy performance: {str(e)}")
        
        # Evaluate ensembles and adjust weights
        for ensemble_id, ensemble in self.ensembles.items():
            try:
                if ensemble.auto_adjust_weights and ensemble.strategies:
                    # Get performance metrics for component strategies
                    strategy_metrics = {}
                    for strategy_id, (strategy, _) in ensemble.strategies.items():
                        if strategy_id in results["strategies"]:
                            strategy_metrics[strategy_id] = results["strategies"][strategy_id]["metrics"]
                    
                    # Adjust weights based on performance
                    old_weights = {strategy_id: weight for strategy_id, (_, weight) in ensemble.strategies.items()}
                    new_weights = self._calculate_adjusted_weights(strategy_metrics, old_weights)
                    
                    # Apply new weights
                    for strategy_id, new_weight in new_weights.items():
                        if strategy_id in ensemble.strategies:
                            strategy, _ = ensemble.strategies[strategy_id]
                            ensemble.update_strategy_weight(strategy, new_weight)
                    
                    # Record in results
                    results["ensembles"][ensemble_id] = {
                        "name": ensemble.name,
                        "old_weights": old_weights,
                        "new_weights": new_weights
                    }
                    
                    # Log weight adjustment
                    self.event_bus.publish(Event(
                        event_type=EventType.ENSEMBLE_WEIGHTS_ADJUSTED,
                        data={
                            "ensemble_id": ensemble_id,
                            "name": ensemble.name,
                            "old_weights": old_weights,
                            "new_weights": new_weights,
                            "time": datetime.now().isoformat()
                        },
                        source="strategy_manager"
                    ))
            
            except Exception as e:
                logger.error(f"Error adjusting ensemble {ensemble_id} weights: {str(e)}")
        
        return results
    
    def _apply_strategy_action(self, strategy_id: str, action: str, evaluation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply a performance-based action to a strategy.
        
        Args:
            strategy_id: ID of the strategy to apply action to
            action: Action type from StrategyAction enum
            evaluation: Evaluation data for the strategy
            
        Returns:
            Dictionary describing the action taken, or None if no action was taken
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return None
        
        action_taken = None
        
        try:
            if action == StrategyAction.ACTIVATE and strategy.state != StrategyState.RUNNING:
                # Activate a paused or stopped strategy
                strategy.start()
                self.active_strategies[strategy_id] = strategy
                
                action_taken = {
                    "strategy_id": strategy_id,
                    "name": strategy.name,
                    "action": "activated",
                    "reason": evaluation.get("reason", "Performance above threshold")
                }
                
                logger.info(f"Activated strategy {strategy.name} ({strategy_id}) due to good performance")
            
            elif action == StrategyAction.DEACTIVATE and strategy.state == StrategyState.RUNNING:
                # Deactivate a running strategy
                strategy.pause()
                if strategy_id in self.active_strategies:
                    del self.active_strategies[strategy_id]
                
                action_taken = {
                    "strategy_id": strategy_id,
                    "name": strategy.name,
                    "action": "deactivated",
                    "reason": evaluation.get("reason", "Performance below threshold")
                }
                
                logger.info(f"Deactivated strategy {strategy.name} ({strategy_id}) due to poor performance")
            
            # Publish event for the action
            if action_taken:
                self.event_bus.publish(Event(
                    event_type=EventType.STRATEGY_STATUS_CHANGED,
                    data={
                        **action_taken,
                        "time": datetime.now().isoformat(),
                        "metrics": evaluation.get("metrics", {})
                    },
                    source="strategy_manager"
                ))
        
        except Exception as e:
            logger.error(f"Error applying action {action} to strategy {strategy_id}: {str(e)}")
        
        return action_taken
    
    def _calculate_adjusted_weights(self, strategy_metrics: Dict[str, Dict[str, Any]], 
                                  old_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate adjusted strategy weights based on performance metrics.
        
        Args:
            strategy_metrics: Performance metrics for each strategy
            old_weights: Current weights of strategies
            
        Returns:
            Dictionary of new weights
        """
        if not strategy_metrics or not old_weights:
            return old_weights
        
        new_weights = old_weights.copy()
        total_score = 0.0
        
        # Calculate performance score for each strategy
        scores = {}
        for strategy_id, metrics in strategy_metrics.items():
            if strategy_id not in old_weights:
                continue
            
            # Use Sharpe ratio if available, otherwise profit factor or win rate
            score = metrics.get("sharpe_ratio", None)
            if score is None:
                score = metrics.get("profit_factor", None)
            if score is None:
                score = metrics.get("win_rate", 0.5)
            
            # Apply minimum weight to ensure some diversity
            score = max(score, 0.1)  # Ensure no negative scores
            scores[strategy_id] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            for strategy_id, score in scores.items():
                # Weight is proportional to performance score
                new_weights[strategy_id] = score / total_score
        
        return new_weights
    
    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """
        Return information about currently active strategies.
        
        Returns:
            List of strategy information dictionaries
        """
        active_strategies = []
        
        for strategy_id, strategy in self.active_strategies.items():
            strategy_info = {
                "strategy_id": strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "type": strategy.strategy_type.value if hasattr(strategy, 'strategy_type') else "unknown",
                "symbols": strategy.symbols,
                "asset_type": strategy.asset_type,
                "state": strategy.state.value,
                "running_since": strategy.running_since.isoformat() if strategy.running_since else None,
                "broker_id": strategy.broker_id
            }
            
            active_strategies.append(strategy_info)
        
        return active_strategies
    
    def get_strategy_details(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific strategy.
        
        Args:
            strategy_id: ID of the strategy to get details for
            
        Returns:
            Dictionary of strategy details or None if not found
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return None
        
        # Get performance metrics if available
        metrics = {}
        if self.performance_manager:
            try:
                metrics = self.performance_manager.get_strategy_metrics(strategy_id) or {}
            except Exception as e:
                logger.error(f"Error getting metrics for strategy {strategy_id}: {str(e)}")
        
        # Get recent signals
        recent_signals = []
        for signal in reversed(self.signal_history):
            if signal.strategy_id == strategy_id:
                recent_signals.append(signal.to_dict())
            if len(recent_signals) >= 10:  # Limit to 10 most recent
                break
        
        # Compile details
        details = {
            "strategy_id": strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "type": strategy.strategy_type.value if hasattr(strategy, 'strategy_type') else "unknown",
            "symbols": strategy.symbols,
            "asset_type": strategy.asset_type,
            "state": strategy.state.value,
            "parameters": strategy.parameters,
            "running_since": strategy.running_since.isoformat() if strategy.running_since else None,
            "broker_id": strategy.broker_id,
            "metrics": metrics,
            "recent_signals": recent_signals
        }
        
        return details
