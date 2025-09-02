#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator Strategy Framework Usage Example

This example demonstrates how to integrate indicator-based strategies with
the Enhanced Strategy Manager and event system.
"""

import os
import logging
import time
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus
from trading_bot.core.constants import EventType
from trading_bot.core.adaptive_scheduler import AdaptiveScheduler, SymbolTier
from trading_bot.core.adaptive_scheduler_factory import create_scheduler
from trading_bot.core.state_manager import StateManager
from trading_bot.core.recovery_controller import RecoveryController
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.strategies.indicator.indicator_strategy_factory import IndicatorStrategyFactory
from trading_bot.strategies.indicator.indicator_data_provider import IndicatorDataProvider

logger = logging.getLogger(__name__)

class IndicatorStrategyManager:
    """
    Manager for indicator-based strategies that integrates with the trading bot framework.
    Handles strategy loading, event processing, and signal generation.
    """
    
    def __init__(self, 
                 broker_manager: MultiBrokerManager,
                 event_bus: EventBus,
                 state_manager: StateManager = None,
                 recovery_controller: RecoveryController = None,
                 config_dir: str = None):
        """
        Initialize the indicator strategy manager.
        
        Args:
            broker_manager: Multi-broker manager instance
            event_bus: Event bus instance
            state_manager: Optional state manager for persistence
            recovery_controller: Optional recovery controller
            config_dir: Directory containing strategy configurations
        """
        self.broker_manager = broker_manager
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.recovery_controller = recovery_controller
        
        # Initialize strategy factory
        self.strategy_factory = IndicatorStrategyFactory(
            broker_manager=broker_manager,
            config_dir=config_dir or 'config/strategies'
        )
        
        # Initialize data provider
        self.data_provider = IndicatorDataProvider(
            broker_manager=broker_manager,
            prefer_broker_indicators=True,
            cache_indicators=True
        )
        
        # Load strategies
        self.strategies = {}
        self.load_strategies()
        
        # Track active symbols
        self.active_symbols = set()
        self._update_active_symbols()
        
        # Register with state manager if available
        if self.state_manager:
            self.state_manager.register_component(
                name="indicator_strategy_manager",
                component=self,
                get_state_method="get_state",
                restore_state_method="restore_state"
            )
        
        # Register with recovery controller if available
        if self.recovery_controller:
            self.recovery_controller.register_component(
                name="indicator_strategy_manager",
                component=self,
                health_check_method="get_health_status",
                restart_method="restart",
                health_check_interval=60
            )
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info(f"Initialized IndicatorStrategyManager with {len(self.strategies)} strategies")
    
    def load_strategies(self) -> None:
        """
        Load all strategy configurations from the config directory.
        """
        self.strategies = self.strategy_factory.load_all_strategies()
        logger.info(f"Loaded {len(self.strategies)} indicator strategies")
    
    def add_strategy(self, config_file: str, symbol: str = None) -> bool:
        """
        Add a strategy from a configuration file.
        
        Args:
            config_file: Path to configuration file
            symbol: Optional symbol to override configuration
            
        Returns:
            True if strategy was successfully added
        """
        try:
            strategy = self.strategy_factory.create_strategy_from_config(config_file, symbol)
            if strategy:
                strategy_id = f"{strategy.name}_{strategy.symbol}"
                self.strategies[strategy_id] = strategy
                self._update_active_symbols()
                logger.info(f"Added strategy: {strategy_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding strategy: {str(e)}")
            return False
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy by ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if strategy was successfully removed
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self._update_active_symbols()
            logger.info(f"Removed strategy: {strategy_id}")
            return True
        return False
    
    def get_active_symbols(self) -> List[str]:
        """
        Get list of active symbols used by strategies.
        
        Returns:
            List of symbol strings
        """
        return list(self.active_symbols)
    
    def _update_active_symbols(self) -> None:
        """Update the set of active symbols from loaded strategies."""
        self.active_symbols = {
            strategy.symbol for strategy in self.strategies.values() 
            if strategy.symbol is not None
        }
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events for strategy execution."""
        if not self.event_bus:
            logger.warning("No event bus available for event subscription")
            return
        
        # Market data updates
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATE, self._on_market_data_update)
        
        # Bar closes
        self.event_bus.subscribe(EventType.BAR_CLOSED, self._on_bar_closed)
        
        # Trade execution
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        
        # System events
        self.event_bus.subscribe(EventType.SYSTEM_START, self._on_system_start)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self._on_system_shutdown)
        
        logger.info("Subscribed to events for strategy execution")
    
    def _on_market_data_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle market data update events.
        
        Args:
            event_type: Event type
            data: Event data
        """
        symbol = data.get('symbol')
        
        if not symbol or symbol not in self.active_symbols:
            return
        
        # Get strategies for this symbol
        symbol_strategies = [
            s for s in self.strategies.values() 
            if s.symbol == symbol
        ]
        
        if not symbol_strategies:
            return
        
        # Check if we need to generate signals (only for non-bar data)
        if data.get('tick_data') and not data.get('bar_data'):
            for strategy in symbol_strategies:
                # Skip strategies that operate on bar data only
                if strategy.timeframe != '1m':
                    continue
                
                try:
                    # Convert tick data to OHLCV format for strategy
                    tick_data = data.get('tick_data')
                    ohlcv_data = {
                        'ohlcv': pd.DataFrame({
                            'timestamp': [datetime.now()],
                            'open': [tick_data.get('price')],
                            'high': [tick_data.get('price')],
                            'low': [tick_data.get('price')],
                            'close': [tick_data.get('price')],
                            'volume': [tick_data.get('volume', 0)]
                        }).set_index('timestamp')
                    }
                    
                    # Generate signal
                    signal = strategy.generate_signal(ohlcv_data)
                    
                    if signal != 0:
                        # Forward signal to the broker/execution engine
                        self._publish_signal_event(strategy, signal)
                
                except Exception as e:
                    logger.error(f"Error processing tick data for strategy {strategy.name}: {str(e)}")
    
    def _on_bar_closed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle bar closed events.
        
        Args:
            event_type: Event type
            data: Event data
        """
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not symbol or symbol not in self.active_symbols:
            return
        
        # Get strategies for this symbol and timeframe
        matching_strategies = [
            s for s in self.strategies.values() 
            if s.symbol == symbol and s.timeframe == timeframe
        ]
        
        if not matching_strategies:
            return
        
        # Get bar data
        bar_data = data.get('bar_data')
        if not bar_data:
            logger.warning(f"No bar data in BAR_CLOSED event for {symbol}")
            return
        
        # Convert to OHLCV format for strategy
        if not isinstance(bar_data, pd.DataFrame):
            try:
                bar_df = pd.DataFrame(bar_data)
            except Exception as e:
                logger.error(f"Error converting bar data to DataFrame: {str(e)}")
                return
        else:
            bar_df = bar_data
        
        # Ensure DataFrame has expected columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in bar_df.columns for col in required_columns):
            logger.warning(f"Bar data missing required columns: {bar_df.columns}")
            return
        
        # Process with each matching strategy
        for strategy in matching_strategies:
            try:
                # Prepare market data
                market_data = {
                    'ohlcv': bar_df
                }
                
                # Update with indicator data
                market_data = self.strategy_factory.update_strategy_data(strategy, market_data)
                
                # Generate signal
                signal = strategy.generate_signal(market_data)
                
                if signal != 0:
                    # Forward signal to the broker/execution engine
                    self._publish_signal_event(strategy, signal)
            
            except Exception as e:
                logger.error(f"Error processing bar data for strategy {strategy.name}: {str(e)}")
    
    def _on_order_filled(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle order filled events to update strategy performance.
        
        Args:
            event_type: Event type
            data: Event data
        """
        order_id = data.get('order_id')
        symbol = data.get('symbol')
        fill_price = data.get('fill_price')
        strategy_id = data.get('strategy_id')
        
        if not symbol or not strategy_id or strategy_id not in self.strategies:
            return
        
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return
        
        # Update strategy performance based on order execution
        # For simplicity, we're just logging here, but you would calculate actual performance
        logger.info(f"Order filled for strategy {strategy.name}: symbol={symbol}, price={fill_price}")
    
    def _on_system_start(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle system start events.
        
        Args:
            event_type: Event type
            data: Event data
        """
        logger.info("System started, initializing all strategies")
        
        # Register active symbols with adaptive scheduler if provided
        scheduler = data.get('scheduler')
        if scheduler and isinstance(scheduler, AdaptiveScheduler):
            for symbol in self.active_symbols:
                scheduler.add_symbol(symbol, SymbolTier.TIER_1)
    
    def _on_system_shutdown(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle system shutdown events.
        
        Args:
            event_type: Event type
            data: Event data
        """
        logger.info("System shutting down, saving strategy states")
        
        # Create state snapshot if state manager is available
        if self.state_manager:
            self.state_manager.create_snapshot()
    
    def _publish_signal_event(self, strategy, signal_value: float) -> None:
        """
        Publish a signal event to the event bus.
        
        Args:
            strategy: Strategy instance that generated the signal
            signal_value: Signal value (-1.0 to 1.0)
        """
        if not self.event_bus:
            return
        
        # Create signal event
        signal_event = {
            'timestamp': datetime.now(),
            'strategy_id': f"{strategy.name}_{strategy.symbol}",
            'strategy_name': strategy.name,
            'symbol': strategy.symbol,
            'signal_value': signal_value,
            'timeframe': strategy.timeframe,
            'position_sizing': strategy.position_sizing,
            'metadata': {
                'in_position': strategy.in_position,
                'current_position': strategy.current_position,
                'entry_price': strategy.entry_price,
                'trade_start_time': strategy.trade_start_time
            }
        }
        
        # Publish the event
        self.event_bus.publish(EventType.STRATEGY_SIGNAL, signal_event)
        logger.info(f"Published signal: {strategy.name} for {strategy.symbol}: {signal_value}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get component state for persistence.
        
        Returns:
            Dictionary with component state
        """
        strategy_states = {}
        for strategy_id, strategy in self.strategies.items():
            strategy_states[strategy_id] = strategy.get_state()
        
        return {
            'strategy_states': strategy_states,
            'active_symbols': list(self.active_symbols)
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore component state from persistence.
        
        Args:
            state: Dictionary with component state
        """
        if not state:
            return
        
        strategy_states = state.get('strategy_states', {})
        for strategy_id, strategy_state in strategy_states.items():
            if strategy_id in self.strategies:
                self.strategies[strategy_id].restore_state(strategy_state)
                logger.info(f"Restored state for strategy: {strategy_id}")
        
        self._update_active_symbols()
    
    def restart(self) -> bool:
        """
        Restart the component (for recovery).
        
        Returns:
            True if restart was successful
        """
        try:
            logger.info("Restarting IndicatorStrategyManager")
            
            # Reload strategies
            self.load_strategies()
            self._update_active_symbols()
            
            # Re-subscribe to events
            self._subscribe_to_events()
            
            return True
        except Exception as e:
            logger.error(f"Error restarting IndicatorStrategyManager: {str(e)}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the component.
        
        Returns:
            Dictionary with health status information
        """
        # Check strategy health
        strategy_status = {}
        for strategy_id, strategy in self.strategies.items():
            if hasattr(strategy, 'get_health_status'):
                strategy_status[strategy_id] = strategy.get_health_status()
        
        # Overall status
        return {
            "status": "healthy",
            "strategy_count": len(self.strategies),
            "active_symbols": len(self.active_symbols),
            "strategy_status": strategy_status
        }


def main():
    """Main example function to demonstrate indicator strategy integration."""
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create event bus
    event_bus = EventBus()
    
    # Create broker manager
    broker_manager = MultiBrokerManager()
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Initialize recovery controller
    recovery_controller = RecoveryController(event_bus=event_bus)
    recovery_controller.register_state_manager(state_manager)
    
    # Create adaptive scheduler
    scheduler = create_scheduler(
        broker_manager=broker_manager,
        event_bus=event_bus
    )
    
    # Register core components with recovery controller
    recovery_controller.register_component(
        "broker_manager",
        broker_manager,
        health_check_method="get_health_status",
        restart_method="restart"
    )
    
    recovery_controller.register_component(
        "scheduler",
        scheduler,
        health_check_method="get_health_status",
        restart_method="restart"
    )
    
    # Create indicator strategy manager
    strategy_manager = IndicatorStrategyManager(
        broker_manager=broker_manager,
        event_bus=event_bus,
        state_manager=state_manager,
        recovery_controller=recovery_controller,
        config_dir='config/strategies'
    )
    
    # Start the system
    event_bus.publish(EventType.SYSTEM_START, {
        'timestamp': datetime.now(),
        'scheduler': scheduler
    })
    
    # Set up scheduler with active symbols
    active_symbols = strategy_manager.get_active_symbols()
    for symbol in active_symbols:
        scheduler.add_symbol(symbol, SymbolTier.TIER_1)
    
    # Register data update tasks
    for timeframe in ['1m', '5m', '15m', '1h', '1d']:
        scheduler.register_task(
            name=f"update_market_data_{timeframe}",
            task_function=lambda tf=timeframe: update_market_data(
                broker_manager, event_bus, tf, active_symbols
            ),
            interval_seconds=get_timeframe_seconds(timeframe),
            priority=3
        )
    
    # Start the scheduler
    scheduler.start()
    
    # Run until interrupted
    try:
        logger.info("Running example indicator strategy system...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping example...")
    finally:
        # Shutdown
        scheduler.stop()
        event_bus.publish(EventType.SYSTEM_SHUTDOWN, {
            'timestamp': datetime.now()
        })


def update_market_data(broker_manager, event_bus, timeframe, symbols):
    """
    Update market data for specified symbols and publish events.
    
    Args:
        broker_manager: Broker manager instance
        event_bus: Event bus instance
        timeframe: Timeframe string
        symbols: List of symbols to update
    """
    logger.debug(f"Updating {timeframe} market data for {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            # Get broker for this symbol
            broker_id = broker_manager.get_preferred_broker_for_symbol(symbol)
            if not broker_id:
                continue
                
            broker = broker_manager.get_broker(broker_id)
            
            # Get latest bar data
            bars = broker.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=1
            )
            
            if not bars or len(bars) == 0:
                continue
            
            # Publish bar closed event
            event_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'bar_data': bars,
                'timestamp': datetime.now()
            }
            
            event_bus.publish(EventType.BAR_CLOSED, event_data)
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {str(e)}")


def get_timeframe_seconds(timeframe):
    """
    Convert timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h')
        
    Returns:
        Seconds as integer
    """
    timeframe = timeframe.lower()
    
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        return minutes * 60
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        return hours * 3600
    elif timeframe.endswith('d'):
        days = int(timeframe[:-1])
        return days * 86400
    else:
        # Default to 5 minutes
        return 300


if __name__ == "__main__":
    main()
