#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Orchestrator

This module orchestrates the trading bot's components and lifecycle.
"""

import json
import logging
import signal
import sys
import time
from typing import Dict, List, Any, Optional, Set

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import EventType, TradingPaused, TradingResumed, ForcedExitOrder
from trading_bot.core.risk_manager import RiskManager, PortfolioCircuitBreaker
from trading_bot.core.strategy_executor import StrategyExecutor
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.analytics.trade_logger import TradeLogger
from trading_bot.core.adaptive_position_manager import AdaptivePositionManager
from trading_bot.dashboard.zmq_server import ZMQServer
from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.persistence.event_handlers import PersistenceEventHandler, sync_hot_state_to_durable_storage
from trading_bot.persistence.recovery_manager import RecoveryManager
from trading_bot.persistence.idempotency import IdempotencyManager
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.interfaces import DataProvider, StrategyInterface as Strategy, RiskManager, OrderManager
from trading_bot.utils.config_parser import load_config_file, validate_calendar_spread_config
from trading_bot.config.typed_settings import (
    load_config as typed_load_config, 
    TradingBotSettings,
    OrchestratorSettings,
    RiskSettings,
    BrokerSettings,
    DataSettings
)
from trading_bot.data.data_manager import DataManager
from trading_bot.strategies.options.spreads.calendar_spread import CalendarSpread as CalendarSpreadStrategy
from trading_bot.strategies.stocks.swing import StockSwingTradingStrategy

logger = logging.getLogger(__name__)

class MainOrchestrator:
    """
    Orchestrates the trading bot's components and lifecycle.
    """
    
    def __init__(self, config_path: str = 'config/'):
        """
        Initialize the orchestrator with config.
        
        Args:
            config_path: Path to configuration files
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Initialize state
        self.running = False
        self.paused = False
        self.shutdown_requested = False
        self.paused_reason = None
        
        # Initialize configs
        self.configs = self._load_configs()
        
        # Initialize persistence layer
        self.connection_manager = None
        self.persistence_handler = None
        self.recovery_manager = None
        self.idempotency_manager = None
        
        # Initialize components
        self.broker_manager = None
        self.strategy_executor = None
        self.trade_logger = None
        self.position_manager = None
        self.zmq_server = None
        self.risk_manager = None
        self.circuit_breaker = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize traditional config (for backward compatibility)
        self.config = {}
        self.typed_settings = None
        if TYPED_SETTINGS_AVAILABLE:
            try:
                self.typed_settings = typed_load_config(config_path)
                logger.info(f"Loaded typed settings from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load typed settings: {e}, falling back to legacy config")
        
        if not self.typed_settings:
            self.config = load_config_file(config_path)
            if not self.config:
                raise ValueError(f"Failed to load configuration from {config_path}")
        
        # Initialize components based on configuration
        self._initialize_components()
        
        logger.info("MainOrchestrator initialized")
    
    def initialize(self) -> None:
        """Initialize all components"""
        self.logger.info("Initializing orchestrator...")
        
        # Initialize persistence layer
        self._initialize_persistence()
        
        # Initialize broker manager
        self.broker_manager = MultiBrokerManager(
            self.configs['broker_config'],
            self.event_bus
        )
        
        # Add idempotency_manager to broker_manager if available
        if self.idempotency_manager:
            self.broker_manager.idempotency_manager = self.idempotency_manager
        
        # Initialize position manager
        self.position_manager = AdaptivePositionManager(
            event_bus=self.event_bus
        )
        
        # Initialize risk manager and circuit breaker
        self._initialize_risk_management()
        
        # Initialize trade logger
        self.trade_logger = TradeLogger(
            log_dir='logs',
            event_bus=self.event_bus
        )
        
        # Initialize strategy executor
        self.strategy_executor = StrategyExecutor(
            strategy_config=self.configs['strategy_config'],
            broker_manager=self.broker_manager,
            position_manager=self.position_manager,
            event_bus=self.event_bus
        )
        
        # Initialize ZMQ server for dashboard communication
        self.zmq_server = ZMQServer(
            port=5555,
            event_bus=self.event_bus
        )
        
        # Recover state if configured
        if self.configs['persistence_config']['recovery']['recover_on_startup']:
            self._recover_state()
        
        # Subscribe to events
        self._subscribe_to_events()
    
    def _load_configs(self) -> Dict[str, Any]:
        """
        Load configuration files.
        
        Returns:
            Dictionary of configuration objects
        """
        configs = {}
        config_files = [
            'broker_config.json',
            'strategy_config.json',
            'risk_management.json',
            'persistence_config.json'
        ]
        
        for file in config_files:
            try:
                with open(f"{self.config_path}{file}", 'r') as f:
                    configs[file.replace('.json', '')] = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading {file}: {str(e)}")
                raise
                
        return configs
    
    def _initialize_persistence(self) -> None:
        """Initialize persistence layer"""
        try:
            # Initialize connection manager
            self.connection_manager = ConnectionManager(
                self.configs['persistence_config']['connection']
            )
            
            # Initialize persistence handler
            self.persistence_handler = PersistenceEventHandler(
                self.connection_manager,
                self.event_bus
            )
            
            # Initialize recovery manager
            self.recovery_manager = RecoveryManager(
                self.persistence_handler,
                self.event_bus
            )
            
            # Initialize idempotency manager
            self.idempotency_manager = IdempotencyManager(
                self.persistence_handler
            )
            
            self.logger.info("Persistence layer initialized")
        
        except Exception as e:
            self.logger.error(f"Error initializing persistence layer: {str(e)}")
    
    def _initialize_risk_management(self) -> None:
        """Initialize risk manager and circuit breaker"""
        try:
            # Initialize risk manager
            self.risk_manager = RiskManager(
                self.configs['risk_management'],
                self.event_bus
            )
            
            # Initialize circuit breaker
            self.circuit_breaker = PortfolioCircuitBreaker(
                self.configs['risk_management'],
                self.event_bus
            )
            
            self.logger.info("Risk manager and circuit breaker initialized")
        
        except Exception as e:
            self.logger.error(f"Error initializing risk manager and circuit breaker: {str(e)}")
    
    def _recover_state(self) -> None:
        """Recover state from persistence layer"""
        try:
            self.recovery_manager.recover_state()
            self.logger.info("State recovered from persistence layer")
        
        except Exception as e:
            self.logger.error(f"Error recovering state: {str(e)}")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to events"""
        self.event_bus.on(EventType.TRADING_PAUSED, self._on_trading_paused)
        self.event_bus.on(EventType.TRADING_RESUMED, self._on_trading_resumed)
        self.event_bus.on(EventType.FORCED_EXIT_ORDER, self._on_forced_exit)
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle signals"""
        self.logger.info(f"Received signal {sig}")
        self.shutdown_requested = True
    
    def _on_trading_paused(self, event: EventType) -> None:
        """Handle trading paused event"""
        self.paused = True
        self.paused_reason = event.data['reason']
        self.logger.info(f"Trading paused due to {self.paused_reason}")
    
    def _on_trading_resumed(self, event: EventType) -> None:
        """Handle trading resumed event"""
        self.paused = False
        self.paused_reason = None
        self.logger.info("Trading resumed")
    
    def _on_forced_exit(self, event: EventType) -> None:
        """Handle forced exit event"""
        self.logger.info("Forced exit requested")
        self.shutdown_requested = True
    
    def _main_loop(self) -> None:
        """Main trading loop"""
        last_heartbeat = time.time()
        last_state_sync = time.time()
        sync_interval = self.configs['persistence_config']['sync'].get('sync_interval_seconds', 3600)
        
        self.logger.info("Entering main trading loop")
        
        try:
            while self.running and not self.shutdown_requested:
                # Handle broker polling cycle
                self.broker_manager.poll_cycle()
                
                # Handle strategies
                if not self.paused:
                    self.strategy_executor.execute_strategies()
                    
                # Handle risk management
                self.risk_manager.check_risk_metrics()
                
                # Send heartbeat every 10 seconds
                current_time = time.time()
                if current_time - last_heartbeat >= 10:
                    self.event_bus.emit(EventType.HEARTBEAT, {'timestamp': current_time})
                    last_heartbeat = current_time
                
                # Sync hot state to durable storage periodically
                if (self.persistence_handler and 
                    self.configs['persistence_config']['sync'].get('periodic_sync_enabled', True) and
                    current_time - last_state_sync >= sync_interval):
                    
                    try:
                        sync_hot_state_to_durable_storage(self.persistence_handler)
                        last_state_sync = current_time
                    except Exception as e:
                        self.logger.error(f"Failed to sync state to durable storage: {str(e)}")
                
                # Sleep briefly to avoid CPU spikes
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.stop()
            
        self.logger.info("Exiting main trading loop")
    
    def start(self) -> None:
        """Start the trading bot"""
        if self.running:
            self.logger.warning("Trading bot already running")
            return
            
        self.logger.info("Starting trading bot...")
        
        # Initialize if needed
        if not self.broker_manager:
            self.initialize()
            
        # Connect to brokers
        self.broker_manager.connect_all()
        
        # Start ZMQ server
        self.zmq_server.start()
        
        # Set running state
        self.running = True
        self.paused = False
        
        # Start main loop
        try:
            self._main_loop()
            # Sleep to avoid excessive CPU usage
            time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
        except Exception as e:
            logger.error(f"Error in orchestrator main loop: {e}")
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the orchestrator and all components."""
        if not self.running:
            return
        
        logger.info("Stopping MainOrchestrator")
        self.should_stop.set()
        self.running = False
        
        # Perform cleanup of components
        try:
            # Stop data manager and real-time streams
            if ServiceRegistry.has_service("data_manager"):
                data_manager = ServiceRegistry.get("data_manager")
                data_manager.stop_realtime_stream()
                data_manager.shutdown()
            
            logger.info("Components stopped and cleaned up")
        
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
    
    def _process_cycle(self) -> None:
        """Process a single cycle of the trading system."""
        current_time = datetime.now()
        
        for strategy_name in self.active_strategies:
            # Check if it's time to run this strategy
            last_run = self.last_run_time.get(strategy_name, datetime.min)
            strategy_interval = self._get_strategy_interval(strategy_name)
            
            if current_time - last_run >= strategy_interval:
                self._run_strategy_cycle(strategy_name)
                self.last_run_time[strategy_name] = current_time
    
    def _get_strategy_interval(self, strategy_name: str) -> timedelta:
        """Get the execution interval for a strategy."""
        # Default to 5 minutes if not specified
        interval_minutes = self.config.get("strategies", {}).get(
            strategy_name, {}).get("interval_minutes", 5)
        return timedelta(minutes=interval_minutes)
    
    def _run_strategy_cycle(self, strategy_name: str) -> None:
        """Run a complete cycle for a specific strategy."""
        logger.info(f"Running cycle for strategy: {strategy_name}")
        
        try:
            # Check if trading is paused
            if self.trading_paused:
                logger.info(f"Trading is paused due to {self.pause_reason}, skipping strategy cycle for {strategy_name}")
                return
            
            # Get strategy configuration
            strategy_config = self.config.get("strategies", {}).get(strategy_name, {})
            symbols = strategy_config.get("symbols", [])
            
            if not symbols:
                logger.warning(f"No symbols configured for strategy: {strategy_name}")
                return
            
            # 1. Fetch market data using data manager
            data_manager = ServiceRegistry.get("data_manager")
            market_data = self._fetch_market_data(strategy_name, symbols, data_manager)
            
            if not market_data:
                logger.warning(f"No market data available for strategy: {strategy_name}")
                return
            
            # 2. Generate signals using the strategy
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                logger.warning(f"Strategy not found: {strategy_name}")
                return
                
            signals = strategy.generate_signals(market_data)
            
            if not signals:
                logger.info(f"No signals generated for strategy: {strategy_name}")
                return
            
            logger.info(f"Generated {len(signals)} signals for strategy: {strategy_name}")
            
            # 3. Validate signals with risk manager
            risk_manager = ServiceRegistry.get("risk_manager", None)
            
            if risk_manager:
                approved_signals = []
                for signal in signals:
                    # Convert signal to order format for risk check
                    order = {
                        "symbol": signal.get("symbol"),
                        "side": signal.get("action"),  # 'buy' or 'sell'
                        "quantity": signal.get("quantity", 0),
                        "price": signal.get("price", 0),
                        "stop_price": signal.get("stop_price"),
                        "dollar_amount": signal.get("estimated_value", 0),
                        "strategy": strategy_name
                    }
                    
                    # Perform risk check
                    if hasattr(risk_manager, 'check_trade'):
                        result = risk_manager.check_trade(order)
                        if result.get("approved", False):
                            approved_signals.append(signal)
                            if result.get("warnings"):
                                logger.warning(f"Risk warnings for {signal.get('symbol')}: {', '.join(result.get('warnings', []))}")
                        else:
                            logger.warning(f"Signal rejected by risk manager: {result.get('reason')}")
                    else:
                        # Risk manager doesn't have check_trade method
                        approved_signals.append(signal)
                        logger.warning(f"Risk manager doesn't have check_trade method, signal approved without risk check")
            else:
                # No risk manager available
                approved_signals = signals
                logger.warning("No risk manager available, all signals approved without risk check")
            
            # 4. Execute approved signals
            if approved_signals:
                logger.info(f"Executing {len(approved_signals)} signals for strategy: {strategy_name}")
                self._execute_signals(strategy_name, approved_signals)
            else:
                logger.info(f"No approved signals for strategy: {strategy_name}")
        
        except Exception as e:
            logger.error(f"Error running cycle for strategy '{strategy_name}': {e}")
    
    def _fetch_market_data(self, strategy_name: str, symbols: List[str], data_manager: Any) -> Dict[str, Any]:
        """
        Fetch market data for a strategy.
        
        Args:
            strategy_name: Strategy name
            symbols: List of symbols to fetch data for
            data_manager: Data manager instance
            
        Returns:
            Dictionary with market data
        """
        try:
            # Determine required lookback period (default to 1 year)
            lookback_days = self.config.get("strategies", {}).get(
                strategy_name, {}).get("lookback_days", 365)
            
            # Calculate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Fetch market data
            market_data = data_manager.get_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # If this is a strategy that needs option data, fetch it
            if strategy_name == "calendar_spread":
                # Fetch option chains for each symbol
                option_data = {}
                for symbol in symbols:
                    option_data[symbol] = data_manager.get_option_chain(symbol)
                
                # Add option data to market data
                for symbol, data in market_data.items():
                    if symbol in option_data:
                        data['options'] = option_data[symbol]
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for strategy '{strategy_name}': {e}")
            return {}

    def _execute_signals(self, strategy_name: str, signals: List[Dict[str, Any]]) -> None:
        """
        Execute approved trading signals using the order manager.
        
        Args:
            strategy_name: Name of the strategy that generated the signals
            signals: List of approved trading signals to execute
        """
        try:
            # Get the order manager from service registry
            order_manager = ServiceRegistry.get("order_manager", None)
            
            if not order_manager:
                logger.warning("No order manager available, cannot execute signals")
                return
            
            for signal in signals:
                try:
                    symbol = signal.get("symbol")
                    action = signal.get("action")  # 'buy' or 'sell'
                    price = signal.get("price")   # Can be None for market orders
                    stop_price = signal.get("stop_price")
                    target_price = signal.get("target_price")
                    quantity = signal.get("quantity")
                    risk_pct = signal.get("risk_pct")
                    
                    # Add metadata to the trade
                    metadata = {
                        "strategy": strategy_name,
                        "signal_time": datetime.now().isoformat(),
                        "confidence": signal.get("confidence", 0),
                        "signal_id": signal.get("id", str(uuid.uuid4()))
                    }
                    
                    # Execute the trade
                    logger.info(f"Executing {action} signal for {symbol} from {strategy_name}")
                    
                    result = order_manager.execute_trade(
                        symbol=symbol,
                        side=action,
                        entry_price=price,
                        stop_price=stop_price,
                        target_price=target_price,
                        shares=quantity,
                        risk_pct=risk_pct,
                        strategy_name=strategy_name,
                        metadata=metadata
                    )
                    
                    if result.get("status") == "rejected":
                        logger.warning(f"Order rejected: {result.get('message')}")
                    elif result.get("status") == "error":
                        logger.error(f"Order error: {result.get('message')}")
                    else:
                        logger.info(f"Order placed successfully: {result.get('order_id')}")
                        
                except Exception as signal_error:
                    logger.error(f"Error executing signal for {signal.get('symbol', 'unknown')}: {signal_error}")
        
        except Exception as e:
            logger.error(f"Error executing signals for strategy {strategy_name}: {e}")


def setup_signal_handlers(orchestrator: MainOrchestrator) -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        orchestrator.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) 