#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Orchestrator

This module provides the central orchestration for the trading bot system,
coordinating different components like data fetching, strategy execution,
risk management, and order management.

It connects all system components, including the hybrid strategy system,
and provides the main execution flow for automated trading.
"""

import logging
import time
import signal
import threading
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
import os
import uuid
import json
import traceback

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.interfaces import DataProvider, StrategyInterface, RiskManager, OrderManager
from trading_bot.data.data_manager import DataManager

# Import strategy components
try:
    # Try the import from the correct location first
    from trading_bot.strategies.factory.strategy_factory import StrategyFactory
except ImportError:
    # Fall back to the old location if that doesn't work
    try:
        from trading_bot.strategies.strategy_factory import StrategyFactory
    except ImportError:
        # Create a simple mock if neither location works
        import logging
        logging.warning("StrategyFactory not found, creating mock implementation")
        
        class StrategyFactory:
            """Mock implementation of StrategyFactory"""
            
            def __init__(self, *args, **kwargs):
                self.active_strategies = {}
            
            def create_strategy(self, strategy_name, parameters=None, metadata=None):
                """Mock method that returns None"""
                logging.warning(f"Mock StrategyFactory: create_strategy called for {strategy_name}")
                return None

# Import hybrid strategy system
try:
    from trading_bot.strategies.hybrid_strategy_adapter import HybridStrategyAdapter
    from trading_bot.strategies.hybrid_strategy_system import HybridStrategySystem
    HYBRID_STRATEGY_AVAILABLE = True
except ImportError:
    HYBRID_STRATEGY_AVAILABLE = False
    logging.warning("Hybrid Strategy System not available")

# Import broker components
try:
    from trading_bot.brokers.broker_interface import BrokerInterface
    from trading_bot.brokers.multi_broker_executor import MultiBrokerExecutor
    MULTI_BROKER_AVAILABLE = True
except ImportError:
    MULTI_BROKER_AVAILABLE = False
    logging.warning("Multi-broker executor not available")

# Import typed settings if available
try:
    from trading_bot.config.typed_settings import (
        load_config as typed_load_config, 
        TradingBotSettings,
        OrchestratorSettings
    )
    TYPED_SETTINGS_AVAILABLE = True
except ImportError:
    TYPED_SETTINGS_AVAILABLE = False
    # Define placeholder classes for when imports are not available
    class TradingBotSettings:
        """Placeholder for TradingBotSettings when import fails"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class OrchestratorSettings:
        """Placeholder for OrchestratorSettings when import fails"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def typed_load_config(config_path):
        """Placeholder for typed_load_config when import fails"""
        logging.warning(f"Typed settings not available, using placeholder for {config_path}")
        return None

logger = logging.getLogger(__name__)

class MainOrchestrator:
    """
    Main orchestrator that coordinates the trading bot components.
    """
    
    def __init__(self, config_path: str = None, settings: Optional[TradingBotSettings] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to the main configuration file
            settings: Optional typed settings object
        """
        # Store initialization parameters
        self.config_path = config_path
        self.typed_settings = settings
        
        # Initialize with default configuration if none provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            if TYPED_SETTINGS_AVAILABLE:
                try:
                    self.typed_settings = typed_load_config(config_path)
                    logger.info(f"Loaded typed settings from {config_path}")
                except Exception as e:
                    logger.warning(f"Could not load typed settings: {e}, falling back to legacy config")
            
            # Fallback to legacy config if needed
            if not self.typed_settings:
                try:
                    from trading_bot.utils.config_parser import load_config_file
                    self.config = load_config_file(config_path)
                except ImportError:
                    logger.warning("Could not load legacy config parser")
        
        self.running = False
        self.should_stop = threading.Event()
        self.active_strategies = set()
        self.last_run_time = {}
        
        # Store registered strategies
        self.strategies = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("MainOrchestrator initialized")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize data manager
            self._initialize_data_manager()
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Initialize risk manager
            self._initialize_risk_manager()
            
            # Initialize order manager
            self._initialize_order_manager()
            
            logger.info("All components initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_data_manager(self):
        """Initialize data manager based on configuration."""
        # Create and register data manager
        try:
            if ServiceRegistry.has_service("data_manager"):
                logger.info("Using existing data manager from registry")
            else:
                data_manager = DataManager(config=self.config.get("data", {}))
                ServiceRegistry.register("data_manager", data_manager)
                logger.info("Initialized and registered data manager")
        except Exception as e:
            logger.error(f"Failed to initialize data manager: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize trading strategies based on configuration."""
        # Create and register strategies
        try:
            # Get strategy config
            strategy_config = self.config.get("strategy", {})
            
            # Check if strategies are already in the registry
            if ServiceRegistry.has_service("strategies"):
                logger.info("Using existing strategies from registry")
                self.strategies = ServiceRegistry.get("strategies")
            else:
                # Initialize the strategy factory
                self.strategy_factory = StrategyFactory()
                logger.info("Initialized strategy factory")
                
                # Initialize strategies
                self.strategies = {}
                active_strategies = strategy_config.get("active_strategies", [])
                
                # Create hybrid strategy if available
                if HYBRID_STRATEGY_AVAILABLE:
                    try:
                        # Create hybrid strategy with configuration
                        hybrid_config = strategy_config.get("hybrid", {})
                        hybrid_strategy = self.strategy_factory.create_strategy(
                            "hybrid",
                            config=hybrid_config
                        )
                        self.strategies["hybrid"] = hybrid_strategy
                        logger.info("Initialized hybrid strategy system")
                        
                        # Add to active strategies if configured
                        if "hybrid" in active_strategies or not active_strategies:
                            self.active_strategies.add("hybrid")
                    except Exception as e:
                        logger.error(f"Error initializing hybrid strategy: {e}")
                        logger.error(traceback.format_exc())
                
                # Create traditional strategies
                for strategy_name in self.strategy_factory.available_strategies():
                    if strategy_name == "hybrid":
                        continue  # Already handled above
                        
                    if strategy_name in active_strategies or not active_strategies:
                        try:
                            strategy_specific_config = strategy_config.get(strategy_name, {})
                            strategy = self.strategy_factory.create_strategy(
                                strategy_name,
                                config=strategy_specific_config
                            )
                            self.strategies[strategy_name] = strategy
                            self.active_strategies.add(strategy_name)
                            logger.info(f"Initialized {strategy_name} strategy")
                        except Exception as e:
                            logger.error(f"Error initializing {strategy_name} strategy: {e}")
                
                # Register strategies with service registry
                ServiceRegistry.register("strategies", self.strategies)
                logger.info(f"Initialized and registered {len(self.strategies)} strategies")
                logger.info(f"Active strategies: {self.active_strategies}")
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            logger.error(traceback.format_exc())
    
    def _initialize_risk_manager(self):
        """Initialize risk manager based on configuration."""
        try:
            if ServiceRegistry.has_service("risk_manager"):
                logger.info("Using existing risk manager from registry")
            else:
                # This would create the actual risk manager
                logger.info("Risk manager would be initialized here")
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            raise
    
    def _initialize_order_manager(self):
        """Initialize order manager based on configuration."""
        # Create and register order manager
        try:
            if ServiceRegistry.has_service("order_manager"):
                logger.info("Using existing order manager from registry")
            else:
                # Check if multi-broker executor is available
                if MULTI_BROKER_AVAILABLE:
                    try:
                        # Initialize multi-broker executor as order manager
                        broker_config = self.config.get("brokers", {})
                        order_manager = MultiBrokerExecutor(config=broker_config)
                        ServiceRegistry.register("order_manager", order_manager)
                        logger.info("Initialized and registered MultiBrokerExecutor as order manager")
                    except Exception as e:
                        logger.error(f"Error initializing multi-broker executor: {e}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning("Multi-broker executor not available, order execution will be limited")
        except Exception as e:
            logger.error(f"Failed to initialize order manager: {e}")
            raise
    
    def start(self):
        """Start the orchestrator and its components."""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("Starting MainOrchestrator")
        self.running = True
        self.should_stop.clear()
        
        try:
            # Start data manager and real-time streams
            if ServiceRegistry.has_service("data_manager"):
                data_manager = ServiceRegistry.get("data_manager")
                data_manager.start_realtime_stream()
            
            # Main processing loop
            while not self.should_stop.is_set():
                try:
                    self._process_cycle()
                except Exception as e:
                    logger.error(f"Error in processing cycle: {e}")
                
                # Sleep to avoid excessive CPU usage
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
        except Exception as e:
            logger.error(f"Error in orchestrator main loop: {e}")
        
        finally:
            self.stop()
    
    def stop(self):
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
    
    def _process_cycle(self):
        """Process a complete orchestration cycle."""
        logger.debug("Processing orchestration cycle")
        
        # Process each active strategy
        for strategy_name in list(self.active_strategies):
            try:
                self._run_strategy_cycle(strategy_name)
            except Exception as e:
                logger.error(f"Error running cycle for strategy {strategy_name}: {e}")
    
    def _run_strategy_cycle(self, strategy_name):
        """Run a complete cycle for a specific strategy."""
        logger.info(f"Running cycle for strategy: {strategy_name}")
        
        try:
            # Get the strategy
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                logger.warning(f"Strategy {strategy_name} not found")
                return
            
            # Get data manager
            data_manager = ServiceRegistry.get("data_manager", None)
            if not data_manager:
                logger.warning("No data manager available, cannot get market data")
                return
            
            # Get strategy config
            strategy_config = self.config.get("strategy", {}).get(strategy_name, {})
            
            # Get symbols for this strategy
            symbols = strategy_config.get("symbols", [])
            if not symbols:
                # Get default symbols from general config
                symbols = self.config.get("symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
            
            # Get timeframes for this strategy
            timeframes = strategy_config.get("timeframes", ["1d"])
            if not timeframes:
                timeframes = ["1d"]  # Default to daily timeframe
                
            # Fetch market data for this strategy
            market_data = self._fetch_market_data(strategy_name, symbols, data_manager)
            if not market_data:
                return
            
            # Generate signals
            signals = []
            for symbol, data in market_data.items():
                # For each available timeframe
                for timeframe in timeframes:
                    try:
                        # For hybrid strategy, pass additional context
                        if strategy_name == "hybrid" and hasattr(strategy, "generate_signals"):
                            signal = strategy.generate_signals(
                                data, 
                                ticker=symbol, 
                                timeframe=timeframe
                            )
                        else:
                            # Standard strategy interface
                            signal = strategy.generate_signals(data)
                            
                        if signal:
                            # Add symbol if not present
                            if "symbol" not in signal:
                                signal["symbol"] = symbol
                                
                            # Add timeframe if not present
                            if "timeframe" not in signal:
                                signal["timeframe"] = timeframe
                                
                            # Add strategy name for tracking
                            signal["strategy"] = strategy_name
                                
                            # Add to signals list if actionable
                            if signal.get("action", "hold").lower() != "hold":
                                signals.append(signal)
                                
                    except Exception as e:
                        logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
                        logger.error(traceback.format_exc())
            
            # Execute approved signals
            if signals:
                logger.info(f"Generated {len(signals)} actionable signals from {strategy_name}")
                self._execute_signals(strategy_name, signals)
            else:
                logger.info(f"No actionable signals generated from {strategy_name}")
                
            # Update last run time
            self.last_run_time[strategy_name] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in strategy cycle for {strategy_name}: {e}")
            logger.error(traceback.format_exc())
    
    def _fetch_market_data(self, strategy_name, symbols, data_manager):
        """Fetch market data for a strategy."""
        try:
            # Get strategy config
            strategy_config = self.config.get("strategy", {}).get(strategy_name, {})
            
            # Get data sources
            data_source = strategy_config.get("data_source", "default")
            
            # Support for multiple timeframes
            timeframes = strategy_config.get("timeframes", ["1d"])
            if isinstance(timeframes, str):
                timeframes = [timeframes]  # Convert single timeframe to list
            
            # Use default if empty
            if not timeframes:
                timeframes = ["1d"]  # Default to daily timeframe
            
            # Get lookback periods
            lookback_periods = strategy_config.get("lookback_periods", 100)
            
            # Initialize market data dict
            market_data = {}
            
            # Fetch data for each symbol
            for symbol in symbols:
                # Create a dictionary for each timeframe
                symbol_data = {}
                
                for timeframe in timeframes:
                    try:
                        # Fetch data for this timeframe
                        df = data_manager.get_historical_data(
                            data_source=data_source,
                            symbol=symbol,
                            timeframe=timeframe,
                            periods=lookback_periods
                        )
                        
                        if df is not None and not df.empty:
                            symbol_data[timeframe] = df
                            logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
                        else:
                            logger.warning(f"No data returned for {symbol} {timeframe}")
                            
                    except Exception as e:
                        logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
                
                # Use primary timeframe (first in list) for now
                # In future, we could pass all timeframes to strategies that support it
                if symbol_data and timeframes[0] in symbol_data:
                    market_data[symbol] = symbol_data[timeframes[0]]
                    
            if not market_data:
                logger.warning(f"No market data fetched for any symbols in {strategy_name}")
                return None
                
            return market_data
        
        except Exception as e:
            logger.error(f"Error fetching market data for strategy {strategy_name}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _execute_signals(self, strategy_name, signals):
        """
        Execute approved trading signals using the order manager.
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
                    order_manager.place_order(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        stop_price=stop_price,
                        target_price=target_price,
                        metadata=metadata
                    )
                
                except Exception as e:
                    logger.error(f"Error executing signal for {signal.get('symbol', 'unknown')}: {e}")
            
        except Exception as e:
            logger.error(f"Error in signal execution for strategy {strategy_name}: {e}")
    
    # Public API methods used by other components like the assistant
    
    def run_pipeline(self, strategy_name=None):
        """Run the trading pipeline, optionally for a specific strategy."""
        result = {
            "success": False,
            "message": "",
            "executed_strategies": [],
            "signals_generated": 0,
            "orders_placed": 0,
            "errors": []
        }
        
        try:
            # Run for specific strategy
            if strategy_name:
                if strategy_name in self.active_strategies:
                    logger.info(f"Running pipeline for strategy: {strategy_name}")
                    self._run_strategy_cycle(strategy_name)
                    result["success"] = True
                    result["message"] = f"Pipeline executed for strategy: {strategy_name}"
                    result["executed_strategies"] = [strategy_name]
                else:
                    logger.warning(f"Strategy {strategy_name} not active or does not exist")
                    result["message"] = f"Strategy {strategy_name} not active or does not exist"
                    result["errors"] = [f"Strategy {strategy_name} not found in active strategies"]
            # Run for all active strategies
            else:
                logger.info("Running pipeline for all active strategies")
                self._process_cycle()
                result["success"] = True
                result["message"] = "Pipeline executed for all active strategies"
                result["executed_strategies"] = list(self.active_strategies)
        except Exception as e:
            error_msg = f"Error running pipeline: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            result["message"] = error_msg
            result["errors"] = [error_msg]
        
        return result
    
    def get_approved_opportunities(self):
        """Get current approved trading opportunities."""
        # This would normally query from the risk manager or opportunity database
        # For now, return a simple demo result
        return [
            {"symbol": "AAPL", "strategy": "MomentumStrategy", "confidence": 0.85},
            {"symbol": "MSFT", "strategy": "BreakoutStrategy", "confidence": 0.78}
        ]
    
    def get_market_regime(self):
        """Get the current market regime assessment."""
        # Check if hybrid strategy is available and has regime detection
        if "hybrid" in self.strategies and hasattr(self.strategies["hybrid"], "hybrid_system"):
            try:
                # Get market conditions from hybrid strategy system
                hybrid_strategy = self.strategies["hybrid"]
                market_conditions = hybrid_strategy.hybrid_system.get_market_conditions()
                
                if market_conditions:
                    return market_conditions
            except Exception as e:
                logger.error(f"Error getting market regime from hybrid strategy: {e}")
        
        # Fallback to simple demo result
        return {"regime": "Bullish", "confidence": 0.82, "trend_strength": "Medium"} 