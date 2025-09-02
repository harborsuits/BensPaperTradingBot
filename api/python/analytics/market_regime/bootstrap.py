"""
Market Regime System Bootstrap

This module initializes and configures the market regime detection system,
and integrates it with the main BensBot trading system.
"""

import logging
import os
import glob
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Import system components
from trading_bot.core.event_bus import EventBus
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.accounting.trade_accounting import TradeAccounting
from trading_bot.portfolio.capital_allocator import CapitalAllocator

# Import regime components
from trading_bot.analytics.market_regime.integration import MarketRegimeManager, initialize_regime_system
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.analytics.market_regime.ml_classifier import MarketRegimeMLClassifier, load_ml_classifier

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "monitoring_interval_seconds": 300,  # 5 minutes
    "update_interval_seconds": 3600,     # 1 hour
    "primary_timeframe": "1d",           # Primary timeframe for regime detection
    "min_data_points": 100,              # Minimum data points for regime detection
    "default_timeframes": ["1d", "4h", "1h"],  # Default timeframes to track
    "watched_symbols": [                 # Default symbols to monitor
        "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "SPY", "QQQ", "IWM"
    ],
    "detector": {
        "auto_update": True,
        "detection_interval_seconds": 1800,  # 30 minutes
        "emit_events": True
    },
    "parameter_optimizer": {
        "transition_smoothing": True,
        "smoothing_factor": 0.3,
        "max_performance_history": 100
    },
    "performance_tracker": {
        "tracked_metrics": ["win_rate", "profit_factor", "sharpe_ratio", "sortino_ratio", 
                          "max_drawdown", "avg_profit", "avg_loss", "expectancy", "returns"],
        "max_metric_history": 100,
        "max_time_series": 500
    },
    "strategy_selector": {
        "scoring_weights": {
            "profit_factor": 0.3,
            "sharpe_ratio": 0.2,
            "win_rate": 0.1,
            "expectancy": 0.2,
            "correlation": 0.1,
            "sample_size": 0.1
        },
        "min_strategies": 1,
        "max_strategies": 5,
        "min_score_threshold": 0.4,
        "correlation_penalty": 0.5
    }
}

# Default initial strategy parameters by regime
DEFAULT_PARAMETERS = {
    "trend_following": {
        MarketRegimeType.TRENDING_UP: {
            "trailing_stop_pct": 5.0,            # Wider trailing stop in uptrend
            "take_profit_pct": 15.0,             # Higher take profit in uptrend
            "stop_loss_pct": 7.0,                # Wider stop loss in uptrend
            "entry_delay": 1,                    # Quick entry in uptrend
            "exit_delay": 3,                     # Patient exit in uptrend
            "position_size_factor": 1.0          # Standard position size
        },
        MarketRegimeType.TRENDING_DOWN: {
            "trailing_stop_pct": 4.0,            # Tighter trailing stop in downtrend
            "take_profit_pct": 12.0,             # Lower take profit in downtrend
            "stop_loss_pct": 5.0,                # Tighter stop loss in downtrend
            "entry_delay": 2,                    # Careful entry in downtrend
            "exit_delay": 1,                     # Quick exit in downtrend
            "position_size_factor": 0.8          # Reduced position size
        },
        MarketRegimeType.VOLATILE: {
            "trailing_stop_pct": 8.0,            # Wider trailing stop in volatile markets
            "take_profit_pct": 20.0,             # Higher take profit in volatile markets
            "stop_loss_pct": 10.0,               # Wider stop loss in volatile markets
            "entry_delay": 3,                    # Careful entry in volatile markets
            "exit_delay": 1,                     # Quick exit in volatile markets
            "position_size_factor": 0.6          # Reduced position size
        },
        MarketRegimeType.RANGE_BOUND: {
            "trailing_stop_pct": 3.0,            # Tighter trailing stop in range-bound markets
            "take_profit_pct": 8.0,              # Lower take profit in range-bound markets
            "stop_loss_pct": 4.0,                # Tighter stop loss in range-bound markets
            "entry_delay": 2,                    # Standard entry in range-bound markets
            "exit_delay": 2,                     # Standard exit in range-bound markets
            "position_size_factor": 0.9          # Slightly reduced position size
        },
        MarketRegimeType.NORMAL: {
            "trailing_stop_pct": 5.0,            # Standard trailing stop
            "take_profit_pct": 12.0,             # Standard take profit
            "stop_loss_pct": 6.0,                # Standard stop loss
            "entry_delay": 2,                    # Standard entry delay
            "exit_delay": 2,                     # Standard exit delay
            "position_size_factor": 1.0          # Standard position size
        }
    },
    "mean_reversion": {
        MarketRegimeType.TRENDING_UP: {
            "entry_threshold": 2.0,              # Higher entry threshold in uptrend
            "exit_threshold": 0.0,               # Return to mean exit threshold
            "stop_loss_pct": 5.0,                # Moderate stop loss in uptrend
            "take_profit_pct": 7.0,              # Moderate take profit in uptrend
            "lookback_periods": 20,              # Standard lookback
            "position_size_factor": 0.8          # Reduced position size (not ideal in uptrend)
        },
        MarketRegimeType.TRENDING_DOWN: {
            "entry_threshold": 2.0,              # Higher entry threshold in downtrend
            "exit_threshold": 0.0,               # Return to mean exit threshold
            "stop_loss_pct": 5.0,                # Moderate stop loss in downtrend
            "take_profit_pct": 7.0,              # Moderate take profit in downtrend
            "lookback_periods": 20,              # Standard lookback
            "position_size_factor": 0.8          # Reduced position size (not ideal in downtrend)
        },
        MarketRegimeType.VOLATILE: {
            "entry_threshold": 3.0,              # Higher entry threshold in volatile markets
            "exit_threshold": 0.0,               # Return to mean exit threshold
            "stop_loss_pct": 7.0,                # Wider stop loss in volatile markets
            "take_profit_pct": 10.0,             # Higher take profit in volatile markets
            "lookback_periods": 15,              # Shorter lookback in volatile markets
            "position_size_factor": 0.6          # Reduced position size in volatile markets
        },
        MarketRegimeType.RANGE_BOUND: {
            "entry_threshold": 1.5,              # Lower entry threshold in range-bound markets
            "exit_threshold": 0.0,               # Return to mean exit threshold
            "stop_loss_pct": 3.0,                # Tighter stop loss in range-bound markets
            "take_profit_pct": 5.0,              # Lower take profit in range-bound markets
            "lookback_periods": 25,              # Longer lookback in range-bound markets
            "position_size_factor": 1.2          # Increased position size (ideal for range-bound)
        },
        MarketRegimeType.NORMAL: {
            "entry_threshold": 2.0,              # Standard entry threshold
            "exit_threshold": 0.0,               # Standard exit threshold
            "stop_loss_pct": 4.0,                # Standard stop loss
            "take_profit_pct": 6.0,              # Standard take profit
            "lookback_periods": 20,              # Standard lookback
            "position_size_factor": 1.0          # Standard position size
        }
    },
    "breakout": {
        MarketRegimeType.TRENDING_UP: {
            "breakout_threshold": 1.5,           # Lower threshold in uptrend (already breaking out)
            "stop_loss_pct": 5.0,                # Moderate stop loss in uptrend
            "take_profit_pct": 15.0,             # Higher take profit in uptrend
            "consolidation_periods": 10,         # Shorter consolidation detection in uptrend
            "confirmation_periods": 2,           # Quick confirmation in uptrend
            "position_size_factor": 1.2          # Increased position size (good in uptrend)
        },
        MarketRegimeType.TRENDING_DOWN: {
            "breakout_threshold": 1.5,           # Lower threshold in downtrend (already breaking out)
            "stop_loss_pct": 5.0,                # Moderate stop loss in downtrend
            "take_profit_pct": 12.0,             # Moderate take profit in downtrend
            "consolidation_periods": 10,         # Shorter consolidation detection in downtrend
            "confirmation_periods": 2,           # Quick confirmation in downtrend
            "position_size_factor": 1.0          # Standard position size
        },
        MarketRegimeType.VOLATILE: {
            "breakout_threshold": 2.5,           # Higher threshold in volatile markets
            "stop_loss_pct": 8.0,                # Wider stop loss in volatile markets
            "take_profit_pct": 18.0,             # Higher take profit in volatile markets
            "consolidation_periods": 8,          # Shorter consolidation detection in volatile markets
            "confirmation_periods": 3,           # More confirmation needed in volatile markets
            "position_size_factor": 0.8          # Reduced position size in volatile markets
        },
        MarketRegimeType.RANGE_BOUND: {
            "breakout_threshold": 2.0,           # Standard threshold in range-bound markets
            "stop_loss_pct": 4.0,                # Tighter stop loss in range-bound markets
            "take_profit_pct": 10.0,             # Moderate take profit in range-bound markets
            "consolidation_periods": 15,         # Longer consolidation detection in range-bound markets
            "confirmation_periods": 3,           # More confirmation needed in range-bound markets
            "position_size_factor": 0.9          # Slightly reduced position size
        },
        MarketRegimeType.NORMAL: {
            "breakout_threshold": 2.0,           # Standard threshold
            "stop_loss_pct": 5.0,                # Standard stop loss
            "take_profit_pct": 12.0,             # Standard take profit
            "consolidation_periods": 12,         # Standard consolidation detection
            "confirmation_periods": 2,           # Standard confirmation
            "position_size_factor": 1.0          # Standard position size
        }
    }
}

# Default timeframe preferences for different regimes
DEFAULT_TIMEFRAME_MAPPINGS = {
    "default": {
        MarketRegimeType.TRENDING_UP: "1d",     # Longer timeframe for trending markets
        MarketRegimeType.TRENDING_DOWN: "1d",   # Longer timeframe for trending markets
        MarketRegimeType.VOLATILE: "4h",        # Medium timeframe for volatile markets
        MarketRegimeType.RANGE_BOUND: "1h",     # Shorter timeframe for range-bound markets
        MarketRegimeType.NORMAL: "4h"           # Medium timeframe for normal markets
    }
}

# Default strategy configurations
DEFAULT_STRATEGY_CONFIGS = {
    "trend_following": {
        "name": "Trend Following Strategy",
        "compatible_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "SPY", "QQQ"],
        "compatible_timeframes": ["1d", "4h", "1h"],
        "compatible_regimes": ["trending_up", "trending_down", "normal"],
        "description": "Follows market trends using moving averages and momentum indicators"
    },
    "mean_reversion": {
        "name": "Mean Reversion Strategy",
        "compatible_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "SPY", "QQQ", "IWM"],
        "compatible_timeframes": ["1d", "4h", "1h", "15m"],
        "compatible_regimes": ["range_bound", "normal"],
        "description": "Trades price reversions to the mean using oversold/overbought conditions"
    },
    "breakout": {
        "name": "Breakout Strategy",
        "compatible_symbols": ["AAPL", "TSLA", "NFLX", "NVDA", "AMD", "SHOP", "SQ", "ROKU"],
        "compatible_timeframes": ["1d", "4h", "1h"],
        "compatible_regimes": ["trending_up", "trending_down", "volatile"],
        "description": "Capitalizes on price breakouts from consolidation patterns"
    }
}

def load_custom_config(config_path: str) -> Dict[str, Any]:
    """
    Load custom configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return DEFAULT_CONFIG
        
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        
        logger.info(f"Loaded custom regime configuration from {config_path}")
        return custom_config
        
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG

def initialize_regime_system_with_defaults(
    event_bus: EventBus,
    broker_manager: MultiBrokerManager,
    trade_accounting: TradeAccounting,
    capital_allocator: Optional[CapitalAllocator] = None,
    config_path: Optional[str] = None
) -> MarketRegimeManager:
    """
    Initialize the market regime system with default settings.
    
    Args:
        event_bus: System event bus
        broker_manager: Broker manager
        trade_accounting: Trade accounting
        capital_allocator: Optional capital allocator
        config_path: Optional path to custom configuration file
        
    Returns:
        Initialized MarketRegimeManager
    """
    try:
        # Load configuration
        config = DEFAULT_CONFIG
        if config_path:
            custom_config = load_custom_config(config_path)
            config = {**DEFAULT_CONFIG, **custom_config}  # Merge with defaults
        
        logger.info("Initializing market regime system with configuration")
        
        # Initialize regime manager
        manager = initialize_regime_system(
            event_bus, broker_manager, trade_accounting, capital_allocator,
            config["watched_symbols"], config
        )
        
        # Register default strategy configurations
        for strategy_id, strategy_config in DEFAULT_STRATEGY_CONFIGS.items():
            manager.strategy_selector.register_strategy(strategy_id, strategy_config)
            logger.info(f"Registered strategy configuration: {strategy_id}")
        
        # Register default parameters
        for strategy_id, regime_params in DEFAULT_PARAMETERS.items():
            for regime_type, params in regime_params.items():
                manager.parameter_optimizer.update_optimal_parameters(
                    strategy_id, regime_type, params
                )
                logger.info(f"Registered default parameters for {strategy_id} in {regime_type}")
        
        # Set up timeframe mappings
        for symbol, mappings in DEFAULT_TIMEFRAME_MAPPINGS.items():
            for regime_type, timeframe in mappings.items():
                if symbol == "default":
                    # Apply to all watched symbols
                    for watched_symbol in config["watched_symbols"]:
                        manager.strategy_selector.set_timeframe_mapping(
                            watched_symbol, regime_type, timeframe
                        )
                        logger.debug(f"Set timeframe mapping for {watched_symbol} in {regime_type} to {timeframe}")
                else:
                    manager.strategy_selector.set_timeframe_mapping(
                        symbol, regime_type, timeframe
                    )
                    logger.debug(f"Set timeframe mapping for {symbol} in {regime_type} to {timeframe}")
        
        logger.info(f"Market regime system initialized with {len(config['watched_symbols'])} symbols")
        return manager
        
    except Exception as e:
        logger.error(f"Error initializing regime system: {str(e)}")
        raise

def setup_performance_collection(manager: MarketRegimeManager) -> None:
    """
    Configure performance data collection for regime classification improvement.
    
    Args:
        manager: Market regime manager
    """
    try:
        # Set up additional event subscriptions for performance tracking
        
        # Trade events (already handled by manager)
        
        # Position update events
        manager.event_bus.register("position_update", manager._handle_position_update)
        
        # Market data quality events
        manager.event_bus.register("data_quality_issue", manager._handle_data_quality_issue)
        
        # Strategy update events
        manager.event_bus.register("strategy_update", manager._handle_strategy_update)
        
        # Execution quality events
        manager.event_bus.register("execution_report", manager._handle_execution_report)
        
        logger.info("Performance collection configured")
        
    except Exception as e:
        logger.error(f"Error setting up performance collection: {str(e)}")

def register_with_main_system(
    manager: MarketRegimeManager,
    main_system: Any
) -> None:
    """
    Register the regime manager with the main system.
    
    Args:
        manager: Market regime manager
        main_system: Main trading system
    """
    try:
        # Add regime manager to main system
        if hasattr(main_system, "register_component"):
            main_system.register_component("market_regime_manager", manager)
        
        # Register with dashboard if available
        if hasattr(main_system, "dashboard") and hasattr(main_system.dashboard, "register_module"):
            from trading_bot.dashboard.components.regime_dashboard import setup_regime_routes
            setup_regime_routes(main_system.dashboard.app)
        
        logger.info("Market regime manager registered with main system")
        
    except Exception as e:
        logger.error(f"Error registering with main system: {str(e)}")

# Initialize ML classifier and load models if available
def setup_ml_classifier(detector_config: Dict[str, Any]) -> Optional[MarketRegimeMLClassifier]:
    """
    Set up ML classifier for regime detection if models are available.
    
    Args:
        detector_config: Configuration for detector
        
    Returns:
        MarketRegimeMLClassifier instance or None if no models available
    """
    try:
        # Check if ML classifier is enabled
        if not detector_config.get('use_ml_classifier', True):
            logger.info("ML classifier disabled in configuration")
            return None
        
        # Configure model directory
        model_dir = detector_config.get('ml_model_dir', 'data/market_regime/models')
        
        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if any models exist
        model_files = glob.glob(os.path.join(model_dir, '*_model_*.pkl'))
        
        if not model_files:
            logger.info("No ML models found, using rule-based classification")
            return None
        
        # Get preferred model type
        model_type = detector_config.get('ml_model_type', 'rf')
        
        # Load classifier
        classifier = load_ml_classifier(model_dir, model_type)
        
        if classifier and classifier.model is not None:
            logger.info(f"Loaded {model_type} ML classifier for regime detection")
            return classifier
        else:
            logger.info("Could not load ML classifier, using rule-based classification")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up ML classifier: {str(e)}")
        return None

# Setup scheduled optimization
def setup_parameter_optimization(manager: MarketRegimeManager, config: Dict[str, Any]) -> None:
    """
    Set up scheduled parameter optimization based on collected data.
    
    Args:
        manager: Market regime manager
        config: Configuration parameters
    """
    try:
        # Check if scheduled optimization is enabled
        if not config.get('parameter_optimizer', {}).get('enable_scheduled_optimization', False):
            return
        
        # Get optimization interval
        optimization_interval = config.get('parameter_optimizer', {}).get(
            'optimization_interval_hours', 168)  # Default: weekly
        
        # Convert to seconds
        optimization_interval_seconds = optimization_interval * 3600
        
        # Register a timer event for optimization
        def schedule_optimization():
            logger.info("Scheduling parameter optimization")
            manager.event_bus.emit("scheduled_optimization", {
                "timestamp": datetime.now().isoformat(),
                "strategy_ids": list(DEFAULT_PARAMETERS.keys())
            })
        
        # Register timer event
        manager.event_bus.register_timer(
            "scheduled_optimization", 
            optimization_interval_seconds,
            schedule_optimization
        )
        
        # Register event handler for optimization
        def handle_optimization_event(event):
            from trading_bot.scripts.regime_parameter_optimizer import optimize_all_regimes
            
            strategies = event.data.get("strategy_ids", [])
            logger.info(f"Running scheduled optimization for {len(strategies)} strategies")
            
            # Data directory relative to project root
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "data", "market_regime")
            
            output_dir = os.path.join(data_dir, "parameters")
            
            for strategy_id in strategies:
                try:
                    logger.info(f"Optimizing parameters for {strategy_id}")
                    optimize_all_regimes(strategy_id, data_dir, output_dir)
                    
                    # Load optimized parameters
                    param_file = os.path.join(output_dir, f"{strategy_id}_parameters.json")
                    if os.path.exists(param_file):
                        with open(param_file, 'r') as f:
                            optimized_params = json.load(f)
                        
                        # Update parameter optimizer
                        for regime_str, params in optimized_params.items():
                            try:
                                regime_type = MarketRegimeType(regime_str)
                                manager.parameter_optimizer.update_optimal_parameters(
                                    strategy_id, regime_type, params
                                )
                                logger.info(f"Updated {strategy_id} parameters for {regime_str}")
                            except Exception as e:
                                logger.error(f"Error updating {strategy_id} parameters for {regime_str}: {str(e)}")
                                
                except Exception as e:
                    logger.error(f"Error optimizing {strategy_id}: {str(e)}")
        
        # Register event handler
        manager.event_bus.register("scheduled_optimization", handle_optimization_event)
        
        logger.info(f"Scheduled parameter optimization configured (every {optimization_interval} hours)")
        
    except Exception as e:
        logger.error(f"Error setting up parameter optimization: {str(e)}")

# Main initialization function
def setup_market_regime_system(
    main_system: Any,
    config_path: Optional[str] = None
) -> MarketRegimeManager:
    """
    Complete setup for the market regime system.
    
    Args:
        main_system: Main trading system object
        config_path: Optional path to configuration file
        
    Returns:
        Initialized MarketRegimeManager
    """
    try:
        logger.info("Setting up market regime system")
        
        # Extract required components from main system
        event_bus = main_system.event_bus
        broker_manager = main_system.broker_manager
        trade_accounting = main_system.trade_accounting
        capital_allocator = main_system.capital_allocator if hasattr(main_system, "capital_allocator") else None
        
        # Load configuration
        config = DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        
        # Setup ML classifier
        ml_classifier = setup_ml_classifier(config.get('detector', {}))
        
        # Initialize regime system with ML classifier
        manager = initialize_regime_system_with_defaults(
            event_bus, broker_manager, trade_accounting, capital_allocator, config_path
        )
        
        # If ML classifier is available, add it to detector
        if ml_classifier:
            manager.detector.set_ml_classifier(ml_classifier)
        
        # Set up performance collection
        setup_performance_collection(manager)
        
        # Setup parameter optimization
        setup_parameter_optimization(manager, config)
        
        # Register with main system
        register_with_main_system(manager, main_system)
        
        logger.info("Market regime system setup complete")
        return manager
        
    except Exception as e:
        logger.error(f"Failed to set up market regime system: {str(e)}")
        raise
