#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot Main Module

This module integrates the feature engineering, model training, trade analysis,
visualization components, and secure authentication for brokers.
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import components
from trading_bot.utils.feature_engineering import FeatureEngineering
from trading_bot.models.model_trainer import ModelTrainer
from trading_bot.analysis.trade_analyzer import TradeAnalyzer
from trading_bot.visualization.model_dashboard import ModelDashboard

# Import broker components
from trading_bot.brokers.auth_manager import (
    initialize_auth_system, load_config as load_auth_config
)
from trading_bot.brokers.broker_factory import create_broker_manager
from trading_bot.core.event_bus import get_global_event_bus, Event

# Import strategy components
from trading_bot.core.enhanced_strategy_manager_impl import EnhancedStrategyManager
from trading_bot.core.strategy_manager import StrategyPerformanceManager

# Import recovery components
from trading_bot.core.recovery_controller import RecoveryController
from trading_bot.core.state_manager import StateManager
from trading_bot.monitoring.system_monitor import SystemMonitor
from trading_bot.core.constants import EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_bot")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return config


def initialize_broker_system(broker_config_path: str) -> Tuple[Any, Any, Any, Any]:
    """
    Initialize the broker system with secure authentication, credential store,
    audit logging, and event bus integration.
    
    Args:
        broker_config_path: Path to broker configuration file
        
    Returns:
        Tuple of (broker_manager, credential_store, audit_log, audit_listener)
    """
    # Load broker configuration
    logger.info(f"Loading broker configuration from {broker_config_path}")
    try:
        broker_config = load_auth_config(broker_config_path)
    except Exception as e:
        logger.error(f"Error loading broker configuration: {str(e)}")
        return None, None, None, None
    
    # Initialize the global event bus
    event_bus = get_global_event_bus()
    logger.info("Global event bus initialized")
    
    # Initialize the complete authentication system
    logger.info("Initializing authentication and audit system")
    credential_store, audit_log, audit_listener = initialize_auth_system(broker_config)
    
    if not credential_store:
        logger.error("Failed to initialize credential store")
        return None, None, None, None
    
    # Initialize broker manager with credentials
    logger.info("Creating broker manager with secure credential store")
    broker_manager = create_broker_manager(broker_config)
    
    # Test broker connections
    logger.info("Testing broker connections")
    connection_results = broker_manager.connect_all()
    
    # Log connection results
    for broker_id, success in connection_results.items():
        if success:
            logger.info(f"Successfully connected to {broker_id}")
        else:
            logger.warning(f"Failed to connect to {broker_id}")
    
    # Return initialized components
    return broker_manager, credential_store, audit_log, audit_listener

def prepare_output_dirs(config: Dict[str, Any]) -> None:
    """
    Create necessary output directories.
    
    Args:
        config: Configuration dictionary
    """
    base_dir = config.get('output_dir', './output')
    
    # Create subdirectories
    dirs = [
        base_dir,
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'logs/trades'),
        os.path.join(base_dir, 'visualizations')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_data(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load market data from source.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with loaded dataframes
    """
    data_sources = config.get('data_sources', {})
    result = {}
    
    for source_name, source_config in data_sources.items():
        path = source_config.get('path')
        if path and os.path.exists(path):
            df = pd.read_csv(path, parse_dates=True, index_col=source_config.get('index_col', 0))
            result[source_name] = df
        else:
            print(f"Warning: Could not load data source {source_name} from {path}")
    
    return result

def setup_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all system components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with initialized components
    """
    # Setup parameters for each component
    fe_params = config.get('feature_engineering', {})
    mt_params = config.get('model_trainer', {})
    ta_params = config.get('trade_analyzer', {})
    vd_params = config.get('visualization', {})
    
    # Set output directories
    base_dir = config.get('output_dir', './output')
    fe_params['output_dir'] = os.path.join(base_dir, 'features')
    mt_params['output_dir'] = os.path.join(base_dir, 'models')
    mt_params['model_dir'] = os.path.join(base_dir, 'models')
    ta_params['log_dir'] = os.path.join(base_dir, 'logs/trades')
    vd_params['output_dir'] = os.path.join(base_dir, 'visualizations')
    
    # Initialize broker system if broker configuration is provided
    broker_config_path = config.get('broker_config_path', 'config/broker_config.json')
    broker_manager, credential_store, audit_log, audit_listener = None, None, None, None
    
    if os.path.exists(broker_config_path):
        logger.info(f"Initializing broker system from {broker_config_path}")
        broker_manager, credential_store, audit_log, audit_listener = initialize_broker_system(broker_config_path)
    else:
        logger.warning(f"Broker configuration not found at {broker_config_path}, broker system not initialized")
    
    # Initialize components
    feature_engineering = FeatureEngineering(fe_params)
    model_trainer = ModelTrainer(mt_params)
    trade_analyzer = TradeAnalyzer(ta_params)
    model_dashboard = ModelDashboard(vd_params)
    
    # Connect visualization dashboard to other components
    model_dashboard.connect_components(
        trade_analyzer=trade_analyzer,
        model_trainer=model_trainer,
        feature_engineering=feature_engineering
    )
    
    # Create component dictionary including broker system
    components = {
        'feature_engineering': feature_engineering,
        'model_trainer': model_trainer,
        'trade_analyzer': trade_analyzer,
        'model_dashboard': model_dashboard
    }
    
    # Add broker components if initialized
    if broker_manager:
        components['broker_manager'] = broker_manager
        components['credential_store'] = credential_store
        components['audit_log'] = audit_log
        components['audit_listener'] = audit_listener
        logger.info("Broker system components added to system")
    
    return components

def generate_features(components: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Generate features for all data sources.
    
    Args:
        components: Component dictionary
        data: Data dictionary
        
    Returns:
        Dictionary with features
    """
    feature_engineering = components['feature_engineering']
    features = {}
    
    for source_name, df in data.items():
        print(f"Generating features for {source_name}...")
        features[source_name] = feature_engineering.generate_features(df)
        
    return features

def train_models(components: Dict[str, Any], features: Dict[str, pd.DataFrame], 
                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train models based on configuration.
    
    Args:
        components: Component dictionary
        features: Features dictionary
        config: Configuration dictionary
        
    Returns:
        Dictionary with training results
    """
    model_trainer = components['model_trainer']
    feature_engineering = components['feature_engineering']
    model_configs = config.get('models', [])
    results = {}
    
    for model_config in model_configs:
        model_name = model_config.get('name', 'default')
        model_type = model_config.get('type', 'classification')
        source_name = model_config.get('data_source')
        target_horizon = model_config.get('target_horizon', 5)
        
        if source_name not in features:
            print(f"Warning: Data source {source_name} not found for model {model_name}")
            continue
            
        # Prepare data
        feature_df = features[source_name]
        
        # Generate target labels
        if not model_config.get('target_column'):
            # Generate labels if not specified
            print(f"Generating target labels for model {model_name}...")
            data_df = data.get(source_name)
            if data_df is not None:
                feature_df = feature_engineering.add_return_labels(
                    df=feature_df,
                    future_windows=[target_horizon],
                    thresholds=[0.0, 0.01, 0.02]
                )
                
                # Set target column based on type
                if model_type == 'classification':
                    target_column = f'label_{target_horizon}d_{int(model_config.get("threshold", 1) * 100)}pct'
                else:
                    target_column = f'future_return_{target_horizon}'
            else:
                print(f"Warning: Original data not found for {source_name}, cannot generate labels")
                continue
        else:
            target_column = model_config.get('target_column')
            
        if target_column not in feature_df.columns:
            print(f"Warning: Target column {target_column} not found in features")
            continue
            
        # Prepare train/test dataset
        X, y, metadata = feature_engineering.to_ml_dataset(feature_df, target_column)
        
        # Skip if not enough data
        if len(X) < 100:
            print(f"Warning: Not enough data for model {model_name} ({len(X)} samples)")
            continue
            
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
        
        # Check for regime column
        regime_column = None
        if 'market_regime' in feature_df.columns:
            regime_column = 'market_regime'
            
        # Train model
        print(f"Training model {model_name}...")
        model = model_trainer.train_model(
            X=X_train,
            y=y_train,
            model_type=model_type,
            model_name=model_name
        )
        
        # Cross-validation
        print(f"Performing cross-validation for model {model_name}...")
        cv_results = model_trainer.time_series_cv(
            X=X_train,
            y=y_train,
            model_type=model_type,
            model_name=model_name
        )
        
        # Train regime-specific models if applicable
        if regime_column and model_config.get('use_regime_models', True):
            print(f"Training regime-specific models for {model_name}...")
            regime_models = model_trainer.train_regime_specific_models(
                X=X_train.copy(),
                y=y_train.copy(),
                regime_column=regime_column,
                model_type=model_type,
                base_model_name=model_name
            )
        
        # Evaluate on test set
        print(f"Evaluating model {model_name}...")
        evaluation = model_trainer.evaluate_model(
            X=X_test,
            y=y_test,
            model_name=model_name
        )
        
        # Save model
        print(f"Saving model {model_name}...")
        model_path = model_trainer.save_model(model_name=model_name)
        
        # Store results
        results[model_name] = {
            'model_path': model_path,
            'evaluation': evaluation,
            'cv_results': cv_results,
            'feature_importance': model_trainer.get_top_features(model_name, top_n=20)
        }
        
    return results

def analyze_features_and_models(components: Dict[str, Any], 
                              training_results: Dict[str, Any],
                              features: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze feature importance and model performance.
    
    Args:
        components: Component dictionary
        training_results: Training results
        features: Features dictionary
    """
    model_dashboard = components['model_dashboard']
    
    # Create visualizations
    print("Creating model dashboard...")
    dashboard_paths = model_dashboard.create_dashboard(interactive=True)
    
    # Print dashboard location
    if 'index' in dashboard_paths:
        print(f"Dashboard created at: {dashboard_paths['index']}")
    
    # Print model performance summaries
    for model_name, results in training_results.items():
        print(f"\nModel: {model_name}")
        print("Evaluation metrics:")
        for metric, value in results['evaluation'].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
                
        print("Top 5 features:")
        top_features = list(results['feature_importance'].items())[:5]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
            
def run_backtest(components: Dict[str, Any], features: Dict[str, pd.DataFrame],
               data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> None:
    """
    Run backtest to generate trade signals and analyze performance.
    
    Args:
        components: Component dictionary
        features: Features dictionary
        data: Original data dictionary
        config: Configuration dictionary
    """
    model_trainer = components['model_trainer']
    trade_analyzer = components['trade_analyzer']
    
    # Get backtest configuration
    backtest_config = config.get('backtest', {})
    model_name = backtest_config.get('model_name', 'default')
    data_source = backtest_config.get('data_source')
    
    if data_source not in features:
        print(f"Warning: Data source {data_source} not found for backtest")
        return
        
    # Get feature and price data
    feature_df = features[data_source]
    price_df = data[data_source]
    
    # Check for regime column
    regime_column = None
    if 'market_regime' in feature_df.columns:
        regime_column = 'market_regime'
    
    # Backtest period
    test_size = int(len(feature_df) * backtest_config.get('test_size', 0.2))
    feature_test = feature_df.iloc[-test_size:]
    price_test = price_df.iloc[-test_size:]
    
    print(f"Running backtest with {test_size} periods...")
    
    # Generate predictions for each period
    for i in range(len(feature_test)):
        # Get features for current period
        current_features = feature_test.iloc[i:i+1]
        current_timestamp = feature_test.index[i]
        
        # Get regime if available
        regime = current_features[regime_column].iloc[0] if regime_column else None
        
        # Generate prediction
        try:
            # Get prediction and confidence
            if hasattr(model_trainer.models[model_name], 'predict_proba'):
                # Classification model
                pred_proba = model_trainer.predict_proba(current_features, model_name, 
                                                      regime=regime, regime_column=regime_column)
                prediction = model_trainer.predict(current_features, model_name, 
                                                 regime=regime, regime_column=regime_column)[0]
                confidence = np.max(pred_proba[0])
            else:
                # Regression model
                prediction = model_trainer.predict(current_features, model_name, 
                                                 regime=regime, regime_column=regime_column)[0]
                confidence = None
                
            # Get feature explanation
            explanations = model_trainer.get_feature_explanation(current_features, model_name, regime)
            top_features = explanations[0]['top_features'] if explanations else {}
            
            # Log prediction
            prediction_entry = trade_analyzer.log_prediction(
                timestamp=current_timestamp,
                features=current_features,
                prediction=prediction,
                confidence=confidence,
                top_features=top_features,
                regime=regime if regime else 'unknown',
                model_name=model_name,
                metadata={'price': price_test.iloc[i]['close'] if 'close' in price_test.columns else None}
            )
            
            # Determine actual outcome (if we have future data)
            if i < len(feature_test) - 5:  # Assuming 5-period forward returns
                # For classification models
                if isinstance(prediction, (int, np.integer)):
                    # Get actual direction (1, 0, -1)
                    future_return = price_test.iloc[i+5]['close'] / price_test.iloc[i]['close'] - 1
                    actual_outcome = 1 if future_return > 0.01 else (-1 if future_return < -0.01 else 0)
                    pnl = future_return if prediction == np.sign(future_return) else -future_return
                else:
                    # For regression models
                    future_return = price_test.iloc[i+5]['close'] / price_test.iloc[i]['close'] - 1
                    actual_outcome = future_return
                    pnl = future_return if np.sign(prediction) == np.sign(future_return) else -future_return
                
                # Log outcome
                trade_analyzer.log_trade_outcome(
                    prediction_id=current_timestamp,
                    actual_outcome=actual_outcome,
                    pnl=pnl,
                    trade_metadata={'future_price': price_test.iloc[i+5]['close']}
                )
        except Exception as e:
            print(f"Error in prediction at {current_timestamp}: {str(e)}")
    
    # Analyze backtest results
    print("Analyzing backtest results...")
    
    # Overall performance
    performance = trade_analyzer.analyze_model_performance()
    
    print("\nBacktest Performance:")
    print(f"Total trades: {performance['total_trades']}")
    print(f"Accuracy: {performance['accuracy']:.4f}")
    print(f"Win rate: {performance['win_rate']:.4f}")
    print(f"Profit factor: {performance['profit_factor']:.4f}")
    print(f"Total P&L: {performance['total_pnl']:.4f}")
    
    # Performance by regime
    regime_performance = trade_analyzer.compare_regimes()
    
    print("\nPerformance by Regime:")
    for regime, metrics in regime_performance.items():
        if regime != 'overall':
            print(f"\n{regime.capitalize()} Regime:")
            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Win rate: {metrics['win_rate']:.4f}")
            print(f"  Profit factor: {metrics['profit_factor']:.4f}")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Trading Bot")
    parser.add_argument(
        "--config", type=str, default="config/config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--broker-config", type=str, default="config/multi_broker_config.json",
        help="Path to broker configuration file"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "backtest", "dashboard", "broker_status", "live"], 
        default="train", help="Operation mode"
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return 1
    
    # Initialize broker system if needed
    broker_manager, credential_store, audit_log, audit_listener = None, None, None, None
    if args.mode in ["broker_status", "live", "dashboard"]:
        broker_manager, credential_store, audit_log, audit_listener = initialize_broker_system(args.broker_config)
        if not broker_manager and args.mode != "dashboard":
            logger.error("Failed to initialize broker system. Exiting.")
            return 1
    
    # Prepare output directories
    prepare_output_dirs(config)
    
        # Initialize components dictionary
    components = {
        'broker_manager': broker_manager,
        'credential_store': credential_store,
        'audit_log': audit_log,
        'audit_listener': audit_listener
    }
    
    # Initialize recovery controller
    if args.mode in ["live", "dashboard"]:
        logger.info("Initializing recovery controller")
        try:
            # Create state directory if it doesn't exist
            state_dir = os.path.join(os.path.dirname(args.config), "state")
            os.makedirs(state_dir, exist_ok=True)
            
            # Initialize recovery controller
            recovery_controller = RecoveryController(
                state_dir=state_dir,
                snapshot_interval_seconds=60,  # Take snapshots every minute
                heartbeat_interval_seconds=10,  # Check heartbeats every 10 seconds
                startup_recovery=True,         # Attempt recovery on startup
                enable_auto_restart=True       # Auto-restart failed components
            )
            
            # Add to components
            components['recovery_controller'] = recovery_controller
            
            # Register broker manager with recovery controller if it exists
            if broker_manager:
                recovery_controller.register_component(
                    component_id="broker_manager",
                    component=broker_manager,
                    is_critical=True,
                    recovery_method="connect_all",
                    health_check_method="get_health_status"
                )
            
            logger.info("Recovery controller initialized")
        except Exception as e:
            logger.error(f"Error initializing recovery controller: {str(e)}")
            # Continue without recovery controller
    
    # Initialize strategy manager for live trading and dashboard modes
    strategy_manager = None
    if args.mode in ["live", "dashboard"] and broker_manager:
        logger.info("Initializing Enhanced Strategy Manager")
        strategy_manager = initialize_strategy_manager(components)
        if not strategy_manager and args.mode == "live":
            logger.error("Failed to initialize strategy manager. Exiting.")
            return 1
        
        # Register strategy manager with recovery controller if both exist
        if strategy_manager and 'recovery_controller' in components:
            recovery_controller = components['recovery_controller']
            recovery_controller.register_component(
                component_id="strategy_manager",
                component=strategy_manager,
                is_critical=True,
                recovery_method="restart",
                health_check_method="get_health_status"
            )
            
            # Register strategy manager with state manager
            logger.info("Registering strategy manager with state manager")
            try:
                recovery_controller.state_manager.register_component(
                    name="strategy_manager",
                    component=strategy_manager,
                    get_state_method="get_state",
                    restore_state_method="restore_state"
                )
            except Exception as e:
                logger.error(f"Error registering strategy manager with state manager: {str(e)}")
    
    # Execute based on mode
    if args.mode == "train":
        logger.info("Starting training mode")
        
        # Load data
        data = load_data(config)
        
        # Set up components
        components.update(setup_components(config))
        
        # Generate features
        features = generate_features(components, data)
        
        # Train models
        training_results = train_models(components, features, config)
        
        # Analyze features and model performance
        analyze_features_and_models(components, training_results, features)
        
        logger.info("Training completed")
    
    elif args.mode == "backtest":
        logger.info("Starting backtest mode")
        
        # Load data
        data = load_data(config)
        
        # Set up components
        components.update(setup_components(config))
        
        # Generate features
        features = generate_features(components, data)
        
        # Run backtest
        run_backtest(components, features, data, config)
        
        logger.info("Backtest completed")
    
    elif args.mode == "dashboard":
        logger.info("Starting dashboard mode")
        
        # Connect strategy manager to dashboard
        try:
            # Import dashboard and connect components
            from dashboard.app import app, strategy_dashboard
            
            # Connect strategy manager to dashboard
            if strategy_manager:
                logger.info("Connecting strategy manager to dashboard")
                strategy_dashboard.strategy_manager = strategy_manager
            
            logger.info("Starting dashboard on http://localhost:8050")
            app.run_server(debug=True, port=8050)
        except Exception as e:
            logger.error(f"Error starting dashboard: {str(e)}")
            return 1
    
    elif args.mode == "broker_status":
        logger.info("Checking broker status")
        
        # Show broker status
        show_broker_status(components)
    
    elif args.mode == "live":
        logger.info("Starting live trading mode")
        
        # Set up signal handlers for graceful shutdown
        if 'recovery_controller' in components:
            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, shutting down gracefully...")
                # Stop recovery controller first to take a final snapshot
                recovery_controller = components.get('recovery_controller')
                if recovery_controller:
                    recovery_controller.stop()
                # Publish shutdown event
                event_bus = get_global_event_bus()
                event_bus.publish(Event(EventType.SYSTEM_SHUTDOWN, {"reason": "user_interrupt"}))
                sys.exit(0)
            
            # Register signal handlers
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        
        # Start recovery controller if available
        recovery_controller = components.get('recovery_controller')
        if recovery_controller:
            logger.info("Starting recovery controller")
            recovery_controller.start()
        
        # Start live trading with the strategy manager
        run_live_trading(components)
        
        # Stop recovery controller on shutdown
        if recovery_controller:
            logger.info("Stopping recovery controller")
            recovery_controller.stop()
    
    return 0

def show_broker_status(components: Dict[str, Any]):
    """
    Display status of all configured brokers.
    
    Args:
        components: Component dictionary containing broker manager
    """
    broker_manager = components.get('broker_manager')
    if not broker_manager:
        logger.error("Broker manager not initialized.")
        return
    
    credential_store = components.get('credential_store')
    audit_log = components.get('audit_log')
    
    # Show broker manager status
    logger.info("\n=== Broker System Status ===")
    logger.info(f"Primary broker: {broker_manager.primary_broker_id}")
    logger.info(f"Active broker: {broker_manager.active_broker_id}")
    
    # Show available brokers
    available_brokers = broker_manager.get_available_brokers()
    logger.info(f"\nAvailable brokers: {', '.join(available_brokers) if available_brokers else 'None'}")
    
    # Show broker-specific status
    logger.info("\nBroker Status:")
    for broker_id, broker in broker_manager.brokers.items():
        try:
            status = "Connected" if broker.is_connected() else "Disconnected"
            logger.info(f"{broker_id}: {status}")
            
            # Show account information if connected
            if broker.is_connected():
                try:
                    account_info = broker.get_account_info()
                    logger.info(f"  Account Balance: {account_info.get('balance', 'N/A')}")
                    logger.info(f"  Account Type: {account_info.get('type', 'N/A')}")
                except Exception as e:
                    logger.warning(f"  Could not retrieve account info: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking {broker_id} status: {str(e)}")
    
    # Show asset routing
    logger.info("\nAsset Routing:")
    for asset_type, broker_id in broker_manager.asset_routing.items():
        logger.info(f"  {asset_type}: {broker_id}")
    
    # Show credential store status
    if credential_store:
        broker_ids = credential_store.list_brokers()
        logger.info(f"\nCredential store contains {len(broker_ids)} broker configurations")
    
    # Show audit log status
    if audit_log:
        try:
            # Get count of recent events
            recent_events = audit_log.query_events(
                start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            )
            logger.info(f"\nAudit log contains {len(recent_events)} events today")
        except Exception as e:
            logger.error(f"Error querying audit log: {str(e)}")


def initialize_strategy_manager(components: Dict[str, Any]) -> EnhancedStrategyManager:
    """
    Initialize the Enhanced Strategy Manager and load strategies from configuration.
    
    Args:
        components: Component dictionary containing broker manager
        
    Returns:
        Initialized EnhancedStrategyManager instance
    """
    broker_manager = components.get('broker_manager')
    if not broker_manager:
        logger.error("Broker manager not initialized. Cannot initialize strategy manager.")
        return None
    
    # Initialize the performance manager
    logger.info("Initializing StrategyPerformanceManager")
    performance_manager = StrategyPerformanceManager()
    
    # Initialize the Enhanced Strategy Manager with broker and performance managers
    logger.info("Initializing EnhancedStrategyManager")
    strategy_manager = EnhancedStrategyManager(
        broker_manager=broker_manager,
        performance_manager=performance_manager,
        config={
            "risk_limits": {
                "max_position_per_symbol": 0.05,  # Max 5% allocation per symbol
                "max_allocation_per_strategy": 0.20,  # Max 20% allocation per strategy
                "max_allocation_per_asset_type": 0.50,  # Max 50% allocation per asset type
                "max_total_allocation": 0.80,  # Max 80% of portfolio allocated
                "max_drawdown": 0.10,  # Max 10% drawdown before intervention
                "correlation_threshold": 0.70  # Max correlation between strategies
            }
        }
    )
    
    # Load strategy configurations from the config directory
    strategies_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'strategies')
    if not os.path.exists(strategies_dir):
        logger.warning(f"Strategy configuration directory {strategies_dir} not found")
        return strategy_manager
    
    # Load all strategy configurations
    strategy_configs = []
    for filename in os.listdir(strategies_dir):
        if filename.endswith('_strategies.json'):
            filepath = os.path.join(strategies_dir, filename)
            logger.info(f"Loading strategy configurations from {filepath}")
            
            try:
                with open(filepath, 'r') as f:
                    config_data = json.load(f)
                    if 'strategies' in config_data:
                        strategy_configs.extend(config_data['strategies'])
            except Exception as e:
                logger.error(f"Error loading strategies from {filepath}: {str(e)}")
    
    # Load ensemble configurations if they exist
    ensemble_configs = []
    ensemble_path = os.path.join(strategies_dir, 'strategy_ensembles.json')
    if os.path.exists(ensemble_path):
        logger.info(f"Loading ensemble configurations from {ensemble_path}")
        try:
            with open(ensemble_path, 'r') as f:
                ensemble_data = json.load(f)
                if 'ensembles' in ensemble_data:
                    ensemble_configs = ensemble_data['ensembles']
        except Exception as e:
            logger.error(f"Error loading ensembles from {ensemble_path}: {str(e)}")
    
    # Initialize strategies and ensembles
    if strategy_configs:
        logger.info(f"Loading {len(strategy_configs)} strategies into the EnhancedStrategyManager")
        strategy_manager.load_strategies(strategy_configs)
    
    if ensemble_configs:
        logger.info(f"Creating {len(ensemble_configs)} ensembles in the EnhancedStrategyManager")
        strategy_manager.create_ensembles(ensemble_configs)
    
    # Add the strategy manager to components for access elsewhere
    components['strategy_manager'] = strategy_manager
    
    # Return the initialized strategy manager
    return strategy_manager


def run_live_trading(components: Dict[str, Any]) -> None:
    """
    Run live trading using the configured brokers and strategy manager.
    
    Args:
        components: Component dictionary containing broker manager
    """
    # Get broker manager
    broker_manager = components.get('broker_manager')
    if not broker_manager:
        logger.error("Broker manager not found in components")
        return
    
    # Get recovery controller
    recovery_controller = components.get('recovery_controller')
    
    # Check broker connections
    logger.info("Checking broker connections")
    available_brokers = []
    for broker_id, broker in broker_manager.brokers.items():
        if broker.is_connected():
            available_brokers.append(broker_id)
            logger.info(f"Connected to {broker_id}")
        else:
            try:
                connected = broker.connect()
                if connected:
                    available_brokers.append(broker_id)
                    logger.info(f"Successfully connected to {broker_id}")
                else:
                    logger.warning(f"Failed to connect to {broker_id}")
            except Exception as e:
                logger.error(f"Error connecting to {broker_id}: {str(e)}")
                logger.warning(f"Failed to connect to {broker_id}")
    
    if not available_brokers:
        logger.error("No brokers available. Cannot start trading.")
        return
    
    # Connect to all brokers
    logger.info("Connecting to all brokers...")
    connection_results = broker_manager.connect_all()
    
    available_brokers = []
    for broker_id, success in connection_results.items():
        if success:
            logger.info(f"Successfully connected to {broker_id}")
            available_brokers.append(broker_id)
        else:
            logger.warning(f"Failed to connect to {broker_id}")
    
    if not available_brokers:
        logger.error("No brokers available. Cannot start trading.")
        return
    
    # Check primary broker
    if not broker_manager.is_primary_broker_available():
        logger.warning(f"Primary broker {broker_manager.primary_broker_id} is not available")
        if not broker_manager.reset_to_primary_broker():
            logger.error("Failed to set active broker. Cannot start trading.")
            return
    
    logger.info(f"Using {broker_manager.active_broker_id} as active broker")
    
    # Initialize the strategy manager if not already done
    strategy_manager = components.get('strategy_manager')
    if not strategy_manager:
        logger.info("Strategy manager not initialized. Initializing now...")
        strategy_manager = initialize_strategy_manager(components)
        if not strategy_manager:
            logger.error("Failed to initialize strategy manager. Cannot start trading.")
            return
    
    # Start the strategy manager
    logger.info("Starting strategy manager...")
    try:
        strategy_manager.start_strategies()
        logger.info(f"Started {len(strategy_manager.active_strategies)} active strategies and {len(strategy_manager.ensembles)} ensembles")
    except Exception as e:
        logger.error(f"Error starting strategy manager: {str(e)}")
        return
    
    # Set up market data sources
    logger.info("Starting market data feed...")
    # This would be your implementation of market data subscription
    # based on the symbols required by active strategies
    
    logger.info("Starting live trading session...")
    logger.info("Press Ctrl+C to stop")
    
    event_bus = get_global_event_bus()
    
    try:
        # Main trading loop
        while True:
            try:
                # Record heartbeats for critical components
                if recovery_controller:
                    recovery_controller.record_heartbeat("main_loop")
                    if strategy_manager:
                        recovery_controller.record_heartbeat("strategy_manager")
                    if broker_manager:
                        recovery_controller.record_heartbeat("broker_manager")
                
                # Get positions from all brokers
                try:
                    all_positions = broker_manager.get_all_positions()
                    for broker_id, positions in all_positions.items():
                        logger.info(f"{broker_id} has {len(positions)} open positions")
                except Exception as e:
                    logger.error(f"Error getting positions: {str(e)}")
                    if recovery_controller:
                        # Log the error for recovery
                        recovery_controller.save_crash_report(
                            error=e, 
                            context={"component": "broker_manager", "operation": "get_all_positions"}
                        )
                
                # Log active strategies
                try:
                    active_strategies = strategy_manager.get_active_strategies()
                    logger.info(f"Active strategies: {len(active_strategies)}")
                except Exception as e:
                    logger.error(f"Error getting active strategies: {str(e)}")
                    if recovery_controller:
                        # Log the error for recovery
                        recovery_controller.save_crash_report(
                            error=e, 
                            context={"component": "strategy_manager", "operation": "get_active_strategies"}
                        )
                
                # Check strategy manager performance periodically
                # You might want to put this on a less frequent timer
                if strategy_manager and len(strategy_manager.active_strategies) > 0:
                    logger.info("Evaluating strategy performance...")
                    try:
                        performance_results = strategy_manager.evaluate_performance()
                        for action in performance_results.get("actions_taken", []):
                            logger.info(f"Strategy action: {action['strategy_id']} - {action['action']} - {action['reason']}")
                            
                            # Record transaction to prevent duplicate action on restart
                            if recovery_controller and action.get('action') in ['promote', 'demote', 'disable', 'enable']:
                                tx_id = recovery_controller.generate_transaction_id(action)
                                recovery_controller.log_transaction(
                                    transaction_type="strategy_action",
                                    transaction_id=tx_id,
                                    data=action,
                                    expiry_seconds=3600  # Expire after 1 hour
                                )
                    except Exception as e:
                        logger.error(f"Error evaluating strategy performance: {str(e)}")
                        if recovery_controller:
                            # Log the error for recovery
                            recovery_controller.save_crash_report(
                                error=e, 
                                context={"component": "strategy_manager", "operation": "evaluate_performance"}
                            )
                
                # Take periodic state snapshots when recovery controller is available
                if recovery_controller:
                    # Get health status of components
                    health_status = recovery_controller.get_health_status()
                    logger.debug(f"System health: {health_status['overall_status']}")
                    
                    # Check if any crashes need recovery
                    if health_status.get('is_recovering'):
                        logger.warning("System is currently in recovery mode")
                
                # Wait before next iteration
                import time
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error in main trading loop: {str(e)}")
                
                if recovery_controller:
                    # Log the crash and attempt recovery
                    recovery_controller.save_crash_report(
                        error=e,
                        context={"component": "main_loop", "operation": "trading_cycle"}
                    )
                    
                    # Attempt manual recovery of all components
                    try:
                        recovery_controller.manual_recovery()
                    except Exception as recovery_error:
                        logger.error(f"Error during recovery attempt: {str(recovery_error)}")
                
                # Sleep before retry to avoid tight error loops
                time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    finally:
        # Take a final state snapshot before shutdown
        if recovery_controller:
            try:
                logger.info("Taking final state snapshot before shutdown")
                recovery_controller.state_manager.create_snapshot()
            except Exception as e:
                logger.error(f"Error taking final snapshot: {str(e)}")
        
        # Stop the strategy manager
        logger.info("Stopping strategy manager...")
        try:
            if strategy_manager:
                strategy_manager.stop_strategies()
                logger.info("Strategy manager stopped")
        except Exception as e:
            logger.error(f"Error stopping strategy manager: {str(e)}")
        
        # Disconnect from brokers
        logger.info("Disconnecting from brokers...")
        for broker_id in available_brokers:
            try:
                broker = broker_manager.brokers.get(broker_id)
                if broker and broker.is_connected():
                    broker.disconnect()
                    logger.info(f"Disconnected from {broker_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from {broker_id}: {str(e)}")
        
        logger.info("Trading session ended")


if __name__ == "__main__":
    main()