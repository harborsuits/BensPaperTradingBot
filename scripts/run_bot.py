#!/usr/bin/env python
"""
BensBot Trading System - Unified Entry Point

This script provides a unified entry point to run the trading system in different modes:
- Live trading with real brokers (Tradier, Alpaca, E*Trade)
- Paper trading with simulated execution
- Backtesting on historical data

With robust crash recovery and persistent state management.
"""

import argparse
import logging
import sys
import os
import yaml
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Event system
from trading_bot.event_system.event_bus import EventBus

# Broker components
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.brokers.credential_store import EncryptedFileStore
from trading_bot.core.main_orchestrator import MainOrchestrator

# Broker adapters
from trading_bot.brokers.tradier.adapter import TradierAdapter
from trading_bot.brokers.alpaca.adapter import AlpacaAdapter
from trading_bot.brokers.paper.adapter import PaperTradeAdapter, PaperTradeConfig

# Persistence and recovery
from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.persistence.recovery_manager import RecoveryManager
from trading_bot.persistence.idempotency import IdempotencyManager
from trading_bot.persistence.order_repository import OrderRepository
from trading_bot.persistence.position_repository import PositionRepository
from trading_bot.persistence.fill_repository import FillRepository
from trading_bot.persistence.pnl_repository import PnLRepository

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bensbot.log')
    ]
)
logger = logging.getLogger('run_bot')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                import json
                config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        sys.exit(1)


def setup_broker(mode: str, broker_name: str, config: Dict[str, Any], event_bus: EventBus):
    """Set up and connect to the appropriate broker."""
    if mode == 'live':
        # Live trading with real broker
        if broker_name == 'tradier':
            broker = TradierAdapter(event_bus)
            broker.connect(config.get('brokers', {}).get('tradier', {}))
        elif broker_name == 'alpaca':
            broker = AlpacaAdapter(event_bus)
            broker.connect(config.get('brokers', {}).get('alpaca', {}))
        elif broker_name == 'etrade':
            # TODO: Implement E*Trade adapter
            logger.error("E*Trade adapter not implemented yet")
            sys.exit(1)
        else:
            logger.error(f"Unknown broker: {broker_name}")
            sys.exit(1)
    else:
        # Paper trading or backtesting
        paper_config = PaperTradeConfig.from_dict(config.get('paper_trading', {}))
        
        # For backtesting, configure the paper trading mode
        if mode == 'backtest':
            paper_config.simulation_mode = 'backtest'
        
        broker = PaperTradeAdapter(event_bus)
        broker.connect(paper_config)
    
    return broker


def load_historical_data(data_path: str, symbols: list, start_date: datetime, end_date: datetime):
    """Load historical data for backtesting."""
    import pandas as pd
    
    if not os.path.exists(data_path):
        logger.error(f"Historical data path not found: {data_path}")
        sys.exit(1)
    
    data = {}
    for symbol in symbols:
        try:
            # Try to load CSV data
            file_path = os.path.join(data_path, f"{symbol}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                
                # Filter by date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                data[symbol] = df[mask]
                
                logger.info(f"Loaded {len(data[symbol])} bars for {symbol}")
            else:
                logger.warning(f"No data file found for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    return data


def run_backtest(trading_engine, broker, symbols, start_date, end_date, interval='1d'):
    """Run backtest."""
    # Assuming the first symbol has the main timeline
    if not symbols or not broker.historical_data:
        logger.error("No symbols or historical data available for backtesting")
        return
    
    main_symbol = symbols[0]
    if main_symbol not in broker.historical_data:
        main_symbol = list(broker.historical_data.keys())[0]
    
    data = broker.historical_data[main_symbol]
    
    # Prepare for backtest
    trading_engine.initialize_backtest(start_date, end_date)
    
    # Iterate through each time point
    for timestamp in data.index:
        if timestamp < start_date or timestamp > end_date:
            continue
        
        # Set the current time for simulation
        broker.set_backtest_time(timestamp)
        
        # Process this time step
        trading_engine.process_time_update(timestamp)
    
    # Finalize backtest and generate report
    trading_engine.finalize_backtest()


def initialize_persistence(config: Dict[str, Any], event_bus: EventBus):
    """Initialize the persistence layer and connection managers."""
    logger.info("Initializing persistence layer...")
    
    # Get database configuration
    db_config = config.get('persistence', {})
    mongodb_uri = db_config.get('mongodb_uri', os.environ.get('MONGODB_URI', 'mongodb://localhost:27017'))
    redis_uri = db_config.get('redis_uri', os.environ.get('REDIS_URI', 'redis://localhost:6379/0'))
    db_name = db_config.get('database_name', 'bensbot')
    
    # Set up connection manager
    connection_manager = ConnectionManager(
        mongodb_uri=mongodb_uri,
        redis_uri=redis_uri,
        database_name=db_name
    )
    
    # Initialize repositories
    order_repo = OrderRepository(connection_manager)
    position_repo = PositionRepository(connection_manager)
    fill_repo = FillRepository(connection_manager)
    pnl_repo = PnLRepository(connection_manager)
    idempotency_manager = IdempotencyManager(connection_manager)
    
    # Initialize recovery manager
    recovery_manager = RecoveryManager(
        connection_manager=connection_manager,
        event_bus=event_bus,
        order_repo=order_repo,
        position_repo=position_repo,
        fill_repo=fill_repo,
        pnl_repo=pnl_repo,
        idempotency_manager=idempotency_manager
    )
    
    return {
        'connection_manager': connection_manager,
        'recovery_manager': recovery_manager,
        'order_repo': order_repo,
        'position_repo': position_repo,
        'fill_repo': fill_repo,
        'pnl_repo': pnl_repo,
        'idempotency_manager': idempotency_manager
    }


def perform_recovery(recovery_manager: RecoveryManager, broker_manager: MultiBrokerManager):
    """Perform crash recovery and state reconciliation."""
    logger.info("Starting crash recovery process...")
    
    # Recover full state from persistent storage
    recovery_summary = recovery_manager.recover_full_state()
    
    # Log recovery summary
    logger.info(f"Recovery complete: {recovery_summary}")
    
    # Cross-check with broker if not in backtest mode
    # This will be implemented if needed by the broker adapters
    if hasattr(broker_manager, 'reconcile_positions'):
        logger.info("Reconciling positions with broker(s)...")
        broker_manager.reconcile_positions()
    
    # Recover idempotency state to prevent duplicate orders
    pending_ops = recovery_manager.recover_idempotency_state()
    logger.info(f"Recovered {pending_ops} pending idempotent operations")
    
    return recovery_summary


def main():
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description='BensBot Trading System')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], default='paper',
                      help='Trading mode')
    parser.add_argument('--broker', choices=['tradier', 'alpaca', 'etrade'], default='tradier',
                      help='Broker to use for live trading')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--backtest-data', type=str,
                      help='Path to historical data directory for backtesting')
    parser.add_argument('--start-date', type=str,
                      help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                      help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str,
                      help='Comma-separated list of symbols to trade')
    parser.add_argument('--no-recovery', action='store_true',
                      help='Skip state recovery (use with caution)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse symbols
    symbols = []
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        # Get symbols from config
        symbols = config.get('symbols', [])
    
    if not symbols:
        logger.error("No symbols specified")
        sys.exit(1)
    
    # Set up event system
    event_bus = EventBus()
    
    # Initialize persistence layer (except for backtest mode)
    persistence = None
    if args.mode != 'backtest':
        persistence = initialize_persistence(config, event_bus)
    
    # Setup broker
    broker = setup_broker(args.mode, args.broker, config, event_bus)
    
    # Create and configure multi-broker manager
    broker_manager = MultiBrokerManager(event_bus=event_bus)
    broker_id = args.broker if args.mode == 'live' else 'paper'
    
    # Set the idempotency manager for the broker_manager if available
    if persistence:
        broker_manager.idempotency_manager = persistence['idempotency_manager']
    
    # Add broker to manager
    broker_manager.add_broker(broker_id, broker, None, make_primary=True)
    
    # Get start/end dates for backtest
    start_date = None
    end_date = None
    if args.mode == 'backtest':
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=30)
            
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        
        # Load historical data
        if args.backtest_data:
            historical_data = load_historical_data(args.backtest_data, symbols, start_date, end_date)
            if broker and hasattr(broker, 'load_historical_data'):
                broker.load_historical_data(historical_data)
    
    # Perform crash recovery (except for backtest mode or if explicitly disabled)
    if args.mode != 'backtest' and not args.no_recovery and persistence:
        recovery_summary = perform_recovery(persistence['recovery_manager'], broker_manager)
        
        # Store recovery information in broker manager's metadata
        broker_manager.metadata['recovery'] = recovery_summary
    
    # Set up the trading engine with repositories if available
    trading_engine_kwargs = {
        'broker_manager': broker_manager,
        'config': config,
        'event_bus': event_bus,
        'mode': args.mode
    }
    
    # Add repositories if available
    if persistence:
        trading_engine_kwargs.update({
            'order_repository': persistence['order_repo'],
            'position_repository': persistence['position_repo'],
            'fill_repository': persistence['fill_repo'],
            'pnl_repository': persistence['pnl_repo']
        })
    
    trading_engine = MainOrchestrator(**trading_engine_kwargs)
    
    # Run the trading engine based on mode
    if args.mode == 'backtest':
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        run_backtest(trading_engine, broker, symbols, start_date, end_date)
    else:
        logger.info(f"Starting {args.mode} trading with {broker_id}")
        
        # Set up periodic state persistence (sync every 5 minutes)
        def sync_state_periodically():
            while True:
                try:
                    time.sleep(300)  # 5 minutes
                    if persistence and hasattr(persistence['position_repo'], 'sync_to_durable_storage'):
                        logger.info("Syncing state to durable storage...")
                        persistence['position_repo'].sync_to_durable_storage()
                except Exception as e:
                    logger.error(f"Error during periodic state sync: {e}")
        
        # Start the trading engine
        trading_engine.start()
        
        # Start periodic state sync in a separate thread
        if args.mode != 'backtest' and persistence:
            import threading
            sync_thread = threading.Thread(target=sync_state_periodically, daemon=True)
            sync_thread.start()
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping trading engine...")
            
            # Perform final state sync before stopping
            if args.mode != 'backtest' and persistence:
                try:
                    logger.info("Performing final state sync...")
                    persistence['position_repo'].sync_to_durable_storage()
                except Exception as e:
                    logger.error(f"Error during final state sync: {e}")
            
            trading_engine.stop()


if __name__ == '__main__':
    main()
