#!/usr/bin/env python3
"""
Integration test for the Enhanced Strategy Manager with existing broker connections.
This script tests:
1. Loading strategy configurations
2. Creating strategy ensembles
3. Connecting to brokers
4. Simulating market data events
5. Validating signal generation and processing

Run this in paper trading/sandbox mode only!
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy_manager_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.brokers.broker_factory import load_from_config_file
from trading_bot.core.enhanced_strategy_manager_impl import EnhancedStrategyManager
from trading_bot.core.event_bus import Event, get_global_event_bus
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_manager import StrategyPerformanceManager
from trading_bot.models.quote import Quote


def load_strategy_configs(config_dir):
    """Load all strategy configuration files from the specified directory."""
    strategy_configs = []
    
    # Load each strategy config file
    for filename in os.listdir(config_dir):
        if filename.endswith('_strategies.json'):
            filepath = os.path.join(config_dir, filename)
            logger.info(f"Loading strategy config from {filepath}")
            
            try:
                with open(filepath, 'r') as f:
                    config_data = json.load(f)
                    if 'strategies' in config_data:
                        strategy_configs.extend(config_data['strategies'])
            except Exception as e:
                logger.error(f"Error loading {filepath}: {str(e)}")
    
    logger.info(f"Loaded {len(strategy_configs)} strategy configurations")
    return strategy_configs


def load_ensemble_configs(config_path):
    """Load ensemble configurations from file."""
    ensemble_configs = []
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            if 'ensembles' in config_data:
                ensemble_configs = config_data['ensembles']
    except Exception as e:
        logger.error(f"Error loading ensembles from {config_path}: {str(e)}")
    
    logger.info(f"Loaded {len(ensemble_configs)} ensemble configurations")
    return ensemble_configs


def simulate_market_data(event_bus, symbols, duration_seconds=30):
    """
    Simulate market data events for testing.
    """
    logger.info(f"Starting market data simulation for {len(symbols)} symbols")
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < duration_seconds:
        # Generate mock data for each symbol
        for symbol in symbols:
            # Create mock quote data
            mock_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "last": 100.0 + (hash(symbol + str(time.time())) % 10) / 10.0,
                "bid": 99.8 + (hash(symbol + str(time.time())) % 10) / 10.0,
                "ask": 100.2 + (hash(symbol + str(time.time())) % 10) / 10.0,
                "volume": 1000 + (hash(symbol + str(time.time())) % 1000)
            }
            
            # Publish quote update event
            event_bus.publish(Event(
                event_type=EventType.QUOTE_UPDATE,
                data=mock_data,
                source="market_data_simulator"
            ))
            
            # Also publish as market data for more general handlers
            event_bus.publish(Event(
                event_type=EventType.MARKET_DATA_UPDATE,
                data=mock_data,
                source="market_data_simulator"
            ))
        
        # Small delay between updates
        time.sleep(0.5)
    
    logger.info("Market data simulation completed")


def main():
    """Main test function"""
    try:
        logger.info("Starting Enhanced Strategy Manager integration test")
        
        # Load broker manager
        broker_config_path = os.path.join('config', 'multi_broker_config.json')
        logger.info(f"Loading broker manager from {broker_config_path}")
        broker_manager = load_from_config_file(broker_config_path)
        
        # Try connecting to all brokers
        logger.info("Connecting to brokers...")
        connection_results = broker_manager.connect_all()
        logger.info(f"Broker connections: {connection_results}")
        
        # Create performance manager
        performance_manager = StrategyPerformanceManager()
        
        # Create Enhanced Strategy Manager
        strategy_manager = EnhancedStrategyManager(
            broker_manager=broker_manager,
            performance_manager=performance_manager,
            config={
                "risk_limits": {
                    "max_position_per_symbol": 0.05,
                    "max_allocation_per_strategy": 0.20,
                    "max_allocation_per_asset_type": 0.50,
                    "max_total_allocation": 0.80,
                    "max_drawdown": 0.10
                }
            }
        )
        
        # Load strategy configurations
        strategies_dir = os.path.join('config', 'strategies')
        strategy_configs = load_strategy_configs(strategies_dir)
        
        # Load ensemble configurations
        ensemble_config_path = os.path.join('config', 'strategies', 'strategy_ensembles.json')
        ensemble_configs = load_ensemble_configs(ensemble_config_path)
        
        # Initialize strategies and ensembles
        strategy_manager.load_strategies(strategy_configs)
        strategy_manager.create_ensembles(ensemble_configs)
        
        # Start the strategy manager
        logger.info("Starting strategy manager...")
        strategy_manager.start_strategies()
        
        # Collect symbols from all strategies for simulation
        all_symbols = set()
        for strategy_id, strategy in strategy_manager.strategies.items():
            if strategy.symbols:
                all_symbols.update(strategy.symbols)
        
        # Run simulation with market data
        logger.info(f"Simulating market data for symbols: {all_symbols}")
        simulate_market_data(get_global_event_bus(), list(all_symbols), duration_seconds=60)
        
        # Get active strategies
        active_strategies = strategy_manager.get_active_strategies()
        logger.info(f"Active strategies: {len(active_strategies)}")
        for strat in active_strategies:
            logger.info(f"  - {strat['name']} ({strat['strategy_id']}): {strat['state']}")
        
        # Check if any signals were generated
        logger.info(f"Total signals generated: {len(strategy_manager.signal_history)}")
        for i, signal in enumerate(strategy_manager.signal_history[-10:]):
            logger.info(f"Signal {i+1}: {signal.symbol} - {signal.action} - {signal.strength}")
        
        # Stop the strategy manager
        logger.info("Stopping strategy manager...")
        strategy_manager.stop_strategies()
        
        logger.info("Enhanced Strategy Manager integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
