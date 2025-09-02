#!/usr/bin/env python3
"""
Migration Examples

This script demonstrates the differences between the old and new ways of using BensBot.
It contains example code for both approaches to help users transition to the new system.

These examples cover:
1. Starting the trading bot
2. Running a backtest
3. Initializing a strategy
4. Configuring brokers

Run with:
    python examples/migration_examples.py
"""

import os
import sys
import yaml
import json
from datetime import datetime
from pprint import pprint


def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def show_old_vs_new_bot_initialization():
    """Show how bot initialization has changed."""
    print_section("TRADING BOT INITIALIZATION")
    
    print("OLD APPROACH - Multiple entry points and config files:")
    print("```python")
    print("""
# Old approach - Starting live trading
from trading_bot.main import TradingBot
from trading_bot.config import load_config

# Load separate config files
trading_config = load_config('config/trading_config.json')
broker_config = load_config('config/broker_config.json')
strategy_config = load_config('config/strategy_config.json')

# Initialize the bot with multiple configs
bot = TradingBot(
    trading_config=trading_config,
    broker_config=broker_config,
    strategy_config=strategy_config
)

# Start trading
bot.start()
    """)
    print("```\n")
    
    print("NEW APPROACH - Unified entry point:")
    print("```python")
    print("""
# New approach - Everything through run_bot.py
# Command line:
# python run_bot.py --config config.yaml --mode live

# If you need to access from code:
from trading_bot.cli.run_bot import create_bot

# Create and run bot with unified config
bot = create_bot(
    config_path='config.yaml',
    mode='live'
)

# Run the bot (this is normally handled by run_bot.py)
bot.run()
    """)
    print("```\n")


def show_old_vs_new_backtest_approach():
    """Show how backtesting has changed."""
    print_section("BACKTESTING")
    
    print("OLD APPROACH - Separate backtest script:")
    print("```python")
    print("""
# Old approach - Separate backtest module
from trading_bot.backtest import Backtester
from trading_bot.strategies import MomentumStrategy
from trading_bot.config import load_config

# Load backtest config
backtest_config = load_config('config/backtest_config.json')

# Create strategy instance manually
strategy = MomentumStrategy(
    lookback_period=20,
    volatility_factor=1.5,
    symbols=['AAPL', 'MSFT']
)

# Initialize backtester
backtester = Backtester(
    strategy=strategy,
    initial_capital=backtest_config['initial_capital'],
    start_date=backtest_config['start_date'],
    end_date=backtest_config['end_date'],
    data_file=backtest_config['data_file']
)

# Run backtest
results = backtester.run()
backtester.plot_results()
    """)
    print("```\n")
    
    print("NEW APPROACH - Unified entry point with mode flag:")
    print("```python")
    print("""
# New approach - Everything through run_bot.py
# Command line:
# python run_bot.py --config config.yaml --mode backtest

# If you need to access from code:
from trading_bot.cli.run_bot import create_bot

# Create and run bot in backtest mode
backtest_bot = create_bot(
    config_path='config.yaml',
    mode='backtest'
)

# Run the backtest (this is normally handled by run_bot.py)
results = backtest_bot.run()

# Access backtest results if needed
print(f"Final portfolio value: ${results['final_equity']:.2f}")
print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.2f}")
    """)
    print("```\n")


def show_old_vs_new_strategy_initialization():
    """Show how strategy initialization has changed."""
    print_section("STRATEGY INITIALIZATION")
    
    print("OLD APPROACH - Direct instantiation:")
    print("```python")
    print("""
# Old approach - Direct instantiation with parameters
from trading_bot.strategies import MomentumStrategy

# Create strategy with parameters
strategy = MomentumStrategy(
    lookback_period=20,
    volatility_factor=1.5,
    symbols=['AAPL', 'MSFT'],
    timeframe='1h',
    take_profit_pct=5.0,
    stop_loss_pct=2.0,
    # ... more parameters
)
    """)
    print("```\n")
    
    print("NEW APPROACH - Factory pattern with config:")
    print("```python")
    print("""
# New approach - Factory pattern with configuration
from trading_bot.strategies import strategy_factory

# Strategy config from YAML
strategy_config = {
    'name': 'momentum_breakout',
    'class': 'trading_bot.strategies.momentum.BreakoutStrategy',
    'parameters': {
        'lookback_period': 20,
        'volatility_factor': 1.5,
        'profit_target_pct': 5.0,
        'stop_loss_pct': 2.0,
    }
}

# Create strategy from config
strategy = strategy_factory.create_strategy(
    strategy_config,
    symbols=['AAPL', 'MSFT'],  # Override or supplement config values
    timeframe='1h'
)
    """)
    print("```\n")


def show_old_vs_new_broker_configuration():
    """Show how broker configuration has changed."""
    print_section("BROKER CONFIGURATION")
    
    print("OLD APPROACH - Direct instantiation:")
    print("```python")
    print("""
# Old approach - Direct instantiation of specific broker
from trading_bot.brokers.tradier import TradierBroker

# Create broker with credentials
broker = TradierBroker(
    token='your-api-token',
    account_id='your-account-id',
    sandbox=True
)

# Connect to broker
broker.connect()

# Get account information
account_info = broker.get_account_info()
print(account_info)
    """)
    print("```\n")
    
    print("NEW APPROACH - Multi-broker manager with abstraction:")
    print("```python")
    print("""
# New approach - MultiBrokerManager with adapter pattern
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.brokers.alpaca.adapter import AlpacaAdapter
from trading_bot.brokers.tradier.adapter import TradierAdapter
from trading_bot.core.event_bus import EventBus

# Create event bus
event_bus = EventBus()

# Create broker manager
broker_manager = MultiBrokerManager(event_bus=event_bus)

# Add brokers from config
broker_manager.add_broker(
    broker_id='alpaca',
    broker=AlpacaAdapter(event_bus),
    credentials={
        'api_key': 'your-alpaca-key',
        'api_secret': 'your-alpaca-secret'
    },
    make_primary=True
)

broker_manager.add_broker(
    broker_id='tradier',
    broker=TradierAdapter(event_bus),
    credentials={
        'token': 'your-tradier-token',
        'account_id': 'your-tradier-account'
    }
)

# Get aggregated account information
all_accounts = broker_manager.get_all_account_info()
print(all_accounts)

# Place an order through the primary broker
order = {
    'symbol': 'AAPL',
    'side': 'buy',
    'quantity': 10,
    'order_type': 'market',
    'time_in_force': 'day'
}
broker_manager.place_order(order)
    """)
    print("```\n")


def show_configuration_examples():
    """Show examples of old vs new configuration formats."""
    print_section("CONFIGURATION FORMAT")
    
    # Old-style config examples
    old_broker_config = {
        "tradier_token": "your-api-token",
        "tradier_account_id": "your-account-id",
        "alpaca_key_id": "your-alpaca-key",
        "alpaca_secret_key": "your-alpaca-secret",
        "broker_api_url": "https://api.tradier.com",
        "paper_trading": True
    }
    
    old_strategy_config = {
        "strategy_name": "MomentumBreakout",
        "strategy_class": "trading_bot.strategies.MomentumStrategy",
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "timeframe": "1h",
        "lookback_period": 20,
        "volatility_factor": 1.5,
        "take_profit_pct": 5.0,
        "stop_loss_pct": 2.0
    }
    
    # New-style unified config
    new_unified_config = {
        "metadata": {
            "version": "1.0.0",
            "description": "BensBot unified configuration"
        },
        "account": {
            "initial_balance": 10000.0,
            "paper_trading": True
        },
        "brokers": {
            "enabled": ["alpaca", "tradier"],
            "primary": "alpaca",
            "failover": True,
            "credentials": {
                "alpaca": {
                    "api_key": "${ALPACA_API_KEY}",
                    "api_secret": "${ALPACA_SECRET_KEY}"
                },
                "tradier": {
                    "token": "${TRADIER_TOKEN}",
                    "account_id": "${TRADIER_ACCOUNT_ID}"
                }
            },
            "settings": {
                "alpaca": {
                    "paper": True,
                    "api_url": "https://paper-api.alpaca.markets"
                },
                "tradier": {
                    "sandbox": True
                }
            }
        },
        "strategy": {
            "name": "momentum_breakout",
            "class": "trading_bot.strategies.momentum.BreakoutStrategy",
            "parameters": {
                "lookback_period": 20,
                "volatility_factor": 1.5,
                "profit_target_pct": 5.0,
                "stop_loss_pct": 2.0
            }
        },
        "data": {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "timeframe": "1h",
            "source": "alpaca"
        },
        "risk": {
            "max_risk_per_trade": 1.0,
            "max_concurrent_trades": 5,
            "max_drawdown_pct": 15.0,
            "use_trailing_stop": True,
            "trailing_stop_percent": 1.5
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/trading.log",
            "console": True
        }
    }
    
    # Print examples
    print("OLD APPROACH - Multiple JSON configuration files:\n")
    
    print("broker_config.json:")
    print(json.dumps(old_broker_config, indent=2))
    print()
    
    print("strategy_config.json:")
    print(json.dumps(old_strategy_config, indent=2))
    print("\n")
    
    print("NEW APPROACH - Single unified YAML configuration:\n")
    print("config.yaml:")
    print(yaml.dump(new_unified_config, sort_keys=False))


def main():
    """Run all examples."""
    print("\nBENSBOT MIGRATION EXAMPLES\n")
    print("This script demonstrates the differences between old and new usage patterns.\n")
    
    # Show examples
    show_old_vs_new_bot_initialization()
    show_old_vs_new_backtest_approach()
    show_old_vs_new_strategy_initialization()
    show_old_vs_new_broker_configuration()
    show_configuration_examples()
    
    print_section("CONCLUSION")
    print("The new BensBot architecture provides several advantages:")
    print("✓ Unified entry point with 'run_bot.py'")
    print("✓ Clean separation of concerns with adapter patterns and dependency injection")
    print("✓ Structured, hierarchical configuration")
    print("✓ Support for multiple brokers with failover")
    print("✓ Production-ready features like persistence, recovery, and containers")
    print("✓ Standardized logging and comprehensive testing\n")
    print("For more details, refer to the documentation:")
    print("- docs/CONFIG_MIGRATION.md: Detailed configuration migration guide")
    print("- docs/MIGRATION_GUIDE.md: Overall system migration guide")
    print("- README.md: Getting started with the new system\n")


if __name__ == "__main__":
    main()
