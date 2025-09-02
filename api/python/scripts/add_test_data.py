#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Add Test Data Script

This script adds test data to our portfolio state manager.
It creates simulated trades for a few popular stocks.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import random
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import our components
from trading_bot.portfolio_state import PortfolioStateManager
from trading_bot.data.market_data_provider import create_data_provider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_DIR = os.path.join(current_dir.parent, "config")
DATA_PROVIDER_CONFIG = os.path.join(CONFIG_DIR, "data_providers.json")
PORTFOLIO_STATE_FILE = os.path.join(CONFIG_DIR, "portfolio_state.json")

def add_test_data():
    """Add test data to the portfolio state manager"""
    # Initialize portfolio state manager
    portfolio = PortfolioStateManager(initial_cash=100000.0, state_file=PORTFOLIO_STATE_FILE)
    
    # Try to initialize a real data provider or use sample data
    try:
        # Check if we have API credentials configured
        data_provider = create_data_provider("yahoo")
        use_real_data = True
        logger.info("Using Yahoo Finance for test data")
    except Exception as e:
        logger.warning(f"Could not initialize data provider: {e}")
        logger.warning("Using sample price data instead")
        use_real_data = False
    
    # Symbols to add to the portfolio
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "GLD"]
    
    # Sample prices in case real data is not available
    sample_prices = {
        "AAPL": 185.92,
        "MSFT": 417.88,
        "GOOGL": 175.53,
        "AMZN": 178.75,
        "TSLA": 175.68,
        "NVDA": 1023.55,
        "SPY": 520.94,
        "QQQ": 445.94,
        "GLD": 233.96
    }
    
    # Get current prices for these symbols
    if use_real_data:
        current_prices = data_provider.get_current_price(symbols)
        logger.info(f"Got real prices for {len(current_prices)} symbols")
    else:
        current_prices = sample_prices
        logger.info("Using sample prices")
    
    # Generate some historical trades (going back a few months)
    start_date = datetime.now() - timedelta(days=90)
    
    # Create buy trades for each symbol at a random time in the past
    for symbol in symbols:
        # Skip if price not available
        if symbol not in current_prices:
            logger.warning(f"No price available for {symbol}, skipping")
            continue
        
        # Calculate a random date in the past 90 days
        trade_date = start_date + timedelta(days=random.randint(0, 90))
        
        # Calculate a reasonable historical price (slight variation from current)
        price_factor = 1.0 + (random.random() * 0.2 - 0.1)  # -10% to +10%
        historical_price = current_prices[symbol] / price_factor
        
        # Calculate a quantity based on a target allocation
        target_allocation = random.uniform(0.05, 0.15)  # 5% to 15% of portfolio
        target_value = 100000 * target_allocation
        quantity = int(target_value / historical_price)
        
        if quantity > 0:
            # Add the trade to the portfolio
            portfolio.update_position(symbol, quantity, historical_price, trade_date)
            logger.info(f"Added position: {quantity} shares of {symbol} at ${historical_price:.2f}")
    
    # Update with current prices
    portfolio.update_prices(current_prices)
    
    # Set some strategy allocations
    strategy_allocations = {
        "Momentum": 35.0,
        "MeanReversion": 25.0,
        "Trend": 20.0,
        "VolatilityBreakout": 15.0,
        "MacroRegime": 5.0
    }
    portfolio.update_strategy_allocations(strategy_allocations)
    
    # Set some strategy performance metrics
    strategy_performance = {
        "Momentum": {
            "return": 12.5,
            "sharpe": 1.4,
            "drawdown": -8.3,
            "win_rate": 62.0
        },
        "MeanReversion": {
            "return": 8.2,
            "sharpe": 1.1,
            "drawdown": -5.7,
            "win_rate": 58.0
        },
        "Trend": {
            "return": 15.8,
            "sharpe": 1.7,
            "drawdown": -12.4,
            "win_rate": 55.0
        },
        "VolatilityBreakout": {
            "return": 10.3,
            "sharpe": 1.2,
            "drawdown": -9.6,
            "win_rate": 51.0
        },
        "MacroRegime": {
            "return": 7.1,
            "sharpe": 1.0,
            "drawdown": -6.2,
            "win_rate": 53.0
        }
    }
    portfolio.update_strategy_performance(strategy_performance)
    
    # Save the portfolio state
    portfolio.save_state(PORTFOLIO_STATE_FILE)
    
    # Print the final portfolio summary
    state = portfolio.get_full_state()
    portfolio_value = state["portfolio"]["total_value"]
    cash = state["portfolio"]["cash"]
    
    logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
    logger.info(f"Cash: ${cash:.2f}")
    logger.info(f"Invested: ${portfolio_value - cash:.2f}")
    
    num_positions = len(state["portfolio"]["positions"])
    logger.info(f"Number of positions: {num_positions}")
    
    # Print the performance metrics
    metrics = state["performance_metrics"]
    logger.info(f"Performance Metrics:")
    logger.info(f"  Return: {metrics['cumulative_return']:.2f}%")
    logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  Volatility: {metrics['volatility']:.2f}%")
    
    logger.info(f"Test data successfully added to {PORTFOLIO_STATE_FILE}")

if __name__ == "__main__":
    # Make sure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Add test data
    add_test_data() 