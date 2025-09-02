#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortfolioStateManager Demonstration

This script demonstrates the key features and functionality of the PortfolioStateManager class
from the trading_bot package. It showcases:

1. Creating and initializing a portfolio state
2. Saving and loading state from disk
3. Updating various portfolio components
4. Retrieving state summaries and specific data
5. Working with the event system

Run this script to see a complete demonstration of the PortfolioStateManager's capabilities
with detailed logging of what's happening at each step.
"""

import os
import sys
import json
import logging
import shutil
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_bot.portfolio_state import PortfolioStateManager

# Set up logging with formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('PortfolioStateDemo')

# Create temp directory for state files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
STATE_FILE_PATH = os.path.join(TEMP_DIR, "demo_portfolio_state.json")

def demo_create_portfolio():
    """
    Demonstrates creating and initializing a portfolio state.
    """
    logger.info("=== Demonstrating Portfolio Creation ===")
    
    # Create a new portfolio state
    portfolio = PortfolioStateManager(state_file_path=STATE_FILE_PATH)
    
    # Initialize with sample data
    portfolio.update_portfolio_data(
        cash=50000.0,
        total_value=100000.0,
        positions={
            "AAPL": {
                "quantity": 100,
                "avg_price": 150.0,
                "current_price": 170.0,
                "current_value": 17000.0,
                "unrealized_pnl": 2000.0,
                "unrealized_pnl_pct": 13.33
            },
            "MSFT": {
                "quantity": 50,
                "avg_price": 250.0,
                "current_price": 280.0,
                "current_value": 14000.0,
                "unrealized_pnl": 1500.0,
                "unrealized_pnl_pct": 12.0
            },
            "GOOGL": {
                "quantity": 20,
                "avg_price": 2500.0,
                "current_price": 2700.0,
                "current_value": 54000.0,
                "unrealized_pnl": 4000.0,
                "unrealized_pnl_pct": 8.0
            }
        },
        asset_allocation={
            "Technology": 85.0,
            "Cash": 15.0
        }
    )
    
    # Update performance metrics
    portfolio.update_performance_metrics({
        "cumulative_return": 18.5,
        "sharpe_ratio": 1.8,
        "max_drawdown": -8.2,
        "volatility": 12.5,
        "win_rate": 0.68,
        "profit_factor": 2.1,
        "recent_daily_returns": [0.8, -0.3, 1.2, 0.5, -0.1]
    })
    
    # Add sample trades
    for i, (symbol, action, price, quantity) in enumerate([
        ("AAPL", "BUY", 145.0, 50),
        ("MSFT", "BUY", 245.0, 30),
        ("AAPL", "BUY", 155.0, 50),
        ("GOOGL", "BUY", 2480.0, 10),
        ("MSFT", "BUY", 258.0, 20),
        ("GOOGL", "BUY", 2520.0, 10)
    ]):
        trade_time = datetime.now().isoformat()
        portfolio.add_trade({
            "id": f"trade-{i+1}",
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "timestamp": trade_time,
            "status": "FILLED",
            "commission": 1.0
        })
    
    # Add sample signals
    for i, (symbol, signal_type, confidence) in enumerate([
        ("AAPL", "BUY", 0.85),
        ("MSFT", "BUY", 0.75),
        ("GOOGL", "BUY", 0.82),
        ("SPY", "NEUTRAL", 0.60),
        ("QQQ", "BULLISH", 0.78)
    ]):
        signal_time = datetime.now().isoformat()
        portfolio.add_signal({
            "id": f"signal-{i+1}",
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "timestamp": signal_time,
            "strategy": "MomentumStrategy",
            "expiration": None
        })
    
    # Update strategy data
    portfolio.update_strategy_data(
        active_strategies=["MomentumStrategy", "MeanReversionStrategy", "TrendFollowingStrategy"],
        strategy_allocations={
            "MomentumStrategy": 40,
            "MeanReversionStrategy": 30,
            "TrendFollowingStrategy": 30
        },
        strategy_performance={
            "MomentumStrategy": {
                "return": 22.5,
                "sharpe": 1.9,
                "drawdown": -7.8
            },
            "MeanReversionStrategy": {
                "return": 15.2,
                "sharpe": 1.6,
                "drawdown": -9.3
            },
            "TrendFollowingStrategy": {
                "return": 18.7,
                "sharpe": 1.7,
                "drawdown": -8.5
            }
        }
    )
    
    # Update system status
    portfolio.update_system_status(
        is_market_open=True,
        market_hours="9:30 AM - 4:00 PM EST",
        data_providers=["Alpha Vantage", "IEX Cloud", "Yahoo Finance"],
        connected_brokers=["Alpaca", "Interactive Brokers"],
        system_health={
            "cpu_usage": 35.2,
            "memory_usage": 42.8,
            "api_rate_limit_remaining": 950,
            "last_health_check": datetime.now().isoformat(),
            "status": "HEALTHY"
        }
    )
    
    # Update learning status
    portfolio.update_learning_status(
        training_in_progress=False,
        models_status={
            "price_predictor": {
                "last_trained": (datetime.now().isoformat()),
                "accuracy": 0.68,
                "status": "active"
            },
            "volatility_model": {
                "last_trained": (datetime.now().isoformat()),
                "accuracy": 0.72,
                "status": "active"
            },
            "sentiment_analyzer": {
                "last_trained": (datetime.now().isoformat()),
                "accuracy": 0.65,
                "status": "active"
            }
        },
        recent_learning_metrics={
            "training_duration": 1200,
            "samples_processed": 50000,
            "validation_accuracy": 0.71
        }
    )
    
    # Save the state
    portfolio.save_state()
    logger.info(f"Portfolio state saved to {STATE_FILE_PATH}")
    
    return portfolio

def demo_save_load_state(portfolio):
    """
    Demonstrates saving the current state and loading into a new instance.
    """
    logger.info("\n=== Demonstrating Save and Load Functionality ===")
    
    # Save current state
    save_result = portfolio.save_state()
    logger.info(f"State saved successfully: {save_result}")
    
    # Create a new instance and load the state
    new_portfolio = PortfolioStateManager(state_file_path=STATE_FILE_PATH)
    load_result = new_portfolio.load_state()
    logger.info(f"State loaded successfully: {load_result}")
    
    # Verify the loaded state matches
    original_summary = portfolio.get_portfolio_summary()
    loaded_summary = new_portfolio.get_portfolio_summary()
    
    logger.info("Original portfolio summary:")
    logger.info(json.dumps(original_summary, indent=2))
    
    logger.info("Loaded portfolio summary:")
    logger.info(json.dumps(loaded_summary, indent=2))
    
    # Verify state equality
    is_equal = original_summary == loaded_summary
    logger.info(f"States are equal: {is_equal}")
    
    return new_portfolio

def demo_update_portfolio(portfolio):
    """
    Demonstrates updating various aspects of the portfolio state.
    """
    logger.info("\n=== Demonstrating Portfolio Updates ===")
    
    # Update portfolio data
    portfolio.update_portfolio_data(
        cash=55000.0,  # Increased cash
        total_value=110000.0  # Increased total value
    )
    logger.info("Updated cash and total value")
    
    # Add new position
    positions = portfolio.get_positions()
    positions["AMZN"] = {
        "quantity": 10,
        "avg_price": 3200.0,
        "current_price": 3300.0,
        "current_value": 33000.0,
        "unrealized_pnl": 1000.0,
        "unrealized_pnl_pct": 3.1
    }
    portfolio.update_portfolio_data(positions=positions)
    logger.info("Added new position: AMZN")
    
    # Update asset allocation
    portfolio.update_portfolio_data(
        asset_allocation={
            "Technology": 80.0,
            "E-commerce": 10.0,
            "Cash": 10.0
        }
    )
    logger.info("Updated asset allocation to include E-commerce category")
    
    # Add a new trade
    portfolio.add_trade({
        "id": "trade-7",
        "symbol": "AMZN",
        "action": "BUY",
        "price": 3200.0,
        "quantity": 10,
        "timestamp": datetime.now().isoformat(),
        "status": "FILLED",
        "commission": 1.0
    })
    logger.info("Added new trade for AMZN")
    
    # Add a new signal
    portfolio.add_signal({
        "id": "signal-6",
        "symbol": "AMZN",
        "signal_type": "BUY",
        "confidence": 0.88,
        "timestamp": datetime.now().isoformat(),
        "strategy": "TrendFollowingStrategy",
        "expiration": None
    })
    logger.info("Added new signal for AMZN")
    
    # Update system status
    portfolio.update_system_status(
        system_health={
            "cpu_usage": 40.5,
            "memory_usage": 45.3,
            "api_rate_limit_remaining": 920,
            "last_health_check": datetime.now().isoformat(),
            "status": "HEALTHY"
        }
    )
    logger.info("Updated system health metrics")
    
    # Save updated state
    portfolio.save_state()
    logger.info("Updated portfolio state saved")
    
    return portfolio

def demo_retrieving_data(portfolio):
    """
    Demonstrates retrieving various data from the portfolio state.
    """
    logger.info("\n=== Demonstrating Data Retrieval ===")
    
    # Get full state
    full_state = portfolio.get_full_state()
    logger.info(f"Retrieved full state with {len(full_state)} top-level keys")
    
    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    logger.info("Portfolio summary:")
    logger.info(json.dumps(summary, indent=2))
    
    # Get specific position
    aapl_position = portfolio.get_position("AAPL")
    logger.info("AAPL position details:")
    logger.info(json.dumps(aapl_position, indent=2))
    
    # Get all positions
    positions = portfolio.get_positions()
    logger.info(f"Retrieved {len(positions)} positions")
    
    # Get performance metrics
    metrics = portfolio.get_performance_metrics()
    logger.info("Performance metrics:")
    logger.info(json.dumps(metrics, indent=2))
    
    # Get recent activity
    activity = portfolio.get_recent_activity(limit=3)
    logger.info("Recent activity (limited to 3 items):")
    logger.info(json.dumps(activity, indent=2))
    
    # Get strategy data
    strategy_data = portfolio.get_strategy_data()
    logger.info(f"Retrieved data for {len(strategy_data['active_strategies'])} active strategies")
    
    # Get system status
    system_status = portfolio.get_system_status()
    logger.info("System status:")
    logger.info(json.dumps(system_status, indent=2))
    
    # Get learning status
    learning_status = portfolio.get_learning_status()
    logger.info(f"Learning status with {len(learning_status['models_status'])} models")
    
    return portfolio

def demo_event_handling(portfolio):
    """
    Demonstrates event handling system.
    """
    logger.info("\n=== Demonstrating Event Handling ===")
    
    # Define an event handler
    def event_handler(event_type, data):
        logger.info(f"Event received - Type: {event_type}")
        logger.info(f"Event data: {json.dumps(data, indent=2)}")
    
    # Register the event handler
    portfolio.register_event_handler(event_handler)
    logger.info("Event handler registered")
    
    # Make updates to trigger events
    logger.info("Making updates to trigger events...")
    
    # Update portfolio data to trigger portfolio_update event
    portfolio.update_portfolio_data(
        cash=60000.0,
        total_value=115000.0
    )
    
    # Update performance metrics to trigger performance_update event
    portfolio.update_performance_metrics({
        "cumulative_return": 20.5,
        "sharpe_ratio": 1.9
    })
    
    # Add trade to trigger trade_added event
    portfolio.add_trade({
        "id": "trade-8",
        "symbol": "NVDA",
        "action": "BUY",
        "price": 780.0,
        "quantity": 5,
        "timestamp": datetime.now().isoformat(),
        "status": "FILLED",
        "commission": 1.0
    })
    
    # Unregister event handler
    portfolio.unregister_event_handler(event_handler)
    logger.info("Event handler unregistered")
    
    # Make another update (should not trigger any logged events)
    portfolio.update_system_status(is_market_open=False)
    logger.info("Made update after handler was unregistered (no event should be logged)")
    
    return portfolio

def main():
    """
    Run the full demonstration of the PortfolioStateManager.
    """
    logger.info("Starting PortfolioStateManager demonstration")
    
    try:
        # Create and initialize portfolio
        portfolio = demo_create_portfolio()
        
        # Demonstrate save and load functionality
        new_portfolio = demo_save_load_state(portfolio)
        
        # Demonstrate updating portfolio components
        updated_portfolio = demo_update_portfolio(new_portfolio)
        
        # Demonstrate retrieving data
        demo_retrieving_data(updated_portfolio)
        
        # Demonstrate event handling
        demo_event_handling(updated_portfolio)
        
        logger.info("\n=== Demonstration Completed Successfully ===")
    
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}", exc_info=True)
    
    finally:
        # Clean up - remove temporary directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")

if __name__ == "__main__":
    main() 