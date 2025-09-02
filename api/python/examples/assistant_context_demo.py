#!/usr/bin/env python3
"""
Demo script for AssistantContext usage.

This script:
1. Creates a sample portfolio with the PortfolioStateManager
2. Initializes an AssistantContext with this portfolio data
3. Demonstrates processing different types of user queries
4. Shows how the formatted responses look
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path to import trading_bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.portfolio_state import PortfolioStateManager
from trading_bot.assistant_context import AssistantContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_portfolio():
    """Create a sample portfolio with test data."""
    portfolio = PortfolioStateManager()
    
    # Initialize with starting portfolio data
    portfolio.initialize_portfolio(
        total_value=100000.0,
        cash=25000.0,
        positions={
            "AAPL": {
                "symbol": "AAPL",
                "quantity": 50,
                "avg_price": 175.25,
                "current_price": 187.50,
                "market_value": 50 * 187.50
            },
            "MSFT": {
                "symbol": "MSFT",
                "quantity": 30,
                "avg_price": 320.15,
                "current_price": 337.42,
                "market_value": 30 * 337.42
            },
            "NVDA": {
                "symbol": "NVDA",
                "quantity": 25,
                "avg_price": 420.75,
                "current_price": 450.23,
                "market_value": 25 * 450.23
            },
            "AMZN": {
                "symbol": "AMZN",
                "quantity": 20,
                "avg_price": 130.50,
                "current_price": 141.25,
                "market_value": 20 * 141.25
            }
        },
        asset_allocation={
            "Stocks": 0.75,
            "Cash": 0.25,
            "Bonds": 0.0,
            "Crypto": 0.0
        }
    )
    
    # Add performance metrics
    portfolio.update_metrics({
        "cumulative_return": 0.12,
        "sharpe_ratio": 1.85,
        "max_drawdown": -0.08,
        "volatility": 0.15,
        "win_rate": 0.65,
        "recent_daily_returns": [0.01, -0.005, 0.02, 0.01, -0.01]
    })
    
    # Add some recent trades
    now = datetime.now()
    portfolio.update_trades([
        {
            "timestamp": (now - timedelta(days=1)).isoformat(),
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 10,
            "price": 180.25,
            "value": 1802.50,
            "fees": 1.99
        },
        {
            "timestamp": (now - timedelta(days=3)).isoformat(),
            "symbol": "MSFT",
            "action": "BUY",
            "quantity": 5,
            "price": 322.50,
            "value": 1612.50,
            "fees": 1.99
        },
        {
            "timestamp": (now - timedelta(days=5)).isoformat(),
            "symbol": "NVDA",
            "action": "BUY",
            "quantity": 8,
            "price": 412.75,
            "value": 3302.00,
            "fees": 1.99
        },
        {
            "timestamp": (now - timedelta(days=7)).isoformat(),
            "symbol": "TSLA",
            "action": "SELL",
            "quantity": 15,
            "price": 178.30,
            "value": 2674.50,
            "fees": 1.99
        }
    ])
    
    # Add some recent signals
    portfolio.update_signals([
        {
            "timestamp": (now - timedelta(hours=3)).isoformat(),
            "symbol": "AAPL",
            "signal_type": "BUY",
            "confidence": 0.85,
            "source": "momentum_strategy"
        },
        {
            "timestamp": (now - timedelta(hours=5)).isoformat(),
            "symbol": "MSFT",
            "signal_type": "HOLD",
            "confidence": 0.65,
            "source": "mean_reversion_strategy"
        },
        {
            "timestamp": (now - timedelta(hours=8)).isoformat(),
            "symbol": "AMZN",
            "signal_type": "BUY",
            "confidence": 0.75,
            "source": "breakout_strategy"
        }
    ])
    
    # Add strategies information
    portfolio.update_strategies(
        active_strategies=["momentum", "breakout", "mean_reversion"],
        strategy_allocations={
            "momentum": 0.4,
            "breakout": 0.3,
            "mean_reversion": 0.3
        },
        strategy_performance={
            "momentum": {
                "return": 0.18,
                "sharpe": 1.95,
                "win_rate": 0.68
            },
            "breakout": {
                "return": 0.12,
                "sharpe": 1.65,
                "win_rate": 0.62
            },
            "mean_reversion": {
                "return": 0.09,
                "sharpe": 1.45,
                "win_rate": 0.57
            }
        }
    )
    
    # Update system status
    portfolio.update_system_status(
        is_market_open=True,
        market_hours="09:30 - 16:00 ET",
        data_providers=["alpha_vantage", "yahoo_finance"],
        connected_brokers=["alpaca", "interactive_brokers"],
        system_health={
            "data_feed": "healthy",
            "execution": "healthy",
            "database": "healthy",
            "api": "healthy"
        }
    )
    
    # Update learning status
    portfolio.update_learning_status(
        training_in_progress=False,
        last_training_time=(now - timedelta(days=1)).isoformat(),
        models_status={
            "momentum": "active",
            "breakout": "active",
            "mean_reversion": "active"
        },
        recent_learning_metrics={
            "momentum": {
                "train_accuracy": 0.85,
                "val_accuracy": 0.82,
                "f1_score": 0.84
            },
            "breakout": {
                "train_accuracy": 0.83,
                "val_accuracy": 0.80,
                "f1_score": 0.81
            }
        }
    )
    
    return portfolio


def test_queries(assistant_context):
    """Test various query types and print the formatted responses."""
    test_cases = [
        "Show me my portfolio summary",
        "What's my position in AAPL?",
        "How much NVDA do I own?",
        "Show me my performance metrics",
        "What strategies are currently active?",
        "What are my recent trades?",
        "What's the system status?",
        "Tell me about my MSFT holdings",
        "How am I doing overall?",
        "What's my asset allocation?",
        "AMZN position details",
        "Is the market open right now?",
        "What's the status of my learning models?"
    ]
    
    for query in test_cases:
        print("\n" + "="*80)
        print(f"USER QUERY: '{query}'")
        print("-"*80)
        
        # Get query classification and context data
        context = assistant_context.process_query(query)
        query_type = context.get("query_type", "UNKNOWN")
        symbols = context.get("symbols", [])
        
        print(f"Query classified as: {query_type}" + (f" for symbols: {symbols}" if symbols else ""))
        
        # Get formatted response
        formatted_response = assistant_context.get_formatted_context(query)
        print("\nFORMATTED RESPONSE:")
        print(formatted_response)


def main():
    """Main function to demo the AssistantContext."""
    logger.info("Creating sample portfolio...")
    portfolio = create_sample_portfolio()
    
    logger.info("Initializing AssistantContext...")
    assistant_context = AssistantContext(portfolio)
    
    logger.info("Testing different query types...")
    test_queries(assistant_context)
    
    logger.info("Demo completed!")


if __name__ == "__main__":
    main() 