#!/usr/bin/env python3
"""
BenBot Assistant Demo with Portfolio State

This demo script shows how to:
1. Initialize a PortfolioStateManager with realistic trading data
2. Connect it to the AssistantContext for query processing
3. Simulate user interactions with the BenBot Assistant
4. Update portfolio state and observe how responses change
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path to import trading_bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.portfolio_state import PortfolioStateManager
from trading_bot.assistant_context import AssistantContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_portfolio() -> PortfolioStateManager:
    """
    Create a sample portfolio with realistic data.
    
    Returns:
        Initialized PortfolioStateManager with sample data
    """
    # Create a portfolio state manager with an in-memory state (no file persistence)
    portfolio = PortfolioStateManager(state_file_path=None)
    
    # Set up initial portfolio data
    portfolio.update_portfolio_data(
        cash=25432.18,
        total_value=152845.63,
        positions={
            "AAPL": {
                "symbol": "AAPL", 
                "quantity": 100,
                "avg_price": 150.75,
                "current_price": 186.32,
                "market_value": 18632.00,
                "unrealized_pnl": 3557.00,
                "unrealized_pnl_percent": 23.60,
                "sector": "Technology"
            },
            "MSFT": {
                "symbol": "MSFT",
                "quantity": 75,
                "avg_price": 290.25,
                "current_price": 342.88,
                "market_value": 25716.00,
                "unrealized_pnl": 3947.25,
                "unrealized_pnl_percent": 18.13,
                "sector": "Technology"
            },
            "AMZN": {
                "symbol": "AMZN",
                "quantity": 50,
                "avg_price": 135.60,
                "current_price": 173.22,
                "market_value": 8661.00,
                "unrealized_pnl": 1881.00,
                "unrealized_pnl_percent": 27.74,
                "sector": "Consumer Discretionary"
            },
            "GOOGL": {
                "symbol": "GOOGL",
                "quantity": 60,
                "avg_price": 120.35,
                "current_price": 159.13,
                "market_value": 9547.80,
                "unrealized_pnl": 2326.80,
                "unrealized_pnl_percent": 32.22,
                "sector": "Communication Services"
            },
            "NVDA": {
                "symbol": "NVDA",
                "quantity": 120,
                "avg_price": 430.22,
                "current_price": 509.69,
                "market_value": 61162.80,
                "unrealized_pnl": 9536.40,
                "unrealized_pnl_percent": 18.47,
                "sector": "Technology"
            },
            "SPY": {
                "symbol": "SPY",
                "quantity": 15,
                "avg_price": 420.15,
                "current_price": 516.72,
                "market_value": 7750.80,
                "unrealized_pnl": 1448.55,
                "unrealized_pnl_percent": 22.97,
                "sector": "ETF"
            }
        },
        asset_allocation={
            "Technology": 68.67,
            "Consumer Discretionary": 5.67,
            "Communication Services": 6.25,
            "ETF": 5.07,
            "Cash": 16.64
        }
    )
    
    # Set up performance metrics
    portfolio.update_performance_metrics({
        "cumulative_return": 32.45,
        "sharpe_ratio": 1.85,
        "max_drawdown": -12.3,
        "volatility": 18.7,
        "win_rate": 64.2,
        "profit_factor": 2.31,
        "recent_daily_returns": [0.45, -0.32, 1.23, 0.87, -0.12, 0.54, 0.95]
    })
    
    # Add some recent trades
    recent_trades = [
        {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 25,
            "price": 186.32,
            "total": 4658.00,
            "strategy": "Momentum"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "symbol": "MSFT",
            "action": "BUY",
            "quantity": 15,
            "price": 342.88,
            "total": 5143.20,
            "strategy": "Value"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
            "symbol": "IBM",
            "action": "SELL",
            "quantity": 30,
            "price": 175.24,
            "total": 5257.20,
            "strategy": "Rebalancing"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
            "symbol": "NVDA",
            "action": "BUY",
            "quantity": 20,
            "price": 502.35,
            "total": 10047.00,
            "strategy": "Momentum"
        }
    ]
    
    for trade in recent_trades:
        portfolio.add_trade(trade)
    
    # Add some recent signals
    recent_signals = [
        {
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "symbol": "TSLA",
            "signal_type": "BUY",
            "confidence": 0.78,
            "strategy": "Technical"
        },
        {
            "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
            "symbol": "AMZN",
            "signal_type": "HOLD",
            "confidence": 0.65,
            "strategy": "Sentiment"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "symbol": "META",
            "signal_type": "BUY",
            "confidence": 0.82,
            "strategy": "Value"
        }
    ]
    
    for signal in recent_signals:
        portfolio.add_signal(signal)
    
    # Update strategy data
    portfolio.update_strategy_data(
        active_strategies=["Momentum", "Value", "Technical", "Sentiment", "Rebalancing"],
        strategy_allocations={
            "Momentum": 35.0,
            "Value": 25.0,
            "Technical": 20.0,
            "Sentiment": 10.0,
            "Rebalancing": 10.0
        },
        strategy_performance={
            "Momentum": {
                "returns": 42.3,
                "sharpe_ratio": 2.1,
                "max_drawdown": -15.2
            },
            "Value": {
                "returns": 18.7,
                "sharpe_ratio": 1.4,
                "max_drawdown": -10.5
            },
            "Technical": {
                "returns": 26.5,
                "sharpe_ratio": 1.8,
                "max_drawdown": -14.3
            },
            "Sentiment": {
                "returns": 22.1,
                "sharpe_ratio": 1.6,
                "max_drawdown": -12.1
            },
            "Rebalancing": {
                "returns": 15.4,
                "sharpe_ratio": 1.2,
                "max_drawdown": -8.7
            }
        }
    )
    
    # Update system status
    portfolio.update_system_status(
        is_market_open=True,
        market_hours="9:30 AM - 4:00 PM ET",
        data_providers=["Alpha Vantage", "IEX Cloud", "Yahoo Finance"],
        connected_brokers=["Alpaca", "Interactive Brokers"],
        system_health={
            "data_feed": "Operational",
            "order_execution": "Operational",
            "strategy_engine": "Operational",
            "ml_pipeline": "Processing",
            "database": "Operational"
        }
    )
    
    # Update learning status
    portfolio.update_learning_status(
        training_in_progress=False,
        models_status={
            "Price Predictor": "Trained",
            "Sentiment Analyzer": "Trained",
            "Market Regime Classifier": "Training",
            "Volatility Predictor": "Trained"
        },
        recent_learning_metrics={
            "Price Predictor": {
                "mae": 0.023,
                "mse": 0.0015,
                "accuracy": 0.73
            },
            "Sentiment Analyzer": {
                "precision": 0.82,
                "recall": 0.79,
                "f1_score": 0.81
            },
            "Volatility Predictor": {
                "mae": 0.031,
                "mse": 0.0022,
                "accuracy": 0.68
            }
        }
    )
    
    logger.info("Sample portfolio created with realistic data")
    return portfolio

def simulate_user_interaction(assistant_context):
    """
    Simulate a user interaction with the BenBot Assistant.
    
    Args:
        assistant_context: The AssistantContext instance
    """
    # List of sample queries
    sample_queries = [
        "What's in my portfolio?",
        "Show me my AAPL position",
        "How is my portfolio performing?",
        "What are my recent trades?",
        "Which strategies are being used?",
        "Is the market open right now?",
        "What's the status of the AI models?"
    ]
    
    # Randomly select a query
    query = random.choice(sample_queries)
    
    # Process the query
    response = assistant_context.process_query(query)
    
    # Print the interaction
    print("\n" + "="*80)
    print(f"USER: {query}")
    print("-"*80)
    print(f"BENBOT: {response['formatted_response']}")
    print("="*80)

def simulate_market_update(portfolio_state):
    """
    Simulate a market update with price changes and new trades.
    
    Args:
        portfolio_state: The PortfolioStateManager instance
    """
    logger.info("Simulating market update...")
    
    # Get current positions
    current_positions = portfolio_state.get_positions()
    
    # Update prices with small random changes
    updated_positions = {}
    for symbol, position in current_positions.items():
        # Random price change (-2% to +2%)
        price_change_pct = (random.random() * 4) - 2
        new_price = position["current_price"] * (1 + (price_change_pct / 100))
        
        # Update position
        updated_position = position.copy()
        updated_position["current_price"] = round(new_price, 2)
        updated_position["market_value"] = round(updated_position["quantity"] * new_price, 2)
        updated_position["unrealized_pnl"] = round(updated_position["market_value"] - 
                                                  (updated_position["quantity"] * updated_position["avg_price"]), 2)
        updated_position["unrealized_pnl_percent"] = round((updated_position["unrealized_pnl"] / 
                                                          (updated_position["quantity"] * updated_position["avg_price"])) * 100, 2)
        
        updated_positions[symbol] = updated_position
    
    # Calculate new total value
    total_value = sum(pos["market_value"] for pos in updated_positions.values())
    cash = portfolio_state.get_portfolio_summary()["cash"]
    total_value += cash
    
    # Update portfolio data
    portfolio_state.update_portfolio_data(
        positions=updated_positions,
        total_value=round(total_value, 2)
    )
    
    # 30% chance to add a new trade
    if random.random() < 0.3:
        # Pick a random position
        symbol = random.choice(list(updated_positions.keys()))
        
        # Decide buy or sell
        action = "BUY" if random.random() > 0.4 else "SELL"
        
        # Random quantity (5-20% of current position)
        current_quantity = updated_positions[symbol]["quantity"]
        quantity = max(1, int(current_quantity * (random.random() * 0.15 + 0.05)))
        
        # Ensure we don't sell more than we have
        if action == "SELL":
            quantity = min(quantity, current_quantity - 1)  # Keep at least 1 share
        
        # Random strategy
        strategies = ["Momentum", "Value", "Technical", "Sentiment", "Rebalancing"]
        strategy = random.choice(strategies)
        
        # Create and add trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": updated_positions[symbol]["current_price"],
            "total": round(quantity * updated_positions[symbol]["current_price"], 2),
            "strategy": strategy
        }
        
        portfolio_state.add_trade(trade)
        logger.info(f"Added new trade: {action} {quantity} {symbol}")
    
    # 20% chance to add a new signal
    if random.random() < 0.2:
        # Pick a random symbol (including ones we don't own)
        potential_symbols = list(updated_positions.keys()) + ["TSLA", "META", "AMD", "JPM", "DIS"]
        symbol = random.choice(potential_symbols)
        
        # Random signal type
        signal_type = random.choice(["BUY", "SELL", "HOLD"])
        
        # Random confidence (0.6-0.95)
        confidence = round(random.random() * 0.35 + 0.6, 2)
        
        # Random strategy
        strategies = ["Momentum", "Value", "Technical", "Sentiment"]
        strategy = random.choice(strategies)
        
        # Create and add signal
        signal = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "strategy": strategy
        }
        
        portfolio_state.add_signal(signal)
        logger.info(f"Added new signal: {signal_type} {symbol} ({confidence:.2f})")
    
    logger.info("Market update simulation completed")

def main():
    """
    Main function demonstrating the BenBot Assistant with portfolio state.
    """
    logger.info("Starting BenBot Assistant Demo")
    
    # Create a sample portfolio
    portfolio_state = create_sample_portfolio()
    
    # Initialize AssistantContext with portfolio state
    assistant_context = AssistantContext(portfolio_state_manager=portfolio_state)
    
    # Run simulation for a period of time
    simulation_duration = 60  # seconds
    start_time = time.time()
    
    print("\nStarting BenBot Assistant simulation...")
    print(f"Running for {simulation_duration} seconds with market updates and user interactions")
    
    while time.time() - start_time < simulation_duration:
        # Simulate a user interaction
        simulate_user_interaction(assistant_context)
        
        # Wait a bit
        time.sleep(5)
        
        # 40% chance to simulate a market update
        if random.random() < 0.4:
            simulate_market_update(portfolio_state)
    
    # Final portfolio summary
    print("\n" + "="*80)
    print("FINAL PORTFOLIO SUMMARY")
    print("-"*80)
    summary = portfolio_state.get_portfolio_summary()
    positions = portfolio_state.get_positions()
    
    print(f"Total Value: ${summary['total_value']:.2f}")
    print(f"Cash Balance: ${summary['cash']:.2f}")
    print(f"Number of Positions: {len(positions)}")
    print(f"Recent Trades: {len(portfolio_state.get_recent_activity()['recent_trades'])}")
    print(f"Recent Signals: {len(portfolio_state.get_recent_activity()['recent_signals'])}")
    print("="*80)
    
    logger.info("BenBot Assistant Demo completed")

if __name__ == "__main__":
    main() 