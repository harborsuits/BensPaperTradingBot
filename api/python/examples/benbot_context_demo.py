#!/usr/bin/env python
import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
import random

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.core.portfolio_state import PortfolioStateManager
from trading_bot.core.assistant_context import AssistantContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_portfolio_updates(portfolio_state, duration_seconds=60, update_interval=5):
    """
    Simulate portfolio updates over time.
    
    Args:
        portfolio_state: PortfolioStateManager instance
        duration_seconds: Duration of simulation in seconds
        update_interval: Interval between updates in seconds
    """
    logger.info(f"Starting portfolio simulation for {duration_seconds} seconds")
    
    # Sample stock symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B"]
    
    # Sample strategies
    strategies = ["Momentum", "Value", "Mean Reversion", "Trend Following", "Statistical Arbitrage"]
    
    # Initialize with some positions
    for symbol in symbols[:4]:  # Start with 4 positions
        qty = random.randint(10, 100)
        price = random.uniform(50, 500)
        portfolio_state.record_trade({
            "symbol": symbol,
            "quantity": qty,
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "type": "buy",
            "strategy": random.choice(strategies)
        })
    
    # Set initial strategy allocations
    for strategy in strategies:
        portfolio_state.update_strategy_allocation(
            strategy, 
            random.uniform(0.05, 0.3),  # Random allocation between 5-30%
            {"return": random.uniform(-0.05, 0.15)}  # Random performance
        )
    
    # Update system status
    portfolio_state.update_system_status("running", "open")
    
    # Update learning status
    portfolio_state.update_learning_status(
        rl_training=True,
        pattern_learning=True,
        current_episode=0,
        total_episodes=100,
        current_reward=0,
        best_reward=0
    )
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration_seconds:
        iteration += 1
        logger.info(f"Simulation iteration {iteration}")
        
        # Update portfolio values randomly
        total_value = portfolio_state.get_current_state()["portfolio"]["total_value"]
        daily_change = random.uniform(-0.02, 0.025) * total_value
        overall_change = random.uniform(-0.1, 0.15) * total_value
        
        portfolio_state.update_portfolio(
            total_value=total_value + daily_change,
            cash=total_value * random.uniform(0.1, 0.3),
            daily_pnl=daily_change,
            daily_pnl_percent=daily_change / total_value * 100 if total_value > 0 else 0,
            overall_pnl=overall_change,
            overall_pnl_percent=overall_change / total_value * 100 if total_value > 0 else 0
        )
        
        # Update metrics
        portfolio_state.update_metrics(
            daily_return=random.uniform(-0.02, 0.025),
            weekly_return=random.uniform(-0.05, 0.07),
            monthly_return=random.uniform(-0.1, 0.15),
            yearly_return=random.uniform(-0.2, 0.3),
            sharpe_ratio=random.uniform(0.5, 2.5),
            volatility=random.uniform(0.1, 0.4),
            max_drawdown=random.uniform(-0.3, -0.05),
            win_rate=random.uniform(0.4, 0.7)
        )
        
        # Update risk metrics
        portfolio_state.update_risk_metrics(
            var_95=random.uniform(-0.03, -0.01),
            var_99=random.uniform(-0.05, -0.02),
            expected_shortfall=random.uniform(-0.07, -0.03),
            beta=random.uniform(0.8, 1.2),
            correlation_to_spy=random.uniform(0.5, 0.9)
        )
        
        # Occasionally record a new trade
        if random.random() < 0.3:  # 30% chance of a trade each iteration
            trade_type = random.choice(["buy", "sell"])
            symbol = random.choice(symbols)
            qty = random.randint(5, 50)
            price = random.uniform(50, 500)
            
            portfolio_state.record_trade({
                "symbol": symbol,
                "quantity": qty,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "type": trade_type,
                "strategy": random.choice(strategies)
            })
            
            logger.info(f"Recorded {trade_type} trade for {symbol}: {qty} shares at ${price:.2f}")
        
        # Occasionally record a new signal
        if random.random() < 0.2:  # 20% chance of a signal each iteration
            signal_type = random.choice(["buy", "sell", "hold"])
            symbol = random.choice(symbols)
            
            portfolio_state.record_signal({
                "symbol": symbol,
                "signal": signal_type,
                "timestamp": datetime.now().isoformat(),
                "confidence": random.uniform(0.5, 0.95),
                "strategy": random.choice(strategies),
                "reason": f"Signal generated by {random.choice(['pattern detection', 'trend analysis', 'RL model', 'fundamental analysis'])}"
            })
            
            logger.info(f"Recorded {signal_type} signal for {symbol}")
        
        # Update learning status
        if portfolio_state.get_current_state()["learning"]["rl_training"]:
            current_episode = portfolio_state.get_current_state()["learning"]["current_episode"]
            total_episodes = portfolio_state.get_current_state()["learning"]["total_episodes"]
            
            if current_episode < total_episodes:
                new_episode = min(current_episode + random.randint(1, 5), total_episodes)
                current_reward = random.uniform(-10, 50)
                best_reward = max(portfolio_state.get_current_state()["learning"]["best_reward"], current_reward)
                
                portfolio_state.update_learning_status(
                    rl_training=True,
                    pattern_learning=True,
                    current_episode=new_episode,
                    total_episodes=total_episodes,
                    current_reward=current_reward,
                    best_reward=best_reward
                )
                
                if new_episode == total_episodes:
                    logger.info("RL training completed")
                    portfolio_state.update_learning_status(
                        rl_training=False,
                        pattern_learning=True,
                        current_episode=total_episodes,
                        total_episodes=total_episodes,
                        last_model_update=datetime.now().isoformat()
                    )
        
        time.sleep(update_interval)
    
    logger.info("Portfolio simulation completed")

def test_assistant_queries(assistant_context):
    """
    Test various types of queries to the assistant context.
    
    Args:
        assistant_context: AssistantContext instance
    """
    logger.info("Testing assistant queries")
    
    # Test queries and print results
    test_queries = [
        "How is my portfolio doing?",
        "What are my current positions?",
        "How are my strategies performing?",
        "What is my portfolio's performance?",
        "What trades have been executed recently?",
        "How is the system status?",
        "Is any training happening right now?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        context = assistant_context.get_response_context(query)
        print(f"\nContext for query '{query}':")
        print(json.dumps(context, indent=2))
        print("\n" + "-"*80)

def main():
    """
    Main function to demonstrate the integration of PortfolioStateManager and AssistantContext.
    """
    logger.info("Starting BenBot Context Demo")
    
    # Create instances
    portfolio_state = PortfolioStateManager(initial_capital=1000000)
    assistant_context = AssistantContext(portfolio_state)
    
    # Print initial state
    print("\nInitial Portfolio State:")
    print(json.dumps(portfolio_state.get_current_state(), indent=2))
    
    print("\nInitial Assistant Context:")
    print(json.dumps(assistant_context.get_context(), indent=2))
    
    # Run simulation
    simulate_portfolio_updates(portfolio_state, duration_seconds=30, update_interval=5)
    
    # Print updated state
    print("\nUpdated Portfolio State:")
    print(json.dumps(portfolio_state.get_current_state(), indent=2))
    
    print("\nUpdated Assistant Context:")
    print(json.dumps(assistant_context.get_context(), indent=2))
    
    # Test assistant queries
    test_assistant_queries(assistant_context)
    
    logger.info("BenBot Context Demo completed")

if __name__ == "__main__":
    main() 