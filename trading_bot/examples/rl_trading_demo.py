#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL Trading Demo

This script demonstrates how to:
1. Generate backtest data using the strategy rotator
2. Train an RL agent to learn optimal strategy allocations
3. Evaluate the RL agent's performance
4. Integrate the RL agent with the trading system
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading_bot.examples.strategy_rotator_demo import run_demo as run_rotator_demo
from trading_bot.backtesting.data_manager import DataManager
from trading_bot.backtesting.pattern_learner import PatternLearner
from trading_bot.learning.rl_agent import RLStrategyAgent
from trading_bot.learning.rl_trainer import RLTrainer
from trading_bot.learning.rl_environment import RLTradingEnv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RLTradingDemo")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import gymnasium
        import stable_baselines3
        logger.info("All required dependencies are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install the required packages: pip install gymnasium stable-baselines3")
        return False

def generate_backtest_data():
    """Generate backtest data using the strategy rotator demo"""
    logger.info("Generating backtest data...")
    data_path = "data/demo/backtest_history.json"
    
    # Check if data already exists
    if os.path.exists(data_path):
        logger.info(f"Backtest data already exists at {data_path}")
        
        # Ask user if they want to regenerate
        response = input("Do you want to regenerate the backtest data? (y/n): ")
        if response.lower() != 'y':
            logger.info("Using existing backtest data.")
            return data_path
    
    # Run strategy rotator demo to generate data
    run_rotator_demo()
    logger.info(f"Backtest data generated and saved to {data_path}")
    
    return data_path

def analyze_backtest_data(data_path):
    """Analyze backtest data using the PatternLearner"""
    logger.info("Analyzing backtest data...")
    
    # Initialize DataManager and PatternLearner
    data_manager = DataManager(save_path=data_path)
    pattern_learner = PatternLearner(data_manager=data_manager)
    
    # Analyze data
    analysis_results = pattern_learner.analyze(save_results=True)
    
    # Print key insights
    logger.info("\n===== PATTERN ANALYSIS RESULTS =====")
    
    if "win_rates" in analysis_results and "by_strategy" in analysis_results["win_rates"]:
        logger.info("\nStrategy Win Rates:")
        for strategy in analysis_results["win_rates"]["by_strategy"]:
            logger.info(f"  {strategy['strategy']}: {strategy['win_rate']*100:.1f}% ({strategy['trade_count']} trades)")
    
    # Get recommendations
    recommendations = pattern_learner.get_recommendations(analysis_results)
    
    logger.info("\n===== TRADING RECOMMENDATIONS =====")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['description']} (Confidence: {rec['confidence']:.2f})")
    
    logger.info("\nAnalysis complete.")
    
    return analysis_results

def train_rl_agent(data_path, strategies, timesteps=50000):
    """Train an RL agent on the backtest data"""
    logger.info(f"Training RL agent for {timesteps} timesteps...")
    
    # Initialize RLTrainer
    trainer = RLTrainer(
        strategies=strategies,
        data_path=data_path,
        model_dir="models/rl_demo",
        results_dir="results/rl_demo",
        algorithm="PPO",
        reward_type="sharpe",
        use_pattern_insights=True,
        window_size=30,
        episode_length=252,
        verbose=1
    )
    
    # Train agent
    training_results = trainer.train(total_timesteps=timesteps, eval_freq=5000)
    
    logger.info("\n===== TRAINING RESULTS =====")
    logger.info(f"Training duration: {training_results['duration_seconds'] / 60:.1f} minutes")
    logger.info(f"Mean reward: {training_results['mean_episode_reward']:.2f}")
    logger.info(f"Episodes: {training_results['n_episodes']}")
    
    logger.info("\n===== EVALUATION RESULTS =====")
    if "evaluation" in training_results:
        eval_results = training_results["evaluation"]
        logger.info(f"Mean reward: {eval_results['mean_reward']:.2f}")
        logger.info(f"Mean return: {eval_results['mean_return']*100:.2f}%")
        logger.info(f"Sharpe ratio: {eval_results['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {eval_results['mean_max_drawdown']*100:.2f}%")
    
    logger.info("\nTraining complete.")
    
    return trainer

def evaluate_rl_agent(trainer, n_episodes=10):
    """Evaluate the RL agent's performance"""
    logger.info(f"Evaluating RL agent for {n_episodes} episodes...")
    
    # Evaluate agent
    eval_results = trainer.evaluate(n_episodes=n_episodes)
    
    logger.info("\n===== DETAILED EVALUATION RESULTS =====")
    logger.info(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    logger.info(f"Mean return: {eval_results['mean_return']*100:.2f}%")
    logger.info(f"Sharpe ratio: {eval_results['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {eval_results['mean_max_drawdown']*100:.2f}%")
    
    # Print individual episode results
    logger.info("\nEpisode Results:")
    for i, metrics in enumerate(eval_results['episode_metrics'], 1):
        logger.info(f"Episode {i}: Return: {metrics['portfolio_return']*100:.2f}%, "
                   f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    logger.info("\nEvaluation complete.")
    
    return eval_results

def demonstrate_continuous_learning(trainer, demo_time=60):
    """Demonstrate continuous learning in a background thread"""
    logger.info(f"Starting continuous learning demo for {demo_time} seconds...")
    
    # Start continuous improvement
    thread = trainer.start_continuous_improvement(train_timesteps=10000, check_interval=10)
    
    # Monitor status periodically
    start_time = time.time()
    while time.time() - start_time < demo_time:
        status = trainer.get_system_status()
        
        logger.info("\n===== SYSTEM STATUS =====")
        logger.info(f"Is training: {status['is_training']}")
        logger.info(f"Is evaluating: {status['is_evaluating']}")
        logger.info(f"Training runs: {status['training_runs']}")
        logger.info(f"Evaluation runs: {status['evaluation_runs']}")
        if status['latest_performance'] is not None:
            logger.info(f"Latest performance: {status['latest_performance']:.2f}")
        
        # Wait a bit before checking again
        time.sleep(15)
    
    logger.info("Continuous learning demo completed.")

def demonstrate_integration(trainer, strategies):
    """Demonstrate integration with the trading system"""
    logger.info("Demonstrating integration with the trading system...")
    
    # Create some dummy market data and signals
    market_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(30)],
        'SPY': [100 + i * 0.1 for i in range(30)],
        'QQQ': [200 + i * 0.2 for i in range(30)],
        'IWM': [150 + i * 0.15 for i in range(30)],
        'volatility': [0.2 - i * 0.002 for i in range(30)]
    })
    
    signals = {
        strategy: {
            'signals': [
                {'type': 'entry', 'direction': 'buy', 'confidence': 0.7 + 0.1 * i, 'symbol': 'SPY'}
                for i in range(3)
            ],
            'performance': {
                'win_rate': 0.6,
                'sharpe': 1.2,
                'returns': 0.05
            }
        }
        for i, strategy in enumerate(strategies)
    }
    
    # Get recommended allocations
    allocations = trainer.get_recommended_allocations(market_data, signals)
    
    logger.info("\n===== RECOMMENDED ALLOCATIONS =====")
    for strategy, weight in allocations.items():
        logger.info(f"{strategy}: {weight:.2%}")
    
    logger.info("\nIntegration demo complete.")
    
    return allocations

def run_demo():
    """Run the complete RL trading demo"""
    # Check dependencies
    if not check_dependencies():
        return
    
    # Make sure data directories exist
    os.makedirs("data/demo", exist_ok=True)
    os.makedirs("models/rl_demo", exist_ok=True)
    os.makedirs("results/rl_demo", exist_ok=True)
    
    # Define strategies
    strategies = ['momentum', 'trend_following', 'mean_reversion']
    
    # Step 1: Generate backtest data
    data_path = generate_backtest_data()
    
    # Step 2: Analyze backtest data
    analysis_results = analyze_backtest_data(data_path)
    
    # Step 3: Train RL agent
    # Use a smaller number of timesteps for the demo
    timesteps = int(input("Enter number of training timesteps (recommended: 10000-50000): "))
    trainer = train_rl_agent(data_path, strategies, timesteps=timesteps)
    
    # Step 4: Evaluate RL agent
    eval_results = evaluate_rl_agent(trainer, n_episodes=5)
    
    # Step 5: Demonstrate continuous learning (optional)
    run_continuous = input("Run continuous learning demo? (y/n): ")
    if run_continuous.lower() == 'y':
        demonstrate_continuous_learning(trainer, demo_time=60)
    
    # Step 6: Demonstrate integration
    demonstrate_integration(trainer, strategies)
    
    logger.info("\n===== RL TRADING DEMO COMPLETED =====")
    logger.info("The RL agent has been trained to allocate between strategies based on market conditions.")
    logger.info("You can now use the agent in your trading system to dynamically adjust strategy allocations.")
    logger.info(f"Model saved to: {trainer.agent.model_path}")
    logger.info(f"Best model saved to: {trainer.best_model_path}")

if __name__ == "__main__":
    run_demo() 