#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to demonstrate how the PatternLearner works with backtest history data.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading_bot.backtesting.data_manager import DataManager
from trading_bot.backtesting.pattern_learner import PatternLearner
from trading_bot.examples.strategy_rotator_demo import run_demo as run_rotator_demo

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternLearnerDemo")

def run_demo(regenerate_data=True):
    """
    Run a demonstration of the PatternLearner.
    
    Args:
        regenerate_data: Whether to regenerate backtest data
    """
    # Backtest data path
    data_path = "data/demo/backtest_history.json"
    
    # Step 1: Generate backtest data if needed
    if regenerate_data or not os.path.exists(data_path):
        logger.info("Generating backtest data...")
        run_rotator_demo()
    
    # Step 2: Initialize the DataManager and PatternLearner
    logger.info("Initializing PatternLearner...")
    data_manager = DataManager(save_path=data_path)
    pattern_learner = PatternLearner(data_manager=data_manager)
    
    # Step 3: Analyze the backtest data
    logger.info("Analyzing backtest data...")
    analysis_results = pattern_learner.analyze(save_results=True)
    
    # Step 4: Print key insights
    logger.info("\n===== PATTERN ANALYSIS RESULTS =====")
    
    if "win_rates" in analysis_results and "by_strategy" in analysis_results["win_rates"]:
        logger.info("\nStrategy Win Rates:")
        for strategy in analysis_results["win_rates"]["by_strategy"]:
            logger.info(f"  {strategy['strategy']}: {strategy['win_rate']*100:.1f}% ({strategy['trade_count']} trades)")
    
    if "regime_performance" in analysis_results and "regime_returns" in analysis_results["regime_performance"]:
        logger.info("\nRegime Performance:")
        for regime in analysis_results["regime_performance"]["regime_returns"]:
            logger.info(f"  {regime['regime']}: Return: {regime['avg_daily_return']*100:.2f}%, Sharpe: {regime['sharpe_ratio']:.2f}")
    
    if "time_patterns" in analysis_results and "by_hour" in analysis_results["time_patterns"]:
        logger.info("\nTime Pattern Analysis:")
        top_hours = sorted(analysis_results["time_patterns"]["by_hour"], key=lambda x: x["win_rate"], reverse=True)[:3]
        for hour in top_hours:
            logger.info(f"  Hour {hour['hour']}: {hour['win_rate']*100:.1f}% win rate")
    
    # Step 5: Get recommendations
    logger.info("\n===== TRADING RECOMMENDATIONS =====")
    recommendations = pattern_learner.get_recommendations(analysis_results)
    
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['description']} (Confidence: {rec['confidence']:.2f})")
    
    # Step 6: Generate visualizations
    logger.info("\nGenerating visualizations...")
    plots_dir = "data/pattern_analysis/plots"
    os.makedirs(plots_dir, exist_ok=True)
    pattern_learner.plot_analysis_results(analysis_results, save_path=plots_dir)
    logger.info(f"Visualizations saved to {plots_dir}")
    
    logger.info("\nPattern analysis demo completed!")

if __name__ == "__main__":
    run_demo() 