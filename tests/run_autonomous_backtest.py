#!/usr/bin/env python
"""
Autonomous Backtesting Pipeline Demo
This script demonstrates the end-to-end autonomous backtesting pipeline with strategy rotation
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add trading_bot to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.backtesting.unified_backtester import UnifiedBacktester
from trading_bot.config.typed_settings import TradingBotSettings, load_config
from trading_bot.core.main_orchestrator import MainOrchestrator
from trading_bot.strategies.strategy_factory import StrategyFactory

# Create output directory if it doesn't exist
os.makedirs("backtest_results", exist_ok=True)

def load_test_settings():
    """Load settings or create minimal test settings."""
    try:
        # Try to load from config.yaml first
        if os.path.exists('config.yaml'):
            return load_config('config.yaml')
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}")
    
    # Create minimal test settings if loading failed
    return TradingBotSettings.parse_obj({
        "broker": {
            "name": "paper",
            "paper_trading": True,
        },
        "risk": {
            "max_position_pct": 0.1,
            "portfolio_stop_loss_pct": 0.15,
            "max_drawdown_pct": 0.20,
            "max_positions": 10,
            "max_single_order_size": 5000,
            "enable_max_daily_loss": True,
            "max_daily_loss_pct": 0.05,
            "max_correlation": 0.7
        },
        "backtest": {
            "default_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "default_start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "default_end_date": datetime.now().strftime("%Y-%m-%d"),
            "initial_capital": 100000.0
        }
    })

def run_autonomous_backtest():
    """Run the autonomous backtesting pipeline with the orchestrator."""
    logger.info("Starting autonomous backtesting pipeline")
    
    # Load settings
    settings = load_test_settings()
    logger.info(f"Loaded settings with {len(settings.backtest.default_symbols)} symbols")
    
    # Get all available strategies
    strategy_factory = StrategyFactory()
    available_strategies = strategy_factory.get_available_strategies()
    logger.info(f"Available strategies: {available_strategies}")
    
    # Create backtester with all strategies for autonomous selection
    backtester = UnifiedBacktester(
        strategies=available_strategies,
        start_date=settings.backtest.default_start_date,
        end_date=settings.backtest.default_end_date,
        rebalance_frequency="weekly",
        benchmark_symbol="SPY",
        initial_capital=settings.backtest.initial_capital,
        settings=settings
    )
    
    # Run autonomous strategy rotation backtest (the core of your system)
    logger.info("Running autonomous strategy rotation backtest")
    results = backtester.run(mode="rotation")
    
    # Generate performance report
    report = backtester.generate_performance_report()
    
    # Print the autonomous strategy selections made by the system
    logger.info("\n=== AUTONOMOUS STRATEGY ROTATION RESULTS ===")
    for date, strategies in backtester.strategy_allocations.items():
        logger.info(f"{date}: Selected strategies: {', '.join(strategies)}")
    
    # Plot the equity curve with strategy changes highlighted
    plt.figure(figsize=(12, 8))
    plt.plot(results['equity_curve'], label='Portfolio Value', linewidth=2)
    
    # Mark strategy rotation points
    for date, strategies in backtester.strategy_allocations.items():
        if pd.to_datetime(date) in results['equity_curve'].index:
            plt.axvline(x=pd.to_datetime(date), color='r', linestyle='--', alpha=0.5)
    
    plt.title('Autonomous Strategy Rotation Results', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig('backtest_results/autonomous_strategy_rotation.png')
    logger.info("Saved strategy rotation plot to backtest_results/autonomous_strategy_rotation.png")
    
    # Save detailed results for each period
    with open('backtest_results/autonomous_backtest_report.txt', 'w') as f:
        f.write("=== AUTONOMOUS BACKTESTING REPORT ===\n\n")
        f.write(f"Initial Capital: ${settings.backtest.initial_capital:,.2f}\n")
        f.write(f"Date Range: {settings.backtest.default_start_date} to {settings.backtest.default_end_date}\n")
        f.write(f"Rebalance Frequency: Weekly\n\n")
        
        f.write("=== PERFORMANCE METRICS ===\n")
        for metric, value in report.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        f.write("\n=== AUTONOMOUS STRATEGY SELECTIONS ===\n")
        for date, strategies in backtester.strategy_allocations.items():
            f.write(f"{date}: {', '.join(strategies)}\n")
            
            # Include why the system selected this strategy
            if hasattr(backtester, 'strategy_selection_reasons') and date in backtester.strategy_selection_reasons:
                f.write(f"  Reason: {backtester.strategy_selection_reasons[date]}\n")
            
            # Include performance for this period
            if date in backtester.period_performance:
                period_perf = backtester.period_performance[date]
                f.write(f"  Period Return: {period_perf.get('return', 0):.2%}\n")
                f.write(f"  Period Sharpe: {period_perf.get('sharpe', 0):.2f}\n")
                f.write(f"  Period Max DD: {period_perf.get('max_drawdown', 0):.2%}\n")
            
            f.write("\n")
    
    logger.info("Saved detailed autonomous backtest report to backtest_results/autonomous_backtest_report.txt")
    
    return results, report, backtester

if __name__ == "__main__":
    try:
        results, report, backtester = run_autonomous_backtest()
        logger.info("Autonomous backtesting pipeline completed successfully!")
        
        print("\n========== AUTONOMOUS BACKTESTING RESULTS ==========")
        print(f"Final Portfolio Value: ${results['equity_curve'].iloc[-1]:,.2f}")
        print(f"Total Return: {report['total_return']:.2%}")
        print(f"Annualized Return: {report['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {report['max_drawdown']:.2%}")
        print(f"Win Rate: {report['win_rate']:.2%}")
        print("====================================================")
        print("\nStrategy Rotation Summary:")
        
        for date, strategies in backtester.strategy_allocations.items():
            print(f"  {date}: {', '.join(strategies)}")
        
        print("\nFor detailed results, see backtest_results/autonomous_backtest_report.txt")
        print("For the strategy rotation chart, see backtest_results/autonomous_strategy_rotation.png")
    
    except Exception as e:
        logger.error(f"Error in autonomous backtesting pipeline: {e}", exc_info=True)
        sys.exit(1)
