#!/usr/bin/env python
import os
import sys
import argparse
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.enhanced_backtester import EnhancedBacktester
from backtest.backtest_visualizer import BacktestVisualizer
from trading_strategy import TradingStrategy, MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from brokers.tradier_client import TradierClient
from config_manager import ConfigManager


def setup_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger for backtesting.
    
    Args:
        log_dir: Directory to store logs
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("backtest")
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"backtest_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration for backtesting.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    # Use ConfigManager if available
    config = {}
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_configuration()
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        
        # Use default config if ConfigManager fails
        config = {
            "tradier": {
                "api_key": os.environ.get("TRADIER_API_KEY", ""),
                "account_id": os.environ.get("TRADIER_ACCOUNT_ID", ""),
                "use_sandbox": True
            },
            "backtest": {
                "data_directory": "data/historical",
                "results_directory": "results/backtests",
                "default_initial_capital": 100000.0,
                "default_commission": 0.001,  # 0.1%
                "default_slippage": 0.001    # 0.1%
            }
        }
    
    return config


def get_strategy_from_name(name: str) -> TradingStrategy:
    """
    Get strategy class from name.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy instance
    """
    strategies = {
        "moving_average_crossover": MovingAverageCrossover,
        "rsi": RSIStrategy,
        "bollinger_bands": BollingerBandsStrategy
    }
    
    if name.lower() in strategies:
        return strategies[name.lower()]()
    else:
        raise ValueError(f"Unknown strategy: {name}")


def run_backtest(
    strategy_name: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    config: Dict[str, Any] = None,
    logger: logging.Logger = None
) -> str:
    """
    Run a backtest with the specified parameters.
    
    Args:
        strategy_name: Name of the strategy to backtest
        symbols: List of symbols to backtest
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        initial_capital: Initial capital for backtest
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Path to backtest results directory
    """
    if logger is None:
        logger = setup_logger()
    
    if config is None:
        config = load_config()
    
    logger.info(f"Running backtest for {strategy_name} on {symbols} from {start_date} to {end_date}")
    
    # Get strategy instance
    strategy = get_strategy_from_name(strategy_name)
    
    # Initialize Tradier client if credentials available
    tradier_client = None
    tradier_config = config.get("tradier", {})
    api_key = tradier_config.get("api_key")
    account_id = tradier_config.get("account_id")
    use_sandbox = tradier_config.get("use_sandbox", True)
    
    if api_key and account_id:
        logger.info("Initializing Tradier client")
        tradier_client = TradierClient(
            api_key=api_key,
            account_id=account_id,
            sandbox=use_sandbox
        )
    
    # Get backtest configuration
    backtest_config = config.get("backtest", {})
    data_directory = backtest_config.get("data_directory", "data/historical")
    results_directory = backtest_config.get("results_directory", "results/backtests")
    commission = backtest_config.get("default_commission", 0.001)
    slippage = backtest_config.get("default_slippage", 0.001)
    
    # Create backtester
    backtester = EnhancedBacktester(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        data_directory=data_directory,
        results_directory=results_directory,
        commission=commission,
        slippage=slippage,
        tradier_client=tradier_client,
        logger=logger
    )
    
    # Run backtest
    results = backtester.run()
    
    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        return None
    
    # Get results directory
    result_dir = os.path.join(
        results_directory,
        f"{strategy.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create visualizations
    visualizer = BacktestVisualizer(result_dir, logger)
    visualizer.create_all_visualizations()
    
    logger.info(f"Backtest completed. Results saved to {result_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"Backtest Summary for {strategy_name}")
    print("="*50)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Annual Return: {results['performance_metrics']['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {results['performance_metrics']['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['performance_metrics']['profit_factor']:.2f}")
    print("="*50)
    
    return result_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest")
    
    # Required arguments
    parser.add_argument("--strategy", type=str, required=True, 
                        help="Strategy name (moving_average_crossover, rsi, bollinger_bands)")
    parser.add_argument("--symbols", type=str, required=True, 
                        help="Comma-separated list of symbols to backtest")
    
    # Optional arguments
    parser.add_argument("--start_date", type=str, default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"), 
                        help="Start date (YYYY-MM-DD), default: 1 year ago")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"), 
                        help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--capital", type=float, default=100000.0, 
                        help="Initial capital, default: $100,000")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to configuration file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Logging level")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Setup logger
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level=log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse symbols
    symbols = [symbol.strip() for symbol in args.symbols.split(",")]
    
    # Run backtest
    result_dir = run_backtest(
        strategy_name=args.strategy,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        config=config,
        logger=logger
    )
    
    if result_dir:
        print(f"\nDetailed results and visualizations saved to {result_dir}")
    else:
        print("\nBacktest failed. Check logs for details.") 