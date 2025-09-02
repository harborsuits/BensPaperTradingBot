#!/usr/bin/env python
"""
Hybrid Strategy Optimization Script

This script runs a complete optimization of the hybrid strategy system 
using actual market data from Tradier and Alpha Vantage APIs.
"""

import os
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("hybrid_optimization")

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import trading bot components
from trading_bot.strategies.hybrid_strategy_optimizer import HybridStrategyOptimizer
from trading_bot.ml_pipeline.market_regime_detector import MarketRegimeDetector
from trading_bot.data_handlers.data_loader import DataLoader
from trading_bot.config.config_loader import ConfigLoader

def load_market_data(config, symbols, timeframes, days=180):
    """
    Load market data from APIs for the specified symbols and timeframes
    
    Args:
        config: Configuration dictionary
        symbols: List of symbol strings
        timeframes: List of timeframe strings
        days: Number of days of historical data to load
        
    Returns:
        Dictionary of timeframe -> symbol -> DataFrame with historical data
    """
    logger.info(f"Loading market data for {len(symbols)} symbols across {len(timeframes)} timeframes")
    
    # Initialize data loader
    data_loader = DataLoader(config=config)
    
    # Load data for each symbol and timeframe
    historical_data = {}
    
    for timeframe in timeframes:
        historical_data[timeframe] = {}
        
        for symbol in symbols:
            try:
                # Load historical data using configured API (Tradier, Alpha Vantage, etc.)
                df = data_loader.load_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days
                )
                
                if df is not None and not df.empty:
                    logger.info(f"Loaded {len(df)} data points for {symbol} on {timeframe} timeframe")
                    historical_data[timeframe][symbol] = df
                else:
                    logger.warning(f"No data loaded for {symbol} on {timeframe}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol} on {timeframe}: {e}")
    
    # Validate we got data
    total_datasets = sum(len(symbol_data) for symbol_data in historical_data.values())
    logger.info(f"Loaded {total_datasets} datasets across all symbols and timeframes")
    
    if total_datasets == 0:
        logger.error("No data was loaded! Check API keys and connectivity")
        return None
    
    return historical_data

def save_optimization_results(results, output_dir="optimization_results"):
    """
    Save optimization results to disk
    
    Args:
        results: Optimization results dictionary
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hybrid_optimization_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Prepare serializable results
    serializable_results = convert_numpy(results)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Saved optimization results to {filepath}")
    return filepath

def main():
    """Main function to run the hybrid strategy optimization"""
    logger.info("Starting hybrid strategy optimization")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Define symbols to optimize for - use high liquidity stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    
    # Define timeframes to test
    timeframes = ["1h", "4h", "1d"]
    
    # Load market data
    market_data = load_market_data(config, symbols, timeframes, days=180)
    
    if not market_data:
        logger.error("Failed to load market data")
        return
    
    # Initialize the hybrid strategy optimizer
    optimizer_config = {
        'results_dir': 'optimization_results',
        'market_regime': {
            'lookback_periods': 120,
            'regime_threshold': 0.75,
            'volatility_window': 20
        }
    }
    optimizer = HybridStrategyOptimizer(config=optimizer_config)
    
    # Run genetic algorithm optimization with multi-timeframe testing
    logger.info("Running genetic algorithm optimization with multi-timeframe testing")
    try:
        genetic_results = optimizer.optimize_strategy_weights(
            symbols=symbols,
            timeframes=timeframes,
            optimization_method="genetic",
            metric="sharpe_ratio",
            use_multi_timeframe=True,
            include_regime_detection=True,
            days=180
        )
        
        # Save genetic optimization results
        save_optimization_results(genetic_results, "optimization_results/genetic")
        
        # Log best parameters
        if 'best_params' in genetic_results:
            logger.info(f"Best genetic algorithm parameters: {genetic_results['best_params']}")
            if 'best_metrics' in genetic_results:
                logger.info(f"Best genetic algorithm metrics: {genetic_results['best_metrics']}")
    except Exception as e:
        logger.error(f"Error during genetic optimization: {e}")
    
    # Run Bayesian optimization on a single timeframe
    logger.info("Running Bayesian optimization on daily timeframe")
    try:
        bayesian_results = optimizer.optimize_strategy_weights(
            symbols=symbols[:3],  # Use fewer symbols for faster Bayesian optimization
            timeframes=["1d"],    # Just use daily data
            optimization_method="bayesian",
            metric="sortino_ratio",  # Different metric for comparison
            use_multi_timeframe=False,
            include_regime_detection=True,
            days=180
        )
        
        # Save Bayesian optimization results
        save_optimization_results(bayesian_results, "optimization_results/bayesian")
        
        # Log best parameters
        if 'best_params' in bayesian_results:
            logger.info(f"Best Bayesian parameters: {bayesian_results['best_params']}")
            if 'best_metrics' in bayesian_results:
                logger.info(f"Best Bayesian metrics: {bayesian_results['best_metrics']}")
    except Exception as e:
        logger.error(f"Error during Bayesian optimization: {e}")
    
    # Run walk-forward testing on best parameters
    logger.info("Running walk-forward validation")
    try:
        # Use best parameters from either genetic or Bayesian
        best_params = None
        if 'best_params' in genetic_results:
            best_params = genetic_results['best_params']
        elif 'best_params' in bayesian_results:
            best_params = bayesian_results['best_params']
        
        if best_params:
            walk_forward_results = optimizer.optimize_strategy_weights(
                symbols=["AAPL", "MSFT"],  # Use fewer symbols for faster walk-forward
                timeframes=["1d"],         # Just use daily data
                optimization_method="genetic", # Method used for in-sample optimization
                use_walk_forward=True,
                days=365  # Use more data for walk-forward
            )
            
            # Save walk-forward results
            save_optimization_results(walk_forward_results, "optimization_results/walk_forward")
            
            # Log walk-forward robustness
            if 'robustness_metrics' in walk_forward_results:
                logger.info(f"Walk-forward robustness metrics: {walk_forward_results['robustness_metrics']}")
    except Exception as e:
        logger.error(f"Error during walk-forward validation: {e}")
    
    logger.info("Hybrid strategy optimization completed")

if __name__ == "__main__":
    main()
