#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Evolution Runner

This script executes the genetic algorithm-based strategy evolution process
to discover and optimize trading strategies across diverse market conditions.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evolution_log.txt')
    ]
)
logger = logging.getLogger('evolution_runner')

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from trading_bot.core.strategy_evolution import StrategyEvolution
from trading_bot.core.strategy_templates import get_all_templates, get_templates_by_type
from trading_bot.data.persistence import PersistenceManager
from trading_bot.data.data_provider import DataProvider
from trading_bot.core.constants import MarketRegime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run strategy evolution')
    
    # General evolution parameters
    parser.add_argument('--population-size', type=int, default=100,
                        help='Population size per generation')
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of generations to evolve')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation rate (0.0-1.0)')
    parser.add_argument('--crossover-rate', type=float, default=0.7,
                        help='Crossover rate (0.0-1.0)')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol to use for evolution')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe to use (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-01-01',
                        help='End date for data (YYYY-MM-DD)')
    
    # Evolution mode options
    parser.add_argument('--strategy-type', type=str, default='all',
                        choices=['all', 'indicator', 'pattern', 'ml'],
                        help='Type of strategies to evolve')
    parser.add_argument('--max-time', type=int, default=3600,
                        help='Maximum evolution time in seconds')
    
    # Output options
    parser.add_argument('--output', type=str, default='evolution_results.json',
                        help='Output file for evolution results')
    parser.add_argument('--save-top', type=int, default=20,
                        help='Number of top strategies to save')
    
    return parser.parse_args()

def load_market_data(
    data_provider: DataProvider,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Load and prepare market data for different market conditions
    
    Args:
        data_provider: Data provider instance
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date string
        end_date: End date string
        
    Returns:
        Dictionary of market data for different conditions
    """
    logger.info(f"Loading market data for {symbol} {timeframe} from {start_date} to {end_date}")
    
    try:
        # Convert date strings to datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get historical data
        full_data = data_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end
        )
        
        if full_data.empty:
            logger.error(f"No data found for {symbol} {timeframe}")
            return {}
        
        logger.info(f"Loaded {len(full_data)} bars of data")
        
        # Add market regime classification
        # For the purpose of this script, we're using a simple regime classifier
        # In production, you would use your actual advanced regime detection
        full_data = add_market_regime(full_data)
        
        # Split data by market regime
        market_data = {}
        for regime in MarketRegime:
            regime_name = regime.name.lower()
            regime_data = full_data[full_data['regime'] == regime_name]
            
            if not regime_data.empty:
                logger.info(f"Found {len(regime_data)} bars for {regime_name} regime")
                market_data[regime_name] = regime_data
        
        return market_data
    
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        return {}

def add_market_regime(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime classification to data
    
    Args:
        data: Market data DataFrame
        
    Returns:
        DataFrame with regime column added
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate some basic indicators for regime detection
    # 20-day SMA
    df['sma20'] = df['close'].rolling(window=20).mean()
    # 50-day SMA
    df['sma50'] = df['close'].rolling(window=50).mean()
    # 20-day ATR for volatility
    df['tr'] = calculate_tr(df)
    df['atr20'] = df['tr'].rolling(window=20).mean()
    
    # Calculate rate of change for trend detection
    df['roc20'] = df['close'].pct_change(periods=20)
    
    # Initialize regime column
    df['regime'] = 'unknown'
    
    # Trending up regime
    trending_up = (df['close'] > df['sma50']) & (df['roc20'] > 0.01)
    df.loc[trending_up, 'regime'] = 'trending_up'
    
    # Trending down regime
    trending_down = (df['close'] < df['sma50']) & (df['roc20'] < -0.01)
    df.loc[trending_down, 'regime'] = 'trending_down'
    
    # Ranging regime
    df['percent_from_sma20'] = (df['close'] - df['sma20']) / df['sma20']
    ranging = (df['percent_from_sma20'].abs() < 0.02) & (df['roc20'].abs() < 0.01)
    df.loc[ranging, 'regime'] = 'ranging'
    
    # Volatile regime - use ATR relative to its average
    df['atr20_mean'] = df['atr20'].rolling(window=50).mean()
    volatile = df['atr20'] > (df['atr20_mean'] * 1.5)
    df.loc[volatile, 'regime'] = 'volatile'
    
    # Breakout regime - price breaking out of recent range
    df['high20'] = df['high'].rolling(window=20).max()
    df['low20'] = df['low'].rolling(window=20).min()
    breakout_up = (df['close'] > df['high20'].shift(1)) & (df['volume'] > df['volume'].rolling(window=20).mean())
    breakout_down = (df['close'] < df['low20'].shift(1)) & (df['volume'] > df['volume'].rolling(window=20).mean())
    df.loc[breakout_up | breakout_down, 'regime'] = 'breakout'
    
    # Low volatility regime
    low_vol = df['atr20'] < (df['atr20_mean'] * 0.5)
    df.loc[low_vol, 'regime'] = 'low_volatility'
    
    # Clean up NaN values for indicators
    df = df.dropna()
    
    return df

def calculate_tr(data: pd.DataFrame) -> pd.Series:
    """Calculate True Range"""
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift(1)).abs()
    low_close = (data['low'] - data['close'].shift(1)).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range

def save_results(results: Dict[str, Any], filename: str):
    """Save evolution results to a file"""
    try:
        # Convert complex objects to serializable format
        serializable_results = {
            'generations_completed': results['generations_completed'],
            'best_fitness_per_generation': results['best_fitness_per_generation'],
            'elapsed_time': results['elapsed_time'],
            'total_strategies_evaluated': results['total_strategies_evaluated'],
            'top_strategies': []
        }
        
        # Convert datetime objects in top strategies
        for strategy in results['top_strategies']:
            serializable_strategy = strategy.copy()
            
            # Convert creation_time if it's a datetime
            if 'creation_time' in serializable_strategy and isinstance(serializable_strategy['creation_time'], datetime):
                serializable_strategy['creation_time'] = serializable_strategy['creation_time'].isoformat()
                
            # Convert any datetime in fitness
            if 'fitness' in serializable_strategy and 'evaluation_time' in serializable_strategy['fitness']:
                if isinstance(serializable_strategy['fitness']['evaluation_time'], datetime):
                    serializable_strategy['fitness']['evaluation_time'] = serializable_strategy['fitness']['evaluation_time'].isoformat()
                    
            serializable_results['top_strategies'].append(serializable_strategy)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved evolution results to {filename}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize components
    persistence = PersistenceManager()
    data_provider = DataProvider()
    
    logger.info("Initializing strategy evolution")
    evolution = StrategyEvolution(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        persistence=persistence
    )
    
    # Get strategy templates
    if args.strategy_type == 'all':
        templates = get_all_templates()
    else:
        templates = get_templates_by_type(args.strategy_type)
        
    if not templates:
        logger.error(f"No templates found for strategy type: {args.strategy_type}")
        return
        
    logger.info(f"Using {len(templates)} strategy templates")
    
    # Load market data
    market_data = load_market_data(
        data_provider,
        args.symbol,
        args.timeframe,
        args.start_date,
        args.end_date
    )
    
    if not market_data:
        logger.error("No market data available for evolution")
        return
    
    # Run evolution
    logger.info(f"Starting evolution with population={args.population_size}, generations={args.generations}")
    results = evolution.evolve(
        strategy_templates=templates,
        market_data=market_data,
        max_time_seconds=args.max_time
    )
    
    # Register top strategies
    if results:
        registered_ids = evolution.register_top_strategies(top_n=args.save_top)
        logger.info(f"Registered {len(registered_ids)} top strategies")
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        print("\n==== Evolution Summary ====")
        print(f"Generations completed: {results['generations_completed']}")
        print(f"Total strategies evaluated: {results['total_strategies_evaluated']}")
        print(f"Elapsed time: {results['elapsed_time']:.2f} seconds")
        print(f"Top strategy fitness: {results['top_strategies'][0]['fitness']['overall']:.4f}")
        print(f"Strategy name: {results['top_strategies'][0]['strategy_name']}")
        print(f"Results saved to: {args.output}")
    else:
        logger.error("Evolution failed to return results")

if __name__ == "__main__":
    main()
