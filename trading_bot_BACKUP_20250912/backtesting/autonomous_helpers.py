#!/usr/bin/env python3
"""
Autonomous Backtester Helper Functions

This module provides support functions for the truly autonomous backtester system
which follows the 10-rule framework for backtesting without manual intervention.
"""

import os
import time
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np
import requests

# Set up logger
logger = logging.getLogger("autonomous_backtest")
if not logger.handlers:
    # Set up handler if not already configured
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# RULE 1: Data Freshness & Availability Functions
def check_news_cache_freshness() -> Dict[str, Any]:
    """
    Check if the news/sentiment cache is fresh according to Rule 1.1.
    
    Returns:
        Dict with 'fresh' boolean and 'last_update' timestamp
    """
    try:
        # Check cache freshness
        cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "data", "cache", "news_sentiment_cache.json")
        
        if not os.path.exists(cache_path):
            logger.warning("News cache not found, will create new cache")
            return {"fresh": False, "last_update": None}
        
        # Get the file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        now = datetime.now()
        
        # Check if it's market hours (9:30 AM to 4:00 PM ET)
        is_market_hours = (
            now.hour > 9 or (now.hour == 9 and now.minute >= 30)
        ) and now.hour < 16
        
        # Define freshness threshold based on market hours
        if is_market_hours:
            # During market hours: 15 minutes
            threshold = timedelta(minutes=15)
        else:
            # After hours: 60 minutes
            threshold = timedelta(minutes=60)
            
        # Check if cache is fresh
        is_fresh = now - mod_time < threshold
        
        return {
            "fresh": is_fresh,
            "last_update": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
            "age_minutes": (now - mod_time).total_seconds() / 60
        }
        
    except Exception as e:
        logger.error(f"Error checking news cache freshness: {str(e)}")
        return {"fresh": False, "last_update": None}

def update_news_sentiment_cache() -> bool:
    """
    Update the news/sentiment cache per Rule 1.1
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        logger.info("Updating news and sentiment cache")
        
        # In a real implementation, this would call APIs to fetch fresh news
        # and sentiment data for various symbols
        
        # For simulation, we'll just create a mock cache
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "data", "cache")
        
        # Create directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_path = os.path.join(cache_dir, "news_sentiment_cache.json")
        
        # Create a mock cache file with a timestamp
        with open(cache_path, 'w') as f:
            f.write('{"last_updated": "' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '"}')
            
        logger.info("News and sentiment cache updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating news sentiment cache: {str(e)}")
        return False

def fetch_market_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Fetch live market indicators for all symbols per Rule 1.2
    
    Returns:
        Dict mapping symbols to their indicator data
    """
    try:
        logger.info("Fetching live market indicators")
        
        # In a real implementation, this would call market data APIs
        # to fetch real-time price, volume, and volatility data
        
        # For simulation, we'll generate mock data
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX"]
        
        indicators = {}
        now = datetime.now()
        
        for ticker in tickers:
            # Randomly assign age in minutes, with majority being fresh (<2 min)
            # and only a small portion being stale (for testing error handling)
            age_minutes = random.choices(
                [0.5, 1, 1.5, 2, 3, 6, 10],
                weights=[30, 30, 20, 10, 5, 3, 2],  # Higher weights for fresh data
                k=1
            )[0]
            
            indicators[ticker] = {
                "price": round(random.uniform(50, 500), 2),
                "volume": random.randint(200000, 10000000),
                "volatility": round(random.uniform(0.1, 0.5), 2),
                "timestamp": (now - timedelta(minutes=age_minutes)).strftime("%Y-%m-%d %H:%M:%S"),
                "age_minutes": age_minutes
            }
            
        logger.info(f"Fetched indicators for {len(indicators)} symbols")
        return indicators
        
    except Exception as e:
        logger.error(f"Error fetching market indicators: {str(e)}")
        return {}

def log_warning(message: str) -> None:
    """Log a warning message to the logger"""
    logger.warning(message)

def log_error(message: str) -> None:
    """Log an error message to the logger"""
    logger.error(message)

# RULE 2: Symbol Discovery & Selection Functions
def calculate_opportunity_scores() -> List[Dict[str, Any]]:
    """
    Calculate opportunity scores for symbols based on Rule 2.1
    
    Returns:
        List of dicts with symbol opportunity data
    """
    try:
        logger.info("Calculating opportunity scores")
        
        # In a real implementation, this would:
        # 1. Fetch sentiment data from news API
        # 2. Normalize sentiment scores
        # 3. Get indicator signal strength
        # 4. Compute weighted opportunity score
        
        # For simulation, we'll create mock opportunity data
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", 
                 "NVDA", "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE"]
        
        opportunities = []
        
        for symbol in symbols:
            # Generate random normalized sentiment (0 to 1)
            sentiment = random.uniform(0, 1)
            
            # Generate random signal strength (0 to 1)
            signal_strength = random.uniform(0, 1)
            
            # Calculate weighted opportunity score (w1=0.6, w2=0.4)
            opportunity_score = 0.6 * sentiment + 0.4 * signal_strength
            
            # Generate random news article count
            news_articles_24h = random.randint(0, 10)
            
            # Generate random volume
            volume = random.randint(100000, 10000000)
            
            opportunities.append({
                "symbol": symbol,
                "sentiment": sentiment,
                "signal_strength": signal_strength,
                "opportunity_score": opportunity_score,
                "news_articles_24h": news_articles_24h,
                "volume": volume
            })
        
        logger.info(f"Calculated opportunity scores for {len(opportunities)} symbols")
        return opportunities
        
    except Exception as e:
        logger.error(f"Error calculating opportunity scores: {str(e)}")
        return []

# RULE 3: Strategy Assignment Functions
def get_symbol_indicators(symbol: str) -> Dict[str, float]:
    """
    Get technical indicators for a symbol per Rule 3.1
    
    Args:
        symbol: The ticker symbol
        
    Returns:
        Dict containing indicator values
    """
    try:
        # In a real implementation, this would fetch actual technical indicators
        # For simulation, we'll generate mock indicator data
        
        return {
            "ma_50": random.uniform(100, 200),
            "ma_200": random.uniform(90, 210),
            "rsi": random.uniform(20, 80),
            "vix": random.uniform(10, 35),
            "momentum": random.uniform(-1, 1),
            "adx": random.uniform(0, 50),
            "trend_strength": random.uniform(0, 1.0),
            "mean_reversion": random.uniform(0, 1.0),
            "volatility": random.uniform(0, 1.0)
        }
        
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {str(e)}")
        return {
            "ma_50": 0,
            "ma_200": 0,
            "rsi": 50,
            "vix": 15,
            "momentum": 0,
            "adx": 0,
            "trend_strength": 0.5,
            "mean_reversion": 0.5,
            "volatility": 0.5
        }

def get_symbol_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Get sentiment data for a symbol per Rule 3.2
    
    Args:
        symbol: The ticker symbol
        
    Returns:
        Dict containing sentiment data
    """
    try:
        # In a real implementation, this would fetch actual sentiment data
        # For simulation, we'll generate mock sentiment data
        
        score = random.uniform(-1, 1)
        
        return {
            "score": score,
            "magnitude": abs(score),
            "articles_count": random.randint(1, 20),
            "sources": ["CNBC", "WSJ", "Bloomberg"]
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
        return {
            "score": 0,
            "magnitude": 0,
            "articles_count": 0,
            "sources": []
        }

# RULE 4: Parameter Search & Limits Functions
def get_strategy_baseline_parameters(strategy: str) -> Dict[str, Any]:
    """
    Get baseline strategy parameters based on strategy type
    
    Args:
        strategy: The strategy name
        
    Returns:
        Dict containing parameter values
    """
    # Define base parameters for all strategies
    base_params = {
        "position_size": 0.05,  # 5% of capital by default
        "stop_loss": 0.05,      # 5% stop loss
        "take_profit": 0.15,    # 15% take profit
        "max_trades": 5         # Max 5 trades open at once
    }
    
    # Add strategy-specific parameters
    if strategy == "Trend Following":
        base_params.update({
            "ma_short": 20,
            "ma_long": 50,
            "breakout_periods": 20,
            "trailing_stop": 0.02
        })
    
    elif strategy == "Mean Reversion":
        base_params.update({
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "mean_period": 20
        })
    
    elif strategy == "Volatility Breakout":
        base_params.update({
            "volatility_period": 20,
            "volatility_multiple": 2.0,
            "breakout_threshold": 1.5,
            "consolidation_days": 10
        })
    
    elif strategy == "Momentum":
        base_params.update({
            "momentum_period": 14,
            "momentum_threshold": 0.001,
            "volume_filter": 1.5,
            "macd_fast": 12,
            "macd_slow": 26
        })
    
    elif strategy == "Contrarian":
        base_params.update({
            "sentiment_threshold": -0.3,
            "oversold_rsi": 30,
            "reversion_period": 3,
            "recovery_target": 0.1
        })
        
    else:  # Balanced or fallback
        base_params.update({
            "ma_period": 20,
            "rsi_period": 14,
            "volatility_period": 20,
            "sentiment_weight": 0.3
        })
    
    return base_params

def generate_parameter_variants(base_params: Dict[str, Any], num_variants: int = 3) -> List[Dict[str, Any]]:
    """
    Generate parameter variants within ±20% of baseline per Rule 4.1
    
    Args:
        base_params: The baseline parameters
        num_variants: Number of variants to generate
        
    Returns:
        List of parameter dictionaries
    """
    variants = []
    
    for _ in range(num_variants):
        variant = {}
        
        for key, value in base_params.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                variant[key] = value
                continue
                
            # Apply random variation within ±20%
            if isinstance(value, int):
                min_val = max(1, int(value * 0.8))
                max_val = int(value * 1.2)
                variant[key] = random.randint(min_val, max_val)
            else:  # float
                min_val = value * 0.8
                max_val = value * 1.2
                variant[key] = round(random.uniform(min_val, max_val), 3)
        
        variants.append(variant)
    
    return variants

# RULE 5 & 6: Backtest Execution and Evaluation Functions
def execute_backtest_with_controls(
    symbol: str, 
    strategy: str, 
    params: Dict[str, Any], 
    simulation_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Execute backtest with realistic slippage and commission per Rules 5.1 and 5.2
    
    Args:
        symbol: The ticker symbol
        strategy: Strategy name
        params: Strategy parameters
        simulation_config: Configuration for slippage, commissions, etc.
        
    Returns:
        Dict with backtest results or None if failed
    """
    try:
        logger.info(f"Executing backtest for {symbol} with {strategy} strategy")
        
        # In a real implementation, this would:
        # 1. Fetch historical data for the symbol
        # 2. Apply the strategy with its parameters
        # 3. Simulate trades with slippage and commission
        # 4. Track equity curve and performance metrics
        
        # For simulation, we'll generate mock backtest results
        
        # Generate random performance metrics
        sharpe_ratio = round(random.uniform(-0.5, 3.0), 2)
        total_return = round(random.uniform(-20, 60), 2)
        max_drawdown = round(random.uniform(0, 30), 2)
        win_rate = round(random.uniform(30, 80), 2)
        
        # Generate mock equity curve (100 days)
        days = 100
        equity_start = 10000
        daily_returns = np.random.normal(0.001, 0.02, days)  # mean 0.1%, std 2%
        equity_curve = [equity_start]
        
        for ret in daily_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        # Generate mock trades
        num_trades = random.randint(5, 30)
        trades = []
        
        for i in range(num_trades):
            is_win = random.random() < (win_rate / 100)
            
            if is_win:
                pnl_pct = random.uniform(1, 10)
            else:
                pnl_pct = random.uniform(-8, 0)
                
            trades.append({
                "id": i + 1,
                "entry_date": (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
                "exit_date": (datetime.now() - timedelta(days=random.randint(0, 89))).strftime("%Y-%m-%d"),
                "type": random.choice(["Long", "Short"]),
                "pnl_pct": pnl_pct,
                "pnl_amount": round(10000 * (pnl_pct / 100), 2)
            })
        
        # Return backtest results
        return {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "strategy": strategy,
            "params": params,
            "sharpe_ratio": sharpe_ratio,
            "return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "trades": trades,
            "equity_curve": equity_curve,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error executing backtest for {symbol} with {strategy}: {str(e)}")
        return None

def evaluate_performance_targets(result: Dict[str, Any], strategy: str) -> bool:
    """
    Evaluate if backtest meets performance targets per Rule 6
    
    Args:
        result: Backtest result dict
        strategy: Strategy name for target lookup
        
    Returns:
        True if all targets met, False otherwise
    """
    # Get strategy-specific targets
    targets = get_strategy_targets(strategy)
    
    # Evaluate each target
    sharpe_ok = result.get("sharpe_ratio", 0) >= targets.get("min_sharpe", 1.0)
    return_ok = result.get("return", 0) >= targets.get("target_return", 10.0)
    drawdown_ok = result.get("max_drawdown", 100) <= targets.get("max_drawdown", 20.0)
    
    # Log evaluation
    logger.info(f"Strategy {strategy} target evaluation: "
               f"Sharpe {sharpe_ok}, Return {return_ok}, Drawdown {drawdown_ok}")
    
    # Return True only if all targets are met
    return sharpe_ok and return_ok and drawdown_ok

def get_strategy_targets(strategy: str) -> Dict[str, float]:
    """
    Get performance targets for a strategy per Rule 6.1
    
    Args:
        strategy: Strategy name
        
    Returns:
        Dict with target metrics
    """
    # Define default targets
    default_targets = {
        "target_return": 10.0,    # 10% minimum return
        "min_sharpe": 1.0,        # Minimum Sharpe of 1.0
        "max_drawdown": 20.0      # Maximum 20% drawdown
    }
    
    # Define strategy-specific targets
    strategy_targets = {
        "Trend Following": {
            "target_return": 15.0,
            "min_sharpe": 1.2,
            "max_drawdown": 25.0
        },
        "Mean Reversion": {
            "target_return": 12.0,
            "min_sharpe": 1.5,
            "max_drawdown": 15.0
        },
        "Volatility Breakout": {
            "target_return": 20.0,
            "min_sharpe": 0.8,
            "max_drawdown": 30.0
        },
        "Momentum": {
            "target_return": 18.0,
            "min_sharpe": 1.2,
            "max_drawdown": 25.0
        },
        "Contrarian": {
            "target_return": 15.0,
            "min_sharpe": 1.3,
            "max_drawdown": 20.0
        }
    }
    
    # Return strategy-specific targets or default if not found
    return strategy_targets.get(strategy, default_targets)

# RULE 7: AI/ML Enhancement Functions
def get_ml_suggested_parameters(
    symbol: str, 
    strategy: str, 
    previous_results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Get ML-suggested parameters based on previous results per Rule 7
    
    Args:
        symbol: The ticker symbol
        strategy: Strategy name
        previous_results: List of previous backtest results for this symbol-strategy pair
        
    Returns:
        Dict of ML-suggested parameters or None if not available
    """
    try:
        if not previous_results:
            return None
            
        logger.info(f"Generating ML parameter suggestions for {symbol} with {strategy}")
        
        # In a real implementation, this would:
        # 1. Analyze parameter patterns from previous results
        # 2. Use ML to find optimal parameter adjustments
        # 3. Return a single parameter tweak
        
        # For simulation, we'll:
        # 1. Sort previous results by performance
        # 2. Take the best performer
        # 3. Make a small random adjustment to one parameter
        
        # Sort by performance (using Sharpe ratio as primary metric)
        sorted_results = sorted(previous_results, 
                               key=lambda x: x.get("sharpe_ratio", 0), 
                               reverse=True)
        
        if not sorted_results:
            return None
            
        # Get best performer parameters
        best_params = sorted_results[0].get("params", {})
        
        if not best_params:
            return None
            
        # Make a copy to avoid modifying the original
        suggested_params = best_params.copy()
        
        # Select one parameter to modify
        numeric_params = [k for k, v in best_params.items() if isinstance(v, (int, float))]
        
        if not numeric_params:
            return suggested_params
            
        param_to_adjust = random.choice(numeric_params)
        
        # Make a small adjustment (±10%)
        current_value = suggested_params[param_to_adjust]
        adjustment = current_value * random.uniform(-0.1, 0.1)
        
        if isinstance(current_value, int):
            # Ensure we change by at least 1 for integers
            adjustment = max(1, int(abs(adjustment))) * (1 if adjustment > 0 else -1)
            suggested_params[param_to_adjust] = max(1, current_value + adjustment)
        else:
            # For floats, just add the adjustment
            suggested_params[param_to_adjust] = max(0.001, current_value + adjustment)
        
        logger.info(f"ML suggested adjusting {param_to_adjust} from {current_value} to {suggested_params[param_to_adjust]}")
        
        return suggested_params
        
    except Exception as e:
        logger.error(f"Error getting ML parameter suggestions: {str(e)}")
        return None

# RULE 9: Logging, Monitoring & Alerts Functions
def log_backtest_completion(stats: Dict[str, Any]) -> None:
    """
    Log backtest completion stats per Rule 9.1
    
    Args:
        stats: Backtest completion statistics
    """
    logger.info(f"Autonomous backtest completed: "
               f"{stats.get('total_results', 0)} results from "
               f"{stats.get('total_jobs', 0)} jobs, "
               f"{stats.get('top_strategies', 0)} top strategies")

# RULE 10: Scheduling & On-Demand Triggers Functions
def get_next_scheduled_run_time() -> str:
    """
    Calculate the next scheduled backtest run time per Rule 10.1
    
    Returns:
        String representation of next run time
    """
    now = datetime.now()
    
    # If it's before 2 AM, next run is today at 2 AM
    if now.hour < 2:
        next_run = now.replace(hour=2, minute=0, second=0)
    # If it's before 6 AM, next run is today at 6 AM
    elif now.hour < 6:
        next_run = now.replace(hour=6, minute=0, second=0)
    # Otherwise, next run is tomorrow at 2 AM
    else:
        next_run = (now + timedelta(days=1)).replace(hour=2, minute=0, second=0)
    
    return next_run.strftime("%Y-%m-%d %H:%M:%S ET")

def update_backtest_schedule(schedule: Dict[str, str]) -> None:
    """
    Update the backtest schedule per Rule 10.1
    
    Args:
        schedule: Dict with schedule settings
    """
    logger.info(f"Updated backtest schedule. Next run: {schedule.get('next_run', 'unknown')}")

def auto_promote_strategy_to_paper(strategy: Dict[str, Any]) -> bool:
    """
    Automatically promote a strategy to paper trading
    
    Args:
        strategy: Strategy dict with details
        
    Returns:
        True if promotion successful, False otherwise
    """
    try:
        strategy_id = strategy.get("id", "unknown")
        symbol = strategy.get("symbol", "unknown")
        strategy_type = strategy.get("strategy", "unknown")
        
        logger.info(f"Auto-promoting strategy {strategy_id} for {symbol} using {strategy_type}")
        
        # In a real implementation, this would add the strategy to a paper trading system
        
        return True
        
    except Exception as e:
        logger.error(f"Error promoting strategy to paper trading: {str(e)}")
        return False
