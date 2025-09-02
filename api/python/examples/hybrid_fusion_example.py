#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Fusion Strategy Example - Demonstrates using fundamental analysis
and sentiment data with the hybrid fusion approach.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Import strategy components
from trading_bot.strategy.hybrid_fusion_strategy import (
    HybridFusionStrategy, 
    AdaptiveTimeframeStrategy, 
    BuffettMomentumHybridStrategy
)
from trading_bot.strategy.strategy_rotator import (
    MomentumStrategy, 
    TrendFollowingStrategy, 
    MeanReversionStrategy
)
from trading_bot.common.market_types import MarketRegime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_stocks(count: int = 10, days: int = 180):
    """
    Generate sample stock data for a portfolio.
    
    Args:
        count: Number of stocks
        days: Number of days of history
        
    Returns:
        Dict mapping stock symbols to DataFrames
    """
    stocks = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create different stock types (value, growth, cyclical)
    stock_types = ['value', 'growth', 'cyclical'] * (count // 3 + 1)
    
    for i in range(count):
        symbol = f"STOCK{i+1}"
        stock_type = stock_types[i]
        
        # Base price and trend
        if stock_type == 'value':
            # Value stock: low volatility, steady growth
            base_price = 50 + np.random.randint(0, 50)
            daily_return_mean = 0.0002
            volatility = 0.008
        elif stock_type == 'growth':
            # Growth stock: higher volatility, stronger trend
            base_price = 100 + np.random.randint(0, 150)
            daily_return_mean = 0.0005
            volatility = 0.015
        else:  # cyclical
            # Cyclical stock: periodic pattern
            base_price = 75 + np.random.randint(0, 75)
            daily_return_mean = 0.0003
            volatility = 0.012
        
        # Generate price series
        prices = [base_price]
        for d in range(1, len(dates)):
            if stock_type == 'cyclical':
                # Add cyclical component 
                cycle = 0.2 * np.sin(d / 30 * np.pi)
                daily_return = np.random.normal(daily_return_mean + cycle, volatility)
            else:
                daily_return = np.random.normal(daily_return_mean, volatility)
            
            # Add some regime effects (bear market in the middle)
            if len(dates) // 3 < d < 2 * len(dates) // 3:
                # Middle third is a bear market
                daily_return -= 0.001
            
            # Calculate new price
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Create OHLC data
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': [np.random.randint(100000, 1000000) for _ in range(len(dates))],
            'type': stock_type
        })
        
        df.set_index('date', inplace=True)
        stocks[symbol] = df
    
    return stocks

def add_fundamental_data(stocks, valuation_factor=0.6):
    """
    Add mock fundamental data to stocks.
    
    Args:
        stocks: Dictionary of stock DataFrames
        valuation_factor: How closely fundamentals track price (0-1)
    
    Returns:
        Dict with fundamental data for each stock
    """
    fundamentals = {}
    
    for symbol, data in stocks.items():
        # Get latest price and stock type
        latest_price = data['close'].iloc[-1]
        stock_type = data['type'].iloc[0]
        
        # Calculate base valuation metrics based on stock type
        if stock_type == 'value':
            # Value stocks: lower P/E, higher dividend
            pe_ratio = np.random.uniform(8, 15)
            pb_ratio = np.random.uniform(1, 2)
            dividend_yield = np.random.uniform(0.02, 0.04)
            debt_equity = np.random.uniform(0.3, 0.7)
            
        elif stock_type == 'growth':
            # Growth stocks: higher P/E, minimal dividend
            pe_ratio = np.random.uniform(25, 50)
            pb_ratio = np.random.uniform(4, 10)
            dividend_yield = np.random.uniform(0, 0.01)
            debt_equity = np.random.uniform(0.5, 1.2)
            
        else:  # cyclical
            # Cyclical stocks: moderate metrics
            pe_ratio = np.random.uniform(12, 25)
            pb_ratio = np.random.uniform(1.5, 4)
            dividend_yield = np.random.uniform(0.01, 0.03)
            debt_equity = np.random.uniform(0.4, 1.0)
        
        # Calculate earnings, book value from price and ratios
        earnings = latest_price / pe_ratio
        book_value = latest_price / pb_ratio
        
        # Create fundamental data
        fundamentals[symbol] = {
            "valuation": {
                "pe_ratio": pe_ratio,
                "price_to_book": pb_ratio,
                "ev_to_ebitda": pe_ratio * 0.8,  # Approximation
                "price_to_fcf": pe_ratio * 1.1,  # Approximation
                "dividend_yield": dividend_yield
            },
            "quality": {
                "return_on_equity": np.random.uniform(0.08, 0.25),
                "return_on_assets": np.random.uniform(0.04, 0.15),
                "gross_margin": np.random.uniform(0.3, 0.6),
                "operating_margin": np.random.uniform(0.1, 0.25),
                "net_margin": np.random.uniform(0.05, 0.15)
            },
            "financial_health": {
                "current_ratio": np.random.uniform(1.2, 2.5),
                "debt_to_equity": debt_equity,
                "interest_coverage": np.random.uniform(5, 15),
                "debt_to_ebitda": np.random.uniform(1, 3),
                "free_cash_flow": earnings * np.random.uniform(0.8, 1.2) * 1e6
            },
            "growth": {
                "revenue_growth_3yr": np.random.uniform(0.05, 0.2),
                "eps_growth_3yr": np.random.uniform(0.05, 0.25),
                "fcf_growth_3yr": np.random.uniform(0.03, 0.15)
            },
            "dcf": {
                "intrinsic_value": latest_price * (1 + np.random.normal(0, 0.2)),
                "current_price": latest_price,
                "margin_of_safety": np.random.uniform(-0.2, 0.3)
            }
        }
        
        # Adjust intrinsic value based on stock type
        if stock_type == 'value':
            # Value stocks tend to be undervalued
            fundamentals[symbol]["dcf"]["intrinsic_value"] = latest_price * (1 + np.random.uniform(0.1, 0.3))
            fundamentals[symbol]["dcf"]["margin_of_safety"] = np.random.uniform(0.1, 0.3)
        elif stock_type == 'growth':
            # Growth stocks could be fairly valued or overvalued
            fundamentals[symbol]["dcf"]["intrinsic_value"] = latest_price * (1 + np.random.uniform(-0.2, 0.1))
            fundamentals[symbol]["dcf"]["margin_of_safety"] = np.random.uniform(-0.2, 0.1)
    
    return fundamentals

def add_sentiment_data(stocks):
    """
    Add mock sentiment data to stocks.
    
    Args:
        stocks: Dictionary of stock DataFrames
    
    Returns:
        Dict with sentiment data for each stock
    """
    sentiment = {}
    now = datetime.now()
    
    for symbol, data in stocks.items():
        # Get price trend to align sentiment somewhat with price action
        recent_return = data['close'].iloc[-1] / data['close'].iloc[-30] - 1
        sentiment_bias = np.clip(recent_return * 5, -0.3, 0.3)  # Scale return to sentiment
        
        # News sentiment (articles from last 7 days)
        news_items = []
        for i in range(10):  # 10 mock news articles
            days_ago = np.random.randint(0, 7)
            date = (now - timedelta(days=days_ago)).isoformat()
            
            # News sentiment correlated with recent price action
            news_sentiment = np.random.normal(sentiment_bias, 0.3)
            
            news_items.append({
                "date": date,
                "headline": f"Mock headline {i+1} for {symbol}",
                "source": f"Source {i % 3 + 1}",
                "sentiment": news_sentiment,
                "relevance": np.random.uniform(0.5, 1.0)
            })
        
        # Social media sentiment (posts from last 3 days)
        social_items = []
        for i in range(20):  # 20 mock social media posts
            hours_ago = np.random.randint(0, 72)
            date = (now - timedelta(hours=hours_ago)).isoformat()
            
            # Social media more volatile with stronger bias
            social_sentiment = np.random.normal(sentiment_bias * 1.5, 0.5)
            
            social_items.append({
                "date": date,
                "platform": ["Twitter", "Reddit", "StockTwits"][i % 3],
                "sentiment": social_sentiment,
                "engagement": np.random.randint(1, 1000)
            })
        
        # SEC filings sentiment (last 4 quarterly reports)
        filing_items = []
        for i in range(4):
            months_ago = i * 3
            date = (now - timedelta(days=months_ago * 30)).isoformat()
            
            # Fillings less biased by recent price action
            filing_sentiment = np.random.normal(sentiment_bias * 0.3, 0.2)
            
            filing_items.append({
                "date": date,
                "type": "10-Q" if i < 3 else "10-K",
                "sentiment": filing_sentiment,
                "risk_score": np.random.uniform(0, 1),
                "key_topics": ["revenue", "growth", "expenses", "outlook"][i % 4]
            })
        
        sentiment[symbol] = {
            "news": news_items,
            "social": social_items,
            "filings": filing_items,
            "aggregated": {
                "overall_sentiment": np.random.normal(sentiment_bias, 0.3),
                "sentiment_trend": np.random.normal(0, 0.1),
                "volume_trend": np.random.uniform(-0.2, 0.2)
            }
        }
    
    return sentiment

def detect_regime(prices, window=20):
    """
    Simple market regime detection based on recent returns and volatility.
    
    Args:
        prices: Series of prices
        window: Window size for calculation
    
    Returns:
        MarketRegime enum value
    """
    if len(prices) < window * 2:
        return MarketRegime.UNKNOWN
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Recent period statistics
    recent_returns = returns[-window:]
    recent_mean = np.mean(recent_returns)
    recent_vol = np.std(recent_returns)
    
    # Prior period statistics for comparison
    prior_returns = returns[-2*window:-window]
    prior_mean = np.mean(prior_returns)
    prior_vol = np.std(prior_returns)
    
    # Detect regime
    if recent_mean > 0.001:  # Strong positive trend
        if recent_vol > prior_vol * 1.5:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.BULL
    elif recent_mean < -0.001:  # Strong negative trend
        if recent_vol > prior_vol * 1.5:
            return MarketRegime.CRISIS
        else:
            return MarketRegime.BEAR
    else:  # Sideways market
        if recent_vol < prior_vol * 0.7:
            return MarketRegime.LOW_VOL
        else:
            return MarketRegime.SIDEWAYS

def prepare_market_data(stock_data, symbol, window=30):
    """
    Prepare market data dictionary for signal generation.
    
    Args:
        stock_data: DataFrame with stock data
        symbol: Stock symbol
        window: Window size for historical data
    
    Returns:
        Dict with market data
    """
    if len(stock_data) < window:
        return None
    
    # Extract recent data
    recent_data = stock_data.iloc[-window:].copy()
    
    # Create market data dictionary
    market_data = {
        "symbol": symbol,
        "price": recent_data['close'].iloc[-1],
        "volume": recent_data['volume'].iloc[-1],
        "prices": recent_data['close'].tolist(),
        "volumes": recent_data['volume'].tolist(),
        "open": recent_data['open'].tolist(),
        "high": recent_data['high'].tolist(),
        "low": recent_data['low'].tolist(),
        "regime": detect_regime(recent_data['close'].values)
    }
    
    return market_data

def test_hybrid_strategies(stocks, fundamentals, sentiment_data):
    """
    Test different hybrid strategies on the generated data.
    
    Args:
        stocks: Dictionary of stock DataFrames
        fundamentals: Dictionary of fundamental data
        sentiment_data: Dictionary of sentiment data
    
    Returns:
        DataFrame with strategy performance
    """
    # Initialize strategies
    strategies = {
        # Base strategies
        "Momentum": MomentumStrategy("Momentum", {"fast_period": 5, "slow_period": 20}),
        "TrendFollowing": TrendFollowingStrategy("TrendFollowing", {"short_ma_period": 10, "long_ma_period": 30}),
        "MeanReversion": MeanReversionStrategy("MeanReversion", {"period": 20, "std_dev_factor": 2.0}),
        
        # Hybrid strategies
        "HybridFusion": HybridFusionStrategy("HybridFusion", {
            "technical_weight": 0.4, 
            "fundamental_weight": 0.3, 
            "sentiment_weight": 0.3
        }),
        "AdaptiveTimeframe": AdaptiveTimeframeStrategy("AdaptiveTimeframe", {
            "default_timeframe": "medium_term"
        }),
        "BuffettMomentum": BuffettMomentumHybridStrategy("BuffettMomentum", {
            "base_fundamental_weight": 0.6,
            "momentum_weight": 0.25,
            "sentiment_weight": 0.15
        })
    }
    
    # Add technical strategies to hybrid strategies
    for name in ["HybridFusion", "AdaptiveTimeframe", "BuffettMomentum"]:
        hybrid = strategies[name]
        hybrid.add_technical_strategy(strategies["Momentum"])
        hybrid.add_technical_strategy(strategies["TrendFollowing"])
    
    # Track performance
    results = []
    
    # Test each stock
    for symbol, stock_data in stocks.items():
        window = 30  # Window size for signals
        
        # We'll backtest on the last 90 days
        backtest_period = min(90, len(stock_data) - window)
        
        for i in range(backtest_period):
            # Current index in the full dataset
            idx = len(stock_data) - backtest_period + i
            
            # Get current market data
            market_data = prepare_market_data(stock_data.iloc[:idx], symbol, window=window)
            if market_data is None:
                continue
            
            # Add fundamental and sentiment data
            market_data["fundamental_data"] = fundamentals.get(symbol, {})
            market_data["sentiment_data"] = sentiment_data.get(symbol, {})
            
            # Detect regime
            regime = detect_regime(stock_data.iloc[:idx]['close'].values)
            
            # Calculate future return (for performance evaluation)
            if idx < len(stock_data) - 1:
                future_return = stock_data['close'].iloc[idx+1] / stock_data['close'].iloc[idx] - 1
            else:
                future_return = 0
            
            # Generate signals from all strategies
            signals = {}
            for name, strategy in strategies.items():
                # Update regime for hybrid strategies
                if name in ["HybridFusion", "AdaptiveTimeframe", "BuffettMomentum"]:
                    strategy.update_market_regime(regime)
                
                # Generate signal
                try:
                    signal = strategy.generate_signal(market_data)
                    signals[name] = signal
                    
                    # Save result
                    results.append({
                        "date": stock_data.index[idx],
                        "symbol": symbol,
                        "strategy": name,
                        "signal": signal,
                        "price": market_data["price"],
                        "regime": regime.name,
                        "future_return": future_return,
                        "performance": signal * future_return  # Simple performance metric
                    })
                except Exception as e:
                    logger.error(f"Error generating signal for {name} on {symbol}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_results(results_df):
    """
    Plot strategy performance.
    
    Args:
        results_df: DataFrame with results
    """
    # Aggregate performance by strategy
    performance = results_df.groupby('strategy')['performance'].sum().sort_values(ascending=False)
    
    # Plot overall performance
    plt.figure(figsize=(12, 6))
    performance.plot(kind='bar')
    plt.title('Cumulative Strategy Performance')
    plt.ylabel('Performance')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Plot performance by regime
    regimes = results_df['regime'].unique()
    
    plt.figure(figsize=(14, 8))
    for i, regime in enumerate(regimes):
        regime_perf = results_df[results_df['regime'] == regime].groupby('strategy')['performance'].sum()
        
        plt.subplot(2, 3, i+1)
        regime_perf.plot(kind='bar')
        plt.title(f'Performance in {regime} Regime')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Plot signals over time for a sample stock
    symbol = results_df['symbol'].iloc[0]
    symbol_data = results_df[results_df['symbol'] == symbol]
    
    plt.figure(figsize=(14, 10))
    
    # Plot price
    plt.subplot(2, 1, 1)
    symbol_data.pivot(index='date', columns='strategy', values='price').iloc[:, 0].plot()
    plt.title(f'Price for {symbol}')
    plt.grid(True)
    
    # Plot signals
    plt.subplot(2, 1, 2)
    symbol_data.pivot(index='date', columns='strategy', values='signal').plot()
    plt.title(f'Strategy Signals for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Signal (-1 to 1)')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def main():
    """Run the hybrid fusion strategy example."""
    # Generate sample data
    logger.info("Generating sample data...")
    stocks = generate_sample_stocks(count=5, days=180)
    
    # Add fundamental and sentiment data
    logger.info("Adding fundamental data...")
    fundamentals = add_fundamental_data(stocks)
    
    logger.info("Adding sentiment data...")
    sentiment_data = add_sentiment_data(stocks)
    
    # Test strategies
    logger.info("Testing hybrid strategies...")
    results = test_hybrid_strategies(stocks, fundamentals, sentiment_data)
    
    # Print summary
    logger.info("Performance summary:")
    performance = results.groupby('strategy')['performance'].sum().sort_values(ascending=False)
    for strategy, perf in performance.items():
        logger.info(f"  {strategy}: {perf:.6f}")
    
    # Plot results
    logger.info("Plotting results...")
    plot_results(results)

if __name__ == "__main__":
    main() 