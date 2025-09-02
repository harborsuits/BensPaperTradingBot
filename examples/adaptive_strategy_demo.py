#!/usr/bin/env python
"""
Adaptive Strategy Controller Demo

This script demonstrates how to use the Adaptive Strategy Controller
to manage trading strategies with dynamic allocation, position sizing,
and risk management based on market conditions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.analytics.performance_tracker import PerformanceTracker
from trading_bot.analytics.market_regime_detector import MarketRegimeDetector, MarketRegime


def generate_sample_data(symbol, days=200, regime='trending_up'):
    """Generate sample price data for testing"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days + 1)]
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    
    close_prices = [100.0]
    
    if regime == 'trending_up':
        # Trending up with noise
        for i in range(1, days + 1):
            close_prices.append(close_prices[-1] * (1 + np.random.normal(0.001, 0.008)))
    
    elif regime == 'trending_down':
        # Trending down with noise
        for i in range(1, days + 1):
            close_prices.append(close_prices[-1] * (1 + np.random.normal(-0.001, 0.008)))
    
    elif regime == 'ranging':
        # Ranging market around 100
        for i in range(1, days + 1):
            close_prices.append(100.0 + np.random.normal(0, 2.0))
    
    elif regime == 'volatile':
        # Volatile market
        for i in range(1, days + 1):
            close_prices.append(close_prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    elif regime == 'breakout':
        # Ranging then breakout
        for i in range(1, int(days * 0.7)):
            close_prices.append(100.0 + np.random.normal(0, 1.5))
        
        # Breakout point
        close_prices.append(close_prices[-1] * 1.03)
        
        # Trending after breakout
        for i in range(int(days * 0.7) + 1, days + 1):
            close_prices.append(close_prices[-1] * (1 + np.random.normal(0.001, 0.008)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_strs,
        'open': close_prices,
        'high': [p * (1 + np.random.uniform(0.001, 0.01)) for p in close_prices],
        'low': [p * (1 - np.random.uniform(0.001, 0.01)) for p in close_prices],
        'close': close_prices,
        'volume': [1000000 + np.random.randint(-200000, 500000) for _ in range(days + 1)]
    })
    
    return df


def simulate_trades(controller, strategy_id, symbol, num_trades=20, win_rate=0.6):
    """Simulate trades for a strategy"""
    logger.info(f"Simulating {num_trades} trades for {strategy_id} on {symbol} (win rate: {win_rate:.0%})")
    
    for i in range(num_trades):
        # Determine if trade is a winner
        is_winner = np.random.random() < win_rate
        
        # Simulate random entry and exit prices
        entry_price = 100.0 + np.random.normal(0, 2.0)
        
        # Calculate profit or loss
        pnl_pct = np.random.uniform(0.01, 0.03) if is_winner else -np.random.uniform(0.01, 0.02)
        exit_price = entry_price * (1 + pnl_pct)
        
        # Generate trade data
        trade_data = {
            'entry_time': (datetime.now() - timedelta(days=20) + timedelta(days=i)).isoformat(),
            'exit_time': (datetime.now() - timedelta(days=20) + timedelta(days=i, hours=6)).isoformat(),
            'symbol': symbol,
            'direction': 'long',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': 10,
            'pnl': (exit_price - entry_price) * 10,
            'pnl_pct': pnl_pct,
            'fees': 1.0,
            'slippage': 0.02 if is_winner else 0.05
        }
        
        # Record trade
        metrics = controller.record_trade_result(strategy_id, trade_data)
        
        if i % 5 == 0:
            logger.info(f"Trade {i+1}: {'PROFIT' if is_winner else 'LOSS'} - PnL: ${trade_data['pnl']:.2f}")
    
    # Get final metrics
    status = controller.get_strategy_status(strategy_id)
    if 'performance' in status:
        perf = status['performance']
        logger.info(f"Strategy {strategy_id} final metrics: Win Rate: {perf['win_rate']:.2%}, Profit Factor: {perf['profit_factor']:.2f}")


def plot_allocations(controller, title="Strategy Allocations"):
    """Plot strategy allocations"""
    allocations = controller.get_all_allocations()
    
    plt.figure(figsize=(10, 6))
    plt.bar(allocations.keys(), allocations.values())
    plt.title(title)
    plt.ylabel('Allocation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('strategy_allocations.png')
    logger.info(f"Saved allocation chart to strategy_allocations.png")


def plot_regime_comparison(symbol, regimes, title="Market Regime Comparison"):
    """Plot regime strength over time"""
    plt.figure(figsize=(12, 6))
    
    dates = [datetime.fromisoformat(entry['date']).date() for entry in regimes]
    
    strengths = {
        'trending_up': [
            entry['strength'] if entry['regime'] == 'trending_up' else 0 
            for entry in regimes
        ],
        'trending_down': [
            entry['strength'] if entry['regime'] == 'trending_down' else 0 
            for entry in regimes
        ],
        'ranging': [
            entry['strength'] if entry['regime'] == 'ranging' else 0 
            for entry in regimes
        ],
        'volatile': [
            entry['strength'] if entry['regime'] == 'volatile' else 0 
            for entry in regimes
        ],
        'breakout': [
            entry['strength'] if entry['regime'] == 'breakout' else 0 
            for entry in regimes
        ]
    }
    
    plt.plot(dates, strengths['trending_up'], label='Trending Up', color='green')
    plt.plot(dates, strengths['trending_down'], label='Trending Down', color='red')
    plt.plot(dates, strengths['ranging'], label='Ranging', color='blue')
    plt.plot(dates, strengths['volatile'], label='Volatile', color='orange')
    plt.plot(dates, strengths['breakout'], label='Breakout', color='purple')
    
    plt.title(f"{title} for {symbol}")
    plt.xlabel('Date')
    plt.ylabel('Regime Strength')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{symbol}_regimes.png')
    logger.info(f"Saved regime chart to {symbol}_regimes.png")


def main():
    """Main demo function"""
    logger.info("Starting Adaptive Strategy Controller Demo")
    
    # Initialize controller with configuration
    config = {
        'initial_equity': 10000.0,
        'allocation_frequency': 'daily',
        'parameter_update_frequency': 'daily',
        'performance_tracker': {
            'data_dir': './data/performance',
            'auto_save': True
        },
        'market_regime_detector': {
            'data_dir': './data/regimes',
            'auto_save': True
        },
        'snowball_allocator': {
            'rebalance_frequency': 'daily',
            'snowball_reinvestment_ratio': 0.5,
            'min_weight': 0.05,
            'max_weight': 0.9,
            'normalization_method': 'simple'
        }
    }
    
    controller = AdaptiveStrategyController(config=config)
    
    # Register test strategies
    strategies = {
        'trend_following': {
            'name': 'Trend Following Strategy',
            'description': 'Follows market trends using moving averages',
            'category': 'trend_following',
            'symbols': ['SPY', 'QQQ'],
            'timeframes': ['1d'],
            'parameters': {
                'ma_fast': 20,
                'ma_slow': 50,
                'trailing_stop_pct': 2.0,
                'profit_target_pct': 3.0
            }
        },
        'mean_reversion': {
            'name': 'Mean Reversion Strategy',
            'description': 'Trades mean reversion using Bollinger Bands',
            'category': 'mean_reversion',
            'symbols': ['SPY', 'IWM'],
            'timeframes': ['1d'],
            'parameters': {
                'entry_threshold': 2.0,
                'profit_target_pct': 1.5,
                'stop_loss_pct': 1.0
            }
        },
        'breakout_strategy': {
            'name': 'Breakout Strategy',
            'description': 'Trades range breakouts',
            'category': 'breakout',
            'symbols': ['QQQ', 'IWM'],
            'timeframes': ['1d'],
            'parameters': {
                'breakout_threshold': 2.0,
                'confirmation_period': 3,
                'trailing_stop_pct': 2.0
            }
        },
        'volatility_strategy': {
            'name': 'Volatility Strategy',
            'description': 'Trades volatility expansion and contraction',
            'category': 'volatility',
            'symbols': ['VXX', 'SPY'],
            'timeframes': ['1d'],
            'parameters': {
                'vix_threshold': 20,
                'position_size_scale': 0.8,
                'profit_target_mult': 1.3
            }
        }
    }
    
    # Register strategies
    for strategy_id, metadata in strategies.items():
        controller.register_strategy(strategy_id, metadata)
    
    logger.info(f"Registered {len(strategies)} strategies")
    
    # Generate and update sample market data for different regimes
    market_data = {
        'SPY': generate_sample_data('SPY', days=200, regime='trending_up'),
        'QQQ': generate_sample_data('QQQ', days=200, regime='breakout'),
        'IWM': generate_sample_data('IWM', days=200, regime='ranging'),
        'VXX': generate_sample_data('VXX', days=200, regime='volatile')
    }
    
    # Update market data in controller
    for symbol, data in market_data.items():
        controller.update_market_data(symbol, data)
    
    logger.info(f"Updated market data for {len(market_data)} symbols")
    
    # Display market regimes
    regimes = controller.get_market_regimes()
    logger.info("\nCurrent Market Regimes:")
    for symbol, regime_data in regimes.items():
        logger.info(f"{symbol}: {regime_data['regime']} (Confidence: {regime_data['confidence']:.2f}, Strength: {regime_data['strength']:.2f})")
    
    # Simulate trades with different success rates
    simulate_trades(controller, 'trend_following', 'SPY', num_trades=30, win_rate=0.7)
    simulate_trades(controller, 'mean_reversion', 'IWM', num_trades=25, win_rate=0.5)
    simulate_trades(controller, 'breakout_strategy', 'QQQ', num_trades=20, win_rate=0.6)
    simulate_trades(controller, 'volatility_strategy', 'VXX', num_trades=15, win_rate=0.4)
    
    # Update equity to trigger allocation recalculation
    controller.update_equity(12500.0)
    
    # Display allocations
    allocations = controller.get_all_allocations()
    logger.info("\nStrategy Allocations:")
    for strategy_id, allocation in allocations.items():
        logger.info(f"{strategy_id}: {allocation:.2%}")
    
    # Display optimal parameters for different strategies in different regimes
    logger.info("\nOptimal Strategy Parameters:")
    for strategy_id, metadata in strategies.items():
        symbol = metadata['symbols'][0]
        params = controller.get_strategy_parameters(strategy_id, symbol)
        logger.info(f"{strategy_id} for {symbol}: {json.dumps(params, indent=2)}")
    
    # Calculate position sizes
    logger.info("\nSample Position Sizes:")
    for strategy_id, metadata in strategies.items():
        symbol = metadata['symbols'][0]
        position_info = controller.get_position_size(
            strategy_id=strategy_id,
            symbol=symbol,
            entry_price=100.0,
            stop_loss=98.0
        )
        logger.info(f"{strategy_id} for {symbol}: {position_info['size']:.2f} shares (${position_info['notional']:.2f})")
    
    # Plot allocations
    plot_allocations(controller)
    
    # Plot regime changes over time for SPY
    spy_history = controller.market_regime_detector.get_regime_history('SPY', days=200)
    if spy_history:
        plot_regime_comparison('SPY', spy_history)
    
    logger.info("Adaptive Strategy Controller Demo completed successfully")


if __name__ == "__main__":
    main()
