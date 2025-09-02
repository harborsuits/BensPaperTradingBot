#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Strategies Integration and Performance Test

This script tests the integration of our forex strategies with the autonomous
rotation system and benchmarks their performance in different market conditions.

It focuses on:
1. Strategy signal generation
2. Market regime compatibility scoring
3. Strategy rotation based on market conditions
4. Performance metrics comparison
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import strategy-related classes directly
from trading_bot.strategies.strategy_template import Signal, SignalType, MarketRegime, TimeFrame
from trading_bot.strategies.forex.trend_following_strategy import ForexTrendFollowingStrategy
from trading_bot.strategies.forex.range_trading_strategy import ForexRangeTradingStrategy
from trading_bot.strategies.forex.breakout_strategy import ForexBreakoutStrategy

# Create a mock event handler system
class MockEventSystem:
    def __init__(self):
        self.events = []
        
    def publish(self, event):
        self.events.append(event)
        logger.info(f"Event published: {event.event_type}")

# Market data generator for different regimes
def generate_market_data(market_regime: MarketRegime, days: int = 100) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic market data for a specific market regime.
    
    Args:
        market_regime: The market regime to generate data for
        days: Number of days of data to generate
        
    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV data
    """
    base_price = 1.2000
    dates = pd.date_range(end=datetime.now(), periods=days)
    
    # Generate data based on market regime
    if market_regime == MarketRegime.BULL_TREND:
        # Steady uptrend
        daily_return = 0.0010  # 0.1% daily return
        volatility = 0.003
        trend = np.exp(np.linspace(0, days * daily_return, days))
        noise = np.random.normal(0, volatility, days)
        close = base_price * trend * (1 + noise)
        
    elif market_regime == MarketRegime.BEAR_TREND:
        # Steady downtrend
        daily_return = -0.0010  # -0.1% daily return
        volatility = 0.003
        trend = np.exp(np.linspace(0, days * daily_return, days))
        noise = np.random.normal(0, volatility, days)
        close = base_price * trend * (1 + noise)
        
    elif market_regime == MarketRegime.CONSOLIDATION:
        # Sideways with support and resistance
        volatility = 0.0015
        range_amplitude = 0.02  # 2% range
        close = base_price * (1 + range_amplitude * np.sin(np.linspace(0, 3*np.pi, days)))
        close = close + np.random.normal(0, volatility, days)
        
    elif market_regime == MarketRegime.HIGH_VOLATILITY:
        # High volatility with breakouts
        volatility = 0.010
        close = base_price + np.random.normal(0, volatility, days).cumsum() * 0.001
        
        # Add some sharp breakouts
        breakout_points = np.random.choice(range(10, days-10), 3, replace=False)
        for point in breakout_points:
            direction = np.random.choice([-1, 1])
            breakout_size = 0.02 * direction  # 2% breakout
            close[point:] = close[point:] * (1 + breakout_size)
            
    elif market_regime == MarketRegime.LOW_VOLATILITY:
        # Very tight range
        volatility = 0.0008
        range_amplitude = 0.005  # 0.5% range
        close = base_price * (1 + range_amplitude * np.sin(np.linspace(0, 4*np.pi, days)))
        close = close + np.random.normal(0, volatility, days)
    
    else:  # Default to mixed regime
        # Mix of trends and ranges
        volatility = 0.005
        close = base_price + np.random.normal(0, volatility, days).cumsum() * 0.0005
    
    # Create OHLCV data
    high = close + np.random.uniform(0.0005, 0.002, days)
    low = close - np.random.uniform(0.0005, 0.002, days)
    open_price = close.copy()
    open_price[1:] = close[:-1]  # Shift close prices by 1 to get open
    volume = np.random.uniform(1000, 5000, days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Create multiple pairs with slight variations
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    data = {}
    
    for symbol in symbols:
        # Add some random variation to each pair
        variation = np.random.uniform(0.98, 1.02)
        symbol_df = df.copy()
        symbol_df[['open', 'high', 'low', 'close']] *= variation
        data[symbol] = symbol_df
    
    return data

# Performance tracking class
class StrategyPerformance:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.equity_curve = [1000.0]  # Start with $1000
        self.returns = []
        self.win_count = 0
        self.loss_count = 0
        self.positions = {}  # Currently open positions
        self.signals = []    # Signal history
        
    def apply_signal(self, symbol: str, signal: Signal, current_price: float):
        """Record and apply a trading signal"""
        self.signals.append({
            'symbol': symbol,
            'type': signal.signal_type,
            'confidence': signal.confidence,
            'timestamp': signal.timestamp,
            'price': current_price
        })
        
        # If we have an open position for this symbol, close it
        if symbol in self.positions:
            self.close_position(symbol, current_price)
            
        # Open new position if signal is not FLAT
        if signal.signal_type != SignalType.FLAT:
            self.positions[symbol] = {
                'type': signal.signal_type,
                'entry_price': current_price,
                'stop_loss': signal.metadata.get('stop_loss', current_price * 0.99 if signal.signal_type == SignalType.LONG else current_price * 1.01),
                'take_profit': signal.metadata.get('take_profit', current_price * 1.01 if signal.signal_type == SignalType.LONG else current_price * 0.99),
                'size': 100  # Fixed position size for simplicity
            }
    
    def update(self, current_data: Dict[str, pd.DataFrame]):
        """Update performance based on current prices"""
        # Get latest equity
        current_equity = self.equity_curve[-1]
        pnl = 0.0
        
        # Check all open positions
        closed_symbols = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close'].iloc[-1]
            
            # Calculate unrealized P&L
            if position['type'] == SignalType.LONG:
                position_pnl = (current_price - position['entry_price']) * position['size']
                
                # Check if stop loss or take profit hit
                if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                    closed_symbols.append(symbol)
                    pnl += position_pnl
                    
                    if position_pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                        
            elif position['type'] == SignalType.SHORT:
                position_pnl = (position['entry_price'] - current_price) * position['size']
                
                # Check if stop loss or take profit hit
                if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                    closed_symbols.append(symbol)
                    pnl += position_pnl
                    
                    if position_pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
        
        # Remove closed positions
        for symbol in closed_symbols:
            del self.positions[symbol]
        
        # Update equity curve
        new_equity = current_equity + pnl
        self.equity_curve.append(new_equity)
        
        # Calculate return
        if len(self.equity_curve) > 1:
            daily_return = (new_equity - current_equity) / current_equity
            self.returns.append(daily_return)
    
    def close_position(self, symbol: str, current_price: float):
        """Close a specific position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Calculate P&L
        if position['type'] == SignalType.LONG:
            position_pnl = (current_price - position['entry_price']) * position['size']
        else:
            position_pnl = (position['entry_price'] - current_price) * position['size']
            
        # Update counters
        if position_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        # Update equity
        self.equity_curve[-1] += position_pnl
        
        # Remove position
        del self.positions[symbol]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        if len(self.returns) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            }
            
        # Calculate metrics
        total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
        
        # Annualized Sharpe Ratio (assuming daily data)
        if np.std(self.returns) > 0:
            sharpe_ratio = np.mean(self.returns) / np.std(self.returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        # Win rate
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0.0
        
        # Maximum drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0.0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }

# Main test function
def test_forex_strategy_integration():
    """Test forex strategy integration and performance across market regimes"""
    logger.info("Starting Forex Strategy Integration Test")
    
    # Initialize strategies
    trend_strategy = ForexTrendFollowingStrategy()
    range_strategy = ForexRangeTradingStrategy()
    breakout_strategy = ForexBreakoutStrategy()
    
    # Create mock event bus
    mock_event_bus = MockEventSystem()
    
    # Assign to strategies
    trend_strategy.event_bus = mock_event_bus
    range_strategy.event_bus = mock_event_bus
    breakout_strategy.event_bus = mock_event_bus
    
    # Create performance trackers
    trend_performance = StrategyPerformance("Trend Following")
    range_performance = StrategyPerformance("Range Trading")
    breakout_performance = StrategyPerformance("Breakout")
    rotation_performance = StrategyPerformance("Autonomous Rotation")
    
    # Test across different market regimes
    regimes = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.CONSOLIDATION,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY,
    ]
    
    # Store performance by regime
    performance_by_regime = {}
    
    # For each regime, run a simulation
    for regime in regimes:
        logger.info(f"\n\n===== Testing Strategies in {regime.name} Regime =====")
        
        # Generate data for this regime
        market_data = generate_market_data(regime, days=60)
        
        # Reset performance trackers for this regime
        regime_trend_performance = StrategyPerformance("Trend Following")
        regime_range_performance = StrategyPerformance("Range Trading")
        regime_breakout_performance = StrategyPerformance("Breakout")
        regime_rotation_performance = StrategyPerformance("Autonomous Rotation")
        
        # Get compatibility scores for rotation
        trend_score = trend_strategy.get_compatibility_score(regime)
        range_score = range_strategy.get_compatibility_score(regime)
        breakout_score = breakout_strategy.get_compatibility_score(regime)
        
        logger.info(f"Compatibility Scores:")
        logger.info(f"  Trend Following: {trend_score:.2f}")
        logger.info(f"  Range Trading:   {range_score:.2f}")
        logger.info(f"  Breakout:        {breakout_score:.2f}")
        
        # Determine best strategy for rotation
        best_strategy = max([
            (trend_score, "trend"),
            (range_score, "range"),
            (breakout_score, "breakout")
        ])[1]
        
        logger.info(f"Best strategy for {regime.name}: {best_strategy}\n")
        
        # Simulate day by day
        for i in range(10, len(market_data['EURUSD'])):
            current_time = market_data['EURUSD'].index[i]
            
            # Slice data up to current point
            current_data = {
                symbol: df.iloc[:i+1] for symbol, df in market_data.items()
            }
            
            # Get signals from each strategy
            trend_signals = trend_strategy.generate_signals(current_data, current_time)
            range_signals = range_strategy.generate_signals(current_data, current_time)
            breakout_signals = breakout_strategy.generate_signals(current_data, current_time)
            
            # Apply signals to performance trackers
            for symbol in current_data.keys():
                current_price = current_data[symbol]['close'].iloc[-1]
                
                # Trend strategy
                if symbol in trend_signals and trend_signals[symbol] is not None:
                    regime_trend_performance.apply_signal(symbol, trend_signals[symbol], current_price)
                
                # Range strategy
                if symbol in range_signals and range_signals[symbol] is not None:
                    regime_range_performance.apply_signal(symbol, range_signals[symbol], current_price)
                
                # Breakout strategy
                if symbol in breakout_signals and breakout_signals[symbol] is not None:
                    regime_breakout_performance.apply_signal(symbol, breakout_signals[symbol], current_price)
                
                # Rotation - apply signal from best strategy
                if best_strategy == "trend" and symbol in trend_signals and trend_signals[symbol] is not None:
                    regime_rotation_performance.apply_signal(symbol, trend_signals[symbol], current_price)
                elif best_strategy == "range" and symbol in range_signals and range_signals[symbol] is not None:
                    regime_rotation_performance.apply_signal(symbol, range_signals[symbol], current_price)
                elif best_strategy == "breakout" and symbol in breakout_signals and breakout_signals[symbol] is not None:
                    regime_rotation_performance.apply_signal(symbol, breakout_signals[symbol], current_price)
            
            # Update performance with new prices
            regime_trend_performance.update(current_data)
            regime_range_performance.update(current_data)
            regime_breakout_performance.update(current_data)
            regime_rotation_performance.update(current_data)
        
        # Calculate final metrics for this regime
        trend_metrics = regime_trend_performance.get_performance_metrics()
        range_metrics = regime_range_performance.get_performance_metrics()
        breakout_metrics = regime_breakout_performance.get_performance_metrics()
        rotation_metrics = regime_rotation_performance.get_performance_metrics()
        
        # Display results
        logger.info(f"Performance in {regime.name} Regime:")
        logger.info(f"  Trend Following: Return: {trend_metrics['total_return']:.2%}, Sharpe: {trend_metrics['sharpe_ratio']:.2f}, Win Rate: {trend_metrics['win_rate']:.2%}")
        logger.info(f"  Range Trading:   Return: {range_metrics['total_return']:.2%}, Sharpe: {range_metrics['sharpe_ratio']:.2f}, Win Rate: {range_metrics['win_rate']:.2%}")
        logger.info(f"  Breakout:        Return: {breakout_metrics['total_return']:.2%}, Sharpe: {breakout_metrics['sharpe_ratio']:.2f}, Win Rate: {breakout_metrics['win_rate']:.2%}")
        logger.info(f"  Rotation System: Return: {rotation_metrics['total_return']:.2%}, Sharpe: {rotation_metrics['sharpe_ratio']:.2f}, Win Rate: {rotation_metrics['win_rate']:.2%}\n")
        
        # Store results
        performance_by_regime[regime.name] = {
            'trend': trend_metrics,
            'range': range_metrics,
            'breakout': breakout_metrics,
            'rotation': rotation_metrics
        }
        
        # Accumulate results for overall performance
        for equity in regime_trend_performance.equity_curve[1:]:
            trend_performance.equity_curve.append(trend_performance.equity_curve[-1] * equity / regime_trend_performance.equity_curve[0])
        
        for equity in regime_range_performance.equity_curve[1:]:
            range_performance.equity_curve.append(range_performance.equity_curve[-1] * equity / regime_range_performance.equity_curve[0])
        
        for equity in regime_breakout_performance.equity_curve[1:]:
            breakout_performance.equity_curve.append(breakout_performance.equity_curve[-1] * equity / regime_breakout_performance.equity_curve[0])
        
        for equity in regime_rotation_performance.equity_curve[1:]:
            rotation_performance.equity_curve.append(rotation_performance.equity_curve[-1] * equity / regime_rotation_performance.equity_curve[0])
    
    # Calculate overall metrics
    overall_trend_return = (trend_performance.equity_curve[-1] - trend_performance.equity_curve[0]) / trend_performance.equity_curve[0]
    overall_range_return = (range_performance.equity_curve[-1] - range_performance.equity_curve[0]) / range_performance.equity_curve[0]
    overall_breakout_return = (breakout_performance.equity_curve[-1] - breakout_performance.equity_curve[0]) / breakout_performance.equity_curve[0]
    overall_rotation_return = (rotation_performance.equity_curve[-1] - rotation_performance.equity_curve[0]) / rotation_performance.equity_curve[0]
    
    # Display overall results
    logger.info("===== OVERALL PERFORMANCE ACROSS ALL REGIMES =====")
    logger.info(f"Trend Following: {overall_trend_return:.2%}")
    logger.info(f"Range Trading:   {overall_range_return:.2%}")
    logger.info(f"Breakout:        {overall_breakout_return:.2%}")
    logger.info(f"Rotation System: {overall_rotation_return:.2%}")
    
    # Plot equity curves
    plt.figure(figsize=(12, 8))
    plt.plot(trend_performance.equity_curve, label='Trend Following')
    plt.plot(range_performance.equity_curve, label='Range Trading')
    plt.plot(breakout_performance.equity_curve, label='Breakout')
    plt.plot(rotation_performance.equity_curve, label='Autonomous Rotation', linewidth=2)
    plt.title('Strategy Performance Across All Market Regimes')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('forex_strategy_performance.png')
    
    # Create performance table by regime
    performance_table = {
        'Strategy': ['Trend Following', 'Range Trading', 'Breakout', 'Rotation'],
    }
    
    for regime in regimes:
        regime_name = regime.name
        performance_table[regime_name] = [
            f"{performance_by_regime[regime_name]['trend']['total_return']:.2%}",
            f"{performance_by_regime[regime_name]['range']['total_return']:.2%}",
            f"{performance_by_regime[regime_name]['breakout']['total_return']:.2%}",
            f"{performance_by_regime[regime_name]['rotation']['total_return']:.2%}"
        ]
    
    performance_table['OVERALL'] = [
        f"{overall_trend_return:.2%}",
        f"{overall_range_return:.2%}",
        f"{overall_breakout_return:.2%}",
        f"{overall_rotation_return:.2%}"
    ]
    
    # Convert to DataFrame and save
    performance_df = pd.DataFrame(performance_table)
    performance_df.to_csv('forex_strategy_performance.csv', index=False)
    
    logger.info("\nPerformance results saved to forex_strategy_performance.csv")
    logger.info("Performance chart saved to forex_strategy_performance.png")

if __name__ == "__main__":
    test_forex_strategy_integration()
