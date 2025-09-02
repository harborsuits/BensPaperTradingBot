#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Comparison Test: Contextual vs Static

This script compares the performance of a trading strategy with and without 
contextual awareness to demonstrate the benefits of adaptive decision-making.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComparisonTest")

# Import necessary components
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

# Simple trade simulation
class TradeSimulator:
    """Simple trade simulator for backtesting."""
    
    def __init__(self, data, initial_balance=5000.0):
        """
        Initialize the trade simulator.
        
        Args:
            data: Dictionary of symbol -> DataFrame with market data
            initial_balance: Starting account balance
        """
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = {}
    
    def reset(self):
        """Reset the simulator to initial state."""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = {}
    
    def execute_trade(self, 
                    symbol: str, 
                    entry_time: datetime, 
                    direction: str, 
                    position_size: float, 
                    risk_amount: float, 
                    stop_loss_pips: float = 20,
                    take_profit_pips: float = 40,
                    max_bars: int = 100,
                    strategy_id: str = None,
                    strategy_name: str = None):
        """
        Execute a simulated trade.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            direction: 'buy' or 'sell'
            position_size: Position size in lots
            risk_amount: Amount risked on the trade
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            max_bars: Maximum bars to hold the trade
            strategy_id: Strategy ID
            strategy_name: Strategy name
            
        Returns:
            Trade result dictionary
        """
        if symbol not in self.data:
            logger.warning(f"Symbol {symbol} not found in data")
            return None
        
        # Find entry bar
        df = self.data[symbol]
        entry_idx = -1
        
        for i, time_idx in enumerate(df.index):
            if time_idx >= entry_time:
                entry_idx = i
                break
        
        if entry_idx == -1 or entry_idx >= len(df) - 1:
            logger.warning(f"Entry time {entry_time} not found in data")
            return None
        
        # Entry price
        entry_price = df.iloc[entry_idx]['close']
        
        # Calculate pip value (simplified for major forex pairs)
        pip_value = 0.0001  # Standard pip for forex
        
        # Calculate stop loss and take profit levels
        if direction == 'buy':
            stop_loss = entry_price - (stop_loss_pips * pip_value)
            take_profit = entry_price + (take_profit_pips * pip_value)
        else:  # 'sell'
            stop_loss = entry_price + (stop_loss_pips * pip_value)
            take_profit = entry_price - (take_profit_pips * pip_value)
        
        # Track trade through subsequent bars
        exit_idx = -1
        exit_price = entry_price
        exit_reason = 'max_bars'
        
        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            bar = df.iloc[i]
            
            # Check for stop loss
            if direction == 'buy':
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_idx = i
                    exit_reason = 'stop_loss'
                    break
            else:  # 'sell'
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_idx = i
                    exit_reason = 'stop_loss'
                    break
            
            # Check for take profit
            if direction == 'buy':
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
            else:  # 'sell'
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
        
        # If we haven't exited yet, exit at the last processed bar
        if exit_idx == -1:
            exit_idx = min(entry_idx + max_bars, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
        
        # Calculate profit/loss
        if direction == 'buy':
            pips_gained = (exit_price - entry_price) / pip_value
        else:  # 'sell'
            pips_gained = (entry_price - exit_price) / pip_value
        
        # Calculate PnL
        if exit_reason == 'stop_loss':
            pnl = -risk_amount
        elif exit_reason == 'take_profit':
            pnl = risk_amount * (take_profit_pips / stop_loss_pips)
        else:
            # Calculate based on actual pips gained/lost
            risk_per_pip = risk_amount / stop_loss_pips
            pnl = pips_gained * risk_per_pip
        
        # Update balance
        self.balance += pnl
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': df.index[exit_idx],
            'exit_price': exit_price,
            'direction': direction,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': (risk_amount / (self.balance - pnl)) * 100,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'pnl': pnl,
            'pips': pips_gained,
            'exit_reason': exit_reason,
            'bars_held': exit_idx - entry_idx,
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'balance_after': self.balance
        }
        
        # Add to trade history
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append({
            'time': df.index[exit_idx],
            'balance': self.balance
        })
        
        return trade
    
    def get_results(self):
        """Get performance results."""
        if not self.trades:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Calculate performance metrics
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        losses = sum(1 for t in self.trades if t['pnl'] <= 0)
        
        profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        loss = sum(abs(t['pnl']) for t in self.trades if t['pnl'] <= 0)
        
        win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0
        profit_factor = profit / loss if loss > 0 else float('inf')
        
        # Calculate max drawdown
        peak = self.initial_balance
        drawdown = 0
        max_drawdown = 0
        
        for trade in self.trades:
            if trade['balance_after'] > peak:
                peak = trade['balance_after']
            else:
                drawdown = (peak - trade['balance_after']) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': (self.balance / self.initial_balance - 1) * 100,
            'total_trades': len(self.trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

class ContextualStrategy:
    """Strategy that adapts to market regimes."""
    
    def __init__(self, event_bus, account_balance=5000.0):
        """
        Initialize the contextual strategy.
        
        Args:
            event_bus: Event bus for communication
            account_balance: Initial account balance
        """
        self.event_bus = event_bus
        self.account_balance = account_balance
        
        # Current context
        self.current_context = {
            'market_regime': 'unknown',
            'volatility_state': 'medium'
        }
        
        # Strategy selection based on regime
        self.regime_strategies = {
            'trending_up': {'id': 'trend_following', 'name': 'Trend Following'},
            'trending_down': {'id': 'downtrend', 'name': 'Downtrend Strategy'},
            'ranging': {'id': 'mean_reversion', 'name': 'Mean Reversion'},
            'breakout': {'id': 'breakout', 'name': 'Breakout Strategy'},
            'unknown': {'id': 'balanced', 'name': 'Balanced Strategy'}
        }
        
        # Subscribe to context events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self.handle_regime_change)
        self.event_bus.subscribe(EventType.VOLATILITY_UPDATE, self.handle_volatility_update)
    
    def handle_regime_change(self, event):
        """Handle market regime change events."""
        regime = event.data.get('regime', 'unknown')
        self.current_context['market_regime'] = regime
    
    def handle_volatility_update(self, event):
        """Handle volatility update events."""
        volatility = event.data.get('volatility_state', 'medium')
        self.current_context['volatility_state'] = volatility
    
    def select_strategy(self, symbol, market_data):
        """Select strategy based on current context."""
        regime = self.current_context.get('market_regime', 'unknown')
        return self.regime_strategies.get(regime, self.regime_strategies['unknown'])
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_pips):
        """Calculate position size based on account balance and context."""
        # Starting risk table
        risk_table = {
            500: 0.95,    # 95% risk up to $500
            1000: 0.90,   # 90% risk up to $1,000
            2500: 0.85,   # 85% risk up to $2,500
            5000: 0.80,   # 80% risk up to $5,000
            7000: 0.75,   # 75% risk up to $7,000
            10000: 0.65,  # 65% risk up to $10,000
            15000: 0.55,  # 55% risk up to $15,000
            20000: 0.45,  # 45% risk up to $20,000
            24999: 0.35,  # 35% risk up to $24,999
            25000: 0.15,  # 15% risk at $25,000 (PDT threshold)
            35000: 0.12,  # 12% risk up to $35,000
            50000: 0.10,  # 10% risk up to $50,000
            100000: 0.08, # 8% risk up to $100,000
            250000: 0.06, # 6% risk up to $250,000
            500000: 0.04, # 4% risk up to $500,000
            1000000: 0.02 # 2% risk above $1,000,000
        }
        
        # Determine risk percentage
        risk_percentage = 0.02  # Default minimum
        
        for threshold, risk in sorted(risk_table.items()):
            if self.account_balance <= threshold:
                risk_percentage = risk
                break
        
        # Adjust for volatility
        volatility = self.current_context.get('volatility_state', 'medium')
        if volatility == 'high':
            risk_percentage *= 0.8  # Reduce risk in high volatility
        elif volatility == 'low':
            risk_percentage *= 1.1  # Slightly increase risk in low volatility
        
        # Calculate risk amount
        risk_amount = self.account_balance * risk_percentage
        
        # Calculate pip value (simplified for major forex pairs)
        pip_value = 0.0001
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
        
        return {
            'position_size': position_size,
            'risk_percentage': risk_percentage * 100,
            'risk_amount': risk_amount
        }
    
    def update_balance(self, new_balance):
        """Update account balance."""
        self.account_balance = new_balance

class StaticStrategy:
    """Strategy with fixed parameters."""
    
    def __init__(self, account_balance=5000.0):
        """
        Initialize the static strategy.
        
        Args:
            account_balance: Initial account balance
        """
        self.account_balance = account_balance
        
        # Fixed strategy
        self.static_strategy = {'id': 'balanced', 'name': 'Balanced Strategy'}
        
        # Fixed risk percentage (2% of account)
        self.risk_percentage = 0.02
    
    def select_strategy(self, symbol, market_data):
        """Return fixed strategy."""
        return self.static_strategy
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_pips):
        """Calculate position size using fixed risk percentage."""
        # Calculate risk amount
        risk_amount = self.account_balance * self.risk_percentage
        
        # Calculate pip value
        pip_value = 0.0001
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
        
        return {
            'position_size': position_size,
            'risk_percentage': self.risk_percentage * 100,
            'risk_amount': risk_amount
        }
    
    def update_balance(self, new_balance):
        """Update account balance."""
        self.account_balance = new_balance

def run_comparison_test():
    """Compare contextual vs static strategies."""
    # Initialize event bus
    event_bus = EventBus()
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data/market_data'):
        os.makedirs('data/market_data', exist_ok=True)
    
    # Test symbols
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Load or generate test data
    market_data = {}
    for symbol in symbols:
        file_path = f'data/market_data/{symbol}_1h.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            market_data[symbol] = df
    
    # Check if we have data
    if not market_data:
        logger.error("No market data found. Please run regime_detector_test.py first.")
        return None
    
    # Initialize strategies
    contextual_strategy = ContextualStrategy(event_bus)
    static_strategy = StaticStrategy()
    
    # Initialize trade simulators
    contextual_simulator = TradeSimulator(market_data)
    static_simulator = TradeSimulator(market_data)
    
    # Run simulation
    logger.info("Starting strategy comparison test...")
    
    # Define market regimes for testing
    # These would normally come from the MarketRegimeDetector
    regime_periods = [
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[0], 
         'end': market_data['EURUSD'].index[len(market_data['EURUSD'])//4], 'regime': 'ranging'},
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[len(market_data['EURUSD'])//4], 
         'end': market_data['EURUSD'].index[len(market_data['EURUSD'])//2], 'regime': 'trending_up'},
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[len(market_data['EURUSD'])//2], 
         'end': market_data['EURUSD'].index[3*len(market_data['EURUSD'])//4], 'regime': 'ranging'},
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[3*len(market_data['EURUSD'])//4], 
         'end': market_data['EURUSD'].index[-1], 'regime': 'trending_down'}
    ]
    
    # Simulate each regime
    for regime_period in regime_periods:
        symbol = regime_period['symbol']
        start = regime_period['start']
        end = regime_period['end']
        regime = regime_period['regime']
        
        logger.info(f"Processing {regime} regime from {start} to {end} for {symbol}")
        
        # Publish regime change event
        event_bus.publish(Event(
            event_type=EventType.MARKET_REGIME_CHANGE,
            data={
                'symbol': symbol,
                'regime': regime,
                'previous_regime': 'unknown',
                'confidence': 0.85
            }
        ))
        
        # Process each trading day during this regime
        df = market_data[symbol]
        current_idx = df.index.get_indexer([start], method='nearest')[0]
        end_idx = df.index.get_indexer([end], method='nearest')[0]
        
        while current_idx < end_idx:
            # Current time
            current_time = df.index[current_idx]
            
            # Simulate volatility update occasionally
            if current_idx % 24 == 0:  # Once per day
                # Determine volatility based on regime
                if regime == 'ranging' and current_idx >= len(df)//2:
                    volatility = 'high'
                elif regime == 'trending_up':
                    volatility = 'medium'
                elif regime == 'trending_down':
                    volatility = 'high' if current_idx % 48 == 0 else 'medium'
                else:
                    volatility = 'medium'
                
                # Publish volatility update
                event_bus.publish(Event(
                    event_type=EventType.VOLATILITY_UPDATE,
                    data={
                        'symbol': symbol,
                        'volatility_state': volatility,
                        'previous_state': 'medium'
                    }
                ))
            
            # Every 12 hours, make a trading decision
            if current_idx % 12 == 0:
                # Get current price data
                price = df.iloc[current_idx]['close']
                
                # Make trading decisions
                
                # 1. Contextual strategy
                context_strategy = contextual_strategy.select_strategy(symbol, None)
                context_position = contextual_strategy.calculate_position_size(symbol, price, 20)
                
                # Execute contextual trade
                contextual_trade = contextual_simulator.execute_trade(
                    symbol=symbol,
                    entry_time=current_time,
                    direction='buy' if regime in ['trending_up', 'breakout'] else 'sell' if regime == 'trending_down' else 
                              ('buy' if np.random.random() > 0.5 else 'sell'),
                    position_size=context_position['position_size'],
                    risk_amount=context_position['risk_amount'],
                    stop_loss_pips=20,
                    take_profit_pips=40 if regime in ['trending_up', 'trending_down'] else 30,
                    strategy_id=context_strategy['id'],
                    strategy_name=context_strategy['name']
                )
                
                if contextual_trade:
                    # Update balance
                    contextual_strategy.update_balance(contextual_simulator.balance)
                
                # 2. Static strategy
                static_strategy_info = static_strategy.select_strategy(symbol, None)
                static_position = static_strategy.calculate_position_size(symbol, price, 20)
                
                # Execute static trade
                static_trade = static_simulator.execute_trade(
                    symbol=symbol,
                    entry_time=current_time,
                    direction='buy' if np.random.random() > 0.5 else 'sell',  # Random direction
                    position_size=static_position['position_size'],
                    risk_amount=static_position['risk_amount'],
                    stop_loss_pips=20,
                    take_profit_pips=40,
                    strategy_id=static_strategy_info['id'],
                    strategy_name=static_strategy_info['name']
                )
                
                if static_trade:
                    # Update balance
                    static_strategy.update_balance(static_simulator.balance)
            
            # Move to next bar
            current_idx += 1
    
    # Get results
    contextual_results = contextual_simulator.get_results()
    static_results = static_simulator.get_results()
    
    # Log results
    logger.info("\n=== STRATEGY COMPARISON RESULTS ===")
    logger.info("\nContextual Strategy:")
    logger.info(f"Initial Balance: ${contextual_results['initial_balance']:.2f}")
    logger.info(f"Final Balance: ${contextual_results['final_balance']:.2f}")
    logger.info(f"Total Return: {contextual_results['total_return']:.2f}%")
    logger.info(f"Win Rate: {contextual_results['win_rate']:.2f}%")
    logger.info(f"Profit Factor: {contextual_results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {contextual_results['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {contextual_results['total_trades']}")
    
    logger.info("\nStatic Strategy:")
    logger.info(f"Initial Balance: ${static_results['initial_balance']:.2f}")
    logger.info(f"Final Balance: ${static_results['final_balance']:.2f}")
    logger.info(f"Total Return: {static_results['total_return']:.2f}%")
    logger.info(f"Win Rate: {static_results['win_rate']:.2f}%")
    logger.info(f"Profit Factor: {static_results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {static_results['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {static_results['total_trades']}")
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    
    # Contextual equity curve
    contextual_equity = pd.DataFrame(contextual_results['equity_curve'])
    if not contextual_equity.empty:
        plt.plot(contextual_equity['time'], contextual_equity['balance'], 
                 label='Contextual Strategy', color='blue')
    
    # Static equity curve
    static_equity = pd.DataFrame(static_results['equity_curve'])
    if not static_equity.empty:
        plt.plot(static_equity['time'], static_equity['balance'], 
                 label='Static Strategy', color='red')
    
    plt.title('Equity Curve Comparison: Contextual vs Static Strategy')
    plt.xlabel('Time')
    plt.ylabel('Account Balance ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()
    
    logger.info(f"Comparison chart saved to strategy_comparison.png")
    
    return {
        'contextual': contextual_results,
        'static': static_results
    }

if __name__ == "__main__":
    run_comparison_test()
