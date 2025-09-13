#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio State Manager

This module manages the current state of the trading portfolio, including positions,
performance metrics, and trade history.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioStateManager:
    """
    Manages the current state of a trading portfolio
    """
    
    def __init__(self, initial_cash: float = 100000.0, state_file: Optional[str] = None):
        """
        Initialize the portfolio state manager.
        
        Args:
            initial_cash: Initial cash amount
            state_file: Optional path to a JSON file with saved state
        """
        # Initialize with default state
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.positions = {}  # Symbol -> Position data
        self.trades = []  # List of trade records
        self.performance_history = {
            'timestamp': [],
            'portfolio_value': [],
            'cash': []
        }
        self.strategy_allocations = {}  # Strategy name -> allocation percentage
        self.strategy_performance = {}  # Strategy name -> performance metrics
        
        # Load state from file if provided
        if state_file and os.path.exists(state_file):
            self._load_state(state_file)
        else:
            # Record initial state
            self._record_performance_snapshot()
        
        logger.info(f"Portfolio state manager initialized with {self.current_cash:.2f} cash")
    
    def _load_state(self, state_file: str) -> None:
        """
        Load portfolio state from a JSON file.
        
        Args:
            state_file: Path to JSON state file
        """
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Load basic state
            self.initial_cash = state.get('initial_cash', self.initial_cash)
            self.current_cash = state.get('current_cash', self.initial_cash)
            self.positions = state.get('positions', {})
            self.trades = state.get('trades', [])
            self.strategy_allocations = state.get('strategy_allocations', {})
            self.strategy_performance = state.get('strategy_performance', {})
            
            # Convert timestamp strings to datetime for performance history
            history = state.get('performance_history', {})
            self.performance_history = {
                'timestamp': [datetime.fromisoformat(ts) for ts in history.get('timestamp', [])],
                'portfolio_value': history.get('portfolio_value', []),
                'cash': history.get('cash', [])
            }
            
            logger.info(f"Loaded portfolio state from {state_file}")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            # Initialize with default state
            self.current_cash = self.initial_cash
            self._record_performance_snapshot()
    
    def save_state(self, state_file: str) -> None:
        """
        Save current portfolio state to a JSON file.
        
        Args:
            state_file: Path to save the state
        """
        try:
            # Convert datetime objects to ISO format strings
            history = {
                'timestamp': [ts.isoformat() for ts in self.performance_history['timestamp']],
                'portfolio_value': self.performance_history['portfolio_value'],
                'cash': self.performance_history['cash']
            }
            
            state = {
                'initial_cash': self.initial_cash,
                'current_cash': self.current_cash,
                'positions': self.positions,
                'trades': self.trades,
                'performance_history': history,
                'strategy_allocations': self.strategy_allocations,
                'strategy_performance': self.strategy_performance,
                'last_updated': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved portfolio state to {state_file}")
        except Exception as e:
            logger.error(f"Error saving portfolio state: {e}")
    
    def update_position(self, symbol: str, quantity: int, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update a position by adding shares at the specified price.
        
        Args:
            symbol: The ticker symbol
            quantity: Number of shares (negative for selling)
            price: Price per share
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update cash
        trade_value = quantity * price
        self.current_cash -= trade_value
        
        # Update position
        if symbol in self.positions:
            current_position = self.positions[symbol]
            current_quantity = current_position['quantity']
            current_value = current_position['current_value']
            
            # Calculate new position details
            new_quantity = current_quantity + quantity
            
            if new_quantity == 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Update position with weighted average price
                if quantity > 0:  # Buying more
                    current_cost_basis = current_position['avg_price'] * current_quantity
                    new_cost_basis = current_cost_basis + (price * quantity)
                    new_avg_price = new_cost_basis / new_quantity
                else:  # Selling some
                    new_avg_price = current_position['avg_price']
                
                # Update position
                self.positions[symbol] = {
                    'quantity': new_quantity,
                    'avg_price': new_avg_price,
                    'current_price': price,
                    'current_value': new_quantity * price,
                    'unrealized_pnl': (price - new_avg_price) * new_quantity,
                    'unrealized_pnl_pct': ((price / new_avg_price) - 1) * 100 if new_avg_price > 0 else 0
                }
        else:
            # New position
            if quantity != 0:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'current_price': price,
                    'current_value': quantity * price,
                    'unrealized_pnl': 0,
                    'unrealized_pnl_pct': 0
                }
        
        # Record the trade
        self.trades.append({
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'action': 'BUY' if quantity > 0 else 'SELL',
            'quantity': abs(quantity),
            'price': price,
            'value': abs(trade_value)
        })
        
        # Update performance history
        self._record_performance_snapshot(timestamp)
        
        logger.info(f"Updated position for {symbol}: {quantity} shares at ${price:.2f}")
    
    def update_prices(self, price_data: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Update current prices for all positions.
        
        Args:
            price_data: Dictionary mapping symbols to current prices
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update each position
        for symbol, position in self.positions.items():
            if symbol in price_data:
                price = price_data[symbol]
                quantity = position['quantity']
                avg_price = position['avg_price']
                
                # Update position data
                self.positions[symbol].update({
                    'current_price': price,
                    'current_value': quantity * price,
                    'unrealized_pnl': (price - avg_price) * quantity,
                    'unrealized_pnl_pct': ((price / avg_price) - 1) * 100 if avg_price > 0 else 0
                })
        
        # Update performance history
        self._record_performance_snapshot(timestamp)
    
    def _record_performance_snapshot(self, timestamp: Optional[datetime] = None) -> None:
        """
        Record current portfolio value in performance history.
        
        Args:
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate total portfolio value
        portfolio_value = self.current_cash + sum(
            position['current_value'] for position in self.positions.values()
        )
        
        # Add to history
        self.performance_history['timestamp'].append(timestamp)
        self.performance_history['portfolio_value'].append(portfolio_value)
        self.performance_history['cash'].append(self.current_cash)
    
    def update_strategy_allocations(self, allocations: Dict[str, float]) -> None:
        """
        Update strategy allocations.
        
        Args:
            allocations: Dictionary mapping strategy names to allocation percentages (0-100)
        """
        # Validate allocations
        total_allocation = sum(allocations.values())
        if not np.isclose(total_allocation, 100.0, atol=0.1):
            logger.warning(f"Strategy allocations sum to {total_allocation}%, expected 100%")
        
        self.strategy_allocations = allocations
        logger.info(f"Updated strategy allocations: {allocations}")
    
    def update_strategy_performance(self, strategy_performance: Dict[str, Dict[str, float]]) -> None:
        """
        Update performance metrics for strategies.
        
        Args:
            strategy_performance: Dictionary mapping strategy names to performance metrics
        """
        self.strategy_performance = strategy_performance
        logger.info(f"Updated strategy performance metrics")
    
    def get_portfolio_value(self) -> float:
        """
        Get current total portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        return self.current_cash + sum(
            position['current_value'] for position in self.positions.values()
        )
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, float]:
        """
        Calculate performance metrics over the specified time period.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dictionary with performance metrics
        """
        # Get history for the specified period
        if len(self.performance_history['timestamp']) < 2:
            return {
                'cumulative_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'timestamp': self.performance_history['timestamp'],
            'portfolio_value': self.performance_history['portfolio_value']
        })
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff_date]
        
        if df.empty or len(df) < 2:
            return {
                'cumulative_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Calculate metrics
        start_value = df['portfolio_value'].iloc[0]
        end_value = df['portfolio_value'].iloc[-1]
        
        # Daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Metrics
        cumulative_return = ((end_value / start_value) - 1) * 100
        
        # Annualized Sharpe Ratio (assuming risk-free rate of 0)
        daily_returns = df['daily_return'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown
        df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod()
        df['cumulative_max'] = df['cumulative_return'].cummax()
        df['drawdown'] = (df['cumulative_return'] / df['cumulative_max']) - 1
        max_drawdown = df['drawdown'].min() * 100
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0.0
        
        # Win rate and profit factor from trades
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df[trades_df['timestamp'] >= cutoff_date]
            
            # Group by symbol and action to identify round trips
            profitable_trades = 0
            losing_trades = 0
            total_profit = 0.0
            total_loss = 0.0
            
            # This is a simplification - in a real system we'd track P&L per trade
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol]
                
                buys = symbol_trades[symbol_trades['action'] == 'BUY']
                sells = symbol_trades[symbol_trades['action'] == 'SELL']
                
                # Simple approximation of P&L
                if not buys.empty and not sells.empty:
                    avg_buy_price = buys['price'].mean()
                    avg_sell_price = sells['price'].mean()
                    pnl = avg_sell_price - avg_buy_price
                    
                    if pnl > 0:
                        profitable_trades += 1
                        total_profit += pnl
                    else:
                        losing_trades += 1
                        total_loss += abs(pnl)
            
            total_trades = profitable_trades + losing_trades
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else (1.0 if total_profit > 0 else 0.0)
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """
        Calculate current asset allocation.
        
        Returns:
            Dictionary mapping asset classes to allocation percentages
        """
        # In a real implementation, this would categorize positions by sector/asset class
        # For now, we'll just return a simple cash vs equity split
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value == 0:
            return {'Cash': 100.0}
        
        cash_pct = (self.current_cash / portfolio_value) * 100
        equity_pct = 100 - cash_pct
        
        return {
            'Cash': cash_pct,
            'Equity': equity_pct
        }
    
    def get_recent_activity(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recent trading activity.
        
        Args:
            limit: Maximum number of items to return
        
        Returns:
            Dictionary with recent trades and signals
        """
        # Get recent trades
        recent_trades = sorted(
            self.trades,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]
        
        # In a real implementation, we would also include recent signals
        # For now, we'll just return trades
        return {
            'trades': recent_trades,
            'signals': []  # Would be populated in a real implementation
        }
    
    def get_full_state(self) -> Dict[str, Any]:
        """
        Get the complete portfolio state.
        
        Returns:
            Dictionary with complete portfolio state
        """
        # Calculate current metrics
        performance_metrics = self.get_performance_metrics()
        asset_allocation = self.get_asset_allocation()
        recent_activity = self.get_recent_activity()
        portfolio_value = self.get_portfolio_value()
        
        # Construct the state object
        state = {
            'portfolio': {
                'cash': self.current_cash,
                'total_value': portfolio_value,
                'positions': self.positions,
                'asset_allocation': asset_allocation
            },
            'performance_metrics': performance_metrics,
            'recent_activity': recent_activity,
            'strategy_data': {
                'active_strategies': list(self.strategy_allocations.keys()),
                'strategy_allocations': self.strategy_allocations,
                'strategy_performance': self.strategy_performance
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return state 