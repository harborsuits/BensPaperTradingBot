#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic Capital Manager for BensBot
Provides intelligent position sizing and risk management based on:
- Current capital and high water mark
- Performance metrics
- Win/loss streaks
- Market volatility
- Strategy performance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class CapitalManager:
    """
    Dynamic capital and position sizing manager
    Adjusts position sizes based on account performance,
    strategy performance, market conditions, and risk parameters
    """
    
    def __init__(self, initial_capital: float, risk_params: Dict[str, Any] = None,
                persistence_manager=None):
        """
        Initialize the capital manager
        
        Args:
            initial_capital: Initial capital amount
            risk_params: Dictionary of risk parameters
            persistence_manager: Optional persistence manager for state storage
        """
        # Default risk parameters
        default_params = {
            'base_risk_pct': 0.01,          # 1% base risk per trade
            'max_account_risk_pct': 0.05,    # Max 5% of account at risk
            'max_position_risk_pct': 0.02,   # Max 2% risk per position
            'max_drawdown_scaling': 0.5,     # Scale back to 50% on drawdowns
            'winning_streak_boost': 0.2,     # +20% size on winning streaks
            'losing_streak_reduction': 0.3,  # -30% size on losing streaks
            'volatility_scaling': True,      # Scale position based on volatility
            'performance_scaling': True,     # Scale based on strategy performance
            'streak_length': 5,              # How many trades define a streak
            'scale_by_sharpe': True,         # Scale by Sharpe ratio
            'min_trades_for_scaling': 10,    # Min trades before performance scaling
            'max_leverage': 10.0,            # Maximum leverage allowed
            'target_win_pct': 0.55,          # Target win percentage
            'volatility_lookback': 20,       # Periods for volatility calculation
            'use_kelly_criterion': False,    # Use Kelly criterion for sizing
            'kelly_fraction': 0.5,           # Kelly fraction (conservative)
            'enable_martingale': False,      # Enable martingale sizing
            'martingale_factor': 1.5,        # Martingale multiplication factor
            'max_martingale_steps': 3        # Maximum martingale steps
        }
        
        # Use provided parameters or defaults
        self.params = default_params.copy()
        if risk_params:
            self.params.update(risk_params)
            
        # Initialize capital tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0
        self.last_update_time = datetime.now()
        
        # Initialize performance tracking
        self.trade_history = []
        self.strategy_performance = {}
        self.strategy_scaling_factors = {}
        
        # Set persistence manager if provided
        self.persistence_manager = persistence_manager
        
        # Log initialization
        logger.info(f"Capital Manager initialized with {initial_capital:.2f} initial capital")
        logger.info(f"Base risk: {self.params['base_risk_pct']*100:.1f}%, "
                  f"Max account risk: {self.params['max_account_risk_pct']*100:.1f}%")
    
    def update_capital(self, new_capital: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Update current capital and track high water mark
        
        Args:
            new_capital: New capital amount
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            dict: Updated capital metrics
        """
        prev_capital = self.current_capital
        self.current_capital = new_capital
        timestamp = timestamp or datetime.now()
        
        # Update high water mark if needed
        if new_capital > self.high_water_mark:
            self.high_water_mark = new_capital
            
        # Calculate drawdown
        if self.high_water_mark > 0:
            current_drawdown = 1.0 - (new_capital / self.high_water_mark)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate return since last update
        return_pct = ((new_capital - prev_capital) / prev_capital) if prev_capital > 0 else 0
        
        # Create capital update record
        update_record = {
            'timestamp': timestamp,
            'previous_capital': prev_capital,
            'current_capital': new_capital,
            'high_water_mark': self.high_water_mark,
            'current_drawdown': current_drawdown if self.high_water_mark > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'return_pct': return_pct
        }
        
        # Store update if persistence manager available
        if self.persistence_manager:
            self.persistence_manager.save_performance_metrics({
                'type': 'capital_update',
                **update_record
            })
            
        self.last_update_time = timestamp
        
        return update_record
    
    def record_trade(self, strategy_id: str, 
                    trade_data: Dict[str, Any]) -> None:
        """
        Record a completed trade for performance tracking
        
        Args:
            strategy_id: ID of the strategy that generated the trade
            trade_data: Dictionary with trade details
        """
        required_fields = ['symbol', 'profit_loss', 'win', 'timestamp']
        
        # Validate trade data has required fields
        for field in required_fields:
            if field not in trade_data:
                logger.warning(f"Missing required field {field} in trade_data")
                return
                
        # Add strategy ID if not present
        if 'strategy_id' not in trade_data:
            trade_data['strategy_id'] = strategy_id
            
        # Use current time if timestamp not provided
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
            
        # Add trade to history
        self.trade_history.append(trade_data)
        
        # Limit trade history size (keep last 1000)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
            
        # Update strategy performance metrics
        self._update_strategy_performance(strategy_id, trade_data)
        
        # Store trade if persistence manager available
        if self.persistence_manager:
            self.persistence_manager.save_trade(trade_data)
            
    def _update_strategy_performance(self, strategy_id: str, 
                                   trade_data: Dict[str, Any]) -> None:
        """
        Update performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            trade_data: Dictionary with trade details
        """
        # Initialize strategy performance if needed
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0,
                'total_loss': 0,
                'recent_trades': [],  # List of True/False for wins/losses
                'avg_profit': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'sharpe_ratio': 0,
                'daily_returns': {},
                'trade_details': []  # List of recent trade summaries
            }
            
        perf = self.strategy_performance[strategy_id]
        
        # Update trade counts
        perf['total_trades'] += 1
        
        # Update win/loss metrics
        profit_loss = trade_data.get('profit_loss', 0)
        is_win = trade_data.get('win', profit_loss > 0)
        
        if is_win:
            perf['winning_trades'] += 1
            perf['total_profit'] += profit_loss
        else:
            perf['losing_trades'] += 1
            perf['total_loss'] += abs(profit_loss)
            
        # Update recent trades list for streak analysis
        perf['recent_trades'].append(is_win)
        if len(perf['recent_trades']) > self.params['streak_length']:
            perf['recent_trades'].pop(0)
            
        # Update trade details (keep last 50)
        trade_summary = {
            'timestamp': trade_data.get('timestamp'),
            'symbol': trade_data.get('symbol'),
            'profit_loss': profit_loss,
            'win': is_win,
            'size': trade_data.get('position_size', None),
            'risk': trade_data.get('risk_amount', None)
        }
        
        perf['trade_details'].append(trade_summary)
        if len(perf['trade_details']) > 50:
            perf['trade_details'] = perf['trade_details'][-50:]
            
        # Update daily returns for Sharpe calculation
        timestamp = trade_data.get('timestamp', datetime.now())
        date_key = timestamp.strftime('%Y-%m-%d')
        
        if date_key not in perf['daily_returns']:
            perf['daily_returns'][date_key] = 0
            
        perf['daily_returns'][date_key] += profit_loss
        
        # Keep only last 100 days
        if len(perf['daily_returns']) > 100:
            # Sort by date and keep most recent
            sorted_dates = sorted(perf['daily_returns'].keys())
            for old_date in sorted_dates[:-100]:
                del perf['daily_returns'][old_date]
                
        # Recalculate performance metrics
        self._recalculate_performance_metrics(strategy_id)
        
    def _recalculate_performance_metrics(self, strategy_id: str) -> None:
        """
        Recalculate all performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id not in self.strategy_performance:
            return
            
        perf = self.strategy_performance[strategy_id]
        
        # Skip if no trades
        if perf['total_trades'] == 0:
            return
            
        # Win rate
        perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
        
        # Average win/loss
        perf['avg_profit'] = perf['total_profit'] / perf['winning_trades'] if perf['winning_trades'] > 0 else 0
        perf['avg_loss'] = perf['total_loss'] / perf['losing_trades'] if perf['losing_trades'] > 0 else 0
        
        # Profit factor
        perf['profit_factor'] = perf['total_profit'] / perf['total_loss'] if perf['total_loss'] > 0 else float('inf')
        
        # Expectancy
        perf['expectancy'] = (perf['win_rate'] * perf['avg_profit']) - ((1 - perf['win_rate']) * perf['avg_loss'])
        
        # Sharpe ratio (using daily returns)
        daily_returns = list(perf['daily_returns'].values())
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            perf['sharpe_ratio'] = mean_return / std_return if std_return > 0 else 0
            
        # Update scaling factor based on performance
        self._update_strategy_scaling_factor(strategy_id)
        
    def _update_strategy_scaling_factor(self, strategy_id: str) -> None:
        """
        Update the position sizing scaling factor for a strategy
        based on its performance metrics
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id not in self.strategy_performance:
            # Default to 1.0 if no performance data
            self.strategy_scaling_factors[strategy_id] = 1.0
            return
            
        perf = self.strategy_performance[strategy_id]
        
        # Default scaling factor
        scaling_factor = 1.0
        
        # Only apply performance scaling with sufficient history
        if perf['total_trades'] >= self.params['min_trades_for_scaling']:
            # Win rate scaling (linear from 0.6x at 30% win rate to 1.4x at 70% win rate)
            win_rate = perf['win_rate']
            win_rate_factor = 0.6 + (win_rate - 0.3) * 2.0
            win_rate_factor = max(0.6, min(1.4, win_rate_factor))
            
            # Profit factor scaling (from 0.7x at profit factor 0.8 to 1.5x at profit factor 2.0)
            profit_factor = perf['profit_factor']
            profit_factor_scaling = 0.7 + (profit_factor - 0.8) * (0.8 / 1.2)
            profit_factor_scaling = max(0.7, min(1.5, profit_factor_scaling))
            
            # Sharpe ratio scaling if enabled
            sharpe_scaling = 1.0
            if self.params['scale_by_sharpe'] and perf['sharpe_ratio'] > 0:
                # From 0.8x at Sharpe 0 to 1.3x at Sharpe 2.0
                sharpe = perf['sharpe_ratio']
                sharpe_scaling = 0.8 + (sharpe * 0.25)
                sharpe_scaling = min(1.3, sharpe_scaling)
                
            # Combine factors (weighted average)
            scaling_factor = (win_rate_factor * 0.4) + (profit_factor_scaling * 0.4) + (sharpe_scaling * 0.2)
        
        # Apply streak adjustments
        recent = perf['recent_trades']
        if len(recent) >= self.params['streak_length']:
            # Check for winning streak
            if all(recent):
                scaling_factor *= (1.0 + self.params['winning_streak_boost'])
                
            # Check for losing streak
            elif not any(recent):
                scaling_factor *= (1.0 - self.params['losing_streak_reduction'])
                
        # Store the scaling factor
        self.strategy_scaling_factors[strategy_id] = scaling_factor
        
        logger.debug(f"Strategy {strategy_id} scaling factor updated to {scaling_factor:.2f}")
        
    def calculate_position_size(self, strategy_id: str, 
                               symbol: str,
                               entry_price: float,
                               stop_loss_price: float,
                               target_price: Optional[float] = None,
                               current_volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate the optimal position size based on risk parameters,
        account state, strategy performance, and market conditions
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            target_price: Optional take profit price
            current_volatility: Optional current market volatility
            
        Returns:
            dict: Position sizing information
        """
        # Calculate risk per trade in price terms
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            logger.warning(f"Zero price risk for {symbol}, invalid stop loss")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'risk_percent': 0,
                'reason': "Invalid stop loss (zero risk)"
            }
            
        # Get strategy scaling factor
        strategy_factor = self.strategy_scaling_factors.get(strategy_id, 1.0)
        
        # Base position sizing - fixed percent risk
        base_risk_amount = self.current_capital * self.params['base_risk_pct']
        
        # Apply scaling factors
        scaling_factor = 1.0
        scaling_reasons = []
        
        # 1. Strategy performance scaling
        if self.params['performance_scaling'] and strategy_id in self.strategy_performance:
            scaling_factor *= strategy_factor
            scaling_reasons.append(f"Strategy performance: {strategy_factor:.2f}x")
            
        # 2. Drawdown protection
        drawdown_pct = 0
        if self.high_water_mark > 0:
            drawdown_pct = 1 - (self.current_capital / self.high_water_mark)
            
        if drawdown_pct > 0.1:  # If in >10% drawdown
            drawdown_scaling = max(0.5, 1.0 - (drawdown_pct * self.params['max_drawdown_scaling']))
            scaling_factor *= drawdown_scaling
            scaling_reasons.append(f"Drawdown protection: {drawdown_scaling:.2f}x")
            
        # 3. Volatility adjustment
        if current_volatility is not None and self.params['volatility_scaling']:
            # Compare to baseline volatility (normalize against a value of 1.0)
            # If volatility is higher, reduce size; if lower, increase size
            vol_baseline = 1.0  # Calibrated to a specific instrument
            vol_ratio = vol_baseline / current_volatility if current_volatility > 0 else 1.0
            
            # Cap the adjustment between 0.5x and 1.5x
            vol_scaling = min(1.5, max(0.5, vol_ratio))
            scaling_factor *= vol_scaling
            scaling_reasons.append(f"Volatility scaling: {vol_scaling:.2f}x")
            
        # 4. Risk/reward adjustment
        if target_price is not None:
            # Calculate R:R ratio
            reward = abs(target_price - entry_price)
            risk = price_risk
            
            if risk > 0:
                rr_ratio = reward / risk
                
                # Increase size for better R:R trades
                if rr_ratio >= 3.0:
                    rr_scaling = 1.2
                elif rr_ratio >= 2.0:
                    rr_scaling = 1.1
                else:
                    rr_scaling = 1.0
                    
                scaling_factor *= rr_scaling
                scaling_reasons.append(f"Risk/reward scaling: {rr_scaling:.2f}x")
                
        # 5. Kelly criterion if enabled
        if self.params['use_kelly_criterion'] and strategy_id in self.strategy_performance:
            perf = self.strategy_performance[strategy_id]
            
            if perf['total_trades'] >= self.params['min_trades_for_scaling']:
                win_rate = perf['win_rate']
                
                # Need average win/loss ratio
                if perf['avg_loss'] > 0:
                    avg_win_loss_ratio = perf['avg_profit'] / perf['avg_loss']
                    
                    # Kelly formula: f* = (bp - q) / b
                    # where: b = win/loss ratio, p = win probability, q = loss probability
                    kelly_pct = (avg_win_loss_ratio * win_rate - (1 - win_rate)) / avg_win_loss_ratio
                    
                    # Apply Kelly fraction for conservative sizing
                    kelly_pct = max(0, kelly_pct) * self.params['kelly_fraction']
                    
                    # Convert to scaling factor (1.0 = standard risk)
                    kelly_scaling = kelly_pct / self.params['base_risk_pct']
                    
                    # Cap the scaling
                    kelly_scaling = min(2.0, max(0.1, kelly_scaling))
                    
                    # Use this as the primary scaling factor
                    scaling_factor = kelly_scaling
                    scaling_reasons = [f"Kelly criterion: {kelly_scaling:.2f}x"]
                    
        # 6. Apply max account risk limit
        max_risk_amount = self.current_capital * self.params['max_account_risk_pct']
        
        # Calculate final risk amount
        risk_amount = min(base_risk_amount * scaling_factor, max_risk_amount)
        
        # Convert to position size
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        
        # Calculate risk as percentage of account
        risk_percent = (position_size * price_risk) / self.current_capital if self.current_capital > 0 else 0
        
        # Create position sizing result
        result = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'base_risk_amount': base_risk_amount,
            'scaling_factor': scaling_factor,
            'scaling_reasons': scaling_reasons,
            'max_risk_amount': max_risk_amount
        }
        
        logger.debug(f"Position size for {symbol}: {position_size:.4f}, "
                   f"risk: ${risk_amount:.2f} ({risk_percent*100:.2f}%)")
        
        return result
    
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            dict: Performance metrics
        """
        if strategy_id not in self.strategy_performance:
            return {
                'strategy_id': strategy_id,
                'total_trades': 0,
                'performance_available': False
            }
            
        # Return a copy to prevent modification
        perf = self.strategy_performance[strategy_id].copy()
        
        # Add scaling factor
        perf['scaling_factor'] = self.strategy_scaling_factors.get(strategy_id, 1.0)
        perf['performance_available'] = True
        
        return perf
    
    def get_all_strategies_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all strategies
        
        Returns:
            dict: Dictionary mapping strategy IDs to performance metrics
        """
        result = {}
        
        for strategy_id in self.strategy_performance:
            result[strategy_id] = self.get_strategy_performance(strategy_id)
            
        return result
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """
        Get overall portfolio statistics
        
        Returns:
            dict: Portfolio statistics
        """
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'high_water_mark': self.high_water_mark,
            'max_drawdown': self.max_drawdown,
            'return_pct': ((self.current_capital / self.initial_capital) - 1) if self.initial_capital > 0 else 0,
            'total_trades': len(self.trade_history),
            'strategies_count': len(self.strategy_performance),
            'last_update': self.last_update_time.isoformat()
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to persistence manager
        
        Returns:
            dict: Serialized state
        """
        # Create serializable state
        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'high_water_mark': self.high_water_mark,
            'max_drawdown': self.max_drawdown,
            'last_update_time': self.last_update_time.isoformat(),
            'strategy_scaling_factors': self.strategy_scaling_factors
        }
        
        # Save to persistence manager if available
        if self.persistence_manager:
            self.persistence_manager.save_strategy_state('capital_manager', state)
            
        return state
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load state from saved data
        
        Args:
            state: Serialized state dictionary
            
        Returns:
            bool: True if successful
        """
        try:
            self.initial_capital = state.get('initial_capital', self.initial_capital)
            self.current_capital = state.get('current_capital', self.current_capital)
            self.high_water_mark = state.get('high_water_mark', self.high_water_mark)
            self.max_drawdown = state.get('max_drawdown', self.max_drawdown)
            
            # Parse date string to datetime
            if 'last_update_time' in state:
                self.last_update_time = datetime.fromisoformat(state['last_update_time'])
                
            # Load scaling factors
            self.strategy_scaling_factors = state.get('strategy_scaling_factors', {})
            
            logger.info(f"Loaded Capital Manager state: {self.current_capital:.2f} current capital")
            return True
        except Exception as e:
            logger.error(f"Error loading Capital Manager state: {str(e)}")
            return False
