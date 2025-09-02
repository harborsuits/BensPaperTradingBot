#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Risk Manager for Trading Strategies

This module implements an AI-driven risk management system that automatically
adjusts confidence parameters, position sizing, and risk allocation based on
portfolio growth, market conditions, and historical performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

from trading_bot.strategies.forex.confidence_adjusted_position_sizing import ConfidenceAdjustedPositionSizing

logger = logging.getLogger(__name__)

class AdaptiveRiskManager:
    """
    AI-driven risk management system that autonomously adjusts parameters
    based on portfolio size, performance metrics, and market conditions.
    """
    
    # Default scaling parameters
    DEFAULT_PARAMS = {
        # Portfolio scaling parameters
        'base_portfolio_size': 10000,          # Reference portfolio size for baseline risk
        'risk_scale_factor': 0.8,              # How quickly risk scales down (0.5-0.9)
        'min_risk_percent': 0.2,               # Minimum risk % regardless of portfolio size
        'max_risk_percent': 2.0,               # Maximum risk % for small portfolios
        
        # Performance-based adjustment
        'performance_window_days': 30,          # Days to consider for performance analysis
        'min_trades_for_adjustment': 20,        # Minimum trades before auto-adjustment
        'profit_factor_threshold': 1.3,         # Min profit factor to increase risk
        'drawdown_threshold': 10.0,             # Max drawdown % before reducing risk
        'max_adjustment_per_cycle': 0.2,        # Maximum parameter change per adjustment cycle
        
        # Confidence parameter boundaries
        'min_confidence_threshold_range': [0.2, 0.5],  # [min, max] bounds
        'high_confidence_threshold_range': [0.6, 0.9], # [min, max] bounds
        'low_confidence_factor_range': [0.3, 0.7],     # [min, max] bounds
        'high_confidence_factor_range': [1.2, 2.0],    # [min, max] bounds
        
        # Optimization settings
        'optimization_frequency_days': 7,       # How often to update parameters
        'auto_optimization': True,              # Whether to automatically optimize
        'randomized_exploration': 0.1,          # Random exploration factor (0-1)
    }
    
    def __init__(self, 
                 position_sizer: ConfidenceAdjustedPositionSizing,
                 state_dir: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adaptive Risk Manager.
        
        Args:
            position_sizer: ConfidenceAdjustedPositionSizing instance to manage
            state_dir: Directory to store state and history
            parameters: Custom parameters to override defaults
        """
        self.position_sizer = position_sizer
        self.state_dir = state_dir
        
        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)
        
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
            
        # Initialize performance tracking
        self.trade_history = []
        self.parameter_history = []
        self.last_optimization_time = None
        
        # Load existing state if available
        self._load_state()
        
        logger.info(f"Adaptive Risk Manager initialized with parameters: {self.parameters}")
    
    def calculate_adaptive_risk_percent(self, account_balance: float) -> float:
        """
        Calculate adaptive risk percentage based on account balance.
        Risk scales down as the portfolio grows.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Risk percentage to use
        """
        base_size = self.parameters['base_portfolio_size']
        max_risk = self.parameters['max_risk_percent']
        min_risk = self.parameters['min_risk_percent']
        scale_factor = self.parameters['risk_scale_factor']
        
        # Calculate scaled risk (higher scale_factor = slower reduction)
        # This creates a curve that gradually reduces risk as account grows
        if account_balance <= base_size:
            # For small accounts, use the maximum risk
            risk_percent = max_risk
        else:
            # Scaling formula: risk reduces as portfolio grows
            # Uses a power law scaling that gradually decreases
            multiplier = (base_size / account_balance) ** scale_factor
            risk_percent = max(min_risk, min(max_risk, max_risk * multiplier))
        
        return risk_percent
    
    def get_optimized_parameters(self, account_balance: float) -> Dict[str, Any]:
        """
        Get optimized confidence and risk parameters based on 
        current portfolio size and performance history.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Dictionary of optimized parameters
        """
        # Calculate adaptive risk percentage
        risk_percent = self.calculate_adaptive_risk_percent(account_balance)
        
        # Start with current parameters
        optimized = self.position_sizer.confidence_params.copy()
        
        # Update risk percentage
        optimized['max_risk_per_trade_percent'] = risk_percent
        
        # If we don't have enough trading history or auto-optimization is disabled, 
        # just return the scaled risk
        if (len(self.trade_history) < self.parameters['min_trades_for_adjustment'] or
            not self.parameters['auto_optimization']):
            return optimized
        
        # Check if it's time to re-optimize
        current_time = datetime.now()
        if (self.last_optimization_time and 
            (current_time - self.last_optimization_time).days < self.parameters['optimization_frequency_days']):
            # Not time to optimize yet
            return optimized
            
        # Analyze performance
        performance_metrics = self._analyze_performance()
        
        # Adjust parameters based on performance
        optimized = self._adjust_parameters_based_on_performance(
            optimized, performance_metrics, account_balance
        )
        
        # Record optimization time
        self.last_optimization_time = current_time
        
        # Save the new parameters to history
        self.parameter_history.append({
            'timestamp': current_time.isoformat(),
            'account_balance': account_balance,
            'parameters': optimized,
            'performance_metrics': performance_metrics
        })
        
        # Save state after optimization
        self._save_state()
        
        # Log the optimization results
        logger.info(f"Optimized parameters: max_risk={optimized['max_risk_per_trade_percent']:.2f}%, "
                    f"min_confidence={optimized['min_confidence_threshold']:.2f}, "
                    f"performance_factor={performance_metrics.get('profit_factor', 0):.2f}")
                    
        return optimized
    
    def record_trade_result(self, 
                           trade_data: Dict[str, Any], 
                           account_balance: float,
                           confidence_data: Optional[Dict[str, Any]] = None):
        """
        Record a completed trade for performance analysis and optimization.
        
        Args:
            trade_data: Trade result data including profit/loss
            account_balance: Current account balance after the trade
            confidence_data: Confidence metrics used for the trade decision
        """
        # Ensure the trade has a timestamp
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
            
        # Add account balance
        trade_data['account_balance'] = account_balance
        
        # Add confidence data if provided
        if confidence_data:
            trade_data['confidence_data'] = confidence_data
            
        # Add to history
        self.trade_history.append(trade_data)
        
        # Trim history if it gets too large (keep last 1000 trades)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
            
        # Save state periodically (every 10 trades)
        if len(self.trade_history) % 10 == 0:
            self._save_state()
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze trading performance over the specified window.
        
        Returns:
            Dictionary of performance metrics
        """
        # Filter trades to the performance window
        cutoff_date = datetime.now() - timedelta(days=self.parameters['performance_window_days'])
        recent_trades = [
            trade for trade in self.trade_history 
            if datetime.fromisoformat(trade['timestamp']) > cutoff_date
        ]
        
        if not recent_trades:
            return {'profit_factor': 0, 'win_rate': 0, 'drawdown': 0, 'avg_profit': 0}
            
        # Calculate key metrics
        profits = [trade['profit_loss'] for trade in recent_trades if trade.get('profit_loss', 0) > 0]
        losses = [abs(trade['profit_loss']) for trade in recent_trades if trade.get('profit_loss', 0) < 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        
        # Profit factor (total profit / total loss)
        profit_factor = total_profit / total_loss if total_loss > 0 else (2.0 if total_profit > 0 else 0)
        
        # Win rate
        win_count = len(profits)
        trade_count = len(recent_trades)
        win_rate = win_count / trade_count if trade_count > 0 else 0
        
        # Calculate drawdown
        if len(recent_trades) > 1:
            balance_history = []
            current_balance = recent_trades[0].get('account_balance', 0)
            peak_balance = current_balance
            max_drawdown_pct = 0
            
            for trade in recent_trades:
                pnl = trade.get('profit_loss', 0)
                current_balance += pnl
                balance_history.append(current_balance)
                
                if current_balance > peak_balance:
                    peak_balance = current_balance
                elif peak_balance > 0:
                    drawdown_pct = (peak_balance - current_balance) / peak_balance * 100
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        else:
            max_drawdown_pct = 0
        
        # Average values
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / (trade_count - win_count) if (trade_count - win_count) > 0 else 0
        
        # Calculate confidence effectiveness
        confidence_accuracy = self._analyze_confidence_effectiveness(recent_trades)
        
        return {
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'drawdown': max_drawdown_pct,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'trade_count': trade_count,
            'confidence_accuracy': confidence_accuracy
        }
    
    def _analyze_confidence_effectiveness(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze how well confidence metrics predicted trade outcomes.
        
        Args:
            trades: List of trade records with confidence data
            
        Returns:
            Dictionary of confidence effectiveness metrics
        """
        # Filter trades that have confidence data
        confidence_trades = [t for t in trades if 'confidence_data' in t and 'profit_loss' in t]
        
        if not confidence_trades:
            return {'high_win_rate': 0, 'medium_win_rate': 0, 'low_win_rate': 0}
            
        # Group by confidence level
        high_conf_trades = [t for t in confidence_trades 
                           if t['confidence_data'].get('confidence', 0) >= 
                           self.position_sizer.confidence_params['high_confidence_threshold']]
                           
        low_conf_trades = [t for t in confidence_trades 
                          if t['confidence_data'].get('confidence', 0) < 
                          self.position_sizer.confidence_params['baseline_confidence']]
                          
        medium_conf_trades = [t for t in confidence_trades 
                             if t not in high_conf_trades and t not in low_conf_trades]
        
        # Calculate win rates for each group
        high_win_rate = sum(1 for t in high_conf_trades if t['profit_loss'] > 0) / len(high_conf_trades) if high_conf_trades else 0
        medium_win_rate = sum(1 for t in medium_conf_trades if t['profit_loss'] > 0) / len(medium_conf_trades) if medium_conf_trades else 0
        low_win_rate = sum(1 for t in low_conf_trades if t['profit_loss'] > 0) / len(low_conf_trades) if low_conf_trades else 0
        
        return {
            'high_win_rate': high_win_rate,
            'medium_win_rate': medium_win_rate,
            'low_win_rate': low_win_rate,
            'high_conf_count': len(high_conf_trades),
            'medium_conf_count': len(medium_conf_trades),
            'low_conf_count': len(low_conf_trades)
        }
    
    def _adjust_parameters_based_on_performance(self, 
                                              current_params: Dict[str, Any],
                                              performance: Dict[str, Any],
                                              account_balance: float) -> Dict[str, Any]:
        """
        Adjust confidence parameters based on performance metrics.
        
        Args:
            current_params: Current parameters
            performance: Performance metrics
            account_balance: Current account balance
            
        Returns:
            Adjusted parameters
        """
        # Copy current parameters
        adjusted = current_params.copy()
        
        # Maximum adjustment per cycle
        max_adjustment = self.parameters['max_adjustment_per_cycle']
        
        # Get key metrics
        profit_factor = performance.get('profit_factor', 0)
        win_rate = performance.get('win_rate', 0)
        drawdown = performance.get('drawdown', 0)
        confidence_accuracy = performance.get('confidence_accuracy', {})
        
        # 1. Adjust risk based on overall performance
        profit_threshold = self.parameters['profit_factor_threshold']
        drawdown_threshold = self.parameters['drawdown_threshold']
        
        # Base risk has already been calculated based on account size
        current_risk = adjusted.get('max_risk_per_trade_percent', self.parameters['max_risk_percent'])
        
        # Adjust risk if profit factor is high and drawdown is low
        if profit_factor > profit_threshold and drawdown < drawdown_threshold and win_rate > 0.4:
            # Good performance, could potentially increase risk
            risk_adjustment = min(max_adjustment, (profit_factor - profit_threshold) * 0.2)
            new_risk = min(self.parameters['max_risk_percent'], current_risk * (1 + risk_adjustment))
        elif drawdown > drawdown_threshold or profit_factor < 1.0:
            # Poor performance, reduce risk
            risk_adjustment = min(max_adjustment, drawdown / 100)
            new_risk = max(self.parameters['min_risk_percent'], current_risk * (1 - risk_adjustment))
        else:
            # Maintain current risk
            new_risk = current_risk
            
        adjusted['max_risk_per_trade_percent'] = new_risk
        
        # 2. Adjust confidence thresholds based on confidence effectiveness
        high_win_rate = confidence_accuracy.get('high_win_rate', 0)
        medium_win_rate = confidence_accuracy.get('medium_win_rate', 0)
        low_win_rate = confidence_accuracy.get('low_win_rate', 0)
        
        # If we have enough data on confidence levels
        if (confidence_accuracy.get('high_conf_count', 0) > 5 and
            confidence_accuracy.get('medium_conf_count', 0) > 5):
            
            # If high confidence trades aren't performing much better than medium ones
            if high_win_rate < medium_win_rate * 1.1:
                # Increase high confidence threshold to be more selective
                current = adjusted.get('high_confidence_threshold', 0.7)
                max_val = self.parameters['high_confidence_threshold_range'][1]
                adjusted['high_confidence_threshold'] = min(max_val, current + 0.05)
            
            # If high confidence trades are performing significantly better
            elif high_win_rate > medium_win_rate * 1.5:
                # Decrease threshold slightly to include more trades in high confidence
                current = adjusted.get('high_confidence_threshold', 0.7)
                min_val = self.parameters['high_confidence_threshold_range'][0]
                adjusted['high_confidence_threshold'] = max(min_val, current - 0.05)
                
            # If low confidence trades have a decent win rate
            if low_win_rate > 0.4:
                # Lower the minimum confidence threshold to trade more
                current = adjusted.get('min_confidence_threshold', 0.3)
                min_val = self.parameters['min_confidence_threshold_range'][0]
                adjusted['min_confidence_threshold'] = max(min_val, current - 0.05)
            elif low_win_rate < 0.3:
                # Increase minimum threshold to avoid poor trades
                current = adjusted.get('min_confidence_threshold', 0.3)
                max_val = self.parameters['min_confidence_threshold_range'][1]
                adjusted['min_confidence_threshold'] = min(max_val, current + 0.05)
        
        # 3. Adjust position size multipliers based on overall performance
        if profit_factor > 1.5 and win_rate > 0.55:
            # More aggressive on high confidence since system is performing well
            current = adjusted.get('high_confidence_factor', 1.5)
            max_val = self.parameters['high_confidence_factor_range'][1]
            adjusted['high_confidence_factor'] = min(max_val, current + 0.1)
        elif profit_factor < 1.0 or drawdown > drawdown_threshold * 1.5:
            # More conservative on high confidence during poor performance
            current = adjusted.get('high_confidence_factor', 1.5)
            min_val = self.parameters['high_confidence_factor_range'][0]
            adjusted['high_confidence_factor'] = max(min_val, current - 0.1)
            
            # Also reduce low confidence factor
            current = adjusted.get('low_confidence_factor', 0.5)
            min_val = self.parameters['low_confidence_factor_range'][0]
            adjusted['low_confidence_factor'] = max(min_val, current - 0.05)
            
        # 4. Add some randomized exploration for parameter discovery
        # This helps avoid local optimization traps
        if self.parameters['randomized_exploration'] > 0:
            exploration_factor = self.parameters['randomized_exploration']
            
            # Randomly adjust parameters within bounds with small probability
            if np.random.random() < exploration_factor:
                key = np.random.choice([
                    'min_confidence_threshold',
                    'high_confidence_threshold',
                    'low_confidence_factor',
                    'high_confidence_factor'
                ])
                
                current = adjusted.get(key, 0.5)
                
                # Determine parameter range
                if key == 'min_confidence_threshold':
                    param_range = self.parameters['min_confidence_threshold_range']
                elif key == 'high_confidence_threshold':
                    param_range = self.parameters['high_confidence_threshold_range']
                elif key == 'low_confidence_factor':
                    param_range = self.parameters['low_confidence_factor_range']
                elif key == 'high_confidence_factor':
                    param_range = self.parameters['high_confidence_factor_range']
                
                # Random adjustment within range
                adjusted[key] = max(param_range[0], min(param_range[1], 
                                                       current + np.random.uniform(-0.1, 0.1)))
                
                logger.info(f"Exploration: Adjusted {key} from {current:.2f} to {adjusted[key]:.2f}")
            
        return adjusted
    
    def _save_state(self):
        """Save the current state to disk."""
        state = {
            'trade_history': self.trade_history,
            'parameter_history': self.parameter_history,
            'last_optimization_time': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'parameters': self.parameters
        }
        
        state_path = os.path.join(self.state_dir, 'adaptive_risk_manager_state.json')
        
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug(f"Saved AdaptiveRiskManager state to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save risk manager state: {str(e)}")
    
    def _load_state(self):
        """Load state from disk if available."""
        state_path = os.path.join(self.state_dir, 'adaptive_risk_manager_state.json')
        
        if not os.path.exists(state_path):
            logger.info("No existing state file found, starting fresh")
            return
            
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            self.trade_history = state.get('trade_history', [])
            self.parameter_history = state.get('parameter_history', [])
            
            last_optimization_time = state.get('last_optimization_time')
            if last_optimization_time:
                self.last_optimization_time = datetime.fromisoformat(last_optimization_time)
                
            # Only update non-critical parameters from saved state
            saved_params = state.get('parameters', {})
            for key, value in saved_params.items():
                if key in self.parameters and key not in ['base_portfolio_size', 'min_risk_percent', 'max_risk_percent']:
                    self.parameters[key] = value
                    
            logger.info(f"Loaded AdaptiveRiskManager state with {len(self.trade_history)} trade records")
        except Exception as e:
            logger.error(f"Failed to load risk manager state: {str(e)}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics and parameter adjustments.
        
        Returns:
            Dictionary with performance summary
        """
        # Calculate overall performance
        performance = self._analyze_performance()
        
        # Get parameter history summary
        param_changes = []
        if len(self.parameter_history) > 1:
            for i in range(1, min(5, len(self.parameter_history))):
                prev = self.parameter_history[-i-1]['parameters']
                curr = self.parameter_history[-i]['parameters']
                
                changes = {}
                for key in ['max_risk_per_trade_percent', 'min_confidence_threshold', 'high_confidence_threshold']:
                    if key in prev and key in curr:
                        changes[key] = {
                            'from': prev[key],
                            'to': curr[key],
                            'change': ((curr[key] - prev[key]) / prev[key] * 100) if prev[key] != 0 else 0
                        }
                        
                param_changes.append({
                    'timestamp': self.parameter_history[-i]['timestamp'],
                    'changes': changes
                })
        
        return {
            'performance': performance,
            'recent_adjustments': param_changes,
            'trade_count': len(self.trade_history),
            'current_parameters': self.position_sizer.confidence_params,
            'last_optimization': self.last_optimization_time.isoformat() if self.last_optimization_time else None
        }
