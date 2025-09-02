#!/usr/bin/env python3
"""
Forex Smart Compliance Module - Advanced Prop Firm Compliance

This module provides intelligent risk management and compliance monitoring:
- Adaptive position sizing based on risk utilization
- Monte Carlo risk projection and drawdown forecasting
- Intelligent profit target management
- Dynamic trading frequency controls
"""

import os
import sys
import yaml
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import math
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_smart_compliance')


class SmartComplianceMonitor:
    """
    Advanced prop firm compliance monitoring and intelligent risk management.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the compliance monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default prop firm rules
        self.rules = {
            'max_drawdown_percent': 5.0,
            'daily_loss_limit_percent': 3.0,
            'target_profit_percent': 8.0,
            'weekly_loss_limit_percent': 5.0,
            'max_trades_per_day': 20,
            'max_lot_size': 1.0,
            'restricted_hours': [],
            'restricted_days': ['Saturday', 'Sunday'],
            'max_simultaneous_trades': 3,
            'min_trade_duration_minutes': 2
        }
        
        # Update with config values if provided
        if config and 'prop_firm_rules' in config:
            self.rules.update(config['prop_firm_rules'])
        
        # Base position size (to be scaled based on risk)
        self.base_position_size = self.config.get('base_position_size', 0.1)
        
        # Risk budget trackers
        self.current_drawdown = 0.0
        self.current_daily_loss = 0.0
        self.current_weekly_loss = 0.0
        self.current_equity = 1000.0  # Will be updated on first call
        self.starting_equity = 1000.0
        self.high_water_mark = 1000.0
        self.current_open_risk = 0.0
        
        # Trade history for Monte Carlo
        self.trade_history = []
        self.last_trade_timestamp = None
        self.trade_count_today = 0
        self.last_day_reset = datetime.datetime.now().date()
        
        # Success tracker
        self.weekly_win_rate = 0.0
        self.daily_win_rate = 0.0
        self.session_win_rates = {}
        self.pair_win_rates = {}
        
        # Load risk profile if available
        self._load_risk_profile()
        
        logger.info("Smart Compliance Monitor initialized")
    
    def _load_risk_profile(self) -> None:
        """Load risk profile from file if available."""
        risk_profile_file = self.config.get('risk_profile_file', 'prop_risk_profile.yaml')
        
        if os.path.exists(risk_profile_file):
            try:
                with open(risk_profile_file, 'r') as f:
                    risk_profile = yaml.safe_load(f)
                
                if 'prop_firm_rules' in risk_profile:
                    self.rules.update(risk_profile['prop_firm_rules'])
                
                logger.info(f"Loaded risk profile from {risk_profile_file}")
            except Exception as e:
                logger.error(f"Error loading risk profile: {e}")
    
    def update_account_state(self, equity: float, 
                            daily_pnl: float = None, 
                            weekly_pnl: float = None,
                            open_positions: List[Dict[str, Any]] = None) -> None:
        """
        Update the current account state.
        
        Args:
            equity: Current account equity
            daily_pnl: Current daily P&L (optional)
            weekly_pnl: Current weekly P&L (optional)
            open_positions: List of open positions (optional)
        """
        # First update - set starting values
        if self.starting_equity == 1000.0 and equity != 1000.0:
            self.starting_equity = equity
            self.high_water_mark = equity
        
        self.current_equity = equity
        
        # Update high water mark if we have a new equity high
        if equity > self.high_water_mark:
            self.high_water_mark = equity
        
        # Calculate drawdown
        self.current_drawdown = (1.0 - (equity / self.high_water_mark)) * 100.0
        
        # Update daily P&L if provided
        if daily_pnl is not None:
            self.current_daily_loss = -daily_pnl if daily_pnl < 0 else 0.0
        
        # Update weekly P&L if provided
        if weekly_pnl is not None:
            self.current_weekly_loss = -weekly_pnl if weekly_pnl < 0 else 0.0
        
        # Reset trade count if it's a new day
        today = datetime.datetime.now().date()
        if today != self.last_day_reset:
            self.trade_count_today = 0
            self.last_day_reset = today
        
        # Calculate open risk if positions are provided
        if open_positions:
            total_risk = 0.0
            for position in open_positions:
                size = position.get('size', 0.0)
                stop_loss = position.get('stop_loss')
                entry_price = position.get('entry_price')
                
                if stop_loss and entry_price:
                    # Calculate risk in pips
                    is_buy = position.get('direction', '').lower() == 'buy'
                    pair = position.get('pair', '')
                    
                    if is_buy:
                        pips_at_risk = (entry_price - stop_loss) * self._get_pip_multiplier(pair)
                    else:
                        pips_at_risk = (stop_loss - entry_price) * self._get_pip_multiplier(pair)
                    
                    # Convert pips to money
                    pip_value = position.get('pip_value', 10.0)  # Default $10 per pip for 1.0 lot
                    risk_amount = pips_at_risk * pip_value * size
                    total_risk += risk_amount
            
            self.current_open_risk = total_risk
    
    def _get_pip_multiplier(self, pair: str) -> float:
        """Get pip multiplier for a currency pair."""
        if 'JPY' in pair:
            return 100  # JPY pairs have 2 decimal places
        else:
            return 10000  # Most pairs have 4 decimal places
    
    def calculate_position_size(self, base_size: Optional[float] = None, 
                               pair: str = 'EURUSD',
                               strategy_id: Optional[str] = None) -> float:
        """
        Adaptively adjust position size based on remaining risk budget.
        
        Args:
            base_size: Base position size to adjust (default: self.base_position_size)
            pair: Currency pair 
            strategy_id: Strategy ID for tracking performance
            
        Returns:
            Adjusted position size
        """
        if base_size is None:
            base_size = self.base_position_size
        
        # Check if we can trade at all
        if not self.is_trading_allowed():
            return 0.0
        
        # Calculate risk utilization
        max_dd_allowed = self.rules['max_drawdown_percent']
        dd_utilization = self.current_drawdown / max_dd_allowed
        
        max_daily_loss = self.rules['daily_loss_limit_percent']
        daily_utilization = self.current_daily_loss / (self.starting_equity * max_daily_loss / 100.0)
        
        # Use the most conservative risk utilization
        risk_utilization = max(dd_utilization, daily_utilization)
        
        # Calculate adaptive position size
        if risk_utilization > 0.9:
            # Critical risk level - no trading
            return 0.0
        elif risk_utilization > 0.8:
            position_factor = 0.25
        elif risk_utilization > 0.6:
            position_factor = 0.5
        elif risk_utilization > 0.4:
            position_factor = 0.75
        else:
            position_factor = 1.0
        
        # Factor in winning probability for this pair/strategy
        win_rate_factor = 1.0
        if pair in self.pair_win_rates:
            win_rate = self.pair_win_rates.get(pair, 0.5)
            # Increase size for high win rate pairs, decrease for low win rate
            if win_rate > 0.6:
                win_rate_factor = 1.2
            elif win_rate < 0.4:
                win_rate_factor = 0.8
        
        # Calculate projected profit percent
        profit_pct = (self.current_equity / self.starting_equity - 1.0) * 100.0
        target_pct = self.rules['target_profit_percent']
        
        # As we approach profit target, reduce position size
        if profit_pct > target_pct * 0.75:
            target_factor = 0.75
        elif profit_pct > target_pct * 0.5:
            target_factor = 0.9
        else:
            target_factor = 1.0
        
        # Combine all factors for final position size
        adjusted_size = base_size * position_factor * win_rate_factor * target_factor
        
        # Enforce maximum lot size
        adjusted_size = min(adjusted_size, self.rules.get('max_lot_size', 1.0))
        
        return adjusted_size
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on current account state and rules.
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check max drawdown
        if self.current_drawdown >= self.rules['max_drawdown_percent']:
            return False, f"Max drawdown exceeded: {self.current_drawdown:.2f}% > {self.rules['max_drawdown_percent']}%"
        
        # Check daily loss limit
        daily_loss_amount = self.current_daily_loss
        daily_loss_limit = self.starting_equity * (self.rules['daily_loss_limit_percent'] / 100.0)
        if daily_loss_amount >= daily_loss_limit:
            return False, f"Daily loss limit exceeded: ${daily_loss_amount:.2f} > ${daily_loss_limit:.2f}"
        
        # Check if we've hit profit target
        profit_pct = (self.current_equity / self.starting_equity - 1.0) * 100.0
        if profit_pct >= self.rules['target_profit_percent']:
            return False, f"Profit target reached: {profit_pct:.2f}% >= {self.rules['target_profit_percent']}%"
        
        # Check trade count for today
        if self.trade_count_today >= self.rules.get('max_trades_per_day', 20):
            return False, f"Maximum trades per day reached: {self.trade_count_today} >= {self.rules.get('max_trades_per_day', 20)}"
        
        # Check if current time is in restricted hours
        now = datetime.datetime.now()
        for restricted_range in self.rules.get('restricted_hours', []):
            start_hour, end_hour = restricted_range
            if start_hour <= now.hour < end_hour:
                return False, f"Trading not allowed during restricted hours: {start_hour}-{end_hour}"
        
        # Check if current day is restricted
        day_name = now.strftime('%A')
        if day_name in self.rules.get('restricted_days', ['Saturday', 'Sunday']):
            return False, f"Trading not allowed on {day_name}"
        
        return True, "Trading allowed"
    
    def record_trade_result(self, trade_data: Dict[str, Any]) -> None:
        """
        Record a completed trade for analysis and Monte Carlo simulation.
        
        Args:
            trade_data: Trade result data
        """
        # Add trade to history
        self.trade_history.append(trade_data)
        
        # Update counters
        self.trade_count_today += 1
        self.last_trade_timestamp = datetime.datetime.now()
        
        # Update win rates
        pair = trade_data.get('pair', '')
        session = trade_data.get('session', '')
        is_win = trade_data.get('pnl', 0.0) > 0.0
        
        # Update pair win rate
        if pair not in self.pair_win_rates:
            self.pair_win_rates[pair] = 0.5  # Start with neutral win rate
        
        # Simple EMA update of win rate
        alpha = 0.1  # Weight for new observation
        current_win_rate = self.pair_win_rates[pair]
        new_win_value = 1.0 if is_win else 0.0
        self.pair_win_rates[pair] = (1 - alpha) * current_win_rate + alpha * new_win_value
        
        # Update session win rate
        if session:
            if session not in self.session_win_rates:
                self.session_win_rates[session] = 0.5
            
            current_win_rate = self.session_win_rates[session]
            self.session_win_rates[session] = (1 - alpha) * current_win_rate + alpha * new_win_value
        
        # Update daily/weekly win rates
        # For simplicity, using the same exponential update
        self.daily_win_rate = (1 - alpha) * self.daily_win_rate + alpha * new_win_value
        self.weekly_win_rate = (1 - alpha) * self.weekly_win_rate + alpha * new_win_value
        
        # Limit history size
        max_history = 1000
        if len(self.trade_history) > max_history:
            self.trade_history = self.trade_history[-max_history:]
    
    def project_drawdown_risk(self, 
                            trades_history: Optional[List[Dict[str, Any]]] = None, 
                            open_positions: Optional[List[Dict[str, Any]]] = None,
                            num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Project drawdown likelihood using Monte Carlo simulation.
        
        Args:
            trades_history: Historical trades for simulation (uses internal if None)
            open_positions: Current open positions
            num_simulations: Number of Monte Carlo simulations to run
            
        Returns:
            Risk projection results
        """
        # Use provided history or internal history
        history = trades_history if trades_history else self.trade_history
        
        # Need at least 20 trades for a meaningful simulation
        if len(history) < 20:
            return {
                'drawdown_risk': {
                    'hit_max_dd_probability': 0.0,
                    'expected_drawdown': self.current_drawdown,
                    'worst_case_drawdown': self.current_drawdown * 1.5,
                    'confidence': 0.0  # No confidence with insufficient data
                },
                'profit_target': {
                    'hit_probability': 0.0,
                    'expected_days': 0,
                    'confidence': 0.0
                }
            }
        
        # Calculate statistics from trade history
        pnl_values = [trade.get('pnl', 0.0) for trade in history]
        pnl_pct_values = [trade.get('pnl_percent', 0.0) for trade in history]
        
        mean_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values)
        
        # Run Monte Carlo simulations
        hit_max_dd_count = 0
        hit_target_count = 0
        max_drawdowns = []
        days_to_target = []
        
        # Current account state
        current_equity = self.current_equity
        current_high_water = self.high_water_mark
        starting_equity = self.starting_equity
        
        # Simulation parameters
        avg_trades_per_day = min(self.trade_count_today, 5)  # Avoid division by zero
        if avg_trades_per_day == 0:
            avg_trades_per_day = 5  # Default assumption
        
        max_dd_pct = self.rules['max_drawdown_percent']
        target_pct = self.rules['target_profit_percent']
        
        target_equity = starting_equity * (1.0 + target_pct / 100.0)
        
        # Run simulations
        for sim in range(num_simulations):
            # Start from current state
            sim_equity = current_equity
            sim_high_water = current_high_water
            sim_drawdown = self.current_drawdown
            sim_days = 0
            reached_target = False
            hit_max_dd = False
            
            # Run for up to 30 trading days
            for day in range(30):
                sim_days += 1
                daily_trades = max(1, int(random.normalvariate(avg_trades_per_day, avg_trades_per_day / 2)))
                
                # Simulate each trade for the day
                for _ in range(daily_trades):
                    # Randomly sample from historical trade results
                    if pnl_pct_values:
                        # Use percentage-based PNL to account for changing equity
                        idx = random.randint(0, len(pnl_pct_values) - 1)
                        trade_pnl_pct = pnl_pct_values[idx]
                        trade_pnl = sim_equity * (trade_pnl_pct / 100.0)
                    else:
                        # Fallback to absolute PNL if percentage not available
                        idx = random.randint(0, len(pnl_values) - 1)
                        trade_pnl = random.normalvariate(mean_pnl, std_pnl)
                    
                    # Update equity
                    sim_equity += trade_pnl
                    
                    # Update high water mark and drawdown
                    if sim_equity > sim_high_water:
                        sim_high_water = sim_equity
                    
                    sim_drawdown = (1.0 - (sim_equity / sim_high_water)) * 100.0
                    
                    # Check if we hit max drawdown
                    if sim_drawdown >= max_dd_pct:
                        hit_max_dd = True
                        break
                    
                    # Check if we hit profit target
                    if sim_equity >= target_equity and not reached_target:
                        reached_target = True
                        days_to_target.append(sim_days)
                
                # Break early if we hit max drawdown
                if hit_max_dd:
                    break
            
            # Record results
            if hit_max_dd:
                hit_max_dd_count += 1
            
            if reached_target:
                hit_target_count += 1
            
            max_drawdowns.append(sim_drawdown)
        
        # Calculate probabilities
        max_dd_probability = hit_max_dd_count / num_simulations
        expected_drawdown = np.mean(max_drawdowns)
        worst_case_drawdown = np.percentile(max_drawdowns, 95)  # 95th percentile
        
        # Calculate profit target probabilities
        target_probability = hit_target_count / num_simulations
        avg_days_to_target = np.mean(days_to_target) if days_to_target else 30
        
        # Confidence based on sample size
        confidence = min(len(history) / 100.0, 0.9)
        
        return {
            'drawdown_risk': {
                'hit_max_dd_probability': max_dd_probability,
                'expected_drawdown': expected_drawdown,
                'worst_case_drawdown': worst_case_drawdown,
                'confidence': confidence
            },
            'profit_target': {
                'hit_probability': target_probability,
                'expected_days': avg_days_to_target,
                'confidence': confidence
            }
        }
    
    def suggest_trade_adjustments(self, 
                                open_positions: List[Dict[str, Any]],
                                risk_projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Suggest adjustments to open positions based on risk projection.
        
        Args:
            open_positions: Current open positions
            risk_projection: Risk projection data (will run if not provided)
            
        Returns:
            List of suggested adjustments
        """
        if not risk_projection:
            risk_projection = self.project_drawdown_risk(open_positions=open_positions)
        
        suggestions = []
        
        # Check drawdown risk
        drawdown_risk = risk_projection['drawdown_risk']
        dd_probability = drawdown_risk['hit_max_dd_probability']
        
        # Only make suggestions if we're confident
        if drawdown_risk['confidence'] > 0.5:
            # High risk of hitting max drawdown
            if dd_probability > 0.4:
                suggestions.append({
                    'action': 'reduce_risk',
                    'reason': f"High drawdown risk ({dd_probability:.1%} probability of hitting max drawdown)",
                    'severity': 'high',
                    'details': {
                        'current_drawdown': self.current_drawdown,
                        'expected_drawdown': drawdown_risk['expected_drawdown'],
                        'recommendation': 'Close losing positions or reduce position size by 50%'
                    }
                })
            # Moderate risk
            elif dd_probability > 0.2:
                suggestions.append({
                    'action': 'caution',
                    'reason': f"Moderate drawdown risk ({dd_probability:.1%} probability of hitting max drawdown)",
                    'severity': 'medium',
                    'details': {
                        'current_drawdown': self.current_drawdown,
                        'expected_drawdown': drawdown_risk['expected_drawdown'],
                        'recommendation': 'Consider reducing position size by 25%'
                    }
                })
        
        # Check profit target proximity
        profit_pct = (self.current_equity / self.starting_equity - 1.0) * 100.0
        target_pct = self.rules['target_profit_percent']
        profit_ratio = profit_pct / target_pct
        
        if profit_ratio > 0.8:
            suggestions.append({
                'action': 'secure_profits',
                'reason': f"Nearing profit target ({profit_pct:.2f}% of {target_pct:.0f}% target)",
                'severity': 'medium',
                'details': {
                    'current_profit': profit_pct,
                    'target_profit': target_pct,
                    'recommendation': 'Tighten stop losses to secure profits'
                }
            })
        
        # Check open positions risk balance
        if len(open_positions) > 1:
            # Check for position concentration
            pairs = [pos['pair'] for pos in open_positions]
            pair_counts = {}
            for pair in pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            max_pair_count = max(pair_counts.values())
            if max_pair_count > len(open_positions) * 0.5:
                # More than 50% of positions in one pair - concentration risk
                concentrated_pair = [pair for pair, count in pair_counts.items() if count == max_pair_count][0]
                suggestions.append({
                    'action': 'diversify',
                    'reason': f"Position concentration risk: {max_pair_count} positions on {concentrated_pair}",
                    'severity': 'medium',
                    'details': {
                        'concentrated_pair': concentrated_pair,
                        'recommendation': 'Consider diversifying or reducing exposure'
                    }
                })
        
        return suggestions


# Test function
def test_smart_compliance():
    """Test the compliance monitor functionality."""
    monitor = SmartComplianceMonitor({
        'prop_firm_rules': {
            'max_drawdown_percent': 5.0,
            'daily_loss_limit_percent': 3.0,
            'target_profit_percent': 8.0,
        }
    })
    
    # Test basic functionality
    monitor.update_account_state(990.0, daily_pnl=-10.0)
    position_size = monitor.calculate_position_size(0.1)
    print(f"Position size with 1% drawdown: {position_size:.2f} lots")
    
    # Test with higher drawdown
    monitor.update_account_state(960.0, daily_pnl=-15.0)
    position_size = monitor.calculate_position_size(0.1)
    print(f"Position size with 4% drawdown: {position_size:.2f} lots")
    
    # Test risk projection
    # Create some mock trade history
    history = []
    for i in range(50):
        # Random PnL between -20 and +30 pips
        pnl = random.uniform(-20, 30)
        history.append({
            'pair': 'EURUSD',
            'pnl': pnl,
            'pnl_percent': pnl / 100.0,  # Convert to percentage of account
            'session': random.choice(['London', 'New York', 'Asian']),
            'timestamp': datetime.datetime.now() - datetime.timedelta(hours=i*4)
        })
    
    for trade in history:
        monitor.record_trade_result(trade)
    
    risk = monitor.project_drawdown_risk()
    print(f"Drawdown risk projection: {risk['drawdown_risk']}")
    print(f"Profit target projection: {risk['profit_target']}")
    
    # Test trading allowance
    is_allowed, reason = monitor.is_trading_allowed()
    print(f"Trading allowed: {is_allowed} - {reason}")
    
    return "Smart compliance tests completed"


if __name__ == "__main__":
    test_smart_compliance()
