#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Module for Volatility Strategies

This module handles risk management for volatility-based options strategies,
focusing on position sizing, stop loss implementation, and dynamic risk adjustment
based on volatility conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class VolatilityRiskManager:
    """
    Advanced risk management for volatility-based options strategies.
    
    This class provides:
    - Vega-based position sizing
    - Dynamic stop loss based on volatility changes
    - Time decay acceleration triggers
    - Portfolio-level volatility exposure management
    """
    
    def __init__(self,
                 account_value: float = 100000.0,
                 max_position_size_pct: float = 0.05,
                 max_portfolio_vega_pct: float = 0.20,
                 profit_target_pct: float = 0.35,
                 stop_loss_pct: float = 0.60):
        """
        Initialize the risk manager.
        
        Args:
            account_value: Total account value in dollars
            max_position_size_pct: Maximum position size as percentage of account
            max_portfolio_vega_pct: Maximum vega exposure as percentage of account
            profit_target_pct: Profit target as percentage of premium
            stop_loss_pct: Stop loss as percentage of premium
        """
        self.account_value = account_value
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_vega_pct = max_portfolio_vega_pct
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Current portfolio state
        self.active_positions = {}
        self.total_vega_exposure = 0.0
        self.total_delta_exposure = 0.0
        self.total_theta_exposure = 0.0
        self.position_correlations = {}
        
    def update_account_value(self, account_value: float) -> None:
        """
        Update the current account value.
        
        Args:
            account_value: Current account value in dollars
        """
        self.account_value = account_value
        
    def calculate_position_size(self,
                               option_data: Dict[str, Any],
                               volatility_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            option_data: Selected option combination data
            volatility_data: Volatility metrics for dynamic sizing
            
        Returns:
            Dictionary with position sizing recommendations
        """
        if not option_data:
            return {'contracts': 0, 'notional_value': 0.0, 'account_pct': 0.0}
            
        # Extract key values from option data
        total_premium = option_data.get('total_premium', 0)
        if total_premium <= 0:
            logger.warning("Invalid option premium for position sizing")
            return {'contracts': 0, 'notional_value': 0.0, 'account_pct': 0.0}
            
        # Calculate base position size based on account percentage
        max_position_value = self.account_value * self.max_position_size_pct
        
        # Calculate contract value (premium per contract × 100 shares)
        contract_value = total_premium * 100
        
        # Calculate maximum number of contracts based on position value limit
        max_contracts = int(max_position_value / contract_value)
        
        # Adjust based on volatility risk if data provided
        if volatility_data:
            # Extract volatility metrics
            iv_percentile = volatility_data.get('iv_percentile', 50)
            vol_regime = volatility_data.get('regime', 'neutral')
            
            # Apply adjustments based on volatility conditions
            multiplier = 1.0
            
            # Reduce size in high volatility regime
            if vol_regime == 'high_volatility':
                multiplier *= 0.7
            # Increase size slightly in low volatility regime
            elif vol_regime == 'low_volatility':
                multiplier *= 1.1
                
            # Adjust based on IV percentile
            if iv_percentile > 80:  # Very high IV levels
                multiplier *= 0.8  # Reduce size further
            elif iv_percentile < 20:  # Very low IV levels
                multiplier *= 1.1  # Increase size slightly
                
            # Apply vega-based adjustment if available
            net_vega = option_data.get('net_vega', 0)
            if net_vega > 0:
                # Calculate vega-adjusted size
                max_vega_exposure = self.account_value * self.max_portfolio_vega_pct
                remaining_vega = max_vega_exposure - self.total_vega_exposure
                
                # Don't take position if vega exposure would be too high
                if remaining_vega <= 0:
                    logger.warning("Maximum portfolio vega exposure reached")
                    return {'contracts': 0, 'notional_value': 0.0, 'account_pct': 0.0}
                    
                # Calculate vega-based contracts
                vega_based_contracts = int(remaining_vega / (net_vega * 100))
                
                # Take the minimum of vega-based and regular sizing
                max_contracts = min(max_contracts, vega_based_contracts)
            
            # Apply the final multiplier
            adjusted_contracts = int(max_contracts * multiplier)
            
            # Ensure at least 1 contract if we're taking a position
            final_contracts = max(1, adjusted_contracts) if adjusted_contracts > 0 else 0
        else:
            final_contracts = max(1, max_contracts) if max_contracts > 0 else 0
            
        # Calculate the resulting position value and account percentage
        position_value = final_contracts * contract_value
        account_percentage = (position_value / self.account_value) * 100
        
        return {
            'contracts': final_contracts,
            'notional_value': position_value,
            'premium_per_contract': total_premium,
            'account_pct': account_percentage,
            'max_loss': position_value,  # For a long straddle/strangle, max loss is premium paid
            'max_loss_pct': account_percentage
        }
        
    def calculate_exit_conditions(self,
                                option_data: Dict[str, Any],
                                position_size: Dict[str, Any],
                                volatility_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate exit conditions including profit targets and stop losses.
        
        Args:
            option_data: Selected option combination data
            position_size: Position sizing data
            volatility_data: Volatility metrics for dynamic exit conditions
            
        Returns:
            Dictionary with exit condition parameters
        """
        if not option_data or not position_size or position_size.get('contracts', 0) <= 0:
            return {}
            
        # Extract premium information
        total_premium = option_data.get('total_premium', 0)
        position_value = position_size.get('notional_value', 0)
        
        # Base exit conditions
        profit_target = total_premium * (1 + self.profit_target_pct)
        stop_loss = total_premium * (1 - self.stop_loss_pct)
        
        # Calculate exit prices for each leg
        profit_exit = {
            'target_value': profit_target * 100 * position_size.get('contracts', 0),
            'target_premium': profit_target,
            'target_pct': self.profit_target_pct * 100
        }
        
        stop_loss_exit = {
            'stop_value': stop_loss * 100 * position_size.get('contracts', 0),
            'stop_premium': stop_loss,
            'stop_pct': self.stop_loss_pct * 100
        }
        
        # Dynamic adjustments based on volatility if available
        if volatility_data:
            vol_regime = volatility_data.get('regime', 'neutral')
            iv_hv_spread = volatility_data.get('iv_hv_spread', 0)
            days_to_expiry = option_data.get('days_to_expiration', 30)
            
            # Time-based adjustments
            time_adjustments = {}
            
            # If close to expiration, tighten profit targets
            if days_to_expiry <= 7:
                time_adjustments['profit_target_multiplier'] = 0.7  # Tighten profit target
                time_adjustments['accelerated_time_decay'] = True
                time_adjustments['days_to_expiry'] = days_to_expiry
                
                # Adjust profit target based on time remaining
                profit_target *= 0.7
                profit_exit['target_premium'] = profit_target
                profit_exit['target_value'] = profit_target * 100 * position_size.get('contracts', 0)
                profit_exit['target_pct'] = (profit_target / total_premium - 1) * 100
                profit_exit['time_adjusted'] = True
            
            # Volatility regime adjustments
            vol_adjustments = {}
            
            # In high volatility regime, consider taking profits sooner
            if vol_regime == 'high_volatility':
                vol_adjustments['profit_target_multiplier'] = 0.8
                vol_adjustments['volatility_regime'] = 'high'
                
                # Adjust profit target based on volatility
                profit_target *= 0.8
                profit_exit['target_premium'] = profit_target
                profit_exit['target_value'] = profit_target * 100 * position_size.get('contracts', 0)
                profit_exit['target_pct'] = (profit_target / total_premium - 1) * 100
                profit_exit['vol_adjusted'] = True
                
            # If IV is much higher than HV, consider tighter profit targets
            if iv_hv_spread > 0.10:  # IV at least 10 percentage points above HV
                vol_adjustments['iv_rich'] = True
                vol_adjustments['iv_hv_spread'] = iv_hv_spread
                
                # Adjust profit target if not already adjusted
                if not profit_exit.get('vol_adjusted'):
                    profit_multiplier = max(0.6, 1 - iv_hv_spread)
                    profit_target *= profit_multiplier
                    profit_exit['target_premium'] = profit_target
                    profit_exit['target_value'] = profit_target * 100 * position_size.get('contracts', 0)
                    profit_exit['target_pct'] = (profit_target / total_premium - 1) * 100
                    profit_exit['vol_adjusted'] = True
                    
            # Save the adjustment factors
            if time_adjustments:
                profit_exit['time_adjustments'] = time_adjustments
                
            if vol_adjustments:
                profit_exit['vol_adjustments'] = vol_adjustments
        
        # Calculate expected value
        win_probability = option_data.get('win_probability', 0.4)  # Default probability
        expected_gain = profit_target * win_probability
        expected_loss = total_premium * (1 - self.stop_loss_pct) * (1 - win_probability)
        expected_value = expected_gain - expected_loss
        
        # Time-based maximum holding period
        max_days_held = min(30, option_data.get('days_to_expiration', 30) - 7)
        
        return {
            'profit_exit': profit_exit,
            'stop_loss': stop_loss_exit,
            'expected_value': expected_value,
            'expected_value_ratio': expected_value / total_premium if total_premium > 0 else 0,
            'win_probability': win_probability,
            'max_days_held': max_days_held,
            'time_stop': datetime.now() + timedelta(days=max_days_held)
        }
        
    def track_position(self,
                      position_id: str,
                      option_data: Dict[str, Any],
                      position_size: Dict[str, Any],
                      exit_conditions: Dict[str, Any]) -> None:
        """
        Track a new position for risk management purposes.
        
        Args:
            position_id: Unique identifier for the position
            option_data: Selected option combination data
            position_size: Position sizing data
            exit_conditions: Exit condition parameters
        """
        if position_id in self.active_positions:
            logger.warning(f"Position {position_id} already exists in tracking")
            return
            
        # Extract Greek exposures
        contracts = position_size.get('contracts', 0)
        net_delta = option_data.get('net_delta', 0) * contracts * 100
        net_gamma = option_data.get('net_gamma', 0) * contracts * 100
        net_theta = option_data.get('net_theta', 0) * contracts * 100
        net_vega = option_data.get('net_vega', 0) * contracts * 100
        
        # Create position tracking record
        position = {
            'id': position_id,
            'symbol': option_data.get('symbol', 'unknown'),
            'strategy_type': option_data.get('strategy_type', 'unknown'),
            'entry_time': datetime.now(),
            'expiration': option_data.get('expiration'),
            'call_strike': option_data.get('call_strike'),
            'put_strike': option_data.get('put_strike'),
            'contracts': contracts,
            'premium_paid': option_data.get('total_premium', 0) * contracts * 100,
            'current_value': option_data.get('total_premium', 0) * contracts * 100,
            'profit_target': exit_conditions.get('profit_exit', {}).get('target_value'),
            'stop_loss': exit_conditions.get('stop_loss', {}).get('stop_value'),
            'max_days_held': exit_conditions.get('max_days_held'),
            'time_stop': exit_conditions.get('time_stop'),
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'status': 'active'
        }
        
        # Add to active positions
        self.active_positions[position_id] = position
        
        # Update portfolio-level exposures
        self.total_delta_exposure += net_delta
        self.total_theta_exposure += net_theta
        self.total_vega_exposure += net_vega
        
        logger.info(f"Position {position_id} added to tracking: {contracts} contracts, premium={position['premium_paid']:.2f}")
        
    def update_position_value(self,
                             position_id: str,
                             current_value: float,
                             update_time: datetime = None) -> Dict[str, Any]:
        """
        Update a position's current value and check exit conditions.
        
        Args:
            position_id: Unique identifier for the position
            current_value: Current value of the position
            update_time: Time of the update (default: now)
            
        Returns:
            Action to take (hold, exit due to profit, exit due to stop loss, etc.)
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found in tracking")
            return {'action': 'not_found'}
            
        position = self.active_positions[position_id]
        position['current_value'] = current_value
        position['last_update'] = update_time or datetime.now()
        
        # Calculate P&L
        entry_value = position['premium_paid']
        pnl = current_value - entry_value
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
        
        position['current_pnl'] = pnl
        position['current_pnl_pct'] = pnl_pct
        
        # Check exit conditions
        result = {
            'action': 'hold',
            'position_id': position_id,
            'current_value': current_value,
            'entry_value': entry_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        
        # Check profit target
        if position['profit_target'] and current_value >= position['profit_target']:
            result['action'] = 'exit_profit_target'
            result['exit_type'] = 'profit_target'
            result['profit_target'] = position['profit_target']
            
        # Check stop loss
        elif position['stop_loss'] and current_value <= position['stop_loss']:
            result['action'] = 'exit_stop_loss'
            result['exit_type'] = 'stop_loss'
            result['stop_loss'] = position['stop_loss']
            
        # Check time stop
        elif position['time_stop'] and (update_time or datetime.now()) >= position['time_stop']:
            result['action'] = 'exit_time_stop'
            result['exit_type'] = 'time_stop'
            result['time_stop'] = position['time_stop']
            
        # Add position details to result
        result['position'] = position
        
        return result
        
    def close_position(self,
                      position_id: str,
                      exit_value: float,
                      exit_time: datetime = None,
                      exit_reason: str = 'manual') -> Dict[str, Any]:
        """
        Close a tracked position and update portfolio metrics.
        
        Args:
            position_id: Unique identifier for the position
            exit_value: Exit value of the position
            exit_time: Time of exit (default: now)
            exit_reason: Reason for exit
            
        Returns:
            Closed position data with P&L metrics
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found in tracking")
            return None
            
        position = self.active_positions[position_id]
        
        # Update position with exit data
        position['exit_value'] = exit_value
        position['exit_time'] = exit_time or datetime.now()
        position['exit_reason'] = exit_reason
        position['status'] = 'closed'
        
        # Calculate final P&L
        entry_value = position['premium_paid']
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
        
        position['final_pnl'] = pnl
        position['final_pnl_pct'] = pnl_pct
        
        # Update portfolio-level exposures
        self.total_delta_exposure -= position['net_delta']
        self.total_theta_exposure -= position['net_theta']
        self.total_vega_exposure -= position['net_vega']
        
        # Log the closure
        logger.info(f"Position {position_id} closed: PnL=${pnl:.2f} ({pnl_pct:.2f}%), reason={exit_reason}")
        
        # Return the closed position data
        return position
