#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Module for Complex Spread Strategies

This module provides risk management functionality specifically 
designed for complex options spread strategies like iron condors,
butterflies, etc.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ComplexSpreadRiskManager:
    """
    Risk manager for complex option spread strategies.
    
    This class provides specialized risk management for complex spread strategies
    including position sizing, exit conditions, and adjustment criteria.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the risk manager with strategy parameters.
        
        Args:
            parameters: Dictionary of configuration parameters
        """
        self.params = parameters or {}
        self.logger = logger
        
    def calculate_position_size(self, spread: Dict[str, Any], 
                              account_info: Dict[str, Any]) -> int:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            spread: Spread configuration details
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Number of spreads to trade (contracts per leg)
        """
        if not spread or not account_info:
            return 0
            
        try:
            # Extract necessary data
            account_value = account_info.get('equity', 0)
            buying_power = account_info.get('buying_power', account_value * 0.5)
            
            # Get max risk per spread
            max_risk = spread.get('max_risk', 0)
            
            if max_risk <= 0:
                self.logger.warning("Invalid max risk, cannot calculate position size")
                return 0
                
            # Calculate position size based on risk parameters
            max_risk_pct = self.params.get('max_risk_per_trade_pct', 2.0) / 100
            max_loss_amount = account_value * max_risk_pct
            
            # Calculate position size based on max loss
            position_size = int(max_loss_amount / max_risk)
            
            # Apply additional limits
            max_position_pct = self.params.get('max_position_pct', 15.0) / 100
            max_position_buying_power = buying_power * max_position_pct
            
            # Calculate margin requirement
            strategy_type = spread.get('strategy_type', '')
            if strategy_type == 'iron_condor':
                # Margin is typically the width of the wider spread minus credit received
                call_width = spread.get('call_spread_width', 0)
                put_width = spread.get('put_spread_width', 0)
                credit = spread.get('total_credit', 0)
                
                margin_per_spread = max(call_width, put_width) * 100 - credit * 100
                max_size_by_margin = int(max_position_buying_power / margin_per_spread)
                
                position_size = min(position_size, max_size_by_margin)
                
            elif strategy_type == 'butterfly':
                # Margin for butterfly is typically just the debit paid
                debit = spread.get('total_debit', 0)
                margin_per_spread = debit * 100
                max_size_by_margin = int(max_position_buying_power / margin_per_spread)
                
                position_size = min(position_size, max_size_by_margin)
                
            # Apply min/max position size constraints
            min_position_size = self.params.get('min_position_size', 1)
            max_position_size = self.params.get('max_position_size', 10)
            
            position_size = max(min_position_size, min(position_size, max_position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1  # Default to minimum position size on error
            
    def calculate_exit_conditions(self, spread: Dict[str, Any], 
                                position_size: int) -> Dict[str, Any]:
        """
        Calculate appropriate exit conditions for the spread.
        
        Args:
            spread: Spread configuration details
            position_size: Number of spreads to trade
            
        Returns:
            Dictionary with exit condition parameters
        """
        if not spread:
            return {}
            
        try:
            strategy_type = spread.get('strategy_type', '')
            exit_conditions = {}
            
            # Common exit conditions
            exit_conditions['max_days'] = self.params.get('max_days_to_hold', 45)
            exit_conditions['days_before_earnings'] = self.params.get('exit_days_before_earnings', 5)
            
            # Calculate profit target based on credit received
            profit_target_pct = self.params.get('profit_target_pct', 50)
            profit_target = spread.get('total_credit', 0) * profit_target_pct / 100
            
            exit_conditions['profit_target'] = profit_target
            exit_conditions['profit_target_pct'] = profit_target_pct / 100
            
            # Calculate stop loss
            stop_loss_pct = self.params.get('stop_loss_pct', 100)  # Percentage of max profit
            max_loss = spread.get('max_risk', 0)
            
            stop_loss = max_loss * stop_loss_pct / 100
            
            exit_conditions['stop_loss'] = stop_loss
            exit_conditions['stop_loss_pct'] = stop_loss_pct / 100
            
            # Set time-based exit conditions
            dte = spread.get('dte', 0)
            if dte > 0:
                # Default to exiting at 21 DTE or 50% profit, whichever comes first
                exit_conditions['min_dte'] = self.params.get('min_dte_exit', 21)
                
                # Calculate theta decay acceleration point (typically around 30-45 DTE)
                theta_decay_point = self.params.get('theta_decay_point', 30)
                exit_conditions['theta_decay_point'] = theta_decay_point
                
                # Set early management threshold
                early_management_point = min(dte - 7, theta_decay_point)
                exit_conditions['early_management_point'] = early_management_point
                
            # Strategy-specific exit conditions
            if strategy_type == 'iron_condor':
                # For iron condors, consider breaching short strikes
                exit_conditions['short_strike_breach_buffer'] = self.params.get('short_strike_breach_buffer', 0.5)
                exit_conditions['iv_decrease_exit'] = self.params.get('iv_decrease_exit', 0.3)  # Exit if IV decreases by 30%
                
                # Delta-based management
                exit_conditions['max_delta_exposure'] = self.params.get('max_delta_exposure', 0.25)
                
                # Calculate risk:reward at exit
                exit_conditions['min_credit_remaining'] = spread.get('total_credit', 0) * 0.20
                
            elif strategy_type == 'butterfly':
                # For butterflies, consider profit taking near center strike
                exit_conditions['center_strike_profit_taking'] = self.params.get('center_strike_profit_taking', 0.75)
                
            # Add expected value calculation
            probability_of_profit = spread.get('probability_of_profit', 0.5)
            max_profit = spread.get('max_profit', 0)
            max_loss = spread.get('max_risk', 0)
            
            expected_value = (probability_of_profit * max_profit) - ((1 - probability_of_profit) * max_loss)
            expected_value_ratio = expected_value / max_loss if max_loss > 0 else 0
            
            exit_conditions['expected_value'] = expected_value
            exit_conditions['expected_value_ratio'] = expected_value_ratio
            
            return exit_conditions
            
        except Exception as e:
            self.logger.error(f"Error calculating exit conditions: {e}")
            return {
                'profit_target_pct': 0.5,  # 50% of max profit
                'stop_loss_pct': 1.0,      # 100% of max loss (i.e., max risk)
                'max_days': 45             # Default max days to hold
            }
            
    def evaluate_adjustment_criteria(self, position: Dict[str, Any], 
                                   current_metrics: Dict[str, Any],
                                   market_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate if a complex spread position needs adjustment.
        
        Args:
            position: Current position details
            current_metrics: Current position metrics
            market_data: Current market data (optional)
            
        Returns:
            Dictionary with adjustment decision and details
        """
        result = {
            'needs_adjustment': False,
            'adjustment_type': None,
            'reason': '',
            'details': {}
        }
        
        if not position or not current_metrics:
            return result
            
        try:
            strategy_type = position.get('strategy_type', '')
            
            # Common adjustment criteria
            dte = current_metrics.get('days_to_expiry', position.get('dte', 0))
            pnl_pct = current_metrics.get('pnl_pct', 0)
            current_price = current_metrics.get('current_price', 0)
            
            # Check if close to expiration but not profitable
            if dte <= 7 and pnl_pct < 0.25:
                result['needs_adjustment'] = True
                result['adjustment_type'] = 'roll_out'
                result['reason'] = f"Close to expiration ({dte} DTE) with insufficient profit ({pnl_pct:.1%})"
                return result
                
            # Strategy-specific adjustments
            if strategy_type == 'iron_condor':
                # Check if price is approaching short strikes
                short_call_strike = position.get('short_call_strike', float('inf'))
                short_put_strike = position.get('short_put_strike', 0)
                
                # Calculate proximity to short strikes
                call_proximity = (short_call_strike - current_price) / current_price
                put_proximity = (current_price - short_put_strike) / current_price
                
                # Set threshold for adjustment (e.g., within 2% of short strike)
                proximity_threshold = self.params.get('adjustment_proximity_threshold', 0.02)
                
                if call_proximity <= proximity_threshold:
                    result['needs_adjustment'] = True
                    result['adjustment_type'] = 'defend_call_side'
                    result['reason'] = f"Price approaching short call strike (within {call_proximity:.1%})"
                    result['details']['strike_distance'] = call_proximity
                    return result
                    
                if put_proximity <= proximity_threshold:
                    result['needs_adjustment'] = True
                    result['adjustment_type'] = 'defend_put_side'
                    result['reason'] = f"Price approaching short put strike (within {put_proximity:.1%})"
                    result['details']['strike_distance'] = put_proximity
                    return result
                    
                # Check if IV has significantly changed
                initial_iv = position.get('initial_iv', 0)
                current_iv = current_metrics.get('current_iv', 0)
                
                if initial_iv > 0 and current_iv > 0:
                    iv_change = (current_iv - initial_iv) / initial_iv
                    
                    # If IV expanded significantly, consider closing for a loss
                    if iv_change > 0.3:  # 30% increase in IV
                        result['needs_adjustment'] = True
                        result['adjustment_type'] = 'close_early'
                        result['reason'] = f"Significant IV expansion ({iv_change:.1%})"
                        result['details']['iv_change'] = iv_change
                        return result
                
            elif strategy_type == 'butterfly':
                # Butterfly adjustment criteria would go here
                pass
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating adjustment criteria: {e}")
            return result
            
    def generate_exit_order(self, position: Dict[str, Any], 
                          exit_reason: str) -> Dict[str, Any]:
        """
        Generate exit order details based on position and exit reason.
        
        Args:
            position: Position details
            exit_reason: Reason for exiting
            
        Returns:
            Exit order details
        """
        if not position:
            return {}
            
        try:
            strategy_type = position.get('strategy_type', '')
            exit_order = {
                'action': 'EXIT',
                'position_id': position.get('position_id', ''),
                'symbol': position.get('symbol', ''),
                'strategy_type': strategy_type,
                'reason': exit_reason,
                'timestamp': datetime.now().isoformat(),
                'legs': []
            }
            
            # Strategy-specific exit order details
            if strategy_type == 'iron_condor':
                # For iron condor, we need to close all four legs
                for leg_type in ['short_call', 'long_call', 'short_put', 'long_put']:
                    if f'{leg_type}_symbol' in position:
                        action = 'BUY' if leg_type.startswith('short') else 'SELL'
                        
                        leg = {
                            'symbol': position.get(f'{leg_type}_symbol', ''),
                            'action': action,
                            'quantity': position.get('quantity', 1),
                            'order_type': 'MARKET',  # Default to market order for exits
                            'leg_type': leg_type
                        }
                        
                        exit_order['legs'].append(leg)
                
            elif strategy_type == 'butterfly':
                # Butterfly exit logic would go here
                pass
                
            return exit_order
            
        except Exception as e:
            self.logger.error(f"Error generating exit order: {e}")
            return {
                'action': 'EXIT',
                'position_id': position.get('position_id', ''),
                'symbol': position.get('symbol', ''),
                'reason': 'Error generating detailed exit order'
            }
            
    def generate_adjustment_order(self, position: Dict[str, Any], 
                                adjustment_type: str,
                                current_price: float) -> Dict[str, Any]:
        """
        Generate adjustment order details based on position and adjustment type.
        
        Args:
            position: Position details
            adjustment_type: Type of adjustment needed
            current_price: Current price of the underlying
            
        Returns:
            Adjustment order details
        """
        if not position or not adjustment_type:
            return {}
            
        try:
            strategy_type = position.get('strategy_type', '')
            adjustment_order = {
                'action': 'ADJUST',
                'position_id': position.get('position_id', ''),
                'symbol': position.get('symbol', ''),
                'strategy_type': strategy_type,
                'adjustment_type': adjustment_type,
                'timestamp': datetime.now().isoformat(),
                'legs': []
            }
            
            # Strategy-specific adjustment order details
            if strategy_type == 'iron_condor':
                # Handle different adjustment types
                if adjustment_type == 'defend_call_side':
                    # Example: Roll short call up and out
                    short_call_symbol = position.get('short_call_symbol', '')
                    long_call_symbol = position.get('long_call_symbol', '')
                    
                    if short_call_symbol and long_call_symbol:
                        # Close existing short call
                        adjustment_order['legs'].append({
                            'symbol': short_call_symbol,
                            'action': 'BUY',  # Buy to close short call
                            'quantity': position.get('quantity', 1),
                            'order_type': 'MARKET',
                            'leg_type': 'short_call_close'
                        })
                        
                        # New short call will be determined by the option selector
                        adjustment_order['details'] = {
                            'adjustment_type': 'roll_short_call',
                            'current_price': current_price,
                            'new_short_call_delta': -0.16,  # Target delta for new short call
                            'dte_extension': 15  # Add 15 days to expiration
                        }
                        
                elif adjustment_type == 'defend_put_side':
                    # Example: Roll short put down and out
                    short_put_symbol = position.get('short_put_symbol', '')
                    long_put_symbol = position.get('long_put_symbol', '')
                    
                    if short_put_symbol and long_put_symbol:
                        # Close existing short put
                        adjustment_order['legs'].append({
                            'symbol': short_put_symbol,
                            'action': 'BUY',  # Buy to close short put
                            'quantity': position.get('quantity', 1),
                            'order_type': 'MARKET',
                            'leg_type': 'short_put_close'
                        })
                        
                        # New short put will be determined by the option selector
                        adjustment_order['details'] = {
                            'adjustment_type': 'roll_short_put',
                            'current_price': current_price,
                            'new_short_put_delta': 0.16,  # Target delta for new short put
                            'dte_extension': 15  # Add 15 days to expiration
                        }
                        
                elif adjustment_type == 'roll_out':
                    # Roll entire position to later expiration
                    adjustment_order['details'] = {
                        'adjustment_type': 'roll_all_legs',
                        'current_price': current_price,
                        'dte_extension': 30,  # Add 30 days to expiration
                        'maintain_strikes': True  # Try to keep same strikes
                    }
                    
            elif strategy_type == 'butterfly':
                # Butterfly adjustment logic would go here
                pass
                
            return adjustment_order
            
        except Exception as e:
            self.logger.error(f"Error generating adjustment order: {e}")
            return {
                'action': 'ADJUST',
                'position_id': position.get('position_id', ''),
                'symbol': position.get('symbol', ''),
                'adjustment_type': adjustment_type,
                'error': 'Error generating detailed adjustment order'
            }
            
    def validate_strategy_parameters(self, strategy_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and adjust strategy parameters to ensure they follow risk management guidelines.
        
        Args:
            strategy_parameters: Strategy parameters to validate
            
        Returns:
            Validated and potentially adjusted parameters
        """
        if not strategy_parameters:
            return {}
            
        try:
            # Create a copy of the parameters
            validated_params = strategy_parameters.copy()
            
            # Ensure position sizing parameters are within acceptable ranges
            max_risk_per_trade_pct = validated_params.get('max_risk_per_trade_pct', 2.0)
            if max_risk_per_trade_pct > 5.0:
                self.logger.warning(f"Reducing excessive max_risk_per_trade_pct from {max_risk_per_trade_pct}% to 5.0%")
                validated_params['max_risk_per_trade_pct'] = 5.0
                
            max_position_pct = validated_params.get('max_position_pct', 15.0)
            if max_position_pct > 25.0:
                self.logger.warning(f"Reducing excessive max_position_pct from {max_position_pct}% to 25.0%")
                validated_params['max_position_pct'] = 25.0
                
            # Validate DTE parameters
            min_dte = validated_params.get('min_dte', 30)
            max_dte = validated_params.get('max_dte', 60)
            
            if min_dte < 7:
                self.logger.warning(f"Increasing too low min_dte from {min_dte} to 7")
                validated_params['min_dte'] = 7
                
            if max_dte > 120:
                self.logger.warning(f"Reducing excessive max_dte from {max_dte} to 120")
                validated_params['max_dte'] = 120
                
            # Validate bid-ask spread parameter
            max_bid_ask_spread_pct = validated_params.get('max_bid_ask_spread_pct', 15)
            if max_bid_ask_spread_pct > 25:
                self.logger.warning(f"Reducing excessive max_bid_ask_spread_pct from {max_bid_ask_spread_pct}% to 25%")
                validated_params['max_bid_ask_spread_pct'] = 25
                
            # Validate strike selection parameters
            short_call_delta = validated_params.get('short_call_delta', -0.16)
            short_put_delta = validated_params.get('short_put_delta', 0.16)
            
            # Ensure deltas are within reasonable ranges
            if abs(short_call_delta) > 0.3:
                self.logger.warning(f"Adjusting excessive short_call_delta from {short_call_delta} to -0.3")
                validated_params['short_call_delta'] = -0.3
                
            if abs(short_put_delta) > 0.3:
                self.logger.warning(f"Adjusting excessive short_put_delta from {short_put_delta} to 0.3")
                validated_params['short_put_delta'] = 0.3
                
            return validated_params
            
        except Exception as e:
            self.logger.error(f"Error validating strategy parameters: {e}")
            return strategy_parameters  # Return original parameters on error
