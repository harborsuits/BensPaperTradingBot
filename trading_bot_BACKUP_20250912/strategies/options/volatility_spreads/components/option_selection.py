#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Selection Module for Volatility Strategies

This module handles the selection of optimal options for volatility-based strategies like 
straddles and strangles, using advanced analytics to find the best strike prices, 
expirations, and specific contracts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import logging
import bisect

logger = logging.getLogger(__name__)

class OptionSelector:
    """
    Advanced option contract selector for volatility-based strategies.
    
    This class is responsible for:
    - Finding optimal strike prices based on volatility skew
    - Selecting the most appropriate expiration dates
    - Applying liquidity and cost filters
    - Analyzing Greeks for optimal positioning
    """
    
    def __init__(self, 
                 min_open_interest: int = 100,
                 max_bid_ask_spread_pct: float = 0.10,
                 min_days_to_expiration: int = 14,
                 max_days_to_expiration: int = 45):
        """
        Initialize the option selector.
        
        Args:
            min_open_interest: Minimum open interest for liquidity
            max_bid_ask_spread_pct: Maximum acceptable bid-ask spread as percentage
            min_days_to_expiration: Minimum days to expiration
            max_days_to_expiration: Maximum days to expiration
        """
        # Configuration parameters
        self.min_open_interest = min_open_interest
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.min_dte = min_days_to_expiration
        self.max_dte = max_days_to_expiration
        
        # Working data
        self.indexed_options = {}  # For quick lookup
        
    def prepare_option_chain(self, option_chain: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare an option chain for efficient selection by indexing and pre-filtering.
        
        Args:
            option_chain: Raw option chain DataFrame
            
        Returns:
            Dictionary with indexed and processed option chain data
        """
        if option_chain.empty:
            logger.warning("Empty option chain provided")
            return {}
            
        # Make a copy to avoid modifying the original
        chain = option_chain.copy()
        
        # Add convenience columns
        today = datetime.now().date()
        if 'expiration_date' in chain.columns:
            if isinstance(chain['expiration_date'].iloc[0], str):
                chain['days_to_expiration'] = chain['expiration_date'].apply(
                    lambda x: (datetime.strptime(x, '%Y-%m-%d').date() - today).days
                )
            else:
                chain['days_to_expiration'] = chain['expiration_date'].apply(
                    lambda x: (x - today).days if isinstance(x, datetime) else 0
                )
        elif 'expiration' in chain.columns:
            if isinstance(chain['expiration'].iloc[0], str):
                chain['days_to_expiration'] = chain['expiration'].apply(
                    lambda x: (datetime.strptime(x, '%Y-%m-%d').date() - today).days
                )
            else:
                chain['days_to_expiration'] = chain['expiration'].apply(
                    lambda x: (x - today).days if isinstance(x, datetime) else 0
                )
        
        # Calculate bid-ask spread percentage
        if 'bid' in chain.columns and 'ask' in chain.columns:
            chain['bid_ask_spread_pct'] = (chain['ask'] - chain['bid']) / ((chain['bid'] + chain['ask']) / 2)
            
        # Filter based on configured parameters
        filtered_chain = chain[
            (chain['days_to_expiration'] >= self.min_dte) &
            (chain['days_to_expiration'] <= self.max_dte) &
            (chain.get('open_interest', 1000) >= self.min_open_interest) &
            (chain.get('bid_ask_spread_pct', 0) <= self.max_bid_ask_spread_pct)
        ]
        
        # Index by key fields for quick access
        result = {
            'full_chain': chain,
            'filtered_chain': filtered_chain,
            'expirations': sorted(filtered_chain['expiration'].unique()),
            'call_chains': {},
            'put_chains': {}
        }
        
        # Create indices for fast lookup
        for expiry in result['expirations']:
            expiry_slice = filtered_chain[filtered_chain['expiration'] == expiry]
            
            calls = expiry_slice[expiry_slice['option_type'] == 'call'].sort_values('strike')
            puts = expiry_slice[expiry_slice['option_type'] == 'put'].sort_values('strike')
            
            # Store indexed options
            result['call_chains'][expiry] = calls
            result['put_chains'][expiry] = puts
            
            # Create strike lists for binary search
            result[f'call_strikes_{expiry}'] = calls['strike'].values.tolist()
            result[f'put_strikes_{expiry}'] = puts['strike'].values.tolist()
        
        self.indexed_options = result
        return result
        
    def find_atm_options(self, 
                        current_price: float, 
                        expiration: str,
                        atm_threshold: float = 0.03) -> Dict[str, pd.Series]:
        """
        Find at-the-money options closest to the current price.
        
        Args:
            current_price: Current price of the underlying
            expiration: Target expiration date
            atm_threshold: Maximum allowed distance from ATM as percentage
            
        Returns:
            Dictionary with selected call and put options
        """
        if not self.indexed_options or expiration not in self.indexed_options['expirations']:
            logger.warning(f"Expiration {expiration} not found in indexed options")
            return {}
            
        call_strikes = self.indexed_options.get(f'call_strikes_{expiration}', [])
        put_strikes = self.indexed_options.get(f'put_strikes_{expiration}', [])
        
        if not call_strikes or not put_strikes:
            return {}
            
        # Binary search for closest strikes
        call_idx = bisect.bisect_left(call_strikes, current_price)
        if call_idx >= len(call_strikes):
            call_idx = len(call_strikes) - 1
            
        put_idx = bisect.bisect_right(put_strikes, current_price)
        if put_idx <= 0:
            put_idx = 0
        else:
            put_idx -= 1
            
        # Get the option data
        call_chain = self.indexed_options['call_chains'][expiration]
        put_chain = self.indexed_options['put_chains'][expiration]
        
        # Check if price is within threshold
        call_strike = call_strikes[call_idx]
        put_strike = put_strikes[put_idx]
        
        # For a true straddle, we want the same strike price
        # Find the common strike closest to current price
        common_strikes = set(call_strikes).intersection(set(put_strikes))
        if common_strikes:
            common_strikes_list = sorted(common_strikes)
            closest_idx = bisect.bisect_left(common_strikes_list, current_price)
            if closest_idx >= len(common_strikes_list):
                closest_idx = len(common_strikes_list) - 1
                
            straddle_strike = common_strikes_list[closest_idx]
            
            # Check if within threshold
            if abs(straddle_strike / current_price - 1) <= atm_threshold:
                # Get the option rows
                selected_call = call_chain[call_chain['strike'] == straddle_strike].iloc[0]
                selected_put = put_chain[put_chain['strike'] == straddle_strike].iloc[0]
                
                return {
                    'strategy_type': 'straddle',
                    'call': selected_call,
                    'put': selected_put,
                    'strike': straddle_strike,
                    'expiration': expiration,
                    'atm_distance': abs(straddle_strike / current_price - 1)
                }
        
        # If no suitable straddle found, return empty result
        logger.info(f"No suitable ATM options found within {atm_threshold:.1%} threshold")
        return {}
            
    def find_strangle_options(self,
                             current_price: float,
                             expiration: str,
                             width_pct: float = 0.05) -> Dict[str, pd.Series]:
        """
        Find optimal options for a strangle strategy.
        
        Args:
            current_price: Current price of the underlying
            expiration: Target expiration date
            width_pct: Target width for the strangle as percentage of price
            
        Returns:
            Dictionary with selected call and put options
        """
        if not self.indexed_options or expiration not in self.indexed_options['expirations']:
            logger.warning(f"Expiration {expiration} not found in indexed options")
            return {}
            
        call_chain = self.indexed_options['call_chains'][expiration]
        put_chain = self.indexed_options['put_chains'][expiration]
        
        if call_chain.empty or put_chain.empty:
            return {}
            
        # Target strikes
        target_call_strike = current_price * (1 + width_pct)
        target_put_strike = current_price * (1 - width_pct)
        
        # Find closest strikes
        call_strikes = call_chain['strike'].values
        put_strikes = put_chain['strike'].values
        
        # Use binary search for efficiency
        call_idx = bisect.bisect_left(call_strikes, target_call_strike)
        if call_idx >= len(call_strikes):
            call_idx = len(call_strikes) - 1
            
        put_idx = bisect.bisect_right(put_strikes, target_put_strike)
        if put_idx <= 0:
            put_idx = 0
        else:
            put_idx -= 1
            
        # Get the options
        selected_call = call_chain.iloc[call_idx]
        selected_put = put_chain.iloc[put_idx]
        
        return {
            'strategy_type': 'strangle',
            'call': selected_call,
            'put': selected_put,
            'call_strike': selected_call['strike'],
            'put_strike': selected_put['strike'],
            'expiration': expiration,
            'width_pct': abs(selected_call['strike'] / selected_put['strike'] - 1)
        }
        
    def select_optimal_expiration(self,
                                current_price: float,
                                target_days: int = 30,
                                volatility_curve: pd.Series = None) -> str:
        """
        Select the optimal expiration date based on liquidity, target duration, and vol curve.
        
        Args:
            current_price: Current price of the underlying
            target_days: Target days to expiration
            volatility_curve: Series of IV by expiration (for term structure analysis)
            
        Returns:
            Selected expiration date string
        """
        if not self.indexed_options or not self.indexed_options['expirations']:
            logger.warning("No expirations available in indexed options")
            return None
            
        expirations = self.indexed_options['expirations']
        
        if len(expirations) == 1:
            return expirations[0]
            
        # Calculate days to expiration for each available expiry
        today = datetime.now().date()
        dte_map = {}
        
        for exp in expirations:
            if isinstance(exp, str):
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            else:
                exp_date = exp
                
            dte_map[exp] = (exp_date - today).days
            
        # Find closest to target days
        closest_exp = min(expirations, key=lambda x: abs(dte_map[x] - target_days))
        
        # If we have a volatility curve, we can be smarter about selection
        if volatility_curve is not None and not volatility_curve.empty:
            # Look for volatility term structure anomalies
            # If a nearby expiration has unusually high IV, it might be better
            
            # Calculate average IV change per day
            iv_by_dte = {}
            for exp in expirations:
                if exp in volatility_curve:
                    iv_by_dte[dte_map[exp]] = volatility_curve[exp]
            
            if len(iv_by_dte) >= 2:
                # Look for term structure steepness
                dte_sorted = sorted(iv_by_dte.keys())
                iv_sorted = [iv_by_dte[d] for d in dte_sorted]
                
                # Calculate IV differences
                iv_diffs = [iv_sorted[i+1] - iv_sorted[i] for i in range(len(iv_sorted)-1)]
                dte_diffs = [dte_sorted[i+1] - dte_sorted[i] for i in range(len(dte_sorted)-1)]
                
                # Normalize to daily change
                iv_slopes = [iv_diffs[i] / dte_diffs[i] if dte_diffs[i] > 0 else 0 
                           for i in range(len(iv_diffs))]
                
                # Look for unusually steep changes
                if iv_slopes:
                    mean_slope = sum(iv_slopes) / len(iv_slopes)
                    
                    # If we find a particularly steep part of the curve,
                    # it might be advantageous to position around it
                    for i, slope in enumerate(iv_slopes):
                        # If slope is significantly steeper than average
                        if slope > mean_slope * 1.5:
                            # Choose the expiration before the steep increase
                            candidate_exp = expirations[i]
                            if dte_map[candidate_exp] >= self.min_dte:
                                logger.info(f"Selected expiration {candidate_exp} based on IV term structure anomaly")
                                return candidate_exp
        
        return closest_exp
        
    def evaluate_option_combination(self,
                                  call_option: pd.Series,
                                  put_option: pd.Series,
                                  current_price: float) -> Dict[str, Any]:
        """
        Evaluate a potential option combination for a straddle/strangle strategy.
        
        Args:
            call_option: Selected call option data
            put_option: Selected put option data
            current_price: Current price of the underlying
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate mid prices
        call_price = (call_option['bid'] + call_option['ask']) / 2 if 'bid' in call_option and 'ask' in call_option else call_option.get('price', 0)
        put_price = (put_option['bid'] + put_option['ask']) / 2 if 'bid' in put_option and 'ask' in put_option else put_option.get('price', 0)
        
        # Calculate total premium
        total_premium = call_price + put_price
        
        # Calculate break-even points
        if call_option['strike'] == put_option['strike']:  # Straddle
            upper_breakeven = call_option['strike'] + total_premium
            lower_breakeven = put_option['strike'] - total_premium
            strategy_type = 'straddle'
        else:  # Strangle
            upper_breakeven = call_option['strike'] + total_premium
            lower_breakeven = put_option['strike'] - total_premium
            strategy_type = 'strangle'
            
        # Calculate required move for profit
        required_move_pct = total_premium / current_price
        
        # Get the implied volatility and Greeks if available
        iv_call = call_option.get('implied_volatility', 0)
        iv_put = put_option.get('implied_volatility', 0)
        avg_iv = (iv_call + iv_put) / 2 if iv_call and iv_put else 0
        
        delta_call = call_option.get('delta', 0.5)
        delta_put = put_option.get('delta', -0.5)
        net_delta = delta_call + delta_put
        
        gamma_call = call_option.get('gamma', 0)
        gamma_put = put_option.get('gamma', 0)
        net_gamma = gamma_call + gamma_put
        
        theta_call = call_option.get('theta', 0)
        theta_put = put_option.get('theta', 0)
        net_theta = theta_call + theta_put
        
        vega_call = call_option.get('vega', 0)
        vega_put = put_option.get('vega', 0)
        net_vega = vega_call + vega_put
        
        return {
            'strategy_type': strategy_type,
            'call_strike': call_option['strike'],
            'put_strike': put_option['strike'],
            'call_price': call_price,
            'put_price': put_price,
            'total_premium': total_premium,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'required_move_pct': required_move_pct,
            'average_iv': avg_iv,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'call_option': call_option,
            'put_option': put_option
        }
