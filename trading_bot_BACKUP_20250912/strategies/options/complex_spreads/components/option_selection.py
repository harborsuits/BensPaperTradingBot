#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Selection Module for Complex Spread Strategies

This module provides functionality for selecting optimal options contracts
for complex option spread strategies like iron condors, butterflies, etc.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm

logger = logging.getLogger(__name__)

class ComplexSpreadOptionSelector:
    """
    Option selector for complex option spread strategies.
    
    This class identifies optimal options contracts for complex spread strategies
    based on various criteria including delta, implied volatility, spread width,
    and other relevant parameters.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the option selector with strategy parameters.
        
        Args:
            parameters: Dictionary of configuration parameters
        """
        self.params = parameters or {}
        self.logger = logger
        
    def find_iron_condor_options(self, symbol: str, option_chain: Any, current_price: float,
                               iv_surface: Dict[str, Any], price_levels: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find optimal options for an iron condor spread.
        
        Args:
            symbol: Underlying symbol
            option_chain: Option chain data
            current_price: Current price of the underlying
            iv_surface: Implied volatility surface data
            price_levels: Important price levels (optional)
            
        Returns:
            Iron condor configuration or None if no suitable spread found
        """
        try:
            # Get parameters
            target_dte = self.params.get('target_dte', 45)
            min_dte = self.params.get('min_dte', 30)
            max_dte = self.params.get('max_dte', 60)
            
            call_spread_width = self.params.get('call_spread_width', 5)
            put_spread_width = self.params.get('put_spread_width', 5)
            
            target_short_call_delta = self.params.get('short_call_delta', -0.16)
            target_short_put_delta = self.params.get('short_put_delta', 0.16)
            
            min_credit = self.params.get('min_credit', 0.10)
            min_credit_to_width_ratio = self.params.get('min_credit_to_width_ratio', 0.15)
            
            # Check if we have valid option chain data
            if not option_chain:
                self.logger.warning("Invalid option chain data")
                return None
                
            # Process option chain based on format
            if isinstance(option_chain, pd.DataFrame):
                return self._find_iron_condor_from_dataframe(
                    symbol, option_chain, current_price, iv_surface, 
                    target_dte, min_dte, max_dte, call_spread_width, put_spread_width,
                    target_short_call_delta, target_short_put_delta, 
                    min_credit, min_credit_to_width_ratio, price_levels
                )
            elif isinstance(option_chain, dict):
                return self._find_iron_condor_from_dict(
                    symbol, option_chain, current_price, iv_surface,
                    target_dte, min_dte, max_dte, call_spread_width, put_spread_width,
                    target_short_call_delta, target_short_put_delta, 
                    min_credit, min_credit_to_width_ratio, price_levels
                )
            else:
                self.logger.warning(f"Unsupported option chain format: {type(option_chain)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding iron condor options for {symbol}: {e}")
            return None
            
    def _find_iron_condor_from_dataframe(self, symbol: str, chain: pd.DataFrame, 
                                      current_price: float, iv_surface: Dict[str, Any],
                                      target_dte: int, min_dte: int, max_dte: int,
                                      call_spread_width: float, put_spread_width: float,
                                      target_short_call_delta: float, target_short_put_delta: float,
                                      min_credit: float, min_credit_to_width_ratio: float,
                                      price_levels: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find iron condor options from DataFrame option chain.
        
        Args:
            symbol: Underlying symbol
            chain: Option chain DataFrame
            current_price: Current price of the underlying
            iv_surface: Implied volatility surface data
            target_dte/min_dte/max_dte: Days to expiration parameters
            call_spread_width/put_spread_width: Width of spreads
            target_short_call_delta/target_short_put_delta: Target deltas for short options
            min_credit: Minimum credit for the spread
            min_credit_to_width_ratio: Minimum ratio of credit to spread width
            price_levels: Important price levels (optional)
            
        Returns:
            Iron condor configuration or None if no suitable spread found
        """
        # Check if we have the required columns
        required_cols = ['expiration', 'strike', 'option_type', 'bid', 'ask', 'delta']
        if not all(col in chain.columns for col in required_cols):
            missing = [col for col in required_cols if col not in chain.columns]
            self.logger.warning(f"Missing columns in option chain: {missing}")
            return None
            
        # Calculate days to expiration for each option
        if 'dte' not in chain.columns:
            # Convert expiration to datetime if it's a string
            if isinstance(chain['expiration'].iloc[0], str):
                chain['expiration'] = pd.to_datetime(chain['expiration'])
                
            # Calculate days to expiration
            today = datetime.now().date()
            chain['dte'] = chain['expiration'].apply(lambda x: (x.date() - today).days if hasattr(x, 'date') else 0)
            
        # Filter for valid DTE range
        valid_dte_chain = chain[(chain['dte'] >= min_dte) & (chain['dte'] <= max_dte)]
        
        if valid_dte_chain.empty:
            self.logger.warning(f"No options with DTE between {min_dte} and {max_dte}")
            return None
            
        # Find the expiration closest to target DTE
        expirations = valid_dte_chain['expiration'].unique()
        target_exp = None
        target_diff = float('inf')
        
        for exp in expirations:
            exp_chain = valid_dte_chain[valid_dte_chain['expiration'] == exp]
            if exp_chain.empty:
                continue
                
            dte = exp_chain['dte'].iloc[0]
            diff = abs(dte - target_dte)
            
            if diff < target_diff:
                target_diff = diff
                target_exp = exp
                
        if target_exp is None:
            self.logger.warning(f"Could not find appropriate expiration for {symbol}")
            return None
            
        # Filter for target expiration
        exp_chain = valid_dte_chain[valid_dte_chain['expiration'] == target_exp]
        dte = exp_chain['dte'].iloc[0]
        
        # Split into calls and puts
        calls = exp_chain[exp_chain['option_type'].str.lower() == 'call']
        puts = exp_chain[exp_chain['option_type'].str.lower() == 'put']
        
        if calls.empty or puts.empty:
            self.logger.warning(f"No calls or puts available for {symbol} with expiration {target_exp}")
            return None
            
        # Filter for valid open interest and bid-ask spread if available
        if 'open_interest' in exp_chain.columns:
            min_oi = self.params.get('min_option_open_interest', 10)
            calls = calls[calls['open_interest'] >= min_oi]
            puts = puts[puts['open_interest'] >= min_oi]
            
        if 'bid' in exp_chain.columns and 'ask' in exp_chain.columns:
            max_spread_pct = self.params.get('max_bid_ask_spread_pct', 15)
            
            # Filter calls with reasonable bid-ask spreads
            calls = calls[(calls['ask'] > 0) & (calls['bid'] > 0)]
            calls = calls[((calls['ask'] - calls['bid']) / calls['ask'] * 100) <= max_spread_pct]
            
            # Filter puts with reasonable bid-ask spreads
            puts = puts[(puts['ask'] > 0) & (puts['bid'] > 0)]
            puts = puts[((puts['ask'] - puts['bid']) / puts['ask'] * 100) <= max_spread_pct]
            
        if calls.empty or puts.empty:
            self.logger.warning(f"No valid calls or puts after filtering for {symbol}")
            return None
            
        # Find short options closest to target delta
        short_call = self._find_option_by_delta(calls, target_short_call_delta)
        short_put = self._find_option_by_delta(puts, target_short_put_delta)
        
        if short_call is None or short_put is None:
            self.logger.warning(f"Could not find appropriate short options for {symbol}")
            return None
            
        # Calculate long strikes based on spread width
        long_call_strike = short_call['strike'] + call_spread_width
        long_put_strike = short_put['strike'] - put_spread_width
        
        # Find long options
        long_call = calls[calls['strike'] == long_call_strike]
        long_put = puts[puts['strike'] == long_put_strike]
        
        # If exact strike not found, get closest available strike
        if long_call.empty:
            long_call = calls[calls['strike'] > short_call['strike']].sort_values('strike').iloc[0] if not calls[calls['strike'] > short_call['strike']].empty else None
                
        if long_put.empty:
            long_put = puts[puts['strike'] < short_put['strike']].sort_values('strike', ascending=False).iloc[0] if not puts[puts['strike'] < short_put['strike']].empty else None
                
        if long_call is None or long_put is None:
            self.logger.warning(f"Could not find appropriate long options for {symbol}")
            return None
            
        # Ensure long_call and long_put are Series not DataFrame
        if isinstance(long_call, pd.DataFrame) and not long_call.empty:
            long_call = long_call.iloc[0]
            
        if isinstance(long_put, pd.DataFrame) and not long_put.empty:
            long_put = long_put.iloc[0]
            
        # Calculate credit and ratios
        short_call_credit = short_call['bid']
        short_put_credit = short_put['bid']
        long_call_debit = long_call['ask']
        long_put_debit = long_put['ask']
        
        total_credit = (short_call_credit + short_put_credit) - (long_call_debit + long_put_debit)
        
        # Calculate actual spread widths
        actual_call_width = long_call['strike'] - short_call['strike']
        actual_put_width = short_put['strike'] - long_put['strike']
        
        # Calculate the risk-reward ratio
        max_risk = (actual_call_width + actual_put_width) - total_credit
        risk_reward_ratio = max_risk / total_credit if total_credit > 0 else float('inf')
        
        # Check if the credit is acceptable
        credit_to_width_ratio = total_credit / (actual_call_width + actual_put_width)
        
        if total_credit < min_credit or credit_to_width_ratio < min_credit_to_width_ratio:
            self.logger.info(f"Iron condor for {symbol} has insufficient credit {total_credit:.2f} or ratio {credit_to_width_ratio:.2f}")
            return None
            
        # Calculate break-even points
        lower_breakeven = short_put['strike'] - total_credit
        upper_breakeven = short_call['strike'] + total_credit
        
        # Calculate probability of profit (approximation using delta)
        short_call_delta_abs = abs(short_call['delta'])
        short_put_delta_abs = abs(short_put['delta'])
        
        prob_call_side = 1 - short_call_delta_abs
        prob_put_side = 1 - short_put_delta_abs
        prob_profit = (prob_call_side + prob_put_side) - 1
        
        # Create result
        iron_condor = {
            'symbol': symbol,
            'strategy_type': 'iron_condor',
            'expiration': target_exp,
            'dte': dte,
            'current_price': current_price,
            
            'short_call_strike': short_call['strike'],
            'short_call_bid': short_call['bid'],
            'short_call_ask': short_call['ask'],
            'short_call_delta': short_call['delta'],
            
            'long_call_strike': long_call['strike'],
            'long_call_bid': long_call['bid'],
            'long_call_ask': long_call['ask'],
            'long_call_delta': long_call['delta'],
            
            'short_put_strike': short_put['strike'],
            'short_put_bid': short_put['bid'],
            'short_put_ask': short_put['ask'],
            'short_put_delta': short_put['delta'],
            
            'long_put_strike': long_put['strike'],
            'long_put_bid': long_put['bid'],
            'long_put_ask': long_put['ask'],
            'long_put_delta': long_put['delta'],
            
            'call_spread_width': actual_call_width,
            'put_spread_width': actual_put_width,
            'total_credit': total_credit,
            'max_risk': max_risk,
            'max_profit': total_credit,
            'risk_reward_ratio': risk_reward_ratio,
            'credit_to_width_ratio': credit_to_width_ratio,
            'lower_breakeven': lower_breakeven,
            'upper_breakeven': upper_breakeven,
            'probability_of_profit': max(0, min(prob_profit, 1)),
            
            'distance_to_upper_pct': (short_call['strike'] - current_price) / current_price * 100,
            'distance_to_lower_pct': (current_price - short_put['strike']) / current_price * 100
        }
        
        # Add option symbols if available
        for leg in ['short_call', 'long_call', 'short_put', 'long_put']:
            option_data = locals()[leg]
            if 'symbol' in option_data:
                iron_condor[f'{leg}_symbol'] = option_data['symbol']
                
        return iron_condor
            
    def _find_iron_condor_from_dict(self, symbol: str, chain: Dict[str, Any], 
                                  current_price: float, iv_surface: Dict[str, Any],
                                  target_dte: int, min_dte: int, max_dte: int,
                                  call_spread_width: float, put_spread_width: float,
                                  target_short_call_delta: float, target_short_put_delta: float,
                                  min_credit: float, min_credit_to_width_ratio: float,
                                  price_levels: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find iron condor options from dictionary option chain.
        
        Args:
            symbol: Underlying symbol
            chain: Option chain dictionary
            current_price: Current price of the underlying
            iv_surface: Implied volatility surface data
            target_dte/min_dte/max_dte: Days to expiration parameters
            call_spread_width/put_spread_width: Width of spreads
            target_short_call_delta/target_short_put_delta: Target deltas for short options
            min_credit: Minimum credit for the spread
            min_credit_to_width_ratio: Minimum ratio of credit to spread width
            price_levels: Important price levels (optional)
            
        Returns:
            Iron condor configuration or None if no suitable spread found
        """
        self.logger.warning("Dictionary option chain format processing not implemented")
        return None
        
    def find_butterfly_options(self, symbol: str, option_chain: Any, current_price: float,
                             iv_surface: Dict[str, Any], price_levels: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find optimal options for a butterfly spread.
        
        Args:
            symbol: Underlying symbol
            option_chain: Option chain data
            current_price: Current price of the underlying
            iv_surface: Implied volatility surface data
            price_levels: Important price levels (optional)
            
        Returns:
            Butterfly configuration or None if no suitable spread found
        """
        # This is a placeholder for butterfly spread logic
        # Implementation would be similar to iron_condor but with butterfly-specific logic
        self.logger.info("Butterfly spread selection not fully implemented")
        return None
        
    def _find_option_by_delta(self, options: pd.DataFrame, target_delta: float) -> Optional[pd.Series]:
        """
        Find the option with delta closest to the target delta.
        
        Args:
            options: DataFrame of options
            target_delta: Target delta value
            
        Returns:
            Option Series or None if not found
        """
        if options.empty or 'delta' not in options.columns:
            return None
            
        # Find the option with delta closest to target
        options['delta_diff'] = (options['delta'] - target_delta).abs()
        closest_option = options.sort_values('delta_diff').iloc[0]
        
        return closest_option
        
    def _find_option_by_strike(self, options: pd.DataFrame, target_strike: float) -> Optional[pd.Series]:
        """
        Find the option with strike closest to the target strike.
        
        Args:
            options: DataFrame of options
            target_strike: Target strike value
            
        Returns:
            Option Series or None if not found
        """
        if options.empty or 'strike' not in options.columns:
            return None
            
        # Find the option with strike closest to target
        options['strike_diff'] = (options['strike'] - target_strike).abs()
        closest_option = options.sort_values('strike_diff').iloc[0]
        
        return closest_option
        
    def calculate_probability_of_profit(self, spread: Dict[str, Any], 
                                      current_price: float, 
                                      dte: int, 
                                      iv: float) -> float:
        """
        Calculate probability of profit for a given spread.
        
        Args:
            spread: Spread configuration
            current_price: Current price of the underlying
            dte: Days to expiration
            iv: Implied volatility
            
        Returns:
            Probability of profit (0-1)
        """
        try:
            if not spread or 'strategy_type' not in spread:
                return 0.0
                
            strategy_type = spread.get('strategy_type')
            
            if strategy_type == 'iron_condor':
                # Iron condor is profitable if price stays between short strikes
                short_call_strike = spread.get('short_call_strike', 0)
                short_put_strike = spread.get('short_put_strike', 0)
                
                if short_call_strike <= 0 or short_put_strike <= 0:
                    return 0.0
                    
                # Calculate standard deviation of price
                time_years = dte / 365.0
                std_dev = current_price * iv * np.sqrt(time_years)
                
                # Calculate z-scores for upper and lower bounds
                upper_z = (short_call_strike - current_price) / std_dev
                lower_z = (short_put_strike - current_price) / std_dev
                
                # Probability price stays below short call and above short put
                prob_below_call = norm.cdf(upper_z)
                prob_above_put = 1 - norm.cdf(lower_z)
                
                # Combined probability
                probability = prob_below_call - (1 - prob_above_put)
                
                return max(0, min(probability, 1))
                
            elif strategy_type == 'butterfly':
                # Placeholder for butterfly POP calculation
                return 0.5
                
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating probability of profit: {e}")
            return 0.0
            
    def evaluate_spread_metrics(self, spread: Dict[str, Any], current_price: float,
                              price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate additional metrics for the spread.
        
        Args:
            spread: Spread configuration
            current_price: Current price of the underlying
            price_data: Historical price data (optional)
            
        Returns:
            Dictionary with additional metrics
        """
        if not spread or 'strategy_type' not in spread:
            return {}
            
        metrics = {}
        
        try:
            strategy_type = spread.get('strategy_type')
            
            # Common metrics
            metrics['risk_reward_ratio'] = spread.get('risk_reward_ratio', 0)
            metrics['return_on_risk'] = spread.get('total_credit', 0) / spread.get('max_risk', 1) * 100 if spread.get('max_risk', 0) > 0 else 0
            metrics['probability_of_profit'] = spread.get('probability_of_profit', 0)
            
            # Risk-adjusted expected return
            expected_return = metrics['probability_of_profit'] * spread.get('max_profit', 0) - (1 - metrics['probability_of_profit']) * spread.get('max_risk', 0)
            metrics['expected_return'] = expected_return
            
            # Strategy-specific metrics
            if strategy_type == 'iron_condor':
                # Price range relative to historical volatility
                if price_data is not None and len(price_data) > 20:
                    returns = price_data['close'].pct_change().dropna()
                    historical_std = returns.std() * np.sqrt(252)  # Annualized
                    
                    upper_range_pct = (spread.get('short_call_strike', current_price * 1.1) - current_price) / current_price
                    lower_range_pct = (current_price - spread.get('short_put_strike', current_price * 0.9)) / current_price
                    
                    metrics['upper_range_std_dev'] = upper_range_pct / (historical_std / np.sqrt(252) * np.sqrt(spread.get('dte', 30) / 365.0))
                    metrics['lower_range_std_dev'] = lower_range_pct / (historical_std / np.sqrt(252) * np.sqrt(spread.get('dte', 30) / 365.0))
                    
                # Skew metrics
                metrics['put_call_premium_ratio'] = spread.get('short_put_bid', 0) / spread.get('short_call_bid', 1) if spread.get('short_call_bid', 0) > 0 else 1
                
            elif strategy_type == 'butterfly':
                # Placeholder for butterfly-specific metrics
                pass
                
            return metrics
                
        except Exception as e:
            self.logger.error(f"Error evaluating spread metrics: {e}")
            return metrics
            
    def adjust_for_price_levels(self, spread: Dict[str, Any], 
                              price_levels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust spread parameters based on key price levels.
        
        Args:
            spread: Spread configuration
            price_levels: Important price levels
            
        Returns:
            Adjusted spread configuration
        """
        if not spread or not price_levels:
            return spread
            
        try:
            strategy_type = spread.get('strategy_type')
            
            if strategy_type == 'iron_condor':
                # Adjust for support and resistance levels
                supports = price_levels.get('support', [])
                resistances = price_levels.get('resistance', [])
                
                # Adjust short call to be above nearest resistance
                short_call_strike = spread.get('short_call_strike', 0)
                for resistance in sorted(resistances):
                    if resistance > short_call_strike:
                        # Consider adjusting if resistance is close
                        if (resistance - short_call_strike) / short_call_strike < 0.03:
                            spread['short_call_strike_adjusted'] = resistance + 1  # Add a buffer
                            break
                
                # Adjust short put to be below nearest support
                short_put_strike = spread.get('short_put_strike', 0)
                for support in sorted(supports, reverse=True):
                    if support < short_put_strike:
                        # Consider adjusting if support is close
                        if (short_put_strike - support) / short_put_strike < 0.03:
                            spread['short_put_strike_adjusted'] = support - 1  # Add a buffer
                            break
                
            return spread
                
        except Exception as e:
            self.logger.error(f"Error adjusting for price levels: {e}")
            return spread
