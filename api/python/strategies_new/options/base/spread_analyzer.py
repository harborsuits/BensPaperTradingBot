#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spread Analyzer Module

This module provides functionality for analyzing, filtering, and selecting 
optimal vertical spreads based on technical indicators, implied volatility, 
and strategy parameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import math
from scipy.stats import norm

from trading_bot.strategies_new.options.base.spread_types import (
    OptionType, VerticalSpreadType, OptionContract, VerticalSpread
)

# Configure logging
logger = logging.getLogger(__name__)

class SpreadAnalyzer:
    """
    Analyzes options chains and constructs optimal vertical spreads based on 
    market conditions and strategy parameters.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the spread analyzer with parameters.
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        # Default parameters
        self.default_params = {
            # Strike selection parameters
            'delta_target_call_long': 0.30,     # Target delta for long call in bull call spread
            'delta_target_call_short': 0.15,    # Target delta for short call in bull call spread
            'delta_target_put_long': -0.30,     # Target delta for long put in bear put spread
            'delta_target_put_short': -0.15,    # Target delta for short put in bear put spread
            
            # Filters
            'min_open_interest': 50,            # Minimum open interest
            'max_spread_pct': 0.15,             # Maximum bid-ask spread as percentage
            'min_spread_width': 2.0,            # Minimum width between strikes ($)
            'max_spread_width': 10.0,           # Maximum width between strikes ($)
            
            # Profitability requirements
            'min_risk_reward_ratio': 0.5,       # Minimum risk/reward ratio
            'max_risk_reward_ratio': 3.0,       # Maximum risk/reward ratio
            'min_credit_received_pct': 0.20,    # Minimum credit as percentage of width
            'min_prob_profit': 0.40,            # Minimum probability of profit
            
            # Delta adjustments based on market conditions
            'bullish_delta_adjustment': 0.05,   # Increase delta targets in bullish conditions
            'bearish_delta_adjustment': -0.05,  # Decrease delta targets in bearish conditions
            'high_iv_width_adjustment': 1.05,   # Increase width in high IV conditions
            'low_iv_width_adjustment': 0.95,    # Decrease width in low IV conditions
        }
        
        # Override defaults with provided parameters
        self.parameters = self.default_params.copy()
        if parameters:
            self.parameters.update(parameters)
    
    def filter_option_chain(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the option chain based on basic criteria.
        
        Args:
            option_chain: DataFrame with option chain data
            
        Returns:
            Filtered option chain
        """
        if option_chain is None or option_chain.empty:
            return pd.DataFrame()
        
        # Copy to avoid modifying original
        filtered_chain = option_chain.copy()
        
        # Apply open interest filter
        if 'open_interest' in filtered_chain.columns:
            min_oi = self.parameters['min_open_interest']
            filtered_chain = filtered_chain[filtered_chain['open_interest'] >= min_oi]
        
        # Apply bid-ask spread filter
        if 'bid' in filtered_chain.columns and 'ask' in filtered_chain.columns:
            max_spread_pct = self.parameters['max_spread_pct']
            filtered_chain['spread_pct'] = (filtered_chain['ask'] - filtered_chain['bid']) / filtered_chain['ask']
            filtered_chain = filtered_chain[filtered_chain['spread_pct'] <= max_spread_pct]
        
        return filtered_chain
    
    def select_option_contracts(self, 
                              option_chain: pd.DataFrame, 
                              target_delta: float,
                              option_type: OptionType,
                              underlying_price: float) -> List[OptionContract]:
        """
        Select option contracts close to the target delta.
        
        Args:
            option_chain: Filtered option chain
            target_delta: Target delta value
            option_type: Type of option (CALL/PUT)
            underlying_price: Current price of underlying asset
            
        Returns:
            List of option contracts matching criteria
        """
        if option_chain is None or option_chain.empty:
            return []
        
        # Filter for option type
        option_type_value = option_type.value
        type_filtered = option_chain[option_chain['option_type'] == option_type_value]
        
        if type_filtered.empty:
            return []
        
        # Convert to list of OptionContract objects
        contracts = []
        for _, row in type_filtered.iterrows():
            try:
                # Convert expiration to date if it's a string
                if isinstance(row['expiration'], str):
                    expiration = datetime.strptime(row['expiration'], '%Y-%m-%d').date()
                else:
                    expiration = row['expiration']
                
                contract = OptionContract(
                    symbol=row['symbol'],
                    option_type=OptionType(row['option_type']),
                    strike=row['strike'],
                    expiration=expiration,
                    bid=row['bid'],
                    ask=row['ask'],
                    last=row.get('last'),
                    volume=row.get('volume'),
                    open_interest=row.get('open_interest'),
                    delta=row.get('delta'),
                    gamma=row.get('gamma'),
                    theta=row.get('theta'),
                    vega=row.get('vega'),
                    implied_volatility=row.get('implied_volatility')
                )
                contracts.append(contract)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error creating option contract: {e}")
        
        # Sort by delta proximity to target
        if contracts and contracts[0].delta is not None:
            # For puts, we're working with negative delta values
            if option_type == OptionType.PUT:
                contracts.sort(key=lambda x: abs(x.delta - target_delta) if x.delta is not None else float('inf'))
            else:
                contracts.sort(key=lambda x: abs(x.delta - target_delta) if x.delta is not None else float('inf'))
        # Fallback to sorting by proximity to underlying price
        else:
            # For calls, we want OTM strikes above the price for short calls, ITM below for long calls
            # For puts, we want OTM strikes below the price for short puts, ITM above for long puts
            if option_type == OptionType.CALL:
                if target_delta > 0.5:  # ITM - long call
                    contracts.sort(key=lambda x: abs(underlying_price - x.strike) if x.strike < underlying_price else float('inf'))
                else:  # OTM - short call
                    contracts.sort(key=lambda x: abs(underlying_price - x.strike) if x.strike > underlying_price else float('inf'))
            else:  # PUT
                if target_delta < -0.5:  # ITM - long put
                    contracts.sort(key=lambda x: abs(underlying_price - x.strike) if x.strike > underlying_price else float('inf'))
                else:  # OTM - short put
                    contracts.sort(key=lambda x: abs(underlying_price - x.strike) if x.strike < underlying_price else float('inf'))
        
        return contracts
    
    def construct_vertical_spread(self,
                                spread_type: VerticalSpreadType,
                                option_chain: pd.DataFrame,
                                underlying_price: float,
                                market_bias: str = 'neutral',
                                iv_percentile: float = 50.0) -> Optional[VerticalSpread]:
        """
        Construct an optimal vertical spread based on market conditions.
        
        Args:
            spread_type: Type of vertical spread to construct
            option_chain: DataFrame with option chain data
            underlying_price: Current price of underlying asset
            market_bias: Market bias ('bullish', 'bearish', or 'neutral')
            iv_percentile: Current IV percentile (0-100)
            
        Returns:
            Constructed vertical spread or None if no suitable spread found
        """
        # Filter the chain
        filtered_chain = self.filter_option_chain(option_chain)
        
        if filtered_chain.empty:
            logger.warning("No options passed the filters")
            return None
        
        # Determine option type for this spread
        option_type = VerticalSpreadType.get_option_type(spread_type)
        
        # Apply market condition adjustments
        delta_adjustment = 0.0
        width_adjustment = 1.0
        
        if market_bias == 'bullish':
            delta_adjustment = self.parameters['bullish_delta_adjustment']
        elif market_bias == 'bearish':
            delta_adjustment = self.parameters['bearish_delta_adjustment']
            
        if iv_percentile > 70:
            width_adjustment = self.parameters['high_iv_width_adjustment']
        elif iv_percentile < 30:
            width_adjustment = self.parameters['low_iv_width_adjustment']
        
        # Select option contracts based on spread type
        long_options = None
        short_options = None
        
        if spread_type == VerticalSpreadType.BULL_CALL_SPREAD:
            # Long lower strike call, short higher strike call
            long_delta = self.parameters['delta_target_call_long'] + delta_adjustment
            short_delta = self.parameters['delta_target_call_short'] + delta_adjustment
            
            long_options = self.select_option_contracts(filtered_chain, long_delta, OptionType.CALL, underlying_price)
            short_options = self.select_option_contracts(filtered_chain, short_delta, OptionType.CALL, underlying_price)
            
        elif spread_type == VerticalSpreadType.BEAR_CALL_SPREAD:
            # Short lower strike call, long higher strike call
            short_delta = self.parameters['delta_target_call_long'] + delta_adjustment
            long_delta = self.parameters['delta_target_call_short'] + delta_adjustment
            
            short_options = self.select_option_contracts(filtered_chain, short_delta, OptionType.CALL, underlying_price)
            long_options = self.select_option_contracts(filtered_chain, long_delta, OptionType.CALL, underlying_price)
            
        elif spread_type == VerticalSpreadType.BULL_PUT_SPREAD:
            # Short higher strike put, long lower strike put
            short_delta = self.parameters['delta_target_put_long'] + delta_adjustment
            long_delta = self.parameters['delta_target_put_short'] + delta_adjustment
            
            short_options = self.select_option_contracts(filtered_chain, short_delta, OptionType.PUT, underlying_price)
            long_options = self.select_option_contracts(filtered_chain, long_delta, OptionType.PUT, underlying_price)
            
        elif spread_type == VerticalSpreadType.BEAR_PUT_SPREAD:
            # Long higher strike put, short lower strike put
            long_delta = self.parameters['delta_target_put_long'] + delta_adjustment
            short_delta = self.parameters['delta_target_put_short'] + delta_adjustment
            
            long_options = self.select_option_contracts(filtered_chain, long_delta, OptionType.PUT, underlying_price)
            short_options = self.select_option_contracts(filtered_chain, short_delta, OptionType.PUT, underlying_price)
        
        if not long_options or not short_options:
            logger.warning(f"Couldn't find suitable options for {spread_type.value}")
            return None
        
        # Create and validate spreads
        valid_spreads = []
        
        # Try different combinations of long and short options
        for long_opt in long_options[:5]:  # Limit to first 5 matches
            for short_opt in short_options[:5]:  # Limit to first 5 matches
                # Ensure correct strike relationship
                if spread_type == VerticalSpreadType.BULL_CALL_SPREAD and long_opt.strike >= short_opt.strike:
                    continue
                if spread_type == VerticalSpreadType.BEAR_CALL_SPREAD and long_opt.strike <= short_opt.strike:
                    continue
                if spread_type == VerticalSpreadType.BULL_PUT_SPREAD and long_opt.strike >= short_opt.strike:
                    continue
                if spread_type == VerticalSpreadType.BEAR_PUT_SPREAD and long_opt.strike <= short_opt.strike:
                    continue
                
                # Ensure same expiration
                if long_opt.expiration != short_opt.expiration:
                    continue
                
                # Create the spread
                spread = VerticalSpread(
                    spread_type=spread_type,
                    long_option=long_opt,
                    short_option=short_opt
                )
                
                # Check width constraints
                min_width = self.parameters['min_spread_width']
                max_width = self.parameters['max_spread_width'] * width_adjustment
                
                if spread.width < min_width or spread.width > max_width:
                    continue
                
                # Check risk/reward ratio
                min_ratio = self.parameters['min_risk_reward_ratio']
                max_ratio = self.parameters['max_risk_reward_ratio']
                
                if spread.risk_reward_ratio < min_ratio or spread.risk_reward_ratio > max_ratio:
                    continue
                
                # Credit spread specific check
                if VerticalSpreadType.is_credit(spread_type):
                    min_credit_pct = self.parameters['min_credit_received_pct']
                    credit_pct = abs(spread.net_premium) / spread.width
                    
                    if credit_pct < min_credit_pct:
                        continue
                
                # Calculate probability of profit
                prob_profit = self.calculate_probability_of_profit(spread, underlying_price)
                min_prob = self.parameters['min_prob_profit']
                
                if prob_profit < min_prob:
                    continue
                
                # Add additional metadata to spread
                spread_info = spread.to_dict()
                spread_info['probability_of_profit'] = prob_profit
                
                valid_spreads.append((spread, spread_info))
        
        if not valid_spreads:
            logger.warning(f"No valid spreads found for {spread_type.value}")
            return None
        
        # Sort spreads by risk/reward ratio and select the best one
        if VerticalSpreadType.is_credit(spread_type):
            # For credit spreads, higher credit percentage is better
            valid_spreads.sort(key=lambda x: abs(x[0].net_premium) / x[0].width, reverse=True)
        else:
            # For debit spreads, lower risk/reward ratio is better
            valid_spreads.sort(key=lambda x: x[0].risk_reward_ratio)
        
        best_spread, best_spread_info = valid_spreads[0]
        
        logger.info(f"Selected {spread_type.value} with R/R {best_spread.risk_reward_ratio:.2f}, "
                  f"width: ${best_spread.width:.2f}, premium: ${abs(best_spread.net_premium):.2f}, "
                  f"PoP: {best_spread_info['probability_of_profit']:.2%}")
        
        return best_spread
    
    def calculate_probability_of_profit(self, 
                                      spread: VerticalSpread, 
                                      underlying_price: float) -> float:
        """
        Calculate the probability of profit for a vertical spread.
        
        Args:
            spread: The vertical spread
            underlying_price: Current price of underlying asset
            
        Returns:
            Probability of profit (0.0 to 1.0)
        """
        # Get implied volatility from options
        iv = (spread.long_option.implied_volatility + spread.short_option.implied_volatility) / 2
        
        if iv is None or math.isnan(iv):
            iv = 0.3  # Default to 30% if IV not available
        
        # Days to expiration
        days_to_expiration = (spread.long_option.expiration - datetime.now().date()).days
        if days_to_expiration <= 0:
            return 0.0
        
        # Calculate standard deviation
        annual_std_dev = underlying_price * iv
        daily_std_dev = annual_std_dev / math.sqrt(365)
        expiration_std_dev = daily_std_dev * math.sqrt(days_to_expiration)
        
        if expiration_std_dev == 0:
            return 0.5
        
        # Calculate probability based on spread type
        if spread.spread_type == VerticalSpreadType.BULL_CALL_SPREAD:
            # Profit if price > breakeven at expiration
            z_score = (spread.breakeven - underlying_price) / expiration_std_dev
            return 1 - norm.cdf(z_score)
            
        elif spread.spread_type == VerticalSpreadType.BEAR_CALL_SPREAD:
            # Profit if price <= short strike at expiration
            z_score = (spread.short_option.strike - underlying_price) / expiration_std_dev
            return norm.cdf(z_score)
            
        elif spread.spread_type == VerticalSpreadType.BULL_PUT_SPREAD:
            # Profit if price >= short strike at expiration
            z_score = (underlying_price - spread.short_option.strike) / expiration_std_dev
            return 1 - norm.cdf(-z_score)
            
        elif spread.spread_type == VerticalSpreadType.BEAR_PUT_SPREAD:
            # Profit if price < breakeven at expiration
            z_score = (underlying_price - spread.breakeven) / expiration_std_dev
            return norm.cdf(z_score)
        
        return 0.5  # Default to 50% if spread type not handled
