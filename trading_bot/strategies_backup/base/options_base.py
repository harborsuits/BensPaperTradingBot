#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Base Strategy Module

This module provides the base class for options trading strategies, with
options-specific functionality built in.
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date

from trading_bot.strategies.strategy_template import StrategyOptimizable, Signal, SignalType, TimeFrame, MarketRegime

logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Enum for option types."""
    CALL = "call"
    PUT = "put"

class OptionsBaseStrategy(StrategyOptimizable):
    """
    Base class for options trading strategies.
    
    This class extends the StrategyOptimizable to add options-specific
    functionality including:
    - Option chain filtering
    - Greeks calculation and analysis
    - Implied volatility analysis
    - Expiration management
    - Options-specific position sizing and risk management
    """
    
    # Default parameters specific to options trading
    DEFAULT_OPTIONS_PARAMS = {
        # Underlying asset filters
        'min_stock_price': 20.0,          # Minimum price for underlying
        'max_stock_price': 1000.0,        # Maximum price for underlying
        'min_avg_volume': 500000,         # Minimum average volume
        
        # Option chain filters
        'min_open_interest': 100,         # Minimum open interest
        'min_option_volume': 10,          # Minimum option volume
        'max_bid_ask_spread_pct': 0.10,   # Maximum bid-ask spread (10%)
        
        # Expiration parameters
        'min_dte': 14,                    # Minimum days to expiration
        'max_dte': 45,                    # Maximum days to expiration
        'target_dte': 30,                 # Target days to expiration
        'avoid_earnings': True,           # Avoid expiration near earnings
        
        # Greeks parameters
        'min_delta': 0.30,                # Minimum delta (absolute value)
        'max_delta': 0.70,                # Maximum delta (absolute value)
        'min_theta': 0.0,                 # Minimum theta
        'max_vega': 1.0,                  # Maximum vega
        'max_gamma': 0.1,                 # Maximum gamma
        
        # Volatility parameters
        'min_iv_percentile': 30,          # Minimum IV percentile
        'max_iv_percentile': 60,          # Maximum IV percentile
        'use_iv_rank': True,              # Whether to use IV rank instead of percentile
        
        # Position sizing and risk
        'max_position_size_percent': 0.02, # Maximum position size (2%)
        'max_loss_percent': 0.01,         # Maximum loss per trade (1%)
        'max_options_notional': 10000,    # Maximum notional value per position
    }
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an options trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_OPTIONS_PARAMS)
            metadata: Strategy metadata
        """
        # Start with default options parameters
        options_params = self.DEFAULT_OPTIONS_PARAMS.copy()
        
        # Override with provided parameters
        if parameters:
            options_params.update(parameters)
        
        # Initialize the parent class
        super().__init__(name=name, parameters=options_params, metadata=metadata)
        
        # Options-specific member variables
        self.iv_history = {}  # Track IV history for each underlying
        self.greek_thresholds = {}  # Track adaptive greek thresholds
        
        logger.info(f"Initialized options strategy: {name}")
    
    def filter_option_chains(self, chains: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter option chains based on strategy criteria.
        
        Args:
            chains: Dictionary mapping symbols to DataFrames with option chain data
            
        Returns:
            Filtered option chains
        """
        filtered_chains = {}
        
        for symbol, chain in chains.items():
            # Skip if no data
            if chain.empty:
                continue
            
            # Apply filters to the option chain
            filtered_chain = chain.copy()
            
            # Filter by open interest
            if 'open_interest' in filtered_chain.columns and self.parameters['min_open_interest'] > 0:
                filtered_chain = filtered_chain[filtered_chain['open_interest'] >= self.parameters['min_open_interest']]
            
            # Filter by option volume
            if 'volume' in filtered_chain.columns and self.parameters['min_option_volume'] > 0:
                filtered_chain = filtered_chain[filtered_chain['volume'] >= self.parameters['min_option_volume']]
            
            # Filter by bid-ask spread
            if 'bid' in filtered_chain.columns and 'ask' in filtered_chain.columns:
                # Calculate spread percentage
                mid_price = (filtered_chain['bid'] + filtered_chain['ask']) / 2
                spread_pct = (filtered_chain['ask'] - filtered_chain['bid']) / mid_price
                
                # Filter by max spread
                filtered_chain = filtered_chain[spread_pct <= self.parameters['max_bid_ask_spread_pct']]
            
            # Filter by expiration dates
            if 'expiration_date' in filtered_chain.columns:
                # Convert expiration dates to datetime if needed
                if not pd.api.types.is_datetime64_dtype(filtered_chain['expiration_date']):
                    filtered_chain['expiration_date'] = pd.to_datetime(filtered_chain['expiration_date'])
                
                # Calculate days to expiration
                now = pd.Timestamp(datetime.now().date())
                filtered_chain['dte'] = (filtered_chain['expiration_date'] - now).dt.days
                
                # Filter by DTE range
                filtered_chain = filtered_chain[
                    (filtered_chain['dte'] >= self.parameters['min_dte']) & 
                    (filtered_chain['dte'] <= self.parameters['max_dte'])
                ]
            
            # Filter by Greeks if available
            if 'delta' in filtered_chain.columns and self.parameters['min_delta'] > 0:
                # Take absolute value of delta for both calls and puts
                abs_delta = filtered_chain['delta'].abs()
                filtered_chain = filtered_chain[
                    (abs_delta >= self.parameters['min_delta']) & 
                    (abs_delta <= self.parameters['max_delta'])
                ]
            
            if 'theta' in filtered_chain.columns and self.parameters['min_theta'] > 0:
                filtered_chain = filtered_chain[filtered_chain['theta'] <= self.parameters['min_theta']]
            
            if 'vega' in filtered_chain.columns and self.parameters['max_vega'] > 0:
                filtered_chain = filtered_chain[filtered_chain['vega'] <= self.parameters['max_vega']]
            
            if 'gamma' in filtered_chain.columns and self.parameters['max_gamma'] > 0:
                filtered_chain = filtered_chain[filtered_chain['gamma'] <= self.parameters['max_gamma']]
            
            # Add filtered chain to result
            if not filtered_chain.empty:
                filtered_chains[symbol] = filtered_chain
        
        logger.info(f"Filtered option chains from {len(chains)} to {len(filtered_chains)} symbols")
        return filtered_chains
    
    def calculate_implied_volatility_metrics(self, underlying: str, 
                                           current_iv: float, 
                                           iv_history: pd.Series) -> Dict[str, float]:
        """
        Calculate implied volatility metrics for decision making.
        
        Args:
            underlying: Underlying symbol
            current_iv: Current implied volatility
            iv_history: Series of historical IV values
            
        Returns:
            Dictionary of IV metrics
        """
        metrics = {
            'current_iv': current_iv
        }
        
        # IV Percentile: percentage of days IV was below current IV
        metrics['iv_percentile'] = (iv_history < current_iv).mean() * 100
        
        # IV Rank: current IV relative to 52-week range
        iv_min = iv_history.min()
        iv_max = iv_history.max()
        
        if iv_max > iv_min:
            metrics['iv_rank'] = (current_iv - iv_min) / (iv_max - iv_min) * 100
        else:
            metrics['iv_rank'] = 50  # Default if range is zero
        
        # IV Z-Score: standard deviations from mean
        iv_mean = iv_history.mean()
        iv_std = iv_history.std()
        
        if iv_std > 0:
            metrics['iv_zscore'] = (current_iv - iv_mean) / iv_std
        else:
            metrics['iv_zscore'] = 0  # Default if std is zero
        
        # IV Trend: 20-day moving average direction
        if len(iv_history) >= 20:
            iv_ma20 = iv_history.rolling(window=20).mean()
            metrics['iv_trend'] = 1 if current_iv > iv_ma20.iloc[-1] else -1
        else:
            metrics['iv_trend'] = 0  # Neutral if not enough history
        
        return metrics
    
    def select_expiration(self, chain: pd.DataFrame) -> str:
        """
        Select the best expiration date based on strategy parameters.
        
        Args:
            chain: DataFrame with option chain data
            
        Returns:
            Selected expiration date (as string in format 'YYYY-MM-DD')
        """
        if chain.empty or 'expiration_date' not in chain.columns:
            return None
        
        # Get unique expiration dates
        expirations = chain['expiration_date'].unique()
        
        # Convert to datetime if needed
        if not isinstance(expirations[0], (datetime, pd.Timestamp, date)):
            expirations = pd.to_datetime(expirations)
        
        # Calculate days to expiration for each date
        today = datetime.now().date()
        dte_values = [(exp.date() if hasattr(exp, 'date') else exp) - today 
                     for exp in expirations]
        dte_days = [dte.days for dte in dte_values]
        
        # Pair expirations with their DTE
        exp_dte_pairs = list(zip(expirations, dte_days))
        
        # Filter by min/max DTE
        valid_pairs = [(exp, dte) for exp, dte in exp_dte_pairs 
                      if self.parameters['min_dte'] <= dte <= self.parameters['max_dte']]
        
        if not valid_pairs:
            logger.warning(f"No expirations found within DTE range {self.parameters['min_dte']}-{self.parameters['max_dte']}")
            return None
        
        # Find expiration closest to target DTE
        target_dte = self.parameters['target_dte']
        
        closest_pair = min(valid_pairs, key=lambda x: abs(x[1] - target_dte))
        selected_expiration = closest_pair[0]
        
        # Convert to string format if it's a datetime
        if isinstance(selected_expiration, (datetime, pd.Timestamp, date)):
            if hasattr(selected_expiration, 'date'):
                selected_expiration = selected_expiration.date()
            return selected_expiration.strftime('%Y-%m-%d')
        
        return str(selected_expiration)
    
    def calculate_option_risk_reward(self, 
                                    option_type: OptionType,
                                    underlying_price: float,
                                    strike_price: float,
                                    premium: float,
                                    days_to_expiration: int,
                                    implied_volatility: float) -> Dict[str, float]:
        """
        Calculate risk/reward metrics for an option.
        
        Args:
            option_type: Type of option (CALL or PUT)
            underlying_price: Current price of the underlying
            strike_price: Strike price of the option
            premium: Option premium (per share)
            days_to_expiration: Days until expiration
            implied_volatility: Option's implied volatility
            
        Returns:
            Dictionary of risk/reward metrics
        """
        metrics = {
            'premium': premium,
            'premium_percent': (premium / underlying_price) * 100
        }
        
        if option_type == OptionType.CALL:
            # For long call
            metrics['breakeven'] = strike_price + premium
            metrics['max_loss'] = premium
            metrics['max_loss_percent'] = (premium / underlying_price) * 100
            
            # Theoretical max profit is unlimited, but we'll use a 2-sigma move
            expected_move = underlying_price * implied_volatility * np.sqrt(days_to_expiration / 365)
            metrics['expected_move'] = expected_move
            
            # Potential profit at 1-sigma move up
            upside_price = underlying_price + expected_move
            if upside_price > metrics['breakeven']:
                metrics['potential_profit'] = upside_price - metrics['breakeven']
                metrics['risk_reward_ratio'] = metrics['potential_profit'] / metrics['max_loss']
            else:
                metrics['potential_profit'] = 0
                metrics['risk_reward_ratio'] = 0
            
        elif option_type == OptionType.PUT:
            # For long put
            metrics['breakeven'] = strike_price - premium
            metrics['max_loss'] = premium
            metrics['max_loss_percent'] = (premium / underlying_price) * 100
            
            # Theoretical max profit is limited by stock price going to zero
            expected_move = underlying_price * implied_volatility * np.sqrt(days_to_expiration / 365)
            metrics['expected_move'] = expected_move
            
            # Potential profit at 1-sigma move down
            downside_price = underlying_price - expected_move
            if downside_price < metrics['breakeven']:
                metrics['potential_profit'] = metrics['breakeven'] - downside_price
                metrics['risk_reward_ratio'] = metrics['potential_profit'] / metrics['max_loss']
            else:
                metrics['potential_profit'] = 0
                metrics['risk_reward_ratio'] = 0
        
        return metrics
    
    def calculate_position_size(self, 
                              underlying_price: float,
                              option_price: float,
                              max_loss_amount: float) -> int:
        """
        Calculate position size based on risk parameters.
        
        Args:
            underlying_price: Current price of the underlying
            option_price: Price of the option contract (per share)
            max_loss_amount: Maximum acceptable loss amount in dollars
            
        Returns:
            Number of contracts to trade
        """
        # Calculate max loss per contract
        max_loss_per_contract = option_price * 100  # 100 shares per contract
        
        # Calculate maximum contracts based on max loss
        if max_loss_per_contract > 0:
            max_contracts_by_risk = int(max_loss_amount / max_loss_per_contract)
        else:
            max_contracts_by_risk = 0
        
        # Calculate maximum contracts based on position size
        portfolio_value = self.parameters.get('portfolio_value', 100000)  # Default to $100k if not provided
        max_position_size = portfolio_value * self.parameters['max_position_size_percent']
        contract_value = option_price * 100
        max_contracts_by_size = int(max_position_size / contract_value)
        
        # Calculate maximum contracts based on notional
        notional_per_contract = underlying_price * 100
        max_contracts_by_notional = int(self.parameters['max_options_notional'] / notional_per_contract)
        
        # Take the minimum of all constraints
        max_contracts = min(
            max_contracts_by_risk,
            max_contracts_by_size,
            max_contracts_by_notional
        )
        
        # Ensure at least 1 contract if any are allowed
        return max(1, max_contracts) if max_contracts > 0 else 0
    
    def adjust_for_implied_volatility(self, 
                                    strategy_type: str,
                                    iv_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on implied volatility conditions.
        
        Args:
            strategy_type: Type of options strategy (e.g., 'long_call', 'iron_condor')
            iv_metrics: Dictionary of IV metrics from calculate_implied_volatility_metrics
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjustments = {}
        
        # Determine IV regime
        iv_percentile = iv_metrics.get('iv_percentile', 50)
        iv_rank = iv_metrics.get('iv_rank', 50)
        
        # Use IV rank or percentile based on parameters
        iv_metric = iv_rank if self.parameters['use_iv_rank'] else iv_percentile
        
        # Low IV environment (below 30th percentile)
        if iv_metric < 30:
            if strategy_type in ['long_call', 'long_put', 'long_straddle', 'long_strangle']:
                # Favor long premium strategies in low IV
                adjustments['confidence_multiplier'] = 1.2
                adjustments['position_size_multiplier'] = 1.1
                adjustments['dte_adjustment'] = 5  # Longer expirations in low IV
            else:
                # Reduce size for short premium strategies
                adjustments['confidence_multiplier'] = 0.8
                adjustments['position_size_multiplier'] = 0.7
                adjustments['dte_adjustment'] = -5  # Shorter expirations for credit strategies
        
        # High IV environment (above 70th percentile)
        elif iv_metric > 70:
            if strategy_type in ['short_call', 'short_put', 'iron_condor', 'credit_spread']:
                # Favor short premium strategies in high IV
                adjustments['confidence_multiplier'] = 1.2
                adjustments['position_size_multiplier'] = 1.1
                adjustments['dte_adjustment'] = 0  # Standard expiration
            else:
                # Reduce size for long premium strategies
                adjustments['confidence_multiplier'] = 0.8
                adjustments['position_size_multiplier'] = 0.7
                adjustments['dte_adjustment'] = -5  # Shorter expirations
        
        # Medium IV environment (between 30-70 percentile)
        else:
            # Neutral settings
            adjustments['confidence_multiplier'] = 1.0
            adjustments['position_size_multiplier'] = 1.0
            adjustments['dte_adjustment'] = 0
        
        # Add IV trend adjustments
        iv_trend = iv_metrics.get('iv_trend', 0)
        if iv_trend > 0:  # Rising IV
            if strategy_type in ['long_call', 'long_put', 'long_straddle']:
                adjustments['confidence_multiplier'] *= 1.1  # More confidence in long options with rising IV
        elif iv_trend < 0:  # Falling IV
            if strategy_type in ['short_call', 'short_put', 'iron_condor']:
                adjustments['confidence_multiplier'] *= 1.1  # More confidence in short options with falling IV
        
        return adjustments 