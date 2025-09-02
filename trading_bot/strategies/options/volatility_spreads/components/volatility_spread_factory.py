#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Spread Factory

This module implements a factory pattern for creating various volatility-based 
option spread strategies, making it easy to generate straddles, strangles, and
other volatility spreads with consistent configuration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging

from .market_analysis import VolatilityAnalyzer
from .option_selection import OptionSelector
from .risk_management import VolatilityRiskManager

logger = logging.getLogger(__name__)

class VolatilitySpreadFactory:
    """
    Factory class for creating volatility-based option spread strategies.
    
    This class uses the factory pattern to create different types of volatility
    spreads (straddles, strangles, etc.) with standardized configuration and
    consistent risk management.
    """
    
    def __init__(self, 
                 volatility_analyzer: VolatilityAnalyzer = None,
                 option_selector: OptionSelector = None,
                 risk_manager: VolatilityRiskManager = None,
                 default_params: Dict[str, Any] = None):
        """
        Initialize the volatility spread factory.
        
        Args:
            volatility_analyzer: Instance of VolatilityAnalyzer
            option_selector: Instance of OptionSelector
            risk_manager: Instance of VolatilityRiskManager
            default_params: Default parameters for strategy creation
        """
        # Initialize component classes if not provided
        self.volatility_analyzer = volatility_analyzer or VolatilityAnalyzer()
        self.option_selector = option_selector or OptionSelector()
        self.risk_manager = risk_manager or VolatilityRiskManager()
        
        # Set default parameters
        self.default_params = default_params or {
            'volatility_threshold': 0.20,        # Historical volatility threshold
            'implied_volatility_rank_min': 30,   # Minimum IV rank to consider
            'atm_threshold': 0.03,               # How close to ATM for straddle
            'strangle_width_pct': 0.05,          # Strike width for strangle as % of price
            'min_dte': 20,                       # Minimum days to expiration
            'max_dte': 45,                       # Maximum days to expiration
            'profit_target_pct': 0.35,           # Profit target as percentage of premium
            'stop_loss_pct': 0.60,               # Stop loss as percentage of premium
            'max_positions': 5,                  # Maximum positions
            'position_size_pct': 0.05,           # Position size as portfolio percentage
            'strategy_variant': 'adaptive',      # 'straddle', 'strangle', or 'adaptive'
            'iv_percentile_threshold': 30,       # IV percentile threshold for strategy selection
            'vix_threshold': 18,                 # VIX threshold for strategy selection
            'event_window_days': 5               # Days before earnings/events to consider
        }
        
    def create_strategy(self, 
                       strategy_type: str, 
                       params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a volatility spread strategy of the specified type.
        
        Args:
            strategy_type: Type of strategy ('straddle', 'strangle', 'adaptive')
            params: Strategy-specific parameters that override defaults
            
        Returns:
            Strategy configuration dictionary
        """
        # Merge default params with provided params
        strategy_params = self.default_params.copy()
        if params:
            strategy_params.update(params)
            
        # Build the strategy configuration
        strategy = {
            'type': strategy_type,
            'name': f"{strategy_type.capitalize()} Volatility Strategy",
            'params': strategy_params,
            'components': {
                'volatility_analyzer': self.volatility_analyzer,
                'option_selector': self.option_selector,
                'risk_manager': self.risk_manager
            },
            'entry_rules': self._get_entry_rules(strategy_type, strategy_params),
            'exit_rules': self._get_exit_rules(strategy_type, strategy_params),
            'position_sizing': self._get_position_sizing_rules(strategy_type, strategy_params)
        }
        
        return strategy
        
    def _get_entry_rules(self, strategy_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define entry rules for the specified strategy type.
        
        Args:
            strategy_type: Type of strategy
            params: Strategy parameters
            
        Returns:
            Entry rules configuration
        """
        # Common rules for all volatility strategies
        rules = {
            'min_historical_volatility': params.get('volatility_threshold', 0.20),
            'min_iv_rank': params.get('implied_volatility_rank_min', 30),
            'min_dte': params.get('min_dte', 20),
            'max_dte': params.get('max_dte', 45),
            'avoid_earnings': True
        }
        
        # Strategy-specific rules
        if strategy_type == 'straddle':
            rules.update({
                'atm_threshold': params.get('atm_threshold', 0.03),
                'max_bid_ask_spread_pct': 0.10,
                'min_open_interest': 100
            })
        elif strategy_type == 'strangle':
            rules.update({
                'strangle_width_pct': params.get('strangle_width_pct', 0.05),
                'max_bid_ask_spread_pct': 0.12,
                'min_open_interest': 50
            })
        else:  # adaptive
            rules.update({
                'atm_threshold': params.get('atm_threshold', 0.03),
                'strangle_width_pct': params.get('strangle_width_pct', 0.05),
                'iv_percentile_threshold': params.get('iv_percentile_threshold', 30),
                'vix_threshold': params.get('vix_threshold', 18),
                'max_bid_ask_spread_pct': 0.10,
                'min_open_interest': 75
            })
            
        return rules
        
    def _get_exit_rules(self, strategy_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define exit rules for the specified strategy type.
        
        Args:
            strategy_type: Type of strategy
            params: Strategy parameters
            
        Returns:
            Exit rules configuration
        """
        # Common exit rules for all volatility strategies
        rules = {
            'profit_target_pct': params.get('profit_target_pct', 0.35),
            'stop_loss_pct': params.get('stop_loss_pct', 0.60),
            'max_days_held': 30,
            'exit_days_before_expiration': 7,
            'iv_crash_exit': True  # Exit on significant IV decrease
        }
        
        # Strategy-specific exit rules
        if strategy_type == 'straddle':
            rules.update({
                'vega_exposure_limit': 0.20,  # Exit if vega exposure exceeds this percentage
                'iv_decrease_threshold': 0.20  # Exit if IV drops by this percentage
            })
        elif strategy_type == 'strangle':
            rules.update({
                'vega_exposure_limit': 0.25,
                'iv_decrease_threshold': 0.25
            })
        else:  # adaptive
            rules.update({
                'vega_exposure_limit': 0.20,
                'iv_decrease_threshold': 0.20,
                'dynamic_exit': True  # Adjust exits based on volatility regime
            })
            
        return rules
        
    def _get_position_sizing_rules(self, strategy_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define position sizing rules for the specified strategy type.
        
        Args:
            strategy_type: Type of strategy
            params: Strategy parameters
            
        Returns:
            Position sizing rules configuration
        """
        # Common position sizing rules
        rules = {
            'max_position_size_pct': params.get('position_size_pct', 0.05),
            'max_positions': params.get('max_positions', 5),
            'max_vega_exposure_pct': 0.20
        }
        
        # Strategy-specific position sizing rules
        if strategy_type == 'straddle':
            rules.update({
                'adjust_for_volatility': True,
                'min_contracts': 1,
                'max_premium_pct': 0.03  # Max premium as percentage of account
            })
        elif strategy_type == 'strangle':
            rules.update({
                'adjust_for_volatility': True,
                'min_contracts': 1,
                'max_premium_pct': 0.025
            })
        else:  # adaptive
            rules.update({
                'adjust_for_volatility': True,
                'min_contracts': 1,
                'max_premium_pct': 0.03,
                'dynamic_sizing': True  # Adjust size based on volatility regime
            })
            
        return rules
        
    def generate_volatility_trade(self,
                                 symbol: str,
                                 current_price: float,
                                 option_chain: pd.DataFrame,
                                 market_data: pd.DataFrame,
                                 volatility_data: Dict[str, Any] = None,
                                 strategy_type: str = None,
                                 params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete volatility trade for the specified symbol.
        
        This is the main factory method that creates a fully configured trade
        with selected options, position sizing, and exit conditions.
        
        Args:
            symbol: Trading symbol
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Historical market data
            volatility_data: Pre-calculated volatility metrics (optional)
            strategy_type: Type of strategy to create (default: use params or default)
            params: Strategy parameters that override defaults
            
        Returns:
            Complete trade configuration with selected options and risk parameters
        """
        # Merge parameters
        merged_params = self.default_params.copy()
        if params:
            merged_params.update(params)
            
        # Determine strategy type if not specified
        if not strategy_type:
            strategy_type = merged_params.get('strategy_variant', 'adaptive')
            
        # If adaptive, determine the appropriate strategy based on volatility
        if strategy_type == 'adaptive':
            strategy_type = self._select_adaptive_strategy(volatility_data, merged_params)
            
        # Prepare option chain data
        self.option_selector.min_open_interest = merged_params.get('min_open_interest', 100)
        self.option_selector.max_bid_ask_spread_pct = merged_params.get('max_bid_ask_spread_pct', 0.10)
        self.option_selector.min_dte = merged_params.get('min_dte', 20)
        self.option_selector.max_dte = merged_params.get('max_dte', 45)
        
        indexed_options = self.option_selector.prepare_option_chain(option_chain)
        
        # Calculate or use provided volatility data
        if not volatility_data:
            # Calculate historical volatility
            hist_vol = self.volatility_analyzer.calculate_historical_volatility(
                market_data, period=20, return_type='latest'
            )
            
            # Calculate volatility percentile if we have a series
            hist_vol_series = self.volatility_analyzer.calculate_historical_volatility(
                market_data, period=20, return_type='series'
            )
            
            vol_percentile = self.volatility_analyzer.calculate_volatility_percentile(
                hist_vol, hist_vol_series
            )
            
            volatility_data = {
                'historical_volatility': hist_vol,
                'volatility_percentile': vol_percentile
            }
        
        # Select optimal expiration
        target_dte = (merged_params.get('min_dte', 20) + merged_params.get('max_dte', 45)) // 2
        expiration = self.option_selector.select_optimal_expiration(
            current_price, target_days=target_dte
        )
        
        if not expiration:
            logger.warning(f"No suitable expiration found for {symbol}")
            return {}
            
        # Select options based on strategy type
        option_data = {}
        
        if strategy_type == 'straddle':
            option_data = self.option_selector.find_atm_options(
                current_price, expiration, 
                atm_threshold=merged_params.get('atm_threshold', 0.03)
            )
        else:  # strangle
            option_data = self.option_selector.find_strangle_options(
                current_price, expiration,
                width_pct=merged_params.get('strangle_width_pct', 0.05)
            )
            
        if not option_data:
            logger.warning(f"No suitable {strategy_type} options found for {symbol}")
            return {}
            
        # Evaluate the option combination
        if 'call' in option_data and 'put' in option_data:
            evaluation = self.option_selector.evaluate_option_combination(
                option_data['call'], option_data['put'], current_price
            )
            option_data.update(evaluation)
            
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            option_data, volatility_data
        )
        
        # Calculate exit conditions
        exit_conditions = self.risk_manager.calculate_exit_conditions(
            option_data, position_size, volatility_data
        )
        
        # Build the complete trade
        trade = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'strategy_type': strategy_type,
            'expiration': expiration,
            'option_data': option_data,
            'position_size': position_size,
            'exit_conditions': exit_conditions,
            'volatility_data': volatility_data,
            'params': merged_params
        }
        
        return trade
        
    def _select_adaptive_strategy(self, volatility_data: Dict[str, Any], params: Dict[str, Any]) -> str:
        """
        Select the appropriate strategy type based on volatility conditions.
        
        Args:
            volatility_data: Volatility metrics
            params: Strategy parameters
            
        Returns:
            Selected strategy type ('straddle' or 'strangle')
        """
        # Default to straddle if no volatility data
        if not volatility_data:
            return 'straddle'
            
        # Extract relevant volatility metrics
        iv_percentile = volatility_data.get('volatility_percentile', 50)
        vol_regime = volatility_data.get('regime', 'neutral')
        hist_vol = volatility_data.get('historical_volatility', 0.2)
        
        # Get threshold parameters
        iv_threshold = params.get('iv_percentile_threshold', 30)
        vix_threshold = params.get('vix_threshold', 18)
        vol_threshold = params.get('volatility_threshold', 0.2)
        
        # Decision logic for adaptive strategy selection
        if vol_regime == 'high_volatility' or iv_percentile > 70:
            # In high volatility environment, prefer strangle (cheaper, wider strikes)
            return 'strangle'
        elif vol_regime == 'low_volatility' or iv_percentile < 30:
            # In low volatility environment, prefer straddle (tighter strikes)
            return 'straddle'
        elif hist_vol > vol_threshold * 1.5:
            # If historical volatility is significantly above threshold, use strangle
            return 'strangle'
        else:
            # Default to straddle for moderate volatility
            return 'straddle'
