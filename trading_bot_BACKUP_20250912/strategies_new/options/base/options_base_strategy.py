#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Base Strategy Module

This module provides the base class for options trading strategies, with
options-specific functionality built in. It's designed to be subclassed by
specific options strategies.
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date

from trading_bot.strategies.strategy_template import StrategyOptimizable, Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

class OptionType(Enum, AccountAwareMixin):
    """Enum for option types."""
    CALL = "call"
    PUT = "put"

class OptionsSession:
    """Class representing an options trading session."""
    
    def __init__(self, symbol: str, timeframe: str, expiration_date: str = None, 
                 option_chain: pd.DataFrame = None):
        # Initialize account awareness functionality
        AccountAwareMixin.__init__(self)
        """
        Initialize an options trading session.
        
        Args:
            symbol: Underlying asset symbol
            timeframe: Trading timeframe (e.g., '1d', '1h')
            expiration_date: Target expiration date (optional)
            option_chain: DataFrame with option chain data (optional)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.expiration_date = expiration_date
        self.option_chain = option_chain
        self.min_trade_size = 1  # Minimum number of contracts
        
        # Track current market data
        self.current_price = None
        self.current_iv = None
        self.last_updated = None
        
        # Position tracking
        self.active_positions = {}
        self.position_history = []

class OptionsBaseStrategy:
    """
    Base class for options trading strategies.
    
    This class serves as the foundation for all options trading strategies, providing:
    - Option chain filtering
    - Greeks calculation and analysis
    - Implied volatility analysis
    - Expiration management
    - Options-specific position sizing and risk management
    """
    
    # Default parameters specific to options trading
    DEFAULT_PARAMETERS = {
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
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize an options trading strategy.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize parameters with defaults and override with provided parameters
        self.parameters = self.DEFAULT_PARAMETERS.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Store session and data pipeline
        self.session = session
        self.data_pipeline = data_pipeline
        
        # Options-specific member variables
        self.id = f"{self.__class__.__name__}_{session.symbol}"
        self.iv_history = {}  # Track IV history for each underlying
        self.greek_thresholds = {}  # Track adaptive greek thresholds
        self.signals = {}  # Store current signals
        self.positions = []  # Active positions
        self.market_data = pd.DataFrame()  # Current market data
        self.event_bus = EventBus.get_instance()
        
        # Register event handlers
        self.register_event_handlers()
        
        logger.info(f"Initialized options strategy: {self.id} for {session.symbol}")
    
    def register_event_handlers(self):
        """Register for market events."""
        # Register for market data updates
        self.event_bus.register(EventType.MARKET_DATA, self._on_market_data)
        # Register for option chain updates
        self.event_bus.register(EventType.OPTION_CHAIN_UPDATE, self._on_option_chain_update)
        # Register for session events (market open/close)
        self.event_bus.register(EventType.SESSION_EVENT, self._on_session_event)
    
    def _on_market_data(self, event: Event):
        """Handle market data events."""
        if event.data.get('symbol') == self.session.symbol:
            self.market_data = event.data.get('data')
            self.session.current_price = event.data.get('price')
            self.session.last_updated = datetime.now()
            
            # Process data if strategy is active
            if self.is_active:
                self._process_market_data()
    
    def _on_option_chain_update(self, event: Event):
        """Handle option chain update events."""
        if event.data.get('symbol') == self.session.symbol:
            self.session.option_chain = event.data.get('option_chain')
            self.session.current_iv = event.data.get('implied_volatility')
            
            # Update IV history
            if self.session.current_iv is not None:
                self._update_iv_history()
    
    def _on_session_event(self, event: Event):
        """Handle session events like market open/close."""
        if event.data.get('action') == 'open':
            # Start of trading session actions
            pass
        elif event.data.get('action') == 'close':
            # End of trading session actions
            self._on_session_close()
    
    def _process_market_data(self):
        """Process new market data and generate signals."""
        if self.market_data.empty or self.session.option_chain is None:
            return
            
        # Calculate indicators
        indicators = self.calculate_indicators(self.market_data)
        
        # Generate signals
        self.signals = self.generate_signals(self.market_data, indicators)
        
        # Execute trading decisions
        self._execute_signals()
    
    def _update_iv_history(self):
        """Update implied volatility history."""
        if self.session.symbol not in self.iv_history:
            self.iv_history[self.session.symbol] = []
            
        # Add current IV to history
        self.iv_history[self.session.symbol].append({
            'timestamp': datetime.now(),
            'iv': self.session.current_iv
        })
        
        # Limit history size
        max_history = 60  # Keep last 60 data points
        if len(self.iv_history[self.session.symbol]) > max_history:
            self.iv_history[self.session.symbol] = self.iv_history[self.session.symbol][-max_history:]
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """Execute trading signals."""
        # Implementation depends on the concrete strategy
        pass
    
    def _on_session_close(self):
        """Handle end-of-session activities."""
        # Implementation depends on the concrete strategy
        pass
    
    def filter_option_chains(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Filter option chains based on strategy criteria.
        
        Args:
            option_chain: DataFrame with option chain data
            
        Returns:
            Filtered option chain
        """
        if option_chain is None or option_chain.empty:
            return pd.DataFrame()
            
        # Make a copy to avoid modifying the original
        filtered_chain = option_chain.copy()
        
        # Apply filters based on parameters
        # 1. Open interest filter
        if 'open_interest' in filtered_chain.columns:
            filtered_chain = filtered_chain[filtered_chain['open_interest'] >= self.parameters['min_open_interest']]
            
        # 2. Volume filter
        if 'volume' in filtered_chain.columns:
            filtered_chain = filtered_chain[filtered_chain['volume'] >= self.parameters['min_option_volume']]
            
        # 3. Bid-ask spread filter
        if 'bid' in filtered_chain.columns and 'ask' in filtered_chain.columns:
            # Calculate spread percentage
            filtered_chain['spread_pct'] = (filtered_chain['ask'] - filtered_chain['bid']) / filtered_chain['ask']
            filtered_chain = filtered_chain[filtered_chain['spread_pct'] <= self.parameters['max_bid_ask_spread_pct']]
            
        # 4. Days to expiration filter
        if 'days_to_expiration' in filtered_chain.columns:
            min_dte = self.parameters['min_dte']
            max_dte = self.parameters['max_dte']
            filtered_chain = filtered_chain[
                (filtered_chain['days_to_expiration'] >= min_dte) & 
                (filtered_chain['days_to_expiration'] <= max_dte)
            ]
            
        # 5. Greeks filters
        for greek in ['delta', 'theta', 'vega', 'gamma']:
            if greek in filtered_chain.columns:
                min_param = self.parameters.get(f'min_{greek}', None)
                max_param = self.parameters.get(f'max_{greek}', None)
                
                if min_param is not None:
                    if greek == 'delta':  # Delta can be negative for puts
                        filtered_chain = filtered_chain[filtered_chain[greek].abs() >= min_param]
                    else:
                        filtered_chain = filtered_chain[filtered_chain[greek] >= min_param]
                
                if max_param is not None:
                    if greek == 'delta':  # Delta can be negative for puts
                        filtered_chain = filtered_chain[filtered_chain[greek].abs() <= max_param]
                    else:
                        filtered_chain = filtered_chain[filtered_chain[greek] <= max_param]
        
        return filtered_chain
    
    def calculate_implied_volatility_metrics(self, iv_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate implied volatility metrics for decision making.
        
        Args:
            iv_history: List of historical IV values with timestamps
            
        Returns:
            Dictionary of IV metrics
        """
        if not iv_history:
            return {}
            
        # Extract IV values from history
        iv_values = [entry['iv'] for entry in iv_history]
        
        if not iv_values:
            return {}
            
        current_iv = iv_values[-1]
        
        # Calculate basic statistics
        iv_mean = np.mean(iv_values)
        iv_std = np.std(iv_values)
        iv_min = min(iv_values)
        iv_max = max(iv_values)
        
        # Calculate IV percentile and rank
        if iv_max > iv_min:  # Avoid division by zero
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        else:
            iv_rank = 50  # Default to middle if no range
            
        iv_percentile = np.percentile(iv_values, np.searchsorted(
            np.sort(iv_values), current_iv) / len(iv_values) * 100)
        
        # Calculate IV trend (positive = rising, negative = falling)
        if len(iv_values) >= 5:
            iv_trend = np.polyfit(range(len(iv_values[-5:])), iv_values[-5:], 1)[0]
        else:
            iv_trend = 0
            
        # Compile metrics
        return {
            'current_iv': current_iv,
            'iv_mean': iv_mean,
            'iv_std': iv_std,
            'iv_min': iv_min,
            'iv_max': iv_max,
            'iv_rank': iv_rank,
            'iv_percentile': iv_percentile,
            'iv_trend': iv_trend,
            'iv_zscore': (current_iv - iv_mean) / iv_std if iv_std > 0 else 0
        }
    
    def select_expiration(self, option_chain: pd.DataFrame) -> str:
        """
        Select the best expiration date based on strategy parameters.
        
        Args:
            option_chain: DataFrame with option chain data
            
        Returns:
            Selected expiration date (as string in format 'YYYY-MM-DD')
        """
        if option_chain is None or option_chain.empty or 'expiration_date' not in option_chain.columns:
            return None
            
        # Get unique expiration dates
        expiration_dates = option_chain['expiration_date'].unique()
        
        # If no expirations available, return None
        if len(expiration_dates) == 0:
            return None
            
        # Filter by days to expiration
        min_dte = self.parameters['min_dte']
        max_dte = self.parameters['max_dte']
        target_dte = self.parameters['target_dte']
        
        # Current date
        today = datetime.now().date()
        
        # Calculate days to expiration for each date
        dte_map = {}
        valid_expirations = []
        
        for exp_date_str in expiration_dates:
            # Convert to date object if it's a string
            if isinstance(exp_date_str, str):
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
            else:
                exp_date = exp_date_str
                
            # Calculate days to expiration
            dte = (exp_date - today).days
            
            # Check if within valid range
            if min_dte <= dte <= max_dte:
                valid_expirations.append(exp_date_str)
                dte_map[exp_date_str] = dte
                
        # If no valid expirations, return None
        if len(valid_expirations) == 0:
            return None
            
        # Find expiration closest to target DTE
        closest_expiration = min(valid_expirations, key=lambda x: abs(dte_map[x] - target_dte))
        
        return closest_expiration
    
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
            premium: Option premium
            days_to_expiration: Days until expiration
            implied_volatility: Implied volatility of the option
            
        Returns:
            Dictionary of risk/reward metrics
        """
        # Contract multiplier (typically 100 for US options)
        multiplier = 100
        
        # Basic calculations
        max_loss = premium * multiplier  # Maximum loss is premium paid for long options
        
        # Breakeven calculation
        if option_type == OptionType.CALL:
            breakeven = strike_price + premium
            intrinsic_value = max(0, underlying_price - strike_price)
        else:  # PUT
            breakeven = strike_price - premium
            intrinsic_value = max(0, strike_price - underlying_price)
            
        # Calculate time value
        time_value = premium - (intrinsic_value / multiplier)
        
        # Probability of profit based on implied volatility
        # This is a simplification - in reality would use more sophisticated models
        std_dev = underlying_price * implied_volatility * np.sqrt(days_to_expiration / 365)
        
        if option_type == OptionType.CALL:
            # For call, need underlying > breakeven
            z_score = (breakeven - underlying_price) / std_dev
            prob_profit = 1 - norm.cdf(z_score)
        else:  # PUT
            # For put, need underlying < breakeven
            z_score = (underlying_price - breakeven) / std_dev
            prob_profit = norm.cdf(z_score)
            
        # Return to risk ratio (potential return / max loss)
        # Using expected value at 1 and 2 standard deviations
        if option_type == OptionType.CALL:
            potential_price_1std = underlying_price + std_dev
            potential_price_2std = underlying_price + (2 * std_dev)
        else:  # PUT
            potential_price_1std = underlying_price - std_dev
            potential_price_2std = underlying_price - (2 * std_dev)
            
        # Calculate potential values
        if option_type == OptionType.CALL:
            potential_value_1std = max(0, potential_price_1std - strike_price)
            potential_value_2std = max(0, potential_price_2std - strike_price)
        else:  # PUT
            potential_value_1std = max(0, strike_price - potential_price_1std)
            potential_value_2std = max(0, strike_price - potential_price_2std)
            
        # Convert to per-contract value
        potential_value_1std *= multiplier
        potential_value_2std *= multiplier
        
        # Calculate return ratios
        return_ratio_1std = (potential_value_1std - max_loss) / max_loss if max_loss > 0 else 0
        return_ratio_2std = (potential_value_2std - max_loss) / max_loss if max_loss > 0 else 0
        
        # Compile metrics
        return {
            'max_loss': max_loss,
            'breakeven': breakeven,
            'intrinsic_value': intrinsic_value * multiplier,
            'time_value': time_value * multiplier,
            'prob_profit': prob_profit,
            'return_ratio_1std': return_ratio_1std,
            'return_ratio_2std': return_ratio_2std,
            'potential_value_1std': potential_value_1std,
            'potential_value_2std': potential_value_2std
        }
    
    def calculate_position_size(self, 
                              underlying_price: float,
                              option_price: float,
                              max_loss_amount: float) -> int:
        """
        Calculate position size based on risk parameters.
        
        Args:
            underlying_price: Current price of the underlying
            option_price: Price of the option contract (per share)
            max_loss_amount: Maximum dollar amount to risk
            
        Returns:
            Number of contracts to trade
        """
        # Contract multiplier (typically 100 for US options)
        multiplier = 100
        
        # Calculate maximum number of contracts based on risk per trade
        contract_cost = option_price * multiplier
        max_contracts_by_risk = int(max_loss_amount / contract_cost)
        
        # Calculate maximum number of contracts based on position size limit
        portfolio_value = 100000  # Default to $100k if not provided
        max_position_size = portfolio_value * self.parameters['max_position_size_percent']
        max_contracts_by_size = int(max_position_size / contract_cost)
        
        # Calculate maximum contracts based on notional
        notional_per_contract = underlying_price * multiplier
        max_contracts_by_notional = int(self.parameters['max_options_notional'] / notional_per_contract)
        
        # Take the minimum of all constraints
        max_contracts = min(
            max_contracts_by_risk,
            max_contracts_by_size,
            max_contracts_by_notional
        )
        
        # Ensure at least 1 contract if any are allowed
        
        # Calculate position size based on strategy logic
        original_size = max(1, max_contracts) if max_contracts > 0 else 0
        
        # Apply account-aware constraints
        is_day_trade = hasattr(self, 'is_day_trade') and self.is_day_trade
        max_size, _ = self.calculate_max_position_size(price, is_day_trade=is_day_trade)
        
        return min(original_size, max_size)  # Use lower of the two
    
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

    # Abstract methods to be implemented by subclasses
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        raise NotImplementedError("Subclasses must implement calculate_indicators")
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on calculated indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    @property
    def is_active(self) -> bool:
        """Check if the strategy is currently active."""
        # Implement strategy-specific activation logic
        return True
