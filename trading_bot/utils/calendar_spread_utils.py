#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calendar Spread Strategy Utilities

This module provides utility functions and helpers for the calendar spread strategy.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_calendar_spread_config(config_path: str) -> Dict[str, Any]:
    """
    Load calendar spread strategy configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Loaded calendar spread configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading calendar spread configuration: {e}")
        return {}

def flatten_config_to_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary to parameter format.
    
    Args:
        config: Nested configuration dictionary
        
    Returns:
        Flattened parameters dictionary
    """
    parameters = {}
    
    # Process each section of the config
    for section_key, section_value in config.items():
        if isinstance(section_value, dict):
            # Flatten nested dictionary
            for key, value in section_value.items():
                parameters[key] = value
        else:
            # Add top-level parameters directly
            parameters[section_key] = section_value
    
    return parameters

def calculate_days_to_expiration(expiry_date: datetime, current_date: Optional[datetime] = None) -> int:
    """
    Calculate days to expiration for an option.
    
    Args:
        expiry_date: Option expiration date
        current_date: Current date (defaults to today)
        
    Returns:
        Number of days to expiration
    """
    if current_date is None:
        current_date = datetime.now()
    
    # Calculate business days between dates
    delta = expiry_date - current_date
    
    # Return calendar days
    return max(0, delta.days)

def calculate_theoretical_calendar_value(
    short_leg_price: float,
    long_leg_price: float,
    short_leg_dte: int,
    long_leg_dte: int,
    implied_volatility: float,
    strike_price: float,
    underlying_price: float
) -> Tuple[float, float, float]:
    """
    Calculate theoretical value of a calendar spread.
    
    Args:
        short_leg_price: Price of short (front-month) option
        long_leg_price: Price of long (back-month) option
        short_leg_dte: Days to expiration for short leg
        long_leg_dte: Days to expiration for long leg
        implied_volatility: Current implied volatility
        strike_price: Strike price of both options
        underlying_price: Current price of the underlying
        
    Returns:
        Tuple of (theoretical_value, max_profit, max_loss)
    """
    # TODO: Implement Black-Scholes or binomial model for theoretical pricing
    
    # Simplified calculation
    net_debit = long_leg_price - short_leg_price
    
    # Approximate max profit (at expiration of front month)
    # This is simplified and should be replaced with a proper model
    max_profit = long_leg_price * 0.5  # Rough estimate
    
    # Max loss is typically the net debit paid
    max_loss = net_debit
    
    return net_debit, max_profit, max_loss

def calculate_iv_rank(
    current_iv: float,
    historical_iv: List[float],
    lookback_period: int = 252
) -> float:
    """
    Calculate IV Rank (0-100) based on historical implied volatility.
    
    Args:
        current_iv: Current implied volatility
        historical_iv: List of historical implied volatility values
        lookback_period: Number of periods to look back
        
    Returns:
        IV Rank as a percentage (0-100)
    """
    if not historical_iv or len(historical_iv) < 2:
        return 50.0  # Default to middle if not enough data
    
    # Limit to lookback period
    if len(historical_iv) > lookback_period:
        historical_iv = historical_iv[-lookback_period:]
    
    iv_min = min(historical_iv)
    iv_max = max(historical_iv)
    
    # Avoid division by zero
    if iv_max == iv_min:
        return 50.0
    
    # Calculate IV Rank
    iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
    
    return max(0, min(100, iv_rank))

def filter_option_chain(
    option_chain: pd.DataFrame,
    min_dte: int,
    max_dte: int,
    min_open_interest: int,
    max_bid_ask_spread_pct: float
) -> pd.DataFrame:
    """
    Filter option chain based on strategy criteria.
    
    Args:
        option_chain: DataFrame containing option chain data
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        min_open_interest: Minimum open interest
        max_bid_ask_spread_pct: Maximum bid-ask spread as percentage
        
    Returns:
        Filtered option chain
    """
    try:
        # Filter by DTE
        filtered = option_chain[
            (option_chain['dte'] >= min_dte) & 
            (option_chain['dte'] <= max_dte)
        ]
        
        # Filter by open interest
        filtered = filtered[filtered['open_interest'] >= min_open_interest]
        
        # Calculate and filter by bid-ask spread percentage
        filtered['bid_ask_spread_pct'] = (filtered['ask'] - filtered['bid']) / ((filtered['bid'] + filtered['ask']) / 2) * 100
        filtered = filtered[filtered['bid_ask_spread_pct'] <= max_bid_ask_spread_pct]
        
        return filtered
    
    except Exception as e:
        logger.error(f"Error filtering option chain: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def find_atm_strike(option_chain: pd.DataFrame, underlying_price: float) -> float:
    """
    Find the at-the-money (ATM) strike in an option chain.
    
    Args:
        option_chain: DataFrame containing option chain data
        underlying_price: Current price of the underlying
        
    Returns:
        ATM strike price
    """
    try:
        # Get unique strikes
        strikes = option_chain['strike'].unique()
        
        if len(strikes) == 0:
            return underlying_price  # Fallback
        
        # Find closest strike to current price
        atm_strike = strikes[np.abs(strikes - underlying_price).argmin()]
        
        return atm_strike
    
    except Exception as e:
        logger.error(f"Error finding ATM strike: {e}")
        return underlying_price  # Fallback to underlying price

def should_roll_position(
    short_leg_dte: int,
    long_leg_dte: int,
    roll_trigger_dte: int,
    current_iv: float,
    entry_iv: float,
    early_roll_volatility_change_pct: float
) -> Tuple[bool, str]:
    """
    Determine if a calendar spread position should be rolled.
    
    Args:
        short_leg_dte: Days to expiration for short leg
        long_leg_dte: Days to expiration for long leg
        roll_trigger_dte: DTE trigger for rolling short leg
        current_iv: Current implied volatility
        entry_iv: Implied volatility at entry
        early_roll_volatility_change_pct: Volatility change percentage to trigger early roll
        
    Returns:
        Tuple of (should_roll, reason)
    """
    # Check time-based roll trigger
    if short_leg_dte <= roll_trigger_dte:
        return True, "Time-based roll trigger reached"
    
    # Check volatility-based early roll trigger
    iv_change_pct = abs(current_iv - entry_iv) / entry_iv * 100
    if iv_change_pct >= early_roll_volatility_change_pct:
        if current_iv > entry_iv:
            return True, "IV spike - early roll triggered"
        else:
            return True, "IV crash - early roll triggered"
    
    # No roll needed
    return False, "No roll needed"

def calculate_position_size(
    account_equity: float,
    position_size_pct: float,
    net_debit: float,
    contract_multiplier: int = 100
) -> int:
    """
    Calculate appropriate position size for calendar spread.
    
    Args:
        account_equity: Current account equity
        position_size_pct: Percentage of equity to risk per spread
        net_debit: Net debit per spread
        contract_multiplier: Option contract multiplier (typically 100)
        
    Returns:
        Number of spreads to trade
    """
    # Calculate risk amount
    risk_amount = account_equity * (position_size_pct / 100)
    
    # Calculate spread cost
    spread_cost = net_debit * contract_multiplier
    
    # Calculate number of spreads (rounded down)
    num_spreads = int(risk_amount / spread_cost)
    
    # Ensure at least 1 spread
    return max(1, num_spreads) 