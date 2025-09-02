#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Chains Module

This module provides the OptionChains class for accessing options data
for various symbols and expirations.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

class OptionChains:
    """
    Option Chains class for accessing options data.
    
    This class provides methods to access option chain data for various
    symbols, including retrieving calls, puts, and specific expirations.
    """
    
    def __init__(self, data_source: str = "mock"):
        """
        Initialize the OptionChains object.
        
        Args:
            data_source: Source for options data (default: "mock")
        """
        self.data_source = data_source
        self._chains_cache = {}  # Cache for option chain data
        
        logger.info(f"Initialized OptionChains with source: {data_source}")
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get option chain data for a symbol and optional expiration date.
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date in 'YYYY-MM-DD' format (optional)
            
        Returns:
            DataFrame with option chain data
        """
        try:
            # Generate cache key
            cache_key = f"{symbol}_{expiration_date}" if expiration_date else symbol
            
            # Check cache first
            if cache_key in self._chains_cache:
                return self._chains_cache[cache_key]
            
            # Generate mock data if no real data source is configured
            if self.data_source.lower() == "mock" or True:  # Always use mock for now
                chain = self._generate_mock_option_chain(symbol, expiration_date)
                self._chains_cache[cache_key] = chain
                return chain
                
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_expirations(self, symbol: str) -> List[str]:
        """
        Get available expiration dates for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of expiration dates in 'YYYY-MM-DD' format
        """
        try:
            # Get full option chain
            chain = self.get_option_chain(symbol)
            
            if chain.empty:
                return []
                
            # Extract unique expiration dates
            expiration_dates = chain['expiration_date'].unique().tolist()
            
            return sorted(expiration_dates)
            
        except Exception as e:
            logger.error(f"Error getting expirations for {symbol}: {str(e)}")
            return []
    
    def get_calls(self, symbol: str, expiration_date: str) -> pd.DataFrame:
        """
        Get call options for a symbol and expiration date.
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with call options data
        """
        try:
            # Get option chain for the specified expiration
            chain = self.get_option_chain(symbol, expiration_date)
            
            if chain.empty:
                return pd.DataFrame()
                
            # Filter for call options
            calls = chain[chain['option_type'] == 'call']
            
            return calls
            
        except Exception as e:
            logger.error(f"Error getting calls for {symbol} exp {expiration_date}: {str(e)}")
            return pd.DataFrame()
    
    def get_puts(self, symbol: str, expiration_date: str) -> pd.DataFrame:
        """
        Get put options for a symbol and expiration date.
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with put options data
        """
        try:
            # Get option chain for the specified expiration
            chain = self.get_option_chain(symbol, expiration_date)
            
            if chain.empty:
                return pd.DataFrame()
                
            # Filter for put options
            puts = chain[chain['option_type'] == 'put']
            
            return puts
            
        except Exception as e:
            logger.error(f"Error getting puts for {symbol} exp {expiration_date}: {str(e)}")
            return pd.DataFrame()
    
    def get_atm_options(self, symbol: str, expiration_date: str, current_price: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get at-the-money call and put options for a symbol.
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date in 'YYYY-MM-DD' format
            current_price: Current price of the underlying
            
        Returns:
            Tuple of (calls_df, puts_df) with at-the-money options
        """
        try:
            # Get calls and puts
            calls = self.get_calls(symbol, expiration_date)
            puts = self.get_puts(symbol, expiration_date)
            
            if calls.empty or puts.empty:
                return pd.DataFrame(), pd.DataFrame()
                
            # Find closest strikes to current price
            calls['price_diff'] = abs(calls['strike'] - current_price)
            puts['price_diff'] = abs(puts['strike'] - current_price)
            
            # Sort by price difference
            calls = calls.sort_values('price_diff')
            puts = puts.sort_values('price_diff')
            
            # Get ATM options (closest 3 strikes)
            atm_calls = calls.head(3).drop('price_diff', axis=1)
            atm_puts = puts.head(3).drop('price_diff', axis=1)
            
            return atm_calls, atm_puts
            
        except Exception as e:
            logger.error(f"Error getting ATM options for {symbol} exp {expiration_date}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _generate_mock_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate mock option chain data for testing purposes.
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date (optional)
            
        Returns:
            DataFrame with mock option chain data
        """
        # Generate a base price from the symbol (for consistency)
        base_price = sum(ord(c) for c in symbol) % 100 + 50
        
        # Generate expiration dates if not provided
        if expiration_date:
            expiration_dates = [expiration_date]
        else:
            today = date.today()
            expiration_dates = [
                (today + timedelta(days=7+i*30)).strftime('%Y-%m-%d') 
                for i in range(6)  # 6 monthly expirations
            ]
        
        # Generate strike prices around the base price
        strike_range = np.arange(0.7 * base_price, 1.3 * base_price, base_price * 0.025)
        strikes = [round(strike, 2) for strike in strike_range]
        
        # Generate option data
        data = []
        
        for exp_date in expiration_dates:
            # Calculate days to expiration
            exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
            dte = (exp_date_obj - date.today()).days
            
            # Calculate implied volatility based on DTE
            base_iv = 0.3  # 30% base IV
            iv_adjustment = 0.05 * (dte / 30)  # Increase IV for longer DTE
            iv = base_iv + iv_adjustment
            
            # Define IV skew (higher for OTM puts, lower for OTM calls)
            for strike in strikes:
                # Moneyness
                moneyness = strike / base_price
                
                # Adjust IV for strike (skew)
                if moneyness < 1:  # OTM put / ITM call
                    put_iv = iv * (1.1 + 0.2 * (1 - moneyness))
                    call_iv = iv * (0.9 + 0.1 * (1 - moneyness))
                else:  # ITM put / OTM call
                    put_iv = iv * (0.9 + 0.1 * (moneyness - 1))
                    call_iv = iv * (1.0 + 0.15 * (moneyness - 1))
                
                # Generate prices using a simple Black-Scholes approximation
                t = dte / 365.0
                
                # Call price approximation
                if moneyness < 0.8:
                    call_price = 0.05  # Deep OTM call
                    call_delta = 0.1
                elif moneyness > 1.2:
                    call_price = strike - base_price  # Deep ITM call
                    call_delta = 0.9
                else:
                    # Rough approximation for ATM options
                    call_price = base_price * 0.04 * np.sqrt(t) * call_iv
                    call_delta = 0.5 + 0.5 * (moneyness - 1) / (0.2 * np.sqrt(t))
                    call_delta = max(0.01, min(0.99, call_delta))
                    
                    # Adjust based on moneyness
                    if moneyness < 1:
                        call_price *= (0.5 + 0.5 * moneyness)
                    else:
                        call_price *= (1 + 0.3 * (moneyness - 1))
                
                # Put price approximation using put-call parity
                put_price = call_price + strike - base_price
                put_delta = call_delta - 1
                
                # Ensure prices are not negative
                call_price = max(0.01, call_price)
                put_price = max(0.01, put_price)
                
                # Add bid/ask spread
                call_bid = round(max(0.01, call_price * 0.95), 2)
                call_ask = round(call_price * 1.05, 2)
                put_bid = round(max(0.01, put_price * 0.95), 2)
                put_ask = round(put_price * 1.05, 2)
                
                # Generate call option
                call = {
                    'symbol': symbol,
                    'expiration_date': exp_date,
                    'strike': strike,
                    'option_type': 'call',
                    'bid': call_bid,
                    'ask': call_ask,
                    'last': round((call_bid + call_ask) / 2, 2),
                    'volume': int(np.random.randint(10, 1000) * (1 - abs(moneyness - 1))),
                    'open_interest': int(np.random.randint(100, 5000) * (1 - 0.7 * abs(moneyness - 1))),
                    'implied_volatility': round(call_iv, 4),
                    'delta': round(call_delta, 4),
                    'gamma': round(0.05 * (1 - abs(moneyness - 1)), 4),
                    'theta': round(-call_price * 0.01 / max(1, dte), 4),
                    'vega': round(call_price * 0.1, 4),
                    'dte': dte
                }
                
                # Generate put option
                put = {
                    'symbol': symbol,
                    'expiration_date': exp_date,
                    'strike': strike,
                    'option_type': 'put',
                    'bid': put_bid,
                    'ask': put_ask,
                    'last': round((put_bid + put_ask) / 2, 2),
                    'volume': int(np.random.randint(10, 1000) * (1 - abs(moneyness - 1))),
                    'open_interest': int(np.random.randint(100, 5000) * (1 - 0.7 * abs(moneyness - 1))),
                    'implied_volatility': round(put_iv, 4),
                    'delta': round(put_delta, 4),
                    'gamma': round(0.05 * (1 - abs(moneyness - 1)), 4),
                    'theta': round(-put_price * 0.01 / max(1, dte), 4),
                    'vega': round(put_price * 0.1, 4),
                    'dte': dte
                }
                
                data.append(call)
                data.append(put)
        
        # Create DataFrame
        chain = pd.DataFrame(data)
        
        # Filter for specific expiration date if provided
        if expiration_date:
            chain = chain[chain['expiration_date'] == expiration_date]
        
        return chain 