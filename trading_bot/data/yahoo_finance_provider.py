#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yahoo Finance Data Provider

This module implements a data provider that fetches market data from Yahoo Finance.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.core.interfaces import DataProvider
from trading_bot.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class YahooFinanceProvider(DataProvider):
    """
    Data provider implementation that uses Yahoo Finance API (via yfinance).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Yahoo Finance data provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_expiry_minutes = config.get("cache_expiry", 30)
        self.data_cache = {}
        self.cache_timestamps = {}
        
        logger.info("Yahoo Finance data provider initialized")
    
    def get_market_data(self, symbols: List[str], start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get market data for a list of symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dictionary mapping symbols to their market data
        """
        if not symbols:
            logger.warning("No symbols provided to get_market_data")
            return {}
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Default to 1 year of data
            start_date = end_date - timedelta(days=365)
        
        result = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                
                if self._is_cache_valid(cache_key):
                    result[symbol] = self.data_cache[cache_key]
                    logger.debug(f"Using cached data for {symbol}")
                    continue
                
                # Fetch data from Yahoo Finance
                logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                # Handle empty data
                if hist.empty:
                    logger.warning(f"No data returned for {symbol}")
                    result[symbol] = pd.DataFrame()
                    continue
                
                # Process data
                hist.reset_index(inplace=True)
                
                # Format columns to standard names
                renamed_columns = {
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                
                # Only rename columns that exist
                existing_columns = {k: v for k, v in renamed_columns.items() if k in hist.columns}
                hist.rename(columns=existing_columns, inplace=True)
                
                # Add to cache
                self.data_cache[cache_key] = hist
                self.cache_timestamps[cache_key] = datetime.now()
                
                # Add to result
                result[symbol] = hist
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                result[symbol] = pd.DataFrame()
        
        return result
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get option chain data for a symbol.
        
        Args:
            symbol: Symbol to fetch option chain for
            expiration_date: Optional specific expiration date
            
        Returns:
            Option chain data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_options_{expiration_date or 'all'}"
            
            if self._is_cache_valid(cache_key):
                return self.data_cache[cache_key]
            
            # Fetch data from Yahoo Finance
            logger.info(f"Fetching option chain for {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get current price for underlying
            current_price = ticker.info.get('regularMarketPrice', None)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return {'expirations': [], 'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'underlying_price': current_price}
            
            # Filter by specific expiration date if provided
            if expiration_date and expiration_date in expirations:
                expirations = [expiration_date]
            elif expiration_date and expiration_date not in expirations:
                logger.warning(f"Expiration date {expiration_date} not available for {symbol}")
                return {'expirations': expirations, 'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'underlying_price': current_price}
            
            # Initialize result structure
            options_data = {
                'expirations': expirations,
                'underlying_price': current_price,
                'chains': {}
            }
            
            # Fetch option chain for each expiration
            for exp_date in expirations:
                try:
                    opt = ticker.option_chain(exp_date)
                    
                    # Process calls
                    calls = opt.calls.copy()
                    if not calls.empty:
                        calls['option_type'] = 'call'
                        
                    # Process puts
                    puts = opt.puts.copy()
                    if not puts.empty:
                        puts['option_type'] = 'put'
                    
                    # Combine and store in result
                    options_data['chains'][exp_date] = {
                        'calls': calls,
                        'puts': puts,
                        'all': pd.concat([calls, puts]) if not calls.empty and not puts.empty else calls if not calls.empty else puts
                    }
                    
                except Exception as e:
                    logger.error(f"Error fetching options for {symbol} expiration {exp_date}: {e}")
            
            # Cache the result
            self.data_cache[cache_key] = options_data
            self.cache_timestamps[cache_key] = datetime.now()
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return {'expirations': [], 'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'underlying_price': None}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.data_cache or cache_key not in self.cache_timestamps:
            return False
        
        # Check if cache has expired
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.cache_expiry_minutes * 60)
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Yahoo Finance provider cache cleared") 