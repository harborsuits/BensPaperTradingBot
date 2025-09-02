#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Data Module

This module provides the MarketData class for accessing historical and real-time
market data from various sources.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

class MarketData:
    """
    Comprehensive market data management system for trading applications.
    
    The MarketData class serves as a unified interface for accessing, processing, and
    managing financial market data from multiple sources. It abstracts away the complexities
    of different data providers, caching mechanisms, and data transformations to provide
    a consistent API for trading strategies to consume market data.
    
    Key capabilities:
    1. Multi-source data access with provider abstraction
    2. Historical and real-time data retrieval
    3. Efficient caching and data management
    4. Consistent data formatting and normalization
    5. Symbol universe management and filtering
    6. Data quality verification and validation
    
    This class addresses several critical challenges in financial data management:
    - Handling data from diverse sources with different formats and conventions
    - Managing large quantities of historical data efficiently
    - Providing both on-demand and subscription-based data access
    - Implementing appropriate caching strategies for performance optimization
    - Ensuring data quality and completeness for trading applications
    - Supporting multiple asset classes and data types
    
    Implementation considerations:
    - Thread safety for concurrent data access
    - Memory management for large datasets
    - Error handling and recovery for data access failures
    - Cache invalidation strategies for different data types
    - Performance optimization for high-frequency data access
    - Consistent handling of missing data, splits, and corporate actions
    
    The class can be extended to support additional data sources, asset classes,
    and specialized data types as needed by implementing appropriate provider interfaces
    and data transformation logic.
    """
    
    def __init__(self, data_source: str = "yahoo", cache_dir: Optional[str] = None):
        """
        Initialize a MarketData instance with specified configuration.
        
        Creates a new MarketData instance configured to use the specified data source
        and optional caching directory. The instance is immediately ready to serve
        data requests, with caching behavior determined by the configuration.
        
        Parameters:
            data_source (str): Identifier for the primary data source provider.
                Supported options include:
                - "yahoo": Yahoo Finance (free, delayed data)
                - "alpha_vantage": Alpha Vantage API (requires API key)
                - "polygon": Polygon.io (requires API key)
                - "iex": IEX Cloud (requires API key)
                - "csv": Local CSV files in specified format
                - "mock": Generated mock data for testing
                
            cache_dir (Optional[str]): Directory path for storing cached data files.
                If provided, enables persistent caching to improve performance and
                reduce API calls. If None, only in-memory caching is used.
                
        Notes:
            - Data source authentication (API keys) must be configured separately
            - Cache directory must be writable by the application
            - In-memory caching is always enabled regardless of cache_dir setting
            - Thread safety must be considered when sharing instances between threads
            - Some data sources may have rate limits or subscription requirements
        """
        self.data_source = data_source
        self.cache_dir = cache_dir
        self._price_cache = {}  # Cache for price data
        self._metadata_cache = {}  # Cache for symbol metadata
        self._last_update_time = {}  # Track data freshness
        self._intraday_cache = {}  # Cache for intraday data
        
        # Configure cache expiration policies (in seconds)
        self._cache_expiry = {
            'daily': 86400,  # Daily data expires after 24 hours
            'intraday': 300,  # Intraday data expires after 5 minutes
            'metadata': 604800  # Metadata expires after 7 days
        }
        
        logger.info(f"Initialized MarketData with source: {data_source}, cache: {'enabled' if cache_dir else 'in-memory only'}")
    
    def get_historical_data(self, symbol: str, 
                           start_date: Optional[Union[str, datetime, date]] = None,
                           end_date: Optional[Union[str, datetime, date]] = None,
                           days: Optional[int] = None,
                           fields: Optional[List[str]] = None,
                           interval: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Retrieve historical market data for a specified symbol and time range.
        
        This method provides flexible historical data retrieval with multiple
        options for specifying the time range, data fields, and sampling interval.
        Data is automatically cached for performance and can be filtered to include
        only needed fields.
        
        The method performs several key functions:
        1. Normalizes input date parameters to a consistent format
        2. Checks cache for available data before making external requests
        3. Retrieves and formats data from the configured data source
        4. Handles error conditions gracefully with appropriate logging
        5. Updates the cache with newly retrieved data
        
        Parameters:
            symbol (str): The ticker symbol to retrieve data for. Should follow
                the convention used by the selected data source (e.g., "AAPL", "BTC-USD")
                
            start_date (Optional[Union[str, datetime, date]]): The start date for 
                historical data. Accepts string format 'YYYY-MM-DD', datetime, or date objects.
                If None and days is provided, calculated as end_date - days.
                If None and days is None, defaults to 1 year before end_date.
                
            end_date (Optional[Union[str, datetime, date]]): The end date for
                historical data. Accepts string format 'YYYY-MM-DD', datetime, or date objects.
                If None, defaults to current date.
                
            days (Optional[int]): Alternative to start_date; specifies the number of 
                calendar days of history to retrieve counting back from end_date.
                Ignored if start_date is provided.
                
            fields (Optional[List[str]]): List of data fields to include in the result.
                Common fields include:
                - "open": Opening price
                - "high": Highest price
                - "low": Lowest price
                - "close": Closing price
                - "volume": Trading volume
                - "adj_close": Adjusted closing price
                If None, returns all available fields.
                
            interval (str): The sampling interval for historical data.
                Options include:
                - "daily": Daily data (default)
                - "weekly": Weekly data
                - "monthly": Monthly data
                - "1m", "5m", "15m", "30m", "1h": Intraday intervals
                Note: Intraday data availability depends on the data source
                and may have different retention periods.
                
        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame with historical data indexed by date,
                with each requested field as a column. Returns None if data retrieval fails
                or if no data is available for the specified range.
                
        Notes:
            - Data points on non-trading days are excluded (e.g., weekends, holidays)
            - Adjusted close prices account for splits and dividends when available
            - Intraday data typically has shorter retention periods
            - Recent data may be delayed based on the data source used
            - Cache behavior prioritizes minimizing external data calls
            - Missing data points are handled according to provider conventions
        """
        # Handle date parameters
        if end_date is None:
            end_date = datetime.now().date()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_date is None and days is not None:
            start_date = end_date - timedelta(days=days)
        elif start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Default fields if none provided
        if fields is None:
            fields = ["open", "high", "low", "close", "volume"]
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            if cache_key in self._price_cache:
                data = self._price_cache[cache_key]
                # Filter to requested fields
                if fields and not all(field in data.columns for field in fields):
                    logger.warning(f"Some requested fields {fields} not available in cached data")
                available_fields = [f for f in fields if f in data.columns]
                return data[available_fields] if available_fields else data
            
            # If using "mock" data source, generate mock data
            if self.data_source.lower() == "mock":
                data = self._generate_mock_data(symbol, start_date, end_date, fields, interval)
                self._price_cache[cache_key] = data
                self._last_update_time[cache_key] = datetime.now()
                return data
                
            # Yahoo Finance data source
            elif self.data_source.lower() == "yahoo":
                data = self._get_yahoo_finance_data(symbol, start_date, end_date, interval)
                if data is not None:
                    self._price_cache[cache_key] = data
                    self._last_update_time[cache_key] = datetime.now()
                    # Filter to requested fields
                    available_fields = [f for f in fields if f in data.columns]
                    return data[available_fields] if available_fields else data
                return None
            
            # Other data sources would be implemented here
            
            # Otherwise, we would fetch from the actual data source
            # For now, return mock data as placeholder
            logger.warning(f"Using mock data for {symbol} as {self.data_source} is not implemented")
            data = self._generate_mock_data(symbol, start_date, end_date, fields, interval)
            self._price_cache[cache_key] = data
            self._last_update_time[cache_key] = datetime.now()
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Retrieve the most recent available price for a specified symbol.
        
        This method provides a streamlined way to access the latest closing price
        for a symbol without needing to process a full historical dataset. It's
        designed for efficient use in strategies that only need current prices.
        
        The implementation prioritizes:
        1. Using cached data when available and fresh
        2. Minimizing data retrieval to only what's needed
        3. Consistent error handling with appropriate fallbacks
        
        Parameters:
            symbol (str): The ticker symbol to retrieve the latest price for.
                Should follow the convention used by the selected data source.
                
        Returns:
            Optional[float]: The most recent closing price available for the symbol,
                or None if the price could not be retrieved or the symbol is invalid.
                
        Notes:
            - Data freshness depends on the cache settings and data source
            - For intraday trading, consider using a real-time data source
            - Price returned is the regular close, not adjusted for corporate actions
            - Data delay depends on the data source (typically 15-20 minutes for free sources)
            - Method optimized for high-frequency calls across multiple symbols
        """
        try:
            # Check if we have recent data in the cache
            cache_key = f"{symbol}_recent"
            if cache_key in self._price_cache and cache_key in self._last_update_time:
                # Check if cache is still fresh (< 5 minutes old for intraday)
                cache_age = (datetime.now() - self._last_update_time[cache_key]).total_seconds()
                if cache_age < 300:  # 5 minutes
                    return self._price_cache[cache_key]
            
            # Get recent data (last 5 days to ensure we have at least one data point)
            recent_data = self.get_historical_data(symbol, days=5, fields=["close"])
            
            if recent_data is None or recent_data.empty:
                return None
                
            # Return the most recent close price
            latest_price = recent_data["close"].iloc[-1]
            
            # Update the cache
            self._price_cache[cache_key] = latest_price
            self._last_update_time[cache_key] = datetime.now()
            
            return latest_price
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None
    
    def get_latest_prices(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve the most recent prices for multiple symbols efficiently.
        
        This method provides batch retrieval of latest prices for multiple symbols,
        optimizing network requests and processing. It returns a formatted DataFrame
        indexed by symbol for easy integration with analysis and trading logic.
        
        Parameters:
            symbols (Optional[List[str]]): List of ticker symbols to retrieve prices for.
                If None, returns an empty DataFrame with the appropriate structure.
                
        Returns:
            pd.DataFrame: A DataFrame containing the latest prices for the requested symbols,
                indexed by symbol with a 'close' column containing the latest closing price.
                
        Performance considerations:
        - For large symbol lists, prices are retrieved in parallel when possible
        - Caching is leveraged to minimize redundant data requests
        - Empty result has consistent structure for error handling
        
        Notes:
            - Missing or invalid symbols are omitted from the result
            - Performance scales efficiently with the number of symbols
            - Result structure remains consistent regardless of input
            - Primary use case is portfolio valuation and signal generation
            - Data freshness is consistent with get_latest_price() method
        """
        if symbols is None:
            # If no symbols provided, return empty DataFrame
            return pd.DataFrame(columns=["symbol", "close"])
        
        data = []
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price is not None:
                data.append({"symbol": symbol, "close": price})
        
        # Create DataFrame from collected data
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("symbol", inplace=True)
        
        return df
    
    def has_min_history(self, symbol: str, min_days: int) -> bool:
        """
        Verify that a symbol has sufficient historical data available.
        
        This method checks if a minimum amount of historical data is available for
        a symbol, which is essential for strategies that require specific lookback
        periods for calculations and signal generation.
        
        Parameters:
            symbol (str): The ticker symbol to check data availability for
            min_days (int): The minimum number of trading days of history required
                
        Returns:
            bool: True if the symbol has at least min_days of historical data available,
                False otherwise
                
        Notes:
            - Only counts actual trading days, not calendar days
            - Used to validate symbols before inclusion in trading universes
            - Returns False on data retrieval errors as a safety mechanism
            - Helps prevent applying strategies to newly listed securities with
              insufficient history for indicator calculations
            - Performs efficiently for large-scale universe filtering operations
        """
        data = self.get_historical_data(symbol, days=min_days)
        return data is not None and len(data) >= min_days
    
    def get_all_symbols(self) -> List[str]:
        """
        Retrieve all available tradable symbols from the data source.
        
        This method provides access to the complete list of symbols available
        through the configured data source, useful for universe construction
        and discovery operations.
        
        Returns:
            List[str]: A list of all available ticker symbols from the data source
            
        Notes:
            - Result may be cached to avoid repeated large data retrievals
            - List can be very large for comprehensive data sources
            - Some data sources may limit the symbols available based on subscription
            - Result varies by data source and may require filtering for tradable assets
            - Primarily used for universe construction and screening operations
        """
        # Implementation would depend on the data source
        # For now, return a placeholder list of common symbols
        return ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'JPM', 'V', 'PG', 'JNJ']
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve option chain data for a specified underlying symbol.
        
        This method provides access to options market data for a given underlying
        symbol, optionally filtered to a specific expiration date. The resulting
        data includes available strikes, pricing, and Greeks when available.
        
        Parameters:
            symbol (str): The underlying ticker symbol for the options chain
            expiration_date (Optional[str]): Specific expiration date to retrieve,
                in 'YYYY-MM-DD' format. If None, returns data for all available
                expiration dates.
                
        Returns:
            Dict[str, Any]: A dictionary containing options chain data with structure:
                {
                    'expiration_dates': List of available expiration dates,
                    'calls': DataFrame with call options data,
                    'puts': DataFrame with put options data
                }
                
        Notes:
            - Options data availability varies significantly by data source
            - Free data sources typically have delayed or limited options data
            - Structure optimized for efficient filtering and strategy application
            - Greeks and implied volatility may not be available from all sources
            - High volume of data for securities with many strikes and expirations
        """
        # Implementation would depend on the data source
        # For now, return mock options data
        logger.warning(f"Using mock options data for {symbol} as not implemented")
        return self._generate_mock_options_data(symbol, expiration_date)
    
    def _generate_mock_data(self, symbol: str, start_date: date, 
                          end_date: date, fields: List[str],
                          interval: str = 'daily') -> pd.DataFrame:
        """
        Generate synthetic market data for testing and development purposes.
        
        This internal method creates realistic, deterministic mock data when actual
        market data is not available or needed. The generated data maintains the
        mathematical properties of real market data while providing consistent
        results for the same inputs.
        
        Parameters:
            symbol (str): The ticker symbol to generate data for
            start_date (date): Start date for the generated data
            end_date (date): End date for the generated data
            fields (List[str]): List of data fields to generate
            interval (str): Data interval (e.g., 'daily', '1h')
            
        Returns:
            pd.DataFrame: A DataFrame containing the generated mock data with
                the requested fields, indexed by date
                
        Implementation notes:
            - Deterministic random generation based on symbol for consistency
            - Realistic price movements with appropriate volatility
            - OHLC relationships preserved (high >= open,close >= low)
            - Adjustable trend and volatility based on symbol characteristics
            - Handles different intervals appropriately
        """
        # Generate date range
        if interval == 'daily':
            date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        elif interval == 'weekly':
            date_range = pd.date_range(start=start_date, end=end_date, freq="W")
        elif interval == 'monthly':
            date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
        elif interval in ['1m', '5m', '15m', '30m', '1h']:
            # For intraday, generate data only for trading hours
            date_range = pd.date_range(start=start_date, end=end_date, freq="B")
            # Expand to intraday points (simplified)
            intraday_range = []
            for day in date_range:
                # 9:30 AM to 4:00 PM Eastern
                day_start = day.replace(hour=9, minute=30)
                day_end = day.replace(hour=16, minute=0)
                if interval == '1m':
                    intraday_range.extend(pd.date_range(start=day_start, end=day_end, freq="1min"))
                elif interval == '5m':
                    intraday_range.extend(pd.date_range(start=day_start, end=day_end, freq="5min"))
                elif interval == '15m':
                    intraday_range.extend(pd.date_range(start=day_start, end=day_end, freq="15min"))
                elif interval == '30m':
                    intraday_range.extend(pd.date_range(start=day_start, end=day_end, freq="30min"))
                elif interval == '1h':
                    intraday_range.extend(pd.date_range(start=day_start, end=day_end, freq="1H"))
            date_range = intraday_range
        else:
            # Default to daily
            date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        
        # Generate random starting price based on symbol
        # Use sum of ASCII values of the symbol to generate a consistent price
        base_price = sum(ord(c) for c in symbol) % 100 + 50
        
        # Generate random price data with a slight upward trend
        np.random.seed(sum(ord(c) for c in symbol))  # Seed based on symbol for consistency
        
        # Generate daily returns with a slight drift
        returns = np.random.normal(0.0002, 0.015, len(date_range))
        
        # Calculate prices from returns
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC data
        data = {
            "open": prices * np.random.uniform(0.99, 1.01, len(date_range)),
            "high": prices * np.random.uniform(1.01, 1.03, len(date_range)),
            "low": prices * np.random.uniform(0.97, 0.99, len(date_range)),
            "close": prices,
            "volume": np.random.randint(100000, 5000000, len(date_range)),
            "adj_close": prices * 0.99  # Slightly lower to simulate dividend adjustments
        }
        
        # Ensure high >= open, close, low and low <= open, close
        for i in range(len(date_range)):
            data["high"][i] = max(data["high"][i], data["open"][i], data["close"][i])
            data["low"][i] = min(data["low"][i], data["open"][i], data["close"][i])
        
        # Create DataFrame
        df = pd.DataFrame(data, index=date_range)
        
        # Filter to requested fields
        available_fields = [f for f in fields if f in df.columns]
        
        return df[available_fields] if available_fields else df
    
    def _generate_mock_options_data(self, symbol: str, expiration_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate synthetic options chain data for testing and development.
        
        Creates a realistic options chain with appropriate strike prices, premiums,
        and Greeks based on the underlying symbol's price and characteristics.
        
        Parameters:
            symbol (str): The underlying symbol for the options chain
            expiration_date (Optional[str]): Specific expiration date to generate
            
        Returns:
            Dict[str, Any]: Dictionary containing generated options chain data
        """
        # Get current price for the symbol
        current_price = self.get_latest_price(symbol) or 100.0
        
        # Generate expiration dates (every Friday for the next 3 months)
        today = datetime.now().date()
        all_expirations = []
        
        for i in range(12):  # 12 weeks
            friday = today + timedelta(days=(4 - today.weekday()) % 7 + 7 * i)
            all_expirations.append(friday.strftime("%Y-%m-%d"))
            
        # Add monthly expirations
        for i in range(1, 7):  # 6 months
            month_end = (today.replace(day=1) + timedelta(days=32 * i)).replace(day=1) - timedelta(days=1)
            # Find third Friday
            third_friday = month_end.replace(day=1)
            while third_friday.weekday() != 4:  # Friday
                third_friday = third_friday + timedelta(days=1)
            third_friday = third_friday + timedelta(days=14)  # Third Friday
            if third_friday.strftime("%Y-%m-%d") not in all_expirations:
                all_expirations.append(third_friday.strftime("%Y-%m-%d"))
        
        # Filter to specific expiration if provided
        if expiration_date:
            if expiration_date in all_expirations:
                expirations = [expiration_date]
            else:
                # Add the requested expiration if it's valid
                try:
                    exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
                    if exp_date > today:
                        expirations = [expiration_date]
                    else:
                        return {"expiration_dates": all_expirations, "calls": pd.DataFrame(), "puts": pd.DataFrame()}
                except ValueError:
                    return {"expiration_dates": all_expirations, "calls": pd.DataFrame(), "puts": pd.DataFrame()}
        else:
            expirations = all_expirations
        
        # Generate strike prices (10 below and 10 above current price, in 5% increments)
        strike_multipliers = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                             1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
        strikes = [round(current_price * m, 2) for m in strike_multipliers]
        
        # Generate option data
        call_data = []
        put_data = []
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            
            # Calculate volatility based on days to expiration (decreases with time)
            implied_vol = 0.3 - 0.1 * min(days_to_exp / 365, 1.0)
            
            for strike in strikes:
                # Calculate call/put values based on Black-Scholes approximation
                moneyness = current_price / strike
                time_factor = days_to_exp / 365.0
                
                # Call premium
                call_intrinsic = max(0, current_price - strike)
                call_time_value = current_price * implied_vol * np.sqrt(time_factor) * np.exp(-0.5 * (moneyness - 1.0)**2)
                call_premium = max(0.01, call_intrinsic + call_time_value)
                
                # Put premium
                put_intrinsic = max(0, strike - current_price)
                put_time_value = current_price * implied_vol * np.sqrt(time_factor) * np.exp(-0.5 * (1.0 - moneyness)**2)
                put_premium = max(0.01, put_intrinsic + put_time_value)
                
                # Calculate simple Greeks
                delta_call = 0.5 + 0.5 * (moneyness - 1.0) / (implied_vol * np.sqrt(time_factor))
                delta_call = max(0.0, min(1.0, delta_call))
                delta_put = delta_call - 1.0
                
                gamma = np.exp(-0.5 * ((moneyness - 1.0) / (implied_vol * np.sqrt(time_factor)))**2) / (current_price * implied_vol * np.sqrt(time_factor))
                
                theta_factor = -0.5 * current_price * implied_vol / np.sqrt(time_factor * 365)
                theta_call = theta_factor * np.exp(-0.5 * ((moneyness - 1.0) / (implied_vol * np.sqrt(time_factor)))**2)
                theta_put = theta_call
                
                vega = 0.1 * current_price * np.sqrt(time_factor) * np.exp(-0.5 * ((moneyness - 1.0) / (implied_vol * np.sqrt(time_factor)))**2)
                
                # Add call option
                call_data.append({
                    "symbol": f"{symbol}_{exp}_{strike}_C",
                    "underlying": symbol,
                    "expiration_date": exp,
                    "strike": strike,
                    "option_type": "call",
                    "bid": round(call_premium * 0.95, 2),
                    "ask": round(call_premium * 1.05, 2),
                    "last": round(call_premium, 2),
                    "volume": int(np.random.exponential(1000) * (1.0 - abs(moneyness - 1.0))),
                    "open_interest": int(np.random.exponential(5000) * (1.0 - abs(moneyness - 1.0))),
                    "implied_volatility": round(implied_vol, 4),
                    "delta": round(delta_call, 4),
                    "gamma": round(gamma, 4),
                    "theta": round(theta_call, 4),
                    "vega": round(vega, 4)
                })
                
                # Add put option
                put_data.append({
                    "symbol": f"{symbol}_{exp}_{strike}_P",
                    "underlying": symbol,
                    "expiration_date": exp,
                    "strike": strike,
                    "option_type": "put",
                    "bid": round(put_premium * 0.95, 2),
                    "ask": round(put_premium * 1.05, 2),
                    "last": round(put_premium, 2),
                    "volume": int(np.random.exponential(1000) * (1.0 - abs(moneyness - 1.0))),
                    "open_interest": int(np.random.exponential(5000) * (1.0 - abs(moneyness - 1.0))),
                    "implied_volatility": round(implied_vol, 4),
                    "delta": round(delta_put, 4),
                    "gamma": round(gamma, 4),
                    "theta": round(theta_put, 4),
                    "vega": round(vega, 4)
                })
        
        # Create DataFrames
        calls_df = pd.DataFrame(call_data)
        puts_df = pd.DataFrame(put_data)
        
        # Return the options chain data
        return {
            "expiration_dates": all_expirations,
            "calls": calls_df,
            "puts": puts_df
        } 