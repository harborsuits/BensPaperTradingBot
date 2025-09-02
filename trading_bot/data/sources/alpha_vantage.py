#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage data source implementation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import time
import os
import requests
import pandas as pd
import numpy as np
import json
from dataclasses import asdict

from trading_bot.data.models import MarketData, TimeFrame, DataSource, SymbolMetadata
from trading_bot.data.sources.base import BaseDataSource


logger = logging.getLogger(__name__)


class AlphaVantageDataSource(BaseDataSource):
    """
    Data source implementation for Alpha Vantage API.
    Requires an API key from https://www.alphavantage.co/
    """
    
    # Base URL for Alpha Vantage API
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Rate limiting settings
    CALLS_PER_MINUTE = 5  # Free tier limit
    
    # Map of TimeFrame enum to Alpha Vantage interval strings
    TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: "1min",
        TimeFrame.MINUTE_5: "5min",
        TimeFrame.MINUTE_15: "15min",
        TimeFrame.MINUTE_30: "30min",
        TimeFrame.HOUR_1: "60min",
        TimeFrame.DAY_1: "daily",
        TimeFrame.WEEK_1: "weekly",
        TimeFrame.MONTH_1: "monthly"
    }
    
    def __init__(self, name: str = "alpha_vantage", api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage data source.
        
        Args:
            name: Unique identifier for this data source instance
            api_key: Alpha Vantage API key (required)
        """
        super().__init__(name, DataSource.ALPHA_VANTAGE, api_key)
        
        if not api_key:
            logger.warning("No API key provided for Alpha Vantage. Most endpoints will fail.")
            
        self._last_call_time = 0
    
    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert our TimeFrame enum to Alpha Vantage interval string.
        
        Args:
            timeframe: TimeFrame enum value
            
        Returns:
            Alpha Vantage interval string
        """
        mapping = {
            TimeFrame.MIN_1: "1min",
            TimeFrame.MIN_5: "5min",
            TimeFrame.MIN_15: "15min",
            TimeFrame.MIN_30: "30min",
            TimeFrame.HOUR_1: "60min",
            TimeFrame.DAY_1: "daily",
            TimeFrame.WEEK_1: "weekly",
            TimeFrame.MONTH_1: "monthly",
        }
        
        if timeframe not in mapping:
            logger.warning(f"Unsupported timeframe {timeframe}, defaulting to daily")
            return "daily"
            
        return mapping[timeframe]
    
    def _respect_rate_limit(self):
        """
        Ensure we don't exceed Alpha Vantage's rate limits.
        Implements a simple delay mechanism.
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        # Ensure at least 60/CALLS_PER_MINUTE seconds between calls
        min_interval = 60.0 / self.CALLS_PER_MINUTE
        
        if time_since_last_call < min_interval:
            time_to_sleep = min_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {time_to_sleep:.2f} seconds")
            time.sleep(time_to_sleep)
            
        self._last_call_time = time.time()
    
    def _make_api_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make a request to the Alpha Vantage API with rate limiting.
        
        Args:
            params: Request parameters
            
        Returns:
            API response as a dictionary
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
            
        # Add API key to params
        params['apikey'] = self.api_key
        
        # Respect rate limit
        self._respect_rate_limit()
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}
                
            # Check for information messages (often rate limit warnings)
            if 'Information' in data:
                logger.warning(f"Alpha Vantage API info: {data['Information']}")
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage API note: {data['Note']}")
                return {}
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return {}
        except ValueError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {}
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                timeframe: TimeFrame = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Retrieve market data for a symbol within the specified date range.
        
        Args:
            symbol: Trading symbol to retrieve data for
            start_date: Start date for the data range
            end_date: End date for the data range
            timeframe: Time frame for the data
            
        Returns:
            List of MarketData objects
        """
        interval = self._convert_timeframe(timeframe)
        
        # Determine which function to use based on the interval
        if interval in ['daily', 'weekly', 'monthly']:
            function = f"TIME_SERIES_{interval.upper()}"
            time_series_key = f"Time Series ({interval.capitalize()})"
            # For daily data we can use adjusted to get adjusted close
            if interval == 'daily':
                function = "TIME_SERIES_DAILY_ADJUSTED"
                time_series_key = "Time Series (Daily)"
        else:
            function = "TIME_SERIES_INTRADAY"
            time_series_key = f"Time Series ({interval})"
            
        # Prepare API parameters
        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': 'full'  # To get as much historical data as possible
        }
        
        # Add interval parameter for intraday data
        if function == "TIME_SERIES_INTRADAY":
            params['interval'] = interval
            
        # Make API request
        data = self._make_api_request(params)
        
        if not data or time_series_key not in data:
            logger.warning(f"No data returned for {symbol} with timeframe {timeframe}")
            return []
            
        # Extract time series data and convert to DataFrame
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns to match our schema (column names vary by endpoint)
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
            '5. adjusted close': 'adj_close',
            '6. volume': 'volume'  # For adjusted daily data
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert index to datetime column
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Filter by date range
        df = df[(df['timestamp'] >= pd.Timestamp(start_date)) & 
                (df['timestamp'] <= pd.Timestamp(end_date))]
                
        # Add symbol column
        df['symbol'] = symbol
        
        # Convert columns to appropriate types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close']  # Use regular close if adjusted not available
        
        # Sort by timestamp (descending)
        df.sort_values('timestamp', ascending=False, inplace=True)
        
        # Convert to our data model
        market_data_list = self.from_dataframe(df)
        
        return market_data_list
    
    def get_latest(self, symbol: str, timeframe: TimeFrame = TimeFrame.DAY_1) -> Optional[MarketData]:
        """
        Get the latest market data for a symbol.
        
        Alpha Vantage doesn't have a specific endpoint for latest data,
        so we retrieve recent data and return the most recent entry.
        
        Args:
            symbol: Trading symbol to retrieve data for
            timeframe: Time frame for the data
            
        Returns:
            Latest MarketData object or None if not available
        """
        # For latest data, we'll retrieve the last month and take the most recent
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = self.get_data(symbol, start_date, end_date, timeframe)
        if not data:
            return None
            
        # Data is already sorted by timestamp in descending order
        return data[0]
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of available symbols that match a search query.
        
        Note: Alpha Vantage doesn't provide a complete list of symbols,
        but offers a search endpoint. This implementation returns an empty list
        as a full implementation would require a specific search query.
        
        Returns:
            Empty list (feature not fully supported)
        """
        logger.warning("Alpha Vantage doesn't provide a direct API for listing all available symbols")
        return []
    
    def get_symbol_metadata(self, symbol: str) -> Optional[SymbolMetadata]:
        """
        Get metadata for a specific symbol from Alpha Vantage.
        
        Args:
            symbol: Symbol to retrieve metadata for
            
        Returns:
            SymbolMetadata object or None if not available
        """
        # Use the OVERVIEW function to get company information
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        data = self._make_api_request(params)
        
        if not data or 'Symbol' not in data:
            logger.warning(f"No metadata available for symbol {symbol}")
            # Try the search function as a fallback
            return self._get_metadata_from_search(symbol)
            
        return SymbolMetadata(
            symbol=data.get('Symbol', symbol),
            name=data.get('Name', symbol),
            asset_type='Equity',  # Overview only works for equities
            exchange=data.get('Exchange', 'unknown'),
            currency=data.get('Currency', 'USD'),
            sector=data.get('Sector', None),
            industry=data.get('Industry', None),
            country=data.get('Country', None),
            description=data.get('Description', None)
        )
    
    def _get_metadata_from_search(self, symbol: str) -> Optional[SymbolMetadata]:
        """
        Fallback method to get basic metadata from the SYMBOL_SEARCH endpoint.
        
        Args:
            symbol: Symbol to search for
            
        Returns:
            SymbolMetadata object or None if not available
        """
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': symbol
        }
        
        data = self._make_api_request(params)
        
        if not data or 'bestMatches' not in data or not data['bestMatches']:
            logger.warning(f"No search results for symbol {symbol}")
            return None
            
        # Find the exact match if possible
        exact_matches = [match for match in data['bestMatches'] 
                        if match.get('1. symbol', '').upper() == symbol.upper()]
        
        match = exact_matches[0] if exact_matches else data['bestMatches'][0]
        
        return SymbolMetadata(
            symbol=match.get('1. symbol', symbol),
            name=match.get('2. name', symbol),
            asset_type=match.get('3. type', 'unknown'),
            exchange=match.get('4. region', 'unknown'),
            currency=match.get('8. currency', 'USD'),
            sector=None,
            industry=None,
            country=None,
            description=None
        )
    
    def get_technical_indicator(
        self, 
        symbol: str, 
        indicator: str,
        time_period: int = 14,
        series_type: str = "close",
        timeframe: TimeFrame = TimeFrame.DAY_1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get technical indicator data for a symbol.
        
        Args:
            symbol: The stock symbol to retrieve data for
            indicator: The indicator to retrieve (e.g., SMA, EMA, RSI)
            time_period: The time period for the indicator
            series_type: The price series to use (open, high, low, close)
            timeframe: The time frame to retrieve data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with indicator data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
            # Map timeframe to Alpha Vantage interval
            interval = self.TIMEFRAME_MAP.get(timeframe)
            if not interval or timeframe in [TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
                interval = "daily"  # Default to daily for weekly/monthly or unsupported timeframes
                
            # Build parameters
            params = {
                "function": indicator,
                "symbol": symbol,
                "interval": interval,
                "time_period": time_period,
                "series_type": series_type,
                "apikey": self.api_key
            }
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return pd.DataFrame()
                
            if "Note" in data and "API call frequency" in data["Note"]:
                logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
                
            # Extract indicator data
            indicator_key = None
            for key in data.keys():
                if "Technical Analysis" in key:
                    indicator_key = key
                    break
                    
            if not indicator_key:
                logger.error(f"No indicator data found in response: {data.keys()}")
                return pd.DataFrame()
                
            # Parse indicator data
            indicator_data = []
            time_series = data[indicator_key]
            
            for date_str, values in time_series.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                
                # Skip if outside of date range
                if date < start_date or date > end_date:
                    continue
                    
                # Extract indicator value(s)
                row = {"timestamp": date}
                
                for key, value in values.items():
                    row[key.split(". ")[1] if ". " in key else key] = float(value)
                    
                indicator_data.append(row)
                
            # Create DataFrame
            df = pd.DataFrame(indicator_data)
            
            if df.empty:
                logger.warning(f"No data found for {indicator} on {symbol}")
                return pd.DataFrame()
                
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Retrieved {len(df)} data points for {indicator} on {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving {indicator} for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_indicators(
        self, 
        symbol: str,
        indicators: List[Dict[str, Any]],
        timeframe: TimeFrame = TimeFrame.DAY_1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get multiple technical indicators for a symbol and merge them.
        
        Args:
            symbol: The stock symbol to retrieve data for
            indicators: List of indicator configurations (e.g., [{"name": "SMA", "time_period": 20}, ...])
            timeframe: The time frame to retrieve data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with merged indicator data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
            # Get price data first
            price_data = self.get_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if not price_data:
                logger.error(f"No price data available for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            price_df = pd.DataFrame([asdict(data) for data in price_data])
            price_df.set_index("timestamp", inplace=True)
            
            # Get each indicator and merge
            for indicator_config in indicators:
                indicator_name = indicator_config["name"]
                time_period = indicator_config.get("time_period", 14)
                series_type = indicator_config.get("series_type", "close")
                
                indicator_df = self.get_technical_indicator(
                    symbol=symbol,
                    indicator=indicator_name,
                    time_period=time_period,
                    series_type=series_type,
                    timeframe=timeframe,
                    start_date=start_date - timedelta(days=100),  # Get extra data for indicators that need more history
                    end_date=end_date
                )
                
                if not indicator_df.empty:
                    # Add prefix to avoid column name conflicts
                    prefix = indicator_name.lower()
                    if time_period != 14:
                        prefix += f"_{time_period}"
                        
                    indicator_df = indicator_df.add_prefix(f"{prefix}_")
                    
                    # Merge with price data
                    price_df = price_df.join(indicator_df, how="left")
            
            # Filter to requested date range
            price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
            
            logger.info(f"Retrieved data for {len(indicators)} indicators on {symbol}")
            return price_df
            
        except Exception as e:
            logger.error(f"Error retrieving multiple indicators for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview information.
        
        Args:
            symbol: The stock symbol to retrieve data for
            
        Returns:
            Dictionary with company information
        """
        try:
            # Build parameters
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return {}
                
            data = response.json()
            
            # Check for empty response or error messages
            if not data or "Error Message" in data:
                logger.error(f"No overview data found for {symbol}: {data.get('Error Message', '')}")
                return {}
                
            logger.info(f"Retrieved company overview for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving company overview for {symbol}: {str(e)}")
            return {}
    
    def search_symbol(self, keywords: str) -> List[Dict[str, str]]:
        """
        Search for symbols matching keywords.
        
        Args:
            keywords: Keywords to search for
            
        Returns:
            List of dictionaries with symbol information
        """
        try:
            # Build parameters
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": keywords,
                "apikey": self.api_key
            }
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return []
                
            # Extract search results
            if "bestMatches" in data:
                results = data["bestMatches"]
                logger.info(f"Found {len(results)} matches for '{keywords}'")
                return results
            else:
                logger.warning(f"No matches found for '{keywords}'")
                return []
                
        except Exception as e:
            logger.error(f"Error searching for '{keywords}': {str(e)}")
            return []
            
    def get_earnings_calendar(self, symbol: Optional[str] = None, horizon: str = "3month") -> pd.DataFrame:
        """
        Get earnings calendar for a specific symbol or all symbols.
        
        Args:
            symbol: Optional symbol to filter by
            horizon: Time horizon for earnings (3month, 6month, 12month)
            
        Returns:
            DataFrame with earnings calendar data
        """
        try:
            # Build parameters
            params = {
                "function": "EARNINGS_CALENDAR",
                "horizon": horizon,
                "apikey": self.api_key
            }
            
            if symbol:
                params["symbol"] = symbol
                
            # Make API request with CSV format
            response = requests.get(f"{self.BASE_URL}", params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return pd.DataFrame()
                
            # Parse CSV data
            try:
                df = pd.read_csv(pd.StringIO(response.text))
                
                if df.empty:
                    logger.warning(f"No earnings data found for horizon {horizon}")
                else:
                    logger.info(f"Retrieved {len(df)} earnings entries")
                    
                    # Convert report date to datetime
                    if "reportDate" in df.columns:
                        df["reportDate"] = pd.to_datetime(df["reportDate"])
                        
                return df
                
            except Exception as e:
                logger.error(f"Error parsing earnings calendar data: {str(e)}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving earnings calendar: {str(e)}")
            return pd.DataFrame()
            
    def get_economic_indicator(
        self, 
        indicator: str,
        interval: str = "quarterly",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get economic indicator data.
        
        Args:
            indicator: The economic indicator to retrieve (e.g., GDP, REAL_GDP, CPI)
            interval: Data interval (daily, weekly, monthly, quarterly, annual)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with economic indicator data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365 * 5)  # Default to 5 years
                
            # Build parameters
            params = {
                "function": indicator,
                "interval": interval,
                "apikey": self.api_key
            }
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return pd.DataFrame()
                
            # Extract data
            if "data" in data:
                rows = []
                for item in data["data"]:
                    date = datetime.strptime(item["date"], "%Y-%m-%d")
                    
                    # Skip if outside of date range
                    if date < start_date or date > end_date:
                        continue
                        
                    value = float(item["value"])
                    rows.append({"date": date, "value": value})
                    
                # Create DataFrame
                df = pd.DataFrame(rows)
                
                if df.empty:
                    logger.warning(f"No data found for {indicator}")
                else:
                    df.set_index("date", inplace=True)
                    df.sort_index(inplace=True)
                    logger.info(f"Retrieved {len(df)} data points for {indicator}")
                    
                return df
            else:
                logger.error(f"No data found in response: {data.keys()}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving {indicator}: {str(e)}")
            return pd.DataFrame()
            
    def get_fx_rate(
        self, 
        from_currency: str, 
        to_currency: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> List[Dict[str, Any]]:
        """
        Get foreign exchange rate data.
        
        Args:
            from_currency: The source currency code
            to_currency: The target currency code
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: The time frame to retrieve data for
            
        Returns:
            List of dictionaries with forex data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
            # Map timeframe to Alpha Vantage function
            if timeframe == TimeFrame.DAY_1:
                function = "FX_DAILY"
            elif timeframe == TimeFrame.WEEK_1:
                function = "FX_WEEKLY"
            elif timeframe == TimeFrame.MONTH_1:
                function = "FX_MONTHLY"
            else:
                logger.error(f"Unsupported timeframe for FX data: {timeframe}")
                return []
                
            # Build parameters
            params = {
                "function": function,
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return []
                
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break
                    
            if not time_series_key:
                logger.error(f"No time series data found in response: {data.keys()}")
                return []
                
            # Parse time series data
            fx_data = []
            time_series = data[time_series_key]
            
            for date_str, values in time_series.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                
                # Skip if outside of date range
                if date < start_date or date > end_date:
                    continue
                    
                # Extract values
                open_rate = float(values["1. open"])
                high_rate = float(values["2. high"])
                low_rate = float(values["3. low"])
                close_rate = float(values["4. close"])
                
                # Create data point
                fx_point = {
                    "date": date,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "open": open_rate,
                    "high": high_rate,
                    "low": low_rate,
                    "close": close_rate
                }
                
                fx_data.append(fx_point)
                
            # Sort by date
            fx_data.sort(key=lambda x: x["date"])
            
            logger.info(f"Retrieved {len(fx_data)} FX data points for {from_currency}/{to_currency}")
            return fx_data
            
        except Exception as e:
            logger.error(f"Error retrieving FX data for {from_currency}/{to_currency}: {str(e)}")
            return [] 