import os
import json
import time
import pandas as pd
import requests
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
import logging

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlphaVantageFetcher')

class AlphaVantageFetcher:
    """
    A class to fetch financial data from Alpha Vantage API
    """
    
    def __init__(self, api_key: str = None, use_cache: bool = True, cache_dir: str = 'data/cache'):
        """
        Initialize the AlphaVantageFetcher with API key and settings
        
        Parameters:
        -----------
        api_key : str, optional
            Alpha Vantage API key. If None, will try to load from environment variable 
            ALPHA_VANTAGE_API_KEY or from config.json
        use_cache : bool
            Whether to use cached data
        cache_dir : str
            Directory to store cached data
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # If API key is still None, try loading from config.json
        if self.api_key is None:
            try:
                with open('config.json', 'r') as f:
                    config = json.load(f)
                self.api_key = config.get("alpha_vantage_api_key")
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass
        
        if not self.api_key:
            logger.warning("No Alpha Vantage API key provided. Set the ALPHA_VANTAGE_API_KEY environment variable.")
            self.api_key = 'demo'  # Use demo key for testing
        
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API call counter to avoid hitting rate limits
        self.last_call_timestamp = 0
        self.calls_per_minute = 5  # Free tier limit
    
    def _manage_rate_limit(self):
        """Ensure we don't exceed Alpha Vantage API rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_timestamp
        
        # If less than a minute has passed since our last API call
        if time_since_last_call < 60 / self.calls_per_minute:
            sleep_time = (60 / self.calls_per_minute) - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_timestamp = time.time()
    
    def _get_cache_path(self, function: str, symbol: str, **kwargs) -> Path:
        """Generate a path for caching API responses"""
        # Create a filename based on function, symbol and parameters
        params_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()) 
                             if k not in ['apikey', 'function', 'symbol'])
        filename = f"{function}_{symbol}_{params_str}.csv"
        return self.cache_dir / filename
    
    def _fetch_from_api(self, params: Dict) -> Dict:
        """Make an API request to Alpha Vantage"""
        self._manage_rate_limit()
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            raise Exception(f"Alpha Vantage API error: {data['Error Message']}")
        if "Information" in data and "call frequency" in data["Information"].lower():
            raise Exception(f"API rate limit exceeded: {data['Information']}")
            
        return data
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch daily adjusted time series for a given symbol
        
        Parameters:
        -----------
        symbol : str
            The stock symbol to fetch data for
        outputsize : str
            Output size. 'compact' returns the latest 100 data points, 'full' returns up to 20 years of historical data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the daily adjusted stock data
        """
        function = 'TIME_SERIES_DAILY_ADJUSTED'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, outputsize=outputsize)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': outputsize
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and 'Time Series (Daily)' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or 'Time Series (Daily)' not in data:
            logger.error(f"No data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Rename columns
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend',
            '8. split coefficient': 'split'
        })
        
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_intraday(self, symbol: str, interval: str = '5min', outputsize: str = 'full') -> pd.DataFrame:
        """
        Get intraday price data for symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        interval : str
            Time interval between data points. Options: '1min', '5min', '15min', '30min', '60min'
        outputsize : str
            Output size. 'compact' returns the latest 100 data points, 'full' returns up to 20 years of historical data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with intraday price data
        """
        function = 'TIME_SERIES_INTRADAY'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, interval=interval, outputsize=outputsize)
        cached_data = self._load_from_cache(cache_path, max_age_days=1)  # Expire intraday data after 1 day
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and f'Time Series ({interval})' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or f'Time Series ({interval})' not in data:
            logger.error(f"No intraday data found for {symbol} at {interval} interval")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[f'Time Series ({interval})'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Rename columns
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_sma(self, symbol: str, time_period: int = 20, series_type: str = "close") -> pd.DataFrame:
        """Get Simple Moving Average (SMA) for a symbol"""
        function = 'SMA'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, time_period=time_period, series_type=series_type)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'interval': 'daily',
                'time_period': time_period,
                'series_type': series_type
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and 'Technical Analysis: SMA' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or 'Technical Analysis: SMA' not in data:
            logger.error(f"No SMA data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Technical Analysis: SMA'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_ema(self, symbol: str, time_period: int = 20, series_type: str = "close") -> pd.DataFrame:
        """Get Exponential Moving Average (EMA) for a symbol"""
        function = 'EMA'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, time_period=time_period, series_type=series_type)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'interval': 'daily',
                'time_period': time_period,
                'series_type': series_type
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and 'Technical Analysis: EMA' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or 'Technical Analysis: EMA' not in data:
            logger.error(f"No EMA data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Technical Analysis: EMA'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_rsi(self, symbol: str, time_period: int = 14, series_type: str = "close") -> pd.DataFrame:
        """Get Relative Strength Index (RSI) for a symbol"""
        function = 'RSI'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, time_period=time_period, series_type=series_type)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'interval': 'daily',
                'time_period': time_period,
                'series_type': series_type
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and 'Technical Analysis: RSI' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or 'Technical Analysis: RSI' not in data:
            logger.error(f"No RSI data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Technical Analysis: RSI'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_macd(self, symbol: str, series_type: str = "close", 
                fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """Get Moving Average Convergence/Divergence (MACD) for a symbol"""
        function = 'MACD'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, series_type=series_type, 
                                          fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'interval': 'daily',
                'series_type': series_type,
                'fastperiod': fastperiod,
                'slowperiod': slowperiod,
                'signalperiod': signalperiod
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and 'Technical Analysis: MACD' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or 'Technical Analysis: MACD' not in data:
            logger.error(f"No MACD data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Technical Analysis: MACD'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_bbands(self, symbol: str, time_period: int = 20, series_type: str = "close",
                 nbdevup: int = 2, nbdevdn: int = 2, matype: int = 0) -> pd.DataFrame:
        """Get Bollinger Bands for a symbol"""
        function = 'BBANDS'
        
        # Check cache
        cache_path = self._get_cache_path(function, symbol, time_period=time_period, series_type=series_type,
                                          nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            data = cached_data
        else:
            # Make request
            params = {
                'function': function,
                'symbol': symbol,
                'interval': 'daily',
                'time_period': time_period,
                'series_type': series_type,
                'nbdevup': nbdevup,
                'nbdevdn': nbdevdn,
                'matype': matype
            }
            
            data = self._fetch_from_api(params)
            
            # Save to cache if request was successful
            if data and 'Technical Analysis: BBANDS' in data:
                self._save_to_cache(cache_path, data)
                
        # Parse data
        if not data or 'Technical Analysis: BBANDS' not in data:
            logger.error(f"No Bollinger Bands data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Technical Analysis: BBANDS'], orient='index')
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get comprehensive stock data including price and technical indicators
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with stock data
        """
        # Get daily adjusted price data
        price_df = self.get_daily_adjusted(symbol)
        
        if price_df.empty:
            logger.error(f"No price data found for {symbol}")
            return pd.DataFrame()
            
        # Get SMA data
        sma_short_df = self.get_sma(symbol, time_period=20)
        sma_long_df = self.get_sma(symbol, time_period=50)
        
        # Get MACD data
        macd_df = self.get_macd(symbol)
        
        # Get RSI data
        rsi_df = self.get_rsi(symbol)
        
        # Get Bollinger Bands data
        bbands_df = self.get_bbands(symbol)
        
        # Merge all DataFrames
        df = price_df.copy()
        
        if not sma_short_df.empty:
            df['sma_20'] = sma_short_df['SMA']
            
        if not sma_long_df.empty:
            df['sma_50'] = sma_long_df['SMA']
            
        if not macd_df.empty:
            df['macd'] = macd_df['MACD']
            df['macd_signal'] = macd_df['MACD_Signal']
            df['macd_hist'] = macd_df['MACD_Hist']
            
        if not rsi_df.empty:
            df['rsi'] = rsi_df['RSI']
            
        if not bbands_df.empty:
            df['bb_upper'] = bbands_df['Real Upper Band']
            df['bb_middle'] = bbands_df['Real Middle Band']
            df['bb_lower'] = bbands_df['Real Lower Band']
            
        # Filter by date range if provided
        if start_date and end_date:
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        elif start_date:
            df = df[df.index >= start_date]
        elif end_date:
            df = df[df.index <= end_date]
            
        return df

    def _load_from_cache(self, cache_path: Path, max_age_days: int = 1) -> Optional[Dict]:
        """
        Load data from cache if available and not expired
        
        Parameters:
        -----------
        cache_path : Path
            Path to cached data file
        max_age_days : int
            Maximum age of cached data in days
            
        Returns:
        --------
        Optional[Dict]
            Cached data if available and not expired, None otherwise
        """
        if not self.use_cache or not cache_path.exists():
            return None
            
        # Check if cache is expired
        modification_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - modification_time
        
        if age > timedelta(days=max_age_days):
            logger.info(f"Cache expired for {cache_path}")
            return None
            
        # Load cache
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded data from cache: {cache_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict) -> None:
        """
        Save data to cache
        
        Parameters:
        -----------
        cache_path : Path
            Path to cached data file
        data : Dict
            Data to cache
        """
        if not self.use_cache:
            return
            
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                logger.info(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")


if __name__ == "__main__":
    # Example usage
    fetcher = AlphaVantageFetcher()
    
    # Get daily adjusted data for Apple
    apple_data = fetcher.get_daily_adjusted("AAPL")
    print(f"AAPL data shape: {apple_data.shape}")
    print(apple_data.head())
    
    # Get comprehensive stock data for Apple
    df = fetcher.get_stock_data('AAPL', start_date='2021-01-01', end_date='2021-12-31')
    print(df.head()) 