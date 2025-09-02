"""
Real Market Data Integration for Backtesting

This module provides integration between the backtesting system and real market data
from various API providers (Alpha Vantage, Finnhub, Tradier, Alpaca).
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import API config if available
try:
    from trading_bot.config import API_KEYS
except ImportError:
    # Create a dummy config for testing
    API_KEYS = {
        'alpha_vantage': '',
        'finnhub': '',
        'tradier': '',
        'alpaca': {
            'key': '',
            'secret': '',
            'endpoint': ''
        }
    }

logger = logging.getLogger(__name__)

class RealMarketDataProvider:
    """
    Provides real market data from various API providers for backtesting.
    
    This class fetches historical market data from:
    - Alpha Vantage
    - Finnhub
    - Tradier
    - Alpaca
    
    It implements fallback logic to try alternative providers if one fails.
    """
    
    def __init__(self):
        """Initialize the market data provider"""
        self.api_keys = API_KEYS
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../../data/market_data')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check which APIs are available
        self.available_apis = []
        
        if self.api_keys.get('alpha_vantage'):
            self.available_apis.append('alpha_vantage')
        
        if self.api_keys.get('finnhub'):
            self.available_apis.append('finnhub')
        
        if self.api_keys.get('tradier'):
            self.available_apis.append('tradier')
        
        if self.api_keys.get('alpaca', {}).get('key') and self.api_keys.get('alpaca', {}).get('secret'):
            self.available_apis.append('alpaca')
        
        logger.info(f"Initialized RealMarketDataProvider with {len(self.available_apis)} available APIs")
    
    def get_historical_data(self, 
                          symbol: str, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          timeframe: str = 'daily',
                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (defaults to 1 year ago)
            end_date: End date (defaults to today)
            timeframe: Timeframe ('daily', 'weekly', etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data or None if data cannot be fetched
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Check cache first if enabled
        if use_cache:
            cached_data = self._load_from_cache(symbol, start_date, end_date, timeframe)
            if cached_data is not None:
                logger.info(f"Loaded cached data for {symbol} from {start_date.date()} to {end_date.date()}")
                return cached_data
        
        # Try each available API until we get data
        for api in self.available_apis:
            try:
                if api == 'alpha_vantage':
                    data = self._fetch_alpha_vantage(symbol, timeframe)
                elif api == 'finnhub':
                    data = self._fetch_finnhub(symbol, start_date, end_date, timeframe)
                elif api == 'tradier':
                    data = self._fetch_tradier(symbol, start_date, end_date, timeframe)
                elif api == 'alpaca':
                    data = self._fetch_alpaca(symbol, start_date, end_date, timeframe)
                else:
                    continue
                
                if data is not None and not data.empty:
                    # Filter to requested date range
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[(data['date'] >= pd.Timestamp(start_date)) & 
                               (data['date'] <= pd.Timestamp(end_date))]
                    
                    # Cache the data
                    self._save_to_cache(data, symbol, start_date, end_date, timeframe)
                    
                    logger.info(f"Fetched {len(data)} days of {symbol} data from {api}")
                    return data
            except Exception as e:
                logger.warning(f"Error fetching {symbol} data from {api}: {str(e)}")
                continue
        
        logger.error(f"Failed to fetch data for {symbol} from any available API")
        return None
    
    def get_multi_symbol_data(self,
                            symbols: List[str],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            timeframe: str = 'daily',
                            use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ('daily', 'weekly', etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            data = self.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                use_cache=use_cache
            )
            
            if data is not None and not data.empty:
                results[symbol] = data
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        return results
    
    def _fetch_alpha_vantage(self, 
                          symbol: str, 
                          timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API"""
        if not self.api_keys.get('alpha_vantage'):
            return None
        
        function = 'TIME_SERIES_DAILY' if timeframe == 'daily' else 'TIME_SERIES_WEEKLY'
        outputsize = 'full'  # Get maximum data
        
        url = (
            f"https://www.alphavantage.co/query?"
            f"function={function}&symbol={symbol}&outputsize={outputsize}"
            f"&apikey={self.api_keys['alpha_vantage']}"
        )
        
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Alpha Vantage API returned status code {response.status_code}")
        
        data = response.json()
        
        # Extract time series
        time_series_key = f"Time Series ({timeframe.capitalize()})"
        if time_series_key not in data:
            # Try alternative key format
            time_series_key = next((k for k in data.keys() if k.startswith('Time Series')), None)
            if not time_series_key:
                raise Exception(f"No time series data found in Alpha Vantage response: {list(data.keys())}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        records = []
        for date, values in time_series.items():
            record = {
                'date': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(float(values['5. volume']))
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        df = df.sort_values('date')
        
        return df
    
    def _fetch_finnhub(self, 
                     symbol: str, 
                     start_date: datetime,
                     end_date: datetime,
                     timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub API"""
        if not self.api_keys.get('finnhub'):
            return None
        
        # Convert to Unix timestamp
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())
        
        # Resolution: D=day, W=week
        resolution = 'D' if timeframe == 'daily' else 'W'
        
        url = (
            f"https://finnhub.io/api/v1/stock/candle?"
            f"symbol={symbol}&resolution={resolution}"
            f"&from={from_timestamp}&to={to_timestamp}"
            f"&token={self.api_keys['finnhub']}"
        )
        
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Finnhub API returned status code {response.status_code}")
        
        data = response.json()
        
        # Check for valid response
        if data.get('s') != 'ok':
            raise Exception(f"Finnhub API returned error: {data.get('s')}")
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        
        df['symbol'] = symbol
        df = df.sort_values('date')
        
        return df
    
    def _fetch_tradier(self, 
                     symbol: str, 
                     start_date: datetime,
                     end_date: datetime,
                     timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        """Fetch data from Tradier API"""
        if not self.api_keys.get('tradier'):
            return None
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Interval: daily, weekly
        interval = timeframe
        
        url = f"https://api.tradier.com/v1/markets/history"
        headers = {
            'Authorization': f"Bearer {self.api_keys['tradier']}",
            'Accept': 'application/json'
        }
        params = {
            'symbol': symbol,
            'interval': interval,
            'start': start_str,
            'end': end_str
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Tradier API returned status code {response.status_code}")
        
        data = response.json()
        
        # Check for valid response
        if 'history' not in data or 'day' not in data['history']:
            raise Exception(f"Tradier API returned invalid data: {data}")
        
        # Handle case where only one day is returned
        days = data['history']['day']
        if not isinstance(days, list):
            days = [days]
        
        # Convert to DataFrame
        records = []
        for day in days:
            record = {
                'date': day['date'],
                'open': float(day['open']),
                'high': float(day['high']),
                'low': float(day['low']),
                'close': float(day['close']),
                'volume': int(day['volume'])
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        df = df.sort_values('date')
        
        return df
    
    def _fetch_alpaca(self, 
                    symbol: str, 
                    start_date: datetime,
                    end_date: datetime,
                    timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca API"""
        alpaca_config = self.api_keys.get('alpaca', {})
        if not alpaca_config.get('key') or not alpaca_config.get('secret'):
            return None
        
        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Timeframe: 1D, 1W
        timeframe_str = '1D' if timeframe == 'daily' else '1W'
        
        # Determine endpoint
        endpoint = alpaca_config.get('endpoint', 'https://data.alpaca.markets')
        url = f"{endpoint}/v2/stocks/{symbol}/bars"
        
        headers = {
            'APCA-API-KEY-ID': alpaca_config['key'],
            'APCA-API-SECRET-KEY': alpaca_config['secret']
        }
        
        params = {
            'start': start_str,
            'end': end_str,
            'timeframe': timeframe_str,
            'limit': 10000  # Max limit
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Alpaca API returned status code {response.status_code}")
        
        data = response.json()
        
        # Check for valid response
        if 'bars' not in data:
            raise Exception(f"Alpaca API returned invalid data: {data}")
        
        # Convert to DataFrame
        records = []
        for bar in data['bars']:
            record = {
                'date': bar['t'],
                'open': float(bar['o']),
                'high': float(bar['h']),
                'low': float(bar['l']),
                'close': float(bar['c']),
                'volume': int(bar['v'])
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        df = df.sort_values('date')
        
        return df
    
    def _get_cache_filename(self, 
                          symbol: str, 
                          start_date: datetime,
                          end_date: datetime,
                          timeframe: str) -> str:
        """Generate a cache filename for the given parameters"""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return os.path.join(self.cache_dir, f"{symbol}_{timeframe}_{start_str}_{end_str}.csv")
    
    def _save_to_cache(self,
                     data: pd.DataFrame,
                     symbol: str,
                     start_date: datetime,
                     end_date: datetime,
                     timeframe: str) -> None:
        """Save data to cache"""
        filename = self._get_cache_filename(symbol, start_date, end_date, timeframe)
        data.to_csv(filename, index=False)
    
    def _load_from_cache(self,
                       symbol: str,
                       start_date: datetime,
                       end_date: datetime,
                       timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        filename = self._get_cache_filename(symbol, start_date, end_date, timeframe)
        
        if os.path.exists(filename):
            try:
                data = pd.read_csv(filename)
                data['date'] = pd.to_datetime(data['date'])
                return data
            except Exception as e:
                logger.warning(f"Error loading cached data: {str(e)}")
                return None
        
        return None

class StrategyIntegration:
    """
    Integrates the backtest system with the existing strategy framework.
    
    This class provides utilities to convert between the strategy representations
    used in the trading system and the backtest system.
    """
    
    @staticmethod
    def convert_strategy_to_backtest_format(strategy_class) -> Dict[str, Any]:
        """
        Convert a strategy class to the format expected by the backtest system.
        
        Args:
            strategy_class: Strategy class from the trading system
            
        Returns:
            Dict with strategy metadata in backtest format
        """
        # Extract metadata from strategy class
        strategy_id = getattr(strategy_class, 'strategy_id', strategy_class.__name__)
        strategy_name = getattr(strategy_class, 'name', strategy_class.__name__)
        description = getattr(strategy_class, 'description', '')
        
        # Determine category from class hierarchy or direct attribute
        category = None
        if hasattr(strategy_class, 'category'):
            category = strategy_class.category
        else:
            # Try to infer from class name
            class_name = strategy_class.__name__
            if 'Trend' in class_name:
                category = 'trend_following'
            elif 'Reversion' in class_name:
                category = 'mean_reversion'
            elif 'Breakout' in class_name:
                category = 'breakout'
            elif 'Volatility' in class_name:
                category = 'volatility'
            else:
                category = 'general'
        
        # Get symbols if available
        symbols = getattr(strategy_class, 'symbols', ['SPY'])
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Get timeframes if available
        timeframes = getattr(strategy_class, 'timeframes', ['1d'])
        if isinstance(timeframes, str):
            timeframes = [timeframes]
        
        # Get default parameters if available
        parameters = {}
        try:
            if hasattr(strategy_class, 'default_parameters'):
                parameters = strategy_class.default_parameters
            elif hasattr(strategy_class, 'get_default_parameters'):
                parameters = strategy_class.get_default_parameters()
        except:
            # Use empty dict if can't get parameters
            pass
        
        # Construct strategy metadata
        strategy_metadata = {
            'name': strategy_name,
            'description': description,
            'category': category,
            'symbols': symbols,
            'timeframes': timeframes,
            'parameters': parameters
        }
        
        return strategy_metadata
    
    @staticmethod
    def load_strategies_from_module(module_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load strategies from a module path.
        
        Args:
            module_path: Dotted path to module (e.g., 'trading_bot.strategies.stocks')
            
        Returns:
            Dict mapping strategy IDs to strategy metadata in backtest format
        """
        strategies = {}
        
        try:
            module = __import__(module_path, fromlist=['*'])
            
            # Find all strategy classes (assuming they follow a naming convention)
            for attr_name in dir(module):
                if attr_name.endswith('Strategy') and attr_name != 'Strategy':
                    try:
                        strategy_class = getattr(module, attr_name)
                        
                        # Check if it's a class
                        if isinstance(strategy_class, type):
                            # Get strategy ID
                            strategy_id = getattr(strategy_class, 'strategy_id', attr_name)
                            
                            # Convert to backtest format
                            strategy_metadata = StrategyIntegration.convert_strategy_to_backtest_format(strategy_class)
                            
                            # Add to result
                            strategies[strategy_id] = strategy_metadata
                    except Exception as e:
                        logger.warning(f"Error loading strategy {attr_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading strategies from module {module_path}: {str(e)}")
        
        return strategies
    
    @staticmethod
    def load_all_strategies() -> Dict[str, Dict[str, Any]]:
        """
        Load all available strategies from the trading system.
        
        Returns:
            Dict mapping strategy IDs to strategy metadata in backtest format
        """
        all_strategies = {}
        
        # Try to load from different strategy modules
        modules_to_try = [
            'trading_bot.strategies.stocks',
            'trading_bot.strategies.options',
            'trading_bot.strategies.crypto',
            'trading_bot.strategies.forex',
            'trading_bot.strategies_new.stocks',
            'trading_bot.strategies_new.options',
            'trading_bot.strategies_new.crypto',
            'trading_bot.strategies_new.forex'
        ]
        
        for module_path in modules_to_try:
            try:
                strategies = StrategyIntegration.load_strategies_from_module(module_path)
                all_strategies.update(strategies)
            except Exception as e:
                logger.warning(f"Error loading strategies from {module_path}: {str(e)}")
        
        # If we couldn't load any strategies, use the test strategies
        if not all_strategies:
            logger.warning("Could not load any real strategies, using test strategies")
            all_strategies = {
                'trend_following': {
                    'name': 'Trend Following Strategy',
                    'description': 'Follows market trends using moving averages',
                    'category': 'trend_following',
                    'symbols': ['SPY', 'QQQ', 'IWM', 'EEM'],
                    'timeframes': ['1d'],
                    'parameters': {
                        'ma_fast': 20,
                        'ma_slow': 50,
                        'trailing_stop_pct': 2.0,
                        'profit_target_pct': 3.0
                    }
                },
                'mean_reversion': {
                    'name': 'Mean Reversion Strategy',
                    'description': 'Trades mean reversion using Bollinger Bands',
                    'category': 'mean_reversion',
                    'symbols': ['SPY', 'IWM', 'EEM', 'GLD'],
                    'timeframes': ['1d'],
                    'parameters': {
                        'entry_threshold': 2.0,
                        'profit_target_pct': 1.5,
                        'stop_loss_pct': 1.0
                    }
                },
                'breakout_strategy': {
                    'name': 'Breakout Strategy',
                    'description': 'Trades range breakouts',
                    'category': 'breakout',
                    'symbols': ['QQQ', 'IWM', 'EFA', 'USO'],
                    'timeframes': ['1d'],
                    'parameters': {
                        'breakout_threshold': 2.0,
                        'confirmation_period': 3,
                        'trailing_stop_pct': 2.0
                    }
                },
                'volatility_strategy': {
                    'name': 'Volatility Strategy',
                    'description': 'Trades volatility expansion and contraction',
                    'category': 'volatility',
                    'symbols': ['VXX', 'SPY', 'TLT', 'GLD'],
                    'timeframes': ['1d'],
                    'parameters': {
                        'vix_threshold': 20,
                        'position_size_scale': 0.8,
                        'profit_target_mult': 1.3
                    }
                }
            }
        
        return all_strategies


# Simple test function
def test_real_data_provider():
    """Test the real market data provider"""
    provider = RealMarketDataProvider()
    
    # Test fetching data for SPY
    data = provider.get_historical_data(
        symbol='SPY',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    if data is not None:
        print(f"Successfully fetched {len(data)} days of SPY data")
        print(data.head())
    else:
        print("Failed to fetch SPY data")
    
    return data

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_real_data_provider()
