#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data sources for market data acquisition.
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

from trading_bot.data.models import MarketData, DataSource, TimeFrame

logger = logging.getLogger(__name__)

class DataSourceInterface(ABC):
    """Base interface for all data sources."""
    
    @abstractmethod
    def fetch_data(self, 
                  symbol: str, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Fetch market data for a given symbol and date range.
        
        Args:
            symbol: Asset symbol to fetch data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            
        Returns:
            List of MarketData objects
        """
        pass
    
    @abstractmethod
    def fetch_latest(self, 
                    symbol: str,
                    lookback_periods: int = 1,
                    timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Fetch the latest market data for a given symbol.
        
        Args:
            symbol: Asset symbol to fetch data for
            lookback_periods: Number of periods to look back
            timeframe: Timeframe for the data
            
        Returns:
            List of MarketData objects with the latest data
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from this data source.
        
        Returns:
            List of available symbols
        """
        pass
    
    @abstractmethod
    def supports_timeframe(self, timeframe: Union[str, TimeFrame]) -> bool:
        """
        Check if the data source supports a specific timeframe.
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            True if the timeframe is supported, False otherwise
        """
        pass

class YahooFinanceDataSource(DataSourceInterface):
    """Yahoo Finance data source implementation."""
    
    SOURCE_ID = DataSource.YAHOO
    
    # Map of timeframes to Yahoo Finance interval strings
    TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: "1m",
        TimeFrame.MINUTE_5: "5m",
        TimeFrame.MINUTE_15: "15m",
        TimeFrame.MINUTE_30: "30m", 
        TimeFrame.HOUR_1: "1h",
        TimeFrame.DAY_1: "1d",
        TimeFrame.WEEK_1: "1wk",
        TimeFrame.MONTH_1: "1mo"
    }
    
    # Maximum periods per request for each timeframe
    MAX_PERIODS = {
        TimeFrame.MINUTE_1: 7,      # 7 days
        TimeFrame.MINUTE_5: 60,     # 60 days
        TimeFrame.MINUTE_15: 60,    # 60 days
        TimeFrame.MINUTE_30: 60,    # 60 days
        TimeFrame.HOUR_1: 730,      # 730 days (2 years)
        TimeFrame.DAY_1: 10000,     # Max
        TimeFrame.WEEK_1: 10000,    # Max
        TimeFrame.MONTH_1: 10000,   # Max
    }
    
    def __init__(self):
        """Initialize Yahoo Finance data source."""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.error("yfinance package not found. Please install it with 'pip install yfinance'")
            raise ImportError("yfinance package is required for YahooFinanceDataSource")
    
    def fetch_data(self, 
                  symbol: str, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Fetch market data from Yahoo Finance.
        
        Args:
            symbol: Asset symbol to fetch data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            
        Returns:
            List of MarketData objects
        """
        # Convert string timeframe to enum if needed
        if isinstance(timeframe, str):
            try:
                timeframe = TimeFrame(timeframe)
            except ValueError:
                # Try to find matching timeframe
                for tf in TimeFrame:
                    if tf.value == timeframe:
                        timeframe = tf
                        break
                else:
                    logger.warning(f"Invalid timeframe: {timeframe}, using default 1d")
                    timeframe = TimeFrame.DAY_1
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Default to reasonable lookback based on timeframe
            if timeframe == TimeFrame.MINUTE_1:
                start_date = end_date - timedelta(days=7)
            elif timeframe in [TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.MINUTE_30]:
                start_date = end_date - timedelta(days=60)
            elif timeframe == TimeFrame.HOUR_1:
                start_date = end_date - timedelta(days=730)
            else:
                start_date = end_date - timedelta(days=365)
        
        # Ensure start date is not too far in the past for intraday data
        if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.MINUTE_30]:
            max_days = self.MAX_PERIODS.get(timeframe, 7)
            if (end_date - start_date).days > max_days:
                logger.warning(f"Adjusting start date for {timeframe.value} data (max {max_days} days)")
                start_date = end_date - timedelta(days=max_days)
        
        # Get interval string for Yahoo Finance
        interval = self.TIMEFRAME_MAP.get(timeframe, "1d")
        
        try:
            # Fetch data from Yahoo Finance
            df = self.yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                show_errors=False
            )
            
            # Return empty list if no data
            if df.empty:
                logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
                return []
            
            # Convert to list of MarketData objects
            result = []
            for timestamp, row in df.iterrows():
                # Convert timestamp to datetime if needed
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row['Open']) if not pd.isna(row['Open']) else None,
                    high=float(row['High']) if not pd.isna(row['High']) else None,
                    low=float(row['Low']) if not pd.isna(row['Low']) else None,
                    close=float(row['Close']) if not pd.isna(row['Close']) else None,
                    volume=float(row['Volume']) if not pd.isna(row['Volume']) else None,
                    source=self.SOURCE_ID,
                    metadata={
                        "adjusted_close": float(row['Adj Close']) if not pd.isna(row['Adj Close']) else None,
                        "interval": interval
                    }
                )
                result.append(market_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return []
    
    def fetch_latest(self, 
                   symbol: str,
                   lookback_periods: int = 1,
                   timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Fetch the latest market data for a given symbol.
        
        Args:
            symbol: Asset symbol to fetch data for
            lookback_periods: Number of periods to look back
            timeframe: Timeframe for the data
            
        Returns:
            List of MarketData objects with the latest data
        """
        # Convert timeframe if needed
        if isinstance(timeframe, str):
            try:
                timeframe = TimeFrame(timeframe)
            except ValueError:
                timeframe = TimeFrame.DAY_1
        
        # Calculate appropriate start date based on timeframe and lookback
        end_date = datetime.now()
        
        # Add buffer to ensure we get enough periods
        buffer_multiplier = 2
        if timeframe == TimeFrame.MINUTE_1:
            start_date = end_date - timedelta(minutes=lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.MINUTE_5:
            start_date = end_date - timedelta(minutes=5 * lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.MINUTE_15:
            start_date = end_date - timedelta(minutes=15 * lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.MINUTE_30:
            start_date = end_date - timedelta(minutes=30 * lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.HOUR_1:
            start_date = end_date - timedelta(hours=lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.HOUR_4:
            start_date = end_date - timedelta(hours=4 * lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.DAY_1:
            start_date = end_date - timedelta(days=lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.WEEK_1:
            start_date = end_date - timedelta(weeks=lookback_periods * buffer_multiplier)
        elif timeframe == TimeFrame.MONTH_1:
            start_date = end_date - timedelta(days=30 * lookback_periods * buffer_multiplier)
        else:
            start_date = end_date - timedelta(days=lookback_periods * buffer_multiplier)
        
        # Fetch data
        data = self.fetch_data(symbol, start_date, end_date, timeframe)
        
        # Return only the requested number of periods, most recent first
        return sorted(data, key=lambda x: x.timestamp, reverse=True)[:lookback_periods]
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from Yahoo Finance.
        This is not practical to implement fully as Yahoo has thousands of symbols.
        Instead, return some common indices and ETFs.
        
        Returns:
            List of common index and ETF symbols
        """
        # Common indices and ETFs available on Yahoo Finance
        return [
            "^GSPC",    # S&P 500
            "^DJI",     # Dow Jones Industrial Average
            "^IXIC",    # NASDAQ Composite
            "^RUT",     # Russell 2000
            "^VIX",     # CBOE Volatility Index
            "SPY",      # SPDR S&P 500 ETF
            "QQQ",      # Invesco QQQ Trust
            "IWM",      # iShares Russell 2000 ETF
            "DIA",      # SPDR Dow Jones Industrial Average ETF
            "GLD",      # SPDR Gold Shares
            "SLV",      # iShares Silver Trust
            "USO",      # United States Oil Fund
            "XLE",      # Energy Select Sector SPDR Fund
            "XLF",      # Financial Select Sector SPDR Fund
            "XLK",      # Technology Select Sector SPDR Fund
            "XLV",      # Health Care Select Sector SPDR Fund
        ]
    
    def supports_timeframe(self, timeframe: Union[str, TimeFrame]) -> bool:
        """
        Check if Yahoo Finance supports a specific timeframe.
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            True if the timeframe is supported, False otherwise
        """
        if isinstance(timeframe, str):
            # Check if string matches any timeframe value
            for tf in self.TIMEFRAME_MAP:
                if tf.value == timeframe or self.TIMEFRAME_MAP[tf] == timeframe:
                    return True
            return False
        else:
            # Check if enum is in supported timeframes
            return timeframe in self.TIMEFRAME_MAP

class MockDataSource(DataSourceInterface):
    """Mock data source for testing."""
    
    SOURCE_ID = DataSource.MOCK
    
    def __init__(self, seed: int = 42, volatility: float = 0.01):
        """
        Initialize mock data source.
        
        Args:
            seed: Random seed for reproducibility
            volatility: Volatility parameter for price generation
        """
        import numpy as np
        self.np = np
        self.np.random.seed(seed)
        self.volatility = volatility
        self.price_data = {}  # Cache for generated price data
        
    def _generate_prices(self, 
                         symbol: str, 
                         start_date: datetime, 
                         end_date: datetime,
                         timeframe: TimeFrame) -> List[MarketData]:
        """
        Generate synthetic price data.
        
        Args:
            symbol: Symbol to generate data for
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            List of MarketData objects with synthetic data
        """
        # Determine time delta based on timeframe
        if timeframe == TimeFrame.MINUTE_1:
            delta = timedelta(minutes=1)
        elif timeframe == TimeFrame.MINUTE_5:
            delta = timedelta(minutes=5)
        elif timeframe == TimeFrame.MINUTE_15:
            delta = timedelta(minutes=15)
        elif timeframe == TimeFrame.MINUTE_30:
            delta = timedelta(minutes=30)
        elif timeframe == TimeFrame.HOUR_1:
            delta = timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_4:
            delta = timedelta(hours=4)
        elif timeframe == TimeFrame.DAY_1:
            delta = timedelta(days=1)
        elif timeframe == TimeFrame.WEEK_1:
            delta = timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTH_1:
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)
        
        # Generate timestamps
        current = start_date
        timestamps = []
        while current <= end_date:
            # Skip weekends for daily or higher timeframes
            if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
                if current.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    current += delta
                    continue
            timestamps.append(current)
            current += delta
        
        # Generate price data
        result = []
        
        # Use symbol as a basis for initial price
        # Hash the symbol to get a consistent seed
        symbol_hash = sum(ord(c) for c in symbol)
        self.np.random.seed(symbol_hash)
        
        # Base price between 10 and 1000
        base_price = 10 + (symbol_hash % 990)
        price = base_price
        
        for i, timestamp in enumerate(timestamps):
            # Random price movement
            change_pct = self.np.random.normal(0, self.volatility)
            price *= (1 + change_pct)
            
            # Generate OHLC
            daily_volatility = self.volatility * price
            open_price = price * (1 + self.np.random.normal(0, daily_volatility / price))
            high_price = max(open_price, price) * (1 + abs(self.np.random.normal(0, daily_volatility / price)))
            low_price = min(open_price, price) * (1 - abs(self.np.random.normal(0, daily_volatility / price)))
            
            # Ensure high >= open, close and low <= open, close
            high_price = max(high_price, open_price, price)
            low_price = min(low_price, open_price, price)
            
            # Generate volume
            volume = self.np.random.gamma(shape=2.0, scale=100000) * (1 + abs(change_pct) * 10)
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=float(open_price),
                high=float(high_price),
                low=float(low_price),
                close=float(price),
                volume=float(volume),
                source=self.SOURCE_ID,
                metadata={"is_synthetic": True}
            )
            result.append(market_data)
        
        return result
        
    def fetch_data(self, 
                  symbol: str, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Fetch mock market data.
        
        Args:
            symbol: Asset symbol to fetch data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            
        Returns:
            List of MarketData objects with mock data
        """
        # Convert string timeframe to enum if needed
        if isinstance(timeframe, str):
            try:
                timeframe = TimeFrame(timeframe)
            except ValueError:
                timeframe = TimeFrame.DAY_1
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            if timeframe == TimeFrame.MINUTE_1:
                start_date = end_date - timedelta(days=1)
            elif timeframe in [TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.MINUTE_30]:
                start_date = end_date - timedelta(days=7)
            elif timeframe == TimeFrame.HOUR_1:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=365)
        
        # Generate cache key
        cache_key = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}_{timeframe.value}"
        
        # Check cache
        if cache_key in self.price_data:
            return self.price_data[cache_key]
        
        # Generate data
        data = self._generate_prices(symbol, start_date, end_date, timeframe)
        
        # Cache data
        self.price_data[cache_key] = data
        
        return data
    
    def fetch_latest(self, 
                    symbol: str,
                    lookback_periods: int = 1,
                    timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1) -> List[MarketData]:
        """
        Fetch the latest mock market data.
        
        Args:
            symbol: Asset symbol to fetch data for
            lookback_periods: Number of periods to look back
            timeframe: Timeframe for the data
            
        Returns:
            List of MarketData objects with the latest mock data
        """
        # Convert timeframe if needed
        if isinstance(timeframe, str):
            try:
                timeframe = TimeFrame(timeframe)
            except ValueError:
                timeframe = TimeFrame.DAY_1
        
        # Calculate start date based on timeframe and lookback
        end_date = datetime.now()
        
        if timeframe == TimeFrame.MINUTE_1:
            start_date = end_date - timedelta(minutes=lookback_periods)
        elif timeframe == TimeFrame.MINUTE_5:
            start_date = end_date - timedelta(minutes=5 * lookback_periods)
        elif timeframe == TimeFrame.MINUTE_15:
            start_date = end_date - timedelta(minutes=15 * lookback_periods)
        elif timeframe == TimeFrame.MINUTE_30:
            start_date = end_date - timedelta(minutes=30 * lookback_periods)
        elif timeframe == TimeFrame.HOUR_1:
            start_date = end_date - timedelta(hours=lookback_periods)
        elif timeframe == TimeFrame.HOUR_4:
            start_date = end_date - timedelta(hours=4 * lookback_periods)
        elif timeframe == TimeFrame.DAY_1:
            start_date = end_date - timedelta(days=lookback_periods)
        elif timeframe == TimeFrame.WEEK_1:
            start_date = end_date - timedelta(weeks=lookback_periods)
        elif timeframe == TimeFrame.MONTH_1:
            start_date = end_date - timedelta(days=30 * lookback_periods)
        else:
            start_date = end_date - timedelta(days=lookback_periods)
        
        # Fetch data
        return self.fetch_data(symbol, start_date, end_date, timeframe)
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available mock symbols.
        
        Returns:
            List of mock symbols
        """
        return [
            "MOCK1", "MOCK2", "MOCK3", "MOCK4", "MOCK5",
            "MOCKTECH", "MOCKFIN", "MOCKENERGY", "MOCKHEALTHCARE", "MOCKCONSUMER"
        ]
    
    def supports_timeframe(self, timeframe: Union[str, TimeFrame]) -> bool:
        """
        Check if the mock data source supports a specific timeframe.
        Mock data source supports all timeframes.
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            True always
        """
        return True 