#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class definition for data sources.
"""

import abc
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from trading_bot.data.models import MarketData, TimeFrame, DataSource, SymbolMetadata


class BaseDataSource(abc.ABC):
    """
    Abstract base class for all data sources.
    
    This class defines the interface that all data sources must implement.
    Data sources are responsible for retrieving market data from various providers
    and converting it to a standardized format.
    """
    
    def __init__(self, name: str, source_type: DataSource, api_key: Optional[str] = None):
        """
        Initialize the data source.
        
        Args:
            name: Unique identifier for this data source instance
            source_type: Type of data source
            api_key: API key for accessing the data source (if required)
        """
        self.name = name
        self.source_type = source_type
        self.api_key = api_key
    
    @abc.abstractmethod
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                timeframe: TimeFrame) -> List[MarketData]:
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
        pass
    
    @abc.abstractmethod
    def get_latest(self, symbol: str, timeframe: TimeFrame = TimeFrame.DAY_1) -> Optional[MarketData]:
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol: Trading symbol to retrieve data for
            timeframe: Time frame for the data
            
        Returns:
            Latest MarketData object or None if not available
        """
        pass
    
    @abc.abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols from this data source.
        
        Returns:
            List of symbol strings
        """
        pass
    
    @abc.abstractmethod
    def get_symbol_metadata(self, symbol: str) -> Optional[SymbolMetadata]:
        """
        Get metadata for a specific symbol.
        
        Args:
            symbol: Symbol to retrieve metadata for
            
        Returns:
            SymbolMetadata object or None if not available
        """
        pass
    
    def to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """
        Convert a list of MarketData objects to a pandas DataFrame.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            DataFrame with market data
        """
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            record = {
                'symbol': item.symbol,
                'timestamp': item.timestamp,
            }
            
            # Add OHLCV data if available
            if item.open is not None:
                record['open'] = item.open
            if item.high is not None:
                record['high'] = item.high
            if item.low is not None:
                record['low'] = item.low
            if item.close is not None:
                record['close'] = item.close
            if item.volume is not None:
                record['volume'] = item.volume
                
            # Add any additional data
            record.update(item.additional_data)
            
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # Set timestamp as index if available
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            
        return df
    
    def from_dataframe(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[MarketData]:
        """
        Convert a pandas DataFrame to a list of MarketData objects.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol to use for the market data (if not included in the DataFrame)
            
        Returns:
            List of MarketData objects
        """
        if df.empty:
            return []
        
        # Reset index if it's the timestamp
        if df.index.name == 'timestamp':
            df = df.reset_index()
            
        result = []
        for _, row in df.iterrows():
            # Get symbol from row or use provided symbol
            data_symbol = row.get('symbol', symbol)
            if data_symbol is None:
                raise ValueError("Symbol must be provided either in the DataFrame or as an argument")
                
            # Get timestamp from row
            timestamp = row.get('timestamp')
            if timestamp is None:
                raise ValueError("Timestamp must be provided in the DataFrame")
                
            # Get OHLCV data
            market_data = MarketData(
                symbol=data_symbol,
                timestamp=timestamp,
                open=row.get('open'),
                high=row.get('high'),
                low=row.get('low'),
                close=row.get('close'),
                volume=row.get('volume'),
                source=self.source_type
            )
            
            # Add any additional columns as additional data
            additional_data = {}
            for col in df.columns:
                if col not in ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    additional_data[col] = row.get(col)
            
            if additional_data:
                market_data.additional_data = additional_data
                
            result.append(market_data)
            
        return result


class MockDataSource(BaseDataSource):
    """
    Mock data source for testing and development.
    
    Generates synthetic market data with realistic patterns.
    """
    
    def __init__(self, name: str = "Mock", api_key: Optional[str] = None):
        """
        Initialize the mock data source.
        
        Args:
            name: Unique identifier for this data source instance
            api_key: Not used, but included for interface compatibility
        """
        super().__init__(name, DataSource.MOCK, api_key)
        self.available_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
            "TSLA", "NVDA", "JPM", "BRK.B", "JNJ"
        ]
        self.symbol_metadata = {
            symbol: SymbolMetadata(
                symbol=symbol,
                name=f"{symbol} Inc.",
                exchange="NASDAQ" if symbol not in ["JPM", "BRK.B", "JNJ"] else "NYSE",
                asset_type="stock",
                sector="Technology" if symbol not in ["JPM", "BRK.B", "JNJ"] else "Finance",
                industry="Software" if symbol not in ["JPM", "BRK.B", "JNJ"] else "Banking"
            )
            for symbol in self.available_symbols
        }
        
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                timeframe: TimeFrame) -> List[MarketData]:
        """
        Generate mock market data for a symbol within the specified date range.
        
        Args:
            symbol: Trading symbol to retrieve data for
            start_date: Start date for the data range
            end_date: End date for the data range
            timeframe: Time frame for the data
            
        Returns:
            List of MarketData objects
        """
        if symbol not in self.available_symbols:
            return []
        
        # Generate dates based on timeframe
        dates = []
        current_date = start_date
        
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
        else:
            delta = timedelta(days=1)  # Default to daily
            
        while current_date <= end_date:
            # Skip weekends for daily/weekly data
            if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1] and current_date.weekday() >= 5:
                current_date += delta
                continue
                
            dates.append(current_date)
            current_date += delta
            
        # Generate synthetic price data
        base_price = 100 + hash(symbol) % 900  # Different starting price based on symbol
        price_data = []
        
        # Generate a random walk with drift and volatility
        volatility = 0.01 + (hash(symbol) % 5) / 100  # Different volatility per symbol
        drift = 0.0001 + (hash(symbol) % 10) / 10000  # Different drift per symbol
        
        price = base_price
        for i in range(len(dates)):
            # Generate daily movement
            daily_return = np.random.normal(drift, volatility)
            price *= (1 + daily_return)
            
            # Generate OHLC
            daily_volatility = price * volatility
            open_price = price
            close_price = price * (1 + np.random.normal(0, volatility/2))
            high_price = max(open_price, close_price) + abs(np.random.normal(0, daily_volatility))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, daily_volatility))
            
            # Generate volume (higher for more volatile days)
            volume = int(random.normalvariate(1000000, 500000) * (1 + abs(daily_return) * 10))
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                timestamp=dates[i],
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                source=DataSource.MOCK
            )
            
            price_data.append(market_data)
            
        return price_data
    
    def get_latest(self, symbol: str, timeframe: TimeFrame = TimeFrame.DAY_1) -> Optional[MarketData]:
        """
        Get the latest mock market data for a symbol.
        
        Args:
            symbol: Trading symbol to retrieve data for
            timeframe: Time frame for the data
            
        Returns:
            Latest MarketData object or None if not available
        """
        if symbol not in self.available_symbols:
            return None
            
        # Get yesterday's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        data = self.get_data(symbol, start_date, end_date, timeframe)
        return data[-1] if data else None
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols from this data source.
        
        Returns:
            List of symbol strings
        """
        return self.available_symbols
    
    def get_symbol_metadata(self, symbol: str) -> Optional[SymbolMetadata]:
        """
        Get metadata for a specific symbol.
        
        Args:
            symbol: Symbol to retrieve metadata for
            
        Returns:
            SymbolMetadata object or None if not available
        """
        return self.symbol_metadata.get(symbol) 