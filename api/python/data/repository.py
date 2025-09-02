#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data repository for managing market data sources and providing a unified API.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from trading_bot.data.models import MarketData, DataSource, TimeFrame, DatasetMetadata
from trading_bot.data.sources import DataSourceInterface, YahooFinanceDataSource, MockDataSource

logger = logging.getLogger(__name__)

class MarketDataRepository:
    """
    Repository for managing market data access through multiple data sources.
    Provides a unified API for data acquisition and handling.
    """
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize the market data repository.
        
        Args:
            use_mock: Whether to use mock data instead of real data sources
        """
        self.sources: Dict[DataSource, DataSourceInterface] = {}
        
        # Initialize data sources
        if use_mock:
            self.sources[DataSource.MOCK] = MockDataSource()
            self.default_source = DataSource.MOCK
        else:
            try:
                self.sources[DataSource.YAHOO] = YahooFinanceDataSource()
                self.default_source = DataSource.YAHOO
            except ImportError:
                logger.warning("Yahoo Finance data source not available, falling back to mock data")
                self.sources[DataSource.MOCK] = MockDataSource()
                self.default_source = DataSource.MOCK
    
    def add_source(self, source_id: DataSource, source: DataSourceInterface) -> None:
        """
        Add a data source to the repository.
        
        Args:
            source_id: ID of the data source
            source: Data source implementation
        """
        self.sources[source_id] = source
    
    def get_source(self, source_id: Optional[DataSource] = None) -> DataSourceInterface:
        """
        Get a data source by ID.
        
        Args:
            source_id: ID of the data source, or None for default
            
        Returns:
            The requested data source
            
        Raises:
            ValueError: If the source ID is not registered
        """
        if source_id is None:
            source_id = self.default_source
            
        if source_id not in self.sources:
            raise ValueError(f"Data source {source_id} not registered")
            
        return self.sources[source_id]
    
    def set_default_source(self, source_id: DataSource) -> None:
        """
        Set the default data source.
        
        Args:
            source_id: ID of the data source to set as default
            
        Raises:
            ValueError: If the source ID is not registered
        """
        if source_id not in self.sources:
            raise ValueError(f"Data source {source_id} not registered")
            
        self.default_source = source_id
    
    def get_data(self, 
                symbol: str, 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1,
                source_id: Optional[DataSource] = None) -> List[MarketData]:
        """
        Fetch market data for a given symbol and date range.
        
        Args:
            symbol: Asset symbol to fetch data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            source_id: Specific data source to use, or None for default
            
        Returns:
            List of MarketData objects
        """
        source = self.get_source(source_id)
        return source.fetch_data(symbol, start_date, end_date, timeframe)
    
    def get_latest(self, 
                  symbol: str,
                  lookback_periods: int = 1,
                  timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1,
                  source_id: Optional[DataSource] = None) -> List[MarketData]:
        """
        Fetch the latest market data for a given symbol.
        
        Args:
            symbol: Asset symbol to fetch data for
            lookback_periods: Number of periods to look back
            timeframe: Timeframe for the data
            source_id: Specific data source to use, or None for default
            
        Returns:
            List of MarketData objects with the latest data
        """
        source = self.get_source(source_id)
        return source.fetch_latest(symbol, lookback_periods, timeframe)
    
    def get_available_symbols(self, source_id: Optional[DataSource] = None) -> List[str]:
        """
        Get list of available symbols from a data source.
        
        Args:
            source_id: Specific data source to use, or None for default
            
        Returns:
            List of available symbols
        """
        source = self.get_source(source_id)
        return source.get_available_symbols()
    
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
        
        # Extract data into a dictionary
        data_dict = {
            'symbol': [],
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'source': []
        }
        
        for item in data:
            data_dict['symbol'].append(item.symbol)
            data_dict['timestamp'].append(item.timestamp)
            data_dict['open'].append(item.open)
            data_dict['high'].append(item.high)
            data_dict['low'].append(item.low)
            data_dict['close'].append(item.close)
            data_dict['volume'].append(item.volume)
            data_dict['source'].append(item.source.value if item.source else None)
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def from_dataframe(self, 
                      df: pd.DataFrame, 
                      symbol: Optional[str] = None,
                      source: DataSource = DataSource.UNKNOWN) -> List[MarketData]:
        """
        Convert a pandas DataFrame to a list of MarketData objects.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol to use if not in DataFrame
            source: Data source to use
            
        Returns:
            List of MarketData objects
        """
        if df.empty:
            return []
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column")
        
        result = []
        
        for timestamp, row in df.iterrows():
            # Get symbol from row or use provided symbol
            row_symbol = row.get('symbol', symbol)
            if row_symbol is None:
                raise ValueError("Symbol must be provided or be a column in the DataFrame")
            
            # Create MarketData object
            market_data = MarketData(
                symbol=row_symbol,
                timestamp=timestamp if isinstance(timestamp, datetime) else pd.Timestamp(timestamp).to_pydatetime(),
                open=float(row['open']) if 'open' in row and not pd.isna(row['open']) else None,
                high=float(row['high']) if 'high' in row and not pd.isna(row['high']) else None,
                low=float(row['low']) if 'low' in row and not pd.isna(row['low']) else None,
                close=float(row['close']) if 'close' in row and not pd.isna(row['close']) else None,
                volume=float(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else None,
                source=source
            )
            result.append(market_data)
        
        return result
    
    def create_dataset_metadata(self,
                               name: str,
                               symbols: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               timeframe: TimeFrame,
                               source: DataSource = None,
                               features: List[str] = None) -> DatasetMetadata:
        """
        Create dataset metadata for a collection of market data.
        
        Args:
            name: Name of the dataset
            symbols: List of symbols in the dataset
            start_date: Start date of the dataset
            end_date: End date of the dataset
            timeframe: Timeframe of the dataset
            source: Data source used
            features: List of features included
            
        Returns:
            DatasetMetadata object
        """
        if source is None:
            source = self.default_source
            
        if features is None:
            features = ["open", "high", "low", "close", "volume"]
            
        return DatasetMetadata(
            name=name,
            source=source,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            created_at=datetime.now(),
            features=features,
            statistics=None  # Could calculate statistics here if needed
        )
    
    def add_market_data(self, data: List[MarketData]) -> None:
        """
        Add market data directly to the repository.
        
        Args:
            data: List of MarketData objects
        """
        if not data:
            return
            
        # Convert to DataFrame for storage
        df = self.to_dataframe(data)
        
        # Get symbol from first data point
        symbol = data[0].symbol
        
        # Check if we already have data for this symbol
        if hasattr(self, 'data_cache') and self.data_cache.get(symbol) is not None:
            # Append new data to existing data
            existing_df = self.data_cache[symbol]
            combined_df = pd.concat([existing_df, df])
            
            # Remove duplicates based on index (timestamp)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            # Sort by timestamp
            combined_df.sort_index(inplace=True)
            
            # Update cache
            self.data_cache[symbol] = combined_df
        else:
            # Initialize data cache if not exists
            if not hasattr(self, 'data_cache'):
                self.data_cache = {}
                
            # Store new data
            self.data_cache[symbol] = df
            
        logger.info(f"Added {len(data)} market data points for {symbol}")
    
    def update_indicators(self, symbol: str, indicators: Dict[str, Any]) -> None:
        """
        Update indicators for a symbol.
        
        Args:
            symbol: Symbol to update indicators for
            indicators: Dictionary of indicator values
        """
        if not hasattr(self, 'indicator_cache'):
            self.indicator_cache = {}
            
        # Store the latest indicators for this symbol
        self.indicator_cache[symbol] = {
            'timestamp': datetime.now(),
            'indicators': indicators
        }
        
        logger.info(f"Updated indicators for {symbol}")
    
    def get_latest_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest indicators for a symbol.
        
        Args:
            symbol: Symbol to get indicators for
            
        Returns:
            Dictionary of indicator values or None if not available
        """
        if not hasattr(self, 'indicator_cache') or symbol not in self.indicator_cache:
            return None
            
        return self.indicator_cache[symbol]['indicators']
    
    def get_data_with_indicators(self, 
                               symbol: str, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1,
                               source_id: Optional[DataSource] = None) -> pd.DataFrame:
        """
        Get market data with indicators.
        
        Args:
            symbol: Symbol to get data for
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            source_id: Data source
            
        Returns:
            DataFrame with market data and indicators
        """
        # Get market data
        data = self.get_data(symbol, start_date, end_date, timeframe, source_id)
        df = self.to_dataframe(data)
        
        # If we have indicators for this symbol, add them to the most recent row
        if hasattr(self, 'indicator_cache') and symbol in self.indicator_cache:
            indicators = self.indicator_cache[symbol]['indicators']
            
            # Create indicator columns if they don't exist
            for indicator, value in indicators.items():
                if indicator not in df.columns:
                    df[indicator] = None
                    
            # Update the most recent row with the latest indicators
            latest_idx = df.index[-1] if not df.empty else None
            if latest_idx is not None:
                for indicator, value in indicators.items():
                    df.at[latest_idx, indicator] = value
        
        return df 