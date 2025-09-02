#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yahoo Finance data source implementation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import yfinance as yf

from trading_bot.data.models import MarketData, TimeFrame, DataSource, SymbolMetadata
from trading_bot.data.sources.base import BaseDataSource


logger = logging.getLogger(__name__)


class YahooFinanceDataSource(BaseDataSource):
    """
    Data source implementation for Yahoo Finance.
    Uses the yfinance library to access Yahoo Finance data.
    """
    
    def __init__(self, name: str = "yahoo_finance", api_key: Optional[str] = None):
        """
        Initialize Yahoo Finance data source.
        
        Args:
            name: Unique identifier for this data source instance
            api_key: Not used for Yahoo Finance but kept for consistency
        """
        super().__init__(name, DataSource.YAHOO, api_key)
        self._symbol_cache = {}  # Cache for Yahoo Ticker objects
    
    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get or create a Yahoo Finance Ticker object for the given symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Yahoo Finance Ticker object
        """
        if symbol not in self._symbol_cache:
            self._symbol_cache[symbol] = yf.Ticker(symbol)
        return self._symbol_cache[symbol]
    
    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert our TimeFrame enum to Yahoo Finance interval string.
        
        Args:
            timeframe: TimeFrame enum value
            
        Returns:
            Yahoo Finance interval string
        """
        mapping = {
            TimeFrame.MIN_1: "1m",
            TimeFrame.MIN_2: "2m",
            TimeFrame.MIN_5: "5m",
            TimeFrame.MIN_15: "15m",
            TimeFrame.MIN_30: "30m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.HOUR_4: "4h",
            TimeFrame.DAY_1: "1d",
            TimeFrame.WEEK_1: "1wk",
            TimeFrame.MONTH_1: "1mo",
        }
        
        if timeframe not in mapping:
            logger.warning(f"Unsupported timeframe {timeframe}, defaulting to 1d")
            return "1d"
            
        return mapping[timeframe]
    
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
        
        try:
            # Note: For some intervals (e.g., 1m), Yahoo might restrict the date range
            # or return an empty result for requests that are too far in the past
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date + timedelta(days=1),  # Add a day to include the end_date
                interval=interval,
                progress=False
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
                return []
                
            # Convert to standard format
            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }, inplace=True)
            
            df['symbol'] = symbol
            
            # Convert to our data model
            market_data_list = self.from_dataframe(df)
            
            return market_data_list
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            return []
    
    def get_latest(self, symbol: str, timeframe: TimeFrame = TimeFrame.DAY_1) -> Optional[MarketData]:
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol: Trading symbol to retrieve data for
            timeframe: Time frame for the data
            
        Returns:
            Latest MarketData object or None if not available
        """
        # For latest data, we'll retrieve the last 7 days and take the most recent
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        data = self.get_data(symbol, start_date, end_date, timeframe)
        if not data:
            return None
            
        # Sort by timestamp and return the most recent
        data.sort(key=lambda x: x.timestamp, reverse=True)
        return data[0]
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols from Yahoo Finance.
        
        Note: Yahoo Finance doesn't provide a direct API for this, so we return an empty list.
        External sources or a predefined list would be needed for a real implementation.
        
        Returns:
            Empty list (feature not supported by Yahoo Finance API)
        """
        logger.warning("Yahoo Finance does not provide a direct API for listing all available symbols")
        return []
    
    def get_symbol_metadata(self, symbol: str) -> Optional[SymbolMetadata]:
        """
        Get metadata for a specific symbol from Yahoo Finance.
        
        Args:
            symbol: Symbol to retrieve metadata for
            
        Returns:
            SymbolMetadata object or None if not available
        """
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            if not info:
                logger.warning(f"No metadata available for symbol {symbol}")
                return None
                
            # Create metadata object with available information
            return SymbolMetadata(
                symbol=symbol,
                name=info.get('shortName', info.get('longName', symbol)),
                asset_type=info.get('quoteType', 'unknown'),
                exchange=info.get('exchange', 'unknown'),
                currency=info.get('currency', 'USD'),
                sector=info.get('sector', None),
                industry=info.get('industry', None),
                country=info.get('country', None),
                description=info.get('longBusinessSummary', None)
            )
                
        except Exception as e:
            logger.error(f"Error retrieving metadata for {symbol}: {str(e)}")
            return None 