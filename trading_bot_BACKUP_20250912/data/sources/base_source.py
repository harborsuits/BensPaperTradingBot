#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base DataSource - Abstract class for all data sources.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.common.market_types import MarketData

logger = logging.getLogger("DataSource")

class DataSourceInterface(ABC):
    """
    Interface defining the contract for data sources.
    """
    
    @abstractmethod
    def fetch_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1d",
        **kwargs
    ) -> List[MarketData]:
        """
        Fetch market data for a symbol within a date range.
        
        Args:
            symbol: Asset symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Data interval/timeframe
            **kwargs: Additional arguments specific to the data source
            
        Returns:
            List of MarketData objects
        """
        pass
    
    @abstractmethod
    def fetch_latest(
        self, 
        symbol: str,
        lookback_periods: int = 1,
        timeframe: str = "1d",
        **kwargs
    ) -> List[MarketData]:
        """
        Fetch latest market data for a symbol.
        
        Args:
            symbol: Asset symbol
            lookback_periods: Number of periods to look back
            timeframe: Data interval/timeframe
            **kwargs: Additional arguments specific to the data source
            
        Returns:
            List of MarketData objects
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

class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    A data source is responsible for fetching data from a specific provider,
    such as market data APIs, fundamental data providers, or alternative data sources.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data source.
        
        Args:
            name: Name of the data source
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.is_connected = False
        self.last_update_time = None
        
        # Set up any additional attributes from config
        self.init_from_config()
        
        logger.info(f"Initialized {self.name} data source")
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Default implementation does nothing
        # Override in subclasses to set up specific configuration
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
        **kwargs
    ) -> List[MarketData]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Asset symbol
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to current time)
            interval: Data interval (e.g., "1m", "1h", "1d")
            **kwargs: Additional arguments specific to the data source
            
        Returns:
            List of MarketData objects
        """
        pass
    
    @abstractmethod
    def get_current_data(self, symbol: str, **kwargs) -> MarketData:
        """
        Get current (latest) data for a symbol.
        
        Args:
            symbol: Asset symbol
            **kwargs: Additional arguments specific to the data source
            
        Returns:
            MarketData object
        """
        pass
    
    def get_data_for_timeframe(
        self, 
        symbol: str, 
        timeframe: str,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> List[MarketData]:
        """
        Helper method to get data for a specific timeframe from now.
        
        Args:
            symbol: Asset symbol
            timeframe: Timeframe string (e.g., "1d", "1w", "1m", "1y")
            end_date: End date (defaults to current time)
            **kwargs: Additional arguments for get_historical_data
            
        Returns:
            List of MarketData objects
        """
        # Parse timeframe
        if end_date is None:
            end_date = datetime.now()
            
        amount = int(timeframe[:-1])
        unit = timeframe[-1]
        
        # Calculate start date based on timeframe
        if unit == 'd':
            start_date = end_date - timedelta(days=amount)
        elif unit == 'w':
            start_date = end_date - timedelta(weeks=amount)
        elif unit == 'm':
            # Approximate months as 30 days
            start_date = end_date - timedelta(days=amount * 30)
        elif unit == 'y':
            # Approximate years as 365 days
            start_date = end_date - timedelta(days=amount * 365)
        else:
            raise ValueError(f"Invalid timeframe unit: {unit}. Use 'd', 'w', 'm', or 'y'.")
        
        # Get historical data
        return self.get_historical_data(
            symbol, 
            start_date, 
            end_date,
            **kwargs
        )
        
    def __str__(self) -> str:
        """String representation of the data source."""
        return f"{self.name} DataSource (connected: {self.is_connected})" 