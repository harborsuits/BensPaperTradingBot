#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data models for market data structures and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union


class TimeFrame(str, Enum):
    """Time frame for market data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"
    
    @classmethod
    def from_string(cls, value: str) -> "TimeFrame":
        """
        Convert a string to a TimeFrame enum value.
        
        Args:
            value: String representation of timeframe
            
        Returns:
            Corresponding TimeFrame enum value
            
        Raises:
            ValueError: If no matching timeframe is found
        """
        for tf in cls:
            if tf.value == value:
                return tf
        raise ValueError(f"Invalid timeframe: {value}")


class DataSource(str, Enum):
    """Data source identifiers."""
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    BINANCE = "binance"
    TRADINGVIEW = "tradingview"
    MOCK = "mock"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "DataSource":
        """
        Convert a string to a DataSource enum value.
        
        Args:
            value: String representation of data source
            
        Returns:
            Corresponding DataSource enum value
            
        Raises:
            ValueError: If no matching data source is found
        """
        for ds in cls:
            if ds.value == value:
                return ds
        raise ValueError(f"Invalid data source: {value}")


@dataclass
class MarketData:
    """
    Represents a single market data point in time.
    Includes OHLCV data and associated metadata.
    """
    symbol: str
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    source: Optional[DataSource] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the market data."""
        if self.close is None and self.open is None:
            raise ValueError("At least one of open or close must be provided")


@dataclass
class DatasetMetadata:
    """
    Metadata for a market data dataset.
    Includes information about data source, time range, symbols, etc.
    """
    name: str
    source: DataSource
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    timeframe: TimeFrame
    created_at: datetime = field(default_factory=datetime.now)
    features: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    statistics: Optional[Dict[str, Any]] = None
    
    @property
    def duration_days(self) -> float:
        """
        Calculate the duration of the dataset in days.
        
        Returns:
            Duration in days
        """
        return (self.end_date - self.start_date).total_seconds() / (60 * 60 * 24)


@dataclass
class SymbolMetadata:
    """
    Metadata for a trading symbol.
    Includes information about the asset type, exchange, etc.
    """
    symbol: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    is_tradable: bool = True
    additional_info: Dict[str, Any] = field(default_factory=dict) 