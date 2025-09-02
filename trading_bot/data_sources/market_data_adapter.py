"""
Market Data Adapter Module

This module provides adapters for connecting to real market data sources.
It abstracts the data retrieval process for different sources and normalizes the data
for use in the BenBot trading system.
"""

import logging
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum

# Import base models
from pydantic import BaseModel, Field

logger = logging.getLogger("market_data_adapter")

class DataSourceType(str, Enum):
    """Types of data sources supported by the adapter"""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    TWELVE_DATA = "twelve_data"
    INTERNAL_DB = "internal_db"

class TimeFrame(str, Enum):
    """Common timeframes for market data"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class MarketDataConfig(BaseModel):
    """Configuration for market data sources"""
    source_type: DataSourceType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_symbols: List[str] = []
    rate_limit: Optional[int] = None  # requests per minute
    timeframes: List[TimeFrame] = [TimeFrame.DAY_1]
    additional_params: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True

class OHLCV(BaseModel):
    """Open, High, Low, Close, Volume data point"""
    timestamp: Union[int, str]
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime"""
        if isinstance(self.timestamp, int):
            return datetime.fromtimestamp(self.timestamp)
        else:
            return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

class MarketIndicator(BaseModel):
    """Technical or fundamental market indicator"""
    name: str
    value: float
    timestamp: Union[int, str]
    symbol: Optional[str] = None
    timeframe: Optional[TimeFrame] = None
    
    class Config:
        use_enum_values = True

class MarketNewsItem(BaseModel):
    """News item with market impact information"""
    id: str
    title: str
    summary: str
    url: str
    source: str
    image_url: Optional[str] = None
    published_at: str
    related_symbols: List[str] = []
    sentiment_score: Optional[float] = None
    
    class Config:
        allow_population_by_field_name = True

class MarketDataAdapter:
    """Base adapter class for market data sources"""
    
    def __init__(self, config: MarketDataConfig):
        self.config = config
        self.last_request_time: Dict[str, datetime] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"market_data.{config.source_type}")
        
    async def initialize(self):
        """Initialize the adapter for use"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self.logger.info(f"Initialized {self.config.source_type} adapter")
        
    async def close(self):
        """Close any open connections"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info(f"Closed {self.config.source_type} adapter")
    
    async def get_price_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame = TimeFrame.DAY_1,
        limit: int = 100,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> List[OHLCV]:
        """
        Retrieve price data for the specified symbol and timeframe
        
        Args:
            symbol: The market symbol to retrieve data for
            timeframe: The time resolution of the data
            limit: Maximum number of data points to retrieve
            start_time: Optional start time for the data range
            end_time: Optional end time for the data range
            
        Returns:
            List of OHLCV data points
        """
        raise NotImplementedError("Subclasses must implement get_price_data")
    
    async def get_indicators(
        self,
        symbol: str,
        indicators: List[str],
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> Dict[str, List[MarketIndicator]]:
        """
        Retrieve technical indicators for the specified symbol
        
        Args:
            symbol: The market symbol to retrieve indicators for
            indicators: List of indicator names to retrieve
            timeframe: The time resolution for the indicators
            
        Returns:
            Dictionary mapping indicator names to lists of indicator values
        """
        raise NotImplementedError("Subclasses must implement get_indicators")
    
    async def get_latest_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 20,
        categories: Optional[List[str]] = None
    ) -> List[MarketNewsItem]:
        """
        Retrieve latest market news relevant to specified symbols
        
        Args:
            symbols: Optional list of market symbols to filter news for
            limit: Maximum number of news items to retrieve
            categories: Optional list of news categories to filter by
            
        Returns:
            List of market news items
        """
        raise NotImplementedError("Subclasses must implement get_latest_news")
    
    async def _make_request(
        self, 
        endpoint: str, 
        params: Dict[str, Any] = {},
        headers: Dict[str, str] = {},
        request_id: str = "default"
    ) -> Any:
        """
        Make a rate-limited request to the data source API
        
        Args:
            endpoint: API endpoint to request
            params: Query parameters for the request
            headers: HTTP headers for the request
            request_id: Identifier for rate limiting purposes
            
        Returns:
            Parsed JSON response
        """
        if not self.session:
            await self.initialize()
            
        # Apply rate limiting if configured
        if self.config.rate_limit:
            now = datetime.now()
            if request_id in self.last_request_time:
                # Calculate time to wait based on rate limit
                seconds_per_request = 60 / self.config.rate_limit
                time_since_last = (now - self.last_request_time[request_id]).total_seconds()
                if time_since_last < seconds_per_request:
                    wait_time = seconds_per_request - time_since_last
                    self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            self.last_request_time[request_id] = datetime.now()
        
        # Add API key to params if present in config
        if self.config.api_key:
            # Different services use different parameter names for API keys
            if self.config.source_type == DataSourceType.ALPHA_VANTAGE:
                params["apikey"] = self.config.api_key
            elif self.config.source_type == DataSourceType.FINNHUB:
                headers["X-Finnhub-Token"] = self.config.api_key
            else:
                params["api_key"] = self.config.api_key
                
        # Construct the full URL
        base_url = self.config.base_url or ""
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    self.logger.error(f"API error: {response.status} - {await response.text()}")
                    return None
                
                result = await response.json()
                return result
        except aiohttp.ClientError as e:
            self.logger.error(f"Request error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return None


# Create adapter factory based on configuration
def create_adapter(config: MarketDataConfig) -> MarketDataAdapter:
    """
    Create the appropriate market data adapter based on configuration
    
    Args:
        config: Adapter configuration
        
    Returns:
        Instance of the appropriate MarketDataAdapter subclass
    """
    if config.source_type == DataSourceType.ALPHA_VANTAGE:
        from trading_bot.data_sources.alpha_vantage_adapter import AlphaVantageAdapter
        return AlphaVantageAdapter(config)
    elif config.source_type == DataSourceType.YAHOO_FINANCE:
        from trading_bot.data_sources.yahoo_finance_adapter import YahooFinanceAdapter
        return YahooFinanceAdapter(config)
    elif config.source_type == DataSourceType.POLYGON:
        from trading_bot.data_sources.polygon_adapter import PolygonAdapter
        return PolygonAdapter(config)
    elif config.source_type == DataSourceType.FINNHUB:
        from trading_bot.data_sources.finnhub_adapter import FinnhubAdapter
        return FinnhubAdapter(config)
    elif config.source_type == DataSourceType.TWELVE_DATA:
        from trading_bot.data_sources.twelve_data_adapter import TwelveDataAdapter
        return TwelveDataAdapter(config)
    elif config.source_type == DataSourceType.INTERNAL_DB:
        from trading_bot.data_sources.internal_db_adapter import InternalDBAdapter
        return InternalDBAdapter(config)
    else:
        raise ValueError(f"Unsupported data source type: {config.source_type}")
