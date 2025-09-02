#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Data Provider

This module provides interfaces and implementations for accessing market data
from various sources like Alpaca, Yahoo Finance, etc.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    def get_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get current price for a list of symbols"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str, 
                           interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data for a list of symbols"""
        pass
    
    @abstractmethod
    def get_latest_bars(self, symbols: List[str], limit: int = 1) -> Dict[str, pd.DataFrame]:
        """Get the latest N bars for a list of symbols"""
        pass
    
    @abstractmethod
    def subscribe_to_real_time_updates(self, symbols: List[str], callback) -> None:
        """Subscribe to real-time updates for symbols"""
        pass
    
    @abstractmethod
    def unsubscribe_from_real_time_updates(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time updates for symbols"""
        pass

class AlpacaDataProvider(MarketDataProvider):
    """Alpaca implementation of MarketDataProvider"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = 'https://paper-api.alpaca.markets'):
        """Initialize the Alpaca data provider with API credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.api = None
        self.ws = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Alpaca client"""
        try:
            # Import Alpaca SDK - we use a try-except to handle missing dependencies
            from alpaca_trade_api import REST, Stream
            
            # Initialize REST client
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url
            )
            
            # Initialize WebSocket client
            self.ws = Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                data_feed='iex'  # Use 'sip' for paid subscription
            )
            
            logger.info("Successfully initialized Alpaca client")
        except ImportError:
            logger.error("Failed to import alpaca_trade_api. Make sure it's installed with: pip install alpaca-trade-api")
            raise
        except Exception as e:
            logger.error(f"Error initializing Alpaca client: {e}")
            raise
    
    def get_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get current price for a list of symbols"""
        try:
            # Get the latest quotes
            quotes = {}
            for symbol in symbols:
                try:
                    quote = self.api.get_latest_quote(symbol)
                    # Use midpoint price (average of bid and ask)
                    if hasattr(quote, 'bidprice') and hasattr(quote, 'askprice'):
                        quotes[symbol] = (quote.bidprice + quote.askprice) / 2
                    else:
                        last_trade = self.api.get_latest_trade(symbol)
                        quotes[symbol] = last_trade.price
                except Exception as e:
                    logger.warning(f"Could not get current price for {symbol}: {e}")
            return quotes
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str, 
                           interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data for a list of symbols"""
        try:
            # Map interval string to timeframe string expected by Alpaca
            timeframe_map = {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '30m': '30Min',
                '1h': '1Hour',
                '1d': '1Day'
            }
            timeframe = timeframe_map.get(interval, '1Day')
            
            # Get historical bars for each symbol
            data = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start_date,
                        end=end_date,
                        adjustment='all'  # Apply all adjustments
                    ).df
                    
                    if not bars.empty:
                        # Rename columns to be consistent with our expected format
                        bars = bars.rename(columns={
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume',
                            'trade_count': 'trades',
                            'vwap': 'vwap'
                        })
                        data[symbol] = bars
                except Exception as e:
                    logger.warning(f"Could not get historical data for {symbol}: {e}")
            return data
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {}
    
    def get_latest_bars(self, symbols: List[str], limit: int = 1) -> Dict[str, pd.DataFrame]:
        """Get the latest N bars for a list of symbols"""
        try:
            end = datetime.now()
            start = end - timedelta(days=limit * 2)  # Request more days to ensure we get enough bars
            
            data = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(
                        symbol,
                        '1Day',
                        start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        limit=limit
                    ).df
                    
                    if not bars.empty:
                        data[symbol] = bars.tail(limit)
                except Exception as e:
                    logger.warning(f"Could not get latest bars for {symbol}: {e}")
            return data
        except Exception as e:
            logger.error(f"Error getting latest bars: {e}")
            return {}
    
    def subscribe_to_real_time_updates(self, symbols: List[str], callback) -> None:
        """Subscribe to real-time updates for symbols"""
        try:
            if self.ws:
                # Register the callback for trade updates
                @self.ws.on_trade_updates
                async def handle_trade_update(trade):
                    callback('trade_update', trade)
                
                # Register the callback for bar updates
                @self.ws.on_bars(symbols)
                async def handle_bar(bar):
                    callback('bar', bar)
                
                # Register the callback for quote updates
                @self.ws.on_quotes(symbols)
                async def handle_quote(quote):
                    callback('quote', quote)
                
                # Start the websocket connection
                self.ws.run()
                
                logger.info(f"Subscribed to real-time updates for: {', '.join(symbols)}")
            else:
                logger.warning("WebSocket client not initialized.")
        except Exception as e:
            logger.error(f"Error subscribing to real-time updates: {e}")
    
    def unsubscribe_from_real_time_updates(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time updates for symbols"""
        try:
            if self.ws:
                # Close the websocket connection
                self.ws.stop()
                logger.info(f"Unsubscribed from real-time updates for: {', '.join(symbols)}")
            else:
                logger.warning("WebSocket client not initialized.")
        except Exception as e:
            logger.error(f"Error unsubscribing from real-time updates: {e}")

class YahooFinanceDataProvider(MarketDataProvider):
    """Yahoo Finance implementation of MarketDataProvider"""
    
    def __init__(self):
        """Initialize the Yahoo Finance data provider"""
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Yahoo Finance client"""
        try:
            # Import yfinance - we use a try-except to handle missing dependencies
            import yfinance as yf
            self.yf = yf
            logger.info("Successfully initialized Yahoo Finance client")
        except ImportError:
            logger.error("Failed to import yfinance. Make sure it's installed with: pip install yfinance")
            raise
        except Exception as e:
            logger.error(f"Error initializing Yahoo Finance client: {e}")
            raise
    
    def get_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get current price for a list of symbols"""
        try:
            # Get the latest quotes
            tickers = self.yf.Tickers(" ".join(symbols))
            quotes = {}
            
            for symbol in symbols:
                try:
                    quotes[symbol] = tickers.tickers[symbol].info.get('regularMarketPrice', None)
                except Exception as e:
                    logger.warning(f"Could not get current price for {symbol}: {e}")
            
            return quotes
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str, 
                           interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data for a list of symbols"""
        try:
            data = {}
            for symbol in symbols:
                try:
                    ticker = self.yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval
                    )
                    
                    if not hist.empty:
                        # Rename columns to be consistent with our expected format
                        hist = hist.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        data[symbol] = hist
                except Exception as e:
                    logger.warning(f"Could not get historical data for {symbol}: {e}")
            
            return data
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {}
    
    def get_latest_bars(self, symbols: List[str], limit: int = 1) -> Dict[str, pd.DataFrame]:
        """Get the latest N bars for a list of symbols"""
        try:
            end = datetime.now()
            start = end - timedelta(days=limit * 2)  # Request more days to ensure we get enough bars
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = self.yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if not hist.empty:
                        # Rename columns to be consistent with our expected format
                        hist = hist.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        data[symbol] = hist.tail(limit)
                except Exception as e:
                    logger.warning(f"Could not get latest bars for {symbol}: {e}")
            
            return data
        except Exception as e:
            logger.error(f"Error getting latest bars: {e}")
            return {}
    
    def subscribe_to_real_time_updates(self, symbols: List[str], callback) -> None:
        """Subscribe to real-time updates for symbols"""
        logger.warning("Real-time updates not supported by Yahoo Finance provider")
    
    def unsubscribe_from_real_time_updates(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time updates for symbols"""
        logger.warning("Real-time updates not supported by Yahoo Finance provider")

def create_data_provider(provider_name: str, config_path: Optional[str] = None) -> MarketDataProvider:
    """
    Factory function to create the appropriate data provider
    
    Args:
        provider_name: Name of the provider ('alpaca', 'yahoo')
        config_path: Path to the config file with API credentials
        
    Returns:
        Initialized data provider instance
    """
    # Load configuration if a path is provided
    config = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Create the appropriate provider
    if provider_name.lower() == 'alpaca':
        api_key = config.get('alpaca', {}).get('api_key', os.environ.get('ALPACA_API_KEY'))
        api_secret = config.get('alpaca', {}).get('api_secret', os.environ.get('ALPACA_API_SECRET'))
        base_url = config.get('alpaca', {}).get('base_url', 'https://paper-api.alpaca.markets')
        
        if not api_key or not api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        return AlpacaDataProvider(api_key, api_secret, base_url)
    
    elif provider_name.lower() == 'yahoo':
        return YahooFinanceDataProvider()
    
    else:
        raise ValueError(f"Unsupported provider: {provider_name}") 