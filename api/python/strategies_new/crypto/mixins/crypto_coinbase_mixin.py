#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase API Integration Mixin for Crypto Strategies

This mixin provides Coinbase-specific functionality to crypto trading strategies.
It handles market data retrieval, order execution, and account management 
specifically for Coinbase.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CoinbaseAPIMixin:
    """
    Mixin class that adds Coinbase API functionality to crypto strategies.
    
    This mixin is designed to be used with the CryptoBaseStrategy class and its
    derivatives. It provides methods to fetch market data, execute orders, and
    manage accounts specifically on Coinbase.
    
    Attributes:
        broker_name (str): The name of the broker as registered in the system.
    """
    
    def __init__(self, *args, **kwargs):
        # This will be called by the Python multiple inheritance mechanism
        super().__init__(*args, **kwargs)
        
        # Flag to indicate if we're using Coinbase
        self.is_using_coinbase = False
        self.broker_name = kwargs.get('broker_name', None)
        
        # Check if we're using Coinbase
        if self.broker_name == 'coinbase':
            self.is_using_coinbase = True
            logger.info(f"Strategy {self.name} is using Coinbase API integration")
    
    def get_market_data(self, symbol: str = None, timeframe: str = None, 
                      start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """
        Fetch market data from Coinbase when the broker is Coinbase.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            timeframe: Candle interval (e.g., '1h', '1d')
            start: Start time for historical data
            end: End time for historical data
            
        Returns:
            DataFrame with market data
        """
        if not self.is_using_coinbase:
            # Use the parent class implementation if not using Coinbase
            return super().get_market_data(symbol, timeframe, start, end)
        
        # Default to strategy symbol and timeframe if not provided
        symbol = symbol or self.session.symbol
        
        # Convert timeframe to Coinbase format if needed
        coinbase_timeframe = self._convert_timeframe_to_coinbase(timeframe or self.session.timeframe)
        
        # Set default time range if not provided
        if not end:
            end = datetime.now()
        if not start:
            # Default to 100 candles
            if timeframe == '1d':
                start = end - timedelta(days=100)
            elif timeframe == '1h':
                start = end - timedelta(hours=100)
            elif timeframe == '15m':
                start = end - timedelta(minutes=15 * 100)
            else:
                # Default fallback
                start = end - timedelta(days=7)
        
        try:
            # Access broker through the session
            broker = self.session.broker
            
            if hasattr(broker, 'get_bars'):
                # Get historical data from broker
                bars = broker.get_bars(symbol, coinbase_timeframe, start, end)
                
                if isinstance(bars, tuple) and len(bars) == 2:
                    # Handle case where broker returns (success, data) tuple
                    success, bars_data = bars
                    if not success:
                        logger.error(f"Failed to get market data from Coinbase: {bars_data}")
                        return pd.DataFrame()
                    bars = bars_data
                
                # Convert to DataFrame if it's not already
                if not isinstance(bars, pd.DataFrame):
                    # Handle case where bars is a list of dictionaries
                    if isinstance(bars, list) and bars and isinstance(bars[0], dict):
                        df = pd.DataFrame(bars)
                        # Standard column renaming if needed
                        if 'timestamp' in df.columns:
                            df['datetime'] = pd.to_datetime(df['timestamp'])
                            df.set_index('datetime', inplace=True)
                        return df
                    else:
                        logger.error(f"Unexpected format for bars from Coinbase: {type(bars)}")
                        return pd.DataFrame()
                
                return bars
            else:
                logger.warning("Broker does not implement get_bars method")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching market data from Coinbase: {str(e)}")
            return pd.DataFrame()
    
    def _convert_timeframe_to_coinbase(self, timeframe: str) -> str:
        """
        Convert internal timeframe format to Coinbase format.
        
        Args:
            timeframe: Internal timeframe format
            
        Returns:
            Coinbase timeframe format
        """
        # Map our internal timeframes to Coinbase granularity
        # Coinbase uses seconds for granularity
        timeframe_map = {
            'M1': '60',           # 1 minute = 60 seconds
            'M5': '300',          # 5 minutes = 300 seconds
            'M15': '900',         # 15 minutes = 900 seconds
            'M30': '1800',        # 30 minutes = 1800 seconds
            'H1': '3600',         # 1 hour = 3600 seconds
            'H4': '14400',        # 4 hours = 14400 seconds
            'H6': '21600',        # 6 hours = 21600 seconds
            'H12': '43200',       # 12 hours = 43200 seconds
            'D1': '86400',        # 1 day = 86400 seconds
        }
        
        # Alternative format that might be used
        alt_format_map = {
            '1m': '60',
            '5m': '300',
            '15m': '900',
            '30m': '1800',
            '1h': '3600',
            '4h': '14400',
            '6h': '21600',
            '12h': '43200',
            '1d': '86400',
        }
        
        # Try both formats
        coinbase_timeframe = timeframe_map.get(timeframe)
        if not coinbase_timeframe:
            coinbase_timeframe = alt_format_map.get(timeframe)
        
        # Default to 1 hour if timeframe not recognized
        if not coinbase_timeframe:
            logger.warning(f"Unrecognized timeframe {timeframe}, defaulting to 1 hour")
            coinbase_timeframe = '3600'
        
        return coinbase_timeframe
    
    def place_order_on_coinbase(self, symbol: str, side: str, quantity: float, order_type: str = 'market',
                              limit_price: float = None, stop_price: float = None) -> Dict[str, Any]:
        """
        Place an order on Coinbase.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            order_type: Order type ('market', 'limit', etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order information
        """
        if not self.is_using_coinbase:
            # Use the parent class implementation if not using Coinbase
            return super().place_order(symbol, side, quantity, order_type, limit_price, stop_price)
        
        try:
            # Access broker through the session
            broker = self.session.broker
            
            if hasattr(broker, 'place_order'):
                # Place order using broker
                order_info = broker.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price
                )
                
                logger.info(f"Placed {order_type} {side} order for {quantity} {symbol} on Coinbase")
                return order_info
            else:
                logger.warning("Broker does not implement place_order method")
                return {"error": "Broker does not implement place_order method"}
                
        except Exception as e:
            logger.error(f"Error placing order on Coinbase: {str(e)}")
            return {"error": str(e)}
    
    def get_account_balances_from_coinbase(self) -> Dict[str, Dict[str, float]]:
        """
        Get account balances from Coinbase.
        
        Returns:
            Dictionary of balances by currency
        """
        if not self.is_using_coinbase:
            # Use the parent class implementation if not using Coinbase
            return super().get_account_balances()
        
        try:
            # Access broker through the session
            broker = self.session.broker
            
            if hasattr(broker, 'get_account_balances'):
                # Get balances using broker
                balances = broker.get_account_balances()
                
                logger.info(f"Retrieved account balances from Coinbase")
                return balances
            else:
                logger.warning("Broker does not implement get_account_balances method")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting account balances from Coinbase: {str(e)}")
            return {}
