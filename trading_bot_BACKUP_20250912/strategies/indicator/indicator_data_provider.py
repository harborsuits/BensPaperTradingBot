#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator Data Provider

This module provides a unified interface to obtain indicator data from broker APIs
or calculate them locally using TA-Lib when not available from the broker.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import talib

from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

logger = logging.getLogger(__name__)

class IndicatorDataProvider:
    """
    Provider for technical indicator data that leverages broker APIs 
    when available and falls back to local calculation when needed.
    """
    
    def __init__(self, 
                 broker_manager: MultiBrokerManager = None,
                 prefer_broker_indicators: bool = True,
                 cache_indicators: bool = True,
                 cache_ttl_seconds: int = 300):
        """
        Initialize the indicator data provider.
        
        Args:
            broker_manager: MultiBrokerManager instance for broker access
            prefer_broker_indicators: Whether to prefer broker-provided indicators
            cache_indicators: Whether to cache indicator results
            cache_ttl_seconds: How long to cache indicator data
        """
        self.broker_manager = broker_manager
        self.prefer_broker_indicators = prefer_broker_indicators
        self.cache_indicators = cache_indicators
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Indicator cache
        self.indicator_cache = {}
        self.cache_timestamps = {}
        
        # Track indicator availability by broker
        self.broker_indicator_support = {
            'alpaca': ['SMA', 'EMA', 'MACD', 'RSI', 'BBANDS'],  # Example - verify with actual API
            'tradier': ['SMA', 'EMA', 'MACD', 'RSI'],           # Example - verify with actual API
            'etrade': ['SMA', 'EMA']                           # Example - verify with actual API
        }
        
        logger.info("Initialized IndicatorDataProvider")
    
    def get_indicator(self, 
                     symbol: str, 
                     indicator_type: str, 
                     timeframe: str, 
                     params: Dict[str, Any],
                     ohlcv_data: Optional[pd.DataFrame] = None) -> Union[pd.Series, Tuple[pd.Series, ...], None]:
        """
        Get indicator data for a symbol.
        
        Args:
            symbol: Trading symbol
            indicator_type: Indicator type (e.g., 'SMA', 'RSI')
            timeframe: Timeframe string (e.g., '1h', '1d')
            params: Indicator parameters
            ohlcv_data: Optional OHLCV data (if not provided, will be fetched)
            
        Returns:
            Indicator data as Series or tuple of Series
        """
        # Check cache first if enabled
        cache_key = f"{symbol}_{timeframe}_{indicator_type}_{str(params)}"
        if self.cache_indicators and cache_key in self.indicator_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl_seconds:
                logger.debug(f"Using cached indicator data for {cache_key}")
                return self.indicator_cache[cache_key]
        
        # Try to get from broker if preferred and supported
        indicator_data = None
        if self.prefer_broker_indicators and self.broker_manager:
            indicator_data = self._get_indicator_from_broker(symbol, indicator_type, timeframe, params)
        
        # Fall back to local calculation if needed
        if indicator_data is None:
            # Get OHLCV data if not provided
            if ohlcv_data is None:
                ohlcv_data = self._get_ohlcv_data(symbol, timeframe)
                
            if ohlcv_data is not None and len(ohlcv_data) > 0:
                indicator_data = self._calculate_indicator_locally(indicator_type, ohlcv_data, params)
            else:
                logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
                return None
        
        # Cache the result if enabled
        if self.cache_indicators and indicator_data is not None:
            self.indicator_cache[cache_key] = indicator_data
            self.cache_timestamps[cache_key] = datetime.now()
        
        return indicator_data
    
    def get_multiple_indicators(self, 
                               symbol: str, 
                               indicators: Dict[str, Dict[str, Any]], 
                               timeframe: str) -> Dict[str, Any]:
        """
        Get multiple indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of indicator configurations
            timeframe: Timeframe string (e.g., '1h', '1d')
            
        Returns:
            Dictionary of indicator values
        """
        # First get the OHLCV data to avoid multiple fetches
        ohlcv_data = self._get_ohlcv_data(symbol, timeframe)
        if ohlcv_data is None or len(ohlcv_data) == 0:
            logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
            return {}
        
        result = {}
        
        # Calculate each indicator
        for indicator_name, indicator_config in indicators.items():
            indicator_type = indicator_config.get('type')
            params = {k: v for k, v in indicator_config.items() if k != 'type'}
            
            indicator_data = self.get_indicator(
                symbol=symbol,
                indicator_type=indicator_type,
                timeframe=timeframe,
                params=params,
                ohlcv_data=ohlcv_data
            )
            
            if indicator_data is not None:
                # Handle multi-value indicators
                if isinstance(indicator_data, tuple):
                    if indicator_type == 'MACD':
                        result[f"{indicator_name}_macd"] = indicator_data[0]
                        result[f"{indicator_name}_signal"] = indicator_data[1]
                        result[f"{indicator_name}_hist"] = indicator_data[2]
                    elif indicator_type == 'BBANDS':
                        result[f"{indicator_name}_upper"] = indicator_data[0]
                        result[f"{indicator_name}_middle"] = indicator_data[1]
                        result[f"{indicator_name}_lower"] = indicator_data[2]
                    elif indicator_type in ['STOCH', 'STOCHF']:
                        result[f"{indicator_name}_k"] = indicator_data[0]
                        result[f"{indicator_name}_d"] = indicator_data[1]
                    else:
                        # Generic handling
                        for i, val in enumerate(indicator_data):
                            result[f"{indicator_name}_{i}"] = val
                else:
                    result[indicator_name] = indicator_data
        
        # Add original price data
        result['open'] = ohlcv_data['open']
        result['high'] = ohlcv_data['high']
        result['low'] = ohlcv_data['low']
        result['close'] = ohlcv_data['close']
        result['volume'] = ohlcv_data.get('volume', pd.Series([0] * len(ohlcv_data)))
        
        return result
    
    def _get_indicator_from_broker(self, 
                                 symbol: str, 
                                 indicator_type: str, 
                                 timeframe: str, 
                                 params: Dict[str, Any]) -> Optional[Union[pd.Series, Tuple[pd.Series, ...]]]:
        """
        Get indicator data directly from broker API if supported.
        
        Args:
            symbol: Trading symbol
            indicator_type: Indicator type
            timeframe: Timeframe string
            params: Indicator parameters
            
        Returns:
            Indicator data if available from broker, None otherwise
        """
        if not self.broker_manager:
            return None
        
        try:
            # Determine which broker to use
            broker_id = self._get_preferred_broker_for_indicator(indicator_type)
            if not broker_id:
                logger.debug(f"No broker supports indicator {indicator_type}")
                return None
            
            broker = self.broker_manager.get_broker(broker_id)
            if not broker or not broker.is_connected():
                logger.warning(f"Broker {broker_id} not available or not connected")
                return None
            
            # Check if the broker supports technical indicators
            if not hasattr(broker, 'get_technical_indicator'):
                logger.debug(f"Broker {broker_id} does not support technical indicators")
                return None
            
            # Convert our timeframe format to broker's format if needed
            broker_timeframe = self._convert_timeframe_for_broker(timeframe, broker_id)
            
            # Call broker's indicator method
            indicator_data = broker.get_technical_indicator(
                symbol=symbol,
                indicator=indicator_type,
                timeframe=broker_timeframe,
                params=params
            )
            
            if indicator_data is not None:
                logger.debug(f"Got indicator {indicator_type} for {symbol} from {broker_id}")
                return indicator_data
            
        except Exception as e:
            logger.error(f"Error getting indicator from broker: {str(e)}")
        
        return None
    
    def _get_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol from the broker or other data source.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.broker_manager:
            logger.warning("No broker manager available for OHLCV data")
            return None
        
        try:
            # Try to get from broker
            broker_id = self.broker_manager.get_preferred_broker_for_symbol(symbol)
            if not broker_id:
                logger.warning(f"No broker available for symbol {symbol}")
                return None
            
            broker = self.broker_manager.get_broker(broker_id)
            if not broker or not broker.is_connected():
                logger.warning(f"Broker {broker_id} not available or not connected")
                return None
            
            # Convert timeframe format if needed
            broker_timeframe = self._convert_timeframe_for_broker(timeframe, broker_id)
            
            # Determine how much history we need
            lookback_bars = 200  # Default to 200 bars
            
            # Get historical data
            historical_data = broker.get_historical_data(
                symbol=symbol,
                timeframe=broker_timeframe,
                limit=lookback_bars
            )
            
            if historical_data is not None and len(historical_data) > 0:
                # Convert to DataFrame if needed
                if not isinstance(historical_data, pd.DataFrame):
                    historical_data = pd.DataFrame(historical_data)
                
                # Ensure columns are properly named
                expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in historical_data.columns for col in expected_columns):
                    logger.warning(f"Historical data missing expected columns: {historical_data.columns}")
                    return None
                
                # Set timestamp as index if it's not already
                if 'timestamp' in historical_data.columns and historical_data.index.name != 'timestamp':
                    historical_data = historical_data.set_index('timestamp')
                
                return historical_data
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {str(e)}")
        
        return None
    
    def _calculate_indicator_locally(self, 
                                   indicator_type: str, 
                                   ohlcv_data: pd.DataFrame, 
                                   params: Dict[str, Any]) -> Optional[Union[pd.Series, Tuple[pd.Series, ...]]]:
        """
        Calculate indicator locally using TA-Lib.
        
        Args:
            indicator_type: Indicator type
            ohlcv_data: OHLCV DataFrame
            params: Indicator parameters
            
        Returns:
            Calculated indicator data
        """
        try:
            # Get indicator function from talib
            if not hasattr(talib, indicator_type):
                logger.warning(f"Indicator {indicator_type} not supported by TA-Lib")
                return None
            
            indicator_func = getattr(talib, indicator_type)
            
            # Extract price data
            open_prices = ohlcv_data['open'].values
            high_prices = ohlcv_data['high'].values
            low_prices = ohlcv_data['low'].values
            close_prices = ohlcv_data['close'].values
            volume = ohlcv_data.get('volume', pd.Series([0] * len(close_prices))).values
            
            # Calculate based on indicator type
            if indicator_type in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'RSI', 'ROC', 'MOM']:
                # Single input indicators
                result = indicator_func(close_prices, **params)
            elif indicator_type == 'BBANDS':
                # Bollinger Bands
                upper, middle, lower = indicator_func(
                    close_prices,
                    timeperiod=params.get('timeperiod', 20),
                    nbdevup=params.get('nbdevup', 2),
                    nbdevdn=params.get('nbdevdn', 2),
                    matype=params.get('matype', 0)
                )
                # Convert to pandas Series
                result = (
                    pd.Series(upper, index=ohlcv_data.index),
                    pd.Series(middle, index=ohlcv_data.index),
                    pd.Series(lower, index=ohlcv_data.index)
                )
            elif indicator_type == 'MACD':
                # MACD
                macd, signal, hist = indicator_func(
                    close_prices,
                    fastperiod=params.get('fastperiod', 12),
                    slowperiod=params.get('slowperiod', 26),
                    signalperiod=params.get('signalperiod', 9)
                )
                # Convert to pandas Series
                result = (
                    pd.Series(macd, index=ohlcv_data.index),
                    pd.Series(signal, index=ohlcv_data.index),
                    pd.Series(hist, index=ohlcv_data.index)
                )
            elif indicator_type in ['STOCH', 'STOCHF']:
                # Stochastic oscillator
                if indicator_type == 'STOCH':
                    k, d = indicator_func(
                        high_prices, low_prices, close_prices,
                        fastk_period=params.get('fastk_period', 5),
                        slowk_period=params.get('slowk_period', 3),
                        slowk_matype=params.get('slowk_matype', 0),
                        slowd_period=params.get('slowd_period', 3),
                        slowd_matype=params.get('slowd_matype', 0)
                    )
                else:  # STOCHF
                    k, d = indicator_func(
                        high_prices, low_prices, close_prices,
                        fastk_period=params.get('fastk_period', 5),
                        fastd_period=params.get('fastd_period', 3),
                        fastd_matype=params.get('fastd_matype', 0)
                    )
                # Convert to pandas Series
                result = (
                    pd.Series(k, index=ohlcv_data.index),
                    pd.Series(d, index=ohlcv_data.index)
                )
            elif indicator_type in ['ATR', 'NATR', 'ADX', 'ADXR', 'CCI']:
                # Indicators that need high, low, close
                result = indicator_func(high_prices, low_prices, close_prices, **params)
            elif indicator_type in ['OBV', 'AD', 'ADOSC']:
                # Indicators that need volume
                if indicator_type == 'OBV':
                    result = indicator_func(close_prices, volume)
                else:
                    result = indicator_func(high_prices, low_prices, close_prices, volume, **params)
            else:
                # Default to using close prices
                result = indicator_func(close_prices, **params)
            
            # Convert to pandas Series if it's not already a tuple
            if not isinstance(result, tuple):
                result = pd.Series(result, index=ohlcv_data.index)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicator locally: {str(e)}")
            return None
    
    def _get_preferred_broker_for_indicator(self, indicator_type: str) -> Optional[str]:
        """
        Get the preferred broker for a specific indicator type.
        
        Args:
            indicator_type: Indicator type
            
        Returns:
            Broker ID or None if no broker supports the indicator
        """
        if not self.broker_manager:
            return None
        
        # Check which brokers support this indicator
        supporting_brokers = []
        for broker_id, indicators in self.broker_indicator_support.items():
            if indicator_type in indicators:
                supporting_brokers.append(broker_id)
        
        if not supporting_brokers:
            return None
        
        # Return the first available broker that supports this indicator
        for broker_id in supporting_brokers:
            broker = self.broker_manager.get_broker(broker_id)
            if broker and broker.is_connected():
                return broker_id
        
        return None
    
    def _convert_timeframe_for_broker(self, timeframe: str, broker_id: str) -> str:
        """
        Convert timeframe format for specific broker.
        
        Args:
            timeframe: Standard timeframe string (e.g., '1h', '1d')
            broker_id: Broker ID
            
        Returns:
            Broker-specific timeframe format
        """
        # Timeframe conversion maps for each broker
        conversion_maps = {
            'alpaca': {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '30m': '30Min',
                '1h': '1Hour',
                '2h': '2Hour',
                '4h': '4Hour',
                '1d': '1Day',
                '1w': '1Week'
            },
            'tradier': {
                '1m': 'minute',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': 'hourly',
                '1d': 'daily',
                '1w': 'weekly'
            },
            'etrade': {
                '1m': 'm1',
                '5m': 'm5',
                '15m': 'm15',
                '30m': 'm30',
                '1h': 'h1',
                '1d': 'd1',
                '1w': 'w1'
            }
        }
        
        # Get conversion map for the specified broker
        broker_map = conversion_maps.get(broker_id, {})
        
        # Return converted timeframe or original if not found
        return broker_map.get(timeframe, timeframe)
