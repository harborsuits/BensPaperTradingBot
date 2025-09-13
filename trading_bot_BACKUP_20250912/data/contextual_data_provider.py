#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Data Provider

This module integrates various data sources and markets them available to the 
contextual integration manager. It handles retrieving market data, volatility 
metrics, and regime information.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import os

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, MarketRegime
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger(__name__)

class ContextualDataProvider:
    """
    Provides data to the contextual integration system with awareness
    of market regimes, volatility, and other market conditions.
    """
    
    def __init__(self, 
                event_bus: EventBus,
                persistence: PersistenceManager,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the contextual data provider.
        
        Args:
            event_bus: Central event bus for communication
            persistence: Persistence manager for state
            config: Configuration options
        """
        self.event_bus = event_bus
        self.persistence = persistence
        self.config = config or {}
        
        # Configure data sources
        self.data_sources = self.config.get('data_sources', ['csv', 'oanda'])
        self.default_source = self.config.get('default_source', 'csv')
        
        # Data cache
        self._market_data = {}
        self._volatility_data = {}
        self._correlation_data = {}
        self._regime_data = {}
        
        # Last update timestamps
        self._last_updates = {
            'market_data': {},
            'volatility': {},
            'correlation': datetime.min,
            'regime': {}
        }
        
        # Data update intervals (in seconds)
        self.update_intervals = {
            'market_data': self.config.get('market_data_interval', 60),
            'volatility': self.config.get('volatility_interval', 300),
            'correlation': self.config.get('correlation_interval', 3600),
            'regime': self.config.get('regime_interval', 300)
        }
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Contextual Data Provider initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events from the event bus."""
        # Subscribe to data request events
        self.event_bus.subscribe(EventType.DATA_REQUEST, self.handle_data_request)
        self.event_bus.subscribe(EventType.REGIME_REQUEST, self.handle_regime_request)
        self.event_bus.subscribe(EventType.VOLATILITY_REQUEST, self.handle_volatility_request)
        
        logger.info("Subscribed to data request events")
    
    def handle_data_request(self, event: Event):
        """Handle data request events."""
        request_type = event.data.get('request_type')
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe', '1h')
        source = event.data.get('source', self.default_source)
        
        if request_type == 'market_data':
            data = self.get_market_data(symbol, timeframe, source)
            
            # Publish data response event
            self.event_bus.publish(Event(
                event_type=EventType.DATA_RESPONSE,
                data={
                    'request_type': request_type,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'source': source,
                    'data': data
                }
            ))
        
        elif request_type == 'all':
            # Get all data types for comprehensive context
            market_data = self.get_market_data(symbol, timeframe, source)
            volatility = self.get_volatility(symbol, timeframe, source)
            regime = self.get_market_regime(symbol, timeframe, source)
            
            # Publish comprehensive data response
            self.event_bus.publish(Event(
                event_type=EventType.DATA_RESPONSE,
                data={
                    'request_type': 'all',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'source': source,
                    'market_data': market_data,
                    'volatility': volatility,
                    'regime': regime
                }
            ))
    
    def handle_regime_request(self, event: Event):
        """Handle market regime request events."""
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe', '1h')
        source = event.data.get('source', self.default_source)
        
        regime = self.get_market_regime(symbol, timeframe, source)
        
        # Publish regime response event
        self.event_bus.publish(Event(
            event_type=EventType.REGIME_RESPONSE,
            data={
                'symbol': symbol,
                'timeframe': timeframe,
                'source': source,
                'regime': regime
            }
        ))
    
    def handle_volatility_request(self, event: Event):
        """Handle volatility request events."""
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe', '1h')
        source = event.data.get('source', self.default_source)
        
        volatility = self.get_volatility(symbol, timeframe, source)
        
        # Publish volatility response event
        self.event_bus.publish(Event(
            event_type=EventType.VOLATILITY_RESPONSE,
            data={
                'symbol': symbol,
                'timeframe': timeframe,
                'source': source,
                'volatility': volatility
            }
        ))
    
    def get_market_data(self, 
                      symbol: str, 
                      timeframe: str = '1h', 
                      source: str = None) -> Optional[pd.DataFrame]:
        """
        Get market data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            source: Data source
            
        Returns:
            DataFrame with market data or None if not available
        """
        source = source or self.default_source
        key = f"{symbol}_{timeframe}"
        
        # Check if we need to update the data
        current_time = datetime.now()
        last_update = self._last_updates['market_data'].get(key, datetime.min)
        update_interval = timedelta(seconds=self.update_intervals['market_data'])
        
        if current_time - last_update > update_interval or key not in self._market_data:
            # Need to fetch new data
            try:
                logger.debug(f"Fetching new market data for {symbol} ({timeframe}) from {source}")
                
                if source == 'csv':
                    data = self._load_from_csv(symbol, timeframe)
                elif source == 'oanda':
                    data = self._load_from_oanda(symbol, timeframe)
                else:
                    # Default to CSV
                    data = self._load_from_csv(symbol, timeframe)
                
                if data is not None:
                    self._market_data[key] = data
                    self._last_updates['market_data'][key] = current_time
                    logger.debug(f"Updated market data for {symbol} ({timeframe})")
                else:
                    logger.warning(f"Failed to update market data for {symbol} ({timeframe})")
            
            except Exception as e:
                logger.error(f"Error updating market data for {symbol} ({timeframe}): {str(e)}")
                # Return existing data if we have it
                if key in self._market_data:
                    return self._market_data[key]
                return None
        
        # Return the data
        return self._market_data.get(key)
    
    def get_volatility(self, 
                     symbol: str, 
                     timeframe: str = '1h', 
                     source: str = None) -> Dict[str, Any]:
        """
        Get volatility metrics for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            source: Data source
            
        Returns:
            Dictionary with volatility metrics
        """
        source = source or self.default_source
        key = f"{symbol}_{timeframe}"
        
        # Check if we need to update the data
        current_time = datetime.now()
        last_update = self._last_updates['volatility'].get(key, datetime.min)
        update_interval = timedelta(seconds=self.update_intervals['volatility'])
        
        if current_time - last_update > update_interval or key not in self._volatility_data:
            # Get market data first
            market_data = self.get_market_data(symbol, timeframe, source)
            
            if market_data is not None:
                try:
                    # Calculate volatility metrics
                    logger.debug(f"Calculating volatility for {symbol} ({timeframe})")
                    
                    # Calculate daily returns
                    if 'close' in market_data.columns:
                        returns = market_data['close'].pct_change().dropna()
                        
                        # Calculate volatility metrics
                        historical_vol = returns.std() * np.sqrt(252)  # Annualized
                        
                        # Calculate ATR if possible
                        atr = None
                        if all(col in market_data.columns for col in ['high', 'low', 'close']):
                            # Simple ATR calculation
                            high_low = market_data['high'] - market_data['low']
                            high_close = np.abs(market_data['high'] - market_data['close'].shift())
                            low_close = np.abs(market_data['low'] - market_data['close'].shift())
                            
                            ranges = pd.concat([high_low, high_close, low_close], axis=1)
                            true_range = ranges.max(axis=1)
                            atr = true_range.rolling(14).mean().iloc[-1]
                        
                        # Determine volatility state
                        volatility_state = 'medium'  # Default
                        
                        # Simple classification based on historical percentiles
                        if len(returns) > 30:
                            vol_percentile = np.percentile(returns.rolling(30).std(), [25, 75])
                            current_vol = returns.tail(30).std()
                            
                            if current_vol < vol_percentile[0]:
                                volatility_state = 'low'
                            elif current_vol > vol_percentile[1]:
                                volatility_state = 'high'
                        
                        # Create volatility data dictionary
                        volatility_data = {
                            'historical_vol': historical_vol,
                            'atr': atr,
                            'volatility_state': volatility_state,
                            'current_vol': returns.tail(30).std() * np.sqrt(252),
                            'updated_at': current_time
                        }
                        
                        # Store the data
                        self._volatility_data[key] = volatility_data
                        self._last_updates['volatility'][key] = current_time
                        
                        # Publish volatility update event
                        self.event_bus.publish(Event(
                            event_type=EventType.VOLATILITY_UPDATE,
                            data={
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'volatility_state': volatility_state,
                                'historical_vol': float(historical_vol),
                                'atr': float(atr) if atr is not None else None
                            }
                        ))
                        
                        logger.debug(f"Updated volatility for {symbol} ({timeframe}): {volatility_state}")
                    else:
                        logger.warning(f"Market data for {symbol} missing 'close' column")
                
                except Exception as e:
                    logger.error(f"Error calculating volatility for {symbol} ({timeframe}): {str(e)}")
                    # Return existing data if we have it
                    if key in self._volatility_data:
                        return self._volatility_data[key]
                    
                    # Return default
                    return {
                        'historical_vol': None,
                        'atr': None,
                        'volatility_state': 'medium',
                        'current_vol': None,
                        'updated_at': current_time
                    }
        
        # Return the data (with defaults if not available)
        volatility_data = self._volatility_data.get(key, {
            'historical_vol': None,
            'atr': None,
            'volatility_state': 'medium',
            'current_vol': None,
            'updated_at': current_time
        })
        
        return volatility_data
    
    def get_correlation_matrix(self, 
                             symbols: List[str], 
                             timeframe: str = '1h',
                             source: str = None) -> Optional[pd.DataFrame]:
        """
        Get correlation matrix for a list of symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for the data
            source: Data source
            
        Returns:
            DataFrame with correlation matrix or None if not available
        """
        source = source or self.default_source
        key = f"correlation_{timeframe}"
        
        # Check if we need to update the data
        current_time = datetime.now()
        last_update = self._last_updates['correlation']
        update_interval = timedelta(seconds=self.update_intervals['correlation'])
        
        if current_time - last_update > update_interval or key not in self._correlation_data:
            # Need to calculate correlation matrix
            try:
                # Get market data for all symbols
                close_data = {}
                for symbol in symbols:
                    market_data = self.get_market_data(symbol, timeframe, source)
                    if market_data is not None and 'close' in market_data.columns:
                        close_data[symbol] = market_data['close']
                
                if close_data:
                    # Create DataFrame with close prices
                    close_df = pd.DataFrame(close_data)
                    
                    # Calculate correlation matrix
                    correlation_matrix = close_df.pct_change().dropna().corr()
                    
                    # Store the data
                    self._correlation_data[key] = correlation_matrix
                    self._last_updates['correlation'] = current_time
                    
                    # Calculate average correlation
                    corr_values = correlation_matrix.values
                    np.fill_diagonal(corr_values, np.nan)
                    avg_correlation = np.nanmean(np.abs(corr_values))
                    
                    # Determine correlation state
                    correlation_state = 'medium'
                    if avg_correlation < 0.4:
                        correlation_state = 'low'
                    elif avg_correlation > 0.7:
                        correlation_state = 'high'
                    
                    # Publish correlation update event
                    self.event_bus.publish(Event(
                        event_type=EventType.CORRELATION_UPDATE,
                        data={
                            'timeframe': timeframe,
                            'correlation_state': correlation_state,
                            'average_correlation': float(avg_correlation),
                            'correlation_matrix': correlation_matrix.to_dict()
                        }
                    ))
                    
                    logger.debug(f"Updated correlation matrix for {timeframe}: {correlation_state}")
                    
                    return correlation_matrix
                
                else:
                    logger.warning("Failed to get close data for correlation calculation")
                    return self._correlation_data.get(key)
            
            except Exception as e:
                logger.error(f"Error calculating correlation matrix: {str(e)}")
                return self._correlation_data.get(key)
        
        # Return the existing data
        return self._correlation_data.get(key)
    
    def get_market_regime(self, 
                        symbol: str, 
                        timeframe: str = '1h',
                        source: str = None) -> Dict[str, Any]:
        """
        Get market regime for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            source: Data source
            
        Returns:
            Dictionary with market regime information
        """
        source = source or self.default_source
        key = f"{symbol}_{timeframe}"
        
        # Check if we need to update the data
        current_time = datetime.now()
        last_update = self._last_updates['regime'].get(key, datetime.min)
        update_interval = timedelta(seconds=self.update_intervals['regime'])
        
        if current_time - last_update > update_interval or key not in self._regime_data:
            # Get market data
            market_data = self.get_market_data(symbol, timeframe, source)
            
            if market_data is not None:
                try:
                    # Calculate regime indicators
                    logger.debug(f"Calculating market regime for {symbol} ({timeframe})")
                    
                    # Simplified regime detection using common indicators
                    # In a real system, you would use a more sophisticated approach
                    if 'close' in market_data.columns:
                        # Calculate EMAs for trend direction
                        market_data['ema20'] = market_data['close'].ewm(span=20).mean()
                        market_data['ema50'] = market_data['close'].ewm(span=50).mean()
                        
                        # ADX for trend strength
                        adx = self._calculate_adx(market_data)
                        
                        # Bollinger bands for volatility and mean reversion
                        market_data['sma20'] = market_data['close'].rolling(20).mean()
                        market_data['std20'] = market_data['close'].rolling(20).std()
                        market_data['upper_band'] = market_data['sma20'] + (market_data['std20'] * 2)
                        market_data['lower_band'] = market_data['sma20'] - (market_data['std20'] * 2)
                        
                        # Get latest data
                        latest = market_data.iloc[-1]
                        prev = market_data.iloc[-2]
                        
                        # Determine regime
                        regime = 'unknown'
                        confidence = 0.5
                        
                        # Strong trend
                        if adx > 25:
                            if latest['ema20'] > latest['ema50'] and latest['close'] > latest['ema20']:
                                regime = 'trending_up'
                                confidence = min(0.5 + (adx / 100), 0.95)
                            elif latest['ema20'] < latest['ema50'] and latest['close'] < latest['ema20']:
                                regime = 'trending_down'
                                confidence = min(0.5 + (adx / 100), 0.95)
                        
                        # Ranging market
                        elif adx < 20:
                            band_width = (latest['upper_band'] - latest['lower_band']) / latest['sma20']
                            if band_width < 0.05:  # Tight bands
                                regime = 'ranging'
                                confidence = 0.7
                            else:
                                regime = 'ranging'
                                confidence = 0.6
                        
                        # Breakout potential
                        breakout_potential = False
                        if latest['close'] > latest['upper_band'] or latest['close'] < latest['lower_band']:
                            if market_data['close'].pct_change().abs().tail(5).mean() > 0.005:  # Increased volatility
                                regime = 'breakout'
                                confidence = 0.75
                                breakout_potential = True
                        
                        # Create regime data
                        regime_data = {
                            'regime': regime,
                            'confidence': confidence,
                            'adx': adx,
                            'trend_direction': 'up' if latest['ema20'] > latest['ema50'] else 'down',
                            'band_width': (latest['upper_band'] - latest['lower_band']) / latest['sma20'],
                            'breakout_potential': breakout_potential,
                            'updated_at': current_time
                        }
                        
                        # Store the data
                        self._regime_data[key] = regime_data
                        self._last_updates['regime'][key] = current_time
                        
                        # Publish regime change event
                        previous_regime = self._regime_data.get(key, {}).get('regime', 'unknown')
                        if regime != previous_regime:
                            self.event_bus.publish(Event(
                                event_type=EventType.MARKET_REGIME_CHANGE,
                                data={
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'regime': regime,
                                    'previous_regime': previous_regime,
                                    'confidence': confidence,
                                    'adx': float(adx)
                                }
                            ))
                            
                            logger.info(f"Market regime change for {symbol} ({timeframe}): {previous_regime} -> {regime}")
                        
                        return regime_data
                    
                    else:
                        logger.warning(f"Market data for {symbol} missing 'close' column")
                
                except Exception as e:
                    logger.error(f"Error calculating market regime for {symbol} ({timeframe}): {str(e)}")
                    # Return existing data if we have it
                    if key in self._regime_data:
                        return self._regime_data[key]
                    
                    # Return default
                    return {
                        'regime': 'unknown',
                        'confidence': 0.5,
                        'updated_at': current_time
                    }
        
        # Return the data (with defaults if not available)
        regime_data = self._regime_data.get(key, {
            'regime': 'unknown',
            'confidence': 0.5,
            'updated_at': current_time
        })
        
        return regime_data
    
    def update_all(self, symbols: List[str], timeframes: List[str] = None):
        """
        Update all data for a list of symbols and timeframes.
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
        """
        timeframes = timeframes or ['1h']
        
        logger.info(f"Updating all data for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Update market data first
        for symbol in symbols:
            for timeframe in timeframes:
                self.get_market_data(symbol, timeframe)
        
        # Update volatility
        for symbol in symbols:
            for timeframe in timeframes:
                self.get_volatility(symbol, timeframe)
        
        # Update correlation matrix
        self.get_correlation_matrix(symbols, timeframes[0])
        
        # Update regime data
        for symbol in symbols:
            for timeframe in timeframes:
                self.get_market_regime(symbol, timeframe)
        
        logger.info("Data update complete")
    
    def _load_from_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load market data from CSV files."""
        try:
            # Determine the file path based on configuration
            base_path = self.config.get('csv_data_path', 'data/market_data')
            file_path = os.path.join(base_path, f"{symbol}_{timeframe}.csv")
            
            if os.path.exists(file_path):
                # Load the data
                data = pd.read_csv(file_path)
                
                # Convert datetime column if needed
                if 'datetime' in data.columns:
                    data['datetime'] = pd.to_datetime(data['datetime'])
                    data.set_index('datetime', inplace=True)
                
                return data
            else:
                logger.warning(f"CSV file not found: {file_path}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading data from CSV: {str(e)}")
            return None
    
    def _load_from_oanda(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load market data from Oanda API."""
        # This would be implemented with the actual Oanda API
        # For now, we'll just return None
        logger.warning("Oanda data source not implemented")
        return None
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)."""
        try:
            # Calculate +DM and -DM
            plus_dm = data['high'].diff()
            minus_dm = data['low'].diff(-1).abs()
            
            # Filter +DM and -DM
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # Calculate true range
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate smoothed +DM, -DM, and TR
            smoothed_plus_dm = plus_dm.rolling(period).sum()
            smoothed_minus_dm = minus_dm.rolling(period).sum()
            smoothed_tr = true_range.rolling(period).sum()
            
            # Calculate +DI and -DI
            plus_di = 100 * smoothed_plus_dm / smoothed_tr
            minus_di = 100 * smoothed_minus_dm / smoothed_tr
            
            # Calculate DX
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            
            # Calculate ADX
            adx = dx.rolling(period).mean()
            
            # Return the latest ADX value
            return adx.iloc[-1]
        
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return 15.0  # Default value
