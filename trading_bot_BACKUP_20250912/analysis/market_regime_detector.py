#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Detector

This module implements a sophisticated market regime detector that integrates
with the contextual awareness system to identify current market conditions
and feed this information to the trading strategy selection and position
sizing components.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import os

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, MarketRegime
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects and classifies market regimes to guide trading decisions.
    
    Key regime types:
    - Trending Up: Strong directional price movement upwards
    - Trending Down: Strong directional price movement downwards
    - Ranging: Oscillating price movement within bounds
    - Breakout: Sudden price action breaking patterns
    - Reversal: Change in underlying trend direction
    - Volatility Compression: Period of decreasing volatility
    - Volatility Expansion: Period of increasing volatility
    """
    
    def __init__(self, 
                event_bus: EventBus,
                persistence: PersistenceManager,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market regime detector.
        
        Args:
            event_bus: Central event bus for communication
            persistence: Persistence manager for state
            config: Configuration options
        """
        self.event_bus = event_bus
        self.persistence = persistence
        self.config = config or {}
        
        # Regime detection parameters
        self.detection_params = {
            'trend': {
                'adx_threshold': self.config.get('adx_threshold', 25),
                'ema_periods': self.config.get('ema_periods', [20, 50, 200]),
                'rsi_period': self.config.get('rsi_period', 14),
                'rsi_thresholds': self.config.get('rsi_thresholds', [30, 70])
            },
            'range': {
                'bollinger_period': self.config.get('bollinger_period', 20),
                'bollinger_std': self.config.get('bollinger_std', 2.0),
                'width_threshold': self.config.get('width_threshold', 0.05)
            },
            'breakout': {
                'atr_period': self.config.get('atr_period', 14),
                'atr_multiplier': self.config.get('atr_multiplier', 2.0),
                'volume_factor': self.config.get('volume_factor', 1.5)
            },
            'volatility': {
                'atr_period': self.config.get('atr_period', 14),
                'volatility_lookback': self.config.get('volatility_lookback', 30),
                'percentiles': self.config.get('volatility_percentiles', [25, 75])
            }
        }
        
        # Current regime state
        self.current_regimes = {}
        self.regime_history = {}
        self.volatility_states = {}
        
        # Last calculations
        self._last_calculations = {}
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Load previous state if available
        self._load_state()
        
        logger.info("Market Regime Detector initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        # Data events
        self.event_bus.subscribe(EventType.DATA_RESPONSE, self.handle_data_response)
        self.event_bus.subscribe(EventType.REGIME_REQUEST, self.handle_regime_request)
        
        # User/system initiated analysis events
        self.event_bus.subscribe(EventType.ANALYSIS_REQUEST, self.handle_analysis_request)
        
        logger.info("Subscribed to data events")
    
    def handle_data_response(self, event: Event):
        """
        Handle data response events to automatically update regime detection.
        
        Args:
            event: Data response event
        """
        request_type = event.data.get('request_type')
        
        # We only process market data responses
        if request_type in ['market_data', 'all']:
            symbol = event.data.get('symbol')
            timeframe = event.data.get('timeframe', '1h')
            
            # Extract market data
            market_data = None
            if request_type == 'market_data':
                market_data = event.data.get('data')
            else:  # 'all'
                market_data = event.data.get('market_data')
            
            if market_data is not None and isinstance(market_data, pd.DataFrame):
                # Update regime detection
                self.detect_regime(symbol, timeframe, market_data)
    
    def handle_regime_request(self, event: Event):
        """
        Handle explicit regime request events.
        
        Args:
            event: Regime request event
        """
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe', '1h')
        
        # Generate data request to get the latest data
        self.event_bus.publish(Event(
            event_type=EventType.DATA_REQUEST,
            data={
                'request_type': 'market_data',
                'symbol': symbol,
                'timeframe': timeframe
            }
        ))
    
    def handle_analysis_request(self, event: Event):
        """
        Handle user-initiated analysis requests.
        
        Args:
            event: Analysis request event
        """
        analysis_type = event.data.get('analysis_type')
        
        # We only handle regime analysis requests
        if analysis_type == 'regime':
            symbols = event.data.get('symbols', [])
            timeframe = event.data.get('timeframe', '1h')
            
            # Perform regime analysis for all requested symbols
            for symbol in symbols:
                # Request market data for analysis
                self.event_bus.publish(Event(
                    event_type=EventType.DATA_REQUEST,
                    data={
                        'request_type': 'market_data',
                        'symbol': symbol,
                        'timeframe': timeframe
                    }
                ))
    
    def detect_regime(self, 
                    symbol: str, 
                    timeframe: str, 
                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the current market regime for a symbol and timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: Data timeframe
            market_data: DataFrame with OHLCV data
            
        Returns:
            Regime detection results
        """
        key = f"{symbol}_{timeframe}"
        
        try:
            logger.debug(f"Detecting regime for {symbol} ({timeframe})")
            
            # Apply indicators for regime detection
            data = self._apply_indicators(market_data.copy())
            
            # Run classification algorithms
            results = self._classify_regime(data)
            
            # Get previous regime
            previous_regime = self.current_regimes.get(key, 'unknown')
            
            # Update current regime
            self.current_regimes[key] = results['regime']
            
            # Store last calculation time
            self._last_calculations[key] = datetime.now()
            
            # Update regime history
            if key not in self.regime_history:
                self.regime_history[key] = []
            
            # Record regime change in history
            if results['regime'] != previous_regime:
                self.regime_history[key].append({
                    'from_regime': previous_regime,
                    'to_regime': results['regime'],
                    'timestamp': datetime.now(),
                    'confidence': results['confidence']
                })
                
                # Keep history at a reasonable size
                if len(self.regime_history[key]) > 100:
                    self.regime_history[key] = self.regime_history[key][-100:]
            
            # Update volatility state
            self.volatility_states[key] = results['volatility_state']
            
            # When regime changes, publish event
            if results['regime'] != previous_regime:
                self.event_bus.publish(Event(
                    event_type=EventType.MARKET_REGIME_CHANGE,
                    data={
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'regime': results['regime'],
                        'previous_regime': previous_regime,
                        'confidence': results['confidence'],
                        'details': {
                            'adx': results['adx'],
                            'trend_direction': results['trend_direction'],
                            'rsi': results['rsi'],
                            'volatility_state': results['volatility_state']
                        }
                    }
                ))
                
                logger.info(f"Regime change detected for {symbol} ({timeframe}): {previous_regime} -> {results['regime']} (confidence: {results['confidence']:.2f})")
            
            # Publish volatility update if changed
            previous_volatility = self.volatility_states.get(key, 'medium')
            if results['volatility_state'] != previous_volatility:
                self.event_bus.publish(Event(
                    event_type=EventType.VOLATILITY_UPDATE,
                    data={
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'volatility_state': results['volatility_state'],
                        'previous_state': previous_volatility,
                        'atr': results['atr']
                    }
                ))
                
                logger.info(f"Volatility state change for {symbol} ({timeframe}): {previous_volatility} -> {results['volatility_state']}")
            
            # Save state
            self._save_state()
            
            return results
        
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol} ({timeframe}): {str(e)}")
            
            # Return previous regime or unknown if no previous regime
            previous_regime = self.current_regimes.get(key, 'unknown')
            return {
                'regime': previous_regime,
                'confidence': 0.5,
                'volatility_state': self.volatility_states.get(key, 'medium')
            }
    
    def get_current_regime(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Get the current regime for a symbol and timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: Data timeframe
            
        Returns:
            Current regime information
        """
        key = f"{symbol}_{timeframe}"
        
        # Check if regime is available
        if key in self.current_regimes:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'regime': self.current_regimes[key],
                'volatility_state': self.volatility_states.get(key, 'medium'),
                'last_updated': self._last_calculations.get(key, datetime.min),
                'history': self.regime_history.get(key, [])
            }
        
        # If not available, request regime detection
        self.event_bus.publish(Event(
            event_type=EventType.REGIME_REQUEST,
            data={
                'symbol': symbol,
                'timeframe': timeframe
            }
        ))
        
        # Return unknown regime
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'regime': 'unknown',
            'volatility_state': 'medium',
            'last_updated': datetime.min,
            'history': []
        }
    
    def get_volatility_state(self, symbol: str, timeframe: str = '1h') -> str:
        """
        Get the current volatility state for a symbol and timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: Data timeframe
            
        Returns:
            Volatility state (low, medium, high)
        """
        key = f"{symbol}_{timeframe}"
        return self.volatility_states.get(key, 'medium')
    
    def get_regime_history(self, symbol: str, timeframe: str = '1h') -> List[Dict[str, Any]]:
        """
        Get regime change history for a symbol and timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: Data timeframe
            
        Returns:
            List of regime changes
        """
        key = f"{symbol}_{timeframe}"
        return self.regime_history.get(key, [])
    
    def _apply_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply technical indicators for regime detection.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        # Make sure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Fill in missing columns with close if needed
        for col in required_columns:
            if col not in data.columns and 'close' in data.columns:
                if col in ['open', 'high', 'low']:
                    data[col] = data['close']
                elif col == 'volume':
                    data[col] = 0
        
        # Skip if we don't have close prices
        if 'close' not in data.columns:
            logger.warning("Data missing 'close' column, skipping indicator calculation")
            return data
        
        # Trend indicators
        # EMAs
        for period in self.detection_params['trend']['ema_periods']:
            data[f'ema{period}'] = data['close'].ewm(span=period).mean()
        
        # RSI
        rsi_period = self.detection_params['trend']['rsi_period']
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX (Direction and Strength)
        data['adx'] = self._calculate_adx(data, self.detection_params['trend']['adx_threshold'])
        
        # Volatility indicators
        # Bollinger Bands
        bb_period = self.detection_params['range']['bollinger_period']
        bb_std = self.detection_params['range']['bollinger_std']
        data['sma'] = data['close'].rolling(window=bb_period).mean()
        data['std'] = data['close'].rolling(window=bb_period).std()
        data['upper_band'] = data['sma'] + (data['std'] * bb_std)
        data['lower_band'] = data['sma'] - (data['std'] * bb_std)
        data['bandwidth'] = (data['upper_band'] - data['lower_band']) / data['sma']
        
        # ATR
        atr_period = self.detection_params['volatility']['atr_period']
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Normalized ATR (ATR / Close)
        data['atr_pct'] = data['atr'] / data['close']
        
        # Volatility percentiles for state classification
        lookback = self.detection_params['volatility']['volatility_lookback']
        data['atr_pct_roll'] = data['atr_pct'].rolling(window=lookback).mean()
        
        # Returns for recent movement
        data['ret_1d'] = data['close'].pct_change(1)
        data['ret_5d'] = data['close'].pct_change(5)
        data['ret_20d'] = data['close'].pct_change(20)
        
        # Volume indicators
        if 'volume' in data.columns and data['volume'].sum() > 0:
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        else:
            data['volume_ratio'] = 1.0
        
        return data
    
    def _classify_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify the current market regime based on indicators.
        
        Args:
            data: DataFrame with calculated indicators
            
        Returns:
            Classification results
        """
        # Default results
        results = {
            'regime': 'unknown',
            'confidence': 0.5,
            'adx': 0,
            'rsi': 50,
            'atr': 0,
            'trend_direction': 'neutral',
            'volatility_state': 'medium'
        }
        
        # Ensure we have enough data
        if len(data) < 50:
            logger.warning("Not enough data for regime classification")
            return results
        
        # Get latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        # ADX for trend strength
        adx = latest.get('adx', 0)
        
        # RSI for overbought/oversold
        rsi = latest.get('rsi', 50)
        
        # ATR for volatility
        atr = latest.get('atr', 0)
        atr_pct = latest.get('atr_pct', 0)
        
        # EMAs for trend direction
        ema20 = latest.get('ema20', latest['close'])
        ema50 = latest.get('ema50', latest['close'])
        ema200 = latest.get('ema200', latest['close'])
        
        # Bollinger Bands for ranging/breakout
        bandwidth = latest.get('bandwidth', 0.05)
        upper_band = latest.get('upper_band', latest['close'] * 1.02)
        lower_band = latest.get('lower_band', latest['close'] * 0.98)
        
        # Volume for confirmation
        volume_ratio = latest.get('volume_ratio', 1.0)
        
        # Recent returns
        ret_1d = latest.get('ret_1d', 0)
        ret_5d = latest.get('ret_5d', 0)
        ret_20d = latest.get('ret_20d', 0)
        
        # Determine trend direction
        if ema20 > ema50 and ema50 > ema200:
            trend_direction = 'up'
        elif ema20 < ema50 and ema50 < ema200:
            trend_direction = 'down'
        elif ema20 > ema50 and ema50 < ema200:
            trend_direction = 'up_conflict'
        elif ema20 < ema50 and ema50 > ema200:
            trend_direction = 'down_conflict'
        else:
            trend_direction = 'neutral'
        
        # Determine volatility state
        volatility_state = 'medium'
        atr_pct_roll = data['atr_pct'].rolling(window=30).mean().iloc[-30:].values
        if len(atr_pct_roll) >= 30:
            vol_percentiles = np.percentile(atr_pct_roll, [25, 75])
            if atr_pct < vol_percentiles[0]:
                volatility_state = 'low'
            elif atr_pct > vol_percentiles[1]:
                volatility_state = 'high'
        
        # Classify regime
        regime = 'unknown'
        confidence = 0.5
        
        # Strong trend
        if adx > self.detection_params['trend']['adx_threshold']:
            if trend_direction == 'up' or trend_direction == 'up_conflict':
                if ret_5d > 0 and ret_1d > 0:  # Confirming price action
                    regime = 'trending_up'
                    confidence = min(0.5 + (adx / 100), 0.9)
                else:
                    # Potential reversal
                    if rsi < 30 and latest['close'] < lower_band:
                        regime = 'reversal'
                        confidence = 0.7
                    else:
                        regime = 'trending_up'
                        confidence = 0.6
            
            elif trend_direction == 'down' or trend_direction == 'down_conflict':
                if ret_5d < 0 and ret_1d < 0:  # Confirming price action
                    regime = 'trending_down'
                    confidence = min(0.5 + (adx / 100), 0.9)
                else:
                    # Potential reversal
                    if rsi > 70 and latest['close'] > upper_band:
                        regime = 'reversal'
                        confidence = 0.7
                    else:
                        regime = 'trending_down'
                        confidence = 0.6
        
        # Ranging market
        elif adx < 20:
            if bandwidth < self.detection_params['range']['width_threshold']:
                # Tight range - potential volatility compression
                regime = 'volatility_compression'
                confidence = 0.7
            else:
                regime = 'ranging'
                confidence = 0.6 + (1 - (adx / 20)) * 0.3  # Higher confidence with lower ADX
        
        # Breakout potential
        if latest['close'] > upper_band and volume_ratio > self.detection_params['breakout']['volume_factor']:
            if ret_1d > 0.01:  # 1% daily move
                regime = 'breakout'
                confidence = 0.8
        elif latest['close'] < lower_band and volume_ratio > self.detection_params['breakout']['volume_factor']:
            if ret_1d < -0.01:  # 1% daily move
                regime = 'breakout'
                confidence = 0.8
        
        # Volatility expansion overrides other regimes if very significant
        if volatility_state == 'high' and atr_pct > 3 * data['atr_pct'].rolling(window=30).mean().iloc[-1]:
            regime = 'volatility_expansion'
            confidence = 0.85
        
        # Build results
        results = {
            'regime': regime,
            'confidence': confidence,
            'adx': adx,
            'rsi': rsi,
            'atr': atr,
            'trend_direction': trend_direction,
            'volatility_state': volatility_state,
            'bandwidth': bandwidth,
            'volume_ratio': volume_ratio,
            'ema_alignment': f"{trend_direction}",
            'recent_returns': {
                '1d': ret_1d,
                '5d': ret_5d,
                '20d': ret_20d
            }
        }
        
        return results
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: DataFrame with OHLCV data
            period: ADX calculation period
            
        Returns:
            Series with ADX values
        """
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
            
            return adx
        
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series([0] * len(data), index=data.index)
    
    def _load_state(self):
        """Load previous state from persistence."""
        try:
            if self.persistence:
                state = self.persistence.load_system_state('market_regime_detector')
                if state:
                    self.current_regimes = state.get('current_regimes', {})
                    self.regime_history = state.get('regime_history', {})
                    self.volatility_states = state.get('volatility_states', {})
                    
                    logger.info("Loaded market regime detector state from persistence")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
    
    def _save_state(self):
        """Save current state to persistence."""
        try:
            if self.persistence:
                state = {
                    'current_regimes': self.current_regimes,
                    'regime_history': self.regime_history,
                    'volatility_states': self.volatility_states,
                    'last_updated': datetime.now().isoformat()
                }
                
                self.persistence.save_system_state('market_regime_detector', state)
                logger.debug("Saved market regime detector state to persistence")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
