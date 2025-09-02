#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap Trading Strategy (Gap and Go/Fade)

This module implements a gap trading strategy for stocks, capable of both:
- Gap and Go: Trading in the direction of the gap (continuation)
- Gap Fade: Trading against the gap (reversal)

The strategy identifies significant price gaps between previous close and current open,
then applies a set of rules to determine whether to trade with the gap (continuation)
or against it (fade/reversal) based on volume confirmation, price action, and time decay.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.stocks.base.stocks_base_strategy import StocksBaseStrategy
from trading_bot.strategies.factory.strategy_registry import register_strategy, StrategyType, AssetClass, MarketRegime, TimeFrame
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'stocks',
    'strategy_type': 'gap_trading',
    'compatible_market_regimes': ['volatile', 'trending'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.75,       # Good compatibility with trending markets
        'ranging': 0.50,        # Moderate compatibility with ranging markets
        'volatile': 0.90,       # Excellent compatibility with volatile markets
        'low_volatility': 0.40, # Poor compatibility with low volatility
        'all_weather': 0.70     # Good overall compatibility
    },
    'optimal_parameters': {
        'volatile': {
            'min_gap_percent': 2.0,
            'volume_threshold': 2.0,
            'max_entry_time_minutes': 30,
            'atr_multiplier': 2.0,
            'continuation_preference': 0.3  # Prefer fade in volatile markets
        },
        'trending': {
            'min_gap_percent': 1.5,
            'volume_threshold': 1.5,
            'max_entry_time_minutes': 45,
            'atr_multiplier': 2.5,
            'continuation_preference': 0.7  # Prefer continuation in trending markets
        }
    }
})
class GapTradingStrategy(StocksBaseStrategy):
    """
    Gap Trading Strategy for stocks
    
    This strategy identifies and trades price gaps at market open:
    - Gap and Go: Trades in the direction of the gap (continuation)
    - Gap Fade: Trades against the gap direction (reversal)
    
    Key features:
    - Automated gap detection
    - Volume confirmation
    - Time-based signal decay
    - ATR-based position sizing
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Gap detection parameters
        'min_gap_percent': 1.5,     # Minimum gap size as percentage
        'max_gap_percent': 10.0,    # Maximum gap size to filter outliers
        'gap_types': ['up', 'down'], # Which gap directions to trade
        
        # Entry parameters
        'trade_mode': 'both',       # 'continuation', 'fade', or 'both'
        'volume_threshold': 1.5,    # Minimum volume vs average
        'max_entry_time_minutes': 30, # Maximum time to enter after open
        'continuation_preference': 0.5, # 0.0-1.0 preference for continuation vs fade
        
        # Technical indicators
        'vwap_periods': [20],       # VWAP periods for confirmation
        'rsi_period': 14,           # RSI period for overbought/oversold
        'rsi_overbought': 70,       # RSI threshold for overbought
        'rsi_oversold': 30,         # RSI threshold for oversold
        
        # Risk management
        'atr_period': 14,           # ATR period for volatility
        'atr_multiplier': 2.0,      # ATR multiplier for stops
        'max_risk_per_trade_percent': 0.01, # 1% risk per trade
        'max_gap_trades_per_day': 3, # Maximum gap trades per day
        
        # Profit taking
        'profit_atr_multiple': 1.5,  # Profit target as ATR multiple
        'partial_exit_threshold': 1.0, # First partial exit at this ATR multiple
    }
    
    def __init__(self, name: str = "GapTradingStrategy", 
                parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Gap Trading Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Initialize the base class
        super().__init__(name, parameters, metadata)
        
        # Override defaults with provided parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Strategy-specific state
        self.detected_gaps = {}  # Track detected gaps by symbol
        self.entry_signals = {}  # Track entry signals by symbol
        self.market_open_prices = {}  # Track market open prices
        self.trade_count_today = 0  # Count of trades taken today
        self.last_trading_day = None  # Last trading day
        self.market_open_time = None  # Market open time for the current day
        
        logger.info(f"{name} initialized with parameters: {self.parameters}")
    
    def register_events(self, event_bus: EventBus) -> None:
        """
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        """
        self.event_bus = event_bus
        
        # Register for market data events
        event_bus.register(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        event_bus.register(EventType.TIMEFRAME_COMPLETED, self._on_timeframe_completed)
        
        # Register for market open event
        event_bus.register(EventType.MARKET_OPEN, self._on_market_open)
        
        # Register for market close event to reset daily counters
        event_bus.register(EventType.MARKET_CLOSE, self._on_market_close)
        
        logger.info(f"{self.name} registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open events. Reset gap detection for the new day.
        """
        # Reset daily tracking
        self.detected_gaps = {}
        self.entry_signals = {}
        self.market_open_prices = {}
        
        # Check if this is a new trading day
        current_date = datetime.now().date()
        if self.last_trading_day != current_date:
            self.trade_count_today = 0
            self.last_trading_day = current_date
        
        # Store market open time
        self.market_open_time = datetime.now()
        
        logger.info(f"Market open event processed: {self.market_open_time}")
    
    def _on_market_close(self, event: Event) -> None:
        """
        Handle market close events. Clear any pending signals.
        """
        # Clear any pending signals at market close
        self.entry_signals = {}
        
        logger.info(f"Market close event processed. Trades executed today: {self.trade_count_today}")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        This is where we'll check for gaps and generate signals
        during the market open period.
        """
        # Check if we're still in the gap trading window
        if not self._is_in_trading_window():
            return
        
        # Extract data from the event
        data = event.data.get('data', {})
        symbol = event.data.get('symbol')
        
        if not symbol or not data:
            return
        
        # Process the data for gap detection if we haven't already
        if symbol not in self.detected_gaps:
            self._detect_gap(data, symbol)
        
        # If a gap is detected, monitor for entry conditions
        if symbol in self.detected_gaps and symbol not in self.entry_signals:
            self._check_entry_conditions(data, symbol)
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For gap trading, this is where we'll analyze patterns after
        gaps and update our models.
        """
        # Extract data from the event
        data = event.data.get('data', {})
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe')
        
        if not symbol or not data or not timeframe:
            return
        
        # Only process for specific timeframes we care about
        if timeframe not in ['1m', '5m', '15m']:
            return
        
        # Update signals and check for time decay
        self._update_signals(data, symbol)
    
    def _is_in_trading_window(self) -> bool:
        """
        Check if current time is within the gap trading window.
        
        Returns:
            bool: True if in trading window, False otherwise
        """
        if not self.market_open_time:
            return False
        
        # Get max entry time in minutes
        max_entry_time = self.parameters['max_entry_time_minutes']
        current_time = datetime.now()
        
        # Check if we're within the window
        time_since_open = (current_time - self.market_open_time).total_seconds() / 60
        
        return time_since_open <= max_entry_time
    
    def _detect_gap(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Detect price gaps between previous close and current open.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to detect gaps for
        """
        # Check data has necessary columns and enough history
        if len(data) < 2 or 'close' not in data.columns or 'open' not in data.columns:
            return
        
        # Get previous close and current open
        prev_close = data['close'].iloc[-2]
        current_open = data['open'].iloc[-1]
        
        # Calculate gap percentage
        gap_percent = (current_open - prev_close) / prev_close * 100
        
        # Store market open price
        self.market_open_prices[symbol] = current_open
        
        # Check if gap is significant enough
        min_gap = self.parameters['min_gap_percent']
        max_gap = self.parameters['max_gap_percent']
        
        if abs(gap_percent) >= min_gap and abs(gap_percent) <= max_gap:
            # Determine gap direction
            gap_direction = 'up' if gap_percent > 0 else 'down'
            
            # Check if we want to trade this gap direction
            if gap_direction in self.parameters['gap_types']:
                # Record the gap
                self.detected_gaps[symbol] = {
                    'gap_percent': gap_percent,
                    'direction': gap_direction,
                    'prev_close': prev_close,
                    'open': current_open,
                    'detection_time': datetime.now(),
                    'volume': data['volume'].iloc[-1] if 'volume' in data.columns else None,
                    'avg_volume': data['volume'].rolling(20).mean().iloc[-1] if 'volume' in data.columns else None
                }
                
                logger.info(f"Gap detected in {symbol}: {gap_percent:.2f}% {gap_direction}")
    
    def _check_entry_conditions(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Check if entry conditions are met for a detected gap.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to check entry conditions for
        """
        # Get gap details
        gap_info = self.detected_gaps.get(symbol)
        if not gap_info:
            return
        
        # Check if we've exceeded the max trade count for today
        if self.trade_count_today >= self.parameters['max_gap_trades_per_day']:
            return
        
        # Check volume confirmation if we have volume data
        volume_confirmed = True
        if 'volume' in gap_info and 'avg_volume' in gap_info and gap_info['avg_volume']:
            volume_ratio = gap_info['volume'] / gap_info['avg_volume']
            volume_confirmed = volume_ratio >= self.parameters['volume_threshold']
        
        # Calculate indicators
        indicators = self.calculate_indicators(data, symbol)
        
        # Determine if we should go with or against the gap
        continuation_signal = None
        fade_signal = None
        
        # Calculate signal confidence based on multiple factors
        if volume_confirmed:
            # Determine the continuation and fade signals
            if gap_info['direction'] == 'up':
                # For up gaps, continuation is long, fade is short
                if self._should_trade_continuation(indicators, gap_info):
                    continuation_signal = self._create_signal(symbol, SignalType.BUY, data, gap_info, indicators)
                
                if self._should_trade_fade(indicators, gap_info):
                    fade_signal = self._create_signal(symbol, SignalType.SELL, data, gap_info, indicators)
            else:
                # For down gaps, continuation is short, fade is long
                if self._should_trade_continuation(indicators, gap_info):
                    continuation_signal = self._create_signal(symbol, SignalType.SELL, data, gap_info, indicators)
                
                if self._should_trade_fade(indicators, gap_info):
                    fade_signal = self._create_signal(symbol, SignalType.BUY, data, gap_info, indicators)
            
            # Apply continuation preference to decide between signals
            if continuation_signal and fade_signal:
                # Choose based on continuation preference parameter
                if self.parameters['continuation_preference'] >= 0.5:
                    self.entry_signals[symbol] = continuation_signal
                    logger.info(f"Choosing continuation signal for {symbol} based on preference")
                else:
                    self.entry_signals[symbol] = fade_signal
                    logger.info(f"Choosing fade signal for {symbol} based on preference")
            elif continuation_signal:
                self.entry_signals[symbol] = continuation_signal
            elif fade_signal:
                self.entry_signals[symbol] = fade_signal
            
            # If we've generated a signal, increment trade count
            if symbol in self.entry_signals:
                self.trade_count_today += 1
                
                # Publish the signal to the event bus
                if self.event_bus:
                    self.event_bus.publish(
                        EventType.SIGNAL_GENERATED,
                        {
                            'symbol': symbol,
                            'signal': self.entry_signals[symbol],
                            'strategy': self.name
                        }
                    )
    
    def _should_trade_continuation(self, indicators: Dict[str, Any], gap_info: Dict[str, Any]) -> bool:
        """
        Determine if a continuation trade (with the gap) should be taken.
        
        Args:
            indicators: Dictionary of calculated indicators
            gap_info: Dictionary of gap information
            
        Returns:
            bool: True if continuation trade should be taken
        """
        # Check trade mode
        if self.parameters['trade_mode'] == 'fade':
            return False
        
        # Get indicator values
        rsi = indicators.get('rsi', 50)
        vwap_signal = indicators.get('vwap_signal', 0)
        
        # Rules for up gap continuation
        if gap_info['direction'] == 'up':
            # For up gaps, we want RSI not extremely overbought and price above VWAP
            return (rsi < self.parameters['rsi_overbought'] and 
                    vwap_signal > 0)
        # Rules for down gap continuation
        else:
            # For down gaps, we want RSI not extremely oversold and price below VWAP
            return (rsi > self.parameters['rsi_oversold'] and 
                    vwap_signal < 0)
    
    def _should_trade_fade(self, indicators: Dict[str, Any], gap_info: Dict[str, Any]) -> bool:
        """
        Determine if a fade trade (against the gap) should be taken.
        
        Args:
            indicators: Dictionary of calculated indicators
            gap_info: Dictionary of gap information
            
        Returns:
            bool: True if fade trade should be taken
        """
        # Check trade mode
        if self.parameters['trade_mode'] == 'continuation':
            return False
        
        # Get indicator values
        rsi = indicators.get('rsi', 50)
        vwap_signal = indicators.get('vwap_signal', 0)
        
        # Rules for up gap fade
        if gap_info['direction'] == 'up':
            # For up gaps, we want extreme RSI overbought and price potentially below VWAP
            return (rsi > self.parameters['rsi_overbought'] or 
                    vwap_signal < 0)
        # Rules for down gap fade
        else:
            # For down gaps, we want extreme RSI oversold and price potentially above VWAP
            return (rsi < self.parameters['rsi_oversold'] or 
                    vwap_signal > 0)
    
    def _create_signal(self, symbol: str, signal_type: SignalType, 
                      data: pd.DataFrame, gap_info: Dict[str, Any], 
                      indicators: Dict[str, Any]) -> Signal:
        """
        Create a trading signal for the given symbol and type.
        
        Args:
            symbol: Symbol to create signal for
            signal_type: Type of signal (BUY/SELL)
            data: DataFrame with OHLCV data
            gap_info: Gap information dictionary
            indicators: Calculated indicators
            
        Returns:
            Signal: Trading signal
        """
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Get ATR for stop loss calculation
        atr = indicators.get('atr', data['close'].iloc[-1] * 0.01)  # Fallback to 1% if ATR not available
        
        # Calculate stop loss and take profit based on ATR
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
            take_profit = current_price + (atr * self.parameters['profit_atr_multiple'])
        else:  # SELL
            stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
            take_profit = current_price - (atr * self.parameters['profit_atr_multiple'])
        
        # Calculate confidence score (0.0-1.0)
        # Base on gap size, volume confirmation, and VWAP/RSI alignment
        gap_size_factor = min(abs(gap_info['gap_percent']) / self.parameters['min_gap_percent'], 1.5)
        volume_factor = gap_info.get('volume', 0) / gap_info.get('avg_volume', 1) if gap_info.get('avg_volume', 0) > 0 else 1
        volume_factor = min(volume_factor / self.parameters['volume_threshold'], 1.5)
        
        # Indicator alignment factor
        indicator_alignment = 0.5  # Neutral
        if signal_type == SignalType.BUY:
            # For long signals
            if indicators.get('vwap_signal', 0) > 0:
                indicator_alignment += 0.2  # Above VWAP is good for longs
            if indicators.get('rsi', 50) < 70:
                indicator_alignment += 0.1  # Non-overbought is good for longs
        else:
            # For short signals
            if indicators.get('vwap_signal', 0) < 0:
                indicator_alignment += 0.2  # Below VWAP is good for shorts
            if indicators.get('rsi', 50) > 30:
                indicator_alignment += 0.1  # Non-oversold is good for shorts
        
        # Time decay factor (confidence decreases as time passes)
        time_since_open = (datetime.now() - self.market_open_time).total_seconds() / 60
        time_factor = max(0.5, 1.0 - (time_since_open / self.parameters['max_entry_time_minutes']))
        
        # Final confidence calculation
        confidence = min(0.9, (0.6 * gap_size_factor + 0.5 * volume_factor + 0.7 * indicator_alignment) * time_factor)
        
        # Create signal
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            price=current_price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': self.name,
                'gap_percent': gap_info['gap_percent'],
                'gap_direction': gap_info['direction'],
                'atr': atr,
                'indicators': {
                    'rsi': indicators.get('rsi'),
                    'vwap_signal': indicators.get('vwap_signal')
                },
                'trade_type': 'continuation' if (
                    (gap_info['direction'] == 'up' and signal_type == SignalType.BUY) or
                    (gap_info['direction'] == 'down' and signal_type == SignalType.SELL)
                ) else 'fade'
            }
        )
    
    def _update_signals(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Update signals based on new data and check for time decay.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to update signals for
        """
        # Check if we're still in the trading window
        if not self._is_in_trading_window():
            # Clear signals if outside window
            if symbol in self.entry_signals:
                logger.info(f"Gap trade window closed for {symbol}, clearing signal")
                del self.entry_signals[symbol]
            return
        
        # Check if we have an active signal
        if symbol not in self.entry_signals:
            return
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Get signal
        signal = self.entry_signals[symbol]
        
        # Check if stop loss or take profit has been hit
        if signal.signal_type == SignalType.BUY:
            if current_price <= signal.stop_loss:
                logger.info(f"Stop loss hit for {symbol}, clearing signal")
                del self.entry_signals[symbol]
            elif current_price >= signal.take_profit:
                logger.info(f"Take profit hit for {symbol}, clearing signal")
                del self.entry_signals[symbol]
        else:  # SELL
            if current_price >= signal.stop_loss:
                logger.info(f"Stop loss hit for {symbol}, clearing signal")
                del self.entry_signals[symbol]
            elif current_price <= signal.take_profit:
                logger.info(f"Take profit hit for {symbol}, clearing signal")
                del self.entry_signals[symbol]
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Check data has necessary columns and enough history
        if len(data) < 20 or 'close' not in data.columns:
            return indicators
        
        # Calculate VWAP
        vwap_periods = self.parameters['vwap_periods']
        for period in vwap_periods:
            if len(data) >= period:
                # VWAP calculation
                vwap = self._calculate_vwap(data, period)
                indicators[f'vwap_{period}'] = vwap.iloc[-1] if not vwap.empty else None
        
        # Determine VWAP signal (above/below)
        if 'vwap_20' in indicators and indicators['vwap_20'] is not None:
            current_price = data['close'].iloc[-1]
            indicators['vwap_signal'] = 1 if current_price > indicators['vwap_20'] else -1
        else:
            indicators['vwap_signal'] = 0
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        if len(data) >= rsi_period + 1:
            rsi = self._calculate_rsi(data['close'], rsi_period)
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate ATR
        atr_period = self.parameters['atr_period']
        if len(data) >= atr_period and all(col in data.columns for col in ['high', 'low', 'close']):
            atr = self._calculate_atr(data, atr_period)
            indicators['atr'] = atr.iloc[-1] if not atr.empty else data['close'].iloc[-1] * 0.01
        
        return indicators
    
    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for VWAP calculation
            
        Returns:
            Series with VWAP values
        """
        if 'volume' not in data.columns:
            return pd.Series()
        
        # Typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # VWAP calculation
        return (typical_price * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: Period for RSI calculation
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for ATR calculation
            
        Returns:
            Series with ATR values
        """
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Return the current entry signals
        return self.entry_signals.copy()
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size for the signal based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units
        """
        # Extract parameters
        max_risk_percent = self.parameters['max_risk_per_trade_percent']
        risk_amount = account_balance * max_risk_percent
        
        # Extract metadata
        atr = signal.metadata.get('atr', signal.price * 0.01)  # Default to 1% if ATR not available
        
        # Calculate position size based on ATR for stop loss
        if signal.stop_loss is not None and signal.price != signal.stop_loss:
            # Risk per share
            risk_per_share = abs(signal.price - signal.stop_loss)
            
            # Position size
            position_size = risk_amount / risk_per_share
        else:
            # Fallback using ATR
            stop_distance = atr * self.parameters['atr_multiplier']
            position_size = risk_amount / stop_distance
        
        # Return the position size
        return position_size
