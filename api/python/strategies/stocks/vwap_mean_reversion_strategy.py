#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP Mean Reversion Strategy

This module implements a mean reversion strategy based on the Volume Weighted Average Price (VWAP).
The strategy identifies when price deviates significantly from VWAP and takes positions
expecting a reversion back toward the VWAP line.
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
    'strategy_type': 'mean_reversion',
    'compatible_market_regimes': ['ranging', 'low_volatility'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.40,        # Poor in strong trends
        'ranging': 0.90,         # Excellent in range-bound markets
        'volatile': 0.50,        # Moderate in volatile markets
        'low_volatility': 0.80,  # Very good in low volatility
        'all_weather': 0.60      # Good overall compatibility
    },
    'optimal_parameters': {
        'ranging': {
            'deviation_threshold': 1.8,
            'vwap_periods': [20, 50],
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30
        },
        'low_volatility': {
            'deviation_threshold': 1.5,
            'vwap_periods': [20, 50],
            'rsi_period': 14,
            'rsi_overbought': 75,
            'rsi_oversold': 25
        }
    }
})
class VWAPMeanReversionStrategy(StocksBaseStrategy):
    """
    VWAP Mean Reversion Strategy for stocks
    
    This strategy identifies significant deviations from VWAP and takes
    contrarian positions when prices are likely to revert to the mean.
    
    Key features:
    - VWAP deviation calculation
    - RSI filter for overbought/oversold confirmation
    - Multiple timeframe analysis
    - Volatility-based position sizing
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # VWAP parameters
        'vwap_periods': [20, 50, 100],  # VWAP calculation periods
        'deviation_threshold': 1.8,     # Standard deviations from VWAP to trigger signal
        'entry_max_time_minutes': 120,  # Maximum time a signal remains valid
        
        # RSI parameters
        'rsi_period': 14,           # RSI calculation period
        'rsi_overbought': 70,       # RSI overbought threshold
        'rsi_oversold': 30,         # RSI oversold threshold
        'use_rsi_filter': True,     # Whether to use RSI as additional filter
        
        # Bollinger Band parameters
        'bb_period': 20,            # Bollinger Band period
        'bb_std': 2.0,              # Standard deviations for Bollinger Bands
        
        # Entry/exit parameters
        'profit_target_multiplier': 2.0,  # Profit target as multiple of risk
        'max_trades_per_day': 5,    # Maximum trades per day
        'min_volume_percentile': 30,  # Minimum volume percentile to take trades
        
        # Risk management
        'atr_period': 14,           # ATR period for volatility
        'atr_multiplier': 2.0,      # ATR multiplier for stops
        'max_risk_per_trade_percent': 0.01  # 1% risk per trade
    }
    
    def __init__(self, name: str = "VWAPMeanReversionStrategy", 
                parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize VWAP Mean Reversion Strategy.
        
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
        self.active_signals = {}  # Active signals by symbol
        self.vwap_data = {}      # VWAP calculations by symbol
        self.trade_count_today = 0
        self.last_trading_day = None
        
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
        
        # Register for market open/close events
        event_bus.register(EventType.MARKET_OPEN, self._on_market_open)
        event_bus.register(EventType.MARKET_CLOSE, self._on_market_close)
        
        logger.info(f"{self.name} registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """Handle market open events."""
        # Check if this is a new trading day
        current_date = datetime.now().date()
        if self.last_trading_day != current_date:
            self.trade_count_today = 0
            self.last_trading_day = current_date
        
        logger.info(f"Market open event processed. Trade count reset.")
    
    def _on_market_close(self, event: Event) -> None:
        """Handle market close events."""
        # Clear active signals at market close
        self.active_signals = {}
        logger.info(f"Market close event processed. Trades today: {self.trade_count_today}")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        """
        # Extract data from the event
        data = event.data.get('data', {})
        symbol = event.data.get('symbol')
        
        if not symbol or not data:
            return
            
        # Check for max trades per day
        if self.trade_count_today >= self.parameters['max_trades_per_day']:
            return
        
        # Process data for VWAP calculations
        self._update_vwap_data(data, symbol)
        
        # Check for mean reversion signals
        if symbol not in self.active_signals:
            self._check_for_signals(data, symbol)
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        """
        # Extract data from the event
        data = event.data.get('data', {})
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe')
        
        if not symbol or not data or not timeframe:
            return
        
        # Only process for relevant timeframes
        if timeframe not in ['1m', '5m', '15m']:
            return
            
        # Update signals if we have any for this symbol
        if symbol in self.active_signals:
            self._update_signal(data, symbol)
    
    def _update_vwap_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Update VWAP calculations for a symbol.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to update VWAP for
        """
        # Check if we have enough data
        if len(data) < max(self.parameters['vwap_periods']):
            return
            
        # Store or update VWAP data
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = {}
            
        # Calculate VWAP for each period
        for period in self.parameters['vwap_periods']:
            vwap = self._calculate_vwap(data, period)
            self.vwap_data[symbol][f'vwap_{period}'] = vwap.iloc[-1] if not vwap.empty else None
            
            # Calculate standard deviation of price from VWAP
            if not vwap.empty:
                # Calculate daily typical price
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                
                # Calculate standard deviation of price from VWAP
                std_dev = np.std(typical_price[-period:] - vwap[-period:])
                self.vwap_data[symbol][f'vwap_{period}_std'] = std_dev
    
    def _check_for_signals(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Check for mean reversion signals based on VWAP deviation.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to check signals for
        """
        # Check if we have VWAP data for this symbol
        if symbol not in self.vwap_data:
            return
            
        # Calculate indicators
        indicators = self.calculate_indicators(data, symbol)
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Check for significant deviations from VWAP
        signal = None
        
        # Use primary VWAP period (first in the list)
        primary_period = self.parameters['vwap_periods'][0]
        vwap_key = f'vwap_{primary_period}'
        std_key = f'vwap_{primary_period}_std'
        
        if vwap_key in self.vwap_data[symbol] and std_key in self.vwap_data[symbol]:
            vwap = self.vwap_data[symbol][vwap_key]
            std_dev = self.vwap_data[symbol][std_key]
            
            if vwap is not None and std_dev is not None:
                # Calculate deviation in standard deviations
                deviation = (current_price - vwap) / std_dev
                
                # Check volume percentile if we have volume data
                volume_ok = True
                if 'volume_percentile' in indicators and self.parameters['min_volume_percentile'] > 0:
                    volume_ok = indicators['volume_percentile'] >= self.parameters['min_volume_percentile']
                
                # Check for buy signal (price below VWAP by threshold)
                if (deviation <= -self.parameters['deviation_threshold'] and 
                        volume_ok and 
                        self._check_rsi_filter(indicators, 'buy')):
                    
                    signal = self._create_signal(symbol, SignalType.BUY, current_price, indicators)
                    
                # Check for sell signal (price above VWAP by threshold)
                elif (deviation >= self.parameters['deviation_threshold'] and 
                        volume_ok and 
                        self._check_rsi_filter(indicators, 'sell')):
                    
                    signal = self._create_signal(symbol, SignalType.SELL, current_price, indicators)
        
        # If we have a signal, add it to active signals and publish
        if signal:
            self.active_signals[symbol] = {
                'signal': signal,
                'entry_time': datetime.now(),
                'vwap_data': self.vwap_data[symbol].copy()
            }
            
            # Increment trade count
            self.trade_count_today += 1
            
            # Publish signal
            if self.event_bus:
                self.event_bus.publish(
                    EventType.SIGNAL_GENERATED,
                    {
                        'symbol': symbol,
                        'signal': signal,
                        'strategy': self.name
                    }
                )
            
            logger.info(f"Generated {signal.signal_type.value} signal for {symbol} with confidence {signal.confidence:.2f}")
    
    def _check_rsi_filter(self, indicators: Dict[str, Any], signal_type: str) -> bool:
        """
        Check if RSI confirms the signal direction.
        
        Args:
            indicators: Dictionary of calculated indicators
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if RSI confirms or filter is disabled
        """
        # If RSI filter is disabled, always return True
        if not self.parameters['use_rsi_filter']:
            return True
            
        # Get RSI value
        rsi = indicators.get('rsi')
        
        # If RSI not available, pass the filter
        if rsi is None:
            return True
            
        # Check RSI for buy signals
        if signal_type == 'buy':
            return rsi <= self.parameters['rsi_oversold']
        
        # Check RSI for sell signals
        elif signal_type == 'sell':
            return rsi >= self.parameters['rsi_overbought']
            
        return False
    
    def _create_signal(self, symbol: str, signal_type: SignalType, 
                      price: float, indicators: Dict[str, Any]) -> Signal:
        """
        Create a trading signal for the given symbol and type.
        
        Args:
            symbol: Symbol to create signal for
            signal_type: Type of signal
            price: Current price
            indicators: Calculated indicators
            
        Returns:
            Signal: Trading signal
        """
        # Get ATR for stop loss calculation
        atr = indicators.get('atr', price * 0.01)  # Fallback to 1% if ATR not available
        
        # Calculate stop loss and take profit
        if signal_type == SignalType.BUY:
            stop_loss = price - (atr * self.parameters['atr_multiplier'])
            take_profit = price + (atr * self.parameters['atr_multiplier'] * self.parameters['profit_target_multiplier'])
        else:
            stop_loss = price + (atr * self.parameters['atr_multiplier'])
            take_profit = price - (atr * self.parameters['atr_multiplier'] * self.parameters['profit_target_multiplier'])
        
        # Calculate confidence based on multiple factors
        deviation_confidence = min(1.0, abs(indicators.get('vwap_deviation', 0)) / 
                                  (self.parameters['deviation_threshold'] * 1.5))
        
        rsi_confidence = 0.5  # Neutral by default
        rsi = indicators.get('rsi')
        if rsi is not None:
            if signal_type == SignalType.BUY:
                # Lower RSI = higher confidence for buy
                rsi_confidence = max(0.5, 1.0 - (rsi / 100.0))
            else:
                # Higher RSI = higher confidence for sell
                rsi_confidence = max(0.5, rsi / 100.0)
        
        # Bollinger band confirmation
        bb_confidence = 0.5  # Neutral by default
        bb_position = indicators.get('bb_position')
        if bb_position is not None:
            if signal_type == SignalType.BUY and bb_position < 0:
                # Lower BB position = higher confidence for buy
                bb_confidence = min(1.0, 0.5 + abs(bb_position) / 2)
            elif signal_type == SignalType.SELL and bb_position > 0:
                # Higher BB position = higher confidence for sell
                bb_confidence = min(1.0, 0.5 + bb_position / 2)
        
        # Final confidence score
        confidence = min(0.95, (0.5 * deviation_confidence + 0.3 * rsi_confidence + 0.2 * bb_confidence))
        
        # Create and return signal
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': self.name,
                'vwap_deviation': indicators.get('vwap_deviation'),
                'rsi': indicators.get('rsi'),
                'atr': atr,
                'bb_position': indicators.get('bb_position')
            }
        )
    
    def _update_signal(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Update an existing signal based on new data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to update signal for
        """
        # Check if we have a signal for this symbol
        if symbol not in self.active_signals:
            return
            
        # Get signal data
        signal_data = self.active_signals[symbol]
        signal = signal_data['signal']
        entry_time = signal_data['entry_time']
        
        # Check for signal expiration
        current_time = datetime.now()
        time_elapsed = (current_time - entry_time).total_seconds() / 60
        
        if time_elapsed > self.parameters['entry_max_time_minutes']:
            logger.info(f"Signal for {symbol} expired after {time_elapsed:.1f} minutes")
            del self.active_signals[symbol]
            return
            
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Check for stop loss or take profit hit
        if signal.signal_type == SignalType.BUY:
            if current_price <= signal.stop_loss:
                logger.info(f"Stop loss hit for {symbol} buy signal")
                del self.active_signals[symbol]
            elif current_price >= signal.take_profit:
                logger.info(f"Take profit hit for {symbol} buy signal")
                del self.active_signals[symbol]
        else:  # SELL signal
            if current_price >= signal.stop_loss:
                logger.info(f"Stop loss hit for {symbol} sell signal")
                del self.active_signals[symbol]
            elif current_price <= signal.take_profit:
                logger.info(f"Take profit hit for {symbol} sell signal")
                del self.active_signals[symbol]
    
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
        
        # Check if we have enough data
        min_periods = max([
            self.parameters['rsi_period'],
            self.parameters['atr_period'],
            self.parameters['bb_period'],
            max(self.parameters['vwap_periods'])
        ])
        
        if len(data) < min_periods:
            return indicators
            
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], self.parameters['rsi_period'])
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else None
        
        # Calculate ATR
        atr = self._calculate_atr(data, self.parameters['atr_period'])
        indicators['atr'] = atr.iloc[-1] if not atr.empty else None
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            data['close'], 
            self.parameters['bb_period'], 
            self.parameters['bb_std']
        )
        
        # Calculate BB position (-1 to 1, where -1 is at lower band, 0 is at middle, 1 is at upper)
        if not bb_upper.empty and not bb_lower.empty:
            current_price = data['close'].iloc[-1]
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            
            if bb_range > 0:
                # Normalize position between bands to -1 to 1
                bb_position = 2 * ((current_price - bb_lower.iloc[-1]) / bb_range - 0.5)
                indicators['bb_position'] = bb_position
        
        # Calculate volume percentile if volume data is available
        if 'volume' in data.columns and len(data) >= 20:
            volume = data['volume'].iloc[-1]
            volume_percentile = percentileofscore(data['volume'][-20:], volume)
            indicators['volume_percentile'] = volume_percentile
        
        # Calculate VWAP deviation if we have VWAP data
        if symbol in self.vwap_data:
            primary_period = self.parameters['vwap_periods'][0]
            vwap_key = f'vwap_{primary_period}'
            std_key = f'vwap_{primary_period}_std'
            
            if vwap_key in self.vwap_data[symbol] and std_key in self.vwap_data[symbol]:
                vwap = self.vwap_data[symbol][vwap_key]
                std_dev = self.vwap_data[symbol][std_key]
                
                if vwap is not None and std_dev is not None and std_dev > 0:
                    current_price = data['close'].iloc[-1]
                    deviation = (current_price - vwap) / std_dev
                    indicators['vwap_deviation'] = deviation
        
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
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate VWAP
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
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of price data
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (upper band, middle band, lower band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Process each active signal
        for symbol, signal_data in self.active_signals.items():
            signals[symbol] = signal_data['signal']
        
        return signals
    
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
        
        # Calculate position size based on ATR for stop loss
        if signal.stop_loss is not None and signal.price != signal.stop_loss:
            # Risk per share
            risk_per_share = abs(signal.price - signal.stop_loss)
            
            # Position size
            position_size = risk_amount / risk_per_share
        else:
            # Fallback to a percentage of the price
            risk_per_share = signal.price * 0.01  # 1% of price
            position_size = risk_amount / risk_per_share
        
        return position_size

# Import for volume percentile calculation
from scipy.stats import percentileofscore
