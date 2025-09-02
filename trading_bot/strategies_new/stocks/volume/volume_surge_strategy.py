#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume Surge Trading Strategy

A sophisticated strategy for trading unusual volume spikes and surges.
This strategy identifies significant volume anomalies and trades based on:
- Relative volume comparisons to historical averages
- Volume spikes accompanied by price movements
- Accumulation/distribution analysis
- Volume profile and VWAP analysis
"""

import logging
import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.core.signals import Signal, SignalType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="VolumeSurgeStrategy",
    market_type="stocks",
    description="A strategy that identifies and trades abnormal volume spikes with price confirmation",
    timeframes=["5m", "15m", "1h", "4h", "1d"],
    parameters={
        "volume_surge_threshold": {"description": "Volume threshold as multiple of average volume", "type": "float"},
        "lookback_periods": {"description": "Number of periods to use for volume average calculation", "type": "integer"},
        "trade_mode": {"description": "Trading mode (breakout, reversal, adaptive)", "type": "string"},
        "min_consolidation_bars": {"description": "Minimum number of consolidation bars before valid breakout", "type": "integer"}
    }
)
class VolumeSurgeStrategy(StocksBaseStrategy):
    """
    Volume Surge Trading Strategy
    
    This strategy identifies and trades significant volume anomalies:
    - Volume spikes (sudden large increases in volume)
    - Relative volume analysis (comparison to historical averages)
    - Accumulation/distribution patterns
    - Volume-price divergences
    
    Features:
    - Multi-timeframe volume analysis
    - Adaptive volume thresholds based on market conditions
    - Volume trend detection and classification
    - Integration with price action for confirmation
    - Specialized entry/exit timing based on volume patterns
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Volume Surge Trading Strategy.
        
        Args:
            session: StocksSession for the specific symbol and timeframe
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base strategy
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Volume surge detection parameters
            'volume_surge_threshold': 2.5,      # Multiple of average volume to consider a surge
            'volume_lookback_periods': 20,      # Periods to look back for volume average
            'volume_smoothing_periods': 5,      # Periods for smoothing volume
            'relative_volume_threshold': 2.0,   # Relative volume threshold for signals
            
            # Price action parameters
            'require_price_confirmation': True, # Require price movement to confirm volume
            'min_price_move_percent': 1.0,      # Minimum price movement percentage
            'trend_confirmation_bars': 3,       # Bars required to confirm trend
            
            # Trading parameters
            'max_volume_trades_per_day': 5,     # Maximum volume-based trades per day
            'entry_delay_bars': 1,              # Bars to wait after volume signal before entry
            'max_hold_period_bars': 20,         # Maximum bars to hold a volume-based position
            
            # Risk management
            'max_risk_per_trade_percent': 1.0,  # Max risk per trade as % of account
            'atr_multiplier': 2.0,              # ATR multiplier for stop loss
            'trailing_stop_activation': 1.5,    # Profit multiple to activate trailing stop
            'trailing_stop_distance': 1.0,      # ATR multiplier for trailing stop
            
            # Additional filters
            'min_avg_volume': 500000,           # Minimum average trading volume
            'min_price': 5.0,                   # Minimum price for tradable stocks
            'exclude_earnings_days': True,      # Avoid trading around earnings
            
            # Strategy style parameters
            'strategy_mode': 'adaptive',        # 'breakout', 'reversal', or 'adaptive'
            'volume_divergence_enabled': True,  # Enable volume-price divergence signals
            
            # Advanced volume profile parameters
            'calculate_volume_profile': True,   # Whether to calculate volume profile
            'volume_profile_bins': 20,          # Number of bins for volume profile
            'volume_profile_lookback': 50,      # Bars to include in volume profile
            
            # External data integration
            'earnings_calendar': {},            # Dict of symbols -> upcoming earnings dates
        }
        
        # Update parameters with defaults for any missing keys
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Strategy state
        self.volume_trades_today = 0  # Counter for volume trades
        self.last_trade_date = None  # Track the last trade date to reset counters
        self.volume_signals = {}  # Stores active volume-related signals
        self.volume_profile = None  # Will store volume profile analysis
        self.relative_volume = 1.0  # Current relative volume (ratio to average)
        self.volume_trend = 'neutral'  # 'increasing', 'decreasing', or 'neutral'
        self.last_surge_bar = None  # Index of the last detected volume surge
        
        # Register for market events if event bus is available
        if self.event_bus:
            self.register_for_events(self.event_bus)
        
        logger.info(f"Initialized Volume Surge Strategy for {session.symbol} on {session.timeframe}")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register for relevant market events.
        
        Args:
            event_bus: EventBus to register with
        """
        # First register for common events via base class
        super().register_for_events(event_bus)
        
        # Register for volume-specific events
        event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        event_bus.subscribe(EventType.UNUSUAL_VOLUME, self._on_unusual_volume)
        event_bus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self._on_earnings_announcement)
        
        logger.debug(f"Volume Surge Strategy registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open event.
        
        Reset daily counters and check for pre-market volume anomalies.
        
        Args:
            event: Market open event
        """
        # Reset daily counter if it's a new trading day
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.volume_trades_today = 0
            self.last_trade_date = current_date
            logger.info(f"New trading day {current_date}, reset volume trade counter")
        
        # Check for trade opportunities at market open (gap with volume)
        self._check_for_trade_opportunities()
    
    def _on_unusual_volume(self, event: Event) -> None:
        """
        Handle unusual volume events.
        
        Process unexpected volume surges that may indicate trading opportunities.
        
        Args:
            event: Unusual volume event
        """
        # Check if the event is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Extract volume data
        volume_data = event.data.get('volume_data', {})
        current_volume = volume_data.get('current_volume', 0)
        avg_volume = volume_data.get('average_volume', 0)
        relative_volume = volume_data.get('relative_volume', 1.0)
        
        # Update our relative volume tracking
        self.relative_volume = relative_volume
        
        # Log unusual volume event
        logger.info(f"Unusual volume for {self.session.symbol}: " +
                   f"Current: {current_volume:,}, Avg: {avg_volume:,}, " +
                   f"Relative: {relative_volume:.2f}x")
        
        # Check for trading opportunities on significant volume
        if relative_volume >= self.parameters['relative_volume_threshold']:
            self._check_for_trade_opportunities()
    
    def _on_earnings_announcement(self, event: Event) -> None:
        """
        Handle earnings announcement events.
        
        Manage positions around earnings if configured to avoid them.
        
        Args:
            event: Earnings announcement event
        """
        # Check if we should avoid trading on earnings days
        if not self.parameters['exclude_earnings_days']:
            return
        
        # Check if the event data contains our symbol
        symbols = event.data.get('symbols', [])
        symbol = self.session.symbol
        
        if symbol in symbols:
            # Get days to announcement
            days_to_announcement = event.data.get('days_to_announcement', 0)
            
            # If earnings are today or tomorrow, don't generate new signals
            if days_to_announcement <= 1:
                logger.info(f"Avoiding volume signals for {symbol} due to upcoming earnings")
                
                # Close existing positions if earnings are today
                if days_to_announcement == 0:
                    for position in self.positions:
                        if position.status == PositionStatus.OPEN:
                            self._close_position(position.id)
                            logger.info(f"Closed position {position.id} before earnings announcement")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated event.
        
        Check for volume anomalies and update internal state.
        
        Args:
            event: Market data updated event
        """
        # Let the base class handle common functionality first
        super()._on_market_data_updated(event)
        
        # Check if the event data is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Check for volume anomalies on each update
        if len(self.market_data) > self.parameters['volume_lookback_periods']:
            self._detect_volume_anomalies()
            
        # Check for trading opportunities if we've detected a recent surge
        if self.last_surge_bar is not None and len(self.market_data) - self.last_surge_bar <= 3:
            self._check_for_trade_opportunities()
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed event.
        
        Calculate indicators and check for volume-based signals.
        
        Args:
            event: Timeframe completed event
        """
        # Let the base class handle common functionality first
        super()._on_timeframe_completed(event)
        
        # Check if the event data is for our symbol and timeframe
        if (event.data.get('symbol') != self.session.symbol or 
            event.data.get('timeframe') != self.session.timeframe):
            return
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Update volume profile if enabled
        if self.parameters['calculate_volume_profile']:
            self._calculate_volume_profile()
        
        # Check for volume-based trading opportunities
        self._check_for_trade_opportunities()
    
    def _detect_volume_anomalies(self) -> Dict[str, Any]:
        """
        Detect volume anomalies such as spikes, surges, and unusual patterns.
        
        Returns:
            Dictionary with volume anomaly detection results
        """
        results = {
            'volume_surge': False,       # Whether a volume surge was detected
            'relative_volume': 1.0,      # Current volume relative to average
            'volume_trend': 'neutral',   # Direction of volume trend
            'volume_pattern': None,      # Detected volume pattern if any
            'signal_strength': 0.0       # Strength of volume signal (0-1)
        }
        
        # Check if we have enough data
        lookback = self.parameters['volume_lookback_periods']
        if len(self.market_data) <= lookback:
            return results
        
        # Get current and historical volume
        current_volume = self.market_data['volume'].iloc[-1]
        historical_volume = self.market_data['volume'].iloc[-lookback:-1]
        avg_volume = historical_volume.mean()
        
        # Calculate relative volume
        if avg_volume > 0:
            relative_volume = current_volume / avg_volume
            results['relative_volume'] = relative_volume
            self.relative_volume = relative_volume
        
        # Detect volume surge based on threshold
        surge_threshold = self.parameters['volume_surge_threshold']
        if relative_volume >= surge_threshold:
            results['volume_surge'] = True
            self.last_surge_bar = len(self.market_data) - 1
            
            # Calculate signal strength based on how much the threshold was exceeded
            exceed_factor = (relative_volume - surge_threshold) / (5.0 - surge_threshold)  # Normalize to 0-1
            results['signal_strength'] = min(1.0, max(0.0, exceed_factor))
            
            logger.info(f"Volume surge detected for {self.session.symbol}: {relative_volume:.2f}x average volume")
        
        # Determine volume trend
        smooth_periods = self.parameters['volume_smoothing_periods']
        if len(self.market_data) >= smooth_periods * 2:
            recent_avg = self.market_data['volume'].iloc[-smooth_periods:].mean()
            prev_avg = self.market_data['volume'].iloc[-smooth_periods*2:-smooth_periods].mean()
            
            if recent_avg > prev_avg * 1.2:  # 20% increase
                results['volume_trend'] = 'increasing'
            elif recent_avg < prev_avg * 0.8:  # 20% decrease
                results['volume_trend'] = 'decreasing'
            
            self.volume_trend = results['volume_trend']
        
        # Detect specific volume patterns
        if len(self.market_data) >= 5:
            results['volume_pattern'] = self._identify_volume_pattern()
        
        return results
    
    def _identify_volume_pattern(self) -> Optional[str]:
        """
        Identify specific volume patterns in recent bars.
        
        Returns:
            Name of identified pattern or None
        """
        # Get recent volume and price data
        volume = self.market_data['volume'].iloc[-5:].values
        close = self.market_data['close'].iloc[-5:].values
        open_prices = self.market_data['open'].iloc[-5:].values
        
        # Check for volume climax (very high volume followed by lower volume)
        if volume[-2] > volume[-3:].mean() * 2 and volume[-2] > volume[-1] * 1.5:
            return 'volume_climax'
        
        # Check for stepping pattern (consistently increasing volume)
        if all(volume[i] < volume[i+1] for i in range(len(volume)-2)):
            return 'stepping_volume'
        
        # Check for exhaustion pattern (volume spike with price reversal)
        if volume[-1] > volume[-5:-1].mean() * 2 and \
           ((close[-2] > open_prices[-2] and close[-1] < open_prices[-1]) or  # Bullish to bearish
            (close[-2] < open_prices[-2] and close[-1] > open_prices[-1])):   # Bearish to bullish
            return 'exhaustion'
        
        # Check for churn pattern (high volume with minimal price movement)
        if volume[-1] > volume[-5:-1].mean() * 1.5 and \
           abs(close[-1] - open_prices[-1]) / ((close[-1] + open_prices[-1]) / 2) < 0.003:  # <0.3% move
            return 'churn'
            
        return None
    
    def _calculate_volume_profile(self) -> Dict[str, Any]:
        """
        Calculate the volume profile (volume distribution by price).
        
        Returns:
            Dictionary with volume profile analysis
        """
        if not self.parameters['calculate_volume_profile']:
            return {}
            
        results = {
            'poc_price': None,          # Point of Control price (highest volume price)
            'value_area_high': None,    # Value Area High (70% of volume above this price)
            'value_area_low': None,     # Value Area Low (70% of volume below this price)
            'volume_by_price': {},      # Dictionary of volume grouped by price
            'price_bins': [],           # Bin edges for price ranges
            'binned_volume': []         # Volume in each bin
        }
        
        # Check if we have enough data
        lookback = self.parameters['volume_profile_lookback']
        if len(self.market_data) < lookback:
            return results
            
        # Get recent price and volume data
        recent_data = self.market_data.iloc[-lookback:]
        bins = self.parameters['volume_profile_bins']
        
        # Calculate price bins
        price_min = recent_data['low'].min()
        price_max = recent_data['high'].max()
        bin_width = (price_max - price_min) / bins if price_max > price_min else 1.0
        
        # Create price bins
        price_bins = [price_min + i * bin_width for i in range(bins + 1)]
        results['price_bins'] = price_bins
        
        # Simplified volume profile calculation - in a real implementation
        # this would use intraday data to calculate more precisely
        binned_volume = np.zeros(bins)
        
        for i, row in recent_data.iterrows():
            # Find which bin this bar's volume belongs to using midpoint of bar
            mid_price = (row['high'] + row['low']) / 2
            bin_index = min(bins - 1, max(0, int((mid_price - price_min) / bin_width)))
            binned_volume[bin_index] += row['volume']
            
        results['binned_volume'] = binned_volume.tolist()
        
        # Find Point of Control (price level with highest volume)
        max_volume_bin = binned_volume.argmax()
        poc_price = price_min + (max_volume_bin + 0.5) * bin_width  # Middle of bin
        results['poc_price'] = poc_price
        
        # Find Value Area (70% of total volume)
        total_volume = binned_volume.sum()
        value_area_volume = total_volume * 0.7
        
        # Start from POC and expand outward until we've covered 70% of volume
        current_volume = binned_volume[max_volume_bin]
        lower_bin = max_volume_bin - 1
        upper_bin = max_volume_bin + 1
        
        while current_volume < value_area_volume and (lower_bin >= 0 or upper_bin < bins):
            # Add from lower bin if available and has more volume
            lower_volume = binned_volume[lower_bin] if lower_bin >= 0 else 0
            upper_volume = binned_volume[upper_bin] if upper_bin < bins else 0
            
            if lower_bin >= 0 and (upper_bin >= bins or lower_volume >= upper_volume):
                current_volume += lower_volume
                lower_bin -= 1
            elif upper_bin < bins:
                current_volume += upper_volume
                upper_bin += 1
            else:
                break
        
        # Calculate value area high and low
        value_area_low = price_min + (lower_bin + 1) * bin_width
        value_area_high = price_min + upper_bin * bin_width
        
        results['value_area_low'] = value_area_low
        results['value_area_high'] = value_area_high
        
        # Store volume profile
        self.volume_profile = results
        
        return results
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for volume analysis.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        if len(data) < 20:  # Need at least 20 bars for meaningful indicators
            return {}
            
        indicators = {}
        
        # Calculate volume-weighted average price (VWAP)
        try:
            # Formula: Cumulative(Price * Volume) / Cumulative(Volume)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            cumulative_tp_vol = (typical_price * data['volume']).cumsum()
            cumulative_vol = data['volume'].cumsum()
            indicators['vwap'] = cumulative_tp_vol / cumulative_vol
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
        
        # Calculate relative volume (compared to n-day average)
        try:
            lookback = self.parameters['volume_lookback_periods']
            if len(data) > lookback:
                indicators['avg_volume'] = data['volume'].rolling(window=lookback).mean()
                # For days with data, calculate relative volume
                indicators['relative_volume'] = data['volume'] / indicators['avg_volume']
        except Exception as e:
            logger.error(f"Error calculating relative volume: {e}")
        
        # Calculate On-Balance Volume (OBV)
        try:
            obv = pd.Series(0, index=data.index)
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            indicators['obv'] = obv
            
            # Calculate OBV moving average for trend detection
            if len(data) >= 20:
                indicators['obv_ma'] = obv.rolling(window=20).mean()
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
        
        # Calculate Average True Range (ATR) for volatility assessment
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(window=14).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            
        # Calculate Accumulation/Distribution Line
        try:
            clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            clv = clv.replace([np.inf, -np.inf], 0)  # Replace infinities
            clv.fillna(0, inplace=True)  # Replace NaN
            
            adl = (clv * data['volume']).cumsum()
            indicators['adl'] = adl
            
            # Calculate ADL moving average for trend detection
            if len(data) >= 20:
                indicators['adl_ma'] = adl.rolling(window=20).mean()
        except Exception as e:
            logger.error(f"Error calculating ADL: {e}")
        
        # Calculate Chaikin Money Flow (CMF) - 20 period
        try:
            period = 20
            if len(data) >= period:
                money_flow_multiplier = clv
                money_flow_volume = money_flow_multiplier * data['volume']
                cmf = money_flow_volume.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
                indicators['cmf'] = cmf
        except Exception as e:
            logger.error(f"Error calculating CMF: {e}")
        
        # Calculate Volume Oscillator (VO) - percentage difference between fast and slow volume MAs
        try:
            fast_period = 5
            slow_period = 20
            
            if len(data) >= slow_period:
                fast_vol_ma = data['volume'].rolling(window=fast_period).mean()
                slow_vol_ma = data['volume'].rolling(window=slow_period).mean()
                
                volume_oscillator = ((fast_vol_ma - slow_vol_ma) / slow_vol_ma) * 100
                indicators['volume_oscillator'] = volume_oscillator
        except Exception as e:
            logger.error(f"Error calculating Volume Oscillator: {e}")
        
        return indicators
    
    def _check_for_trade_opportunities(self) -> None:
        """
        Check for volume-based trading opportunities and generate signals.
        """
        # Check trade limits
        if self.volume_trades_today >= self.parameters['max_volume_trades_per_day']:
            logger.info(f"Maximum volume trades reached for today: {self.volume_trades_today}")
            return
        
        # Make sure we have market data
        if len(self.market_data) < 20:  # Need sufficient data for analysis
            return
        
        # Detect volume anomalies
        volume_analysis = self._detect_volume_anomalies()
        relative_volume = volume_analysis['relative_volume']
        
        # Skip if volume is not unusual enough
        if relative_volume < self.parameters['relative_volume_threshold']:
            return
        
        # Get current price and market data
        current_price = self.market_data['close'].iloc[-1]
        current_volume = self.market_data['volume'].iloc[-1]
        
        # Check if stock meets basic requirements
        if current_price < self.parameters['min_price']:
            logger.debug(f"Price below minimum: {current_price:.2f} < {self.parameters['min_price']:.2f}")
            return
        
        # Skip if average volume is too low
        avg_volume = self.market_data['volume'].iloc[-self.parameters['volume_lookback_periods']:].mean()
        if avg_volume < self.parameters['min_avg_volume']:
            logger.debug(f"Average volume below minimum: {avg_volume:.0f} < {self.parameters['min_avg_volume']:.0f}")
            return
            
        # Get indicators for analysis
        if not hasattr(self, 'indicators') or not self.indicators:
            self.indicators = self.calculate_indicators(self.market_data)
        
        # Determine trade direction based on strategy mode and volume patterns
        direction = self._determine_trade_direction(volume_analysis)
        if direction is None:
            return  # No clear direction determined
        
        # Check for price confirmation if required
        if self.parameters['require_price_confirmation']:
            if not self._confirm_price_action(direction):
                logger.debug(f"Price action does not confirm {direction} signal")
                return
        
        # Generate signal if all conditions are met
        signal_id = str(uuid.uuid4())
        entry_price = current_price
        
        # Calculate target and stop loss
        atr = self.indicators.get('atr', None)
        if atr is not None and len(atr) > 0:
            atr_value = atr.iloc[-1]
        else:
            # Fallback if ATR not available
            atr_value = current_price * 0.02  # Default to 2% of price
        
        # Set stop loss based on direction and ATR
        stop_loss = None
        target_price = None
        risk_reward_ratio = 1.5  # Target should be 1.5x the risk
        
        if direction == 'long':
            stop_loss = entry_price - (atr_value * self.parameters['atr_multiplier'])
            target_price = entry_price + (atr_value * self.parameters['atr_multiplier'] * risk_reward_ratio)
        else:  # short
            stop_loss = entry_price + (atr_value * self.parameters['atr_multiplier'])
            target_price = entry_price - (atr_value * self.parameters['atr_multiplier'] * risk_reward_ratio)
        
        # Create signal object
        signal_type = SignalType.LONG if direction == 'long' else SignalType.SHORT
        signal = Signal(
            id=signal_id,
            symbol=self.session.symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=datetime.now(),
            expiration=datetime.now() + timedelta(days=1),
            confidence=min(0.9, 0.5 + volume_analysis['signal_strength']),
            metadata={
                'strategy': 'volume_surge',
                'relative_volume': relative_volume,
                'strategy_mode': self.parameters['strategy_mode'],
                'volume_pattern': volume_analysis['volume_pattern'],
            }
        )
        
        # Log signal generation
        logger.info(f"Generated volume-based {direction} signal for {self.session.symbol} " +
                   f"at {entry_price:.2f} with stop at {stop_loss:.2f}")
        
        # Act on the signal - in real implementation this would go through proper channels
        self._act_on_signal(signal)
        
        # Increment trade counter
        self.volume_trades_today += 1
    
    def _determine_trade_direction(self, volume_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Determine trade direction based on volume analysis and price action.
        
        Args:
            volume_analysis: Results from volume anomaly detection
            
        Returns:
            'long', 'short', or None if no clear direction
        """
        # Get current market data and indicators
        if len(self.market_data) < 3:
            return None
        
        # Get current price action
        current_close = self.market_data['close'].iloc[-1]
        current_open = self.market_data['open'].iloc[-1]
        prev_close = self.market_data['close'].iloc[-2]
        
        # Check the strategy mode
        mode = self.parameters['strategy_mode']
        
        # Determine base direction from price action
        base_direction = None
        if current_close > prev_close and current_close > current_open:  # Bullish
            base_direction = 'long'
        elif current_close < prev_close and current_close < current_open:  # Bearish
            base_direction = 'short'
        
        # No clear direction in price
        if base_direction is None:
            return None
        
        # Check for specific volume patterns that might override the direction
        volume_pattern = volume_analysis.get('volume_pattern')
        if volume_pattern == 'exhaustion':
            # Exhaustion can indicate reversal - go against the price direction
            if self.parameters['strategy_mode'] == 'reversal':
                return 'short' if base_direction == 'long' else 'long'
        
        # Check indicators for confirmation
        if 'obv' in self.indicators and 'obv_ma' in self.indicators:
            obv = self.indicators['obv'].iloc[-1]
            obv_ma = self.indicators['obv_ma'].iloc[-1]
            
            # OBV trending up suggests accumulation (bullish)
            if obv > obv_ma and base_direction == 'long':
                return 'long'
            # OBV trending down suggests distribution (bearish)
            elif obv < obv_ma and base_direction == 'short':
                return 'short'
        
        # Check for Chaikin Money Flow (CMF) as additional confirmation
        if 'cmf' in self.indicators:
            cmf = self.indicators['cmf'].iloc[-1]
            
            # Positive CMF suggests buying pressure
            if cmf > 0.05 and base_direction == 'long':
                return 'long'
            # Negative CMF suggests selling pressure
            elif cmf < -0.05 and base_direction == 'short':
                return 'short'
        
        # Apply strategy mode logic
        if mode == 'breakout':
            # For breakout mode, go with the direction of the price movement
            # when volume confirms
            return base_direction
            
        elif mode == 'reversal':
            # For reversal mode, look for signs of exhaustion and go against
            # the prevailing trend if there are reversal signals
            if volume_pattern in ['volume_climax', 'exhaustion']:
                return 'short' if base_direction == 'long' else 'long'
            
        elif mode == 'adaptive':
            # Adaptive mode tries to select the most appropriate approach
            # based on market conditions
            
            # Check for potential reversal signals
            rsi = None
            try:
                # Calculate RSI if not already in indicators
                if 'rsi' not in self.indicators and len(self.market_data) >= 14:
                    delta = self.market_data['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    self.indicators['rsi'] = rsi
                elif 'rsi' in self.indicators:
                    rsi = self.indicators['rsi']
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")
            
            # Check for overbought/oversold conditions as reversal signals
            if rsi is not None and len(rsi) > 0:
                current_rsi = rsi.iloc[-1]
                
                # Oversold with high volume could indicate bullish reversal
                if current_rsi < 30 and base_direction == 'short':
                    return 'long'
                # Overbought with high volume could indicate bearish reversal
                elif current_rsi > 70 and base_direction == 'long':
                    return 'short'
            
            # Default to the base direction if we can't find reversal signals
            return base_direction
            
        # Default case
        return base_direction
    
    def _confirm_price_action(self, direction: str) -> bool:
        """
        Confirm the trade direction with price action analysis.
        
        Args:
            direction: Proposed trade direction ('long' or 'short')
            
        Returns:
            True if price action confirms direction, False otherwise
        """
        # Make sure we have enough data
        if len(self.market_data) < 3:
            return False
        
        # Get price data
        current_close = self.market_data['close'].iloc[-1]
        current_open = self.market_data['open'].iloc[-1]
        prev_close = self.market_data['close'].iloc[-2]
        
        # Calculate price movement
        price_move_percent = abs((current_close - prev_close) / prev_close) * 100
        min_price_move = self.parameters['min_price_move_percent']
        
        # Check if price movement is significant enough
        if price_move_percent < min_price_move:
            logger.debug(f"Price movement too small: {price_move_percent:.2f}% < {min_price_move:.2f}%")
            return False
        
        # Check trend confirmation based on direction
        trend_confirmed = False
        
        if direction == 'long':
            # For long, confirm with positive price movement
            trend_confirmed = current_close > prev_close
            
            # For stronger confirmation, check close above VWAP if available
            if 'vwap' in self.indicators:
                current_vwap = self.indicators['vwap'].iloc[-1]
                trend_confirmed = trend_confirmed and current_close > current_vwap
                
        else:  # direction == 'short'
            # For short, confirm with negative price movement
            trend_confirmed = current_close < prev_close
            
            # For stronger confirmation, check close below VWAP if available
            if 'vwap' in self.indicators:
                current_vwap = self.indicators['vwap'].iloc[-1]
                trend_confirmed = trend_confirmed and current_close < current_vwap
        
        return trend_confirmed
    
    def _act_on_signal(self, signal: Signal) -> None:
        """
        Act on a generated signal.
        
        Args:
            signal: Signal object with trade details
        """
        # Skip if we're already in a position for this symbol
        for position in self.positions:
            if position.symbol == signal.symbol and position.status == PositionStatus.OPEN:
                logger.info(f"Already in position for {signal.symbol}, skipping signal")
                return
        
        # Calculate position size based on risk parameters
        account_balance = 100000.0  # Example, in real implementation this would be retrieved
        position_size = self._calculate_position_size(signal, account_balance)
        
        # Create a new position
        position_id = str(uuid.uuid4())
        position = Position(
            id=position_id,
            symbol=signal.symbol,
            direction='long' if signal.signal_type == SignalType.LONG else 'short',
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            size=position_size,
            entry_time=datetime.now(),
            status=PositionStatus.PENDING,
            metadata={
                'strategy': 'volume_surge',
                'signal_id': signal.id,
                'relative_volume': signal.metadata.get('relative_volume', 0),
                'volume_pattern': signal.metadata.get('volume_pattern', None)
            }
        )
        
        # In a real implementation, this would send the order to a broker
        # and the position would be updated once the order is filled
        logger.info(f"Opening {position.direction} position for {position.symbol} " +
                   f"at {position.entry_price:.2f} with size {position.size:.2f}")
        
        # Add position to our tracker
        self.positions.append(position)
        
        # Update position status to OPEN (in real implementation this would happen after fill)
        position.status = PositionStatus.OPEN
        
        # Emit an event for position opened if event bus is available
        if self.event_bus:
            event = Event(
                event_type=EventType.POSITION_OPENED,
                timestamp=datetime.now(),
                data={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'strategy': 'volume_surge'
                }
            )
            self.event_bus.emit(event)
    
    def _calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Signal with entry and stop loss prices
            account_balance: Current account balance
            
        Returns:
            Position size in shares
        """
        # Calculate risk amount based on max risk per trade
        max_risk_percent = self.parameters['max_risk_per_trade_percent'] / 100.0
        risk_amount = account_balance * max_risk_percent
        
        # Calculate risk per share
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        
        if stop_loss is None:
            # Fallback if stop loss not provided
            if signal.signal_type == SignalType.LONG:
                stop_loss = entry_price * 0.95  # 5% below entry
            else:
                stop_loss = entry_price * 1.05  # 5% above entry
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Ensure minimum risk per share to avoid division by zero
        if risk_per_share < 0.01 or risk_per_share < entry_price * 0.005:
            risk_per_share = max(0.01, entry_price * 0.005)  # At least 0.5% of price
        
        # Calculate number of shares based on risk
        shares = risk_amount / risk_per_share
        
        # Round to nearest lot size (typically 100 for stocks)
        lot_size = self.session.lot_size if hasattr(self.session, 'lot_size') else 100
        shares = max(lot_size, round(shares / lot_size) * lot_size)
        
        # Limit to max position size (example: 20% of account)
        max_position_value = account_balance * 0.20
        max_shares = max_position_value / entry_price
        shares = min(shares, max_shares)
        
        # Ensure minimum position size
        min_shares = lot_size
        shares = max(min_shares, shares)
        
        return shares
    
    def _close_position(self, position_id: str) -> None:
        """
        Close an open position.
        
        Args:
            position_id: ID of the position to close
        """
        # Find the position in our tracker
        position = None
        for p in self.positions:
            if p.id == position_id:
                position = p
                break
        
        if position is None or position.status != PositionStatus.OPEN:
            logger.warning(f"Cannot close position {position_id}: not found or not open")
            return
        
        # Get current price (simulated - in real implementation would get from market)
        current_price = self.market_data['close'].iloc[-1] if len(self.market_data) > 0 else position.entry_price
        
        # Calculate P&L
        if position.direction == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - current_price) * position.size
        
        # Update position status
        position.status = PositionStatus.CLOSED
        position.exit_price = current_price
        position.exit_time = datetime.now()
        position.pnl = pnl
        
        logger.info(f"Closed {position.direction} position for {position.symbol} " +
                   f"at {position.exit_price:.2f} with P&L {position.pnl:.2f}")
        
        # Emit an event for position closed if event bus is available
        if self.event_bus:
            event = Event(
                event_type=EventType.POSITION_CLOSED,
                timestamp=datetime.now(),
                data={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'pnl': position.pnl,
                    'strategy': 'volume_surge'
                }
            )
            self.event_bus.emit(event)
    
    def check_exit_conditions(self) -> None:
        """
        Check exit conditions for open positions.
        Should be called on each price update.
        """
        if not self.positions or len(self.market_data) == 0:
            return
        
        current_price = self.market_data['close'].iloc[-1]
        current_time = datetime.now()
        
        # Check each open position
        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue
            
            # Skip if not for our symbol
            if position.symbol != self.session.symbol:
                continue
            
            # Check stop loss
            if position.direction == 'long' and current_price <= position.stop_loss:
                logger.info(f"Stop loss triggered for long position {position.id}")
                self._close_position(position.id)
                continue
            
            if position.direction == 'short' and current_price >= position.stop_loss:
                logger.info(f"Stop loss triggered for short position {position.id}")
                self._close_position(position.id)
                continue
            
            # Check target price
            if position.direction == 'long' and current_price >= position.target_price:
                logger.info(f"Target reached for long position {position.id}")
                self._close_position(position.id)
                continue
            
            if position.direction == 'short' and current_price <= position.target_price:
                logger.info(f"Target reached for short position {position.id}")
                self._close_position(position.id)
                continue
            
            # Check max hold time
            entry_time = position.entry_time
            max_hold_bars = self.parameters['max_hold_period_bars']
            
            # Approximate bar duration based on timeframe
            bar_duration = {
                TimeFrame.MINUTE_1: timedelta(minutes=1),
                TimeFrame.MINUTE_5: timedelta(minutes=5),
                TimeFrame.MINUTE_15: timedelta(minutes=15),
                TimeFrame.HOUR_1: timedelta(hours=1),
                TimeFrame.DAY_1: timedelta(days=1),
            }.get(self.session.timeframe, timedelta(days=1))
            
            max_hold_time = entry_time + (bar_duration * max_hold_bars)
            
            if current_time > max_hold_time:
                logger.info(f"Max hold time reached for position {position.id}")
                self._close_position(position.id)
                continue
            
            # Check for trailing stop if activated
            if 'highest_price' not in position.metadata:
                position.metadata['highest_price'] = position.entry_price
                position.metadata['lowest_price'] = position.entry_price
            
            if position.direction == 'long':
                # Update highest price seen for long positions
                if current_price > position.metadata['highest_price']:
                    position.metadata['highest_price'] = current_price
                    
                    # Check if we need to activate trailing stop
                    price_move = position.metadata['highest_price'] - position.entry_price
                    initial_risk = position.entry_price - position.stop_loss
                    
                    if price_move >= initial_risk * self.parameters['trailing_stop_activation']:
                        # Calculate new stop loss based on trailing distance
                        trailing_distance = self.indicators['atr'].iloc[-1] * self.parameters['trailing_stop_distance'] \
                                            if 'atr' in self.indicators and len(self.indicators['atr']) > 0 \
                                            else position.entry_price * 0.02
                        
                        new_stop = position.metadata['highest_price'] - trailing_distance
                        
                        # Update stop loss if it's higher than current one
                        if new_stop > position.stop_loss:
                            old_stop = position.stop_loss
                            position.stop_loss = new_stop
                            logger.info(f"Updated stop loss for position {position.id} from {old_stop:.2f} to {new_stop:.2f}")
            else:  # short position
                # Update lowest price seen for short positions
                if current_price < position.metadata['lowest_price']:
                    position.metadata['lowest_price'] = current_price
                    
                    # Check if we need to activate trailing stop
                    price_move = position.entry_price - position.metadata['lowest_price']
                    initial_risk = position.stop_loss - position.entry_price
                    
                    if price_move >= initial_risk * self.parameters['trailing_stop_activation']:
                        # Calculate new stop loss based on trailing distance
                        trailing_distance = self.indicators['atr'].iloc[-1] * self.parameters['trailing_stop_distance'] \
                                            if 'atr' in self.indicators and len(self.indicators['atr']) > 0 \
                                            else position.entry_price * 0.02
                        
                        new_stop = position.metadata['lowest_price'] + trailing_distance
                        
                        # Update stop loss if it's lower than current one
                        if new_stop < position.stop_loss:
                            old_stop = position.stop_loss
                            position.stop_loss = new_stop
                            logger.info(f"Updated stop loss for position {position.id} from {old_stop:.2f} to {new_stop:.2f}")
