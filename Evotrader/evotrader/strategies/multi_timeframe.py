"""
Multi-timeframe strategy implementations for EvoTrader.

This module provides strategies that operate across multiple timeframes,
enabling more sophisticated analysis and trading decisions based on
both short-term and long-term market conditions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import copy

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicator_system import Indicator, IndicatorFactory
from .enhanced_strategy import EnhancedStrategy


class TimeFrame:
    """Enum-like class for timeframes."""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    
    @staticmethod
    def get_ratio(higher_tf: str, lower_tf: str) -> Optional[int]:
        """
        Get the ratio between two timeframes.
        
        Args:
            higher_tf: Higher timeframe
            lower_tf: Lower timeframe
            
        Returns:
            Ratio or None if incompatible
        """
        tf_minutes = {
            TimeFrame.TICK: 1/60,
            TimeFrame.MINUTE_1: 1,
            TimeFrame.MINUTE_5: 5,
            TimeFrame.MINUTE_15: 15,
            TimeFrame.MINUTE_30: 30,
            TimeFrame.HOUR_1: 60,
            TimeFrame.HOUR_4: 240,
            TimeFrame.DAY_1: 1440,
            TimeFrame.WEEK_1: 10080,
            TimeFrame.MONTH_1: 43200
        }
        
        if higher_tf in tf_minutes and lower_tf in tf_minutes:
            h_minutes = tf_minutes[higher_tf]
            l_minutes = tf_minutes[lower_tf]
            
            if h_minutes > l_minutes:
                return int(h_minutes / l_minutes)
        
        return None


class MultiTimeFrameStrategy(EnhancedStrategy):
    """
    Base class for multi-timeframe strategies.
    
    This strategy maintains separate indicators for different timeframes
    and generates signals based on the confluence of timeframes.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        params = super().get_parameters() if hasattr(super(), "get_parameters") else []
        
        # Add multi-timeframe specific parameters
        params.extend([
            StrategyParameter(
                name="primary_timeframe",
                default_value=TimeFrame.HOUR_1,
                is_mutable=False
            ),
            StrategyParameter(
                name="secondary_timeframe",
                default_value=TimeFrame.DAY_1,
                is_mutable=False
            ),
            StrategyParameter(
                name="tertiary_timeframe",
                default_value=TimeFrame.WEEK_1,
                is_mutable=False
            ),
            StrategyParameter(
                name="primary_weight",
                default_value=0.5,
                min_value=0.1,
                max_value=0.9,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="secondary_weight",
                default_value=0.3,
                min_value=0.1,
                max_value=0.9,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="tertiary_weight",
                default_value=0.2,
                min_value=0.1,
                max_value=0.9,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="timeframe_agreement_threshold",
                default_value=0.6,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            )
        ])
        
        return params
    
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the multi-timeframe strategy."""
        super().__init__(strategy_id, parameters)
        
        # Store indicators by timeframe and symbol
        self.timeframe_indicators: Dict[str, Dict[str, Dict[str, Indicator]]] = {
            self.parameters.get("primary_timeframe", TimeFrame.HOUR_1): {},
            self.parameters.get("secondary_timeframe", TimeFrame.DAY_1): {},
            self.parameters.get("tertiary_timeframe", TimeFrame.WEEK_1): {}
        }
        
        # Candle aggregators for higher timeframes
        self.candle_aggregators: Dict[str, Dict[str, 'CandleAggregator']] = {
            tf: {} for tf in self.timeframe_indicators
        }
        
        # Latest higher timeframe data
        self.latest_tf_data: Dict[str, Dict[str, Dict[str, Any]]] = {
            tf: {} for tf in self.timeframe_indicators
        }
    
    def setup_indicators(self, symbol: str) -> None:
        """
        Set up indicators for all timeframes.
        
        Args:
            symbol: Trading symbol
        """
        for timeframe in self.timeframe_indicators:
            if symbol not in self.timeframe_indicators[timeframe]:
                self.timeframe_indicators[timeframe][symbol] = {}
                
            # Setup aggregator for this timeframe and symbol
            if symbol not in self.candle_aggregators[timeframe]:
                self.candle_aggregators[timeframe][symbol] = CandleAggregator(timeframe)
                
            # Implement in subclasses to set up specific indicators for each timeframe
            self.setup_timeframe_indicators(timeframe, symbol)
    
    def setup_timeframe_indicators(self, timeframe: str, symbol: str) -> None:
        """
        Set up indicators for a specific timeframe.
        Override in subclasses.
        
        Args:
            timeframe: Timeframe to setup
            symbol: Trading symbol
        """
        pass
    
    def update_indicators(self, symbol: str, candle: Dict[str, Any]) -> None:
        """
        Update all indicators across all timeframes with new data.
        
        Args:
            symbol: Trading symbol
            candle: Price candle with OHLCV data
        """
        self.ensure_indicators_exist(symbol)
        
        # Update indicators for the primary (lowest) timeframe directly
        primary_tf = self.parameters.get("primary_timeframe", TimeFrame.HOUR_1)
        if symbol in self.timeframe_indicators.get(primary_tf, {}):
            for indicator in self.timeframe_indicators[primary_tf][symbol].values():
                indicator.update(candle)
        
        # Update higher timeframe data using aggregators
        for timeframe, aggregators in self.candle_aggregators.items():
            if timeframe == primary_tf:
                # Skip primary timeframe, it's updated directly
                continue
                
            if symbol in aggregators:
                # Update the aggregator with new data
                higher_tf_candle = aggregators[symbol].update(candle)
                
                if higher_tf_candle:
                    # We have a complete higher timeframe candle
                    # Update all indicators for this timeframe
                    for indicator in self.timeframe_indicators[timeframe][symbol].values():
                        indicator.update(higher_tf_candle)
                    
                    # Store the latest higher timeframe data
                    if symbol not in self.latest_tf_data[timeframe]:
                        self.latest_tf_data[timeframe][symbol] = {}
                        
                    self.latest_tf_data[timeframe][symbol] = higher_tf_candle
    
    def generate_symbol_signals(self, symbol: str, data: Dict[str, Any], current_day: int) -> List[Signal]:
        """
        Generate signals for a specific symbol across all timeframes.
        
        Args:
            symbol: Trading symbol
            data: Market data for the symbol
            current_day: Current simulation day
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Collect signals from each timeframe
        tf_signals = {}
        for timeframe in self.timeframe_indicators:
            tf_signal = self.get_timeframe_signal(timeframe, symbol, data, current_day)
            if tf_signal:
                tf_signals[timeframe] = tf_signal
        
        # If we have signals from multiple timeframes, apply the confluence logic
        if tf_signals:
            final_signal = self.combine_timeframe_signals(symbol, tf_signals, data)
            if final_signal:
                signals.append(final_signal)
        
        return signals
    
    def get_timeframe_signal(
        self, 
        timeframe: str, 
        symbol: str, 
        data: Dict[str, Any], 
        current_day: int
    ) -> Optional[Signal]:
        """
        Get signal for a specific timeframe.
        Override in subclasses.
        
        Args:
            timeframe: Timeframe to check
            symbol: Trading symbol
            data: Market data for the symbol
            current_day: Current simulation day
            
        Returns:
            Signal object or None
        """
        return None
    
    def combine_timeframe_signals(
        self, 
        symbol: str, 
        tf_signals: Dict[str, Signal], 
        data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Combine signals from multiple timeframes into a single signal.
        
        Args:
            symbol: Trading symbol
            tf_signals: Signals from different timeframes
            data: Market data for the symbol
            
        Returns:
            Combined signal or None
        """
        # Get weights for each timeframe
        primary_weight = self.parameters.get("primary_weight", 0.5)
        secondary_weight = self.parameters.get("secondary_weight", 0.3)
        tertiary_weight = self.parameters.get("tertiary_weight", 0.2)
        
        weights = {
            self.parameters.get("primary_timeframe", TimeFrame.HOUR_1): primary_weight,
            self.parameters.get("secondary_timeframe", TimeFrame.DAY_1): secondary_weight,
            self.parameters.get("tertiary_timeframe", TimeFrame.WEEK_1): tertiary_weight
        }
        
        # Calculate weighted signal score
        # Positive for buy, negative for sell, 0 for no signal
        score = 0.0
        total_weight = 0.0
        
        for timeframe, signal in tf_signals.items():
            weight = weights.get(timeframe, 0.0)
            
            if signal.signal_type == SignalType.BUY:
                score += weight * signal.confidence
            elif signal.signal_type == SignalType.SELL:
                score -= weight * signal.confidence
                
            total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            score /= total_weight
            
        # Check if we meet the agreement threshold
        threshold = self.parameters.get("timeframe_agreement_threshold", 0.6)
        
        if abs(score) < threshold:
            return None  # Not enough agreement
            
        # Determine signal type and create a combined signal
        signal_type = SignalType.BUY if score > 0 else SignalType.SELL
        
        # Get the highest confidence timeframe signal of this type
        highest_conf_signal = None
        for tf_signal in tf_signals.values():
            if tf_signal.signal_type == signal_type:
                if not highest_conf_signal or tf_signal.confidence > highest_conf_signal.confidence:
                    highest_conf_signal = tf_signal
        
        if not highest_conf_signal:
            return None
            
        # Create a new signal with the combined information
        combined_signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=abs(score),
            reason=f"Multi-timeframe confluence: {', '.join(tf_signals.keys())}",
            params={
                "timeframe_score": score,
                "price": data.get("price", 0),
                "risk_percent": self.parameters.get("risk_percent", 5.0),
                "primary_timeframe": self.parameters.get("primary_timeframe"),
                "combined_from": list(tf_signals.keys())
            }
        )
        
        # Add additional params from the highest confidence signal
        for key, value in highest_conf_signal.params.items():
            if key not in combined_signal.params:
                combined_signal.params[key] = value
        
        return combined_signal


class CandleAggregator:
    """
    Aggregates lower timeframe candles into higher timeframe candles.
    """
    
    def __init__(self, timeframe: str, base_timeframe: str = TimeFrame.MINUTE_1):
        """
        Initialize candle aggregator.
        
        Args:
            timeframe: Target timeframe
            base_timeframe: Base timeframe for the input candles
        """
        self.timeframe = timeframe
        self.base_timeframe = base_timeframe
        self.bars_needed = TimeFrame.get_ratio(timeframe, base_timeframe) or 1
        
        # Current aggregation state
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: float = 0.0
        self.timestamp: int = 0
        self.bar_count: int = 0
        
        # Previous completed candle
        self.prev_candle: Optional[Dict[str, Any]] = None
    
    def update(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update with a new candle.
        
        Args:
            candle: New candle data
            
        Returns:
            Completed higher timeframe candle or None if not complete
        """
        # Check for required fields
        if not all(field in candle for field in ['open', 'high', 'low', 'close']):
            return None
            
        # For the first candle in the aggregation
        if self.bar_count == 0:
            self.open = candle['open']
            self.high = candle['high']
            self.low = candle['low']
            self.timestamp = candle.get('timestamp', 0)
        else:
            # Update high and low
            if candle['high'] > self.high:
                self.high = candle['high']
            if candle['low'] < self.low:
                self.low = candle['low']
        
        # Always update close with the most recent
        self.close = candle['close']
        
        # Add volume
        self.volume += candle.get('volume', 0)
        
        # Increment bar count
        self.bar_count += 1
        
        # Check if we have a complete higher timeframe candle
        if self.bar_count >= self.bars_needed:
            # Create the completed candle
            completed_candle = {
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume,
                'timestamp': self.timestamp,
                'timeframe': self.timeframe
            }
            
            # Reset for the next aggregation
            self.bar_count = 0
            self.open = None
            self.high = None
            self.low = None
            self.close = None
            self.volume = 0.0
            
            # Save this candle as previous
            self.prev_candle = completed_candle
            
            return completed_candle
        
        return None
    
    def get_latest_complete_candle(self) -> Optional[Dict[str, Any]]:
        """Get the most recently completed higher timeframe candle."""
        return self.prev_candle


class MultiTimeFrameTrendStrategy(MultiTimeFrameStrategy):
    """
    Multi-timeframe trend following strategy.
    
    Uses moving averages and other trend indicators across multiple
    timeframes to identify strong trends and high-probability entry points.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        params = super().get_parameters()
        
        # Add trend-specific parameters
        params.extend([
            StrategyParameter(
                name="fast_ma_period",
                default_value=8,
                min_value=3,
                max_value=20,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="medium_ma_period",
                default_value=21,
                min_value=10,
                max_value=50,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="slow_ma_period",
                default_value=50,
                min_value=20,
                max_value=100,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=2.0,
                min_value=0.5,
                max_value=5.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="atr_multiplier",
                default_value=2.0,
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            )
        ])
        
        return params
    
    def setup_timeframe_indicators(self, timeframe: str, symbol: str) -> None:
        """Set up trend indicators for each timeframe."""
        fast_period = self.parameters["fast_ma_period"]
        medium_period = self.parameters["medium_ma_period"]
        slow_period = self.parameters["slow_ma_period"]
        
        tf_indicators = self.timeframe_indicators[timeframe][symbol]
        
        # Moving averages
        tf_indicators['fast_ma'] = IndicatorFactory.create(
            'ema', symbol, {'period': fast_period}
        )
        
        tf_indicators['medium_ma'] = IndicatorFactory.create(
            'ema', symbol, {'period': medium_period}
        )
        
        tf_indicators['slow_ma'] = IndicatorFactory.create(
            'ema', symbol, {'period': slow_period}
        )
        
        # ATR for volatility
        tf_indicators['atr'] = IndicatorFactory.create(
            'atr', symbol, {'period': 14}
        )
    
    def get_timeframe_signal(
        self, 
        timeframe: str, 
        symbol: str, 
        data: Dict[str, Any], 
        current_day: int
    ) -> Optional[Signal]:
        """Generate signal for a specific timeframe."""
        # Skip if we don't have indicators for this timeframe/symbol
        if (timeframe not in self.timeframe_indicators or 
            symbol not in self.timeframe_indicators[timeframe]):
            return None
            
        tf_indicators = self.timeframe_indicators[timeframe][symbol]
        
        # Check if indicators are ready
        if not all(ind.is_ready for ind in tf_indicators.values()):
            return None
        
        # Get the current values
        fast_ma = tf_indicators['fast_ma'].get_last_value()
        medium_ma = tf_indicators['medium_ma'].get_last_value()
        slow_ma = tf_indicators['slow_ma'].get_last_value()
        atr = tf_indicators['atr'].get_last_value()
        
        # Current price from data
        current_price = data.get('price', 0)
        
        # If any values are None, skip
        if None in [fast_ma, medium_ma, slow_ma, atr, current_price]:
            return None
        
        # Check the ma alignment for trend
        # Bullish: fast > medium > slow
        bullish_alignment = fast_ma > medium_ma > slow_ma
        
        # Bearish: fast < medium < slow
        bearish_alignment = fast_ma < medium_ma < slow_ma
        
        # Check position
        in_position = symbol in self.current_positions
        
        # Generate signals
        signal = None
        
        if bullish_alignment and not in_position:
            # Bullish signal
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.7,
                reason=f"Bullish trend alignment in {timeframe} timeframe",
                params={
                    "risk_percent": self.parameters["risk_percent"],
                    "entry_price": current_price,
                    "stop_loss": current_price - (atr * self.parameters["atr_multiplier"]),
                    "timeframe": timeframe,
                    "fast_ma": fast_ma,
                    "medium_ma": medium_ma,
                    "slow_ma": slow_ma
                }
            )
        elif bearish_alignment and in_position:
            # Bearish signal
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=0.7,
                reason=f"Bearish trend alignment in {timeframe} timeframe",
                params={
                    "risk_percent": self.parameters["risk_percent"],
                    "timeframe": timeframe,
                    "fast_ma": fast_ma,
                    "medium_ma": medium_ma,
                    "slow_ma": slow_ma
                }
            )
        
        return signal


class MultiTimeFrameRSIStrategy(MultiTimeFrameStrategy):
    """
    Multi-timeframe RSI strategy.
    
    Uses RSI across multiple timeframes to identify overbought/oversold
    conditions that are confirmed by higher timeframes.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        params = super().get_parameters()
        
        # Add RSI-specific parameters
        params.extend([
            StrategyParameter(
                name="rsi_period",
                default_value=14,
                min_value=5,
                max_value=30,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="oversold_threshold",
                default_value=30.0,
                min_value=10.0,
                max_value=40.0,
                step=1.0,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="overbought_threshold",
                default_value=70.0,
                min_value=60.0,
                max_value=90.0,
                step=1.0,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=2.0,
                min_value=0.5,
                max_value=5.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            )
        ])
        
        return params
    
    def setup_timeframe_indicators(self, timeframe: str, symbol: str) -> None:
        """Set up RSI indicators for each timeframe."""
        rsi_period = self.parameters["rsi_period"]
        
        tf_indicators = self.timeframe_indicators[timeframe][symbol]
        
        # RSI indicator
        tf_indicators['rsi'] = IndicatorFactory.create(
            'rsi', symbol, {'period': rsi_period}
        )
    
    def get_timeframe_signal(
        self, 
        timeframe: str, 
        symbol: str, 
        data: Dict[str, Any], 
        current_day: int
    ) -> Optional[Signal]:
        """Generate signal for a specific timeframe."""
        # Skip if we don't have indicators for this timeframe/symbol
        if (timeframe not in self.timeframe_indicators or 
            symbol not in self.timeframe_indicators[timeframe]):
            return None
            
        tf_indicators = self.timeframe_indicators[timeframe][symbol]
        
        # Check if indicators are ready
        if 'rsi' not in tf_indicators or not tf_indicators['rsi'].is_ready:
            return None
        
        # Get the current values
        rsi = tf_indicators['rsi'].get_last_value()
        
        # Current price from data
        current_price = data.get('price', 0)
        
        # If any values are None, skip
        if None in [rsi, current_price]:
            return None
        
        # Check RSI conditions
        oversold = rsi < self.parameters["oversold_threshold"]
        overbought = rsi > self.parameters["overbought_threshold"]
        
        # Check position
        in_position = symbol in self.current_positions
        
        # Generate signals
        signal = None
        
        if oversold and not in_position:
            # Oversold signal (potential buy)
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.7,
                reason=f"RSI oversold in {timeframe} timeframe",
                params={
                    "risk_percent": self.parameters["risk_percent"],
                    "entry_price": current_price,
                    "timeframe": timeframe,
                    "rsi": rsi
                }
            )
        elif overbought and in_position:
            # Overbought signal (potential sell)
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=0.7,
                reason=f"RSI overbought in {timeframe} timeframe",
                params={
                    "risk_percent": self.parameters["risk_percent"],
                    "timeframe": timeframe,
                    "rsi": rsi
                }
            )
        
        return signal
