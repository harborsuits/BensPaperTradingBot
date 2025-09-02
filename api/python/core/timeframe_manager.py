#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timeframe Manager

This module provides functionality for managing signals across multiple timeframes,
implementing higher timeframe confirmation of lower timeframe signals, and
correlating patterns and signals across different timeframes.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from enum import Enum

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.strategies.external_signal_strategy import (
    ExternalSignal, SignalSource, SignalType, Direction
)

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Standard timeframes used in trading."""
    M1 = "1m"      # 1 minute
    M5 = "5m"      # 5 minutes
    M15 = "15m"    # 15 minutes
    M30 = "30m"    # 30 minutes
    H1 = "1h"      # 1 hour
    H4 = "4h"      # 4 hours
    D1 = "1d"      # 1 day
    W1 = "1w"      # 1 week
    MN1 = "1M"     # 1 month
    
    @classmethod
    def from_string(cls, timeframe_str: str) -> 'Timeframe':
        """
        Convert a string timeframe to Timeframe enum.
        
        Args:
            timeframe_str: String representation of timeframe
            
        Returns:
            Timeframe enum value
        """
        normalized = timeframe_str.lower().replace(" ", "")
        
        # Try direct mapping
        for tf in cls:
            if tf.value.lower() == normalized:
                return tf
        
        # Try alternative formats
        mappings = {
            "1": cls.M1,
            "5": cls.M5,
            "15": cls.M15,
            "30": cls.M30,
            "60": cls.H1,
            "240": cls.H4,
            "1440": cls.D1,
            "10080": cls.W1,
            "43200": cls.MN1,
            "1min": cls.M1,
            "5min": cls.M5,
            "15min": cls.M15,
            "30min": cls.M30,
            "1hour": cls.H1,
            "4hour": cls.H4,
            "1day": cls.D1,
            "1week": cls.W1,
            "1month": cls.MN1,
        }
        
        if normalized in mappings:
            return mappings[normalized]
        
        # Default to H1 if not recognized
        logger.warning(f"Unrecognized timeframe string: {timeframe_str}, defaulting to H1")
        return cls.H1
    
    def to_minutes(self) -> int:
        """
        Convert timeframe to minutes.
        
        Returns:
            Number of minutes in the timeframe
        """
        mapping = {
            self.M1: 1,
            self.M5: 5,
            self.M15: 15,
            self.M30: 30,
            self.H1: 60,
            self.H4: 240,
            self.D1: 1440,
            self.W1: 10080,
            self.MN1: 43200
        }
        return mapping[self]
    
    def is_higher_than(self, other: 'Timeframe') -> bool:
        """
        Check if this timeframe is higher than another.
        
        Args:
            other: Timeframe to compare with
            
        Returns:
            True if this timeframe is higher than the other
        """
        return self.to_minutes() > other.to_minutes()
    
    def is_lower_than(self, other: 'Timeframe') -> bool:
        """
        Check if this timeframe is lower than another.
        
        Args:
            other: Timeframe to compare with
            
        Returns:
            True if this timeframe is lower than the other
        """
        return self.to_minutes() < other.to_minutes()
    
    def get_higher_timeframes(self) -> List['Timeframe']:
        """
        Get all timeframes higher than this one.
        
        Returns:
            List of higher timeframes
        """
        return [tf for tf in Timeframe if tf.to_minutes() > self.to_minutes()]
    
    def get_lower_timeframes(self) -> List['Timeframe']:
        """
        Get all timeframes lower than this one.
        
        Returns:
            List of lower timeframes
        """
        return [tf for tf in Timeframe if tf.to_minutes() < self.to_minutes()]
    
    def get_next_higher(self) -> Optional['Timeframe']:
        """
        Get the next higher timeframe.
        
        Returns:
            Next higher timeframe or None if this is the highest
        """
        higher_tfs = self.get_higher_timeframes()
        if not higher_tfs:
            return None
        
        return min(higher_tfs, key=lambda tf: tf.to_minutes())
    
    def get_next_lower(self) -> Optional['Timeframe']:
        """
        Get the next lower timeframe.
        
        Returns:
            Next lower timeframe or None if this is the lowest
        """
        lower_tfs = self.get_lower_timeframes()
        if not lower_tfs:
            return None
        
        return max(lower_tfs, key=lambda tf: tf.to_minutes())


class TimeframeAwareSignal:
    """
    Extension of ExternalSignal that is aware of timeframes.
    
    This class wraps an ExternalSignal and adds timeframe awareness,
    allowing signals to be processed in the context of multiple timeframes.
    """
    
    def __init__(
        self,
        signal: ExternalSignal,
        timeframe: Union[str, Timeframe],
        primary_timeframe: bool = False,
        higher_tf_confirmations: Optional[List[str]] = None,
        lower_tf_confirmations: Optional[List[str]] = None
    ):
        """
        Initialize a timeframe-aware signal.
        
        Args:
            signal: The external signal
            timeframe: Timeframe of the signal
            primary_timeframe: Whether this is the primary timeframe for trading
            higher_tf_confirmations: List of higher timeframe confirmations
            lower_tf_confirmations: List of lower timeframe confirmations
        """
        self.signal = signal
        
        # Convert timeframe to Timeframe enum if needed
        if isinstance(timeframe, str):
            self.timeframe = Timeframe.from_string(timeframe)
        else:
            self.timeframe = timeframe
        
        self.primary_timeframe = primary_timeframe
        self.higher_tf_confirmations = higher_tf_confirmations or []
        self.lower_tf_confirmations = lower_tf_confirmations or []
        
        # Metadata for tracking confirmations
        self.confirmed_by = set()
        self.confirmation_status = {}
    
    @property
    def symbol(self) -> str:
        """Get the signal symbol."""
        return self.signal.symbol
    
    @property
    def direction(self) -> Direction:
        """Get the signal direction."""
        return self.signal.direction
    
    @property
    def source(self) -> SignalSource:
        """Get the signal source."""
        return self.signal.source
    
    @property
    def signal_type(self) -> SignalType:
        """Get the signal type."""
        return self.signal.signal_type
    
    @property
    def timestamp(self) -> datetime:
        """Get the signal timestamp."""
        return self.signal.timestamp
    
    @property
    def price(self) -> Optional[float]:
        """Get the signal price."""
        return self.signal.price
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the signal metadata."""
        return self.signal.metadata
    
    def is_confirmed_by_timeframe(self, timeframe: Union[str, Timeframe]) -> bool:
        """
        Check if the signal is confirmed by a specific timeframe.
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            True if confirmed by the timeframe
        """
        if isinstance(timeframe, str):
            tf = Timeframe.from_string(timeframe)
        else:
            tf = timeframe
        
        return str(tf.value) in self.confirmed_by
    
    def add_confirmation(self, timeframe: Union[str, Timeframe], status: bool = True) -> None:
        """
        Add a timeframe confirmation.
        
        Args:
            timeframe: Confirming timeframe
            status: Confirmation status (True for confirmed, False for rejected)
        """
        if isinstance(timeframe, str):
            tf = Timeframe.from_string(timeframe)
        else:
            tf = timeframe
        
        if status:
            self.confirmed_by.add(str(tf.value))
        
        self.confirmation_status[str(tf.value)] = status
    
    def get_confirmation_count(self) -> int:
        """
        Get the number of timeframes that confirmed this signal.
        
        Returns:
            Confirmation count
        """
        return len(self.confirmed_by)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "signal": self.signal.to_dict(),
            "timeframe": self.timeframe.value,
            "primary_timeframe": self.primary_timeframe,
            "higher_tf_confirmations": self.higher_tf_confirmations,
            "lower_tf_confirmations": self.lower_tf_confirmations,
            "confirmed_by": list(self.confirmed_by),
            "confirmation_status": self.confirmation_status
        }
    
    def __str__(self) -> str:
        """String representation."""
        confirmation_str = f", confirmed by {len(self.confirmed_by)} timeframes" if self.confirmed_by else ""
        primary_str = ", primary" if self.primary_timeframe else ""
        return f"{self.symbol} {self.signal_type.value} {self.direction.value} on {self.timeframe.value}{primary_str}{confirmation_str}"


class TimeframeManager:
    """
    Manages signals across multiple timeframes.
    
    This class is responsible for:
    1. Tracking signals across different timeframes
    2. Implementing higher timeframe confirmation
    3. Correlating signals across timeframes
    4. Generating combined signals with multi-timeframe context
    """
    
    def __init__(
        self,
        primary_timeframe: Union[str, Timeframe] = Timeframe.H1,
        confirmation_timeframes: Optional[List[Union[str, Timeframe]]] = None,
        confirmation_required: bool = True,
        min_confirmations: int = 1,
        symbol_timeframe_config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the timeframe manager.
        
        Args:
            primary_timeframe: Primary trading timeframe
            confirmation_timeframes: Timeframes used for confirmation
            confirmation_required: Whether confirmation is required for signals
            min_confirmations: Minimum number of timeframe confirmations required
            symbol_timeframe_config: Configuration for specific symbols
        """
        # Convert primary timeframe to Timeframe enum if needed
        if isinstance(primary_timeframe, str):
            self.primary_timeframe = Timeframe.from_string(primary_timeframe)
        else:
            self.primary_timeframe = primary_timeframe
        
        # Initialize confirmation timeframes
        self.confirmation_timeframes = []
        if confirmation_timeframes:
            for tf in confirmation_timeframes:
                if isinstance(tf, str):
                    self.confirmation_timeframes.append(Timeframe.from_string(tf))
                else:
                    self.confirmation_timeframes.append(tf)
        else:
            # Default to higher and lower timeframes
            higher_tf = self.primary_timeframe.get_next_higher()
            if higher_tf:
                self.confirmation_timeframes.append(higher_tf)
            
            lower_tf = self.primary_timeframe.get_next_lower()
            if lower_tf:
                self.confirmation_timeframes.append(lower_tf)
        
        self.confirmation_required = confirmation_required
        self.min_confirmations = min_confirmations
        self.symbol_timeframe_config = symbol_timeframe_config or {}
        
        # Storage for signals by timeframe
        self.signals_by_timeframe: Dict[str, Dict[str, List[TimeframeAwareSignal]]] = {}
        for tf in [self.primary_timeframe] + self.confirmation_timeframes:
            self.signals_by_timeframe[tf.value] = {}
        
        # Storage for signals by symbol
        self.signals_by_symbol: Dict[str, Dict[str, List[TimeframeAwareSignal]]] = {}
        
        # Get event bus
        self.event_bus = EventBus()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"TimeframeManager initialized with primary timeframe {self.primary_timeframe.value}")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # Listen for external signals
        self.event_bus.subscribe(
            EventType.EXTERNAL_SIGNAL,
            self._handle_external_signal
        )
        
        # Listen for pattern detection
        self.event_bus.subscribe(
            EventType.PATTERN_DETECTED,
            self._handle_pattern_detected
        )
    
    def _handle_external_signal(self, event: Event) -> None:
        """
        Handle an external signal event.
        
        Args:
            event: External signal event
        """
        signal_data = event.data.get("signal", {})
        if not signal_data:
            return
        
        # Extract signal information
        symbol = signal_data.get("symbol")
        if not symbol:
            return
        
        # Extract timeframe from metadata or default to primary
        timeframe_str = signal_data.get("timeframe")
        if timeframe_str:
            timeframe = Timeframe.from_string(timeframe_str)
        else:
            timeframe = self.primary_timeframe
        
        # Check if we have the signal object or need to create one
        signal = event.data.get("signal_object")
        if not signal:
            try:
                signal = ExternalSignal.from_dict(signal_data)
            except Exception as e:
                logger.error(f"Error creating external signal from data: {str(e)}")
                return
        
        # Create timeframe-aware signal
        tf_signal = TimeframeAwareSignal(
            signal=signal,
            timeframe=timeframe,
            primary_timeframe=(timeframe == self.primary_timeframe)
        )
        
        # Process the signal
        self._process_signal(tf_signal)
    
    def _handle_pattern_detected(self, event: Event) -> None:
        """
        Handle a pattern detected event.
        
        Args:
            event: Pattern detected event
        """
        pattern_data = event.data
        if not pattern_data:
            return
        
        # Extract pattern information
        symbol = pattern_data.get("symbol")
        if not symbol:
            return
        
        # Extract timeframe or default to primary
        timeframe_str = pattern_data.get("timeframe")
        if timeframe_str:
            timeframe = Timeframe.from_string(timeframe_str)
        else:
            timeframe = self.primary_timeframe
        
        # Create a signal from the pattern
        pattern_type = pattern_data.get("pattern_type", "unknown")
        direction_str = pattern_data.get("direction")
        direction = Direction.LONG if direction_str == "long" else Direction.SHORT if direction_str == "short" else Direction.UNKNOWN
        
        signal = ExternalSignal(
            symbol=symbol,
            source=SignalSource.CUSTOM_SCRIPT,  # Patterns are internal
            signal_type=SignalType.ENTRY,  # Patterns typically generate entry signals
            direction=direction,
            timestamp=datetime.now(),
            price=pattern_data.get("price"),
            metadata=pattern_data
        )
        
        # Create timeframe-aware signal
        tf_signal = TimeframeAwareSignal(
            signal=signal,
            timeframe=timeframe,
            primary_timeframe=(timeframe == self.primary_timeframe)
        )
        
        # Process the pattern signal
        self._process_signal(tf_signal)
    
    def _process_signal(self, tf_signal: TimeframeAwareSignal) -> None:
        """
        Process a timeframe-aware signal.
        
        Args:
            tf_signal: Timeframe-aware signal
        """
        symbol = tf_signal.symbol
        timeframe_value = tf_signal.timeframe.value
        
        # Store by timeframe
        if timeframe_value not in self.signals_by_timeframe:
            self.signals_by_timeframe[timeframe_value] = {}
        
        if symbol not in self.signals_by_timeframe[timeframe_value]:
            self.signals_by_timeframe[timeframe_value][symbol] = []
        
        self.signals_by_timeframe[timeframe_value][symbol].append(tf_signal)
        
        # Store by symbol
        if symbol not in self.signals_by_symbol:
            self.signals_by_symbol[symbol] = {}
        
        if timeframe_value not in self.signals_by_symbol[symbol]:
            self.signals_by_symbol[symbol][timeframe_value] = []
        
        self.signals_by_symbol[symbol][timeframe_value].append(tf_signal)
        
        logger.info(f"Processed signal: {tf_signal}")
        
        # Check for confirmations
        self._check_timeframe_confirmations(tf_signal)
    
    def _check_timeframe_confirmations(self, tf_signal: TimeframeAwareSignal) -> None:
        """
        Check for confirmations from other timeframes.
        
        Args:
            tf_signal: Signal to check for confirmations
        """
        symbol = tf_signal.symbol
        
        # Skip if the symbol isn't in our signals
        if symbol not in self.signals_by_symbol:
            return
        
        # For primary timeframe signals, check confirmation timeframes
        if tf_signal.primary_timeframe:
            for conf_tf in self.confirmation_timeframes:
                tf_value = conf_tf.value
                if tf_value in self.signals_by_symbol[symbol]:
                    # Get recent signals from this timeframe
                    recent_signals = self._get_recent_signals(
                        self.signals_by_symbol[symbol][tf_value],
                        max_age_minutes=conf_tf.to_minutes() * 2
                    )
                    
                    # Check for confirming signals (same direction)
                    for recent_signal in recent_signals:
                        if recent_signal.direction == tf_signal.direction:
                            tf_signal.add_confirmation(conf_tf)
                            recent_signal.add_confirmation(tf_signal.timeframe)
                            logger.info(f"Signal {tf_signal} confirmed by {conf_tf.value}")
                            break
        
        # For confirmations, check if they confirm primary timeframe signals
        else:
            primary_tf = self.primary_timeframe.value
            if primary_tf in self.signals_by_symbol[symbol]:
                # Get recent signals from primary timeframe
                recent_signals = self._get_recent_signals(
                    self.signals_by_symbol[symbol][primary_tf],
                    max_age_minutes=self.primary_timeframe.to_minutes() * 2
                )
                
                # Check for signals that could be confirmed (same direction)
                for recent_signal in recent_signals:
                    if recent_signal.direction == tf_signal.direction:
                        recent_signal.add_confirmation(tf_signal.timeframe)
                        tf_signal.add_confirmation(self.primary_timeframe)
                        logger.info(f"Primary timeframe signal {recent_signal} confirmed by {tf_signal.timeframe.value}")
                        
                        # Check if we now have enough confirmations for a trade
                        self._check_for_trade_signal(recent_signal)
    
    def _get_recent_signals(
        self,
        signals: List[TimeframeAwareSignal],
        max_age_minutes: int
    ) -> List[TimeframeAwareSignal]:
        """
        Get recent signals within a certain age.
        
        Args:
            signals: List of signals
            max_age_minutes: Maximum age in minutes
            
        Returns:
            List of recent signals
        """
        now = datetime.now()
        return [
            s for s in signals 
            if (now - s.timestamp).total_seconds() / 60 <= max_age_minutes
        ]
    
    def _check_for_trade_signal(self, tf_signal: TimeframeAwareSignal) -> None:
        """
        Check if a signal has enough confirmations for a trade.
        
        Args:
            tf_signal: Signal to check
        """
        # Only generate trades for primary timeframe signals
        if not tf_signal.primary_timeframe:
            return
        
        # Check if confirmation is required
        if self.confirmation_required:
            # Check if we have enough confirmations
            if tf_signal.get_confirmation_count() < self.min_confirmations:
                return
        
        # Generate a trade signal
        self._generate_trade_signal(tf_signal)
    
    def _generate_trade_signal(self, tf_signal: TimeframeAwareSignal) -> None:
        """
        Generate a trade signal from a timeframe-aware signal.
        
        Args:
            tf_signal: Signal to generate trade from
        """
        # Create trade parameters
        trade_params = {
            "symbol": tf_signal.symbol,
            "direction": tf_signal.direction.value,
            "price": tf_signal.price,
            "signal_source": tf_signal.source.value,
            "timeframe": tf_signal.timeframe.value,
            "timeframe_confirmations": list(tf_signal.confirmed_by),
            "confirmation_count": tf_signal.get_confirmation_count(),
            "signal_metadata": tf_signal.metadata
        }
        
        # Create and publish the trade signal event
        event = Event(
            event_type=EventType.TRADE_SIGNAL,
            data=trade_params
        )
        
        self.event_bus.publish(event)
        logger.info(f"Generated trade signal for {tf_signal}")
    
    def get_signals_for_symbol(
        self,
        symbol: str,
        timeframe: Optional[Union[str, Timeframe]] = None
    ) -> List[TimeframeAwareSignal]:
        """
        Get signals for a specific symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Optional timeframe filter
            
        Returns:
            List of signals
        """
        if symbol not in self.signals_by_symbol:
            return []
        
        if timeframe:
            if isinstance(timeframe, str):
                tf = Timeframe.from_string(timeframe)
            else:
                tf = timeframe
            
            tf_value = tf.value
            if tf_value not in self.signals_by_symbol[symbol]:
                return []
            
            return self.signals_by_symbol[symbol][tf_value]
        
        # Return all timeframes
        signals = []
        for tf_signals in self.signals_by_symbol[symbol].values():
            signals.extend(tf_signals)
        
        return signals
    
    def get_signals_for_timeframe(
        self,
        timeframe: Union[str, Timeframe]
    ) -> Dict[str, List[TimeframeAwareSignal]]:
        """
        Get signals for a specific timeframe.
        
        Args:
            timeframe: Timeframe
            
        Returns:
            Dictionary of signals by symbol
        """
        if isinstance(timeframe, str):
            tf = Timeframe.from_string(timeframe)
        else:
            tf = timeframe
        
        tf_value = tf.value
        if tf_value not in self.signals_by_timeframe:
            return {}
        
        return self.signals_by_timeframe[tf_value]
    
    def get_latest_signal(
        self,
        symbol: str,
        timeframe: Optional[Union[str, Timeframe]] = None
    ) -> Optional[TimeframeAwareSignal]:
        """
        Get the latest signal for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Optional timeframe filter
            
        Returns:
            Latest signal or None
        """
        signals = self.get_signals_for_symbol(symbol, timeframe)
        if not signals:
            return None
        
        return max(signals, key=lambda s: s.timestamp)
    
    def clear_old_signals(self, max_age_hours: int = 24) -> int:
        """
        Clear signals older than a certain age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of signals cleared
        """
        now = datetime.now()
        max_age = timedelta(hours=max_age_hours)
        count = 0
        
        # Clear by timeframe
        for tf, symbols in self.signals_by_timeframe.items():
            for symbol, signals in list(symbols.items()):
                filtered_signals = [s for s in signals if now - s.timestamp <= max_age]
                count += len(signals) - len(filtered_signals)
                self.signals_by_timeframe[tf][symbol] = filtered_signals
        
        # Clear by symbol
        for symbol, timeframes in list(self.signals_by_symbol.items()):
            for tf, signals in list(timeframes.items()):
                filtered_signals = [s for s in signals if now - s.timestamp <= max_age]
                self.signals_by_symbol[symbol][tf] = filtered_signals
        
        return count


def create_timeframe_manager(config: Optional[Dict[str, Any]] = None) -> TimeframeManager:
    """
    Factory function to create a TimeframeManager with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TimeframeManager
    """
    if not config:
        return TimeframeManager()
    
    # Extract configuration
    primary_tf = config.get("primary_timeframe", "1h")
    confirmation_tfs = config.get("confirmation_timeframes", ["4h", "15m"])
    confirmation_required = config.get("confirmation_required", True)
    min_confirmations = config.get("min_confirmations", 1)
    symbol_config = config.get("symbol_timeframe_config", {})
    
    return TimeframeManager(
        primary_timeframe=primary_tf,
        confirmation_timeframes=confirmation_tfs,
        confirmation_required=confirmation_required,
        min_confirmations=min_confirmations,
        symbol_timeframe_config=symbol_config
    )


if __name__ == "__main__":
    # Example usage
    tf_manager = TimeframeManager(
        primary_timeframe=Timeframe.H1,
        confirmation_timeframes=[Timeframe.H4, Timeframe.M15],
        confirmation_required=True,
        min_confirmations=1
    )
    
    # Example signal
    signal = ExternalSignal(
        symbol="EURUSD",
        source=SignalSource.TRADINGVIEW,
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        timestamp=datetime.now(),
        price=1.05
    )
    
    # Create timeframe-aware signal
    tf_signal = TimeframeAwareSignal(
        signal=signal,
        timeframe=Timeframe.H1,
        primary_timeframe=True
    )
    
    # Process signal
    tf_manager._process_signal(tf_signal)
    
    # Example of confirmation from higher timeframe
    higher_signal = ExternalSignal(
        symbol="EURUSD",
        source=SignalSource.TRADINGVIEW,
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        timestamp=datetime.now(),
        price=1.05
    )
    
    higher_tf_signal = TimeframeAwareSignal(
        signal=higher_signal,
        timeframe=Timeframe.H4,
        primary_timeframe=False
    )
    
    # Process higher timeframe signal
    tf_manager._process_signal(higher_tf_signal)
    
    # Check if confirmation worked
    print(f"Primary signal confirmations: {tf_signal.confirmed_by}")
    print(f"Higher TF signal confirmations: {higher_tf_signal.confirmed_by}")
