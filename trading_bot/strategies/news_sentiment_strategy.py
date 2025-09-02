#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News Sentiment Trading Strategy

This strategy analyzes news sentiment data and generates trading signals based
on sentiment scores, impact, and category of news events.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from trading_bot.core.strategy_base import StrategyBase
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class NewsSentimentStrategy(StrategyBase):
    """
    A strategy that trades based on news sentiment analysis.
    """
    
    def __init__(self, 
                 name: str = "NewsSentiment", 
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the news sentiment strategy.
        
        Args:
            name: Strategy name
            config: Configuration parameters
            event_bus: Event bus for communication
        """
        super().__init__(name=name, config=config, event_bus=event_bus)
        
        # Strategy specific parameters with defaults
        self.config = config or {}
        self.sentiment_threshold = self.config.get("sentiment_threshold", 0.3)
        self.impact_threshold = self.config.get("impact_threshold", 0.5)
        self.position_sizing_factor = self.config.get("position_sizing_factor", 1.0)
        self.news_decay_hours = self.config.get("news_decay_hours", 24)
        self.symbols = self.config.get("symbols", [])
        self.news_categories = self.config.get("news_categories", ["earnings", "analyst", "regulatory", "product"])
        
        # Strategy state
        self.active_signals = {}
        self.news_events = []
        self.last_execution = {}
        
        # Register for news events if we have an event bus
        if self.event_bus:
            self._register_event_handlers()
            
        logger.info(f"News Sentiment Strategy initialized with {len(self.symbols)} symbols")
        
    def _register_event_handlers(self) -> None:
        """Register for relevant event types."""
        self.event_bus.subscribe(EventType.NEWS, self._on_news_event)
        self.event_bus.subscribe("news_event", self._on_news_event)  # Custom event type used in our system
        
    def _on_news_event(self, event: Event) -> None:
        """
        Process incoming news events.
        
        Args:
            event: News event
        """
        data = event.data
        logger.debug(f"Received news event: {data}")
        
        # Store event for historical analysis
        self.news_events.append({
            "data": data,
            "timestamp": event.timestamp or datetime.now()
        })
        
        # Limit stored events to last 1000
        if len(self.news_events) > 1000:
            self.news_events = self.news_events[-1000:]
        
        # Process the event for trading signals
        self._process_news_for_signals(data)
        
    def _process_news_for_signals(self, news_data: Dict[str, Any]) -> None:
        """
        Process news data to generate trading signals.
        
        Args:
            news_data: News event data
        """
        # Extract key fields
        symbol = news_data.get("symbol")
        sentiment = news_data.get("sentiment", 0)
        impact = news_data.get("impact", 0.5)
        category = news_data.get("category", "general")
        
        # Skip if no symbol or not in our watchlist
        if not symbol or (self.symbols and symbol not in self.symbols):
            return
            
        # Skip if category not of interest
        if self.news_categories and category not in self.news_categories:
            return
        
        # Calculate signal strength based on sentiment and impact
        if isinstance(sentiment, str):
            # Convert text sentiment to numeric
            sentiment_value = {
                "positive": 1.0,
                "negative": -1.0,
                "neutral": 0.0
            }.get(sentiment.lower(), 0.0)
        else:
            sentiment_value = sentiment
            
        signal_strength = sentiment_value * impact
        
        # Check if signal exceeds our threshold (in either direction)
        if abs(signal_strength) < self.sentiment_threshold:
            logger.debug(f"News signal for {symbol} below threshold: {signal_strength}")
            return
            
        # Check if we've traded this symbol recently
        now = datetime.now()
        last_trade_time = self.last_execution.get(symbol)
        if last_trade_time and (now - last_trade_time).total_seconds() < 3600:  # 1 hour cooldown
            logger.debug(f"Skipping {symbol} due to recent trade")
            return
            
        # Determine position size by signal strength
        position_size = round(abs(signal_strength) * self.position_sizing_factor, 2)
        position_size = min(position_size, 1.0)  # Cap at 100% of allocation
        
        # Create signal with direction based on sentiment
        signal = {
            "symbol": symbol,
            "direction": "buy" if signal_strength > 0 else "sell",
            "confidence": abs(signal_strength),
            "size": position_size,
            "source": "news_sentiment",
            "timestamp": now.isoformat(),
            "metadata": {
                "headline": news_data.get("headline", ""),
                "category": category,
                "sentiment": sentiment_value,
                "impact": impact
            }
        }
        
        # Store the signal
        self.active_signals[symbol] = signal
        
        # Emit the signal event if we have an event bus
        if self.event_bus:
            self._emit_signal_event(signal)
        
        # Update last execution time
        self.last_execution[symbol] = now
        
        logger.info(f"Generated {signal['direction']} signal for {symbol} with size {position_size}")
        
    def _emit_signal_event(self, signal: Dict[str, Any]) -> None:
        """
        Emit signal event to the event bus.
        
        Args:
            signal: Trade signal details
        """
        if not self.event_bus:
            return
            
        event = Event(
            event_type=EventType.SIGNAL,
            source=f"strategy.{self.name}",
            data=signal,
            timestamp=datetime.now()
        )
        
        self.event_bus.publish(event)
        
    def update(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update strategy with new market data.
        
        Args:
            market_data: Latest market data
            
        Returns:
            Signal if generated, None otherwise
        """
        # News sentiment strategy primarily relies on news events through the event handler
        # but we can use market data to expire old signals or adjust positions
        
        # Check if any active signals should be closed due to time decay
        now = datetime.now()
        signals_to_remove = []
        
        for symbol, signal in self.active_signals.items():
            # Parse the timestamp
            signal_time = datetime.fromisoformat(signal["timestamp"])
            
            # Check if signal is old
            if (now - signal_time).total_seconds() > self.news_decay_hours * 3600:
                # Generate exit signal
                exit_signal = {
                    "symbol": symbol,
                    "direction": "sell" if signal["direction"] == "buy" else "buy",  # Opposite direction to close
                    "confidence": 1.0,  # High confidence for exits
                    "size": 1.0,  # Close full position
                    "source": "news_sentiment_decay",
                    "timestamp": now.isoformat(),
                    "metadata": {
                        "reason": "time_decay",
                        "original_signal": signal
                    }
                }
                
                # Emit exit signal
                if self.event_bus:
                    self._emit_signal_event(exit_signal)
                    
                # Mark for removal
                signals_to_remove.append(symbol)
                
                logger.info(f"Closing position for {symbol} due to signal decay")
        
        # Remove expired signals
        for symbol in signals_to_remove:
            del self.active_signals[symbol]
            
        # Return the most recent signal if available
        if self.active_signals:
            newest_symbol = max(self.active_signals.items(), 
                               key=lambda x: datetime.fromisoformat(x[1]["timestamp"]))
            return self.active_signals[newest_symbol[0]]
            
        return None
        
    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get the current state of the strategy.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            "name": self.name,
            "active_signals": len(self.active_signals),
            "recent_news_events": len(self.news_events),
            "symbols_watched": len(self.symbols),
            "sentiment_threshold": self.sentiment_threshold,
            "impact_threshold": self.impact_threshold
        }
