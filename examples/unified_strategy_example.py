#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Strategy Example

This example demonstrates how the unified strategy handling system integrates
different types of strategies (indicators, patterns, external signals) with
conflict resolution and coordinated signal processing.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import time
import random
from typing import Dict, Any, List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnifiedStrategyExample")

# Import required components
from trading_bot.core.strategy_base import (
    Strategy, StrategyType, StrategyPriority, ConflictResolutionMode, SignalTag
)
from trading_bot.core.unified_strategy_coordinator import UnifiedStrategyCoordinator
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.strategies.external_signal_strategy import (
    ExternalSignalStrategy, ExternalSignal, SignalSource, SignalType, Direction
)
from trading_bot.core.service_registry import ServiceRegistry


# Create example strategies
class IndicatorStrategy(Strategy):
    """Example indicator-based strategy."""
    
    def __init__(self, strategy_id=None, symbols=None):
        super().__init__(
            strategy_id=strategy_id or "indicator_strategy",
            name="Moving Average Crossover Strategy",
            description="Example indicator-based strategy using moving average crossovers",
            symbols=symbols or ["EURUSD", "GBPUSD"],
            timeframe="1h",
            strategy_type=StrategyType.INDICATOR_BASED,
            priority=StrategyPriority.NORMAL,
            signal_tags={SignalTag.ENTRY.value, SignalTag.EXIT.value, "moving_average_crossover"},
            conflict_resolution=ConflictResolutionMode.PRIORITY_BASED
        )
    
    def on_signal_generation(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signals based on indicator values."""
        # This would normally contain indicator logic
        # For this example, we'll just generate a random signal occasionally
        if random.random() < 0.3:  # 30% chance of generating a signal
            direction = random.choice(["long", "short"])
            return {
                "symbol": data.get("symbol", random.choice(self.symbols)),
                "timeframe": self.timeframe,
                "direction": direction,
                "signal_type": "entry",
                "price": data.get("price", 1.2345),
                "confidence": 0.7,
                "tags": ["ma_crossover"],
                "metadata": {
                    "fast_ma": 10,
                    "slow_ma": 20,
                    "crossover_type": "bullish" if direction == "long" else "bearish"
                }
            }
        return None
    
    def on_data(self, data: Dict[str, Any]) -> None:
        """Process market data updates."""
        # For the example, we'll just pass this to signal generation
        signal = self.generate_signal(data)
        if signal:
            logger.info(f"Indicator strategy generated {signal['direction']} signal for {signal['symbol']}")


class PatternStrategy(Strategy):
    """Example pattern recognition strategy."""
    
    def __init__(self, strategy_id=None, symbols=None):
        super().__init__(
            strategy_id=strategy_id or "pattern_strategy",
            name="Candlestick Pattern Strategy",
            description="Example pattern-based strategy using candlestick patterns",
            symbols=symbols or ["EURUSD", "GBPUSD"],
            timeframe="1h",
            strategy_type=StrategyType.PATTERN_RECOGNITION,
            priority=StrategyPriority.HIGH,  # Higher priority than indicator strategy
            signal_tags={SignalTag.ENTRY.value, "pattern", "candlestick"},
            conflict_resolution=ConflictResolutionMode.PRIORITY_BASED
        )
        self.patterns = ["engulfing", "doji", "hammer", "shooting_star", "morning_star"]
    
    def on_signal_generation(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signals based on detected patterns."""
        # This would normally contain pattern detection logic
        # For this example, we'll just generate a random signal occasionally
        if random.random() < 0.2:  # 20% chance of generating a signal
            pattern = random.choice(self.patterns)
            direction = "long" if pattern in ["engulfing", "hammer", "morning_star"] else "short"
            
            return {
                "symbol": data.get("symbol", random.choice(self.symbols)),
                "timeframe": self.timeframe,
                "direction": direction,
                "signal_type": "entry",
                "price": data.get("price", 1.2345),
                "confidence": 0.8,  # Higher confidence than indicator
                "tags": ["pattern", pattern],
                "metadata": {
                    "pattern_type": pattern,
                    "pattern_strength": random.choice(["weak", "moderate", "strong"]),
                    "confirmation": random.choice([True, False])
                }
            }
        return None
    
    def on_data(self, data: Dict[str, Any]) -> None:
        """Process market data updates."""
        # For the example, we'll just pass this to signal generation
        signal = self.generate_signal(data)
        if signal:
            logger.info(f"Pattern strategy detected {signal['metadata']['pattern_type']} "
                       f"for {signal['symbol']} - {signal['direction']} signal")


class ExternalSignalProcessor:
    """Processes external signals from webhooks, APIs, etc."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.external_strategy = ExternalSignalStrategy(
            name="External Signal Processor",
            auto_trade=False,  # Let the coordinator handle trade generation
            register_webhook=False  # We'll manually feed signals for this example
        )
        logger.info("External Signal Processor initialized")
    
    def process_tradingview_signal(self, signal_data: Dict[str, Any]) -> None:
        """Process a signal from TradingView."""
        # Create an external signal
        signal = ExternalSignal(
            symbol=signal_data.get("symbol", "EURUSD"),
            source=SignalSource.TRADINGVIEW,
            signal_type=SignalType.ENTRY,
            direction=Direction(signal_data.get("direction", "long")),
            price=signal_data.get("price"),
            metadata=signal_data
        )
        
        # Publish the signal
        self.event_bus.publish(Event(
            event_type=EventType.EXTERNAL_SIGNAL,
            data={"signal": signal.to_dict(), "signal_object": signal}
        ))
        
        logger.info(f"Processed TradingView signal: {signal.symbol} {signal.direction.value}")


class UnifiedStrategyExample:
    """Demonstrates the unified strategy handling system."""
    
    def __init__(self):
        self.event_bus = EventBus()
        
        # Create the unified strategy coordinator
        self.coordinator = UnifiedStrategyCoordinator(
            default_conflict_mode=ConflictResolutionMode.PRIORITY_BASED
        )
        
        # Create example strategies
        self.indicator_strategy = IndicatorStrategy(symbols=["EURUSD", "GBPUSD", "USDJPY"])
        self.pattern_strategy = PatternStrategy(symbols=["EURUSD", "GBPUSD", "USDJPY"])
        self.external_processor = ExternalSignalProcessor()
        
        # Register strategies with coordinator
        self.coordinator.register_strategy(self.indicator_strategy)
        self.coordinator.register_strategy(self.pattern_strategy)
        
        # Subscribe to trade signals
        self.event_bus.subscribe(EventType.TRADE_SIGNAL, self._on_trade_signal)
        self.event_bus.subscribe(EventType.SIGNAL_REJECTED, self._on_signal_rejected)
        self.event_bus.subscribe(EventType.STRATEGY_CONFLICT, self._on_strategy_conflict)
        
        logger.info("UnifiedStrategyExample initialized")
    
    def _on_trade_signal(self, event: Event) -> None:
        """Handle trade signals after processing by the coordinator."""
        signal_data = event.data
        logger.info(f"TRADE SIGNAL: {signal_data['symbol']} {signal_data['direction']} "
                   f"from {signal_data['source']} (strategy: {signal_data['strategy_id']})")
    
    def _on_signal_rejected(self, event: Event) -> None:
        """Handle rejected signals."""
        signal_data = event.data
        logger.info(f"REJECTED SIGNAL: {signal_data['symbol']} {signal_data['direction']} "
                   f"- Reason: {signal_data.get('rejection_reason', 'unknown')}")
    
    def _on_strategy_conflict(self, event: Event) -> None:
        """Handle strategy conflicts."""
        conflict_data = event.data
        logger.info(f"CONFLICT DETECTED: {conflict_data['symbol']} - "
                   f"between strategies: {', '.join(conflict_data['strategies'])}")
    
    def start(self) -> None:
        """Start the example."""
        # Start the strategies
        self.indicator_strategy.start()
        self.pattern_strategy.start()
        
        logger.info("UnifiedStrategyExample started")
        
        # Simulate market data and external signals
        self._run_simulation()
    
    def _run_simulation(self) -> None:
        """Run a simulation with sample data."""
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        logger.info("Starting simulation...")
        
        # Simulate 20 market updates
        for i in range(20):
            # Simulate market data
            symbol = random.choice(symbols)
            price = 1.0 + random.random()
            
            market_data = {
                "symbol": symbol,
                "timeframe": "1h",
                "price": price,
                "timestamp": datetime.now()
            }
            
            # Feed to strategies
            self.indicator_strategy.on_data(market_data)
            self.pattern_strategy.on_data(market_data)
            
            # Occasionally send external signals
            if random.random() < 0.2:  # 20% chance
                direction = random.choice(["long", "short"])
                external_signal = {
                    "symbol": symbol,
                    "direction": direction,
                    "price": price,
                    "timeframe": "1h",
                    "strategy": "tradingview_alert",
                    "confidence": 0.75
                }
                self.external_processor.process_tradingview_signal(external_signal)
            
            # Sleep to make the output more readable
            time.sleep(0.5)
        
        logger.info("Simulation complete")
        
        # Show summary
        self._show_summary()
    
    def _show_summary(self) -> None:
        """Show summary of the simulation."""
        normalized_count = len(self.coordinator.normalized_signals)
        processed_count = len(self.coordinator.processed_signals)
        conflicts = len(self.coordinator.resolved_conflicts)
        
        logger.info("\n=== SIMULATION SUMMARY ===")
        logger.info(f"Normalized signals: {normalized_count}")
        logger.info(f"Processed signals: {processed_count}")
        logger.info(f"Conflicts resolved: {conflicts}")
        
        if conflicts > 0:
            logger.info("\n=== CONFLICT RESOLUTIONS ===")
            for conflict in self.coordinator.resolved_conflicts:
                resolution = conflict.resolution
                logger.info(f"Conflict on {conflict.symbol}: Resolved by {resolution['mode']}")
                if "winning_strategy" in resolution:
                    logger.info(f"  Winner: {resolution['winning_strategy']}")


if __name__ == "__main__":
    example = UnifiedStrategyExample()
    example.start()
