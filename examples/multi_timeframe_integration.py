#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Integration Example

This script demonstrates how to integrate the TimeframeManager with pattern recognition
and external signals to enable multi-timeframe confirmation and analysis.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import threading

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiTimeframeIntegration")

# Import trading bot modules
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.integrations.webhook_handler import WebhookHandler
from trading_bot.strategies.external_signal_strategy import (
    ExternalSignalStrategy, ExternalSignal, SignalSource, SignalType, Direction
)
from trading_bot.strategies.pattern_enhanced_strategy import PatternEnhancedStrategy
from trading_bot.analysis.pattern_recognition import PatternRecognition
from trading_bot.core.timeframe_manager import (
    Timeframe, TimeframeManager, TimeframeAwareSignal, create_timeframe_manager
)
from examples.pattern_signal_integration import PatternSignalIntegration


class MultiTimeframeIntegration(PatternSignalIntegration):
    """
    Extension of PatternSignalIntegration that adds multi-timeframe support.
    
    This class demonstrates how to:
    1. Configure and use the TimeframeManager
    2. Handle signals from multiple timeframes
    3. Implement higher timeframe confirmation
    4. Generate trades based on multi-timeframe analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-timeframe integration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Extract timeframe configuration
        tf_config = config.get("timeframe_config", {})
        
        # Create timeframe manager
        self.timeframe_manager = create_timeframe_manager(tf_config)
        
        # Register with service registry
        ServiceRegistry.register("timeframe_manager", self.timeframe_manager)
        
        # Additional state for multi-timeframe tracking
        self.timeframe_signals = {}
        self.asset_correlations = {}
        
        # Configure monitored assets and timeframes
        self.monitored_assets = config.get("monitored_assets", ["EURUSD", "GBPUSD", "USDJPY"])
        self.monitored_timeframes = config.get("monitored_timeframes", ["1h", "4h", "15m"])
        
        # Convert monitored timeframes to Timeframe enum
        self.monitored_tf_enums = [Timeframe.from_string(tf) for tf in self.monitored_timeframes]
        
        # Extend event handlers for multi-timeframe awareness
        self._register_additional_event_handlers()
        
        logger.info("MultiTimeframeIntegration initialized")
    
    def _register_additional_event_handlers(self):
        """Register additional event handlers for multi-timeframe awareness."""
        # We'll add a handler for trade signals after multi-timeframe confirmation
        self.event_bus.subscribe(
            EventType.TRADE_SIGNAL,
            self._handle_multi_timeframe_trade_signal
        )
    
    def _handle_external_signal(self, event: Event):
        """
        Override the parent method to add timeframe awareness.
        
        Args:
            event: External signal event
        """
        # First, let the parent class handle basic signal processing
        super()._handle_external_signal(event)
        
        # Now add multi-timeframe awareness
        signal_data = event.data.get("signal", {})
        if not signal_data:
            return
        
        symbol = signal_data.get("symbol")
        if not symbol or symbol not in self.monitored_assets:
            return
        
        # Extract timeframe from the signal or use default
        timeframe_str = signal_data.get("timeframe", "1h")
        
        # Update our timeframe signals dictionary
        if symbol not in self.timeframe_signals:
            self.timeframe_signals[symbol] = {}
        
        if timeframe_str not in self.timeframe_signals[symbol]:
            self.timeframe_signals[symbol][timeframe_str] = []
        
        # Add the signal to our tracking
        signal_entry = {
            "data": signal_data,
            "timestamp": datetime.now(),
            "processed": False
        }
        
        self.timeframe_signals[symbol][timeframe_str].append(signal_entry)
        
        logger.info(f"Processed {timeframe_str} signal for {symbol}")
        
        # Check for correlations across timeframes
        self._check_timeframe_correlations(symbol)
    
    def _handle_pattern_detected(self, event: Event):
        """
        Override the parent method to add timeframe awareness.
        
        Args:
            event: Pattern detected event
        """
        # First, let the parent class handle basic pattern processing
        super()._handle_pattern_detected(event)
        
        # Now add multi-timeframe awareness
        pattern_data = event.data
        if not pattern_data:
            return
        
        symbol = pattern_data.get("symbol")
        if not symbol or symbol not in self.monitored_assets:
            return
        
        # Extract timeframe from the pattern or use default
        timeframe_str = pattern_data.get("timeframe", "1h")
        
        # Update our pattern tracking by timeframe
        if symbol not in self.timeframe_signals:
            self.timeframe_signals[symbol] = {}
        
        if timeframe_str not in self.timeframe_signals[symbol]:
            self.timeframe_signals[symbol][timeframe_str] = []
        
        # Add pattern detection as a special type of signal
        pattern_entry = {
            "type": "pattern",
            "data": pattern_data,
            "timestamp": datetime.now(),
            "processed": False
        }
        
        self.timeframe_signals[symbol][timeframe_str].append(pattern_entry)
        
        logger.info(f"Processed {timeframe_str} pattern for {symbol}")
        
        # Check for correlations across timeframes
        self._check_timeframe_correlations(symbol)
    
    def _check_timeframe_correlations(self, symbol: str):
        """
        Check for correlations between signals across different timeframes.
        
        Args:
            symbol: Symbol to check correlations for
        """
        if symbol not in self.timeframe_signals:
            return
        
        # Check if we have signals in multiple timeframes
        timeframes_with_signals = set(self.timeframe_signals[symbol].keys())
        if len(timeframes_with_signals) < 2:
            return
        
        logger.info(f"Checking timeframe correlations for {symbol}")
        
        # Get recent signals for each timeframe
        recent_signals = {}
        for tf, signals in self.timeframe_signals[symbol].items():
            # Get signals from the last hour
            recent = [
                s for s in signals 
                if (datetime.now() - s["timestamp"]).total_seconds() < 3600
            ]
            if recent:
                recent_signals[tf] = recent
        
        # Check for alignment between primary and confirmation timeframes
        primary_tf = self.timeframe_manager.primary_timeframe.value
        confirmation_tfs = [tf.value for tf in self.timeframe_manager.confirmation_timeframes]
        
        # If we have signals in both primary and at least one confirmation timeframe
        if primary_tf in recent_signals and any(tf in recent_signals for tf in confirmation_tfs):
            primary_signals = recent_signals[primary_tf]
            
            for tf in confirmation_tfs:
                if tf in recent_signals:
                    confirmation_signals = recent_signals[tf]
                    
                    # Look for direction alignment
                    for p_signal in primary_signals:
                        p_direction = self._get_signal_direction(p_signal)
                        
                        for c_signal in confirmation_signals:
                            c_direction = self._get_signal_direction(c_signal)
                            
                            # If directions match, we have correlation
                            if p_direction and c_direction and p_direction == c_direction:
                                logger.info(f"Found correlated {p_direction} signals for {symbol} on {primary_tf} and {tf}")
                                
                                # Record the correlation
                                self._record_correlation(symbol, primary_tf, tf, p_direction)
                                
                                # Generate a multi-timeframe trade signal
                                self._generate_multi_timeframe_signal(
                                    symbol=symbol,
                                    direction=p_direction,
                                    primary_signal=p_signal,
                                    confirmation_signal=c_signal,
                                    primary_tf=primary_tf,
                                    confirmation_tf=tf
                                )
    
    def _get_signal_direction(self, signal_entry: Dict[str, Any]) -> Optional[str]:
        """
        Extract direction from a signal entry.
        
        Args:
            signal_entry: Signal entry dictionary
            
        Returns:
            Direction string or None
        """
        data = signal_entry.get("data", {})
        
        # For standard signals
        if "action" in data:
            action = data["action"].lower()
            if action in ["buy", "long"]:
                return "long"
            elif action in ["sell", "short"]:
                return "short"
        
        # For pattern detections
        if "direction" in data:
            return data["direction"].lower()
        
        return None
    
    def _record_correlation(self, symbol: str, primary_tf: str, confirmation_tf: str, direction: str):
        """
        Record a correlation between timeframes.
        
        Args:
            symbol: Trading symbol
            primary_tf: Primary timeframe
            confirmation_tf: Confirmation timeframe
            direction: Direction of the correlation
        """
        if symbol not in self.asset_correlations:
            self.asset_correlations[symbol] = []
        
        correlation = {
            "timestamp": datetime.now(),
            "primary_timeframe": primary_tf,
            "confirmation_timeframe": confirmation_tf,
            "direction": direction
        }
        
        self.asset_correlations[symbol].append(correlation)
    
    def _generate_multi_timeframe_signal(
        self,
        symbol: str,
        direction: str,
        primary_signal: Dict[str, Any],
        confirmation_signal: Dict[str, Any],
        primary_tf: str,
        confirmation_tf: str
    ):
        """
        Generate a trade signal from multi-timeframe correlation.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            primary_signal: Signal from primary timeframe
            confirmation_signal: Signal from confirmation timeframe
            primary_tf: Primary timeframe
            confirmation_tf: Confirmation timeframe
        """
        # Extract price information from signals
        price = primary_signal.get("data", {}).get("price")
        if not price:
            price = confirmation_signal.get("data", {}).get("price")
        
        # Create trade parameters
        trade_params = {
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "timeframe": primary_tf,
            "confirmation_timeframe": confirmation_tf,
            "multi_timeframe": True,
            "source": "multi_timeframe_integration"
        }
        
        # Add any available stop loss / take profit
        for signal in [primary_signal, confirmation_signal]:
            data = signal.get("data", {})
            if "stop_loss" in data:
                trade_params["stop_loss"] = data["stop_loss"]
            if "take_profit" in data:
                trade_params["take_profit"] = data["take_profit"]
        
        # Create and publish trade signal event
        event = Event(
            event_type=EventType.TRADE_SIGNAL,
            data=trade_params
        )
        
        self.event_bus.publish(event)
        logger.info(f"Generated multi-timeframe trade signal for {symbol} {direction}")
    
    def _handle_multi_timeframe_trade_signal(self, event: Event):
        """
        Handle a trade signal after multi-timeframe confirmation.
        
        Args:
            event: Trade signal event
        """
        signal_data = event.data
        if not signal_data:
            return
        
        # Check if this is a multi-timeframe signal
        if signal_data.get("multi_timeframe"):
            symbol = signal_data.get("symbol")
            direction = signal_data.get("direction")
            
            logger.info(f"Processing multi-timeframe trade signal: {symbol} {direction}")
            
            # Here you would integrate with your trading system
            # For this example, we'll just log the signal
            
            # Mark the correlating signals as processed
            if symbol in self.timeframe_signals:
                primary_tf = signal_data.get("timeframe")
                confirmation_tf = signal_data.get("confirmation_timeframe")
                
                if primary_tf in self.timeframe_signals[symbol]:
                    for signal in self.timeframe_signals[symbol][primary_tf]:
                        if not signal["processed"]:
                            signal["processed"] = True
                            break
                
                if confirmation_tf in self.timeframe_signals[symbol]:
                    for signal in self.timeframe_signals[symbol][confirmation_tf]:
                        if not signal["processed"]:
                            signal["processed"] = True
                            break
    
    def get_timeframe_signals(self, symbol: Optional[str] = None, 
                             timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Get signals filtered by symbol and timeframe.
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            
        Returns:
            Dictionary of signals
        """
        if symbol:
            if symbol not in self.timeframe_signals:
                return {}
            
            if timeframe:
                if timeframe not in self.timeframe_signals[symbol]:
                    return {}
                
                return {symbol: {timeframe: self.timeframe_signals[symbol][timeframe]}}
            
            return {symbol: self.timeframe_signals[symbol]}
        
        if timeframe:
            result = {}
            for sym, timeframes in self.timeframe_signals.items():
                if timeframe in timeframes:
                    if sym not in result:
                        result[sym] = {}
                    result[sym][timeframe] = timeframes[timeframe]
            return result
        
        return self.timeframe_signals
    
    def get_asset_correlations(self, symbol: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get asset correlations.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dictionary of correlations
        """
        if symbol:
            if symbol not in self.asset_correlations:
                return {}
            
            return {symbol: self.asset_correlations[symbol]}
        
        return self.asset_correlations
    
    def _cleanup_old_signals(self, max_age_hours: int = 24):
        """
        Clean up old signals and correlations.
        
        Args:
            max_age_hours: Maximum age in hours
        """
        now = datetime.now()
        max_age = timedelta(hours=max_age_hours)
        
        # Clean up timeframe signals
        for symbol in list(self.timeframe_signals.keys()):
            for tf in list(self.timeframe_signals[symbol].keys()):
                self.timeframe_signals[symbol][tf] = [
                    s for s in self.timeframe_signals[symbol][tf]
                    if now - s["timestamp"] <= max_age
                ]
        
        # Clean up asset correlations
        for symbol in list(self.asset_correlations.keys()):
            self.asset_correlations[symbol] = [
                c for c in self.asset_correlations[symbol]
                if now - c["timestamp"] <= max_age
            ]
        
        # Also clean up the timeframe manager
        if hasattr(self, "timeframe_manager"):
            self.timeframe_manager.clear_old_signals(max_age_hours)
    
    def start(self):
        """Start the integration services."""
        super().start()
        
        # Start a periodic cleanup task
        self._start_cleanup_thread()
        
        logger.info("Multi-Timeframe Integration started")
    
    def _start_cleanup_thread(self):
        """Start a background thread for periodic cleanup."""
        def cleanup_task():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_signals()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def stop(self):
        """Stop the integration services."""
        super().stop()
        logger.info("Multi-Timeframe Integration stopped")


# Example configuration
default_config = {
    "webhook_port": 5000,
    "webhook_path": "webhook",
    "webhook_auth_token": None,
    
    "pattern_strategy_config": {
        "confidence_threshold": 0.7,
        "lookback_periods": 20,
        "confirmation_required": True
    },
    
    "timeframe_config": {
        "primary_timeframe": "1h",
        "confirmation_timeframes": ["4h", "15m"],
        "confirmation_required": True,
        "min_confirmations": 1
    },
    
    "monitored_assets": ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "AAPL", "MSFT"],
    "monitored_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
}


def main():
    """Run the multi-timeframe integration example."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "multi_timeframe_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = default_config
            # Save default config for future reference
            with open(config_path, 'w') as f:
                json.dump(config, indent=2, fp=f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = default_config
    
    # Create the integration
    integration = MultiTimeframeIntegration(config)
    
    try:
        # Start the integration
        integration.start()
        
        # Keep running until keyboard interrupt
        logger.info("Multi-Timeframe Integration running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Stop the integration
        integration.stop()


if __name__ == "__main__":
    main()
