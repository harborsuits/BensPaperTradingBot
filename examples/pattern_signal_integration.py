#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Recognition and External Signal Integration Example

This script demonstrates how to integrate pattern recognition with the external signal strategy,
allowing TradingView, Alpaca, and Finnhub signals to be combined with internal pattern detection.
"""

import logging
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternSignalIntegration")

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


class PatternSignalIntegration:
    """
    Integration of pattern recognition with external signals.
    
    This class demonstrates how to:
    1. Set up a webhook server for external signals
    2. Initialize the external signal strategy
    3. Configure the pattern enhanced strategy
    4. Set up event handlers for integrating signals with patterns
    5. Process combined signals and patterns for trading decisions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pattern signal integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.event_bus = EventBus()
        
        # Initialize services
        self._initialize_services()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Trading state
        self.active_signals = {}
        self.pattern_confirmations = {}
        self.combined_signals = {}
        
    def _initialize_services(self):
        """Initialize required services for integration."""
        # Initialize webhook handler for TradingView and other external sources
        webhook_port = self.config.get("webhook_port", 5000)
        webhook_path = self.config.get("webhook_path", "webhook")
        
        self.webhook_handler = WebhookHandler(
            port=webhook_port,
            path=webhook_path,
            auth_token=self.config.get("webhook_auth_token")
        )
        
        # Register with service registry
        ServiceRegistry.register("webhook_handler", self.webhook_handler)
        
        # Initialize external signal strategy
        self.external_signal_strategy = ExternalSignalStrategy(
            name="external_signal_strategy",
            max_history=100,
            auto_trade=False,  # We'll handle trades with our combined logic
            register_webhook=True  # Register with webhook handler
        )
        
        # Register with service registry
        ServiceRegistry.register("external_signal_strategy", self.external_signal_strategy)
        
        # Initialize pattern enhanced strategy
        self.pattern_strategy = PatternEnhancedStrategy(
            name="pattern_enhanced_strategy",
            config=self.config.get("pattern_strategy_config", {})
        )
        
        # Register with service registry
        ServiceRegistry.register("pattern_enhanced_strategy", self.pattern_strategy)
        
        logger.info("Services initialized")
    
    def _register_event_handlers(self):
        """Register event handlers for signal processing."""
        # Listen for external signals
        self.event_bus.subscribe(
            EventType.EXTERNAL_SIGNAL, 
            self._handle_external_signal
        )
        
        # Listen for pattern signals
        self.event_bus.subscribe(
            EventType.PATTERN_DETECTED,
            self._handle_pattern_detected
        )
        
        # Listen for trade signals (combined)
        self.event_bus.subscribe(
            EventType.TRADE_SIGNAL,
            self._handle_trade_signal
        )
        
        logger.info("Event handlers registered")
    
    def _handle_external_signal(self, event: Event):
        """
        Handle an external signal event.
        
        Args:
            event: External signal event
        """
        signal_data = event.data.get("signal", {})
        symbol = signal_data.get("symbol")
        signal_type = signal_data.get("signal_type")
        source = signal_data.get("source")
        direction = signal_data.get("direction")
        
        logger.info(f"Received external signal: {symbol} {signal_type} {direction} from {source}")
        
        # Store signal for potential pattern confirmation
        if symbol and signal_type:
            key = f"{symbol}_{signal_type}"
            self.active_signals[key] = {
                "timestamp": datetime.now(),
                "data": signal_data,
                "confirmed_by_pattern": False
            }
            
            # Check if we have a recent pattern detection for this symbol
            self._check_signal_pattern_match(symbol)
    
    def _handle_pattern_detected(self, event: Event):
        """
        Handle a pattern detected event.
        
        Args:
            event: Pattern detected event
        """
        pattern_data = event.data
        symbol = pattern_data.get("symbol")
        pattern_type = pattern_data.get("pattern_type")
        direction = pattern_data.get("direction")
        confidence = pattern_data.get("confidence", 0.0)
        
        logger.info(f"Pattern detected: {symbol} {pattern_type} {direction} (confidence: {confidence})")
        
        # Store pattern for potential signal confirmation
        if symbol and pattern_type:
            key = f"{symbol}_{pattern_type}"
            self.pattern_confirmations[key] = {
                "timestamp": datetime.now(),
                "data": pattern_data,
                "confirmed_by_signal": False
            }
            
            # Check if we have a recent signal for this symbol
            self._check_signal_pattern_match(symbol)
    
    def _check_signal_pattern_match(self, symbol: str):
        """
        Check if we have both a signal and pattern for a symbol.
        
        Args:
            symbol: Trading symbol to check
        """
        # Get recent signals for this symbol
        recent_signals = {
            k: v for k, v in self.active_signals.items() 
            if k.startswith(symbol) and 
            datetime.now() - v["timestamp"] < timedelta(minutes=15)
        }
        
        # Get recent patterns for this symbol
        recent_patterns = {
            k: v for k, v in self.pattern_confirmations.items()
            if k.startswith(symbol) and
            datetime.now() - v["timestamp"] < timedelta(minutes=15)
        }
        
        # If we have both, check for direction match
        if recent_signals and recent_patterns:
            signal_entry = next(iter(recent_signals.values()))
            pattern_entry = next(iter(recent_patterns.values()))
            
            signal_direction = signal_entry["data"].get("direction")
            pattern_direction = pattern_entry["data"].get("direction")
            
            if signal_direction and pattern_direction and signal_direction == pattern_direction:
                logger.info(f"Signal-pattern match for {symbol}: {signal_direction}")
                
                # Mark both as confirmed
                signal_entry["confirmed_by_pattern"] = True
                pattern_entry["confirmed_by_signal"] = True
                
                # Create a combined signal
                self._create_combined_signal(symbol, signal_entry, pattern_entry)
    
    def _create_combined_signal(
        self, 
        symbol: str, 
        signal_entry: Dict[str, Any], 
        pattern_entry: Dict[str, Any]
    ):
        """
        Create a combined signal from an external signal and pattern.
        
        Args:
            symbol: Trading symbol
            signal_entry: External signal data
            pattern_entry: Pattern data
        """
        # Get direction
        direction = signal_entry["data"].get("direction")
        
        # Get prices
        signal_price = signal_entry["data"].get("price")
        pattern_price = pattern_entry["data"].get("price")
        
        # Use the most recent price
        price = signal_price or pattern_price
        
        # Get metadata
        signal_metadata = signal_entry["data"].get("metadata", {})
        pattern_metadata = pattern_entry["data"].get("metadata", {})
        
        # Combine metadata
        metadata = {
            "signal_source": signal_entry["data"].get("source"),
            "signal_type": signal_entry["data"].get("signal_type"),
            "pattern_type": pattern_entry["data"].get("pattern_type"),
            "pattern_confidence": pattern_entry["data"].get("confidence"),
            "combined_source": "signal_pattern_integration"
        }
        
        # Add any additional metadata
        metadata.update(signal_metadata)
        metadata.update(pattern_metadata)
        
        # Create a combined signal key
        key = f"{symbol}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store the combined signal
        self.combined_signals[key] = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "metadata": metadata,
            "processed": False
        }
        
        logger.info(f"Created combined signal: {symbol} {direction}")
        
        # Publish a trade signal event
        event_data = {
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "metadata": metadata,
            "source": "signal_pattern_integration"
        }
        
        # Add stop loss and take profit if available
        if "stop_loss" in metadata:
            event_data["stop_loss"] = metadata["stop_loss"]
        
        if "take_profit" in metadata:
            event_data["take_profit"] = metadata["take_profit"]
        
        # Create and publish the event
        event = Event(
            event_type=EventType.TRADE_SIGNAL,
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _handle_trade_signal(self, event: Event):
        """
        Handle a trade signal event.
        
        This is where you would integrate with your trading system to execute trades.
        
        Args:
            event: Trade signal event
        """
        signal_data = event.data
        symbol = signal_data.get("symbol")
        direction = signal_data.get("direction")
        price = signal_data.get("price")
        source = signal_data.get("source")
        
        logger.info(f"Trade signal: {symbol} {direction} at {price} from {source}")
        
        # Here you would integrate with your trading system
        # For example:
        # self.trading_system.execute_trade(signal_data)
        
        # For this example, we'll just log the signal
        logger.info(f"Would execute trade: {symbol} {direction} at {price}")
    
    def start(self):
        """Start the integration services."""
        logger.info("Starting Pattern-Signal Integration")
        
        # Start webhook handler
        self.webhook_handler.start()
        
        logger.info(f"Webhook handler started on port {self.webhook_handler.port}, path /{self.webhook_handler.path}")
        
        # Initialize Alpaca and Finnhub if configured
        alpaca_config = self.config.get("alpaca", {})
        if alpaca_config.get("enabled", False):
            self._setup_alpaca(alpaca_config)
        
        finnhub_config = self.config.get("finnhub", {})
        if finnhub_config.get("enabled", False):
            self._setup_finnhub(finnhub_config)
        
        logger.info("Pattern-Signal Integration started")
    
    def _setup_alpaca(self, config: Dict[str, Any]):
        """
        Setup Alpaca integration.
        
        Args:
            config: Alpaca configuration
        """
        self.external_signal_strategy.update_source_config(
            SignalSource.ALPACA,
            {
                "enabled": True,
                "api_key": config.get("api_key", ""),
                "secret_key": config.get("secret_key", ""),
                "paper": config.get("paper", True),
                "auto_trade": False,  # We handle trades through our integration
                "signal_types": config.get("signal_types", ["trade_updates"])
            }
        )
        
        # Initialize the source
        if self.external_signal_strategy.initialize_source(SignalSource.ALPACA):
            logger.info("Alpaca integration initialized")
        else:
            logger.error("Failed to initialize Alpaca integration")
    
    def _setup_finnhub(self, config: Dict[str, Any]):
        """
        Setup Finnhub integration.
        
        Args:
            config: Finnhub configuration
        """
        self.external_signal_strategy.update_source_config(
            SignalSource.FINNHUB,
            {
                "enabled": True,
                "api_key": config.get("api_key", ""),
                "auto_trade": False,  # We handle trades through our integration
                "symbols": config.get("symbols", ["AAPL", "MSFT", "AMZN"]),
                "signal_types": config.get("signal_types", ["trade"])
            }
        )
        
        # Initialize the source
        if self.external_signal_strategy.initialize_source(SignalSource.FINNHUB):
            logger.info("Finnhub integration initialized")
        else:
            logger.error("Failed to initialize Finnhub integration")
    
    def stop(self):
        """Stop the integration services."""
        logger.info("Stopping Pattern-Signal Integration")
        
        # Stop webhook handler
        self.webhook_handler.stop()
        
        # Disconnect from external sources
        for source in [SignalSource.ALPACA, SignalSource.FINNHUB]:
            if self.external_signal_strategy.get_source_enabled(source):
                self.external_signal_strategy.disconnect_source(source)
        
        logger.info("Pattern-Signal Integration stopped")


def main():
    """Run the pattern signal integration example."""
    # Example configuration
    config = {
        "webhook_port": 5000,
        "webhook_path": "webhook",
        "webhook_auth_token": None,  # Set to a string for authentication
        
        "pattern_strategy_config": {
            "confidence_threshold": 0.7,
            "lookback_periods": 20,
            "confirmation_required": True
        },
        
        "alpaca": {
            "enabled": False,  # Set to True to enable
            "api_key": "",  # Your Alpaca API key
            "secret_key": "",  # Your Alpaca secret key
            "paper": True,
            "signal_types": ["trade_updates"]
        },
        
        "finnhub": {
            "enabled": False,  # Set to True to enable
            "api_key": "",  # Your Finnhub API key
            "symbols": ["AAPL", "MSFT", "AMZN", "EURUSD", "GBPUSD"],
            "signal_types": ["trade"]
        }
    }
    
    # Create the integration
    integration = PatternSignalIntegration(config)
    
    try:
        # Start the integration
        integration.start()
        
        # Keep running until keyboard interrupt
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Stop the integration
        integration.stop()


if __name__ == "__main__":
    main()
