#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Signal Strategy - Converts webhook signals into trading decisions.

This strategy listens for external signals via webhooks (e.g., from TradingView, 
custom scripts, other APIs) and converts them into trading signals for the bot.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from trading_bot.core.constants import EventType
from trading_bot.core.interfaces import StrategyInterface
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.interfaces import WebhookInterface
from trading_bot.strategies.base.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Enumeration of possible signal sources."""
    TRADINGVIEW = "tradingview"
    API = "api"
    CUSTOM_SCRIPT = "custom_script"
    ALPACA = "alpaca"
    FINNHUB = "finnhub"
    UNKNOWN = "unknown"


class SignalType(Enum):
    """Enumeration of possible signal types."""
    ENTRY = "entry"
    EXIT = "exit"
    POSITION_SIZE = "position_size"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PARAMETER_UPDATE = "parameter_update"
    INDICATOR_UPDATE = "indicator_update"
    STRATEGY_UPDATE = "strategy_update"
    UNKNOWN = "unknown"


class Direction(Enum):
    """Enumeration of possible trade directions."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    UNKNOWN = "unknown"


class ExternalSignal:
    """
    Represents a trading signal received from an external source.
    """
    
    def __init__(
        self,
        symbol: str,
        source: SignalSource,
        signal_type: SignalType,
        direction: Direction = Direction.UNKNOWN,
        timestamp: Optional[datetime] = None,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_payload: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new external signal.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD", "BTCUSD")
            source: Source of the signal
            signal_type: Type of signal
            direction: Direction of the trade (if applicable)
            timestamp: When the signal was generated
            price: Price at which the signal was generated (if applicable)
            metadata: Additional metadata about the signal
            raw_payload: The raw webhook payload that generated this signal
        """
        self.symbol = symbol
        self.source = source
        self.signal_type = signal_type
        self.direction = direction
        self.timestamp = timestamp or datetime.now()
        self.price = price
        self.metadata = metadata or {}
        self.raw_payload = raw_payload or {}
        self.processed = False
        self.result = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the signal to a dictionary representation."""
        return {
            "symbol": self.symbol,
            "source": self.source.value,
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "metadata": self.metadata,
            "processed": self.processed,
            "result": self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExternalSignal':
        """Create a signal from a dictionary representation."""
        return cls(
            symbol=data["symbol"],
            source=SignalSource(data["source"]),
            signal_type=SignalType(data["signal_type"]),
            direction=Direction(data["direction"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price=data.get("price"),
            metadata=data.get("metadata", {}),
            raw_payload=data.get("raw_payload", {})
        )
    
    @classmethod
    def from_webhook_payload(cls, payload: Dict[str, Any]) -> 'ExternalSignal':
        """
        Create a signal from a webhook payload.
        
        This method attempts to interpret the webhook payload and extract
        the necessary information to create a signal.
        
        Args:
            payload: The webhook payload
            
        Returns:
            An ExternalSignal instance
        """
        # Default values
        symbol = payload.get("symbol", "unknown")
        signal_type = SignalType.UNKNOWN
        direction = Direction.UNKNOWN
        price = None
        source = SignalSource.UNKNOWN
        metadata = {}
        
        # Try to determine the source
        if "source" in payload:
            # Explicit source in payload
            source_val = payload["source"].lower()
            if source_val == "tradingview":
                source = SignalSource.TRADINGVIEW
            elif source_val == "api":
                source = SignalSource.API
            elif source_val == "custom_script":
                source = SignalSource.CUSTOM_SCRIPT
        else:
            # Try to guess the source based on payload format
            if "strategy" in payload and "ticker" in payload:
                # Likely TradingView
                source = SignalSource.TRADINGVIEW
                # TradingView often uses "ticker" instead of "symbol"
                if "ticker" in payload and not symbol or symbol == "unknown":
                    symbol = payload["ticker"]
        
        # Try to determine the signal type and direction
        if "action" in payload:
            action = payload["action"].lower()
            
            # Handle common action values
            if action in ["buy", "long", "enter_long"]:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG
            elif action in ["sell", "short", "enter_short"]:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT
            elif action in ["exit", "close", "exit_position", "close_position"]:
                signal_type = SignalType.EXIT
                direction = Direction.FLAT
            elif action in ["stop_loss", "sl"]:
                signal_type = SignalType.STOP_LOSS
            elif action in ["take_profit", "tp"]:
                signal_type = SignalType.TAKE_PROFIT
            elif action in ["update_parameters", "params"]:
                signal_type = SignalType.PARAMETER_UPDATE
        
        # Extract price if available
        if "price" in payload:
            price = float(payload["price"])
        elif "close" in payload:
            price = float(payload["close"])
        
        # Extract additional metadata
        metadata_fields = [
            "timeframe", "volume", "asset_type", "strategy", "indicators", 
            "confidence", "risk_reward", "stop_loss", "take_profit"
        ]
        
        for field in metadata_fields:
            if field in payload:
                metadata[field] = payload[field]
        
        # Create and return the signal
        return cls(
            symbol=symbol,
            source=source,
            signal_type=signal_type,
            direction=direction,
            timestamp=datetime.now(),
            price=price,
            metadata=metadata,
            raw_payload=payload
        )
    
    def __str__(self) -> str:
        """String representation of the signal."""
        direction_str = f", {self.direction.value}" if self.direction != Direction.UNKNOWN else ""
        price_str = f" at {self.price}" if self.price else ""
        return f"{self.symbol} {self.signal_type.value}{direction_str}{price_str} from {self.source.value}"


class ExternalSignalStrategy(StrategyBase):
    """
    Strategy that listens for and acts on external signals from webhooks.
    
    This strategy:
    1. Registers with the webhook handler to receive signals
    2. Converts webhook payloads to standardized ExternalSignal objects
    3. Publishes events to the EventBus for other components to react to
    4. Optionally executes trades directly based on the signals
    5. Maintains a history of received signals
    """
    
    def __init__(
        self,
        name: str = "external_signal_strategy",
        max_history: int = 100,
        auto_trade: bool = False,
        signal_filters: Optional[Dict[str, Any]] = None,
        register_webhook: bool = True,
        source_configs: Optional[Dict[SignalSource, Dict[str, Any]]] = None
    ):
        """
        Initialize the external signal strategy.
        
        Args:
            name: Name of the strategy
            max_history: Maximum number of signals to keep in history
            auto_trade: Whether to automatically generate trades from signals
            signal_filters: Filters to apply to incoming signals
            register_webhook: Whether to automatically register with the webhook handler
        """
        super().__init__(name=name)
        self.signals: List[ExternalSignal] = []
        self.max_history = max_history
        self.auto_trade = auto_trade
        self.signal_filters = signal_filters or {}
        
        # Configure data sources
        self.source_configs = source_configs or self._default_source_configs()
        self.source_connections = {}
        
        # Get event bus
        self.event_bus = EventBus()
        
        # Register with webhook handler if requested
        if register_webhook:
            self._register_webhook_handler()
    
    def _register_webhook_handler(self) -> None:
        """Register with the webhook handler to receive signals."""
        try:
            # Try to get the webhook handler from the service registry
            webhook_handler = ServiceRegistry.get_service('webhook_handler')
            
            if webhook_handler and isinstance(webhook_handler, WebhookInterface):
                webhook_handler.register_handler(self.handle_webhook)
                logger.info(f"ExternalSignalStrategy registered with webhook handler")
            else:
                logger.warning("Webhook handler not found or not of correct type")
        except Exception as e:
            logger.error(f"Error registering with webhook handler: {str(e)}")
    
    def handle_webhook(self, payload: Dict[str, Any]) -> None:
        """
        Handle a webhook payload.
        
        This method is called by the webhook handler when a new webhook is received.
        
        Args:
            payload: The webhook payload
        """
        try:
            # Convert webhook payload to external signal
            signal = ExternalSignal.from_webhook_payload(payload)
            
            # Apply filters
            if not self._apply_filters(signal):
                logger.info(f"Signal filtered out: {signal}")
                return
            
            # Add to history
            self._add_signal(signal)
            
            # Publish event
            self._publish_signal_event(signal)
            
            # Automatically generate trade if enabled
            if self.auto_trade:
                self._generate_trade(signal)
            
            logger.info(f"Processed external signal: {signal}")
            
        except Exception as e:
            logger.error(f"Error processing webhook payload: {str(e)}")
    
    def _apply_filters(self, signal: ExternalSignal) -> bool:
        """
        Apply filters to the signal to determine if it should be processed.
        
        Args:
            signal: The signal to filter
            
        Returns:
            True if the signal passes filters, False otherwise
        """
        # Symbol filter
        if "symbols" in self.signal_filters:
            allowed_symbols = self.signal_filters["symbols"]
            if signal.symbol not in allowed_symbols:
                return False
        
        # Source filter
        if "sources" in self.signal_filters:
            allowed_sources = [SignalSource(s) for s in self.signal_filters["sources"]]
            if signal.source not in allowed_sources:
                return False
        
        # Signal type filter
        if "signal_types" in self.signal_filters:
            allowed_types = [SignalType(t) for t in self.signal_filters["signal_types"]]
            if signal.signal_type not in allowed_types:
                return False
        
        # Direction filter
        if "directions" in self.signal_filters:
            allowed_directions = [Direction(d) for d in self.signal_filters["directions"]]
            if signal.direction not in allowed_directions:
                return False
        
        return True
    
    def _add_signal(self, signal: ExternalSignal) -> None:
        """
        Add a signal to the history.
        
        Args:
            signal: The signal to add
        """
        self.signals.append(signal)
        
        # Trim history if needed
        if len(self.signals) > self.max_history:
            self.signals = self.signals[-self.max_history:]
    
    def _publish_signal_event(self, signal: ExternalSignal) -> None:
        """
        Publish a signal event to the event bus.
        
        Args:
            signal: The signal to publish
        """
        event_data = {
            "signal": signal.to_dict(),
            "strategy": self.name
        }
        
        # Determine the event type based on signal type
        event_type = EventType.EXTERNAL_SIGNAL
        if signal.signal_type == SignalType.ENTRY:
            event_type = EventType.SIGNAL_GENERATED
        elif signal.signal_type == SignalType.EXIT:
            event_type = EventType.EXIT_SIGNAL
        
        # Create and publish the event
        event = Event(
            event_type=event_type,
            data=event_data
        )
        self.event_bus.publish(event)
    
    def _generate_trade(self, signal: ExternalSignal) -> None:
        """
        Generate a trade based on the signal.
        
        Args:
            signal: The signal to generate a trade from
        """
        # Only handle entry and exit signals
        if signal.signal_type not in [SignalType.ENTRY, SignalType.EXIT]:
            return
        
        # Create trade parameters
        trade_params = {
            "symbol": signal.symbol,
            "direction": signal.direction.value if signal.direction != Direction.UNKNOWN else None,
            "price": signal.price,
            "signal_source": signal.source.value,
            "strategy": self.name
        }
        
        # Add stop loss and take profit if available in metadata
        if "stop_loss" in signal.metadata:
            trade_params["stop_loss"] = signal.metadata["stop_loss"]
        
        if "take_profit" in signal.metadata:
            trade_params["take_profit"] = signal.metadata["take_profit"]
        
        # Create and publish the appropriate event
        if signal.signal_type == SignalType.ENTRY:
            event = Event(
                event_type=EventType.TRADE_SIGNAL,
                data=trade_params
            )
        else:  # EXIT
            event = Event(
                event_type=EventType.EXIT_POSITION,
                data=trade_params
            )
        
        self.event_bus.publish(event)
        
        # Mark signal as processed
        signal.processed = True
        signal.result = "trade_generated"
    
    def get_signals(
        self, 
        symbol: Optional[str] = None, 
        source: Optional[SignalSource] = None,
        signal_type: Optional[SignalType] = None,
        limit: int = 10
    ) -> List[ExternalSignal]:
        """
        Get signals from history with optional filtering.
        
        Args:
            symbol: Filter by symbol
            source: Filter by source
            signal_type: Filter by signal type
            limit: Maximum number of signals to return
            
        Returns:
            List of signals matching the filters
        """
        filtered_signals = self.signals
        
        if symbol:
            filtered_signals = [s for s in filtered_signals if s.symbol == symbol]
        
        if source:
            filtered_signals = [s for s in filtered_signals if s.source == source]
        
        if signal_type:
            filtered_signals = [s for s in filtered_signals if s.signal_type == signal_type]
        
        # Return most recent signals first
        return sorted(
            filtered_signals, 
            key=lambda x: x.timestamp, 
            reverse=True
        )[:limit]
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the signals received.
        
        Returns:
            Dictionary with signal statistics
        """
        stats = {
            "total_signals": len(self.signals),
            "by_source": {},
            "by_type": {},
            "by_symbol": {},
            "processed_count": len([s for s in self.signals if s.processed])
        }
        
        # Count by source
        for source in SignalSource:
            count = len([s for s in self.signals if s.source == source])
            if count > 0:
                stats["by_source"][source.value] = count
        
        # Count by type
        for signal_type in SignalType:
            count = len([s for s in self.signals if s.signal_type == signal_type])
            if count > 0:
                stats["by_type"][signal_type.value] = count
        
        # Count by symbol
        symbols = set(s.symbol for s in self.signals)
        for symbol in symbols:
            count = len([s for s in self.signals if s.symbol == symbol])
            stats["by_symbol"][symbol] = count
        
        return stats
    
    def clear_signals(self) -> None:
        """Clear the signal history."""
        self.signals = []
        logger.info("Signal history cleared")
    
    def generate_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        direction: Direction,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExternalSignal:
        """
        Manually generate a signal (for testing or internal use).
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal
            direction: Direction of the trade
            price: Price at the time of signal
            metadata: Additional metadata
            
        Returns:
            The generated signal
        """
        signal = ExternalSignal(
            symbol=symbol,
            source=SignalSource.API,  # Mark as API since it's generated internally
            signal_type=signal_type,
            direction=direction,
            timestamp=datetime.now(),
            price=price,
            metadata=metadata or {}
        )
        
        # Add to history and process
        self._add_signal(signal)
        self._publish_signal_event(signal)
        
        if self.auto_trade:
            self._generate_trade(signal)
        
        return signal

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        if "auto_trade" in params:
            self.auto_trade = bool(params["auto_trade"])
        
        if "max_history" in params:
            self.max_history = int(params["max_history"])
        
        if "signal_filters" in params:
            self.signal_filters = params["signal_filters"]
        
        # Update source configs if provided
        if "source_configs" in params:
            for source_name, config in params["source_configs"].items():
                try:
                    source = SignalSource(source_name)
                    self.update_source_config(source, config)
                except ValueError:
                    logger.warning(f"Unknown signal source: {source_name}")
        
        logger.info(f"ExternalSignalStrategy parameters updated: {params}")
    
    def process_tradingview_data(self, market_data: Dict[str, Any]) -> None:
        """
        Process TradingView data (for integration with strategy rotator).
        
        Args:
            market_data: Market data from TradingView
        """
        # Create a signal based on the TradingView data
        symbol = market_data.get("symbol", "unknown")
        
        # Try to determine if this is an entry or exit signal
        signal_type = SignalType.INDICATOR_UPDATE  # Default to indicator update
        direction = Direction.UNKNOWN
        
        # Look for action in the data
        if "action" in market_data:
            action = market_data["action"].lower()
            if action in ["buy", "long"]:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG
            elif action in ["sell", "short"]:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT
            elif action in ["exit", "close"]:
                signal_type = SignalType.EXIT
                direction = Direction.FLAT
        
        # Create the signal
        signal = ExternalSignal(
            symbol=symbol,
            source=SignalSource.TRADINGVIEW,
            signal_type=signal_type,
            direction=direction,
            timestamp=datetime.now(),
            price=market_data.get("price"),
            metadata=market_data,
            raw_payload=market_data
        )
        
        # Add to history and process like a regular signal
        self._add_signal(signal)
        self._publish_signal_event(signal)
        
        # Generate trade if auto-trade is enabled
        if self.auto_trade:
            self._generate_trade(signal)
            
        return signal
    
    def _default_source_configs(self) -> Dict[SignalSource, Dict[str, Any]]:
        """
        Create default configurations for all supported signal sources.
        
        Returns:
            Dictionary of default configurations by source
        """
        return {
            SignalSource.TRADINGVIEW: {
                "enabled": True,
                "auto_trade": self.auto_trade,
            },
            SignalSource.ALPACA: {
                "enabled": False,
                "api_key": "",
                "secret_key": "",
                "paper": True,
                "auto_trade": False,
                "signal_types": ["trade_updates"]
            },
            SignalSource.FINNHUB: {
                "enabled": False,
                "api_key": "",
                "auto_trade": False,
                "symbols": ["AAPL", "MSFT", "AMZN"],
                "signal_types": ["trade"]
            },
            SignalSource.API: {
                "enabled": True,
                "auto_trade": self.auto_trade,
            },
            SignalSource.CUSTOM_SCRIPT: {
                "enabled": True,
                "auto_trade": self.auto_trade,
            }
        }
    
    def get_source_config(self, source: SignalSource) -> Dict[str, Any]:
        """
        Get the configuration for a specific signal source.
        
        Args:
            source: The signal source
            
        Returns:
            Configuration dictionary for the source
        """
        if source not in self.source_configs:
            return {}
            
        return self.source_configs[source]
    
    def get_source_enabled(self, source: SignalSource) -> bool:
        """
        Check if a signal source is enabled.
        
        Args:
            source: The signal source
            
        Returns:
            True if the source is enabled, False otherwise
        """
        config = self.get_source_config(source)
        return config.get("enabled", False)
    
    def update_source_config(self, source: SignalSource, config: Dict[str, Any]) -> None:
        """
        Update the configuration for a signal source.
        
        Args:
            source: The signal source
            config: New configuration
        """
        # Get current config
        current_config = self.get_source_config(source).copy()
        
        # Update with new values
        current_config.update(config)
        
        # Store updated config
        self.source_configs[source] = current_config
        
        # If the source was enabled or disabled, initialize or disconnect it
        was_enabled = self.source_configs[source].get("enabled", False)
        is_enabled = config.get("enabled", was_enabled)
        
        if was_enabled != is_enabled:
            if is_enabled:
                self.initialize_source(source)
            else:
                self.disconnect_source(source)
        
        logger.info(f"Updated configuration for source {source.value}")
    
    def initialize_source(self, source: SignalSource) -> bool:
        """
        Initialize a connection to an external signal source.
        
        Args:
            source: The signal source to initialize
            
        Returns:
            True if initialization was successful, False otherwise
        """
        # Check if source is enabled
        if not self.get_source_enabled(source):
            logger.warning(f"Cannot initialize source {source.value} because it is disabled")
            return False
            
        # Already connected
        if source in self.source_connections and self.source_connections[source]:
            logger.info(f"Source {source.value} is already initialized")
            return True
            
        # Handle different source types
        try:
            if source == SignalSource.ALPACA:
                return self._initialize_alpaca()
            elif source == SignalSource.FINNHUB:
                return self._initialize_finnhub()
            else:
                # Other sources don't need special initialization
                self.source_connections[source] = True
                return True
        except Exception as e:
            logger.error(f"Error initializing source {source.value}: {str(e)}")
            return False
    
    def disconnect_source(self, source: SignalSource) -> bool:
        """
        Disconnect from an external signal source.
        
        Args:
            source: The signal source to disconnect
            
        Returns:
            True if disconnection was successful, False otherwise
        """
        if source not in self.source_connections or not self.source_connections[source]:
            logger.info(f"Source {source.value} is not connected")
            return True
            
        try:
            if source == SignalSource.ALPACA:
                return self._disconnect_alpaca()
            elif source == SignalSource.FINNHUB:
                return self._disconnect_finnhub()
            else:
                # Other sources don't need special disconnection
                self.source_connections[source] = False
                return True
        except Exception as e:
            logger.error(f"Error disconnecting source {source.value}: {str(e)}")
            return False
    
    def _initialize_alpaca(self) -> bool:
        """
        Initialize connection to Alpaca API for trade updates.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            import alpaca_trade_api as tradeapi
            from alpaca_trade_api.stream import Stream
            
            # Get Alpaca config
            config = self.get_source_config(SignalSource.ALPACA)
            api_key = config.get("api_key")
            api_secret = config.get("secret_key")
            paper = config.get("paper", True)
            
            if not api_key or not api_secret:
                logger.error("Alpaca API key or secret key not configured")
                return False
                
            # Setup API connection
            base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
            
            # Initialize Alpaca API client
            api = tradeapi.REST(api_key, api_secret, base_url=base_url)
            
            # Test the connection
            account = api.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
            
            # Initialize Stream connection for trade updates
            stream = Stream(api_key, api_secret, base_url=base_url, data_feed='iex')
            
            # Register handlers for trade updates and other events
            if "trade_updates" in config.get("signal_types", []):
                stream.subscribe_trade_updates(self._handle_alpaca_trade_update)
                
            # Start stream in a separate thread
            stream.start()
            
            # Store the API and stream objects
            self.source_connections[SignalSource.ALPACA] = {
                "api": api,
                "stream": stream
            }
            
            logger.info("Successfully initialized Alpaca API connection")
            return True
            
        except ImportError:
            logger.error("Could not import alpaca_trade_api. Please install it with: pip install alpaca-trade-api")
            return False
        except Exception as e:
            logger.error(f"Error initializing Alpaca connection: {str(e)}")
            return False
    
    def _disconnect_alpaca(self) -> bool:
        """
        Disconnect from Alpaca API.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        try:
            if SignalSource.ALPACA in self.source_connections:
                connection = self.source_connections[SignalSource.ALPACA]
                if connection and "stream" in connection:
                    connection["stream"].stop()
                    logger.info("Disconnected from Alpaca stream")
                    
                self.source_connections[SignalSource.ALPACA] = None
                return True
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {str(e)}")
            return False
    
    def _handle_alpaca_trade_update(self, trade_update):
        """
        Handle a trade update event from Alpaca.
        
        Args:
            trade_update: Trade update event data from Alpaca
        """
        try:
            # Extract relevant information from the trade update
            data = trade_update.__dict__
            order = data.get("order")
            
            if not order:
                logger.warning("Received trade update with no order data")
                return
                
            # Create signal from trade update
            symbol = order.get("symbol")
            side = order.get("side")
            status = order.get("status")
            
            # Determine signal type and direction
            signal_type = SignalType.UNKNOWN
            direction = Direction.UNKNOWN
            
            if status == "filled":
                if side == "buy":
                    signal_type = SignalType.ENTRY
                    direction = Direction.LONG
                elif side == "sell":
                    # Could be either an exit or a short entry
                    signal_type = SignalType.ENTRY if order.get("position_effect") == "open" else SignalType.EXIT
                    direction = Direction.SHORT if signal_type == SignalType.ENTRY else Direction.FLAT
            elif status == "canceled" or status == "rejected":
                signal_type = SignalType.INDICATOR_UPDATE
            
            # Create and process the signal
            signal = ExternalSignal(
                symbol=symbol,
                source=SignalSource.ALPACA,
                signal_type=signal_type,
                direction=direction,
                timestamp=datetime.now(),
                price=float(order.get("filled_avg_price", 0)) or None,
                metadata={
                    "status": status,
                    "order_id": order.get("id"),
                    "client_order_id": order.get("client_order_id"),
                    "order_type": order.get("type"),
                    "qty": order.get("qty"),
                    "filled_qty": order.get("filled_qty")
                },
                raw_payload=data
            )
            
            # Add to history and process
            self._add_signal(signal)
            self._publish_signal_event(signal)
            
            # Generate trade if auto-trade is enabled and this is from Alpaca
            config = self.get_source_config(SignalSource.ALPACA)
            if config.get("auto_trade", False):
                self._generate_trade(signal)
                
            logger.info(f"Processed Alpaca trade update: {signal}")
            
        except Exception as e:
            logger.error(f"Error handling Alpaca trade update: {str(e)}")
    
    def _initialize_finnhub(self) -> bool:
        """
        Initialize connection to Finnhub API for market data and signals.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            import finnhub
            from websocket import WebSocketApp
            import threading
            import json
            
            # Get Finnhub config
            config = self.get_source_config(SignalSource.FINNHUB)
            api_key = config.get("api_key")
            symbols = config.get("symbols", [])
            
            if not api_key:
                logger.error("Finnhub API key not configured")
                return False
                
            if not symbols:
                logger.warning("No symbols configured for Finnhub streaming")
                
            # Initialize Finnhub client
            client = finnhub.Client(api_key=api_key)
            
            # Test the connection with a simple API call
            try:
                client.symbol_lookup("AAPL")
                logger.info("Successfully connected to Finnhub API")
            except Exception as e:
                logger.error(f"Failed to connect to Finnhub API: {str(e)}")
                return False
            
            # Setup WebSocket connection for real-time data
            self._finnhub_setup_websocket(api_key, symbols)
            
            # Store the client object
            self.source_connections[SignalSource.FINNHUB] = {
                "client": client,
                "websocket": self._finnhub_websocket
            }
            
            return True
            
        except ImportError:
            logger.error("Could not import finnhub-python. Please install it with: pip install finnhub-python websocket-client")
            return False
        except Exception as e:
            logger.error(f"Error initializing Finnhub connection: {str(e)}")
            return False
    
    def _finnhub_setup_websocket(self, api_key: str, symbols: List[str]) -> None:
        """
        Setup Finnhub WebSocket connection for real-time data.
        
        Args:
            api_key: Finnhub API key
            symbols: List of symbols to subscribe to
        """
        import threading
        import json
        from websocket import WebSocketApp
        
        # Define WebSocket callbacks
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                if data["type"] == "trade":
                    for trade in data["data"]:
                        self._handle_finnhub_trade(trade)
                        
            except Exception as e:
                logger.error(f"Error processing Finnhub WebSocket message: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Finnhub WebSocket error: {str(error)}")
            
        def on_close(ws, close_status_code, close_msg):
            logger.info("Finnhub WebSocket connection closed")
            
        def on_open(ws):
            logger.info("Finnhub WebSocket connection opened")
            
            # Subscribe to symbols
            for symbol in symbols:
                ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                logger.info(f"Subscribed to Finnhub data for {symbol}")
        
        # Create WebSocket connection
        websocket_url = f"wss://ws.finnhub.io?token={api_key}"
        self._finnhub_websocket = WebSocketApp(
            websocket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in a separate thread
        self._finnhub_thread = threading.Thread(target=self._finnhub_websocket.run_forever)
        self._finnhub_thread.daemon = True
        self._finnhub_thread.start()
    
    def _disconnect_finnhub(self) -> bool:
        """
        Disconnect from Finnhub API.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        try:
            if SignalSource.FINNHUB in self.source_connections:
                connection = self.source_connections[SignalSource.FINNHUB]
                if connection and "websocket" in connection:
                    connection["websocket"].close()
                    logger.info("Disconnected from Finnhub WebSocket")
                    
                self.source_connections[SignalSource.FINNHUB] = None
                return True
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Finnhub: {str(e)}")
            return False
    
    def _handle_finnhub_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Handle a trade event from Finnhub.
        
        Args:
            trade_data: Trade data from Finnhub
        """
        try:
            # Extract trade information
            symbol = trade_data.get("s")  # Symbol
            price = trade_data.get("p")   # Price
            volume = trade_data.get("v")  # Volume
            timestamp = trade_data.get("t")  # Timestamp
            
            if not symbol or not price:
                return
                
            # Get signal types to process from config
            config = self.get_source_config(SignalSource.FINNHUB)
            signal_types = config.get("signal_types", [])
            
            # Only process trade signals if configured
            if "trade" not in signal_types:
                return
                
            # Create timestamp from milliseconds
            signal_time = datetime.fromtimestamp(timestamp / 1000) if timestamp else datetime.now()
            
            # Create and process signal
            signal = ExternalSignal(
                symbol=symbol,
                source=SignalSource.FINNHUB,
                signal_type=SignalType.INDICATOR_UPDATE,  # Trade data is treated as an indicator update
                direction=Direction.UNKNOWN,  # Cannot determine direction from trade data alone
                timestamp=signal_time,
                price=price,
                metadata={
                    "volume": volume,
                    "trade_timestamp": timestamp
                },
                raw_payload=trade_data
            )
            
            # Add to history and process
            self._add_signal(signal)
            self._publish_signal_event(signal)
            
            # We typically don't auto-trade on raw trade data, but check config
            if config.get("auto_trade", False) and "trade" in signal_types:
                # Here we could implement logic to analyze the trade and potentially generate signals
                # For now, we just log it
                logger.info(f"Received Finnhub trade for {symbol} at {price}")
                
        except Exception as e:
            logger.error(f"Error handling Finnhub trade: {str(e)}")
            
    def process_indicator_data(self, data: Dict[str, Any], source: SignalSource = SignalSource.API) -> ExternalSignal:
        """
        Process indicator data from any source and create a signal.
        
        Args:
            data: Indicator data dictionary containing at minimum 'symbol' and 'indicators'
            source: Source of the indicator data
            
        Returns:
            The created signal
        """
        symbol = data.get("symbol")
        if not symbol:
            logger.warning("Indicator data missing symbol")
            return None
            
        # Create metadata from all available fields
        metadata = {}
        for key, value in data.items():
            if key != "symbol":
                metadata[key] = value
                
        # Create the signal
        signal = ExternalSignal(
            symbol=symbol,
            source=source,
            signal_type=SignalType.INDICATOR_UPDATE,
            direction=Direction.UNKNOWN,
            timestamp=datetime.now(),
            price=data.get("price"),
            metadata=metadata,
            raw_payload=data
        )
        
        # Add to history and process
        self._add_signal(signal)
        self._publish_signal_event(signal)
            
        return signal
        
        if self.auto_trade and signal_type in [SignalType.ENTRY, SignalType.EXIT]:
            self._generate_trade(signal)

# Register the strategy with the ServiceRegistry
ServiceRegistry.register('external_signal_strategy', ExternalSignalStrategy())
