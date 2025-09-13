#!/usr/bin/env python3
"""
Real-Time Data Processor

This module provides classes and functions for streaming and processing
market data in real-time for live trading applications.
"""

import os
import time
import json
import logging
import threading
import queue
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from datetime import datetime, timedelta
import websocket
import requests
from urllib.parse import urlencode

# Import local modules
from trading_bot.optimization.advanced_market_regime_detector import AdvancedMarketRegimeDetector
from trading_bot.optimization.strategy_regime_rotator import StrategyRegimeRotator

# Set up logging
logger = logging.getLogger(__name__)

class MarketDataSource(ABC):
    """
    Abstract base class for market data sources.
    
    This class defines the interface that all data sources must implement.
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market data source.
        
        Args:
            symbols: List of symbols to stream data for
            config: Optional configuration dictionary
        """
        self.symbols = symbols
        self.config = config or {}
        self.callbacks = []
        self.is_running = False
        self.last_update_time = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.config.get('max_reconnect_attempts', 5)
        self.reconnect_delay = self.config.get('reconnect_delay', 5)
        
        # Set up logging for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to be called when new data is received.
        
        Args:
            callback: Function that takes a data dictionary as input
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister a previously registered callback function.
        
        Args:
            callback: Previously registered callback function
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process received data and call registered callbacks.
        
        Args:
            data: Dictionary containing market data
        """
        self.last_update_time = datetime.now()
        
        # Call all registered callbacks with the data
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback: {str(e)}")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            bool: Whether connection was successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    def start_streaming(self) -> None:
        """Start streaming data."""
        pass
    
    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop streaming data."""
        pass
    
    def is_connected(self) -> bool:
        """
        Check if the data source is currently connected.
        
        Returns:
            bool: Whether the data source is connected
        """
        # Default implementation based on is_running flag and recent updates
        if not self.is_running:
            return False
        
        if self.last_update_time is None:
            return False
        
        # Check if we've received data recently (within the last minute)
        time_since_last_update = datetime.now() - self.last_update_time
        return time_since_last_update < timedelta(minutes=1)
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the data source.
        
        Returns:
            bool: Whether reconnection was successful
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        self.reconnect_attempts += 1
        self.logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        # Disconnect first (if still connected)
        try:
            self.disconnect()
        except Exception as e:
            self.logger.warning(f"Error during disconnect before reconnect: {str(e)}")
        
        # Wait before reconnecting
        time.sleep(self.reconnect_delay)
        
        # Try to connect again
        try:
            success = self.connect()
            if success:
                self.reconnect_attempts = 0  # Reset counter on successful reconnect
                self.start_streaming()
                return True
            else:
                self.logger.error("Reconnection attempt failed")
                return False
        except Exception as e:
            self.logger.error(f"Error during reconnection: {str(e)}")
            return False


class AlpacaDataSource(MarketDataSource):
    """
    Market data source implementation for Alpaca API.
    
    Streams real-time market data from Alpaca's API.
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Alpaca market data source.
        
        Args:
            symbols: List of symbols to stream data for
            config: Dictionary containing Alpaca API credentials and settings
        """
        super().__init__(symbols, config)
        
        # Alpaca-specific configuration
        self.api_key = self.config.get('api_key')
        self.api_secret = self.config.get('api_secret')
        self.base_url = self.config.get('base_url', 'https://paper-api.alpaca.markets')
        self.data_url = self.config.get('data_url', 'wss://stream.data.alpaca.markets/v2')
        self.data_feed = self.config.get('data_feed', 'iex')  # or 'sip' for paid subscription
        
        # Validate required configuration
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
    
    def connect(self) -> bool:
        """
        Connect to Alpaca WebSocket API.
        
        Returns:
            bool: Whether connection was successful
        """
        # Define WebSocket callbacks
        def on_open(ws):
            self.logger.info("Connected to Alpaca WebSocket")
            self.ws_connected = True
            
            # Authentication message
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            ws.send(json.dumps(auth_msg))
            
            # Subscribe to market data
            subscribe_msg = {
                "action": "subscribe",
                "bars": self.symbols,
                "quotes": self.symbols,
                "trades": self.symbols
            }
            ws.send(json.dumps(subscribe_msg))
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.process_data(data)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to decode message: {message}")
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {str(error)}")
            self.ws_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
            self.ws_connected = False
        
        # Create WebSocket connection
        try:
            self.ws = websocket.WebSocketApp(
                self.data_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to create WebSocket connection: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Alpaca WebSocket API."""
        if self.ws:
            self.ws.close()
            self.ws = None
        
        self.ws_connected = False
        
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=1.0)
    
    def start_streaming(self) -> None:
        """Start streaming data from Alpaca."""
        if self.is_running:
            self.logger.warning("Streaming is already running")
            return
        
        if not self.ws:
            success = self.connect()
            if not success:
                self.logger.error("Failed to connect to Alpaca API")
                return
        
        # Start WebSocket connection in a separate thread
        self.is_running = True
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.logger.info("Started streaming data from Alpaca")
    
    def stop_streaming(self) -> None:
        """Stop streaming data from Alpaca."""
        self.is_running = False
        self.disconnect()
        self.logger.info("Stopped streaming data from Alpaca")
    
    def is_connected(self) -> bool:
        """
        Check if still connected to Alpaca.
        
        Returns:
            bool: Whether connection is active
        """
        return self.ws_connected and super().is_connected()
    
    def get_latest_bars(self, timeframe: str = '1Min', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Fetch the latest OHLCV bars for the subscribed symbols.
        
        Args:
            timeframe: Bar timeframe (e.g., '1Min', '5Min', '1D')
            limit: Maximum number of bars to retrieve
            
        Returns:
            Dict mapping symbol to DataFrame of OHLCV data
        """
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        # Use Alpaca V2 bars API
        base_url = 'https://data.alpaca.markets/v2'
        
        results = {}
        
        for symbol in self.symbols:
            params = {
                'symbols': symbol,
                'timeframe': timeframe,
                'limit': limit,
                'adjustment': 'raw'
            }
            
            url = f"{base_url}/stocks/bars?{urlencode(params)}"
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                if 'bars' in data and symbol in data['bars']:
                    bars = data['bars'][symbol]
                    df = pd.DataFrame(bars)
                    
                    # Convert timestamp to datetime
                    df['t'] = pd.to_datetime(df['t'])
                    
                    # Rename columns to standard OHLCV
                    df = df.rename(columns={
                        't': 'timestamp',
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    })
                    
                    # Set timestamp as index
                    df.set_index('timestamp', inplace=True)
                    
                    results[symbol] = df
            except Exception as e:
                self.logger.error(f"Error fetching bars for {symbol}: {str(e)}")
        
        return results


class IBDataSource(MarketDataSource):
    """
    Market data source implementation for Interactive Brokers.
    
    Streams real-time market data from Interactive Brokers TWS or IB Gateway.
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Interactive Brokers market data source.
        
        Args:
            symbols: List of symbols to stream data for
            config: Dictionary containing IB connection settings
        """
        super().__init__(symbols, config)
        
        # IB-specific configuration
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 7497)  # 7497 for TWS, 4002 for IB Gateway
        self.client_id = self.config.get('client_id', 1)
        
        # IB connection
        self.ib_connection = None
        self.market_data_queue = queue.Queue()
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway.
        
        Returns:
            bool: Whether connection was successful
        """
        try:
            # Note: This is a placeholder for the actual implementation
            # In a real implementation, you would use the ib_insync or ibapi libraries
            
            # Check if TWS/IBG is running on the specified host/port
            self.logger.info(f"Connecting to IB on {self.host}:{self.port} with client ID {self.client_id}")
            
            # Simulate connection (replace with actual IB API calls)
            # For example, using ib_insync:
            # from ib_insync import IB
            # self.ib_connection = IB()
            # self.ib_connection.connect(self.host, self.port, clientId=self.client_id)
            
            self.connected = True
            self.logger.info("Connected to Interactive Brokers")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to connect to Interactive Brokers: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib_connection:
            # Placeholder for actual disconnection logic
            # E.g.: self.ib_connection.disconnect()
            self.logger.info("Disconnected from Interactive Brokers")
        
        self.connected = False
    
    def start_streaming(self) -> None:
        """Start streaming data from Interactive Brokers."""
        if self.is_running:
            self.logger.warning("Streaming is already running")
            return
        
        if not self.connected:
            success = self.connect()
            if not success:
                self.logger.error("Failed to connect to Interactive Brokers")
                return
        
        # Request market data for all symbols
        for symbol in self.symbols:
            # Placeholder for actual market data subscription
            # E.g.: contract = Stock(symbol, 'SMART', 'USD')
            #       self.ib_connection.reqMktData(contract)
            self.logger.info(f"Subscribed to market data for {symbol}")
        
        self.is_running = True
        
        # Start processing thread
        threading.Thread(target=self._process_market_data, daemon=True).start()
        
        self.logger.info("Started streaming data from Interactive Brokers")
    
    def stop_streaming(self) -> None:
        """Stop streaming data from Interactive Brokers."""
        self.is_running = False
        
        # Cancel market data subscriptions
        for symbol in self.symbols:
            # Placeholder for actual cancellation
            # E.g.: self.ib_connection.cancelMktData(...)
            self.logger.info(f"Unsubscribed from market data for {symbol}")
        
        self.disconnect()
        self.logger.info("Stopped streaming data from Interactive Brokers")
    
    def is_connected(self) -> bool:
        """
        Check if still connected to Interactive Brokers.
        
        Returns:
            bool: Whether connection is active
        """
        # In a real implementation, you would check the actual connection status
        # E.g.: return self.ib_connection.isConnected()
        return self.connected and super().is_connected()
    
    def _process_market_data(self) -> None:
        """Process incoming market data from IB (runs in a separate thread)."""
        while self.is_running:
            try:
                # In a real implementation, you would retrieve and process actual market data
                # For demonstration, we'll simulate receiving data every second
                time.sleep(1)
                
                # Simulate market data for each symbol
                for symbol in self.symbols:
                    timestamp = datetime.now()
                    
                    # Create simulated tick data
                    data = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'price': 100 + np.random.normal(0, 1),
                        'volume': int(np.random.exponential(1000)),
                        'type': 'tick'
                    }
                    
                    # Process the data
                    self.process_data(data)
                
            except Exception as e:
                self.logger.error(f"Error processing market data: {str(e)}")
                time.sleep(1)  # Avoid tight error loops


class DataProcessor:
    """
    Processes incoming market data streams and maintains up-to-date OHLCV bars.
    
    This class converts streaming tick data into OHLCV bars and provides methods
    to access the latest market data for analysis and trading decisions.
    """
    
    def __init__(self, timeframes: List[str] = ['1min', '5min', '15min', '1hour', '1day']):
        """
        Initialize the data processor.
        
        Args:
            timeframes: List of timeframes to maintain bars for
        """
        self.timeframes = timeframes
        self.latest_ticks = {}
        self.bars = {tf: {} for tf in timeframes}
        self.callbacks = []
        
        # Lock for thread safety
        self.data_lock = threading.RLock()
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.DataProcessor")
    
    def register_callback(self, callback: Callable[[str, str, pd.DataFrame], None]) -> None:
        """
        Register a callback function to be called when new bars are created.
        
        The callback function should take: (timeframe, symbol, bar_df) as arguments.
        
        Args:
            callback: Function that takes the specified arguments
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[str, str, pd.DataFrame], None]) -> None:
        """
        Unregister a previously registered callback function.
        
        Args:
            callback: Previously registered callback function
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def process_tick(self, data: Dict[str, Any]) -> None:
        """
        Process a single tick of market data.
        
        Args:
            data: Dictionary containing tick data
        """
        # Extract key information
        symbol = data.get('symbol')
        timestamp = data.get('timestamp')
        price = data.get('price')
        
        if not symbol or not timestamp or price is None:
            self.logger.warning(f"Invalid tick data: {data}")
            return
        
        # Standardize timestamp
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        with self.data_lock:
            # Update latest tick for this symbol
            self.latest_ticks[symbol] = {
                'timestamp': timestamp,
                'price': price,
                'volume': data.get('volume', 0),
                'bid': data.get('bid', price),
                'ask': data.get('ask', price),
                'trade_id': data.get('trade_id', None)
            }
            
            # Update bars for each timeframe
            for timeframe in self.timeframes:
                self._update_bars(symbol, timeframe, timestamp, price, data.get('volume', 0))
    
    def _update_bars(self, symbol: str, timeframe: str, timestamp: datetime, 
                   price: float, volume: int) -> None:
        """
        Update OHLCV bars for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        # Get bar timestamp (floor to timeframe)
        bar_timestamp = self._floor_timestamp(timestamp, timeframe)
        
        # Initialize symbol dict if needed
        if symbol not in self.bars[timeframe]:
            self.bars[timeframe][symbol] = pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume']
            )
        
        # Get current bars for this symbol and timeframe
        df = self.bars[timeframe][symbol]
        
        # Check if we need to create a new bar
        if bar_timestamp not in df.index:
            # Add new bar
            new_bar = pd.DataFrame(
                [[price, price, price, price, volume]],
                index=[bar_timestamp],
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            
            # Append to existing bars
            self.bars[timeframe][symbol] = pd.concat([df, new_bar]).sort_index()
            
            # Call callbacks
            for callback in self.callbacks:
                try:
                    callback(timeframe, symbol, self.bars[timeframe][symbol])
                except Exception as e:
                    self.logger.error(f"Error in callback: {str(e)}")
        else:
            # Update existing bar
            df.at[bar_timestamp, 'high'] = max(df.at[bar_timestamp, 'high'], price)
            df.at[bar_timestamp, 'low'] = min(df.at[bar_timestamp, 'low'], price)
            df.at[bar_timestamp, 'close'] = price
            df.at[bar_timestamp, 'volume'] += volume
    
    def _floor_timestamp(self, timestamp: datetime, timeframe: str) -> datetime:
        """
        Floor a timestamp to the start of its timeframe bar.
        
        Args:
            timestamp: Timestamp to floor
            timeframe: Bar timeframe
            
        Returns:
            datetime: Floored timestamp
        """
        if timeframe.endswith('min'):
            minutes = int(timeframe.replace('min', ''))
            return timestamp.replace(
                second=0, microsecond=0,
                minute=(timestamp.minute // minutes) * minutes
            )
        elif timeframe.endswith('hour'):
            hours = int(timeframe.replace('hour', ''))
            return timestamp.replace(
                second=0, microsecond=0, minute=0,
                hour=(timestamp.hour // hours) * hours
            )
        elif timeframe == '1day':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Default to 1-minute bars
            return timestamp.replace(second=0, microsecond=0)
    
    def get_latest_bars(self, symbol: str, timeframe: str, 
                      n_bars: int = 100) -> pd.DataFrame:
        """
        Get the latest N bars for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            n_bars: Number of bars to return
            
        Returns:
            DataFrame: OHLCV bars
        """
        with self.data_lock:
            if timeframe not in self.bars:
                return pd.DataFrame()
            
            if symbol not in self.bars[timeframe]:
                return pd.DataFrame()
            
            # Return the latest N bars
            return self.bars[timeframe][symbol].tail(n_bars)
    
    def get_latest_tick(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest tick for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Latest tick data
        """
        with self.data_lock:
            return self.latest_ticks.get(symbol, {})


class RealTimeDataManager:
    """
    Manages real-time data processing for market analysis and trading.
    
    This class coordinates data sources, processing, and strategy execution
    for real-time trading applications.
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time data manager.
        
        Args:
            symbols: List of symbols to monitor
            config: Configuration dictionary
        """
        self.symbols = symbols
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.RealTimeDataManager")
        
        # Initialize data processor
        timeframes = self.config.get('timeframes', ['1min', '5min', '15min', '1hour', '1day'])
        self.data_processor = DataProcessor(timeframes=timeframes)
        
        # Initialize data source
        data_source_type = self.config.get('data_source', 'alpaca')
        self.data_source = self._create_data_source(data_source_type, symbols, self.config)
        
        # Register callback for incoming data
        self.data_source.register_callback(self._on_data_received)
        
        # Initialize market regime detector
        regime_config = self.config.get('regime_config', {})
        self.market_regime_detector = None
        if self.config.get('use_market_regimes', True):
            self.market_regime_detector = AdvancedMarketRegimeDetector(regime_config)
            
            # Try to load pre-trained model
            model_path = regime_config.get('model_path', 'models/market_regime_model.joblib')
            if os.path.exists(model_path):
                self.market_regime_detector.load_model(model_path)
        
        # Initialize strategy manager
        self.strategy_manager = None
        if self.config.get('use_strategy_rotation', False):
            strategies = self.config.get('strategies', [])
            if strategies:
                self.strategy_manager = StrategyRegimeRotator(
                    strategies=strategies,
                    regime_config=regime_config,
                    lookback_window=self.config.get('lookback_window', 60),
                    rebalance_frequency=self.config.get('rebalance_frequency', 'daily'),
                    max_allocation_change=self.config.get('max_allocation_change', 0.2)
                )
        
        # Market data cache
        self.market_data_cache = {symbol: None for symbol in symbols}
        
        # Current market regime
        self.current_regime = "unknown"
        
        # Monitor thread
        self.monitor_thread = None
        self.is_running = False
        
        # Event handlers
        self.on_regime_change = None
        self.on_bar_update = None
        self.on_strategy_update = None
    
    def _create_data_source(self, source_type: str, symbols: List[str], 
                          config: Dict[str, Any]) -> MarketDataSource:
        """
        Create a data source instance based on type.
        
        Args:
            source_type: Type of data source ('alpaca', 'ib', etc.)
            symbols: List of symbols to monitor
            config: Configuration dictionary
            
        Returns:
            MarketDataSource: Initialized data source
        """
        if source_type.lower() == 'alpaca':
            return AlpacaDataSource(symbols, config.get('alpaca_config', {}))
        elif source_type.lower() in ('ib', 'interactive_brokers'):
            return IBDataSource(symbols, config.get('ib_config', {}))
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    def _on_data_received(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming market data.
        
        Args:
            data: Market data dictionary
        """
        # Process the tick data
        self.data_processor.process_tick(data)
        
        # Update OHLCV cache for the symbol
        symbol = data.get('symbol')
        if symbol and symbol in self.symbols:
            # Get latest 1-day bar data for this symbol
            bars = self.data_processor.get_latest_bars(symbol, '1day', 100)
            if not bars.empty:
                self.market_data_cache[symbol] = bars
        
        # Once per minute (to reduce computational load), update market regime
        timestamp = data.get('timestamp')
        if timestamp and self.market_regime_detector and timestamp.second == 0:
            self._update_market_regime()
        
        # Call custom event handler if set
        if self.on_bar_update:
            try:
                self.on_bar_update(data)
            except Exception as e:
                self.logger.error(f"Error in on_bar_update handler: {str(e)}")
    
    def _update_market_regime(self) -> None:
        """Update the current market regime based on latest data."""
        if not self.market_regime_detector:
            return
        
        # Combine all symbol data into a single DataFrame
        combined_data = None
        for symbol, bars in self.market_data_cache.items():
            if bars is not None and not bars.empty:
                if combined_data is None:
                    combined_data = bars.copy()
                    combined_data.columns = [f'{col}' for col in combined_data.columns]
                else:
                    # For simplicity, just use the first symbol's data
                    # In a real implementation, you might want to aggregate data across symbols
                    pass
        
        if combined_data is not None:
            # Detect current regime
            regime = self.market_regime_detector.detect_regime(combined_data)
            
            # Check if regime has changed
            if regime != self.current_regime:
                self.logger.info(f"Market regime changed from {self.current_regime} to {regime}")
                self.current_regime = regime
                
                # Update strategy weights if using regime rotation
                if self.strategy_manager:
                    self.strategy_manager.update_market_regime(combined_data)
                    weights = self.strategy_manager.optimize_strategy_weights(
                        combined_data, None  # strategy returns would be needed here
                    )
                    self.logger.info(f"Updated strategy weights: {weights}")
                    
                    # Call custom event handler if set
                    if self.on_strategy_update:
                        try:
                            self.on_strategy_update(weights)
                        except Exception as e:
                            self.logger.error(f"Error in on_strategy_update handler: {str(e)}")
                
                # Call custom event handler if set
                if self.on_regime_change:
                    try:
                        self.on_regime_change(self.current_regime)
                    except Exception as e:
                        self.logger.error(f"Error in on_regime_change handler: {str(e)}")
    
    def start(self) -> None:
        """Start real-time data processing."""
        if self.is_running:
            self.logger.warning("Real-time data processing is already running")
            return
        
        # Connect to data source and start streaming
        self.data_source.connect()
        self.data_source.start_streaming()
        
        # Start monitoring thread
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Started real-time data processing")
    
    def stop(self) -> None:
        """Stop real-time data processing."""
        if not self.is_running:
            return
        
        # Stop streaming and disconnect from data source
        self.data_source.stop_streaming()
        self.data_source.disconnect()
        
        # Stop monitoring thread
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped real-time data processing")
    
    def _monitor_loop(self) -> None:
        """Monitor the data source connection and reconnect if needed."""
        while self.is_running:
            try:
                # Check connection status every 30 seconds
                time.sleep(30)
                
                if not self.data_source.is_connected():
                    self.logger.warning("Data source connection lost, attempting to reconnect")
                    success = self.data_source.reconnect()
                    
                    if success:
                        self.logger.info("Successfully reconnected to data source")
                    else:
                        self.logger.error("Failed to reconnect to data source")
            
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(5)  # Avoid tight error loops
    
    def get_latest_bars(self, symbol: str, timeframe: str, n_bars: int = 100) -> pd.DataFrame:
        """
        Get the latest N bars for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            n_bars: Number of bars to return
            
        Returns:
            DataFrame: OHLCV bars
        """
        return self.data_processor.get_latest_bars(symbol, timeframe, n_bars)
    
    def get_current_regime(self) -> str:
        """
        Get the current market regime.
        
        Returns:
            str: Current market regime
        """
        return self.current_regime
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get the current strategy weights.
        
        Returns:
            Dict: Current strategy weights
        """
        if self.strategy_manager:
            return self.strategy_manager.weights
        return {}


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define symbols to monitor
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
    
    # Configuration
    config = {
        'data_source': 'alpaca',
        'alpaca_config': {
            'api_key': 'YOUR_API_KEY',
            'api_secret': 'YOUR_API_SECRET'
        },
        'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
        'use_market_regimes': True,
        'regime_config': {
            'model_path': 'models/market_regime_model.joblib'
        }
    }
    
    # Create real-time data manager
    manager = RealTimeDataManager(symbols, config)
    
    # Define custom event handlers
    def on_regime_change(regime):
        print(f"Market regime changed to: {regime}")
    
    def on_bar_update(data):
        symbol = data.get('symbol')
        if symbol:
            print(f"New data for {symbol}: {data}")
    
    # Register event handlers
    manager.on_regime_change = on_regime_change
    manager.on_bar_update = on_bar_update
    
    try:
        # Start real-time processing
        manager.start()
        
        # Keep running until user interrupts
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    finally:
        # Stop processing
        manager.stop() 