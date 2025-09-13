#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Data Provider

This module implements a real-time data streaming provider using websockets.
"""

import logging
import json
import time
import threading
import queue
import websocket
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class RealTimeProvider:
    """
    Real-time data provider that streams market data using websockets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real-time data provider.
        
        Args:
            config: Configuration dictionary with connection details
        """
        self.config = config
        self.provider = config.get("provider", "finnhub")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("websocket_url", "wss://ws.finnhub.io")
        
        self.ws = None
        self.running = False
        self.subscribed_symbols = set()
        self.callbacks = {}
        self.data_queue = queue.Queue()
        self.worker_thread = None
        
        logger.info(f"Real-time data provider initialized using {self.provider}")
    
    def start(self) -> bool:
        """
        Start the real-time data provider.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Real-time provider already running")
            return True
        
        try:
            # Start websocket connection
            self._connect_websocket()
            
            # Start worker thread to process data
            self.worker_thread = threading.Thread(target=self._process_queue)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            self.running = True
            logger.info("Real-time data provider started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting real-time provider: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the real-time data provider."""
        if not self.running:
            return
        
        self.running = False
        
        # Close websocket
        if self.ws:
            try:
                # Unsubscribe from all symbols
                for symbol in self.subscribed_symbols:
                    self._send_unsubscribe(symbol)
                
                # Close connection
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
        
        # Clear data
        self.subscribed_symbols.clear()
        self.callbacks.clear()
        
        # Clear queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Real-time data provider stopped")
    
    def subscribe(self, symbol: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        """
        Subscribe to real-time data for a symbol.
        
        Args:
            symbol: Symbol to subscribe to
            callback: Optional callback function for data updates
            
        Returns:
            True if subscribed successfully, False otherwise
        """
        try:
            if not self.running:
                logger.warning("Cannot subscribe when provider is not running")
                return False
            
            # Send subscription request
            success = self._send_subscribe(symbol)
            
            if success:
                self.subscribed_symbols.add(symbol)
                
                # Register callback if provided
                if callback:
                    self.callbacks[symbol] = callback
                
                logger.info(f"Subscribed to {symbol}")
                return True
            else:
                logger.error(f"Failed to subscribe to {symbol}")
                return False
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return False
    
    def unsubscribe(self, symbol: str) -> bool:
        """
        Unsubscribe from real-time data for a symbol.
        
        Args:
            symbol: Symbol to unsubscribe from
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        try:
            if not self.running:
                logger.warning("Cannot unsubscribe when provider is not running")
                return False
            
            # Send unsubscription request
            success = self._send_unsubscribe(symbol)
            
            if success:
                if symbol in self.subscribed_symbols:
                    self.subscribed_symbols.remove(symbol)
                
                if symbol in self.callbacks:
                    del self.callbacks[symbol]
                
                logger.info(f"Unsubscribed from {symbol}")
                return True
            else:
                logger.error(f"Failed to unsubscribe from {symbol}")
                return False
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
            return False
    
    def get_subscribed_symbols(self) -> List[str]:
        """
        Get list of currently subscribed symbols.
        
        Returns:
            List of subscribed symbols
        """
        return list(self.subscribed_symbols)
    
    def _connect_websocket(self) -> None:
        """Establish websocket connection."""
        try:
            # Define websocket callbacks
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self.data_queue.put(data)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse websocket message: {message}")
                except Exception as e:
                    logger.error(f"Error processing websocket message: {e}")
            
            def on_error(ws, error):
                logger.error(f"Websocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"Websocket connection closed: {close_status_code} {close_msg}")
                
                # Attempt to reconnect if still running
                if self.running:
                    logger.info("Attempting to reconnect websocket...")
                    time.sleep(5)  # Wait before reconnecting
                    self._connect_websocket()
            
            def on_open(ws):
                logger.info("Websocket connection established")
                
                # Authenticate
                auth_message = {"type": "auth", "data": {"token": self.api_key}}
                ws.send(json.dumps(auth_message))
                
                # Resubscribe to previous symbols
                for symbol in self.subscribed_symbols:
                    self._send_subscribe(symbol)
            
            # Create websocket connection
            self.ws = websocket.WebSocketApp(
                self.base_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start websocket in a separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
            # Wait for connection to establish
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error connecting websocket: {e}")
            raise
    
    def _send_subscribe(self, symbol: str) -> bool:
        """
        Send subscription request over websocket.
        
        Args:
            symbol: Symbol to subscribe to
            
        Returns:
            True if request sent successfully, False otherwise
        """
        try:
            if not self.ws:
                logger.error("Websocket not connected")
                return False
            
            # Format subscription message based on provider
            if self.provider == "finnhub":
                message = {
                    "type": "subscribe",
                    "symbol": symbol
                }
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return False
            
            # Send subscription request
            self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending subscription request: {e}")
            return False
    
    def _send_unsubscribe(self, symbol: str) -> bool:
        """
        Send unsubscription request over websocket.
        
        Args:
            symbol: Symbol to unsubscribe from
            
        Returns:
            True if request sent successfully, False otherwise
        """
        try:
            if not self.ws:
                logger.error("Websocket not connected")
                return False
            
            # Format unsubscription message based on provider
            if self.provider == "finnhub":
                message = {
                    "type": "unsubscribe",
                    "symbol": symbol
                }
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return False
            
            # Send unsubscription request
            self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending unsubscription request: {e}")
            return False
    
    def _process_queue(self) -> None:
        """Process data from the queue and invoke callbacks."""
        while self.running:
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=1.0)
                
                # Process data based on provider
                if self.provider == "finnhub":
                    if "type" in data and data["type"] == "trade":
                        symbol = data.get("symbol", "")
                        
                        # Format data
                        trade_data = {
                            "symbol": symbol,
                            "price": data.get("data", [{}])[0].get("p", 0),
                            "volume": data.get("data", [{}])[0].get("v", 0),
                            "timestamp": data.get("data", [{}])[0].get("t", 0),
                            "type": "trade"
                        }
                        
                        # Invoke callback if registered
                        if symbol in self.callbacks:
                            try:
                                self.callbacks[symbol](trade_data)
                            except Exception as e:
                                logger.error(f"Error in callback for {symbol}: {e}")
                
                # Mark task as done
                self.data_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, continue
                continue
            except Exception as e:
                logger.error(f"Error processing data queue: {e}")
                time.sleep(1)  # Prevent tight loop on error 