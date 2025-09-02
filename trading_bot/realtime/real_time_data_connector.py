#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeDataConnector - Component to maintain persistent WebSocket connections
to market data providers and handle asynchronous data processing.
"""

import asyncio
import json
import logging
import websockets
from typing import Callable, Dict, Any, Optional
from datetime import datetime

# Setup logging
logger = logging.getLogger("RealTimeDataConnector")

class RealTimeDataConnector:
    """
    Maintains a persistent WebSocket connection to a market data provider
    and processes incoming data asynchronously.
    """
    
    def __init__(
        self, 
        url: str, 
        on_message_callback: Callable, 
        reconnect_interval: int = 5,
        auth_params: Optional[Dict[str, Any]] = None,
        subscriptions: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the real-time data connector.
        
        Args:
            url: WebSocket URL for market data provider
            on_message_callback: Async callable that receives decoded JSON market data
            reconnect_interval: Time in seconds to wait before reconnecting
            auth_params: Authentication parameters to send after connection
            subscriptions: Subscription messages to send after connection
        """
        self.url = url
        self.on_message_callback = on_message_callback
        self.reconnect_interval = reconnect_interval
        self.auth_params = auth_params or {}
        self.subscriptions = subscriptions or {}
        self._stop = False
        self._websocket = None
        self._connect_time = None
        self._last_message_time = None
        self._message_count = 0
        
    async def connect(self):
        """
        Establish and maintain the WebSocket connection.
        Automatically reconnects on failure.
        """
        while not self._stop:
            try:
                logger.info(f"Connecting to {self.url}")
                async with websockets.connect(self.url) as websocket:
                    self._websocket = websocket
                    self._connect_time = datetime.now()
                    self._last_message_time = self._connect_time
                    logger.info("Connected to market data stream")
                    
                    # Handle authentication if needed
                    if self.auth_params:
                        await self._authenticate(websocket)
                        
                    # Send subscription messages
                    await self._subscribe(websocket)
                    
                    # Process messages
                    await self._handle_connection(websocket)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"Connection error: {e}")
            
            if not self._stop:
                logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                await asyncio.sleep(self.reconnect_interval)
                
    async def _authenticate(self, websocket):
        """
        Send authentication message if required by the provider.
        
        Args:
            websocket: Active WebSocket connection
        """
        try:
            auth_message = json.dumps(self.auth_params)
            logger.info("Sending authentication message")
            await websocket.send(auth_message)
            
            # Wait for auth response
            response = await websocket.recv()
            logger.info(f"Authentication response: {response}")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise
    
    async def _subscribe(self, websocket):
        """
        Send subscription messages to the WebSocket server.
        
        Args:
            websocket: Active WebSocket connection
        """
        try:
            for channel, params in self.subscriptions.items():
                subscription = {
                    "channel": channel,
                    **params
                }
                
                subscription_msg = json.dumps(subscription)
                logger.info(f"Subscribing to {channel}")
                await websocket.send(subscription_msg)
                
                # Wait for subscription confirmation
                response = await websocket.recv()
                logger.info(f"Subscription response for {channel}: {response}")
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            raise

    async def _handle_connection(self, websocket):
        """
        Process incoming messages from the WebSocket connection.
        
        Args:
            websocket: Active WebSocket connection
        """
        async for message in websocket:
            try:
                # Parse message as JSON
                data = json.loads(message)
                
                # Update tracking metrics
                self._message_count += 1
                self._last_message_time = datetime.now()
                
                # Log periodic stats
                if self._message_count % 1000 == 0:
                    elapsed = (datetime.now() - self._connect_time).total_seconds()
                    rate = self._message_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {self._message_count} messages ({rate:.2f}/sec)")
                
                # Pass data to callback
                await self.on_message_callback(data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def send_message(self, message: Dict[str, Any]):
        """
        Send a message to the WebSocket server.
        
        Args:
            message: Message to send (will be JSON encoded)
        """
        if not self._websocket:
            logger.warning("Cannot send message - not connected")
            return False
            
        try:
            await self._websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def get_connection_stats(self):
        """
        Get connection statistics.
        
        Returns:
            Dict with connection stats
        """
        uptime = None
        if self._connect_time:
            uptime = (datetime.now() - self._connect_time).total_seconds()
            
        return {
            "connected": self._websocket is not None,
            "connect_time": self._connect_time,
            "last_message_time": self._last_message_time,
            "message_count": self._message_count,
            "uptime_seconds": uptime,
            "messages_per_second": self._message_count / uptime if uptime else 0
        }
    
    def stop(self):
        """Stop the connector and close the connection."""
        self._stop = True
        self._websocket = None
        logger.info("Connector stopped") 