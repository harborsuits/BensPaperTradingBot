"""
WebSocket Handler

This module handles WebSocket connections and broadcasts for real-time updates.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from trading_bot.auth.jwt import decode_jwt_token
from trading_bot.market_analysis.market_analyzer import MarketAnalyzer

# Set up logging
logger = logging.getLogger("api.websocket")

# OAuth2 scheme for token validation
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


class ConnectionManager:
    """
    WebSocket connection manager for handling multiple client connections
    and broadcasting messages to subscribers
    """
    def __init__(self):
        # Map of connection ID to WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Map of channel name to set of connection IDs subscribed to that channel
        self.channel_subscribers: Dict[str, Set[str]] = {
            "portfolio": set(),
            "market_data": set(),
            "trading": set(),
            "context": set(),
            "logs": set(),
            "alerts": set(),
            "evotester": set(),
        }
        
        # Broadcast tasks
        self.tasks = []
        
        # For tracking connection details
        self.connection_info: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_info[connection_id] = {
            "connected_at": datetime.now().isoformat(),
            "subscriptions": set(),
            "user_agent": websocket.headers.get("user-agent", "Unknown"),
            "client_ip": websocket.client.host,
        }
        logger.info(f"New WebSocket connection established: {connection_id}")
    
    async def disconnect(self, connection_id: str) -> None:
        """Handle disconnection of a client"""
        # Remove from active connections
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from all channel subscriptions
        for channel in self.channel_subscribers:
            if connection_id in self.channel_subscribers[channel]:
                self.channel_subscribers[channel].remove(connection_id)
        
        # Remove connection info
        if connection_id in self.connection_info:
            del self.connection_info[connection_id]
            
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def subscribe(self, connection_id: str, channel: str) -> None:
        """Subscribe a connection to a specific channel"""
        if channel not in self.channel_subscribers:
            self.channel_subscribers[channel] = set()
        
        self.channel_subscribers[channel].add(connection_id)
        
        if connection_id in self.connection_info:
            self.connection_info[connection_id]["subscriptions"].add(channel)
            
        logger.debug(f"Connection {connection_id} subscribed to channel: {channel}")
    
    async def unsubscribe(self, connection_id: str, channel: str) -> None:
        """Unsubscribe a connection from a specific channel"""
        if channel in self.channel_subscribers and connection_id in self.channel_subscribers[channel]:
            self.channel_subscribers[channel].remove(connection_id)
            
        if connection_id in self.connection_info and channel in self.connection_info[connection_id]["subscriptions"]:
            self.connection_info[connection_id]["subscriptions"].remove(channel)
            
        logger.debug(f"Connection {connection_id} unsubscribed from channel: {channel}")
    
    async def broadcast(self, channel: str, message_type: str, data: Any) -> None:
        """Broadcast a message to all subscribers of a channel"""
        if channel not in self.channel_subscribers:
            return
        
        # Format the message
        message = {
            "type": message_type,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Get all connection IDs subscribed to this channel
        subscribers = self.channel_subscribers[channel]
        
        # Log broadcast events
        if subscribers:
            logger.debug(f"Broadcasting message type '{message_type}' to {len(subscribers)} subscribers on channel '{channel}'")
        
        # Convert message to JSON
        message_json = json.dumps(message)
        
        # Send to all subscribers
        for connection_id in list(subscribers):
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_text(message_json)
                except Exception as e:
                    logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
                    # Connection might be broken, disconnect it
                    await self.disconnect(connection_id)
    
    async def broadcast_price_updates(self, market_analyzer: MarketAnalyzer, interval: int = 5) -> None:
        """Continuously broadcast price updates for active symbols"""
        while True:
            try:
                subscribers = self.channel_subscribers.get("market_data", set())
                
                # Only proceed if there are subscribers
                if subscribers:
                    # Get price updates from the market analyzer
                    adapter = market_analyzer._get_adapter()
                    active_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN"]  # Example
                    
                    # Build the price updates
                    price_updates = []
                    
                    for symbol in active_symbols:
                        try:
                            # Get the latest price for each symbol
                            candles = await adapter.get_price_data(symbol, limit=1)
                            if candles and len(candles) > 0:
                                latest = candles[0]
                                price_updates.append({
                                    "symbol": symbol,
                                    "price": latest.close,
                                    "timestamp": datetime.fromtimestamp(latest.timestamp).isoformat(),
                                    "change": latest.close - latest.open,
                                    "changePercent": (latest.close - latest.open) / latest.open * 100 if latest.open > 0 else 0,
                                    "volume": latest.volume
                                })
                        except Exception as e:
                            logger.error(f"Error getting price data for {symbol}: {str(e)}")
                    
                    # Broadcast the price updates if we have any
                    if price_updates:
                        await self.broadcast("market_data", "price_updates", price_updates)
            
            except Exception as e:
                logger.error(f"Error in price update broadcast: {str(e)}")
            
            # Wait for the next interval
            await asyncio.sleep(interval)
    
    async def broadcast_portfolio_updates(self, interval: int = 10) -> None:
        """Continuously broadcast portfolio updates"""
        # This would normally fetch data from a portfolio service
        # For now, just a placeholder for demonstration
        pass
    
    async def start_broadcast_tasks(self, market_analyzer: MarketAnalyzer) -> None:
        """Start the background broadcast tasks"""
        # Cancel any existing tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Clear the task list
        self.tasks = []
        
        # Create new broadcast tasks
        price_task = asyncio.create_task(self.broadcast_price_updates(market_analyzer))
        portfolio_task = asyncio.create_task(self.broadcast_portfolio_updates())
        
        # Add to the task list
        self.tasks.extend([price_task, portfolio_task])
        
        logger.info("Started WebSocket broadcast tasks")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the connection manager"""
        return {
            "active_connections": len(self.active_connections),
            "subscribers_by_channel": {
                channel: len(subscribers) 
                for channel, subscribers in self.channel_subscribers.items()
            },
            "connections": [
                {
                    "id": conn_id,
                    "info": info
                }
                for conn_id, info in self.connection_info.items()
            ]
        }


# Create a singleton instance
manager = ConnectionManager()


async def get_current_user(token: str) -> Dict[str, Any]:
    """Validate JWT token and return user info"""
    try:
        return decode_jwt_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


class WebSocketEndpoint:
    """WebSocket endpoint handler"""
    
    def __init__(self, market_analyzer: Optional[MarketAnalyzer] = None):
        self.market_analyzer = market_analyzer
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """Handle WebSocket connections"""
        connection_id = f"ws_{id(websocket)}"
        
        # Verify token from query parameters
        token = websocket.query_params.get("token")
        
        # If token is provided, validate it
        if token:
            try:
                user = await get_current_user(token)
                connection_id = f"ws_{user.get('sub', id(websocket))}"
            except HTTPException:
                # If token is invalid, reject the connection
                await websocket.close(code=status.HTTP_401_UNAUTHORIZED)
                return
        
        await manager.connect(websocket, connection_id)
        
        try:
            while True:
                # Wait for messages from the client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    
                    # Handle subscription messages
                    if message.get("action") == "subscribe" and "channel" in message:
                        channel = message["channel"]
                        await manager.subscribe(connection_id, channel)
                        await websocket.send_text(json.dumps({
                            "type": "subscription_ack",
                            "channel": channel,
                            "status": "subscribed"
                        }))
                    
                    # Handle unsubscription messages
                    elif message.get("action") == "unsubscribe" and "channel" in message:
                        channel = message["channel"]
                        await manager.unsubscribe(connection_id, channel)
                        await websocket.send_text(json.dumps({
                            "type": "subscription_ack",
                            "channel": channel,
                            "status": "unsubscribed"
                        }))
                    
                    # Handle ping messages
                    elif message.get("action") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                except Exception as e:
                    logger.error(f"Error processing message from {connection_id}: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Invalid message format: {str(e)}"
                    }))
        
        except WebSocketDisconnect:
            await manager.disconnect(connection_id)
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {str(e)}")
            await manager.disconnect(connection_id)


def setup_websocket_endpoint(app, market_analyzer: Optional[MarketAnalyzer] = None):
    """Set up the WebSocket endpoint in the FastAPI app"""
    endpoint = WebSocketEndpoint(market_analyzer)
    
    # Add the WebSocket endpoint to the app
    app.add_websocket_route("/ws", endpoint.websocket_endpoint)
    
    # Start the broadcast tasks if a market analyzer is provided
    if market_analyzer:
        asyncio.create_task(manager.start_broadcast_tasks(market_analyzer))
    
    return endpoint
