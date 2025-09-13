import logging
from fastapi import WebSocket, HTTPException, status
from typing import List, Dict, Optional, Set, Any, Tuple, Union
from datetime import datetime
import asyncio
import json

# Import AuthService for token validation
from trading_bot.auth.service import AuthService

# Import WebSocket channel definitions
from trading_bot.api.websocket_channels import (
    ChannelType, MessageType, 
    get_channel_for_message, get_topic_for_message, 
    should_broadcast_to_topic
)

logger = logging.getLogger("websocket_manager")

class ConnectionManager:
    """Enhanced WebSocket connection manager with authentication and channel support."""
    def __init__(self):
        # Map from WebSocket connection to set of subscribed channels
        self.active_connections: Dict[WebSocket, Set[str]] = {}
        # Map from user_id to WebSocket connections
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # Map from channel to set of connected WebSockets
        self.channel_connections: Dict[str, Set[WebSocket]] = {}
        # Track subscription channels for each connection
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        # Message type handlers
        self.handlers: Dict[str, callable] = {}

    async def connect(self, websocket: WebSocket, token: Optional[str] = None):
        await websocket.accept()
        self.active_connections[websocket] = set()
        
        # Send connection status message
        await websocket.send_json({
            "event": "connection_status",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "status": "connected"
            }
        })
        
        # If a token is provided, validate it and associate connection with user
        if token:
            try:
                # This is where you would validate token and get user_id
                # For demonstration, we'll just use a dummy user_id
                user_id = "user_1"
                logging.info(f"User {user_id} connected")
                
                # Add to user connections map
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(websocket)
            except Exception as e:
                logging.error(f"Error validating token: {e}")
                await websocket.close(code=1008, reason="Invalid token")
                if websocket in self.active_connections:
                    del self.active_connections[websocket]
                return

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            # Get the channels this connection was subscribed to
            subscribed_channels = self.active_connections[websocket]
            
            # Remove from active connections
            del self.active_connections[websocket]
            
            # Remove from channel connections
            for channel in subscribed_channels:
                if channel in self.channel_connections and websocket in self.channel_connections[channel]:
                    self.channel_connections[channel].remove(websocket)
                    if not self.channel_connections[channel]:  # If no connections left in the channel
                        del self.channel_connections[channel]
            
            # Remove from user connections
            for user_id, connections in list(self.user_connections.items()):
                if websocket in connections:
                    connections.remove(websocket)
                    if not connections:  # If user has no more connections
                        del self.user_connections[user_id]
            
            logging.info(f"Client disconnected from channels: {subscribed_channels}")
        else:
            logging.warning("Attempted to disconnect a websocket that wasn't connected")

    async def subscribe(self, websocket: WebSocket, channel: str):
        if websocket not in self.active_connections:
            logging.warning(f"Attempted to subscribe non-connected websocket to channel {channel}")
            return
        
        # Add channel to connection's subscriptions
        self.active_connections[websocket].add(channel)
        
        # Add connection to channel's subscriber list
        if channel not in self.channel_connections:
            self.channel_connections[channel] = set()
        self.channel_connections[channel].add(websocket)
        
        logging.info(f"Client subscribed to channel: {channel}")
        
        # Send acknowledgement to client
        await websocket.send_json({
            "event": "subscription_ack",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "channel": channel,
                "action": "subscribe",
                "success": True
            }
        })

    async def unsubscribe(self, websocket: WebSocket, channel: str):
        if websocket not in self.active_connections:
            logging.warning(f"Attempted to unsubscribe non-connected websocket from channel {channel}")
            return
            
        if channel in self.active_connections[websocket]:
            # Remove channel from connection's subscriptions
            self.active_connections[websocket].remove(channel)
            
            # Remove connection from channel's subscriber list
            if channel in self.channel_connections and websocket in self.channel_connections[channel]:
                self.channel_connections[channel].remove(websocket)
                if not self.channel_connections[channel]:  # If no connections left in channel
                    del self.channel_connections[channel]
            
            logging.info(f"Client unsubscribed from channel: {channel}")
            
            # Send acknowledgement to client
            await websocket.send_json({
                "event": "subscription_ack",
                "timestamp": datetime.utcnow().isoformat(),
                "payload": {
                    "channel": channel,
                    "action": "unsubscribe",
                    "success": True
                }
            })
        else:
            logging.warning(f"Client tried to unsubscribe from channel {channel} it wasn't subscribed to")
            
            # Send error to client
            await websocket.send_json({
                "event": "subscription_ack",
                "timestamp": datetime.utcnow().isoformat(),
                "payload": {
                    "channel": channel,
                    "action": "unsubscribe",
                    "success": False,
                    "error": "Not subscribed to this channel"
                }
            })

    async def broadcast(self, message_type: str, data: dict):
        # Broadcast to all connected clients
        for websocket in self.active_connections.keys():
            try:
                await websocket.send_json({
                    "event": message_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": data
                })
            except Exception as e:
                logging.error(f"Error broadcasting to websocket: {e}")

    async def broadcast_to_channel(self, channel: str, message_type: str, data: dict, topic: str = None):
        # Broadcast to clients subscribed to the specified channel
        if channel not in self.channel_connections:
            logging.debug(f"No clients subscribed to channel {channel}")
            return
            
        for websocket in self.channel_connections[channel]:
            try:
                message = {
                    "event": message_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": data
                }
                if topic:
                    message["topic"] = topic
                    
                await websocket.send_json(message)
            except Exception as e:
                logging.error(f"Error broadcasting to channel {channel}: {e}")
                
                # Handle disconnected WebSocket
                try:
                    await websocket.close()
                except:
                    pass
                self.disconnect(websocket)

    async def broadcast_to_user(self, user_id: str, message_type: str, data: dict):
        # Broadcast to all connections for a specific user
        if user_id not in self.user_connections:
            logging.debug(f"User {user_id} has no active connections")
            return
            
        for websocket in self.user_connections[user_id]:
            try:
                await websocket.send_json({
                    "event": message_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": data
                })
            except Exception as e:
                logging.error(f"Error broadcasting to user {user_id}: {e}")
                
                # Handle disconnected WebSocket
                try:
                    await websocket.close()
                except:
                    pass
                self.disconnect(websocket)

    def register_handler(self, message_type: str, handler: callable):
        """Register a handler function for a specific message type"""
        self.handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

# Create a singleton instance to be used across all endpoints
manager = ConnectionManager()
enabled_manager = manager  # This allows easy disabling for testing

# Define standard channels
STANDARD_CHANNELS = [
    "market_data",     # Price updates, market signals, etc.
    "portfolio",      # Portfolio and position updates
    "trades",         # Trade execution updates
    "strategies",     # Strategy updates and decisions
    "market_context", # Market regime, sentiment, etc.
    "alerts",         # Alerts and notifications
    "logs"            # System logs and debug messages
]

# Helper methods for common broadcasts

async def broadcast_price_update(symbol: str, price: float, timestamp: Optional[str] = None):
    """Broadcast a price update to the market_data channel"""
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
        
    await enabled_manager.broadcast_to_channel(
        channel="market_data",
        message_type="price_update",
        data={
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp
        }
    )

async def broadcast_portfolio_update(account: str, summary: dict, positions: list):
    """Broadcast a portfolio update to the portfolio channel"""
    await enabled_manager.broadcast_to_channel(
        channel="portfolio",
        message_type="portfolio_update",
        data={
            "account": account,
            "summary": summary,
            "positions": positions
        }
    )

async def broadcast_trade_executed(account: str, trade: dict):
    """Broadcast a trade execution to the trades channel"""
    await enabled_manager.broadcast_to_channel(
        channel="trades",
        message_type="trade_executed",
        data={
            "account": account,
            "trade": trade
        }
    )

async def broadcast_market_regime_change(previous_regime: dict, current_regime: dict, change_factors: list):
    """Broadcast a market regime change to the market_context channel"""
    await enabled_manager.broadcast_to_channel(
        channel="market_context",
        message_type="market_regime_change",
        data={
            "previous_regime": previous_regime,
            "current_regime": current_regime,
            "change_factors": change_factors
        }
    )

async def broadcast_sentiment_update(sentiment_data: dict):
    """Broadcast a sentiment update to the market_context channel"""
    await enabled_manager.broadcast_to_channel(
        channel="market_context",
        message_type="sentiment_update",
        data=sentiment_data
    )

async def broadcast_strategy_update(strategies: list, reason_for_update: str):
    """Broadcast a strategy update to the strategies channel"""
    await enabled_manager.broadcast_to_channel(
        channel="strategies",
        message_type="strategy_priority_update",
        data={
            "strategies": strategies,
            "reason_for_update": reason_for_update
        }
    )

async def broadcast_news_alert(news_item: dict, priority: str = "medium", requires_attention: bool = False):
    """Broadcast a news alert to the alerts channel"""
    await enabled_manager.broadcast_to_channel(
        channel="alerts",
        message_type="news_alert",
        data={
            **news_item,
            "priority": priority,
            "requires_attention": requires_attention
        }
    )

async def broadcast_system_log(message: str, level: str = "info", source: str = "system"):
    """Broadcast a system log to the logs channel"""
    await enabled_manager.broadcast_to_channel(
        channel="logs",
        message_type="log",
        data={
            "message": message,
            "level": level,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# --- HEALTH SHIM ---
from typing import Dict, Any
from datetime import datetime, timezone

_state_snapshot = {
    "connected": False,
    "last_heartbeat_ts": None,  # epoch seconds
    "subs": [],
    "msg_per_min": 0,
    "reconnects": 0,
    "ping_rtt_ms": None,
}

def get_state_snapshot() -> Dict[str, Any]:
    try:
        connected = any(enabled_manager.active_connections)
        subs = list(enabled_manager.channel_connections.keys())
        snap = dict(_state_snapshot)
        snap.update({"connected": connected, "subs": subs})
        return snap
    except Exception:
        return dict(_state_snapshot)

def health() -> Dict[str, Any]:
    try:
        s = get_state_snapshot()
        stale_ms = None
        if s.get("last_heartbeat_ts"):
            now = datetime.now(timezone.utc).timestamp()
            stale_ms = int((now - s["last_heartbeat_ts"]) * 1000)
        return {
            "service": "websocket",
            "ok": bool(s.get("connected")) and (stale_ms is None or stale_ms < 15_000),
            "connected": bool(s.get("connected")),
            "subs": s.get("subs", []),
            "msg_per_min": s.get("msg_per_min"),
            "reconnects": s.get("reconnects"),
            "ping_rtt_ms": s.get("ping_rtt_ms"),
            "stale_ms": stale_ms,
        }
    except Exception as e:
        return {"service": "websocket", "ok": False, "error": str(e)}
