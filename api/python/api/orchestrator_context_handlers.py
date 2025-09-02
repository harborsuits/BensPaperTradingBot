"""
Orchestrator Context WebSocket Handlers

This module provides WebSocket handlers for the orchestrator context,
enabling real-time updates for market data, strategy decisions, system health, and anomaly detection.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
import random
import logging
from typing import Dict, List, Optional, Union, Literal

from trading_bot.components.anomaly_detector import AnomalyDetector
from trading_bot.api.orchestrator_context_endpoints import get_real_market_anomalies

# Initialize logger
logger = logging.getLogger("api.orchestrator_context_ws")

# WebSocket connection manager for orchestrator context
class OrchestratorContextConnectionManager:
    """Connection manager for orchestrator context WebSocket connections.
    
    Handles connection lifecycle and message broadcasting to specific channels.
    """
    
    def __init__(self):
        # Active connections by channel
        self.active_connections: Dict[str, List[WebSocket]] = {
            "market": [],           # Market regime, volatility, sentiment
            "strategies": [],       # Strategy prioritization and performance
            "decisions": [],        # Trade decision candidates and execution
            "anomalies": [],        # Market anomalies and alerts
            "system": [],           # System health and performance
            "all": []               # All updates (firehose)
        }
        
        # Client identification tracking
        self.connection_channels: Dict[WebSocket, Set[str]] = {}
        
        # Last message cache for new connections
        self.last_messages: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Connect a client to a specific channel"""
        await websocket.accept()
        
        # Validate the requested channel
        if channel not in self.active_connections:
            await websocket.send_text(json.dumps({
                "error": f"Invalid channel: {channel}",
                "validChannels": list(self.active_connections.keys())
            }))
            await websocket.close()
            return False
        
        # Add to the channel
        self.active_connections[channel].append(websocket)
        
        # Track which channels this connection is subscribed to
        if websocket not in self.connection_channels:
            self.connection_channels[websocket] = set()
        self.connection_channels[websocket].add(channel)
        
        # Send the last message for this channel if available
        if channel in self.last_messages:
            await websocket.send_text(json.dumps({
                "type": "lastMessage",
                "channel": channel,
                "data": self.last_messages[channel]
            }))
        
        logger.info(f"Client connected to channel: {channel}")
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a client from all subscribed channels"""
        # Remove from all subscribed channels
        if websocket in self.connection_channels:
            for channel in self.connection_channels[websocket]:
                if websocket in self.active_connections[channel]:
                    self.active_connections[channel].remove(websocket)
            
            # Remove from tracking
            del self.connection_channels[websocket]
            
        logger.info("Client disconnected")
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast a message to all clients in a specific channel"""
        # Cache the last message
        self.last_messages[channel] = message
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
            
        # Add channel information
        wrapped_message = {
            "type": "update",
            "channel": channel,
            "data": message
        }
        
        # Convert to JSON
        message_json = json.dumps(wrapped_message)
        
        # Send to all clients in the channel
        disconnected = []
        
        # First send to the specific channel
        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(message_json)
            except RuntimeError:
                disconnected.append(connection)
        
        # Also send to the "all" channel if this isn't already the "all" channel
        if channel != "all":
            for connection in self.active_connections["all"]:
                if connection not in self.active_connections[channel]:  # Avoid duplicates
                    try:
                        await connection.send_text(message_json)
                    except RuntimeError:
                        disconnected.append(connection)
        
        # Clean up any disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_market_update(self, data: Dict[str, Any]):
        """Broadcast a market update to the market channel"""
        await self.broadcast_to_channel("market", data)
    
    async def broadcast_strategies_update(self, data: Dict[str, Any]):
        """Broadcast a strategies update to the strategies channel"""
        await self.broadcast_to_channel("strategies", data)
    
    async def broadcast_decisions_update(self, data: Dict[str, Any]):
        """Broadcast a decisions update to the decisions channel"""
        await self.broadcast_to_channel("decisions", data)
    
    async def broadcast_anomalies_update(self, data: Dict[str, Any]):
        """Broadcast an anomalies update to the anomalies channel"""
        await self.broadcast_to_channel("anomalies", data)
    
    async def broadcast_system_update(self, data: Dict[str, Any]):
        """Broadcast a system update to the system channel"""
        await self.broadcast_to_channel("system", data)

# Create a global instance of the connection manager
orchestrator_manager = OrchestratorContextConnectionManager()

# WebSocket route handlers
async def orchestrator_ws_endpoint(websocket: WebSocket, channel: str):
    """WebSocket endpoint for orchestrator context updates"""
    connection_successful = await orchestrator_manager.connect(websocket, channel)
    
    if not connection_successful:
        return
    
    try:
        while True:
            # Receive and process any client messages
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                # Handle client messages (e.g., subscription changes, requests for data)
                if "action" in message:
                    if message["action"] == "subscribe" and "channel" in message:
                        new_channel = message["channel"]
                        if new_channel in orchestrator_manager.active_connections:
                            await orchestrator_manager.connect(websocket, new_channel)
                            await websocket.send_text(json.dumps({
                                "type": "subscriptionConfirmed",
                                "channel": new_channel
                            }))
                    elif message["action"] == "unsubscribe" and "channel" in message:
                        channel_to_remove = message["channel"]
                        if websocket in orchestrator_manager.connection_channels:
                            if channel_to_remove in orchestrator_manager.connection_channels[websocket]:
                                orchestrator_manager.connection_channels[websocket].remove(channel_to_remove)
                                if websocket in orchestrator_manager.active_connections[channel_to_remove]:
                                    orchestrator_manager.active_connections[channel_to_remove].remove(websocket)
                                
                                await websocket.send_text(json.dumps({
                                    "type": "unsubscriptionConfirmed",
                                    "channel": channel_to_remove
                                }))
            except json.JSONDecodeError:
                # Invalid JSON, ignore
                pass
            
    except WebSocketDisconnect:
        orchestrator_manager.disconnect(websocket)

# Combines mock and real updates for development and production
async def start_mock_update_task():
    """Start a background task that sends orchestrator context updates periodically"""
    logger.info("Starting orchestrator context update task with real anomaly detection")
    
    while True:
        # Send mock market update
        await orchestrator_manager.broadcast_market_update({
            "regime": "bullish",
            "regimeConfidence": 0.85 + (asyncio.get_event_loop().time() % 0.1),
            "volatility": 0.32 + (asyncio.get_event_loop().time() % 0.05),
            "sentiment": 0.75 - (asyncio.get_event_loop().time() % 0.1),
            "timestamp": datetime.now().isoformat()
        })
        
        # Wait 5 seconds
        await asyncio.sleep(5)
        
        # Send mock strategy update
        await orchestrator_manager.broadcast_strategies_update({
            "topStrategy": "MomentumStrategy",
            "strategyScore": 0.88 + (asyncio.get_event_loop().time() % 0.05),
            "strategyChange": "increased prioritization",
            "timestamp": datetime.now().isoformat()
        })
        
        # Wait 7 seconds
        await asyncio.sleep(7)
        
        # Send mock decision update
        await orchestrator_manager.broadcast_decisions_update({
            "newCandidate": "MSFT",
            "score": 0.82 + (asyncio.get_event_loop().time() % 0.05),
            "action": "buy",
            "strategy": "GapTradingStrategy",
            "timestamp": datetime.now().isoformat()
        })
        
        # Wait 11 seconds
        await asyncio.sleep(11)
        
        # Get real anomaly data (executed every 30 seconds)
        if asyncio.get_event_loop().time() % 30 < 1:
            try:
                # Get real anomaly data from the API endpoint
                anomalies_data = get_real_market_anomalies()
                
                # If we have anomalies, broadcast them
                if anomalies_data and "anomalies" in anomalies_data and len(anomalies_data["anomalies"]) > 0:
                    # Find the highest severity anomaly for the broadcast summary
                    highest_severity_anomaly = None
                    for anomaly in anomalies_data["anomalies"]:
                        if not highest_severity_anomaly or anomaly.get("severity", 0) > highest_severity_anomaly.get("severity", 0):
                            highest_severity_anomaly = anomaly
                    
                    if highest_severity_anomaly:
                        # Create a summary for WebSocket broadcast
                        summary = {
                            "anomalyDetected": highest_severity_anomaly.get("description", "Unknown anomaly"),
                            "type": highest_severity_anomaly.get("type", "unknown"),
                            "severity": highest_severity_anomaly.get("severity", 0.0),
                            "severityLabel": highest_severity_anomaly.get("severityLabel", "low"),
                            "affectedAssets": highest_severity_anomaly.get("affectedAssets", []),
                            "timestamp": datetime.now().isoformat(),
                            "allAnomalies": anomalies_data["anomalies"],  # Include all anomalies for detailed view
                            "recommendedAction": highest_severity_anomaly.get("potentialImpact", {}).get("recommendedActions", [])[0] 
                                if highest_severity_anomaly.get("potentialImpact", {}).get("recommendedActions", []) else "monitor closely"
                        }
                        
                        # Broadcast the anomaly data
                        await orchestrator_manager.broadcast_anomalies_update(summary)
                        logger.info(f"Broadcasted real anomaly data: {highest_severity_anomaly.get('description')}")
                else:
                    # If no real anomalies, occasionally send a "all clear" update
                    if random.random() < 0.2:  # 20% chance
                        await orchestrator_manager.broadcast_anomalies_update({
                            "anomalyDetected": "No anomalies detected",
                            "type": "system_status",
                            "severity": 0.0,
                            "severityLabel": "none",
                            "affectedAssets": [],
                            "timestamp": datetime.now().isoformat(),
                            "recommendedAction": "continue normal operations",
                            "allAnomalies": []
                        })
            except Exception as e:
                logger.error(f"Error getting real anomaly data: {str(e)}")
                # Send a fallback mock anomaly
                if random.random() < 0.1:  # 10% chance to send mock data as fallback
                    await orchestrator_manager.broadcast_anomalies_update({
                        "anomalyDetected": "unusual options activity (mock)",
                        "type": "options_activity",
                        "severity": 0.75,
                        "severityLabel": "high",
                        "affectedAssets": ["SPY", "QQQ"],
                        "timestamp": datetime.now().isoformat(),
                        "recommendedAction": "monitor closely",
                        "isMockData": True
                    })
        
        # Send mock system update
        await orchestrator_manager.broadcast_system_update({
            "cpuUsage": 0.38 + (asyncio.get_event_loop().time() % 0.1),
            "memoryUsage": 0.42 + (asyncio.get_event_loop().time() % 0.05),
            "apiLatency": 231 + int(asyncio.get_event_loop().time() % 50),
            "dataProcessingRate": 1250 + int(asyncio.get_event_loop().time() % 200),
            "timestamp": datetime.now().isoformat()
        })
        
        # Wait 3 seconds before the next cycle
        await asyncio.sleep(3)
