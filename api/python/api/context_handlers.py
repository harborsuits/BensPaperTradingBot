"""
WebSocket handlers for market context data.
These handlers manage real-time updates for market features, regime indicators, 
sentiment analysis, and other market context information.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from pydantic import BaseModel

from trading_bot.api.websocket_handlers import ConnectionManager, WebSocketMessage

logger = logging.getLogger(__name__)

class MarketFeatureUpdate(BaseModel):
    """Model for market feature updates sent via WebSocket"""
    name: str
    value: float
    change: float
    timestamp: datetime


class MarketContextHandler:
    """Handler for market context WebSocket messages and updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.feature_update_task = None
        self.market_features = {}
        self._initialize_mock_features()
    
    def _initialize_mock_features(self):
        """Initialize mock features for development/testing"""
        self.market_features = {
            "momentum": {"value": 0.65, "change": 2.1},
            "volatility": {"value": 0.48, "change": -1.2},
            "breadth": {"value": 0.72, "change": 3.5},
            "liquidity": {"value": 0.52, "change": 0.3},
            "sentiment": {"value": 0.38, "change": -0.8},
            "fundamentals": {"value": 0.61, "change": 0.0},
            "technicals": {"value": 0.58, "change": 1.7},
            "correlation": {"value": 0.42, "change": -0.5}
        }
    
    async def start_feature_updates(self):
        """Start the background task for periodic market feature updates"""
        if self.feature_update_task is None or self.feature_update_task.done():
            self.feature_update_task = asyncio.create_task(self._send_periodic_feature_updates())
            logger.info("Started market feature update background task")
    
    async def stop_feature_updates(self):
        """Stop the background task for market feature updates"""
        if self.feature_update_task and not self.feature_update_task.done():
            self.feature_update_task.cancel()
            try:
                await self.feature_update_task
            except asyncio.CancelledError:
                pass
            self.feature_update_task = None
            logger.info("Stopped market feature update background task")
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming WebSocket messages related to market context"""
        if message.get("type") == "context_request":
            request_type = message.get("request")
            
            if request_type == "market_features":
                return {
                    "type": "market_features",
                    "data": self.market_features,
                    "timestamp": datetime.now().isoformat()
                }
                
            elif request_type == "market_regime":
                # This would normally fetch real regime data from your models
                return {
                    "type": "market_regime",
                    "data": {
                        "current_regime": "bullish_momentum",
                        "regime_strength": 0.82,
                        "regime_duration": "14d",
                        "previous_regime": "neutral_consolidation"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            elif request_type == "sentiment_analysis":
                # This would normally fetch real sentiment data from your models
                return {
                    "type": "sentiment_analysis",
                    "data": {
                        "overall": 0.68,
                        "market_sentiment": 0.72,
                        "social_media": 0.65,
                        "news": 0.58,
                        "institutional": 0.71
                    },
                    "timestamp": datetime.now().isoformat()
                }
        
        return None
    
    async def _send_periodic_feature_updates(self):
        """Send periodic market feature updates to all connected clients"""
        try:
            while True:
                # In a real implementation, you would fetch updated features from your models
                # This is a mock implementation for demo purposes
                await self._update_mock_features()
                
                # Broadcast the updated features to all clients subscribed to the context channel
                await self.connection_manager.broadcast(
                    WebSocketMessage(
                        type="features_update",
                        channel="context",
                        data=self.market_features,
                        timestamp=datetime.now().isoformat()
                    )
                )
                
                # Wait for the next update interval
                await asyncio.sleep(10)  # Update every 10 seconds
        
        except asyncio.CancelledError:
            logger.info("Market feature update task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in market feature update task: {e}")
    
    async def _update_mock_features(self):
        """Update mock features with small random changes for demo purposes"""
        import random
        
        for feature in self.market_features:
            # Generate small random changes
            value_change = random.uniform(-0.05, 0.05)
            pct_change = random.uniform(-0.5, 0.5)
            
            # Update the feature values
            current = self.market_features[feature]["value"]
            new_value = max(0, min(1, current + value_change))
            
            self.market_features[feature] = {
                "value": round(new_value, 2),
                "change": round(pct_change, 2)
            }
    
    async def broadcast_regime_change(self, regime_data: Dict[str, Any]):
        """Broadcast a market regime change event to all connected clients"""
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="market_regime_change",
                channel="context",
                data=regime_data,
                timestamp=datetime.now().isoformat()
            )
        )
    
    async def broadcast_sentiment_update(self, sentiment_data: Dict[str, Any]):
        """Broadcast a sentiment update event to all connected clients"""
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="sentiment_update",
                channel="context",
                data=sentiment_data,
                timestamp=datetime.now().isoformat()
            )
        )
    
    async def broadcast_news_alert(self, news_data: Dict[str, Any]):
        """Broadcast a news alert to all connected clients"""
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="news_alert",
                channel="context",
                data=news_data,
                timestamp=datetime.now().isoformat()
            )
        )
