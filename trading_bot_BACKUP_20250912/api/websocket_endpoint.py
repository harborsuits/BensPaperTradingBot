"""
WebSocket Endpoint for BenBot Trading Dashboard

This module sets up the WebSocket endpoint for the BenBot trading dashboard,
handling client connections, message processing, and channel subscriptions.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional
import json
import logging
import asyncio
from datetime import datetime

from trading_bot.api.websocket_manager import enabled_manager
from trading_bot.auth.service import AuthService, get_optional_user
from trading_bot.api.context_endpoints import (
    broadcast_news_update, 
    broadcast_regime_change,
    broadcast_anomaly,
    generate_sample_news,
    generate_sample_context
)
from trading_bot.api.strategy_endpoints import (
    broadcast_strategy_update,
    broadcast_strategy_rankings,
    broadcast_new_trade_candidate,
    generate_sample_strategies,
    generate_trade_candidates
)

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger("websocket_endpoint")

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, 
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint to establish a connection for real-time updates
    
    - Optionally authenticate with a token
    - Subscribe to specific channels (context, strategy, trading, etc.)
    - Receive real-time updates based on subscriptions
    """
    try:
        # Connect the client
        await enabled_manager.connect(websocket, token)
        
        # Process incoming messages
        while True:
            try:
                # Receive and parse the message
                raw_data = await websocket.receive_text()
                message = json.loads(raw_data)
                
                # Let the manager process the message
                await enabled_manager.process_message(websocket, message)
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
    except WebSocketDisconnect:
        # Handle disconnection
        enabled_manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        # Handle other exceptions
        logger.error(f"WebSocket error: {str(e)}")
        try:
            enabled_manager.disconnect(websocket)
        except:
            pass

# Simulation function to regularly send market context updates for testing
async def simulate_market_updates():
    """
    Simulates market update broadcasts for testing purposes
    This would normally be triggered by real market data
    """
    logger.info("Starting market update simulation")
    
    while True:
        try:
            # Wait between updates
            await asyncio.sleep(15)  # Send updates every 15 seconds
            
            # Get sample data
            context = generate_sample_context()
            news_items = generate_sample_news()
            strategies = generate_sample_strategies()
            trade_candidates = generate_trade_candidates()
            
            # 1. Broadcast market regime updates occasionally
            if context and asyncio.get_event_loop().time() % 60 < 5:  # Roughly every minute
                await broadcast_regime_change(context.regime)
                logger.info("Simulated market regime update broadcast")
            
            # 2. Broadcast news updates regularly
            if news_items and len(news_items) > 0:
                await broadcast_news_update(news_items[0])
                logger.info("Simulated news update broadcast")
            
            # 3. Broadcast anomaly detection occasionally
            if context and context.anomalies and asyncio.get_event_loop().time() % 90 < 5:  # Every ~90 seconds
                await broadcast_anomaly(context.anomalies[0])
                logger.info("Simulated anomaly broadcast")
            
            # 4. Broadcast strategy ranking updates
            if strategies and asyncio.get_event_loop().time() % 45 < 5:  # Every ~45 seconds
                await broadcast_strategy_rankings(strategies)
                logger.info("Simulated strategy rankings broadcast")
            
            # 5. Broadcast trade candidates occasionally
            if trade_candidates and len(trade_candidates) > 0 and asyncio.get_event_loop().time() % 120 < 5:  # Every ~2 minutes
                await broadcast_new_trade_candidate(trade_candidates[0])
                logger.info("Simulated trade candidate broadcast")
                
        except Exception as e:
            logger.error(f"Error in market update simulation: {str(e)}")
            await asyncio.sleep(5)  # Wait a bit before trying again

# Start the simulation when the application starts
@router.on_event("startup")
async def start_simulation():
    """Starts the market update simulation on application startup"""
    asyncio.create_task(simulate_market_updates())
