#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot API - FastAPI application for the trading bot
providing UI and API endpoints for monitoring and control.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import logging
import os
import sys

# Import WebSocket manager and helper functions
from .websocket_manager import (
    enabled_manager as manager,
    broadcast_price_update,
    broadcast_portfolio_update,
    broadcast_trade_executed,
    broadcast_market_regime_change,
    broadcast_sentiment_update,
    broadcast_strategy_update,
    broadcast_news_alert,
    broadcast_system_log
)
from pydantic import BaseModel

# Import auth modules
from trading_bot.auth.api import router as auth_router
from trading_bot.auth.service import AuthService

# Import BenBot Assistant
from trading_bot.assistant.benbot_assistant import BenBotAssistant

# Import event system and backtest endpoints
from trading_bot.api.event_endpoints import event_api
from trading_bot.api.backtest_endpoints import router as backtest_api
from trading_bot.api.evotester_endpoints import router as evo_router
from trading_bot.api.data_ingestion_endpoints import router as data_router
from trading_bot.api.context_endpoints import router as context_router
from trading_bot.api.orchestrator_context_endpoints import router as orchestrator_router

# Import WebSocket handlers for real-time market data
from trading_bot.api.websocket_handler import setup_websocket_endpoint, manager as ws_data_manager
from trading_bot.api.context_handlers import MarketContextHandler
from trading_bot.api.market_data_handlers import MarketDataHandler
from trading_bot.api.orchestrator_context_handlers import orchestrator_manager, orchestrator_ws_endpoint, start_mock_update_task as start_orchestrator_mock_task
from trading_bot.components.anomaly_detector import AnomalyDetector
from trading_bot.api.orchestrator_context_endpoints import register_anomaly_detector

# Import AI chat handlers
from trading_bot.api.ai_chat_handlers import router as ai_chat_router
from trading_bot.api.trade_decision_handlers import TradeDecisionHandler
from trading_bot.api.trade_decision_endpoints import router as decisions_router

# Import routers for orders and positions
from trading_bot.api.routers.orders import router as orders_router
from trading_bot.api.routers.positions import router as positions_router
from trading_bot.api.routers.trades import router as trades_router
from trading_bot.api.routers.recap import router as recap_router
from trading_bot.api.routers.strategies import router as strategies_router
from trading_bot.api.logging_endpoints import router as logging_router
from trading_bot.api.strategy_endpoints import router as strategy_analysis_router
from trading_bot.api.routers.safety import router as safety_router
from trading_bot.api.routers.decisions import router as decisions_router
from trading_bot.api.routers.health_sources import router as health_sources_router
from trading_bot.api.websocket_endpoint import router as websocket_router

# Import engine router
from trading_bot.api.routes.engine import router as engine_router
from trading_bot.api.routes.opportunities import router as opportunities_router

# Import Coinbase API router
try:
    from trading_bot.api.coinbase_market_data import router as coinbase_router
    coinbase_available = True
    logger.info("Coinbase API module available for import")
except ImportError:
    coinbase_available = False
    logger.warning("Coinbase API module not available for import")

# Import enhanced WebSocket manager
from .websocket_manager import enabled_manager as ws_manager

# Import typed settings
from trading_bot.config.typed_settings import APISettings, TradingBotSettings, load_config
from trading_bot.ml_pipeline.model_trainer import ModelTrainer

# Initialize logging
logger = logging.getLogger("TradingBotAPI")

# Load typed settings if available
api_settings = None
try:
    config = load_config()
    api_settings = config.api
    logger.info("Loaded API settings from typed config")
except Exception as e:
    logger.warning(f"Could not load typed API settings: {str(e)}. Using defaults.")
    api_settings = APISettings()

# Initialize FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="API for monitoring and controlling the trading bot",
    version="1.0.0"
)

# Add CORS middleware with settings from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(event_api)  # Add event system endpoints
app.include_router(backtest_api)  # Add backtesting endpoints
app.include_router(evo_router)  # Add EvoTester endpoints
app.include_router(context_router, prefix="/api", tags=["MarketContext"])  # Add context endpoints under /api
# Include orders, positions and trades routers
app.include_router(orders_router, prefix="/api", tags=["Orders"])
app.include_router(positions_router, prefix="/api", tags=["Positions"])
app.include_router(trades_router, prefix="/api", tags=["Trades"])
app.include_router(recap_router, prefix="/api", tags=["Recap"])
app.include_router(strategies_router, prefix="/api", tags=["Strategies"])
app.include_router(logging_router, prefix="/api", tags=["Logging"])
app.include_router(strategy_analysis_router, prefix="/api", tags=["StrategyAnalysis"])
app.include_router(safety_router, prefix="/api", tags=["Safety"])
app.include_router(decisions_router, prefix="/api", tags=["Decisions"])

# Include the new policy-driven decisions router
app.include_router(decisions_router, prefix="/api/policy", tags=["PolicyDecisions"])

# Include the engine router
app.include_router(engine_router)

# Include the opportunities router
app.include_router(opportunities_router)

# Import news router
from trading_bot.api.routers.news import router as news_router
# Include news router
app.include_router(news_router)

# Health sources router
app.include_router(health_sources_router, prefix="")
logger.info("News API endpoints registered")

# Include Coinbase router if available
if coinbase_available:
    app.include_router(coinbase_router)
    logger.info("Coinbase API endpoints registered")

# Include WebSocket endpoints
app.include_router(websocket_router)  # Enhanced WebSocket endpoint

# Include AI chat router with enhanced functionality
app.include_router(ai_chat_router, prefix="/api")

# The main WebSocket endpoint is now managed by the websocket_router
# This legacy endpoint is kept for backward compatibility
@app.websocket("/ws/legacy")
async def legacy_websocket_endpoint(websocket: WebSocket, token: Optional[str] = None):
    """Legacy WebSocket endpoint for backward compatibility."""
    await ws_manager.connect(websocket, token)
    try:
        while True:
            message = await websocket.receive_json()
            await manager.process_message(websocket, message)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# WebSocket endpoint for market context and market features
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    # Get token from query params if provided
    token = None
    if websocket.query_params.get("token"):
        token = websocket.query_params.get("token")
    
    # Connect with optional token
    await manager.connect(websocket, token)
    
    # Subscribe to the channel
    await manager.subscribe(websocket, channel)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Log incoming data
            logging.info(f"Received WebSocket data on channel {channel}: {data}")
            
            # Handle subscription messages
            if data.get("action") in ["subscribe", "unsubscribe"]:
                await manager.handle_subscription_message(websocket, data)
            else:
                # Process other message types
                message_type = data.get("event")
                if message_type:
                    # Send acknowledgement
                    await websocket.send_json({
                        "event": "message_received",
                        "timestamp": datetime.utcnow().isoformat(),
                        "payload": {
                            "original_event": message_type,
                            "status": "processed"
                        }
                    })
                else:
                    # Invalid message format
                    await websocket.send_json({
                        "event": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "payload": {
                            "code": "INVALID_FORMAT",
                            "message": "Message format invalid"
                        }
                    })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# WebSocket endpoint for orchestrator context updates
@app.websocket("/ws/orchestrator/{channel}")
async def orchestrator_ws(websocket: WebSocket, channel: str):
    await orchestrator_ws_endpoint(websocket, channel)

# Reference to strategy rotator and continuous learner
# These will be set when the app is started
_strategy_rotator = None
_continuous_learner = None

# API Models
class StrategyInfo(BaseModel):
    name: str
    enabled: bool
    average_performance: float
    current_weight: float
    last_signal: Optional[float] = None

class StrategyUpdate(BaseModel):
    enabled: Optional[bool] = None
    weight: Optional[float] = None

class PerformanceReport(BaseModel):
    timestamp: str
    overall_performance: float
    strategies: Dict[str, Any]

class MarketData(BaseModel):
    prices: List[float]
    volume: Optional[List[float]] = None
    additional_data: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    strategy_names: List[str]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint returning API info."""
    return {"message": "Trading Bot API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if _strategy_rotator is None:
        return {"status": "degraded", "message": "Strategy rotator not initialized"}
    
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Create handlers for real-time updates
market_context_handler = MarketContextHandler(ws_manager)
market_data_handler = MarketDataHandler(ws_manager)
trade_decision_handler = TradeDecisionHandler(ws_manager)

# Initialize API with references to the components
def initialize_api(strategy_rotator, continuous_learner, engine, market_analyzer=None):
    """Initialize the API with references to the components."""
    global _strategy_rotator, _continuous_learner, trading_engine
    
    _strategy_rotator = strategy_rotator
    _continuous_learner = continuous_learner
    trading_engine = engine
    
    # Create and register the anomaly detector
    try:
        anomaly_detector = AnomalyDetector(engine=engine)
        register_anomaly_detector(anomaly_detector)
        logger.info("Initialized and registered anomaly detector for market anomaly detection")
    except Exception as e:
        logger.error(f"Error initializing anomaly detector: {str(e)}")
        logger.warning("Anomaly detection will not be available")
    
    # Register WebSocket handlers for the engine's events
    if hasattr(engine, 'event_bus'):
        engine.event_bus.register_handler('price_update', _handle_price_update)
        engine.event_bus.register_handler('trade_executed', _handle_trade_executed)
        engine.event_bus.register_handler('order_update', _handle_order_update)
        engine.event_bus.register_handler('position_update', _handle_position_update)
        engine.event_bus.register_handler('error', _handle_error)
        engine.event_bus.register_handler('strategy_decision', _handle_strategy_decision)
        logger.info("Registered event handlers for WebSocket broadcasting")
    else:
        logger.warning("Trading engine does not have an event_bus attribute. WebSocket broadcasting may not work properly.")
    
    # Initialize the real-time market data WebSocket handler if market analyzer is provided
    if market_analyzer:
        # Set up the WebSocket endpoint for real-time market data
        setup_websocket_endpoint(app, market_analyzer)
        logger.info("Initialized real-time market data WebSocket handler with market analyzer")
        
        # Initialize orchestrator context dependencies with market analyzer and anomaly detector
        from trading_bot.api.orchestrator_context_endpoints import initialize_dependencies
        initialize_dependencies(market_analyzer=market_analyzer, anomaly_detector=anomaly_detector if 'anomaly_detector' in locals() else None)
        logger.info("Initialized orchestrator context dependencies")
        
        # Start the orchestrator mock task for development/testing
        # This can be removed when real data is fully implemented
        start_orchestrator_mock_task()
        logger.info("Started orchestrator mock update task for testing")
        
    # Start the background tasks for real-time updates
    import asyncio
    asyncio.create_task(market_context_handler.start_feature_updates())
    asyncio.create_task(market_data_handler.start_price_updates())
    asyncio.create_task(market_data_handler.start_risk_metrics_updates())
    asyncio.create_task(market_data_handler.start_market_signals_updates())
    asyncio.create_task(trade_decision_handler.start_candidate_updates())
    
    logger.info("Started real-time update background tasks")
    
    logger.info("API initialized with strategy rotator, continuous learner, and trading engine")

# Event handlers for WebSocket broadcasting
async def _handle_price_update(event):
    """Handle price update events from the engine"""
    await ws_manager.broadcast_to_channel(
        channel="prices",
        message_type="price_update",
        data={
            "symbol": event.get("symbol"),
            "price": event.get("price"),
            "timestamp": event.get("timestamp", datetime.utcnow().isoformat())
        }
    )

async def _handle_trade_executed(event):
    """Handle trade execution events from the engine"""
    await ws_manager.broadcast_to_channel(
        channel="trades",
        message_type="trade_executed",
        data=event
    )

async def _handle_order_update(event):
    """Handle order update events from the engine"""
    await ws_manager.broadcast_to_channel(
        channel="orders",
        message_type="order_update",
        data=event
    )

async def _handle_position_update(event):
    """Handle position update events from the engine"""
    await ws_manager.broadcast_to_channel(
        channel="positions",
        message_type="position_update",
        data=event
    )

async def _handle_error(event):
    """Handle error events from the engine"""
    await ws_manager.broadcast(
        message_type="error",
        data={
            "message": event.get("message"),
            "source": event.get("source"),
            "timestamp": event.get("timestamp", datetime.utcnow().isoformat()),
            "level": event.get("level", "error")
        }
    )

async def _handle_strategy_decision(event):
    """Handle strategy decision events from the engine"""
    await ws_manager.broadcast_to_channel(
        channel="strategies",
        message_type="strategy_decision",
        data=event
    )


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the API server with settings from config
    uvicorn.run(
        app, 
        host=api_settings.host, 
        port=api_settings.port,
        debug=api_settings.debug
    )
