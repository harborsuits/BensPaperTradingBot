#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtester API Endpoints

This module provides API endpoints for the backtesting pipeline, exposing data
from the event system and backtest processor to the frontend.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field

# Import from event system
from trading_bot.event_system import (
    EventBus, EventType, Event,
    MessageQueue, ChannelManager
)

# Setup logging
logger = logging.getLogger("backtest_api")

# Router
router = APIRouter(
    prefix="/api",
    tags=["backtesting"],
    responses={404: {"description": "Not found"}},
)

# Cache for backtest data
current_backtest = None
strategy_queue = []
processing_stats = {
    "cpu": 65,
    "memory": 60,
    "disk": 45,
    "concurrentTests": 2,
    "completedToday": 35,
    "averageDuration": "00:42:18",
    "queueLength": 5
}

# Recent backtest results
recent_results = []

# Models
class BacktestRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to backtest")
    start_date: str = Field(..., description="Start date for backtesting")
    end_date: str = Field(..., description="End date for backtesting")
    strategy: str = Field(..., description="Strategy to use for backtesting")
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")

class BacktestStatus(BaseModel):
    id: str = Field(..., description="Backtest ID")
    status: str = Field(..., description="Backtest status")
    progress: int = Field(..., description="Backtest progress percentage")
    executionStage: str = Field(..., description="Current execution stage")
    eta: Optional[str] = Field(None, description="Estimated time to completion")
    started_at: str = Field(..., description="Start time")

# Monitor channel for backtest results
def monitor_backtest_events():
    global current_backtest, strategy_queue, processing_stats
    
    try:
        # Get channel manager instance
        channel_manager = ChannelManager()
        
        # Subscribe to backtest results channel
        backtest_channel = channel_manager.get_channel("backtest_results")
        if backtest_channel:
            # Get latest update
            latest = backtest_channel.get_latest()
            if latest:
                # Update current backtest
                if 'id' in latest and 'status' in latest:
                    current_backtest = latest
                    # If backtest completed, add to recent results
                    if latest.get('status') == 'completed' and 'results' in latest:
                        recent_results.append(latest)
                        # Keep only last 10 results
                        if len(recent_results) > 10:
                            recent_results.pop(0)
        
        # Update processing stats
        # In a real implementation, these would come from system monitoring
        processing_stats["cpu"] = min(max(processing_stats["cpu"] + 
                                      ((-5 + int(time.time() % 10)) // 2), 50), 90)
        processing_stats["memory"] = min(max(processing_stats["memory"] + 
                                         ((-3 + int(time.time() % 6)) // 2), 55), 85)
        
        # Update queue length based on recent completions
        if processing_stats["queueLength"] > 0 and time.time() % 30 < 1:
            processing_stats["queueLength"] -= 1
            processing_stats["completedToday"] += 1
    
    except Exception as e:
        logger.error(f"Error monitoring backtest events: {e}")

# Endpoints
@router.get("/current_test")
async def get_current_test():
    """
    Get current backtest data
    """
    monitor_backtest_events()
    
    if not current_backtest:
        # If no current backtest, create placeholder
        current_backtest = {
            "id": f"bt-{int(time.time())}",
            "status": "running",
            "progress": 65,
            "eta": "00:15:30",
            "startedAt": (time.time() - 1800),
            "elapsed": "00:30:15",
            "testPeriod": "2020-01-01 to 2024-12-31",
            "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"],
            "parameters": {
                "lookbackPeriod": 20,
                "overBoughtThreshold": 0.8,
                "overSoldThreshold": 0.2
            },
            "executionStage": "Analyzing news sentiment data",
            "results": {
                "totalReturn": 12.8,
                "annualizedReturn": 8.4,
                "sharpeRatio": 1.2,
                "sortino": 1.8,
                "maxDrawdown": 8.2,
                "winRate": 62,
                "profitFactor": 1.7,
                "expectancy": 0.87,
                "totalTrades": 245,
                "winningTrades": 152,
                "losingTrades": 93,
                "averageWinning": 1.2,
                "averageLosing": 0.6,
                "averageDuration": "3.5 days"
            },
            "newsSentiment": {
                "positiveEvents": 78,
                "negativeEvents": 53,
                "neutralEvents": 114,
                "totalEvents": 245,
                "averageSentimentScore": 0.23,
                "topEventsByImpact": [
                    {
                        "headline": "Positive earnings surprise for AAPL",
                        "date": "2023-08-15",
                        "sentiment": "positive",
                        "impact": 0.85
                    },
                    {
                        "headline": "MSFT announces major cloud expansion",
                        "date": "2023-10-22",
                        "sentiment": "positive",
                        "impact": 0.76
                    },
                    {
                        "headline": "GOOGL faces new regulatory challenges",
                        "date": "2023-11-14",
                        "sentiment": "negative",
                        "impact": -0.67
                    }
                ],
                "sentimentBySymbol": {
                    "AAPL": 0.45,
                    "MSFT": 0.38,
                    "AMZN": 0.12,
                    "GOOGL": -0.21,
                    "META": 0.31,
                    "TSLA": 0.19
                }
            }
        }
    
    return current_backtest

@router.get("/strategy_queue")
async def get_strategy_queue():
    """
    Get strategies in queue for backtesting
    """
    monitor_backtest_events()
    
    # If queue is empty, create placeholder data
    if not strategy_queue:
        strategy_queue = [
            {
                "id": "ST-101",
                "name": "ML-Enhanced Mean Reversion",
                "description": "Machine learning model that identifies overbought/oversold conditions",
                "status": "In Queue",
                "priority": "High",
                "estimatedStart": (time.time() + 1200),
                "assets": ["SPY", "QQQ", "IWM"],
                "parameters": {},
                "complexity": "High",
                "createdAt": (time.time() - 3600),
                "updatedAt": (time.time() - 1800)
            },
            {
                "id": "ST-102",
                "name": "News Sentiment Analysis",
                "description": "Analysis of news sentiment for market signals",
                "status": "In Queue",
                "priority": "Medium",
                "estimatedStart": (time.time() + 3600),
                "assets": ["AAPL", "MSFT", "AMZN", "GOOGL"],
                "parameters": {},
                "complexity": "Medium",
                "createdAt": (time.time() - 7200),
                "updatedAt": (time.time() - 3600)
            },
            {
                "id": "ST-103",
                "name": "Global Macro Rotation",
                "description": "Systematic global macro strategy with allocation shifts based on economic regime",
                "status": "In Queue",
                "priority": "Low",
                "estimatedStart": (time.time() + 7200),
                "assets": ["SPY", "EFA", "EEM", "AGG"],
                "parameters": {},
                "complexity": "High",
                "createdAt": (time.time() - 10800),
                "updatedAt": (time.time() - 7200)
            }
        ]
    
    return strategy_queue

@router.get("/processing_stats")
async def get_processing_stats():
    """
    Get processing statistics for backtesting system
    """
    monitor_backtest_events()
    return processing_stats

@router.post("/backtest/submit")
async def submit_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Submit a new backtest request
    """
    try:
        # Get message queue for backtest requests
        backtest_queue = MessageQueue.get("backtest_queue")
        if not backtest_queue:
            raise HTTPException(status_code=500, detail="Backtest queue not available")
        
        # Create backtest config
        backtest_id = f"bt-{int(time.time())}"
        config = {
            "id": backtest_id,
            "symbols": request.symbols,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "strategy": request.strategy,
            "parameters": request.parameters
        }
        
        # Add to queue
        backtest_queue.put(config)
        
        # Update strategy queue
        strategy_queue.append({
            "id": f"ST-{backtest_id[3:]}",
            "name": f"{request.strategy} Strategy",
            "description": f"Backtest of {request.strategy} on {', '.join(request.symbols[:3])}{'...' if len(request.symbols) > 3 else ''}",
            "status": "In Queue",
            "priority": "Medium",
            "estimatedStart": time.time() + 600,
            "assets": request.symbols,
            "parameters": request.parameters,
            "complexity": "Medium",
            "createdAt": time.time(),
            "updatedAt": time.time()
        })
        
        # Update queue length
        processing_stats["queueLength"] += 1
        
        return {"success": True, "message": "Backtest submitted successfully", "id": backtest_id}
    
    except Exception as e:
        logger.error(f"Error submitting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest/control")
async def control_backtest(action: str = Query(..., description="Action to take"), 
                          backtest_id: str = Query(..., description="Backtest ID")):
    """
    Control running backtests (pause, resume, cancel)
    """
    global current_backtest
    
    if not current_backtest or current_backtest.get("id") != backtest_id:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    if action not in ["pause", "resume", "cancel"]:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    try:
        # Update backtest status
        if action == "pause":
            current_backtest["status"] = "paused"
        elif action == "resume":
            current_backtest["status"] = "running"
        elif action == "cancel":
            current_backtest["status"] = "cancelled"
            
        return {"success": True, "message": f"Backtest {action}d successfully"}
    
    except Exception as e:
        logger.error(f"Error controlling backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))
