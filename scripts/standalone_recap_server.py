#!/usr/bin/env python
"""
Standalone Nightly Recap Server

This script starts a minimal FastAPI server that provides access to the nightly recap system,
avoiding the full trading engine dependencies that might cause import errors.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RecapAPI")

# Create FastAPI application
app = FastAPI(title="Trading Recap API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
router = APIRouter(prefix="/api/recap", tags=["recap"])

# Default reports directory
REPORTS_DIR = "reports/nightly"

# Define models
class StrategyWeightUpdate(BaseModel):
    strategy: str
    weight: float

class RecapDateResponse(BaseModel):
    success: bool
    dates: List[str]
    error: Optional[str] = None

class RecapDataResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RecapTriggerResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class RecapActionResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

# Mock Recap System
class MockRecapSystem:
    """Mock implementation of NightlyRecap for standalone mode"""
    
    def __init__(self):
        self.controller = None
        self.paper_trading = None
    
    def run_nightly_recap(self, force_date=None):
        """Run mock recap"""
        logger.info(f"Running mock recap for date: {force_date or 'today'}")
        # Create sample data file
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        date_str = datetime.now().strftime('%Y%m%d')
        if force_date:
            try:
                date_obj = datetime.strptime(force_date, '%Y-%m-%d')
                date_str = date_obj.strftime('%Y%m%d')
            except ValueError:
                pass
        
        file_path = os.path.join(REPORTS_DIR, f"{date_str}_recap_data.json")
        
        # Create sample data
        sample_data = self._generate_sample_data(date_str)
        
        with open(file_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        return True, "Mock recap completed successfully"
    
    def _generate_sample_data(self, date_str):
        """Generate sample recap data"""
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        formatted_date = date_obj.strftime('%Y-%m-%d')
        
        return {
            "date": formatted_date,
            "daily_pnl": 1250.75,
            "daily_return": 0.85,
            "equity_value": 147822.56,
            "total_trades": 18,
            "win_rate": 67.5,
            "benchmarks": {
                "SPY": {
                    "symbol": "SPY",
                    "date": formatted_date,
                    "close": 427.82,
                    "daily_return": 0.35
                },
                "VIX": {
                    "symbol": "VIX",
                    "date": formatted_date,
                    "close": 18.64,
                    "daily_return": -2.1
                }
            },
            "performance_metrics": [
                {
                    "name": "Sharpe Ratio",
                    "value": 1.8,
                    "change": 0.2,
                    "benchmark": 1.0,
                    "threshold": 0.5,
                    "status": "good"
                },
                {
                    "name": "Win Rate",
                    "value": 67.5,
                    "change": 2.5,
                    "benchmark": 55.0,
                    "threshold": 45.0,
                    "status": "good"
                },
                {
                    "name": "Max Drawdown",
                    "value": -4.2,
                    "change": 1.3,
                    "benchmark": -5.0,
                    "threshold": -10.0,
                    "status": "good"
                },
                {
                    "name": "Profit Factor",
                    "value": 2.1,
                    "change": 0.1,
                    "benchmark": 1.5,
                    "threshold": 1.2,
                    "status": "good"
                }
            ],
            "strategy_performance": [
                {
                    "name": "Momentum Strategy",
                    "pnl": 520.35,
                    "return": 1.2,
                    "win_rate": 71.4,
                    "trades": 7,
                    "alerts": [],
                    "status": "good"
                },
                {
                    "name": "Breakout Strategy",
                    "pnl": 320.75,
                    "return": 0.7,
                    "win_rate": 60.0,
                    "trades": 5,
                    "alerts": [],
                    "status": "good"
                },
                {
                    "name": "Reversal Strategy",
                    "pnl": -180.35,
                    "return": -0.4,
                    "win_rate": 33.3,
                    "trades": 3,
                    "alerts": [
                        {
                            "type": "warning",
                            "message": "Win rate below threshold (33.3% < 45.0%)",
                            "metric": "win_rate"
                        }
                    ],
                    "status": "warning"
                },
                {
                    "name": "Range Strategy",
                    "pnl": 590.00,
                    "return": 1.3,
                    "win_rate": 100.0,
                    "trades": 3,
                    "alerts": [],
                    "status": "good"
                }
            ],
            "insights": [
                {
                    "type": "performance",
                    "message": "Portfolio outperformed SPY by 0.50%",
                    "importance": "high"
                },
                {
                    "type": "strategy",
                    "message": "Range Strategy performed best with 100% win rate",
                    "importance": "medium"
                },
                {
                    "type": "warning",
                    "message": "Reversal Strategy win rate (33.3%) below threshold",
                    "importance": "high"
                },
                {
                    "type": "suggestion",
                    "message": "Consider reducing allocation to Reversal Strategy",
                    "importance": "medium",
                    "action": {
                        "type": "weight_adjustment",
                        "strategy": "Reversal Strategy",
                        "current": 0.25,
                        "suggested": 0.15
                    }
                },
                {
                    "type": "suggestion",
                    "message": "Consider increasing allocation to Range Strategy",
                    "importance": "medium",
                    "action": {
                        "type": "weight_adjustment",
                        "strategy": "Range Strategy",
                        "current": 0.25,
                        "suggested": 0.35
                    }
                }
            ],
            "historical_comparison": {
                "week": {
                    "return": 2.8,
                    "win_rate": 65.0,
                    "sharpe": 1.7
                },
                "month": {
                    "return": 5.2,
                    "win_rate": 63.5,
                    "sharpe": 1.65
                }
            },
            "market_regime": {
                "current": "bullish",
                "volatility": "medium",
                "trend_strength": "strong"
            }
        }

# Initialize recap system
recap_system = MockRecapSystem()

# API endpoints
@router.get("/dates", response_model=RecapDateResponse)
async def get_recap_dates():
    """
    Get available recap dates
    
    Returns:
        List of dates for which recap reports are available
    """
    try:
        # Get all files in the reports directory
        if not os.path.exists(REPORTS_DIR):
            return RecapDateResponse(success=True, dates=[])
        
        files = os.listdir(REPORTS_DIR)
        recap_files = [f for f in files if f.endswith("_recap_data.json")]
        
        # Extract dates from filenames
        dates = []
        for file in recap_files:
            try:
                date_str = file.split("_")[0]
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                formatted_date = date_obj.strftime("%Y-%m-%d")
                dates.append(formatted_date)
            except (ValueError, IndexError):
                continue
        
        # Sort dates in descending order
        dates.sort(reverse=True)
        
        return RecapDateResponse(success=True, dates=dates)
    
    except Exception as e:
        logger.error(f"Error getting recap dates: {e}")
        return RecapDateResponse(success=False, error=str(e), dates=[])

@router.get("/data", response_model=RecapDataResponse)
async def get_recap_data(date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """
    Get recap data for a specific date
    
    Args:
        date: Date in YYYY-MM-DD format
    
    Returns:
        Recap data for the specified date
    """
    try:
        # If no date provided, use latest available
        if not date:
            dates_response = await get_recap_dates()
            if not dates_response.success or not dates_response.dates:
                return RecapDataResponse(success=False, error="No recap data available")
            date = dates_response.dates[0]
        
        # Convert date to filename format
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_str = date_obj.strftime("%Y%m%d")
        except ValueError:
            return RecapDataResponse(success=False, error=f"Invalid date format: {date}")
        
        # Load recap data
        file_path = os.path.join(REPORTS_DIR, f"{date_str}_recap_data.json")
        if not os.path.exists(file_path):
            return RecapDataResponse(success=False, error=f"No recap data available for {date}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return RecapDataResponse(success=True, data=data)
    
    except Exception as e:
        logger.error(f"Error getting recap data: {e}")
        return RecapDataResponse(success=False, error=str(e))

@router.post("/trigger", response_model=RecapTriggerResponse)
async def trigger_recap(background_tasks: BackgroundTasks, force_date: str = None):
    """
    Trigger a new recap
    
    Args:
        force_date: Optional date to generate recap for (YYYY-MM-DD)
        
    Returns:
        Status of the recap trigger
    """
    try:
        # Validate date format if provided
        if force_date:
            try:
                datetime.strptime(force_date, "%Y-%m-%d")
            except ValueError:
                return RecapTriggerResponse(success=False, error=f"Invalid date format: {force_date}")
        
        # Run recap in background
        background_tasks.add_task(_run_recap_task, force_date)
        
        return RecapTriggerResponse(
            success=True,
            message=f"Recap started for {force_date if force_date else 'today'}"
        )
    
    except Exception as e:
        logger.error(f"Error triggering recap: {e}")
        return RecapTriggerResponse(success=False, error=str(e))

@router.post("/action", response_model=RecapActionResponse)
async def perform_recap_action(action_type: str, strategy: str = None, params: Dict = None):
    """
    Perform an action based on recap
    
    Args:
        action_type: Type of action (optimize, retrain, etc.)
        strategy: Optional strategy name
        params: Optional parameters for the action
    
    Returns:
        Status of the action
    """
    try:
        # Handle different action types
        if action_type == "optimize":
            # Mock optimization job
            message = f"[MOCK] Optimization started for {strategy if strategy else 'all strategies'}"
        
        elif action_type == "retrain":
            # Mock retraining job
            message = f"[MOCK] Retraining started for {strategy if strategy else 'all strategies'}"
        
        else:
            return RecapActionResponse(success=False, error=f"Unknown action type: {action_type}")
        
        return RecapActionResponse(success=True, message=message)
    
    except Exception as e:
        logger.error(f"Error performing recap action: {e}")
        return RecapActionResponse(success=False, error=str(e))

@router.post("/strategy/weight", response_model=RecapActionResponse)
async def update_strategy_weight(update: StrategyWeightUpdate):
    """
    Update strategy weight based on recap suggestion
    
    Args:
        update: Strategy weight update request
    
    Returns:
        Status of the update
    """
    try:
        # Mock strategy weight update
        logger.info(f"[MOCK] Updating weight for {update.strategy} to {update.weight:.2f}")
        
        return RecapActionResponse(
            success=True, 
            message=f"[MOCK] Updated weight for {update.strategy} to {update.weight:.2f}"
        )
    
    except Exception as e:
        logger.error(f"Error updating strategy weight: {e}")
        return RecapActionResponse(success=False, error=str(e))

def _run_recap_task(force_date: str = None):
    """
    Run recap task in background
    
    Args:
        force_date: Optional date to generate recap for
    """
    try:
        logger.info(f"Running recap task for date: {force_date or 'today'}")
        success, message = recap_system.run_nightly_recap(force_date)
        
        if success:
            logger.info("Recap completed successfully")
        else:
            logger.error(f"Recap failed: {message}")
    
    except Exception as e:
        logger.error(f"Error in recap background task: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Trading Recap API",
        "description": "API for nightly trading recaps and performance analysis",
        "version": "1.0.0",
        "endpoints": [
            "/api/recap/dates",
            "/api/recap/data",
            "/api/recap/trigger",
            "/api/recap/action",
            "/api/recap/strategy/weight"
        ]
    }

# Include router
app.include_router(router)

if __name__ == "__main__":
    # Create required directories
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Start server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        reload=True
    )
