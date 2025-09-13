#!/usr/bin/env python
"""
Recap Router

This module provides API endpoints for the nightly performance recap system,
including fetching recap data, running recaps, and retrieving available dates.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/recap", tags=["recap"])

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

# Get or create NightlyRecap instance
def get_recap_system():
    """Get or create a NightlyRecap instance"""
    try:
        from trading_bot.monitoring.nightly_recap import NightlyRecap
        return NightlyRecap()
    except ImportError:
        logger.warning("NightlyRecap module not found, using mock implementation")
        return MockRecapSystem()

# Mock implementation for development/testing
class MockRecapSystem:
    """Mock implementation of NightlyRecap for development/testing"""
    
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
                    "value": -5.2,
                    "change": 1.1,
                    "benchmark": -8.0,
                    "threshold": -10.0,
                    "status": "good"
                },
                {
                    "name": "Total P&L",
                    "value": 8245.32,
                    "change": 1250.75,
                    "status": "good"
                }
            ],
            "strategy_performance": [
                {
                    "strategy": "Momentum Strategy",
                    "current_weight": 0.35,
                    "sharpe_ratio": 1.7,
                    "win_rate": 68.0,
                    "max_drawdown": -4.5,
                    "total_pnl": 3850.25,
                    "trades_total": 28,
                    "alerts": [],
                    "status": "healthy"
                },
                {
                    "strategy": "Strangle Strategy",
                    "current_weight": 0.25,
                    "sharpe_ratio": 0.42,
                    "win_rate": 48.0,
                    "max_drawdown": -8.7,
                    "total_pnl": 1250.18,
                    "trades_total": 15,
                    "alerts": [
                        {
                            "metric": "sharpe_ratio",
                            "value": 0.42,
                            "threshold": 0.5,
                            "severity": "medium",
                            "message": "Sharpe ratio (0.42) below threshold (0.50)"
                        },
                        {
                            "metric": "win_rate_10d",
                            "value": 33.0,
                            "compared_to": 48.0,
                            "deterioration": -15.0,
                            "severity": "high",
                            "message": "10-day win rate (33.0%) has deteriorated by 15.0 percentage points compared to overall (48.0%)"
                        }
                    ],
                    "suggestion": {
                        "action": "reduce_allocation",
                        "current_weight": 0.25,
                        "suggested_weight": 0.15,
                        "reduction_percentage": 40.0,
                        "reason": "Significant deterioration in performance metrics",
                        "details": ["10-day win rate (33.0%) has deteriorated by 15.0 percentage points compared to overall (48.0%)"]
                    },
                    "status": "critical"
                },
                {
                    "strategy": "Calendar Spread Strategy",
                    "current_weight": 0.15,
                    "sharpe_ratio": 0.95,
                    "win_rate": 62.0,
                    "max_drawdown": -12.5,
                    "total_pnl": 850.45,
                    "trades_total": 12,
                    "alerts": [
                        {
                            "metric": "max_drawdown",
                            "value": -12.5,
                            "threshold": -10.0,
                            "severity": "warning",
                            "message": "Max drawdown (-12.5%) exceeded threshold (-10.0%)"
                        },
                        {
                            "metric": "max_drawdown_5d",
                            "value": -8.2,
                            "compared_to": -12.5,
                            "deterioration": 4.3,
                            "severity": "warning",
                            "message": "5-day max drawdown (-8.2%) has improved by 4.3 percentage points compared to overall (-12.5%)"
                        }
                    ],
                    "suggestion": {
                        "action": "reduce_allocation",
                        "current_weight": 0.15,
                        "suggested_weight": 0.1,
                        "reduction_percentage": 33.3,
                        "reason": "Minor deterioration in performance metrics on heavily weighted strategy",
                        "details": ["Max drawdown (-12.5%) exceeded threshold (-10.0%)"]
                    },
                    "status": "warning"
                },
                {
                    "strategy": "Gap Trading Strategy",
                    "current_weight": 0.15,
                    "sharpe_ratio": 1.2,
                    "win_rate": 52.0,
                    "max_drawdown": -7.8,
                    "total_pnl": 1450.12,
                    "trades_total": 22,
                    "alerts": [],
                    "status": "healthy"
                },
                {
                    "strategy": "Range Strategy",
                    "current_weight": 0.1,
                    "sharpe_ratio": 1.1,
                    "win_rate": 58.0,
                    "max_drawdown": -5.2,
                    "total_pnl": 844.32,
                    "trades_total": 18,
                    "alerts": [],
                    "status": "healthy"
                }
            ],
            "historical_equity": [
                {"date": (date_obj - timedelta(days=30)).strftime('%Y-%m-%d'), "value": 128500.0},
                {"date": (date_obj - timedelta(days=25)).strftime('%Y-%m-%d'), "value": 131250.0},
                {"date": (date_obj - timedelta(days=20)).strftime('%Y-%m-%d'), "value": 134800.0},
                {"date": (date_obj - timedelta(days=15)).strftime('%Y-%m-%d'), "value": 138200.0},
                {"date": (date_obj - timedelta(days=10)).strftime('%Y-%m-%d'), "value": 140500.0},
                {"date": (date_obj - timedelta(days=5)).strftime('%Y-%m-%d'), "value": 143750.0},
                {"date": (date_obj - timedelta(days=1)).strftime('%Y-%m-%d'), "value": 146571.81},
                {"date": formatted_date, "value": 147822.56}
            ]
        }

# Initialize recap system (lazy loaded)
recap_system = None

def get_recap():
    """Get the recap system instance (lazy initialization)"""
    global recap_system
    if recap_system is None:
        recap_system = get_recap_system()
    return recap_system

@router.get("/dates", response_model=RecapDateResponse)
async def get_recap_dates():
    """
    Get available recap dates
    
    Returns:
        List of dates for which recap reports are available
    """
    try:
        # Check reports directory
        if not os.path.exists(REPORTS_DIR):
            return RecapDateResponse(success=True, dates=[])
        
        # Look for report files
        reports = []
        for filename in os.listdir(REPORTS_DIR):
            if filename.endswith('.json') and '_recap_' in filename:
                # Extract date from filename (format should be YYYYMMDD_recap_*.json)
                try:
                    date_str = filename.split('_')[0]
                    date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                    reports.append(date)
                except (IndexError, ValueError):
                    continue
        
        # Sort by date descending (most recent first)
        reports.sort(reverse=True)
        
        return RecapDateResponse(success=True, dates=reports)
    
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
        if not date:
            # Default to most recent date
            dates_response = await get_recap_dates()
            if not dates_response.success or not dates_response.dates:
                return RecapDataResponse(success=False, error="No recap data available")
            date = dates_response.dates[0]
        
        # Format date for filename
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')
        except ValueError:
            return RecapDataResponse(success=False, error="Invalid date format")
        
        # Look for recap file
        for filename in os.listdir(REPORTS_DIR):
            if filename.startswith(f"{date_str}_recap_") and filename.endswith('.json'):
                file_path = os.path.join(REPORTS_DIR, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return RecapDataResponse(success=True, data=data)
        
        return RecapDataResponse(success=False, error=f"No recap data found for {date}")
    
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
        # Initialize with empty data
        result = {
            "success": True,
            "message": "Recap started in background"
        }
        
        # Run recap in background
        background_tasks.add_task(_run_recap_task, force_date)
        
        return RecapTriggerResponse(**result)
    
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
            # Trigger optimization job
            _trigger_optimization(strategy)
            message = f"Optimization started for {strategy if strategy else 'all strategies'}"
        
        elif action_type == "retrain":
            # Trigger retraining job
            _trigger_retraining(strategy)
            message = f"Retraining started for {strategy if strategy else 'all strategies'}"
        
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
        recap = get_recap()
        
        # Get controller from recap system
        controller = recap.controller
        if not controller:
            # Try to initialize controller
            if not recap.paper_trading:
                return RecapActionResponse(
                    success=False, 
                    error="Trading system not available"
                )
            controller = recap.paper_trading.get_controller()
        
        if not controller:
            # Mock success in development mode
            return RecapActionResponse(
                success=True, 
                message=f"[MOCK] Updated weight for {update.strategy} to {update.weight:.2f}"
            )
        
        # Update strategy weight
        controller.update_strategy_weight(update.strategy, update.weight)
        
        return RecapActionResponse(
            success=True, 
            message=f"Updated weight for {update.strategy} to {update.weight:.2f}"
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
        recap = get_recap()
        success, message = recap.run_nightly_recap(force_date)
        
        if success:
            logger.info("Recap completed successfully")
        else:
            logger.error(f"Recap failed: {message}")
    
    except Exception as e:
        logger.error(f"Error in recap background task: {e}")

def _trigger_optimization(strategy: str = None):
    """
    Trigger optimization job
    
    Args:
        strategy: Optional strategy name
    """
    # Implement optimization trigger
    logger.info(f"Triggering optimization for {strategy or 'all strategies'}")
    # This would be implemented based on your optimization system

def _trigger_retraining(strategy: str = None):
    """
    Trigger retraining job
    
    Args:
        strategy: Optional strategy name
    """
    # Implement retraining trigger
    logger.info(f"Triggering retraining for {strategy or 'all strategies'}")
    # This would be implemented based on your retraining system
