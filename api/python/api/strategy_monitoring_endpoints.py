from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.strategy_trial_workflow import StrategyTrialWorkflow
from trading_bot.core.performance_tracker import PerformanceTracker
from trading_bot.core.strategy_broker_router import StrategyBrokerRouter
from trading_bot.core.constants import StrategyStatus, StrategyPhase

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# Models for request and response
class StrategyIdRequest(BaseModel):
    strategy_id: str

class BulkStrategyRequest(BaseModel):
    strategy_ids: List[str]

class ClosePositionsRequest(BaseModel):
    strategy_id: str
    close_all: bool = True
    symbols: Optional[List[str]] = None

class ListStrategiesResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]

class ActionResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None

# Helper function to get required services
def get_services():
    service_registry = ServiceRegistry.get_instance()
    workflow = service_registry.get_service("strategy_trial_workflow")
    if not workflow:
        raise HTTPException(status_code=500, detail="Strategy trial workflow service not available")

    performance_tracker = service_registry.get_service("performance_tracker")
    if not performance_tracker:
        raise HTTPException(status_code=500, detail="Performance tracker service not available")
        
    broker_router = service_registry.get_service("strategy_broker_router")
    if not broker_router:
        raise HTTPException(status_code=500, detail="Strategy broker router service not available")
        
    return workflow, performance_tracker, broker_router

@router.get("/list", response_model=ListStrategiesResponse)
async def list_strategies():
    """
    List all registered strategies with their status, performance, and positions.
    This endpoint populates the Active Strategies Monitor.
    """
    workflow, performance_tracker, broker_router = get_services()
    
    # Get all strategies from workflow
    strategies = workflow.get_all_strategies()
    
    # Enhance with performance data and positions
    enhanced_strategies = []
    for strategy in strategies:
        strategy_id = strategy["id"]
        
        # Get performance metrics
        performance = performance_tracker.get_strategy_metrics(strategy_id)
        
        # Get current positions
        broker = broker_router.get_broker_for_strategy(strategy_id)
        positions = []
        if broker:
            try:
                positions = broker.get_positions(strategy_id=strategy_id)
            except Exception as e:
                # Log error but continue
                print(f"Error getting positions for {strategy_id}: {e}")
        
        # Add to enhanced list
        enhanced_strategies.append({
            **strategy,
            "performance_metrics": performance,
            "positions": positions
        })
    
    return {
        "success": True,
        "data": {
            "strategies": enhanced_strategies
        }
    }

@router.post("/pause", response_model=ActionResponse)
async def pause_strategy(request: StrategyIdRequest):
    """
    Pause a running strategy. 
    This will prevent new signals but won't close positions.
    """
    workflow, _, _ = get_services()
    
    strategy_id = request.strategy_id
    if not workflow.pause_strategy(strategy_id):
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found or already paused")
    
    return {
        "success": True,
        "message": f"Strategy {strategy_id} paused successfully",
        "data": {
            "strategy_id": strategy_id,
            "status": "PAUSED"
        }
    }

@router.post("/resume", response_model=ActionResponse)
async def resume_strategy(request: StrategyIdRequest):
    """
    Resume a paused strategy.
    This will allow the strategy to generate new signals.
    """
    workflow, _, _ = get_services()
    
    strategy_id = request.strategy_id
    if not workflow.resume_strategy(strategy_id):
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found or not paused")
    
    return {
        "success": True,
        "message": f"Strategy {strategy_id} resumed successfully",
        "data": {
            "strategy_id": strategy_id,
            "status": "ACTIVE"
        }
    }

@router.post("/close-positions", response_model=ActionResponse)
async def close_positions(request: ClosePositionsRequest):
    """
    Close positions for a strategy.
    Can close all positions or specific symbols.
    """
    _, _, broker_router = get_services()
    
    strategy_id = request.strategy_id
    broker = broker_router.get_broker_for_strategy(strategy_id)
    
    if not broker:
        raise HTTPException(status_code=404, detail=f"No broker found for strategy {strategy_id}")
    
    try:
        # Get current positions
        positions = broker.get_positions(strategy_id=strategy_id)
        
        if not positions:
            return {
                "success": True,
                "message": f"No open positions found for strategy {strategy_id}"
            }
        
        # Determine which positions to close
        symbols_to_close = request.symbols if request.symbols and not request.close_all else [p["symbol"] for p in positions]
        
        # Close positions
        closed_positions = []
        for position in positions:
            if position["symbol"] in symbols_to_close:
                # Close position by placing opposite order
                quantity = abs(position["quantity"])
                side = "sell" if position["quantity"] > 0 else "buy"
                
                order = broker.place_equity_order(
                    symbol=position["symbol"],
                    side=side,
                    quantity=quantity,
                    order_type="market",
                    tags=["POSITION_CLOSE", strategy_id]
                )
                
                closed_positions.append({
                    "symbol": position["symbol"],
                    "quantity": position["quantity"],
                    "close_order_id": order.get("order_id") if order else None
                })
        
        return {
            "success": True,
            "message": f"Closed {len(closed_positions)} positions for strategy {strategy_id}",
            "data": {
                "strategy_id": strategy_id,
                "closed_positions": closed_positions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing positions: {str(e)}")

@router.post("/bulk-pause", response_model=ActionResponse)
async def bulk_pause_strategies(request: BulkStrategyRequest):
    """
    Pause multiple strategies at once.
    """
    workflow, _, _ = get_services()
    
    paused_strategies = []
    failed_strategies = []
    
    for strategy_id in request.strategy_ids:
        if workflow.pause_strategy(strategy_id):
            paused_strategies.append(strategy_id)
        else:
            failed_strategies.append(strategy_id)
    
    return {
        "success": len(paused_strategies) > 0,
        "message": f"Paused {len(paused_strategies)} strategies, {len(failed_strategies)} failed",
        "data": {
            "paused_strategies": paused_strategies,
            "failed_strategies": failed_strategies
        }
    }

@router.post("/bulk-resume", response_model=ActionResponse)
async def bulk_resume_strategies(request: BulkStrategyRequest):
    """
    Resume multiple strategies at once.
    """
    workflow, _, _ = get_services()
    
    resumed_strategies = []
    failed_strategies = []
    
    for strategy_id in request.strategy_ids:
        if workflow.resume_strategy(strategy_id):
            resumed_strategies.append(strategy_id)
        else:
            failed_strategies.append(strategy_id)
    
    return {
        "success": len(resumed_strategies) > 0,
        "message": f"Resumed {len(resumed_strategies)} strategies, {len(failed_strategies)} failed",
        "data": {
            "resumed_strategies": resumed_strategies,
            "failed_strategies": failed_strategies
        }
    }

@router.post("/emergency-stop", response_model=ActionResponse)
async def emergency_stop_strategies(request: BulkStrategyRequest):
    """
    Emergency stop for multiple strategies.
    This will pause the strategies AND close all positions.
    """
    workflow, _, broker_router = get_services()
    
    stopped_strategies = []
    failed_strategies = []
    
    for strategy_id in request.strategy_ids:
        try:
            # First pause the strategy
            workflow.pause_strategy(strategy_id)
            
            # Then close all positions
            broker = broker_router.get_broker_for_strategy(strategy_id)
            if broker:
                positions = broker.get_positions(strategy_id=strategy_id)
                
                for position in positions:
                    quantity = abs(position["quantity"])
                    side = "sell" if position["quantity"] > 0 else "buy"
                    
                    broker.place_equity_order(
                        symbol=position["symbol"],
                        side=side,
                        quantity=quantity,
                        order_type="market",
                        tags=["EMERGENCY_STOP", strategy_id]
                    )
            
            stopped_strategies.append(strategy_id)
        except Exception as e:
            failed_strategies.append({
                "strategy_id": strategy_id,
                "error": str(e)
            })
    
    return {
        "success": len(stopped_strategies) > 0,
        "message": f"Emergency stopped {len(stopped_strategies)} strategies, {len(failed_strategies)} failed",
        "data": {
            "stopped_strategies": stopped_strategies,
            "failed_strategies": failed_strategies
        }
    }

@router.get("/{strategy_id}/summary", response_model=ActionResponse)
async def get_strategy_summary(strategy_id: str):
    """
    Get detailed summary for a specific strategy.
    Includes current status, performance, positions, etc.
    """
    workflow, performance_tracker, broker_router = get_services()
    
    # Get strategy info
    strategy_info = workflow.get_strategy_info(strategy_id)
    if not strategy_info:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    
    # Get performance metrics
    performance = performance_tracker.get_strategy_metrics(strategy_id)
    
    # Get positions
    broker = broker_router.get_broker_for_strategy(strategy_id)
    positions = []
    orders = []
    if broker:
        try:
            positions = broker.get_positions(strategy_id=strategy_id)
            orders = broker.get_orders(strategy_id=strategy_id)
        except Exception as e:
            print(f"Error getting positions/orders for {strategy_id}: {e}")
    
    # Get recent trades
    trades = performance_tracker.get_recent_trades(strategy_id, limit=20)
    
    return {
        "success": True,
        "message": f"Strategy {strategy_id} summary retrieved",
        "data": {
            "strategy": strategy_info,
            "performance": performance,
            "positions": positions,
            "orders": orders,
            "recent_trades": trades
        }
    }

@router.get("/phase/{phase}", response_model=ListStrategiesResponse)
async def list_strategies_by_phase(phase: str):
    """
    List strategies filtered by phase (PAPER_TRADE, LIVE, etc).
    Useful for getting only paper or only live strategies.
    """
    workflow, performance_tracker, _ = get_services()
    
    # Validate phase
    try:
        phase_enum = StrategyPhase[phase.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid phase: {phase}")
    
    # Get strategies
    strategies = workflow.get_strategies_by_phase(phase_enum)
    
    # Enhance with performance data
    enhanced_strategies = []
    for strategy in strategies:
        strategy_id = strategy["id"]
        performance = performance_tracker.get_strategy_metrics(strategy_id)
        
        enhanced_strategies.append({
            **strategy,
            "performance_metrics": performance
        })
    
    return {
        "success": True,
        "data": {
            "strategies": enhanced_strategies
        }
    }
