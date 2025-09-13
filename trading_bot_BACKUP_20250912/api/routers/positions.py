"""
Position Management API routes.
Provides endpoints for managing and monitoring portfolio positions.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Import auth service
from trading_bot.auth.service import AuthService

# Import connection to trading engine (replace with your actual trading engine connection)
from trading_bot.core.engine import trading_engine

# Import WebSocket manager for real-time updates
from trading_bot.api.websocket_manager import enabled_manager

logger = logging.getLogger("TradingBotAPI.Positions")

router = APIRouter()

class Position(BaseModel):
    """Model for position information."""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Position quantity")
    entry_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_percent: float = Field(..., description="Unrealized profit/loss percentage")
    cost_basis: float = Field(..., description="Total cost basis")
    market_value: float = Field(..., description="Current market value")
    account: str = Field(..., description="Account identifier")
    strategy_id: Optional[str] = Field(None, description="Strategy that opened the position")
    strategy_name: Optional[str] = Field(None, description="Strategy name")
    open_date: str = Field(..., description="Date position was opened")

class ClosePositionRequest(BaseModel):
    """Model for closing a position."""
    symbol: str = Field(..., description="Symbol to close")
    quantity: Optional[float] = Field(None, description="Quantity to close (if None, close all)")
    order_type: str = Field("market", description="Order type for the close")
    limit_price: Optional[float] = Field(None, description="Limit price if order_type is limit")
    stop_price: Optional[float] = Field(None, description="Stop price if order_type is stop")

class EmergencyActionRequest(BaseModel):
    """Model for emergency actions on the portfolio."""
    action: str = Field(..., description="Action to take (close_all, pause_trading, resume_trading)")
    account: str = Field(..., description="Account to apply action to")
    reason: Optional[str] = Field(None, description="Reason for emergency action")

@router.get("/positions", response_model=List[Position], tags=["Positions"])
async def get_positions(
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get all current positions for the specified account.
    """
    try:
        # Connect to your trading engine to get real position data
        try:
            positions = trading_engine.get_positions(account=account)
            
            # Format the positions for the API response
            response = []
            for position in positions:
                response.append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_percent": position.unrealized_pnl_percent,
                    "cost_basis": position.cost_basis,
                    "market_value": position.market_value,
                    "account": account,
                    "strategy_id": position.strategy_id,
                    "strategy_name": position.strategy_name,
                    "open_date": position.open_date.isoformat()
                })
            
            return response
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for position retrieval")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for position operations"
            )
            
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching positions: {str(e)}"
        )

@router.post("/positions/close", status_code=200, tags=["Positions"])
async def close_position(
    request: ClosePositionRequest,
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Close a position (fully or partially) for the specified account.
    """
    try:
        # Connect to your trading engine to close the position
        try:
            result = trading_engine.close_position(
                account=account,
                symbol=request.symbol,
                quantity=request.quantity,  # None means close all
                order_type=request.order_type,
                limit_price=request.limit_price,
                stop_price=request.stop_price
            )
            
            if result:
                # Broadcast the position close via WebSocket
                await enabled_manager.broadcast_to_channel(
                    channel="positions",
                    message_type="position_closed",
                    data={
                        "symbol": request.symbol,
                        "quantity": request.quantity or "all",
                        "account": account,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                return {
                    "message": f"Position {request.symbol} closed successfully",
                    "order_id": result.order_id if hasattr(result, 'order_id') else None
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Position for {request.symbol} not found or insufficient quantity"
                )
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for position closure")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for position operations"
            )
            
    except Exception as e:
        logger.error(f"Error closing position {request.symbol}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error closing position: {str(e)}"
        )

@router.post("/positions/emergency", status_code=200, tags=["Positions"])
async def emergency_action(
    request: EmergencyActionRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Execute emergency actions on the portfolio.
    This can include closing all positions, pausing all trading, etc.
    """
    try:
        # Validate the action
        valid_actions = ["close_all", "pause_trading", "resume_trading"]
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
            )
        
        # Connect to your trading engine to execute the emergency action
        try:
            # Log the emergency action
            logger.warning(
                f"Emergency action {request.action} requested for account {request.account} "
                f"by user {current_user.username}. Reason: {request.reason or 'None provided'}"
            )
            
            if request.action == "close_all":
                # This might take time, so run in background
                background_tasks.add_task(
                    trading_engine.close_all_positions,
                    account=request.account
                )
                
                # Broadcast the emergency action via WebSocket
                await enabled_manager.broadcast(
                    message_type="emergency_action",
                    data={
                        "action": "close_all",
                        "account": request.account,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "in_progress",
                        "reason": request.reason
                    }
                )
                
                return {
                    "message": f"Emergency close all positions initiated for {request.account}",
                    "status": "in_progress"
                }
                
            elif request.action == "pause_trading":
                result = trading_engine.pause_trading(account=request.account)
                
                # Broadcast the emergency action via WebSocket
                await enabled_manager.broadcast(
                    message_type="emergency_action",
                    data={
                        "action": "pause_trading",
                        "account": request.account,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "completed" if result else "failed",
                        "reason": request.reason
                    }
                )
                
                return {
                    "message": f"Trading paused for {request.account}",
                    "status": "completed" if result else "failed"
                }
                
            elif request.action == "resume_trading":
                result = trading_engine.resume_trading(account=request.account)
                
                # Broadcast the emergency action via WebSocket
                await enabled_manager.broadcast(
                    message_type="emergency_action",
                    data={
                        "action": "resume_trading",
                        "account": request.account,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "completed" if result else "failed",
                        "reason": request.reason
                    }
                )
                
                return {
                    "message": f"Trading resumed for {request.account}",
                    "status": "completed" if result else "failed"
                }
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for emergency actions")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for emergency operations"
            )
            
    except Exception as e:
        logger.error(f"Error executing emergency action {request.action}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error executing emergency action: {str(e)}"
        )

@router.get("/positions/{symbol}", response_model=Position, tags=["Positions"])
async def get_position(
    symbol: str = Path(..., description="Symbol to retrieve position for"),
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get details of a specific position by symbol.
    """
    try:
        # Connect to your trading engine to get the position
        try:
            position = trading_engine.get_position(
                symbol=symbol,
                account=account
            )
            
            if position:
                # Format the position for the API response
                response = {
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_percent": position.unrealized_pnl_percent,
                    "cost_basis": position.cost_basis,
                    "market_value": position.market_value,
                    "account": account,
                    "strategy_id": position.strategy_id,
                    "strategy_name": position.strategy_name,
                    "open_date": position.open_date.isoformat()
                }
                
                return response
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Position for {symbol} not found"
                )
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for position retrieval")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for position operations"
            )
            
    except Exception as e:
        logger.error(f"Error fetching position for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching position: {str(e)}"
        )
