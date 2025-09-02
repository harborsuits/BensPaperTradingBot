"""
Order Management API routes.
Provides endpoints for managing and monitoring orders.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Import auth service
from trading_bot.auth.service import AuthService

# Import connection to trading engine (replace with your actual trading engine connection)
from trading_bot.core.engine import trading_engine

# Import WebSocket manager for real-time updates
from trading_bot.api.websocket_manager import enabled_manager

logger = logging.getLogger("TradingBotAPI.Orders")

router = APIRouter()

class OrderCreate(BaseModel):
    """Model for creating a new order."""
    symbol: str = Field(..., description="The trading symbol (e.g., AAPL)")
    side: str = Field(..., description="Order side", example="buy")
    quantity: float = Field(..., gt=0, description="Order quantity")
    order_type: str = Field(..., description="Order type", example="market")
    limit_price: Optional[float] = Field(None, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")
    time_in_force: str = Field("day", description="Time in force", example="day")
    strategy_id: Optional[str] = Field(None, description="Associated strategy ID")

class OrderResponse(BaseModel):
    """Model for order information returned by the API."""
    id: str = Field(..., description="Order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., description="Order quantity")
    order_type: str = Field(..., description="Order type (market/limit/stop)")
    status: str = Field(..., description="Order status")
    created_at: str = Field(..., description="Order creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    filled_quantity: float = Field(0, description="Quantity filled")
    filled_price: Optional[float] = Field(None, description="Average fill price")
    limit_price: Optional[float] = Field(None, description="Limit price if applicable")
    stop_price: Optional[float] = Field(None, description="Stop price if applicable")
    time_in_force: str = Field(..., description="Time in force")
    account: str = Field(..., description="Account identifier")
    strategy_id: Optional[str] = Field(None, description="Associated strategy ID")
    strategy_name: Optional[str] = Field(None, description="Associated strategy name")

class OrderUpdate(BaseModel):
    """Model for updating an existing order."""
    quantity: Optional[float] = Field(None, gt=0, description="New order quantity")
    limit_price: Optional[float] = Field(None, description="New limit price")
    stop_price: Optional[float] = Field(None, description="New stop price")
    time_in_force: Optional[str] = Field(None, description="New time in force")

@router.get("/orders", response_model=List[OrderResponse], tags=["Orders"])
async def get_open_orders(
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get all open orders for the specified account.
    """
    try:
        # Connect to your trading engine to get real order data
        try:
            orders = trading_engine.get_open_orders(account=account)
            
            # Format the orders for the API response
            response = []
            for order in orders:
                response.append({
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "order_type": order.order_type,
                    "status": order.status,
                    "created_at": order.created_at.isoformat(),
                    "updated_at": order.updated_at.isoformat(),
                    "filled_quantity": order.filled_quantity,
                    "filled_price": order.filled_price,
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "time_in_force": order.time_in_force,
                    "account": account,
                    "strategy_id": order.strategy_id,
                    "strategy_name": order.strategy_name
                })
            
            return response
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for order retrieval")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for order operations"
            )
            
    except Exception as e:
        logger.error(f"Error fetching open orders: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching open orders: {str(e)}"
        )

@router.post("/orders", response_model=OrderResponse, status_code=201, tags=["Orders"])
async def create_order(
    order: OrderCreate,
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Create a new order for the specified account.
    """
    try:
        # Validate order parameters
        if order.order_type == "limit" and order.limit_price is None:
            raise HTTPException(
                status_code=400,
                detail="Limit price is required for limit orders"
            )
        
        if order.order_type == "stop" and order.stop_price is None:
            raise HTTPException(
                status_code=400,
                detail="Stop price is required for stop orders"
            )
        
        # Connect to your trading engine to place the order
        try:
            new_order = trading_engine.place_order(
                account=account,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                strategy_id=order.strategy_id
            )
            
            # Format the new order for the API response
            response = {
                "id": new_order.id,
                "symbol": new_order.symbol,
                "side": new_order.side,
                "quantity": new_order.quantity,
                "order_type": new_order.order_type,
                "status": new_order.status,
                "created_at": new_order.created_at.isoformat(),
                "updated_at": new_order.updated_at.isoformat(),
                "filled_quantity": new_order.filled_quantity,
                "filled_price": new_order.filled_price,
                "limit_price": new_order.limit_price,
                "stop_price": new_order.stop_price,
                "time_in_force": new_order.time_in_force,
                "account": account,
                "strategy_id": new_order.strategy_id,
                "strategy_name": new_order.strategy_name
            }
            
            # Broadcast the new order via WebSocket
            await enabled_manager.broadcast_to_channel(
                channel="orders",
                message_type="order_created",
                data=response
            )
            
            return response
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for order creation")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for order operations"
            )
            
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error creating order: {str(e)}"
        )

@router.delete("/orders/{order_id}", status_code=200, tags=["Orders"])
async def cancel_order(
    order_id: str = Path(..., description="Order ID to cancel"),
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Cancel an open order by ID.
    """
    try:
        # Connect to your trading engine to cancel the order
        try:
            result = trading_engine.cancel_order(
                order_id=order_id,
                account=account
            )
            
            if result:
                # Broadcast the cancelled order via WebSocket
                await enabled_manager.broadcast_to_channel(
                    channel="orders",
                    message_type="order_cancelled",
                    data={"order_id": order_id, "account": account}
                )
                
                return {"message": f"Order {order_id} cancelled successfully"}
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Order {order_id} not found or already cancelled"
                )
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for order cancellation")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for order operations"
            )
            
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error cancelling order: {str(e)}"
        )

@router.get("/orders/{order_id}", response_model=OrderResponse, tags=["Orders"])
async def get_order(
    order_id: str = Path(..., description="Order ID to retrieve"),
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get details of a specific order by ID.
    """
    try:
        # Connect to your trading engine to get the order
        try:
            order = trading_engine.get_order(
                order_id=order_id,
                account=account
            )
            
            if order:
                # Format the order for the API response
                response = {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "order_type": order.order_type,
                    "status": order.status,
                    "created_at": order.created_at.isoformat(),
                    "updated_at": order.updated_at.isoformat(),
                    "filled_quantity": order.filled_quantity,
                    "filled_price": order.filled_price,
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "time_in_force": order.time_in_force,
                    "account": account,
                    "strategy_id": order.strategy_id,
                    "strategy_name": order.strategy_name
                }
                
                return response
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Order {order_id} not found"
                )
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for order retrieval")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for order operations"
            )
            
    except Exception as e:
        logger.error(f"Error fetching order {order_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching order: {str(e)}"
        )

@router.patch("/orders/{order_id}", response_model=OrderResponse, tags=["Orders"])
async def update_order(
    order_update: OrderUpdate,
    order_id: str = Path(..., description="Order ID to update"),
    account: str = Query("live", description="Account identifier (live/paper)"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Update an existing open order.
    Only applicable for certain order types and if the broker supports it.
    """
    try:
        # Connect to your trading engine to update the order
        try:
            updated_order = trading_engine.update_order(
                order_id=order_id,
                account=account,
                quantity=order_update.quantity,
                limit_price=order_update.limit_price,
                stop_price=order_update.stop_price,
                time_in_force=order_update.time_in_force
            )
            
            if updated_order:
                # Format the updated order for the API response
                response = {
                    "id": updated_order.id,
                    "symbol": updated_order.symbol,
                    "side": updated_order.side,
                    "quantity": updated_order.quantity,
                    "order_type": updated_order.order_type,
                    "status": updated_order.status,
                    "created_at": updated_order.created_at.isoformat(),
                    "updated_at": updated_order.updated_at.isoformat(),
                    "filled_quantity": updated_order.filled_quantity,
                    "filled_price": updated_order.filled_price,
                    "limit_price": updated_order.limit_price,
                    "stop_price": updated_order.stop_price,
                    "time_in_force": updated_order.time_in_force,
                    "account": account,
                    "strategy_id": updated_order.strategy_id,
                    "strategy_name": updated_order.strategy_name
                }
                
                # Broadcast the updated order via WebSocket
                await enabled_manager.broadcast_to_channel(
                    channel="orders",
                    message_type="order_updated",
                    data=response
                )
                
                return response
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Order {order_id} not found or cannot be updated"
                )
            
        except AttributeError:
            logger.error(f"Trading engine not properly initialized for order update")
            raise HTTPException(
                status_code=500,
                detail="Trading engine not properly configured for order operations"
            )
            
    except Exception as e:
        logger.error(f"Error updating order {order_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error updating order: {str(e)}"
        )
