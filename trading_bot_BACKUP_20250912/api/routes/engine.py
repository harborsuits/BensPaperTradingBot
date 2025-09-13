"""
Engine state and control endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
from datetime import datetime, timedelta
import uuid

router = APIRouter(prefix="/api/engine", tags=["engine"])

# In-memory state (would be persisted in a real implementation)
ENGINE_STATE = {
    "running": True,
    "last_tick_id": str(uuid.uuid4()),
    "last_tick_ts": int(time.time()),
    "next_tick_eta": int(time.time()) + 60,  # Default to 1 minute from now
    "tick_interval_seconds": 60,
    "total_ticks": 1,
    "uptime_seconds": 0,
    "start_time": int(time.time()),
}

class EngineStateResponse(BaseModel):
    running: bool
    tick_id: str
    last_tick_ts: int
    next_tick_eta: int
    tick_interval_seconds: int
    total_ticks: int
    uptime_seconds: int

class EngineToggleRequest(BaseModel):
    running: bool

@router.get("/state", response_model=EngineStateResponse)
async def get_engine_state():
    """
    Get the current state of the trading engine.
    """
    # Update uptime
    ENGINE_STATE["uptime_seconds"] = int(time.time()) - ENGINE_STATE["start_time"]
    
    # If running, simulate a tick if it's time
    if ENGINE_STATE["running"]:
        current_time = int(time.time())
        if current_time >= ENGINE_STATE["next_tick_eta"]:
            # Simulate a tick
            ENGINE_STATE["last_tick_id"] = str(uuid.uuid4())
            ENGINE_STATE["last_tick_ts"] = current_time
            ENGINE_STATE["next_tick_eta"] = current_time + ENGINE_STATE["tick_interval_seconds"]
            ENGINE_STATE["total_ticks"] += 1
    
    return EngineStateResponse(
        running=ENGINE_STATE["running"],
        tick_id=ENGINE_STATE["last_tick_id"],
        last_tick_ts=ENGINE_STATE["last_tick_ts"],
        next_tick_eta=ENGINE_STATE["next_tick_eta"],
        tick_interval_seconds=ENGINE_STATE["tick_interval_seconds"],
        total_ticks=ENGINE_STATE["total_ticks"],
        uptime_seconds=ENGINE_STATE["uptime_seconds"]
    )

@router.post("/toggle", response_model=EngineStateResponse)
async def toggle_engine(request: EngineToggleRequest):
    """
    Toggle the trading engine on or off.
    """
    ENGINE_STATE["running"] = request.running
    
    # If turning on, schedule next tick
    if request.running:
        ENGINE_STATE["next_tick_eta"] = int(time.time()) + ENGINE_STATE["tick_interval_seconds"]
    
    # Update uptime
    ENGINE_STATE["uptime_seconds"] = int(time.time()) - ENGINE_STATE["start_time"]
    
    return EngineStateResponse(
        running=ENGINE_STATE["running"],
        tick_id=ENGINE_STATE["last_tick_id"],
        last_tick_ts=ENGINE_STATE["last_tick_ts"],
        next_tick_eta=ENGINE_STATE["next_tick_eta"],
        tick_interval_seconds=ENGINE_STATE["tick_interval_seconds"],
        total_ticks=ENGINE_STATE["total_ticks"],
        uptime_seconds=ENGINE_STATE["uptime_seconds"]
    )

@router.post("/dev/force_tick")
async def force_tick():
    """
    Force a tick for development purposes.
    """
    ENGINE_STATE["last_tick_id"] = str(uuid.uuid4())
    ENGINE_STATE["last_tick_ts"] = int(time.time())
    ENGINE_STATE["next_tick_eta"] = int(time.time()) + ENGINE_STATE["tick_interval_seconds"]
    ENGINE_STATE["total_ticks"] += 1
    
    # In a real implementation, this would trigger the actual engine tick
    # and emit the appropriate events
    
    return {
        "success": True,
        "tick_id": ENGINE_STATE["last_tick_id"],
        "message": "Forced tick executed successfully"
    }
