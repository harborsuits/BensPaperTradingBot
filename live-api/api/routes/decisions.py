# api/routes/decisions.py
from typing import List
from fastapi import APIRouter, WebSocket, Depends, Query, Body
from ..models.decision_trace import DecisionTrace
from ..services.trace_bus import manager


router = APIRouter()


# In-memory ring buffer (swap to Mongo/Postgres as needed)
BUFFER: List[DecisionTrace] = []
MAX_BUF = 500


@router.get("/api/decision-traces", response_model=List[DecisionTrace])
def list_decisions(limit: int = Query(50, ge=1, le=MAX_BUF)):
    return list(reversed(BUFFER[-limit:]))


@router.post("/api/decision-traces", response_model=DecisionTrace)
def create_decision(dt: DecisionTrace = Body(...)):
    append_and_broadcast(dt)
    return dt


@router.websocket("/ws/decisions")
async def ws_decisions(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Passive push channel; if clients send pings, read and ignore
            await ws.receive_text()
    except Exception:
        manager.disconnect(ws)


# Helper you can call when you create a new DecisionTrace
def append_and_broadcast(dt: DecisionTrace):
    BUFFER.append(dt)
    if len(BUFFER) > MAX_BUF:
        del BUFFER[0:len(BUFFER)-MAX_BUF]
    # send to clients
    from ..services.trace_bus import trace_bus_broadcast
    trace_bus_broadcast(dt)

