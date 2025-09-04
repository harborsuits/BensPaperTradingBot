# api/services/trace_bus.py
from typing import List
from fastapi.websockets import WebSocket
from ..models.decision_trace import DecisionTrace
import json


class TraceConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload):
        data = json.dumps(payload)
        stale = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


manager = TraceConnectionManager()


# Simple facade used by services
async def trace_bus_broadcast_obj(obj):
    await manager.broadcast(obj)


# Sync-friendly wrapper if you call from non-async code
def trace_bus_broadcast(dt: DecisionTrace):
    # You can refactor to your app's event loop; here we keep it simple
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(trace_bus_broadcast_obj(dt.model_dump()))
    except RuntimeError:
        asyncio.run(trace_bus_broadcast_obj(dt.model_dump()))

