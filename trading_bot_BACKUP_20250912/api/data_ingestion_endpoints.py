from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import random, string
from trading_bot.api.websocket_manager import enabled_manager

router = APIRouter(prefix="/data", tags=["DataIngestion"])

class DataSourceStatusModel(BaseModel):
    id: str
    name: str
    type: str
    status: str
    lastUpdate: datetime
    nextUpdateExpected: Optional[datetime] = None
    healthScore: int
    latency: Optional[float] = None
    errorRate: Optional[float] = None
    message: Optional[str] = None

class IngestionMetricsModel(BaseModel):
    totalSymbolsTracked: int
    activeSymbols: List[str]
    symbolsWithErrors: List[str]
    requestsLastHour: int
    dataPointsIngested: int
    lastFullSyncCompleted: datetime
    averageLatency: float
    errorRate: float

class PriceUpdateModel(BaseModel):
    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    timestamp: datetime
    source: str

class MarketDataBatchModel(BaseModel):
    prices: Dict[str, PriceUpdateModel]
    timestamp: datetime
    batchId: str
    source: str

# Summary model for overall data status
class DataStatusSummaryModel(BaseModel):
    timestamp: datetime
    sources: List[DataSourceStatusModel]
    metrics: IngestionMetricsModel

# In-memory store for recent price updates
_price_queue: List[PriceUpdateModel] = []

@router.get("/sources/status", response_model=List[DataSourceStatusModel])
async def get_data_source_status():
    now = datetime.utcnow()
    status = DataSourceStatusModel(
        id="market_data",
        name="Market Data Feed",
        type="market_data",
        status="online",
        lastUpdate=now,
        nextUpdateExpected=now + timedelta(seconds=1),
        healthScore=100,
        latency=10.5,
        errorRate=0.0,
        message="OK"
    )
    # Broadcast data source status to subscribed data channel
    await enabled_manager.broadcast_to_channel("data", "ingestion_sources_status", [status.dict()])
    return [status]

@router.get("/metrics", response_model=IngestionMetricsModel)
async def get_ingestion_metrics():
    now = datetime.utcnow()
    metrics = IngestionMetricsModel(
        totalSymbolsTracked=100,
        activeSymbols=["AAPL", "MSFT", "GOOGL"],
        symbolsWithErrors=[],
        requestsLastHour=random.randint(1000, 2000),
        dataPointsIngested=random.randint(10000, 20000),
        lastFullSyncCompleted=now - timedelta(minutes=5),
        averageLatency=random.uniform(5, 20),
        errorRate=0.0
    )
    # Broadcast ingestion metrics to subscribed data channel
    await enabled_manager.broadcast_to_channel("data", "ingestion_metrics", metrics.dict())
    return metrics

@router.get("/summary", response_model=DataStatusSummaryModel)
async def get_market_data_summary():
    sources = await get_data_source_status()
    metrics = await get_ingestion_metrics()
    summary = {
        "timestamp": datetime.utcnow(),
        "sources": [s.dict() for s in sources],
        "metrics": metrics.dict()
    }
    return summary

@router.get("/status", response_model=DataStatusSummaryModel)
async def get_data_status():
    """Return overall data ingestion status including sources and metrics."""
    return await get_market_data_summary()

@router.get("/prices", response_model=MarketDataBatchModel)
async def get_latest_prices(symbols: str = Query(..., description="Comma separated symbols")):
    syms = symbols.split(",")
    now = datetime.utcnow()
    prices: Dict[str, PriceUpdateModel] = {}
    for sym in syms:
        price = round(random.uniform(100, 200), 2)
        change = round(random.uniform(-1, 1), 2)
        update = PriceUpdateModel(
            symbol=sym,
            price=price,
            change=change,
            changePercent=round((change / price) * 100, 2),
            volume=random.randint(1000, 10000),
            timestamp=now,
            source="market_data"
        )
        prices[sym] = update
        _price_queue.append(update)
        if len(_price_queue) > 1000:
            _price_queue.pop(0)
        # Broadcast individual price update to subscribed data channel
        await enabled_manager.broadcast_to_channel("data", "price_update", update.dict())
    batch = MarketDataBatchModel(
        prices=prices,
        timestamp=now,
        batchId=''.join(random.choices(string.ascii_lowercase + string.digits, k=8)),
        source="market_data"
    )
    return batch

@router.get("/config", response_model=Dict[str, Any])
async def get_data_ingestion_config():
    config = {
        "dataSources": [
            {"id": "market_data", "name": "Market Data Feed", "enabled": True, "priority": 1, "refreshInterval": 1}
        ],
        "fallbackStrategies": [
            {"sourceId": "market_data", "fallbackSourceIds": []}
        ],
        "refreshRates": {"priceData": 1, "fundamentals": 60, "news": 300}
    }
    return config

@router.put("/config", response_model=Dict[str, str])
async def update_data_ingestion_config(config: Dict[str, Any]):
    # Stub: accept any config
    return {"success": "true", "message": "Config updated"}

@router.post("/refresh", response_model=Dict[str, str])
async def trigger_data_refresh(symbols: Optional[str] = None):
    # Stub: immediate refresh
    return {"success": "true", "message": "Data refresh triggered"}

@router.websocket("/ws/updates")
async def data_updates_ws(websocket: WebSocket):
    """WebSocket endpoint for data ingestion real-time updates."""
    await enabled_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        enabled_manager.disconnect(websocket)
