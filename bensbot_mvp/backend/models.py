from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ---- Account / Positions ----
class AccountBalance(BaseModel):
    equity: float = 0.0
    cash: float = 0.0
    day_pl_dollar: float = 0.0
    day_pl_pct: float = 0.0

class Position(BaseModel):
    symbol: str
    qty: float
    avg_price: float
    last: float
    pl_dollar: float
    pl_pct: float
    stop: Optional[float] = None
    take_profit: Optional[float] = None

class Order(BaseModel):
    id: str
    symbol: str
    side: str  # BUY|SELL
    qty: float
    type: str  # market|limit
    limit_price: Optional[float] = None
    status: str  # open|filled|canceled|rejected
    ts: datetime

class LiveSignal(BaseModel):
    ts: datetime
    strategy: str
    symbol: str
    action: str  # buy|sell|hold
    size: float
    reason: str

class StrategyCard(BaseModel):
    id: str
    name: str
    active: bool
    exposure_pct: float
    last_signal_time: Optional[datetime] = None
    last_signal_strength: Optional[float] = None
    p_l_30d: float = 0.0

# ---- Risk / Health ----
class RiskStatus(BaseModel):
    portfolio_heat: float
    dd_pct: float
    concentration_flag: bool
    blocks: List[str] = []

class Health(BaseModel):
    broker: str  # UP|DOWN
    data: str    # UP|DEGRADED|DOWN
    last_heartbeat: datetime

# ---- Orders: create/cancel ----
class PlaceOrderRequest(BaseModel):
    symbol: str
    side: str
    qty: float
    type: str = "market"
    limit_price: Optional[float] = None

class PlaceOrderResponse(BaseModel):
    order_id: str

# ---- Jobs ----
class JobStartResponse(BaseModel):
    job_id: str

class JobStatus(BaseModel):
    job_id: str
    status: str   # QUEUED|RUNNING|DONE|ERROR
    progress: int # 0-100
    result_ref: Optional[str] = None
    error: Optional[str] = None
