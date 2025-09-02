from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Mode = Literal["paper", "live", "stop"]


class SafetyStatus(BaseModel):
    mode: Mode
    kill_switch: bool = False
    daily_loss_limit: float = 0.0
    cooldown_active: bool = False
    updated_at: str


class Position(BaseModel):
    symbol: str
    qty: float = 0
    avg_price: float = 0.0
    mkt_price: float = 0.0
    unrealized_pl: float = 0.0
    pl_pct: float = 0.0


class Portfolio(BaseModel):
    mode: Mode
    equity: float = 0.0
    cash: float = 0.0
    positions: List[Position] = Field(default_factory=list)
    updated_at: str


class Strategy(BaseModel):
    id: str
    name: str
    status: Literal["running", "paused"] = "running"
    score: float = 0.0
    description: Optional[str] = None


class Strategies(BaseModel):
    items: List[Strategy] = Field(default_factory=list)


class Decision(BaseModel):
    id: str
    timestamp: str
    symbol: str
    action: str
    qty: Optional[float] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None


class Decisions(BaseModel):
    items: List[Decision] = Field(default_factory=list)


class ContextItem(BaseModel):
    id: str
    title: str
    symbol: Optional[str] = None
    headline: Optional[str] = None
    ts: str


class MarketContext(BaseModel):
    regime: str
    volatility: float
    sentiment: float
    items: List[ContextItem] = Field(default_factory=list)


