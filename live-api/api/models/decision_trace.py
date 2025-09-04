# api/models/decision_trace.py
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

class Instrument(BaseModel):
    type: Literal["equity", "option", "futures", "crypto", "forex"]
    symbol: str
    underlier: Optional[str] = None
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[Literal["call", "put"]] = None

class MarketRegime(BaseModel):
    label: str
    confidence: float
    since: Optional[str] = None

class MarketVolatility(BaseModel):
    vix: Optional[float] = None
    percentile: Optional[float] = None
    trend: Optional[Literal["rising", "falling", "stable"]] = None

class MarketSentiment(BaseModel):
    label: Literal["bullish", "bearish", "neutral", "mixed"]
    score: Optional[float] = None
    sources: Optional[List[str]] = None

class MarketBreadth(BaseModel):
    advDecRatio: Optional[float] = None
    newHighsLows: Optional[float] = None
    interpretation: Optional[str] = None

class MarketContext(BaseModel):
    regime: Optional[MarketRegime] = None
    volatility: Optional[MarketVolatility] = None
    sentiment: Optional[MarketSentiment] = None
    breadth: Optional[MarketBreadth] = None

class Signal(BaseModel):
    source: str
    name: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    direction: Optional[Literal["bullish", "bearish"]] = None
    lookback: Optional[str] = None

class NewsEvidence(BaseModel):
    url: str
    headline: str
    snippet: str
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = None
    recency_min: Optional[int] = None
    credibility: Optional[Literal["high", "medium", "low"]] = None

class CandidateScore(BaseModel):
    alpha: float
    rank_in_universe: Optional[int] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)

class RiskGate(BaseModel):
    position_limits_ok: Optional[bool] = None
    portfolio_heat_ok: Optional[bool] = None
    drawdown_ok: Optional[bool] = None
    notes: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    action: Literal["OPEN_LONG", "OPEN_SHORT", "CLOSE", "ADJUST"]
    entry: Dict[str, Any] = Field(default_factory=dict)
    sizing: Dict[str, Any] = Field(default_factory=dict)
    exits: Dict[str, Any] = Field(default_factory=dict)
    expected_move: Dict[str, Any] = Field(default_factory=dict)
    strategyLabel: Optional[str] = None

class Execution(BaseModel):
    status: Literal["PROPOSED", "BLOCKED", "SENT", "PARTIAL", "FILLED", "CANCELED"] = "PROPOSED"
    route: Optional[str] = None
    broker: Optional[str] = None
    broker_ids: Optional[List[str]] = None
    latency_ms: Optional[int] = None

class DecisionTrace(BaseModel):
    trace_id: str
    as_of: str
    schema_version: str = "1.0"
    symbol: str
    instrument: Optional[Instrument] = None
    account: Optional[Dict[str, Any]] = None
    market_context: Optional[MarketContext] = None
    signals: List[Signal] = Field(default_factory=list)
    news_evidence: List[NewsEvidence] = Field(default_factory=list)
    candidate_score: Optional[CandidateScore] = None
    risk_gate: Optional[RiskGate] = None
    plan: Plan
    execution: Execution = Field(default_factory=Execution)
    explain_layman: str = ""
    explain_detail: List[str] = Field(default_factory=list)

