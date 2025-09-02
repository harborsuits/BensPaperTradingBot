"""
Opportunities API endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import time
from datetime import datetime
import json
from pydantic import BaseModel
from enum import Enum

router = APIRouter(prefix="/api/opportunities", tags=["opportunities"])

class GateDecision(str, Enum):
    PASS = "PASS"
    SKIP = "SKIP"
    PROBE = "PROBE"

class Opportunity(BaseModel):
    id: str
    symbol: str
    kind: Optional[str] = None
    timestamp: Optional[int] = None
    
    # WHY
    reason_tags: Optional[List[str]] = None
    novelty_score: Optional[float] = None
    sentiment_z: Optional[float] = None
    iv_change_1d: Optional[float] = None
    headline: Optional[str] = None
    
    # FIT
    regime: Optional[str] = None
    regime_alignment: Optional[str] = None
    
    # COSTS & RISK
    fees_bps: Optional[float] = None
    slip_bps: Optional[float] = None
    spread_bps: Optional[float] = None
    spread_cap_bps: Optional[float] = None
    cluster_heat_delta: Optional[float] = None
    
    # PLAN SUGGESTION
    plan_strategy: Optional[str] = None
    plan_risk_usd: Optional[float] = None
    plan_target_r: Optional[float] = None
    plan_horizon: Optional[str] = None
    
    # GO/NO-GO
    meta_prob: Optional[float] = None
    meta_threshold: Optional[float] = None
    decision: Optional[GateDecision] = None
    skip_codes: Optional[List[str]] = None

# Sample opportunities for testing
SAMPLE_OPPORTUNITIES = [
    {
        "id": "1",
        "symbol": "SPY",
        "kind": "etf",
        "timestamp": int(time.time()),
        "reason_tags": ["tariff", "semiconductors"],
        "novelty_score": 0.85,
        "sentiment_z": -2.1,
        "iv_change_1d": 0.12,
        "regime": "risk_off",
        "regime_alignment": "bearish",
        "fees_bps": 6,
        "slip_bps": 3,
        "spread_bps": 24,
        "spread_cap_bps": 30,
        "cluster_heat_delta": 0.01,
        "plan_strategy": "put_debit_spread",
        "plan_risk_usd": 25,
        "plan_target_r": 2,
        "plan_horizon": "1–2 days",
        "meta_prob": 0.61,
        "meta_threshold": 0.62,
        "decision": "PROBE",
    },
    {
        "id": "2",
        "symbol": "NVDA",
        "kind": "equity",
        "timestamp": int(time.time()),
        "reason_tags": ["earnings", "ai_demand"],
        "novelty_score": 0.72,
        "sentiment_z": 1.8,
        "iv_change_1d": 0.25,
        "regime": "mixed",
        "regime_alignment": "bullish",
        "fees_bps": 8,
        "slip_bps": 5,
        "spread_bps": 18,
        "spread_cap_bps": 30,
        "cluster_heat_delta": 0.015,
        "plan_strategy": "call_debit_spread",
        "plan_risk_usd": 35,
        "plan_target_r": 1.8,
        "plan_horizon": "3–5 days",
        "meta_prob": 0.68,
        "meta_threshold": 0.62,
        "decision": "PASS",
    },
    {
        "id": "3",
        "symbol": "SOFI",
        "kind": "equity",
        "timestamp": int(time.time()),
        "reason_tags": ["fintech", "user_growth"],
        "novelty_score": 0.35,
        "sentiment_z": 0.9,
        "iv_change_1d": -0.05,
        "regime": "risk_on",
        "regime_alignment": "neutral",
        "fees_bps": 12,
        "slip_bps": 15,
        "spread_bps": 92,
        "spread_cap_bps": 90,
        "cluster_heat_delta": 0.008,
        "meta_prob": 0.58,
        "meta_threshold": 0.62,
        "decision": "SKIP",
        "skip_codes": ["SPREAD_TOO_WIDE", "META_PROB_LOW"],
    },
]

@router.get("", response_model=List[Opportunity])
async def get_opportunities(
    edge_min: float = Query(0.15, description="Minimum edge score"),
    spread_max: float = Query(80, description="Maximum spread in bps"),
    adv_min: float = Query(1000000, description="Minimum average daily volume"),
    timeframe: str = Query("ALL", description="Timeframe filter (ALL, ST, MT)"),
    hide_held: bool = Query(False, description="Hide symbols already in portfolio"),
    risk_on: bool = Query(True, description="Show only risk-on opportunities"),
):
    """
    Get a list of trading opportunities with detailed narrative information.
    """
    # In a real implementation, this would filter based on the query parameters
    # For now, return the sample opportunities
    return SAMPLE_OPPORTUNITIES

@router.post("/{opportunity_id}/probe")
async def probe_opportunity(opportunity_id: str):
    """
    Start a probe for an opportunity.
    """
    # Check if opportunity exists
    opportunity = next((o for o in SAMPLE_OPPORTUNITIES if o["id"] == opportunity_id), None)
    if not opportunity:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    # In a real implementation, this would start a probe
    return {"success": True, "message": f"Started probe for opportunity {opportunity_id}"}

@router.post("/{opportunity_id}/paper-order")
async def paper_order_opportunity(opportunity_id: str):
    """
    Submit a paper order for an opportunity.
    """
    # Check if opportunity exists
    opportunity = next((o for o in SAMPLE_OPPORTUNITIES if o["id"] == opportunity_id), None)
    if not opportunity:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    # In a real implementation, this would submit a paper order
    return {"success": True, "message": f"Submitted paper order for opportunity {opportunity_id}"}
