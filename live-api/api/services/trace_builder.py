# api/services/trace_builder.py
from datetime import datetime, timezone
from typing import Dict, Any, List
from ..models.decision_trace import DecisionTrace, Plan, Execution

# NOTE: Replace the placeholder merge logic with your real sources
# (context service, scoring, risk, and evidence builder you already have)

def build_decision_trace(raw: Dict[str, Any]) -> DecisionTrace:
    """
    Server-side aggregator: take your current candidate + context + evidence objects and
    return a single DecisionTrace.

    Expected minimal `raw` keys (example):
      - symbol, plan (action, entry px, sizing), explain_layman, explain_detail
      - signals[], news_evidence[], market_context{}, risk_gate{}, candidate_score{}
    """
    now = datetime.now(timezone.utc).isoformat()

    d = DecisionTrace(
        trace_id=raw.get("trace_id") or f"{now}-{raw.get('symbol','UNK')}",
        as_of=raw.get("as_of", now),
        symbol=raw["symbol"],
        instrument=raw.get("instrument"),
        account=raw.get("account"),
        market_context=raw.get("market_context"),
        signals=raw.get("signals", []),
        news_evidence=raw.get("news_evidence", []),
        candidate_score=raw.get("candidate_score"),
        risk_gate=raw.get("risk_gate"),
        plan=Plan(**raw["plan"]),
        execution=Execution(**raw.get("execution", {})),
        explain_layman=raw.get("explain_layman", ""),
        explain_detail=raw.get("explain_detail", []),
    )

    # Guardrail: require at least 1 signal OR 1 news item
    if len(d.signals) == 0 and len(d.news_evidence) == 0:
        d.execution.status = "BLOCKED"
        d.explain_detail = ["Blocked: insufficient evidence (no signals or news attached)."] + d.explain_detail

    return d


def publish_decision_trace(raw: Dict[str, Any]) -> DecisionTrace:
    """
    Build a DecisionTrace and publish it to subscribers.
    """
    dt = build_decision_trace(raw)
    # Broadcast to WS subscribers and also consider persisting to DB here
    from .trace_bus import trace_bus_broadcast
    trace_bus_broadcast(dt)
    return dt

