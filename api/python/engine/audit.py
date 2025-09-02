"""
Decision audit system for trading decisions.

This module implements a decision audit trail to record and explain
all trading decisions, including those that were rejected.
"""

import time
import json
import logging
from typing import Dict, List, Optional, TypedDict, Literal, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class DecisionAudit(TypedDict):
    """Audit record for a trading decision."""
    id: str
    ts: int  # Timestamp in milliseconds
    instrument: str
    symbol: str
    action: Literal["OPEN", "ADJUST", "SKIP"]
    score: Optional[float]
    reasons: List[str]
    policy_version: str
    gates_failed: Optional[List[str]]


# In-memory ring buffer for recent decisions
_decision_ring: List[DecisionAudit] = []
_MAX_RING_SIZE = 500


def log_decision(decision: DecisionAudit) -> None:
    """
    Log a trading decision to the audit trail.
    
    Args:
        decision: The decision audit record
    """
    # Add to in-memory ring buffer
    _decision_ring.append(decision)
    if len(_decision_ring) > _MAX_RING_SIZE:
        _decision_ring.pop(0)
    
    # Log to file system
    try:
        log_entry = {
            **decision,
            "timestamp": datetime.fromtimestamp(decision["ts"] / 1000).isoformat()
        }
        logger.info(f"DECISION: {json.dumps(log_entry)}")
    except Exception as e:
        logger.error(f"Failed to log decision: {e}")
    
    # Optionally send to external audit system
    # This could be a database, API call, etc.
    # Example: post_to_audit_api(decision)


def recent_decisions(limit: int = 50) -> List[DecisionAudit]:
    """
    Get recent trading decisions from the audit trail.
    
    Args:
        limit: Maximum number of decisions to return
        
    Returns:
        List of recent decision audit records
    """
    return list(reversed(_decision_ring[-limit:]))


def get_decision_stats(time_window_ms: int = 24 * 60 * 60 * 1000) -> Dict[str, Any]:
    """
    Get statistics about recent trading decisions.
    
    Args:
        time_window_ms: Time window in milliseconds
        
    Returns:
        Dictionary with decision statistics
    """
    now = int(time.time() * 1000)
    cutoff = now - time_window_ms
    
    recent = [d for d in _decision_ring if d["ts"] >= cutoff]
    
    # Count by action
    action_counts = {
        "OPEN": 0,
        "ADJUST": 0,
        "SKIP": 0
    }
    
    # Count by instrument
    instrument_counts = {}
    
    # Count by gate failure
    gate_failure_counts = {}
    
    for decision in recent:
        action = decision["action"]
        instrument = decision["instrument"]
        
        # Update action counts
        action_counts[action] = action_counts.get(action, 0) + 1
        
        # Update instrument counts
        instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
        
        # Update gate failure counts
        if decision.get("gates_failed"):
            for gate in decision["gates_failed"]:
                gate_failure_counts[gate] = gate_failure_counts.get(gate, 0) + 1
    
    return {
        "total": len(recent),
        "by_action": action_counts,
        "by_instrument": instrument_counts,
        "gate_failures": gate_failure_counts,
        "time_window_ms": time_window_ms
    }
