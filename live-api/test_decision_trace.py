#!/usr/bin/env python3
"""
Test script for the DecisionTrace API.
This script creates a sample DecisionTrace and posts it to the API.
"""
import json
import sys
import time
import requests
from datetime import datetime, timezone

# Default API URL
API_URL = "http://localhost:8000"

def create_sample_trace():
    """Create a sample DecisionTrace object."""
    now = datetime.now(timezone.utc).isoformat()
    
    return {
        "trace_id": f"{now}-NVDA-test",
        "as_of": now,
        "symbol": "NVDA",
        "instrument": {
            "type": "equity",
            "symbol": "NVDA"
        },
        "account": {
            "mode": "paper",
            "broker": "tradier"
        },
        "market_context": {
            "regime": {
                "label": "Neutral",
                "confidence": 0.58
            },
            "volatility": {
                "vix": 17.2,
                "trend": "stable"
            },
            "sentiment": {
                "label": "bullish",
                "score": 0.62
            }
        },
        "signals": [
            {
                "source": "TA",
                "name": "RSI_14",
                "value": 32.1,
                "threshold": 35,
                "direction": "bullish"
            },
            {
                "source": "TA",
                "name": "MA_CROSS",
                "value": "up",
                "direction": "bullish"
            }
        ],
        "news_evidence": [
            {
                "url": "https://example.com/news/1",
                "headline": "Chip policy tailwind for NVDA",
                "snippet": "Export curbs eased for advanced chips, benefiting NVDA",
                "entities": ["NVDA", "chips", "policy"],
                "sentiment": "positive",
                "recency_min": 47,
                "credibility": "high"
            }
        ],
        "candidate_score": {
            "alpha": 0.79,
            "rank_in_universe": 1
        },
        "risk_gate": {
            "position_limits_ok": True,
            "portfolio_heat_ok": True,
            "drawdown_ok": True,
            "notes": ["All risk gates passed"]
        },
        "plan": {
            "action": "OPEN_LONG",
            "entry": {
                "type": "limit",
                "px": 5.2
            },
            "sizing": {
                "units": 2,
                "notional": 1040,
                "max_loss": 1040
            },
            "exits": {
                "stop": 4.8,
                "take_profit": 6.0
            },
            "expected_move": {
                "days": 2,
                "pct": 5.2,
                "p_up": 0.65
            },
            "strategyLabel": "Momentum Reversal"
        },
        "execution": {
            "status": "PROPOSED"
        },
        "explain_layman": "Buy 2 call contracts because momentum is turning up from oversold; risk capped to premium.",
        "explain_detail": [
            "RSI(14)=32 rising; MA cross-up in last 3 sessions.",
            "Policy/news tailwind on advanced chips.",
            "Risk gates OK; within per-trade cap."
        ]
    }

def post_trace(trace, url=f"{API_URL}/api/decision-traces"):
    """Post a trace to the API."""
    try:
        response = requests.post(url, json=trace)
        response.raise_for_status()
        print(f"Successfully posted trace: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error posting trace: {e}")
        return None

def get_traces(url=f"{API_URL}/api/decision-traces"):
    """Get all traces from the API."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"Successfully retrieved traces: {response.status_code}")
        print(f"Found {len(response.json())} traces")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting traces: {e}")
        return None

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "get":
        get_traces()
    else:
        trace = create_sample_trace()
        post_trace(trace)
        # Wait a moment and then get all traces
        time.sleep(1)
        get_traces()

if __name__ == "__main__":
    main()


