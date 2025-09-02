#!/usr/bin/env python3
"""
Example script demonstrating the decision router system.

This script creates sample opportunities for different instrument types
and processes them through the decision engine to generate trading decisions.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_bot.policy.types import Instrument, Regime
from trading_bot.policy.service import PolicyService
from trading_bot.engine.decision_engine import DecisionEngine
from trading_bot.engine.audit import recent_decisions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_opportunities():
    """Create sample opportunities for testing."""
    return [
        # ETF opportunities
        {
            "instrument": Instrument.ETF.value,
            "symbol": "SPY",
            "alpha": 0.7,
            "sentiment_boost": 0.5,
            "est_cost_bps": 1.5,
            "risk_penalty": 0.1,
            "price": 450.25,
            "volume": 1000000,
            "size_budget_usd": 10000,
            "meta": {
                "venue": "NYSE",
                "jurisdiction": "US",
                "adv_usd": 50000000,
                "spread_bps": 1.2
            }
        },
        {
            "instrument": Instrument.ETF.value,
            "symbol": "QQQ",
            "alpha": 0.6,
            "sentiment_boost": 0.4,
            "est_cost_bps": 1.8,
            "risk_penalty": 0.15,
            "price": 380.75,
            "volume": 800000,
            "size_budget_usd": 10000,
            "meta": {
                "venue": "NASDAQ",
                "jurisdiction": "US",
                "adv_usd": 40000000,
                "spread_bps": 1.5
            }
        },
        # Crypto opportunities
        {
            "instrument": Instrument.CRYPTO.value,
            "symbol": "BTC-USD",
            "alpha": 0.8,
            "sentiment_boost": 0.9,
            "est_cost_bps": 5.0,
            "risk_penalty": 0.4,
            "price": 52000.0,
            "volume": 500,
            "size_budget_usd": 5000,
            "meta": {
                "venue": "Coinbase",
                "jurisdiction": "US"
            }
        },
        {
            "instrument": Instrument.CRYPTO.value,
            "symbol": "ETH-USD",
            "alpha": 0.75,
            "sentiment_boost": 0.8,
            "est_cost_bps": 6.0,
            "risk_penalty": 0.45,
            "price": 3200.0,
            "volume": 1200,
            "size_budget_usd": 5000,
            "meta": {
                "venue": "Kraken",
                "jurisdiction": "US"
            }
        },
        # Options opportunities
        {
            "instrument": Instrument.OPTIONS.value,
            "symbol": "AAPL_230616C180",
            "alpha": 0.65,
            "sentiment_boost": 0.6,
            "est_cost_bps": 10.0,
            "risk_penalty": 0.5,
            "price": 3.50,
            "volume": 500,
            "size_budget_usd": 2000,
            "meta": {
                "venue": "OCC",
                "jurisdiction": "US",
                "iv_pctile": 60,
                "dte_days": 14
            }
        },
        {
            "instrument": Instrument.OPTIONS.value,
            "symbol": "MSFT_230616P300",
            "alpha": 0.6,
            "sentiment_boost": 0.5,
            "est_cost_bps": 12.0,
            "risk_penalty": 0.55,
            "price": 2.75,
            "volume": 300,
            "size_budget_usd": 2000,
            "meta": {
                "venue": "OCC",
                "jurisdiction": "US",
                "iv_pctile": 95,  # This should fail the IV percentile gate
                "dte_days": 14
            }
        },
        # Should fail venue whitelist
        {
            "instrument": Instrument.CRYPTO.value,
            "symbol": "XRP-USD",
            "alpha": 0.7,
            "sentiment_boost": 0.7,
            "est_cost_bps": 7.0,
            "risk_penalty": 0.5,
            "price": 0.55,
            "volume": 10000,
            "size_budget_usd": 5000,
            "meta": {
                "venue": "Binance",  # Not in whitelist
                "jurisdiction": "US"
            }
        },
        # Should fail jurisdiction
        {
            "instrument": Instrument.ETF.value,
            "symbol": "FTSE100",
            "alpha": 0.6,
            "sentiment_boost": 0.5,
            "est_cost_bps": 2.0,
            "risk_penalty": 0.2,
            "price": 7500.0,
            "volume": 500000,
            "size_budget_usd": 10000,
            "meta": {
                "venue": "LSE",
                "jurisdiction": "UK",  # Not in whitelist
                "adv_usd": 30000000,
                "spread_bps": 2.0
            }
        },
        # Should fail public disclosure not lagged
        {
            "instrument": Instrument.ETF.value,
            "symbol": "IWM",
            "alpha": 0.65,
            "sentiment_boost": 0.45,
            "est_cost_bps": 2.2,
            "risk_penalty": 0.25,
            "price": 210.50,
            "volume": 600000,
            "size_budget_usd": 10000,
            "meta": {
                "venue": "NYSE",
                "jurisdiction": "US",
                "adv_usd": 25000000,
                "spread_bps": 2.5,
                "uses_public_disclosure": True,
                "disclosure_lag_ms": 3600000  # Only 1 hour lag, should be 24 hours
            }
        }
    ]


def print_decision(decision):
    """Print a decision in a readable format."""
    action_color = {
        "OPEN": "\033[92m",  # Green
        "ADJUST": "\033[94m",  # Blue
        "SKIP": "\033[93m"  # Yellow
    }.get(decision["action"], "\033[0m")
    
    reset_color = "\033[0m"
    
    print(f"{action_color}{decision['action']}{reset_color} {decision['instrument']} {decision['symbol']}")
    
    if decision.get("score") is not None:
        print(f"  Score: {decision['score']:.4f}")
    
    print(f"  Reasons: {', '.join(decision['reasons'])}")
    
    if decision.get("gates_failed"):
        print(f"  Gates failed: {', '.join(decision['gates_failed'])}")
    
    print(f"  Policy: {decision['policy_version']}")
    print()


def main():
    """Run the example."""
    # Create policy service and decision engine
    policy_service = PolicyService()
    engine = DecisionEngine(policy_service)
    
    # Create sample opportunities
    opportunities = create_sample_opportunities()
    
    # Process opportunities
    logger.info("Processing %d opportunities", len(opportunities))
    routed_opps = engine.process_opportunities(
        opportunities,
        current_regime=Regime.RISK_ON
    )
    
    # Print results
    logger.info("Processed %d opportunities, %d routed", len(opportunities), len(routed_opps))
    
    print("\n=== Routed Opportunities ===\n")
    for i, opp in enumerate(routed_opps):
        print(f"{i+1}. {opp['instrument']} {opp['symbol']} (Score: {opp['score']:.4f})")
        print(f"   Orders: {len(opp['orders'])}")
        for j, order in enumerate(opp['orders']):
            print(f"     {j+1}. {order['side']} {order['qty']} @ {order.get('limit', 'MARKET')}")
            if order.get('meta'):
                print(f"        Playbook: {order['meta'].get('playbook', 'Unknown')}")
        print()
    
    print("\n=== Recent Decisions ===\n")
    decisions = recent_decisions()
    for decision in decisions:
        print_decision(decision)
    
    # Print statistics
    stats = engine.get_status()
    print("\n=== Decision Statistics ===\n")
    print(f"Total decisions: {stats['decision_stats']['total']}")
    print(f"By action: {json.dumps(stats['decision_stats']['by_action'], indent=2)}")
    print(f"By instrument: {json.dumps(stats['decision_stats']['by_instrument'], indent=2)}")
    print(f"Gate failures: {json.dumps(stats['decision_stats']['gate_failures'], indent=2)}")


if __name__ == "__main__":
    main()
