#!/usr/bin/env python
"""
Test script for the Macro Guidance system.

This script tests various components of the Macro Guidance system
to ensure they're working correctly.
"""

import sys
import os
import logging
import json
import yaml
from pprint import pprint
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import macro guidance components
from trading_bot.macro_guidance import (
    MacroGuidanceEngine, 
    MacroGuidanceIntegration,
    MacroEvent,
    EventType,
    TradingBias
)

def load_config():
    """Load configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def test_macro_engine():
    """Test the macro guidance engine."""
    logger.info("Testing MacroGuidanceEngine...")
    
    # Initialize engine
    engine = MacroGuidanceEngine()
    
    # Test market regime guidance
    regime_guidance = engine.get_market_regime_guidance()
    print("\n=== MARKET REGIME GUIDANCE ===")
    print(f"Current regime: {regime_guidance['current_regime']}")
    print(f"Description: {regime_guidance['description']}")
    print(f"Regime probabilities: {regime_guidance['regime_probabilities']}")
    print("\nOptimal positioning:")
    pprint(regime_guidance['guidance'])
    
    # Test upcoming events
    events = engine.get_upcoming_events()
    print("\n=== UPCOMING EVENTS ===")
    if events:
        for event in events:
            print(f"{event['date']} - {event['description']} (Importance: {event['importance']})")
    else:
        print("No upcoming events found (using simulated data)")
    
    # Test CPI event processing
    print("\n=== CPI REPORT ANALYSIS ===")
    cpi_data = {
        "headline_cpi": 3.0,
        "core_cpi": 3.2,
        "month_over_month": 0.2,
        "forecast_headline_cpi": 3.1,
        "forecast_core_cpi": 3.3,
        "forecast_month_over_month": 0.3
    }
    
    cpi_guidance = engine.process_economic_event("cpi", cpi_data)
    print(f"Scenario: {cpi_guidance['scenario']['name']}")
    print(f"Trading bias: {cpi_guidance['trading_bias']}")
    print(f"Market response: {cpi_guidance['scenario']['market_response']['equities']}")
    print("\nStrategy recommendations:")
    pprint(cpi_guidance['strategy_recommendations'])
    
    # Test VIX spike guidance
    print("\n=== VIX SPIKE GUIDANCE ===")
    vix_guidance = engine.get_vix_spike_guidance(32, 35)
    print(f"VIX level: {vix_guidance['vix_level']}, Change: {vix_guidance['vix_change_percent']}%")
    print(f"Severity: {vix_guidance['severity']}")
    print("\nRecommendations:")
    pprint(vix_guidance['recommendations'])
    
    # Test yield curve guidance
    print("\n=== YIELD CURVE GUIDANCE ===")
    yield_curve_guidance = engine.get_yield_curve_guidance(3.8, 4.0, 45)
    print(f"10Y-2Y Spread: {yield_curve_guidance['current_spread']}")
    print(f"Curve status: {yield_curve_guidance['curve_status']}")
    print(f"Phase: {yield_curve_guidance['phase']}")
    print("\nRecommendations:")
    pprint(yield_curve_guidance['recommendations'])
    
    logger.info("MacroGuidanceEngine tests completed")

def test_macro_integration():
    """Test the macro guidance integration."""
    logger.info("Testing MacroGuidanceIntegration...")
    
    # Load config
    config = load_config()
    
    # Initialize integration
    integration = MacroGuidanceIntegration(config)
    
    # Test pre-event guidance
    pre_event = integration.get_pre_event_guidance("AAPL")
    print("\n=== PRE-EVENT GUIDANCE ===")
    if pre_event["status"] == "pre_event":
        print(f"Approaching event: {pre_event['event']['description']}")
        print(f"Date: {pre_event['event']['date']}, Time: {pre_event['event']['time']}")
        print(f"Days until: {pre_event['event']['days_until']}")
        print(f"Position sizing adjustment: {pre_event['position_sizing_adjustment']}")
        
        if "ticker_specific" in pre_event:
            print(f"\nTicker: {pre_event['ticker_specific']['ticker']}")
            print(f"Sector: {pre_event['ticker_specific']['sector']}")
            print(f"Sensitivity: {pre_event['ticker_specific']['sensitivity_to_event']}")
            print(f"Position sizing modifier: {pre_event['ticker_specific']['position_sizing_modifier']}")
    else:
        print("No major events approaching (using simulated data)")
    
    # Test enhancing a trading decision
    base_recommendation = {
        "market_condition": "bullish",
        "bias_confidence": 0.8,
        "position_sizing": {
            "equity_position_size": 1000,
            "options_position_size": 500
        },
        "integrated_strategies": [
            {
                "core_strategy": "breakout_swing",
                "options_strategy": "bull_call_spread",
                "confidence": "high"
            }
        ]
    }
    
    market_data = {
        "vix": 24,
        "vix_change_percent": 15,
        "ten_year_yield": 4.2,
        "two_year_yield": 4.5,
        "close": 175.5
    }
    
    enhanced = integration.enhance_trading_decision("AAPL", base_recommendation, market_data)
    print("\n=== ENHANCED TRADING DECISION ===")
    print(f"Original position sizes: Equity {base_recommendation['position_sizing']['equity_position_size']}, Options {base_recommendation['position_sizing']['options_position_size']}")
    print(f"Enhanced position sizes: Equity {enhanced['position_sizing']['equity_position_size']}, Options {enhanced['position_sizing']['options_position_size']}")
    print(f"Macro adjustment factor: {enhanced['macro_adjusted_position_sizing']}")
    
    if "macro_guidance" in enhanced:
        print("\nMacro guidance applied:")
        for guidance_type, guidance in enhanced["macro_guidance"].items():
            print(f"- {guidance_type}")
    
    # Test strategy adjustment
    strategy_params = {
        "strategy": "breakout_swing",
        "ticker": "AAPL",
        "action": "buy",
        "quantity": 10,
        "price": 175.5,
        "stop_loss": 170.0,
        "take_profit": 185.0
    }
    
    adjusted = integration.adjust_strategy_for_macro_conditions("AAPL", "breakout_swing", strategy_params, market_data)
    print("\n=== ADJUSTED STRATEGY ===")
    print(f"Original quantity: {strategy_params['quantity']}")
    print(f"Adjusted quantity: {adjusted['quantity']}")
    print(f"Original stop loss: {strategy_params['stop_loss']}")
    print(f"Adjusted stop loss: {adjusted['stop_loss']}")
    print(f"Original take profit: {strategy_params['take_profit']}")
    print(f"Adjusted take profit: {adjusted['take_profit']}")
    print(f"Position sizing adjustment: {adjusted['macro_guidance']['position_sizing_adjustment']}")
    
    logger.info("MacroGuidanceIntegration tests completed")

if __name__ == "__main__":
    print("\n===== MACRO GUIDANCE SYSTEM TEST =====")
    print(f"Running tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_macro_engine()
    test_macro_integration()
    
    print("\n===== ALL TESTS COMPLETED =====") 