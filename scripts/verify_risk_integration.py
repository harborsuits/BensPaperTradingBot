#!/usr/bin/env python3
"""
Verification Script for Autonomous Risk Integration

This script demonstrates and verifies the integration between the autonomous 
trading engine and risk management system, showcasing the complete workflow
from strategy generation to risk-aware deployment.
"""

import os
import sys
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_integration_verification.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("risk_verification")

# Import components
from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.autonomous.risk_integration import AutonomousRiskManager, get_autonomous_risk_manager
from trading_bot.risk.risk_manager import RiskLevel, StopLossType
from trading_bot.event_system import EventBus, Event, EventType
from trading_bot.testing.market_data_generator import MarketDataGenerator

# Event tracking
tracked_events = []

def event_handler(event_type, event_data):
    """Record events for analysis"""
    tracked_events.append({
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": event_data
    })
    logger.info(f"Event: {event_type} - {event_data.get('source', 'Unknown')}")

def verify_risk_integration():
    """Run the full verification process"""
    logger.info("=" * 80)
    logger.info("STARTING RISK INTEGRATION VERIFICATION")
    logger.info("=" * 80)
    
    verification_results = {
        "components": {},
        "events": {},
        "workflow": {}
    }
    
    # 1. Set up components
    logger.info("1. Setting up components...")
    
    # Create event bus
    event_bus = EventBus()
    
    # Register for all events
    for event_type in dir(EventType):
        if event_type.isupper():
            event_bus.register(getattr(EventType, event_type), event_handler)
    
    verification_results["components"]["event_bus"] = "OK"
    
    # Create market data generator
    market_gen = MarketDataGenerator(
        num_stocks=10,
        days=60,
        volatility_range=(0.15, 0.35),
        price_range=(50, 500)
    )
    market_data = market_gen.generate_market_data()
    options_data = market_gen.generate_options_data(market_data)
    
    verification_results["components"]["market_data"] = "OK"
    
    # Create autonomous engine with test config
    engine_config = {
        "use_real_data": False,
        "test_mode": True,
        "data_dir": "./test_data"
    }
    engine = AutonomousEngine(config=engine_config)
    engine.event_bus = event_bus
    
    # Initialize with test data
    engine._market_data = market_data
    engine._options_data = options_data
    
    verification_results["components"]["autonomous_engine"] = "OK"
    
    # Create risk manager
    risk_config = {
        "portfolio_value": 100000,
        "max_allocation_pct": 5.0,
        "default_risk_per_trade_pct": 1.0,
        "circuit_breakers": {
            "portfolio_drawdown_pct": 10.0,
            "daily_loss_pct": 5.0
        }
    }
    risk_manager = get_autonomous_risk_manager(
        risk_config=risk_config,
        data_dir="./test_data/risk"
    )
    risk_manager.event_bus = event_bus
    risk_manager.connect_engine(engine)
    
    verification_results["components"]["risk_manager"] = "OK"
    
    # 2. Verify event system integration
    logger.info("2. Verifying event system integration...")
    
    test_event = Event(
        event_type=EventType.SYSTEM_STATUS,
        source="VerificationScript",
        data={"status": "testing", "timestamp": datetime.now().isoformat()},
        timestamp=datetime.now()
    )
    event_bus.publish(test_event)
    
    # Check if event was recorded
    if any(e["type"] == EventType.SYSTEM_STATUS for e in tracked_events):
        verification_results["events"]["event_publishing"] = "OK"
    else:
        verification_results["events"]["event_publishing"] = "FAILED"
        logger.error("Event system verification failed")
    
    # 3. Run autonomous workflow with risk integration
    logger.info("3. Running autonomous workflow...")
    
    # Start the engine
    engine.start_process(
        universe="options",
        strategy_types=["iron_condor", "strangle"],
        thresholds={
            "min_sharpe": 1.0,
            "min_win_rate": 60.0,
            "max_drawdown": 15.0
        }
    )
    
    # Wait for strategies to be generated
    time.sleep(2)
    logger.info("Waiting for strategy generation...")
    
    # Stop the engine
    engine.stop_process()
    
    # Check for generated strategies
    top_candidates = engine.get_top_candidates()
    near_miss = engine.get_near_miss_candidates()
    
    if len(top_candidates) > 0 or len(near_miss) > 0:
        verification_results["workflow"]["strategy_generation"] = "OK"
        logger.info(f"Generated {len(top_candidates)} top strategies and {len(near_miss)} near-miss strategies")
    else:
        verification_results["workflow"]["strategy_generation"] = "FAILED"
        logger.error("Strategy generation failed or produced no results")
    
    # 4. Deploy strategies with risk management
    if len(top_candidates) > 0:
        logger.info("4. Deploying strategies with risk management...")
        
        # Deploy first candidate with risk controls
        candidate = top_candidates[0]
        logger.info(f"Deploying strategy {candidate.strategy_id} with risk controls")
        
        deployed = risk_manager.deploy_strategy(
            strategy_id=candidate.strategy_id,
            allocation_percentage=5.0,
            risk_level=RiskLevel.MEDIUM,
            stop_loss_type=StopLossType.VOLATILITY
        )
        
        if deployed:
            verification_results["workflow"]["strategy_deployment"] = "OK"
            logger.info(f"Strategy {candidate.strategy_id} deployed successfully with risk controls")
            
            # Verify risk metrics are being tracked
            risk_report = risk_manager.get_risk_report()
            if candidate.strategy_id in risk_report["strategy_metrics"]:
                verification_results["workflow"]["risk_tracking"] = "OK"
                logger.info("Risk metrics are being tracked correctly")
            else:
                verification_results["workflow"]["risk_tracking"] = "FAILED"
                logger.error("Risk metrics are not being tracked correctly")
                
            # Test circuit breaker functionality
            logger.info("Testing circuit breaker functionality...")
            
            # Simulate extreme risk metrics to trigger circuit breaker
            engine.strategy_candidates[candidate.strategy_id].drawdown = 30.0
            risk_manager.risk_metrics[candidate.strategy_id]["current_drawdown"] = 30.0
            
            # Check circuit breakers
            should_halt, reasons = risk_manager.check_circuit_breakers({})
            
            if should_halt:
                verification_results["workflow"]["circuit_breakers"] = "OK"
                logger.info(f"Circuit breakers triggered correctly: {reasons}")
            else:
                verification_results["workflow"]["circuit_breakers"] = "FAILED"
                logger.error("Circuit breakers failed to trigger")
                
        else:
            verification_results["workflow"]["strategy_deployment"] = "FAILED"
            logger.error(f"Failed to deploy strategy {candidate.strategy_id}")
    else:
        verification_results["workflow"]["strategy_deployment"] = "SKIPPED"
        logger.warning("No strategies to deploy, skipping deployment verification")
    
    # 5. Analyze events
    logger.info("5. Analyzing events...")
    
    event_counts = {}
    for event in tracked_events:
        event_type = event["type"]
        if event_type not in event_counts:
            event_counts[event_type] = 0
        event_counts[event_type] += 1
    
    logger.info(f"Recorded {len(tracked_events)} events:")
    for event_type, count in event_counts.items():
        logger.info(f"  - {event_type}: {count}")
    
    verification_results["events"]["event_counts"] = event_counts
    
    # 6. Generate verification report
    logger.info("6. Generating verification report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": verification_results,
        "overall_status": "PASSED"
    }
    
    # Check for any failures
    for category, items in verification_results.items():
        for item, status in items.items():
            if isinstance(status, str) and status == "FAILED":
                report["overall_status"] = "FAILED"
    
    # Save report
    with open("risk_integration_verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("=" * 80)
    logger.info(f"VERIFICATION COMPLETE: {report['overall_status']}")
    logger.info("=" * 80)
    
    for category, items in verification_results.items():
        logger.info(f"\n{category.upper()}:")
        for item, status in items.items():
            if isinstance(status, str):
                logger.info(f"  - {item}: {status}")
    
    logger.info("\nSee risk_integration_verification_report.json for full details")
    
    return report["overall_status"] == "PASSED"

if __name__ == "__main__":
    success = verify_risk_integration()
    sys.exit(0 if success else 1)
