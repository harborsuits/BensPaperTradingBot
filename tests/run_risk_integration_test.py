#!/usr/bin/env python3
"""
Risk Integration Test Runner

This script runs integration tests for the autonomous risk management system
using synthetic data and mock components where needed, avoiding external dependencies.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("risk_integration_test")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components with try-except to handle potential missing dependencies
try:
    # Event system
    from trading_bot.event_system import EventBus, Event, EventType
    
    # Mock components if needed
    class EventTracker:
        """Helper class to track events for testing."""
        
        def __init__(self, event_bus):
            self.event_bus = event_bus
            self.events = []
            self.is_tracking = False
        
        def _handle_event(self, event):
            """Event handler to record events."""
            self.events.append(event)
        
        def start_tracking(self):
            """Start tracking events."""
            if not self.is_tracking:
                self.event_bus.subscribe(self._handle_event)
                self.is_tracking = True
        
        def stop_tracking(self):
            """Stop tracking events."""
            if self.is_tracking:
                self.event_bus.unsubscribe(self._handle_event)
                self.is_tracking = False
        
        def get_events(self):
            """Get all tracked events."""
            return self.events
        
        def get_events_by_type(self, event_type):
            """Get events of a specific type."""
            return [e for e in self.events if e.event_type == event_type]
        
        def clear_events(self):
            """Clear tracked events."""
            self.events = []
    
    # Import testing components
    from trading_bot.testing.market_data_generator import MarketDataGenerator
    
    # Import autonomous components
    from trading_bot.autonomous.autonomous_engine import AutonomousEngine, StrategyCandidate
    from trading_bot.autonomous.risk_integration import AutonomousRiskManager, get_autonomous_risk_manager
    from trading_bot.risk.risk_manager import RiskLevel, StopLossType
    
    # Import pipeline
    from trading_bot.autonomous.strategy_deployment_pipeline import (
        StrategyDeploymentPipeline, 
        get_deployment_pipeline,
        DeploymentStatus
    )
    
    # Import event handlers
    from trading_bot.event_system.risk_event_handlers import RiskEventTracker, get_risk_event_tracker
    
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


def run_tests():
    """Run the integration tests."""
    
    # Check if imports were successful
    if not IMPORTS_SUCCESSFUL:
        logger.error("Required imports failed, cannot run tests")
        return False
    
    test_results = {}
    
    # Create test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create event bus and tracker
    event_bus = EventBus()
    event_tracker = EventTracker(event_bus)
    event_tracker.start_tracking()
    
    logger.info("="*80)
    logger.info("RUNNING RISK INTEGRATION TESTS")
    logger.info("="*80)
    
    try:
        # Test 1: Initialize Components
        logger.info("\nTest 1: Initialize Components")
        
        # Create market data generator
        market_gen = MarketDataGenerator(
            num_stocks=5,
            days=30,
            volatility_range=(0.15, 0.35),
            price_range=(50, 500)
        )
        
        # Generate market data
        market_data = market_gen.generate_market_data()
        options_data = market_gen.generate_options_data(market_data)
        
        logger.info("Created market data generator and synthetic data")
        
        # Create autonomous engine
        engine_config = {
            "use_real_data": False,
            "test_mode": True,
            "data_dir": test_dir
        }
        engine = AutonomousEngine(config=engine_config)
        engine.event_bus = event_bus
        
        # Initialize with test data
        engine._market_data = market_data
        engine._options_data = options_data
        
        logger.info("Created autonomous engine")
        
        # Create risk config
        risk_config = {
            "portfolio_value": 100000,
            "risk_per_trade_pct": 1.0,
            "default_stop_loss_type": "VOLATILITY",
            "default_risk_level": "MEDIUM",
            "max_allocation_pct": 5.0,
            "max_position_size_pct": 2.0
        }
        
        # Create risk manager
        risk_manager = get_autonomous_risk_manager(
            risk_config=risk_config,
            data_dir=os.path.join(test_dir, "risk")
        )
        risk_manager.event_bus = event_bus
        risk_manager.connect_engine(engine)
        
        logger.info("Created risk manager")
        
        # Create risk event tracker
        risk_tracker = get_risk_event_tracker(event_bus)
        
        logger.info("Created risk event tracker")
        
        # Create deployment pipeline
        deployment_pipeline = get_deployment_pipeline(
            event_bus=event_bus,
            data_dir=os.path.join(test_dir, "deployments")
        )
        
        logger.info("Created deployment pipeline")
        
        test_results["component_initialization"] = "PASSED"
        
        # Test 2: Create Test Strategies
        logger.info("\nTest 2: Create Test Strategies")
        
        # Create test strategies
        symbols = list(market_data.keys())[:3]
        
        # Create strategies
        strategies = []
        
        # Top candidate 1 - Iron Condor
        candidate1 = StrategyCandidate(
            strategy_id="test_iron_condor_001",
            strategy_type="iron_condor",
            symbols=[symbols[0]],
            universe="options",
            parameters={
                "delta": 0.3,
                "dte": 45,
                "take_profit": 50
            }
        )
        candidate1.returns = 15.2
        candidate1.sharpe_ratio = 1.8
        candidate1.drawdown = 12.4
        candidate1.win_rate = 68.0
        candidate1.profit_factor = 1.9
        candidate1.trades_count = 25
        candidate1.status = "backtested"
        candidate1.meets_criteria = True
        strategies.append(candidate1)
        
        # Top candidate 2 - Strangle
        candidate2 = StrategyCandidate(
            strategy_id="test_strangle_002",
            strategy_type="strangle",
            symbols=[symbols[1]],
            universe="options",
            parameters={
                "delta": 0.2,
                "dte": 30,
                "take_profit": 60
            }
        )
        candidate2.returns = 18.5
        candidate2.sharpe_ratio = 2.1
        candidate2.drawdown = 10.2
        candidate2.win_rate = 72.0
        candidate2.profit_factor = 2.2
        candidate2.trades_count = 18
        candidate2.status = "backtested"
        candidate2.meets_criteria = True
        strategies.append(candidate2)
        
        # Add candidates to engine
        engine.strategy_candidates = {
            strategy.strategy_id: strategy for strategy in strategies
        }
        
        # Set top candidates
        engine.top_candidates = strategies
        
        logger.info(f"Created {len(strategies)} test strategies")
        test_results["create_strategies"] = "PASSED"
        
        # Test 3: Deploy Strategy with Risk Manager
        logger.info("\nTest 3: Deploy Strategy with Risk Manager")
        
        strategy = strategies[0]
        
        # Deploy using risk manager
        success = risk_manager.deploy_strategy(
            strategy_id=strategy.strategy_id,
            allocation_percentage=5.0,
            risk_level=RiskLevel.MEDIUM,
            stop_loss_type=StopLossType.VOLATILITY
        )
        
        if success:
            logger.info(f"Successfully deployed {strategy.strategy_id} with risk manager")
            test_results["deploy_with_risk_manager"] = "PASSED"
            
            # Check risk manager tracking
            if strategy.strategy_id in risk_manager.deployed_strategies:
                logger.info("Risk manager is tracking the deployment correctly")
            else:
                logger.error("Risk manager is not tracking the deployment")
                test_results["deploy_with_risk_manager"] = "FAILED"
        else:
            logger.error(f"Failed to deploy {strategy.strategy_id} with risk manager")
            test_results["deploy_with_risk_manager"] = "FAILED"
        
        # Verify event emission
        deploy_events = event_tracker.get_events_by_type(EventType.STRATEGY_DEPLOYED_WITH_RISK)
        if deploy_events:
            logger.info("Deployment events were emitted correctly")
        else:
            logger.warning("No deployment events were emitted")
        
        # Test 4: Deploy Strategy with Pipeline
        logger.info("\nTest 4: Deploy Strategy with Pipeline")
        
        strategy = strategies[1]  # Use the second strategy
        
        # Deploy using pipeline
        success, result = deployment_pipeline.deploy_strategy(
            strategy_id=strategy.strategy_id,
            allocation_percentage=7.5,
            risk_level=RiskLevel.MEDIUM,
            stop_loss_type=StopLossType.VOLATILITY,
            metadata={"source": "test", "version": "1.0"}
        )
        
        if success:
            logger.info(f"Successfully deployed {strategy.strategy_id} with pipeline, deployment ID: {result}")
            test_results["deploy_with_pipeline"] = "PASSED"
            
            # Get deployment info
            deployment = deployment_pipeline.get_deployment(result)
            if deployment:
                logger.info("Pipeline is tracking the deployment correctly")
            else:
                logger.error("Pipeline is not tracking the deployment")
                test_results["deploy_with_pipeline"] = "FAILED"
        else:
            logger.error(f"Failed to deploy {strategy.strategy_id} with pipeline: {result}")
            test_results["deploy_with_pipeline"] = "FAILED"
        
        # Test 5: Calculate Position Size
        logger.info("\nTest 5: Calculate Position Size")
        
        # Calculate position size for the first strategy
        position_size = risk_manager.calculate_position_size(
            strategy_id=strategies[0].strategy_id,
            symbol=symbols[0],
            entry_price=175.0,
            stop_price=170.0,
            market_data={"price": 175.0}
        )
        
        if position_size > 0:
            logger.info(f"Calculated position size: {position_size:.2f} shares")
            test_results["calculate_position_size"] = "PASSED"
        else:
            logger.error("Position size calculation failed")
            test_results["calculate_position_size"] = "FAILED"
        
        # Test 6: Event Handling
        logger.info("\nTest 6: Test Event Handling")
        
        # Emit trade event
        event_bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            source="TestBroker",
            data={
                "strategy_id": strategies[0].strategy_id,
                "symbol": symbols[0],
                "quantity": 10,
                "price": 175.0,
                "direction": "buy"
            },
            timestamp=datetime.now()
        ))
        
        logger.info("Emitted trade execution event")
        
        # Emit position closed event
        event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            source="TestBroker",
            data={
                "strategy_id": strategies[0].strategy_id,
                "symbol": symbols[0],
                "quantity": 10,
                "price": 180.0,
                "profit_loss": 50.0
            },
            timestamp=datetime.now()
        ))
        
        logger.info("Emitted position closed event")
        
        # Check risk event tracker
        time.sleep(0.5)  # Give events time to propagate
        
        # Get risk status
        risk_status = risk_tracker.get_current_risk_status()
        
        logger.info(f"Current risk level: {risk_status.get('risk_level', 'Unknown')}")
        
        # Check strategy risk data
        strategy_risk = risk_tracker.get_strategy_risk_data(strategies[0].strategy_id)
        
        if strategy_risk:
            logger.info("Risk tracker is monitoring strategy risk")
            test_results["event_handling"] = "PASSED"
        else:
            logger.warning("Risk tracker is not capturing strategy risk data")
            test_results["event_handling"] = "WARNING"
        
        # Test 7: Circuit Breaker Functionality
        logger.info("\nTest 7: Test Circuit Breakers")
        
        # Set up conditions to trigger circuit breaker
        if hasattr(risk_manager, 'risk_metrics') and strategies[0].strategy_id in risk_manager.risk_metrics:
            risk_manager.risk_metrics[strategies[0].strategy_id]["current_drawdown"] = 30.0  # Exceeds threshold
            
            # Check circuit breakers
            should_halt, reasons = risk_manager.check_circuit_breakers({})
            
            if should_halt:
                logger.info(f"Circuit breakers correctly triggered: {reasons}")
                test_results["circuit_breakers"] = "PASSED"
                
                # Check if event was emitted
                time.sleep(0.5)  # Give events time to propagate
                circuit_events = event_tracker.get_events_by_type(EventType.CIRCUIT_BREAKER_TRIGGERED)
                
                if circuit_events:
                    logger.info("Circuit breaker events were emitted correctly")
                else:
                    logger.warning("No circuit breaker events were emitted")
            else:
                logger.error("Circuit breakers failed to trigger despite conditions")
                test_results["circuit_breakers"] = "FAILED"
        else:
            logger.warning("Could not test circuit breakers, risk metrics not available")
            test_results["circuit_breakers"] = "SKIPPED"
        
        # Test 8: Pause/Resume Functionality
        logger.info("\nTest 8: Test Pause/Resume")
        
        if success:  # If a deployment was successful
            # Get the deployment ID of the second strategy
            deployment_id = deployment_pipeline.strategy_to_deployment.get(strategies[1].strategy_id)
            
            if deployment_id:
                # Pause deployment
                pause_success = deployment_pipeline.pause_deployment(
                    deployment_id=deployment_id,
                    reason="Test pause"
                )
                
                if pause_success:
                    logger.info(f"Successfully paused deployment {deployment_id}")
                    
                    # Get deployment status
                    deployment = deployment_pipeline.get_deployment(deployment_id)
                    
                    if deployment and deployment.get("status") == DeploymentStatus.PAUSED:
                        logger.info("Deployment status was correctly updated to PAUSED")
                        
                        # Resume deployment
                        resume_success = deployment_pipeline.resume_deployment(deployment_id)
                        
                        if resume_success:
                            logger.info(f"Successfully resumed deployment {deployment_id}")
                            
                            # Get updated status
                            deployment = deployment_pipeline.get_deployment(deployment_id)
                            
                            if deployment and deployment.get("status") == DeploymentStatus.ACTIVE:
                                logger.info("Deployment status was correctly updated to ACTIVE")
                                test_results["pause_resume"] = "PASSED"
                            else:
                                logger.error("Deployment status was not updated after resume")
                                test_results["pause_resume"] = "FAILED"
                        else:
                            logger.error(f"Failed to resume deployment {deployment_id}")
                            test_results["pause_resume"] = "FAILED"
                    else:
                        logger.error("Deployment status was not updated after pause")
                        test_results["pause_resume"] = "FAILED"
                else:
                    logger.error(f"Failed to pause deployment {deployment_id}")
                    test_results["pause_resume"] = "FAILED"
            else:
                logger.warning("Could not find deployment ID for test strategy")
                test_results["pause_resume"] = "SKIPPED"
        else:
            logger.warning("Skipping pause/resume test as deployment failed")
            test_results["pause_resume"] = "SKIPPED"
        
        # Test 9: Get Risk Reports
        logger.info("\nTest 9: Test Risk Reports")
        
        # Get risk report from risk manager
        risk_report = risk_manager.get_risk_report()
        
        if risk_report:
            logger.info("Successfully generated risk manager report")
            logger.info(f"Report contains {risk_report.get('total_strategies', 0)} strategies")
            
            # Get deployment summary from pipeline
            deployment_summary = deployment_pipeline.get_deployment_summary()
            
            if deployment_summary:
                logger.info("Successfully generated deployment pipeline summary")
                logger.info(f"Summary contains {deployment_summary.get('total_deployments', 0)} deployments")
                
                test_results["risk_reports"] = "PASSED"
            else:
                logger.error("Failed to generate deployment pipeline summary")
                test_results["risk_reports"] = "PARTIAL - Risk manager report only"
        else:
            logger.error("Failed to generate risk manager report")
            test_results["risk_reports"] = "FAILED"
        
    except Exception as e:
        logger.error(f"Error during tests: {e}")
        test_results["overall"] = "ERROR"
    finally:
        # Clean up
        event_tracker.stop_tracking()
    
    # Calculate overall result
    passed = sum(1 for result in test_results.values() if result == "PASSED")
    failed = sum(1 for result in test_results.values() if result == "FAILED")
    skipped = sum(1 for result in test_results.values() if result == "SKIPPED")
    warnings = sum(1 for result in test_results.values() if result == "WARNING")
    
    if "overall" not in test_results:
        if failed == 0:
            if warnings == 0 and skipped == 0:
                test_results["overall"] = "PASSED"
            else:
                test_results["overall"] = "PASSED WITH WARNINGS"
        else:
            test_results["overall"] = "FAILED"
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    for test, result in test_results.items():
        if test != "overall":
            logger.info(f"{test:30}: {result}")
    
    logger.info("-"*80)
    logger.info(f"Passed: {passed}, Failed: {failed}, Warnings: {warnings}, Skipped: {skipped}")
    logger.info(f"Overall Result: {test_results['overall']}")
    logger.info("="*80)
    
    # Save results to file
    results_file = os.path.join(test_dir, "risk_integration_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": test_results,
            "details": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return test_results["overall"] == "PASSED" or test_results["overall"] == "PASSED WITH WARNINGS"

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
