#!/usr/bin/env python3
"""
Simplified Risk Integration Verification

This script verifies the integration between autonomous trading and risk management
using mock components and synthetic data, avoiding external dependencies.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("risk_verify")

# Event system
class MockEventType:
    """Mock event types for testing"""
    STRATEGY_GENERATED = "STRATEGY_GENERATED"
    STRATEGY_OPTIMISED = "STRATEGY_OPTIMISED"
    STRATEGY_DEPLOYED = "STRATEGY_DEPLOYED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    RISK_ALERT = "RISK_ALERT"

class MockEvent:
    """Mock event for testing"""
    def __init__(self, event_type, source, data):
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp = datetime.now()

class MockEventBus:
    """Mock event bus for testing"""
    def __init__(self):
        self.handlers = {}
        self.events = []
    
    def register(self, event_type, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def publish(self, event):
        self.events.append(event)
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                handler(event.event_type, event.data)

# Strategy components
class MockStrategy:
    """Mock strategy for testing"""
    def __init__(self, strategy_id, strategy_type, symbols, parameters):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.symbols = symbols
        self.parameters = parameters
        self.returns = 0.0
        self.sharpe_ratio = 0.0
        self.drawdown = 0.0
        self.win_rate = 0.0
        self.trades_count = 0
        self.status = "pending"
    
    def to_dict(self):
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type,
            "symbols": self.symbols,
            "parameters": self.parameters,
            "performance": {
                "returns": self.returns,
                "sharpe_ratio": self.sharpe_ratio,
                "drawdown": self.drawdown,
                "win_rate": self.win_rate,
                "trades_count": self.trades_count
            },
            "status": self.status
        }

class MockAutonomousEngine:
    """Mock autonomous engine for testing"""
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.strategies = {}
        self.top_candidates = []
        self.near_miss_candidates = []
    
    def generate_mock_strategies(self, count=5):
        """Generate some mock strategies"""
        strategy_types = ["iron_condor", "strangle", "butterfly", "vertical_spread"]
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"]
        
        for i in range(count):
            strategy_id = f"test_strategy_{i+1:03d}"
            strategy_type = strategy_types[i % len(strategy_types)]
            strategy_symbols = [symbols[i % len(symbols)]]
            
            strategy = MockStrategy(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                symbols=strategy_symbols,
                parameters={"param1": i, "param2": i*2}
            )
            
            # Set some performance metrics
            strategy.returns = 10.0 + (i * 2)
            strategy.sharpe_ratio = 1.5 + (i * 0.2)
            strategy.drawdown = 10.0 - (i * 0.5)
            strategy.win_rate = 60.0 + (i * 2)
            strategy.trades_count = 20 + i
            
            # Set status based on index
            if i < 2:
                strategy.status = "optimized"
                self.top_candidates.append(strategy)
            else:
                strategy.status = "near_miss"
                self.near_miss_candidates.append(strategy)
            
            self.strategies[strategy_id] = strategy
            
            # Emit event
            self.event_bus.publish(
                MockEvent(
                    event_type=MockEventType.STRATEGY_GENERATED,
                    source="MockEngine",
                    data={"strategy": strategy.to_dict()}
                )
            )
        
        logger.info(f"Generated {count} mock strategies")
    
    def get_top_candidates(self):
        return self.top_candidates
    
    def get_near_miss_candidates(self):
        return self.near_miss_candidates
    
    def deploy_strategy(self, strategy_id):
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            strategy.status = "deployed"
            
            # Emit event
            self.event_bus.publish(
                MockEvent(
                    event_type=MockEventType.STRATEGY_DEPLOYED,
                    source="MockEngine",
                    data={"strategy_id": strategy_id}
                )
            )
            
            return True
        return False

# Risk management
class MockRiskLevel:
    """Mock risk levels for testing"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class MockStopLossType:
    """Mock stop loss types for testing"""
    FIXED = "FIXED"
    TRAILING = "TRAILING"
    VOLATILITY = "VOLATILITY"

class MockRiskManager:
    """Mock risk manager for testing"""
    def __init__(self):
        self.portfolio_value = 100000.0
    
    def get_risk_metrics(self):
        return {
            "current_drawdown_pct": 5.0,
            "daily_profit_loss_pct": -1.2,
            "total_portfolio_risk": 45.0,
            "open_positions": 3
        }

class AutonomousRiskIntegration:
    """Integration between autonomous engine and risk management"""
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.engine = None
        self.risk_manager = MockRiskManager()
        
        # State tracking
        self.deployed_strategies = {}
        self.strategy_allocations = {}
        self.risk_metrics = {}
        self.circuit_breakers = {
            "portfolio_drawdown": 15.0,
            "strategy_drawdown": 25.0,
            "daily_loss": 5.0
        }
        
        # Track events
        self.events_received = []
        
        # Register for events
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for events"""
        self.event_bus.register(MockEventType.STRATEGY_OPTIMISED, self._handle_event)
        self.event_bus.register(MockEventType.STRATEGY_DEPLOYED, self._handle_event)
        self.event_bus.register(MockEventType.POSITION_OPENED, self._handle_event)
        self.event_bus.register(MockEventType.POSITION_CLOSED, self._handle_event)
    
    def _handle_event(self, event_type, event_data):
        """Handle incoming events"""
        self.events_received.append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Received event: {event_type}")
        
        # Process specific events
        if event_type == MockEventType.STRATEGY_OPTIMISED:
            strategy_id = event_data.get("strategy_id")
            logger.info(f"Strategy optimized: {strategy_id}")
            
        elif event_type == MockEventType.STRATEGY_DEPLOYED:
            strategy_id = event_data.get("strategy_id")
            logger.info(f"Strategy deployed: {strategy_id}")
            
        elif event_type == MockEventType.POSITION_OPENED:
            strategy_id = event_data.get("strategy_id")
            symbol = event_data.get("symbol")
            quantity = event_data.get("quantity", 0)
            
            # Update risk metrics
            if strategy_id in self.risk_metrics:
                self.risk_metrics[strategy_id]["position_count"] += 1
                logger.info(f"Position opened: {strategy_id} - {symbol} - {quantity}")
            
        elif event_type == MockEventType.POSITION_CLOSED:
            strategy_id = event_data.get("strategy_id")
            profit_loss = event_data.get("profit_loss", 0)
            
            # Update risk metrics
            if strategy_id in self.risk_metrics:
                self.risk_metrics[strategy_id]["position_count"] -= 1
                self.risk_metrics[strategy_id]["daily_profit_loss"] += profit_loss
                logger.info(f"Position closed: {strategy_id} - P&L: {profit_loss}")
    
    def connect_engine(self, engine):
        """Connect to the engine"""
        self.engine = engine
        logger.info("Connected to mock engine")
    
    def deploy_strategy(self, strategy_id, allocation_percentage=5.0, 
                        risk_level=MockRiskLevel.MEDIUM,
                        stop_loss_type=MockStopLossType.VOLATILITY):
        """Deploy a strategy with risk controls"""
        if not self.engine:
            logger.error("No engine connected")
            return False
        
        # Find the strategy
        strategy = None
        for candidate in self.engine.get_top_candidates():
            if candidate.strategy_id == strategy_id:
                strategy = candidate
                break
        
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found or not optimized")
            return False
        
        # Cap allocation
        allocation_percentage = min(allocation_percentage, 20.0)
        
        # Store deployment info
        self.deployed_strategies[strategy_id] = {
            "strategy": strategy.to_dict(),
            "risk_params": {
                "allocation_percentage": allocation_percentage,
                "risk_level": risk_level,
                "stop_loss_type": stop_loss_type
            },
            "deploy_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Set allocation
        self.strategy_allocations[strategy_id] = allocation_percentage
        
        # Initialize risk metrics
        self.risk_metrics[strategy_id] = {
            "current_drawdown": 0.0,
            "daily_profit_loss": 0.0,
            "position_count": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Deploy through the engine
        success = self.engine.deploy_strategy(strategy_id)
        
        if success:
            logger.info(f"Deployed strategy {strategy_id} with {allocation_percentage}% allocation")
            
            # Emit event
            self.event_bus.publish(
                MockEvent(
                    event_type="STRATEGY_DEPLOYED_WITH_RISK",
                    source="RiskIntegration",
                    data={
                        "strategy_id": strategy_id,
                        "allocation_percentage": allocation_percentage,
                        "risk_level": risk_level
                    }
                )
            )
        
        return success
    
    def calculate_position_size(self, strategy_id, symbol, entry_price, stop_price):
        """Calculate position size based on risk parameters"""
        if strategy_id not in self.deployed_strategies:
            logger.error(f"Strategy {strategy_id} not deployed")
            return 0
        
        # Get allocation
        allocation = self.strategy_allocations.get(strategy_id, 5.0) / 100.0
        
        # Calculate allocation amount
        allocation_amount = self.risk_manager.portfolio_value * allocation
        
        # Risk per trade
        risk_per_trade = 0.02  # 2% risk
        
        # Calculate risk amount
        risk_amount = allocation_amount * risk_per_trade
        
        # Calculate trade risk (distance to stop)
        trade_risk_pct = abs(entry_price - stop_price) / entry_price
        
        # Avoid division by zero
        if trade_risk_pct == 0:
            trade_risk_pct = 0.01
        
        # Calculate position size in dollars
        position_size_dollars = risk_amount / trade_risk_pct
        
        # Calculate position size in shares
        position_size = position_size_dollars / entry_price
        
        logger.info(f"Calculated position size for {strategy_id}: {position_size:.2f} shares")
        return position_size
    
    def check_circuit_breakers(self):
        """Check if any circuit breakers should be triggered"""
        should_halt = False
        reasons = []
        
        # Check risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Portfolio drawdown check
        current_drawdown = risk_metrics.get("current_drawdown_pct", 0)
        if current_drawdown > self.circuit_breakers["portfolio_drawdown"]:
            should_halt = True
            reasons.append(f"Portfolio drawdown ({current_drawdown:.2f}%) exceeds threshold")
        
        # Daily loss check
        daily_profit_loss = risk_metrics.get("daily_profit_loss_pct", 0)
        if daily_profit_loss < -self.circuit_breakers["daily_loss"]:
            should_halt = True
            reasons.append(f"Daily loss ({-daily_profit_loss:.2f}%) exceeds threshold")
        
        # Strategy-specific checks
        for strategy_id, metrics in self.risk_metrics.items():
            # Strategy drawdown check
            if metrics.get("current_drawdown", 0) > self.circuit_breakers["strategy_drawdown"]:
                should_halt = True
                reasons.append(f"Strategy {strategy_id} drawdown exceeds threshold")
        
        if should_halt:
            logger.warning(f"Circuit breakers triggered: {reasons}")
            
            # Emit event
            self.event_bus.publish(
                MockEvent(
                    event_type=MockEventType.RISK_ALERT,
                    source="RiskIntegration",
                    data={
                        "alert_type": "CIRCUIT_BREAKER",
                        "reasons": reasons,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )
        
        return should_halt, reasons
    
    def get_risk_report(self):
        """Get a risk report for all strategies"""
        return {
            "timestamp": datetime.now().isoformat(),
            "deployed_strategies": len(self.deployed_strategies),
            "total_allocation": sum(self.strategy_allocations.values()),
            "strategy_metrics": {
                strategy_id: {
                    "allocation": self.strategy_allocations.get(strategy_id, 0),
                    "risk_metrics": self.risk_metrics.get(strategy_id, {})
                }
                for strategy_id in self.deployed_strategies
            },
            "circuit_breakers": self.circuit_breakers,
            "portfolio_metrics": self.risk_manager.get_risk_metrics()
        }


def run_verification():
    """Run the verification process"""
    logger.info("Starting autonomous risk integration verification")
    
    # Set up event bus
    event_bus = MockEventBus()
    
    # Create mock engine
    engine = MockAutonomousEngine(event_bus)
    engine.generate_mock_strategies(5)
    
    # Create risk integration
    risk_integration = AutonomousRiskIntegration(event_bus)
    risk_integration.connect_engine(engine)
    
    # 1. Deploy strategy with risk controls
    logger.info("\n1. Deploying strategy with risk controls")
    top_candidates = engine.get_top_candidates()
    
    if top_candidates:
        strategy_id = top_candidates[0].strategy_id
        success = risk_integration.deploy_strategy(strategy_id, allocation_percentage=10.0)
        
        if success:
            logger.info(f"Successfully deployed {strategy_id} with risk controls")
        else:
            logger.error(f"Failed to deploy {strategy_id}")
    else:
        logger.error("No top candidates available for deployment")
    
    # 2. Calculate position size
    logger.info("\n2. Calculating risk-adjusted position size")
    if top_candidates:
        position_size = risk_integration.calculate_position_size(
            strategy_id=top_candidates[0].strategy_id,
            symbol="SPY",
            entry_price=400.0, 
            stop_price=380.0
        )
        logger.info(f"Calculated position size: {position_size:.2f} shares")
    
    # 3. Simulate trading activity
    logger.info("\n3. Simulating trading activity")
    if top_candidates:
        strategy_id = top_candidates[0].strategy_id
        
        # Simulate opening positions
        event_bus.publish(
            MockEvent(
                event_type=MockEventType.POSITION_OPENED,
                source="MockBroker",
                data={
                    "strategy_id": strategy_id,
                    "symbol": "SPY",
                    "quantity": 10,
                    "price": 400.0
                }
            )
        )
        
        # Simulate closing positions with profit
        event_bus.publish(
            MockEvent(
                event_type=MockEventType.POSITION_CLOSED,
                source="MockBroker",
                data={
                    "strategy_id": strategy_id,
                    "symbol": "SPY",
                    "quantity": 10,
                    "price": 410.0,
                    "profit_loss": 100.0
                }
            )
        )
    
    # 4. Test circuit breakers
    logger.info("\n4. Testing circuit breakers")
    
    # Adjust risk metrics to trigger breakers
    if top_candidates:
        strategy_id = top_candidates[0].strategy_id
        risk_integration.risk_metrics[strategy_id]["current_drawdown"] = 30.0
    
    # Check circuit breakers
    should_halt, reasons = risk_integration.check_circuit_breakers()
    if should_halt:
        logger.info(f"Circuit breakers correctly triggered: {reasons}")
    else:
        logger.warning("Circuit breakers not triggered")
    
    # 5. Generate risk report
    logger.info("\n5. Generating risk report")
    risk_report = risk_integration.get_risk_report()
    logger.info(f"Risk report contains {risk_report['deployed_strategies']} strategies")
    logger.info(f"Total allocation: {risk_report['total_allocation']}%")
    
    # 6. Check event handling
    logger.info("\n6. Checking event handling")
    events_received = risk_integration.events_received
    logger.info(f"Risk integration received {len(events_received)} events")
    
    event_types = {}
    for event in events_received:
        event_type = event["type"]
        if event_type not in event_types:
            event_types[event_type] = 0
        event_types[event_type] += 1
    
    for event_type, count in event_types.items():
        logger.info(f"  - {event_type}: {count}")
    
    # 7. Check event bus contents
    logger.info("\n7. Checking event bus")
    logger.info(f"Event bus contains {len(event_bus.events)} events")
    
    # 8. Generate verification report
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "components_initialized": {
            "event_bus": True,
            "engine": True,
            "risk_integration": True
        },
        "functionality_verified": {
            "strategy_deployment": len(risk_integration.deployed_strategies) > 0,
            "position_sizing": position_size > 0 if 'position_size' in locals() else False,
            "event_handling": len(events_received) > 0,
            "circuit_breakers": should_halt
        },
        "event_counts": event_types,
        "overall_status": "PASSED"
    }
    
    # Check for failures
    for category, checks in verification_results["functionality_verified"].items():
        if not checks:
            verification_results["overall_status"] = "FAILED"
            break
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info(f"VERIFICATION COMPLETE: {verification_results['overall_status']}")
    logger.info("="*70)
    
    for category, checks in verification_results["functionality_verified"].items():
        status = "PASS" if checks else "FAIL"
        logger.info(f"  - {category}: {status}")
    
    logger.info("="*70)
    
    # Save report
    with open("autonomous_risk_verification.json", "w") as f:
        json.dump(verification_results, f, indent=2)
    
    logger.info("Verification report saved to autonomous_risk_verification.json")
    
    return verification_results["overall_status"] == "PASSED"


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
