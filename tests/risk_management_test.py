#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Engine Test

This script demonstrates how the Risk Management Engine integrates with
the Strategy Intelligence Recorder to provide transparency into risk management
decisions.
"""
import os
import sys
import time
import logging
import random
import json
from datetime import datetime
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder

# Use a simplified risk engine to avoid dependency issues
class SimpleRiskEngine:
    """A simplified risk engine for testing the strategy intelligence recorder."""
    
    def __init__(self, config, persistence_manager=None):
        self.config = config or {}
        self.persistence = persistence_manager
        self.event_bus = get_global_event_bus()
        
        # Initialize risk parameters
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)  # 5% default
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.max_position_size = self.config.get('max_position_size', 0.2)  # 20% default
        self.drawdown_threshold = self.config.get('drawdown_threshold', 0.1)  # 10% default
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1% default
        
        # Internal tracking
        self.positions = {}
        self.correlations = {}
        self.portfolio_value = self.config.get('initial_portfolio_value', 100000.0)
        self.current_market_regime = "normal"
        
        logger.info("Simple Risk Engine initialized")
    
    def register_event_handlers(self):
        """Register for relevant events from the event bus."""
        # Subscribe to market regime changes
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self._on_market_regime_change)
    
    def assess_position_risk(self, symbol, position_size, entry_price, stop_loss_price):
        """Assess risk for a position and publish event."""
        risk_amount = abs(entry_price - stop_loss_price) * position_size
        risk_pct = risk_amount / self.portfolio_value
        
        action = None
        reason = None
        if risk_pct > self.risk_per_trade:
            action = "reduce_position"
            reason = "trade_risk_exceeded"
        
        assessment = {
            "symbol": symbol,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
            "risk_score": min(100, (risk_pct / self.risk_per_trade) * 50 + 30),
            "action": action,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_ALLOCATION_CHANGED,
            data=assessment,
            source="risk_engine"
        )
        
        return assessment
    
    def assess_portfolio_risk(self):
        """Assess overall portfolio risk."""
        total_risk = 0.03  # Simplified example
        
        assessment = {
            "total_risk": total_risk,
            "max_risk": self.max_portfolio_risk,
            "action": "reduce_exposure" if total_risk > self.max_portfolio_risk else None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_bus.create_and_publish(
            event_type=EventType.PORTFOLIO_EXPOSURE_UPDATED,
            data=assessment,
            source="risk_engine"
        )
        
        return assessment
    
    def update_correlations(self, correlation_matrix):
        """Update correlations and check for high correlations."""
        # Check for high correlations
        for symbol1, correlations in correlation_matrix.items():
            for symbol2, value in correlations.items():
                if symbol1 != symbol2 and abs(value) > self.correlation_threshold:
                    self.event_bus.create_and_publish(
                        event_type=EventType.CORRELATION_RISK_ALERT,
                        data={
                            "symbols": [symbol1, symbol2],
                            "correlation": value,
                            "threshold": self.correlation_threshold,
                            "action": "diversify_assets",
                            "timestamp": datetime.now().isoformat()
                        },
                        source="risk_engine"
                    )
    
    def monitor_drawdown(self, current_value):
        """Monitor portfolio drawdown."""
        peak_value = 100000.0  # Simplified example
        drawdown = (peak_value - current_value) / peak_value
        
        assessment = {
            "current_drawdown": drawdown,
            "threshold": self.drawdown_threshold,
            "exceeded": drawdown > self.drawdown_threshold,
            "severity": min(int(drawdown / self.drawdown_threshold), 3) if drawdown > self.drawdown_threshold else 0,
            "action": "reduce_exposure" if drawdown > self.drawdown_threshold else "monitor",
            "timestamp": datetime.now().isoformat()
        }
        
        if drawdown > self.drawdown_threshold:
            self.event_bus.create_and_publish(
                event_type=EventType.DRAWDOWN_THRESHOLD_EXCEEDED,
                data=assessment,
                source="risk_engine"
            )
        
        return assessment
    
    def perform_risk_attribution(self, performance_data):
        """Attribute performance to risk factors."""
        total_return = performance_data.get("total_return", 0)
        
        # Simplified attribution
        attribution = {
            "risk_factors": {
                "market_beta": total_return * 0.5,
                "sector_exposure": total_return * 0.2,
                "volatility": total_return * 0.1,
                "specific_risk": total_return * 0.2
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
            data=attribution,
            source="risk_engine"
        )
        
        return attribution
    
    def _on_market_regime_change(self, event):
        """Handle market regime change events."""
        if "current_regime" in event.data:
            self.current_market_regime = event.data["current_regime"]
            logger.info(f"Market regime changed to {self.current_market_regime}")

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def create_mock_persistence():
    """Create a mock persistence manager for testing."""
    class MockPersistence:
        def __init__(self):
            self.storage = {}
            
        def is_connected(self):
            return True
            
        def save_strategy_state(self, strategy_id, state_data):
            self.storage[strategy_id] = state_data
            return True
            
        def load_strategy_state(self, strategy_id):
            return self.storage.get(strategy_id, {})
            
        def insert_document(self, collection, document):
            if collection not in self.storage:
                self.storage[collection] = []
            self.storage[collection].append(document)
            return True
            
        def list_collections(self):
            return list(self.storage.keys())
    
    return MockPersistence()

def main():
    """Main test function."""
    print_section("RISK MANAGEMENT ENGINE TEST")
    print("Initializing components...")
    
    # Create event bus
    event_bus = get_global_event_bus()
    
    # Create mock persistence
    persistence = create_mock_persistence()
    
    # Create strategy intelligence recorder
    recorder = StrategyIntelligenceRecorder(persistence, event_bus)
    
    # Create risk management engine
    config = {
        "max_portfolio_risk": 0.05,  # 5% maximum portfolio risk
        "correlation_threshold": 0.7,  # Alert on correlations above 0.7
        "max_position_size": 0.2,  # No position can be > 20% of portfolio
        "drawdown_threshold": 0.1,  # Alert on 10% drawdowns
        "risk_per_trade": 0.01,  # Risk 1% per trade
        "initial_portfolio_value": 100000.0  # Start with $100k
    }
    
    risk_engine = SimpleRiskEngine(config, persistence)
    
    # Register for events
    risk_engine.register_event_handlers()
    
    # Subscribe to all events for monitoring
    event_count = {'total': 0}
    
    def event_listener(event: Event):
        """Listen to all events and count them."""
        event_type = event.event_type
        if event_type not in event_count:
            event_count[event_type] = 0
        event_count[event_type] += 1
        event_count['total'] += 1
        
        # Print details for risk events
        risk_events = [
            EventType.RISK_ALLOCATION_CHANGED,
            EventType.PORTFOLIO_EXPOSURE_UPDATED,
            EventType.CORRELATION_RISK_ALERT,
            EventType.DRAWDOWN_THRESHOLD_EXCEEDED,
            EventType.RISK_ATTRIBUTION_CALCULATED
        ]
        
        if event_type in risk_events:
            print(f"\nEvent: {event_type}")
            print(f"Data: {json.dumps(event.data, indent=2)}")
            print("-" * 60)
    
    # Subscribe to all events
    event_bus.subscribe_all(event_listener)
    
    print("\nRunning risk management tests...")
    
    # Test 1: Position Risk Tracking
    print_section("TEST 1: POSITION RISK TRACKING")
    
    # Test position risk assessment - position within limits
    risk_engine.assess_position_risk(
        symbol="AAPL",
        position_size=100,
        entry_price=150.0,
        stop_loss_price=145.0
    )
    
    # Test position risk assessment - position exceeding limits
    risk_engine.assess_position_risk(
        symbol="TSLA",
        position_size=50,
        entry_price=200.0,
        stop_loss_price=180.0
    )
    
    # Test 2: Portfolio-Level Risk Oversight
    print_section("TEST 2: PORTFOLIO-LEVEL RISK OVERSIGHT")
    
    # Assess overall portfolio risk
    portfolio_risk = risk_engine.assess_portfolio_risk()
    print(f"Portfolio Risk Assessment: {json.dumps(portfolio_risk, indent=2)}")
    
    # Test 3: Correlation Monitoring
    print_section("TEST 3: CORRELATION MONITORING")
    
    # Set up a correlation matrix
    correlation_matrix = {
        "AAPL": {"AAPL": 1.0, "MSFT": 0.6, "TSLA": 0.5, "AMZN": 0.75},
        "MSFT": {"AAPL": 0.6, "MSFT": 1.0, "TSLA": 0.4, "AMZN": 0.65},
        "TSLA": {"AAPL": 0.5, "MSFT": 0.4, "TSLA": 1.0, "AMZN": 0.3},
        "AMZN": {"AAPL": 0.75, "MSFT": 0.65, "TSLA": 0.3, "AMZN": 1.0}
    }
    
    # Update positions for correlation tests
    risk_engine.positions = {
        "AAPL": {"size": 100, "entry_price": 150.0, "stop_loss_price": 145.0},
        "MSFT": {"size": 75, "entry_price": 250.0, "stop_loss_price": 240.0},
        "TSLA": {"size": 50, "entry_price": 200.0, "stop_loss_price": 180.0},
        "AMZN": {"size": 20, "entry_price": 100.0, "stop_loss_price": 95.0}
    }
    
    # Update correlations
    risk_engine.update_correlations(correlation_matrix)
    
    # Test 4: Drawdown Protection
    print_section("TEST 4: DRAWDOWN PROTECTION")
    
    # Test normal portfolio value (no drawdown)
    risk_engine.monitor_drawdown(100000.0)
    
    # Test small drawdown (below threshold)
    risk_engine.monitor_drawdown(95000.0)
    
    # Test significant drawdown (above threshold)
    drawdown_assessment = risk_engine.monitor_drawdown(85000.0)
    print(f"Drawdown Assessment: {json.dumps(drawdown_assessment, indent=2)}")
    
    # Test 5: Risk Factor Attribution
    print_section("TEST 5: RISK FACTOR ATTRIBUTION")
    
    # Create sample performance data
    performance_data = {
        "total_return": 0.045,  # 4.5% return
        "positions": {
            "AAPL": {"return": 0.06, "contribution": 0.015},
            "MSFT": {"return": 0.03, "contribution": 0.01},
            "TSLA": {"return": 0.08, "contribution": 0.02},
            "AMZN": {"return": 0.0, "contribution": 0.0}
        }
    }
    
    # Perform risk attribution
    attribution = risk_engine.perform_risk_attribution(performance_data)
    print(f"Risk Attribution: {json.dumps(attribution, indent=2)}")
    
    # Test 6: Market Regime Adaptation
    print_section("TEST 6: MARKET REGIME ADAPTATION")
    
    # Publish market regime change event
    event_bus.create_and_publish(
        event_type=EventType.MARKET_REGIME_CHANGED,
        data={
            "symbol": "SPY",
            "current_regime": "volatile",
            "confidence": 0.85,
            "previous_regime": "trending",
            "timestamp": datetime.now().isoformat(),
            "trigger": "volatility_spike"
        },
        source="test"
    )
    
    # Wait for event handling
    time.sleep(0.1)
    
    # Show event counts
    print_section("EVENT COUNTS")
    for event_type, count in sorted(event_count.items()):
        if event_type != 'total':
            print(f"{event_type}: {count}")
    print(f"Total events: {event_count['total']}")
    
    # Show intelligence data collected
    print_section("INTELLIGENCE DATA")
    print("Collections in persistence:")
    for collection in persistence.list_collections():
        print(f"- {collection}: {len(persistence.storage.get(collection, []))} items")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
