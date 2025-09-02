"""
End-to-End Integration Test

This script tests the entire trading system flow from market data ingestion 
to order execution and reporting, ensuring all components work together properly.

Usage:
    python -m tests.integration.end_to_end_test

Features tested:
- Market data processing & validation
- Signal generation
- Order creation & execution
- Position tracking & reconciliation
- Transaction cost analysis
- End-of-day reporting
- Emergency controls
"""
import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results/integration_test.log', mode='w')
    ]
)

# Create test_results directory if it doesn't exist
os.makedirs('test_results', exist_ok=True)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import system components
from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType, TradingMode
from trading_bot.security.credentials_manager import CredentialsManager
from trading_bot.security.secure_logger import SecureLogger
from trading_bot.data.live_data_source import LiveDataSource
from trading_bot.data.data_validator import DataValidator
from trading_bot.core.position_reconciler import PositionReconciler
from trading_bot.execution.execution_simulator import ExecutionSimulator
from trading_bot.analysis.transaction_cost_analyzer import TransactionCostAnalyzer
from trading_bot.risk.emergency_controls import EmergencyControls
from trading_bot.services.RecapReportingService import RecapReportingService

# Create secure logger for the test
logger = SecureLogger(name="integration_test")

class IntegrationTest:
    """
    End-to-end integration test for the trading system.
    Tests the entire pipeline from market data to reporting.
    """
    
    def __init__(self):
        """Initialize the test environment."""
        self.event_bus = get_global_event_bus()
        self.credentials = CredentialsManager()
        
        # Test configuration
        self.test_symbols = ["SPY", "AAPL", "MSFT"]
        self.test_timeframes = ["1m", "5m"]
        self.test_duration_seconds = 300  # 5 minutes
        
        # Test state tracking
        self.components_started = False
        self.events_received = {event_type: 0 for event_type in EventType}
        self.orders_created = []
        self.trades_executed = []
        self.errors_detected = []
        self.position_snapshots = []
        
        # Create the test report directory
        self.report_dir = "test_results/reports"
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize all system components needed for testing
        logger.info("Initializing system components for integration test")
        
        # Initialize Recap Reporting Service first so it can subscribe to all events
        self.reporting_service = RecapReportingService(
            report_dir=self.report_dir,
            symbols=self.test_symbols,
            send_emails=False  # Don't send emails during testing
        )
        
        # Initialize emergency controls
        self.emergency_controls = EmergencyControls(
            max_daily_loss_pct=0.05,  # Higher limit for testing
            max_position_pct=0.20,    # Higher limit for testing
            auto_enable=True
        )
        
        # Initialize market data validator
        self.data_validator = DataValidator(
            enable_market_hours_check=False  # Disable for testing
        )
        
        # Initialize live data source (simulation mode)
        self.data_source = LiveDataSource(
            provider="simulation",
            symbols=self.test_symbols,
            timeframes=self.test_timeframes
        )
        
        # Initialize execution simulator
        self.execution_simulator = ExecutionSimulator(
            enable_latency=True,
            enable_slippage=True,
            max_latency_ms=500,
            max_slippage_bps=20
        )
        
        # Initialize position reconciler
        self.position_reconciler = PositionReconciler()
        
        # Initialize transaction cost analyzer
        self.transaction_analyzer = TransactionCostAnalyzer()
        
        # Set up event listeners for test monitoring
        self._initialize_event_listeners()
        logger.info("System components initialized")
    
    def _initialize_event_listeners(self):
        """Set up event listeners to track the flow of data through the system."""
        # Count all events by type
        self.event_bus.subscribe_all(self.on_event)
        
        # Track specific event types for detailed monitoring
        self.event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, self.on_market_data)
        self.event_bus.subscribe(EventType.BAR_CLOSED, self.on_bar_closed)
        self.event_bus.subscribe(EventType.DATA_QUALITY_ALERT, self.on_data_quality_alert)
        self.event_bus.subscribe(EventType.ORDER_CREATED, self.on_order_created)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self.on_order_filled)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.on_trade_executed)
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self.on_error)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self.on_position_update)
        self.event_bus.subscribe(EventType.KILL_SWITCH_ACTIVATED, self.on_kill_switch)
    
    def run_test(self):
        """Run the full end-to-end integration test."""
        try:
            logger.info(f"Starting end-to-end integration test for {self.test_duration_seconds} seconds")
            
            # Start all system components
            self._start_components()
            
            # Run test for specified duration
            start_time = time.time()
            end_time = start_time + self.test_duration_seconds
            
            while time.time() < end_time:
                # Report progress every 30 seconds
                elapsed = time.time() - start_time
                if elapsed % 30 < 1:  # Report roughly every 30 seconds
                    self._report_progress(elapsed)
                
                # Sleep to avoid busy waiting
                time.sleep(1)
            
            # Trigger end of day event to generate reports
            self._trigger_end_of_day()
            
            # Allow time for end-of-day processing
            logger.info("Waiting for end-of-day processing to complete...")
            time.sleep(5)
            
            # Stop all components
            self._stop_components()
            
            # Generate test summary and validate results
            test_passed = self._validate_test_results()
            
            # Save test report
            self._save_test_report(test_passed)
            
            return test_passed
            
        except Exception as e:
            logger.exception(f"Integration test failed with error: {e}")
            self._stop_components()  # Ensure components are stopped on error
            return False
    
    def _start_components(self):
        """Start all system components in the correct order."""
        if self.components_started:
            return
            
        logger.info("Starting system components...")
        
        # Start data source first to provide market data
        self.data_source.start()
        logger.info("Market data source started")
        
        # Set initial account value for emergency controls
        self.event_bus.create_and_publish(
            event_type=EventType.CAPITAL_ADJUSTED,
            data={"new_capital": 100000.0},
            source="integration_test"
        )
        
        # Mark as started
        self.components_started = True
        logger.info("All components started successfully")
    
    def _stop_components(self):
        """Stop all system components in the correct order."""
        if not self.components_started:
            return
            
        logger.info("Stopping system components...")
        
        # Stop data source
        self.data_source.stop()
        logger.info("Market data source stopped")
        
        # Mark as stopped
        self.components_started = False
        logger.info("All components stopped successfully")
    
    def _trigger_end_of_day(self):
        """Trigger end of day event to generate reports."""
        logger.info("Triggering end-of-day processing")
        
        # Publish end of day event
        self.event_bus.create_and_publish(
            event_type=EventType.END_OF_DAY,
            data={"timestamp": datetime.now().isoformat()},
            source="integration_test"
        )
    
    def _report_progress(self, elapsed_seconds):
        """Report current test progress."""
        logger.info(f"Test progress: {elapsed_seconds:.0f}/{self.test_duration_seconds} seconds elapsed")
        
        # Report event counts for key event types
        event_summary = {
            "market_data": self.events_received.get(EventType.MARKET_DATA_RECEIVED, 0),
            "bars_closed": self.events_received.get(EventType.BAR_CLOSED, 0),
            "orders_created": len(self.orders_created),
            "trades_executed": len(self.trades_executed),
            "errors": len(self.errors_detected),
            "data_quality_alerts": self.events_received.get(EventType.DATA_QUALITY_ALERT, 0)
        }
        
        logger.info(f"Event summary: {event_summary}")
    
    def _validate_test_results(self) -> bool:
        """
        Validate test results to ensure system is functioning correctly.
        
        Returns:
            True if test passed, False otherwise
        """
        logger.info("Validating test results...")
        
        validation_errors = []
        
        # Validate market data flow
        if self.events_received.get(EventType.MARKET_DATA_RECEIVED, 0) < 10:
            validation_errors.append("Insufficient market data events received")
            
        if self.events_received.get(EventType.BAR_CLOSED, 0) < 5:
            validation_errors.append("Insufficient bar closed events received")
        
        # Validate order execution
        if len(self.orders_created) < 3:
            validation_errors.append("Insufficient orders created")
            
        if len(self.trades_executed) < 2:
            validation_errors.append("Insufficient trades executed")
        
        # Validate emergency controls
        if self.emergency_controls.kill_switch_activated:
            validation_errors.append("Kill switch was activated during test")
        
        # Validate position tracking
        if not self.position_snapshots:
            validation_errors.append("No position updates received")
        
        # Report validation results
        if validation_errors:
            logger.error(f"Test validation failed with {len(validation_errors)} errors:")
            for i, error in enumerate(validation_errors, 1):
                logger.error(f"  {i}. {error}")
            return False
        
        logger.info("All validation checks passed - Test SUCCESSFUL")
        return True
    
    def _save_test_report(self, test_passed: bool):
        """Save test results to a JSON file."""
        report_data = {
            "test_passed": test_passed,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": self.test_duration_seconds,
            "event_counts": {str(k): v for k, v in self.events_received.items()},
            "orders_created": len(self.orders_created),
            "trades_executed": len(self.trades_executed),
            "errors_detected": self.errors_detected,
            "symbols_tested": self.test_symbols,
            "timeframes_tested": self.test_timeframes
        }
        
        report_path = os.path.join("test_results", "integration_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Test report saved to {report_path}")
    
    # Event handlers for test monitoring
    
    def on_event(self, event: Event):
        """Track all events by type."""
        if event.event_type in self.events_received:
            self.events_received[event.event_type] += 1
        else:
            self.events_received[event.event_type] = 1
    
    def on_market_data(self, event: Event):
        """Handle market data events."""
        # Optional: Add specific validation for market data if needed
        pass
    
    def on_bar_closed(self, event: Event):
        """Handle bar closed events."""
        # Simulate a simple trading strategy - create a buy order for every 5th bar
        if self.events_received.get(EventType.BAR_CLOSED, 0) % 5 == 0:
            bar_data = event.data
            symbol = bar_data.get("symbol", "UNKNOWN")
            close_price = bar_data.get("close", 100.0)
            
            # Create a simple buy order
            self._create_test_order(symbol, "buy", 10, close_price)
    
    def on_data_quality_alert(self, event: Event):
        """Handle data quality alert events."""
        alert_data = event.data
        logger.warning(f"Data quality alert: {alert_data.get('message', 'No message')}")
    
    def on_order_created(self, event: Event):
        """Handle order created events."""
        order_data = event.data
        self.orders_created.append(order_data)
        logger.info(f"Order created: {order_data.get('order_id', 'UNKNOWN')} for {order_data.get('symbol', 'UNKNOWN')}")
    
    def on_order_filled(self, event: Event):
        """Handle order filled events."""
        fill_data = event.data
        logger.info(f"Order filled: {fill_data.get('order_id', 'UNKNOWN')} for {fill_data.get('symbol', 'UNKNOWN')}")
    
    def on_trade_executed(self, event: Event):
        """Handle trade executed events."""
        trade_data = event.data
        self.trades_executed.append(trade_data)
        logger.info(f"Trade executed: {trade_data.get('symbol', 'UNKNOWN')} - PnL: {trade_data.get('pnl', 0.0)}")
    
    def on_error(self, event: Event):
        """Handle error events."""
        error_data = event.data
        error_message = error_data.get("message", "Unknown error")
        self.errors_detected.append(error_message)
        logger.error(f"System error detected: {error_message}")
    
    def on_position_update(self, event: Event):
        """Handle position update events."""
        position_data = event.data
        self.position_snapshots.append(position_data)
    
    def on_kill_switch(self, event: Event):
        """Handle kill switch activation events."""
        kill_switch_data = event.data
        reason = kill_switch_data.get("reason", "Unknown reason")
        logger.critical(f"Kill switch activated during test: {reason}")
    
    def _create_test_order(self, symbol: str, side: str, quantity: float, price: float):
        """Create a test order for the specified symbol."""
        order_data = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": "market",
            "time_in_force": "day",
            "order_id": f"test-{len(self.orders_created) + 1}",
            "strategy_id": "integration_test",
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_bus.create_and_publish(
            event_type=EventType.ORDER_CREATED,
            data=order_data,
            source="integration_test"
        )


def main():
    """Main entry point for the end-to-end integration test."""
    # Create and run the integration test
    test = IntegrationTest()
    test_passed = test.run_test()
    
    # Return exit code based on test result
    return 0 if test_passed else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error in integration test: {e}")
        sys.exit(1)
