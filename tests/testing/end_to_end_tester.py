#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Testing Framework for Pattern-Signal Integration

This module provides a comprehensive testing framework to validate the flow of signals
from external sources (TradingView, Alpaca, Finnhub) through the pattern recognition
and signal processing pipeline to trade execution.
"""

import os
import sys
import json
import time
import logging
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import concurrent.futures

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import trading bot modules
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.integrations.webhook_handler import WebhookHandler
from trading_bot.strategies.external_signal_strategy import (
    ExternalSignalStrategy, ExternalSignal, SignalSource, SignalType, Direction
)
from trading_bot.strategies.pattern_enhanced_strategy import PatternEnhancedStrategy
from trading_bot.analysis.pattern_recognition import PatternRecognition
from examples.pattern_signal_integration import PatternSignalIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("E2ETester")


class TestEvent(Enum):
    """Enumeration of test events for monitoring signal flow."""
    SIGNAL_SENT = "signal_sent"
    SIGNAL_RECEIVED = "signal_received"
    PATTERN_DETECTED = "pattern_detected"
    PATTERN_CONFIRMED = "pattern_confirmed"
    TRADE_SIGNAL_GENERATED = "trade_signal_generated"
    TRADE_EXECUTED = "trade_executed"
    ERROR = "error"


class TestResult:
    """Class to store and report test results."""

    def __init__(self, test_name: str):
        """
        Initialize a test result.
        
        Args:
            test_name: Name of the test
        """
        self.test_name = test_name
        self.start_time = datetime.now()
        self.end_time = None
        self.success = False
        self.events = []
        self.metrics = {}
        self.errors = []
    
    def add_event(self, event_type: TestEvent, data: Dict[str, Any]):
        """Add a test event."""
        self.events.append({
            "type": event_type.value,
            "timestamp": datetime.now(),
            "data": data
        })
    
    def add_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Add an error."""
        self.errors.append({
            "message": error_msg,
            "exception": str(exception) if exception else None,
            "timestamp": datetime.now()
        })
        self.add_event(TestEvent.ERROR, {"message": error_msg})
    
    def complete(self, success: bool = True):
        """Mark the test as complete."""
        self.end_time = datetime.now()
        self.success = success
        self.metrics["duration_seconds"] = (self.end_time - self.start_time).total_seconds()
        
        # Calculate signal flow metrics
        events_by_type = {}
        for event in self.events:
            event_type = event["type"]
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        # Count events by type
        for event_type, events in events_by_type.items():
            self.metrics[f"{event_type}_count"] = len(events)
        
        # Check if signals were properly received and processed
        signals_sent = events_by_type.get(TestEvent.SIGNAL_SENT.value, [])
        signals_received = events_by_type.get(TestEvent.SIGNAL_RECEIVED.value, [])
        trades_generated = events_by_type.get(TestEvent.TRADE_SIGNAL_GENERATED.value, [])
        
        if signals_sent and signals_received:
            self.metrics["signal_reception_rate"] = len(signals_received) / len(signals_sent)
        
        if signals_received and trades_generated:
            self.metrics["trade_generation_rate"] = len(trades_generated) / len(signals_received)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test result to a dictionary."""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "events": self.events,
            "metrics": self.metrics,
            "errors": self.errors,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }
    
    def report(self) -> str:
        """Generate a text report of the test results."""
        lines = [
            f"Test: {self.test_name}",
            f"Status: {'SUCCESS' if self.success else 'FAILURE'}",
            f"Duration: {self.metrics.get('duration_seconds', 0):.2f} seconds",
            f"Events: {len(self.events)}",
            f"Errors: {len(self.errors)}"
        ]
        
        # Add metrics
        lines.append("\nMetrics:")
        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value}")
        
        # Add errors if any
        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  {error['timestamp'].strftime('%H:%M:%S.%f')} - {error['message']}")
        
        # Add event summary
        event_counts = {}
        for event in self.events:
            event_type = event["type"]
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
        
        lines.append("\nEvent Summary:")
        for event_type, count in event_counts.items():
            lines.append(f"  {event_type}: {count}")
        
        return "\n".join(lines)


class SignalTester:
    """
    Tests end-to-end signal flow through the trading system.
    
    This class provides tools to validate the complete signal flow:
    1. Sending test signals from various sources
    2. Monitoring signal reception and processing
    3. Validating pattern recognition and confirmation
    4. Tracking trade signal generation
    5. Measuring performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal tester.
        
        Args:
            config: Configuration for the integration and testing
        """
        self.config = config
        self.integration = PatternSignalIntegration(config)
        self.test_results = []
        self.current_test = None
        
        # Custom event handler for monitoring
        self.event_bus = EventBus()
        self._setup_event_monitors()
    
    def _setup_event_monitors(self):
        """Set up event monitors to track signal flow."""
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.EXTERNAL_SIGNAL, self._handle_external_signal_event)
        self.event_bus.subscribe(EventType.PATTERN_DETECTED, self._handle_pattern_detected_event)
        self.event_bus.subscribe(EventType.TRADE_SIGNAL, self._handle_trade_signal_event)
    
    def _handle_external_signal_event(self, event: Event):
        """Handle external signal event."""
        if self.current_test:
            signal_data = event.data.get("signal", {})
            self.current_test.add_event(TestEvent.SIGNAL_RECEIVED, {
                "signal_data": signal_data,
                "event_id": id(event)
            })
    
    def _handle_pattern_detected_event(self, event: Event):
        """Handle pattern detected event."""
        if self.current_test:
            self.current_test.add_event(TestEvent.PATTERN_DETECTED, {
                "pattern_data": event.data,
                "event_id": id(event)
            })
    
    def _handle_trade_signal_event(self, event: Event):
        """Handle trade signal event."""
        if self.current_test:
            self.current_test.add_event(TestEvent.TRADE_SIGNAL_GENERATED, {
                "trade_data": event.data,
                "event_id": id(event)
            })
    
    def start_test(self, test_name: str) -> TestResult:
        """
        Start a new test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test result object
        """
        self.current_test = TestResult(test_name)
        logger.info(f"Starting test: {test_name}")
        return self.current_test
    
    def end_test(self, success: bool = True) -> TestResult:
        """
        End the current test.
        
        Args:
            success: Whether the test was successful
            
        Returns:
            Completed test result
        """
        if self.current_test:
            self.current_test.complete(success)
            self.test_results.append(self.current_test)
            
            logger.info(f"Test completed: {self.current_test.test_name} - {'SUCCESS' if success else 'FAILURE'}")
            test_result = self.current_test
            self.current_test = None
            return test_result
        
        return None
    
    def send_test_tradingview_signal(self, signal_data: Dict[str, Any]) -> requests.Response:
        """
        Send a test TradingView webhook signal.
        
        Args:
            signal_data: Signal data to send
            
        Returns:
            HTTP response from the webhook endpoint
        """
        if not self.integration.webhook_handler:
            raise ValueError("Webhook handler not initialized")
        
        try:
            webhook_url = f"http://localhost:{self.integration.webhook_handler.port}/{self.integration.webhook_handler.path}"
            
            # Log the signal being sent
            if self.current_test:
                self.current_test.add_event(TestEvent.SIGNAL_SENT, {
                    "source": "tradingview",
                    "data": signal_data
                })
            
            # Send the webhook request
            response = requests.post(
                webhook_url,
                json=signal_data,
                headers={"Content-Type": "application/json"}
            )
            
            return response
            
        except Exception as e:
            if self.current_test:
                self.current_test.add_error("Error sending TradingView signal", e)
            logger.error(f"Error sending TradingView signal: {str(e)}")
            raise
    
    def send_test_alpaca_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Simulate an Alpaca trade update signal.
        
        Args:
            signal_data: Alpaca trade update data
            
        Returns:
            True if signal was sent successfully
        """
        try:
            # Get the external signal strategy
            strategy = self.integration.external_signal_strategy
            
            # Log the signal being sent
            if self.current_test:
                self.current_test.add_event(TestEvent.SIGNAL_SENT, {
                    "source": "alpaca",
                    "data": signal_data
                })
            
            # Call the handler directly
            # We need to create a mock trade update object
            class MockTradeUpdate:
                def __init__(self, data):
                    self.__dict__ = {"order": data}
            
            # Create and process mock update
            mock_update = MockTradeUpdate(signal_data)
            strategy._handle_alpaca_trade_update(mock_update)
            
            return True
            
        except Exception as e:
            if self.current_test:
                self.current_test.add_error("Error sending Alpaca signal", e)
            logger.error(f"Error sending Alpaca signal: {str(e)}")
            return False
    
    def send_test_finnhub_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Simulate a Finnhub trade signal.
        
        Args:
            signal_data: Finnhub trade data
            
        Returns:
            True if signal was sent successfully
        """
        try:
            # Get the external signal strategy
            strategy = self.integration.external_signal_strategy
            
            # Log the signal being sent
            if self.current_test:
                self.current_test.add_event(TestEvent.SIGNAL_SENT, {
                    "source": "finnhub",
                    "data": signal_data
                })
            
            # Call the handler directly
            strategy._handle_finnhub_trade(signal_data)
            
            return True
            
        except Exception as e:
            if self.current_test:
                self.current_test.add_error("Error sending Finnhub signal", e)
            logger.error(f"Error sending Finnhub signal: {str(e)}")
            return False
    
    def simulate_pattern_detection(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Simulate internal pattern detection.
        
        Args:
            pattern_data: Pattern data
            
        Returns:
            True if pattern was simulated successfully
        """
        try:
            # Get the pattern strategy
            strategy = self.integration.pattern_strategy
            
            # Log the pattern being simulated
            if self.current_test:
                self.current_test.add_event(TestEvent.PATTERN_DETECTED, {
                    "pattern_data": pattern_data
                })
            
            # Create and publish pattern event
            event = Event(
                event_type=EventType.PATTERN_DETECTED,
                data=pattern_data
            )
            
            self.event_bus.publish(event)
            
            return True
            
        except Exception as e:
            if self.current_test:
                self.current_test.add_error("Error simulating pattern detection", e)
            logger.error(f"Error simulating pattern detection: {str(e)}")
            return False
    
    def monitor_signal_flow(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Monitor signal flow through the system for a given duration.
        
        Args:
            duration_seconds: Duration to monitor in seconds
            
        Returns:
            Signal flow metrics
        """
        if not self.current_test:
            raise ValueError("No active test")
        
        logger.info(f"Monitoring signal flow for {duration_seconds} seconds")
        
        # Sleep for the specified duration
        time.sleep(duration_seconds)
        
        # Compile metrics
        events_by_type = {}
        for event in self.current_test.events:
            event_type = event["type"]
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        metrics = {}
        
        # Count events by type
        for event_type in TestEvent:
            type_value = event_type.value
            metrics[f"{type_value}_count"] = len(events_by_type.get(type_value, []))
        
        # Calculate signal flow rates
        signals_sent = events_by_type.get(TestEvent.SIGNAL_SENT.value, [])
        signals_received = events_by_type.get(TestEvent.SIGNAL_RECEIVED.value, [])
        trades_generated = events_by_type.get(TestEvent.TRADE_SIGNAL_GENERATED.value, [])
        
        if signals_sent and signals_received:
            metrics["signal_reception_rate"] = len(signals_received) / len(signals_sent)
        
        if signals_received and trades_generated:
            metrics["trade_generation_rate"] = len(trades_generated) / len(signals_received)
        
        logger.info(f"Signal flow monitoring complete: {metrics}")
        
        # Update test metrics
        self.current_test.metrics.update(metrics)
        
        return metrics
    
    def run_basic_test(self, signal_data: Dict[str, Any], source: str = "tradingview", 
                     duration_seconds: int = 10) -> TestResult:
        """
        Run a basic end-to-end test with a single signal.
        
        Args:
            signal_data: Signal data to send
            source: Signal source (tradingview, alpaca, finnhub)
            duration_seconds: Duration to monitor signal flow
            
        Returns:
            Test result
        """
        # Start the integration if not already started
        if not hasattr(self.integration, '_started') or not self.integration._started:
            self.integration.start()
        
        # Start the test
        test_name = f"Basic {source} signal test"
        self.start_test(test_name)
        
        try:
            # Send the signal based on source
            if source == "tradingview":
                response = self.send_test_tradingview_signal(signal_data)
                if response.status_code != 200:
                    self.current_test.add_error(f"TradingView webhook returned status {response.status_code}: {response.text}")
            elif source == "alpaca":
                success = self.send_test_alpaca_signal(signal_data)
                if not success:
                    self.current_test.add_error("Failed to send Alpaca signal")
            elif source == "finnhub":
                success = self.send_test_finnhub_signal(signal_data)
                if not success:
                    self.current_test.add_error("Failed to send Finnhub signal")
            else:
                self.current_test.add_error(f"Unknown source: {source}")
                return self.end_test(False)
            
            # Monitor signal flow
            self.monitor_signal_flow(duration_seconds)
            
            # Check for errors
            success = len(self.current_test.errors) == 0
            
            # End the test
            return self.end_test(success)
            
        except Exception as e:
            self.current_test.add_error(f"Unexpected error in test: {str(e)}", e)
            return self.end_test(False)
        finally:
            # Stop the integration
            if not self.config.get("keep_running", False):
                self.integration.stop()
    
    def run_pattern_confirmation_test(self, signal_data: Dict[str, Any], pattern_data: Dict[str, Any],
                                  source: str = "tradingview", duration_seconds: int = 15) -> TestResult:
        """
        Run a test for pattern confirmation of a signal.
        
        Args:
            signal_data: Signal data to send
            pattern_data: Pattern data to simulate
            source: Signal source
            duration_seconds: Duration to monitor signal flow
            
        Returns:
            Test result
        """
        # Start the integration if not already started
        if not hasattr(self.integration, '_started') or not self.integration._started:
            self.integration.start()
        
        # Start the test
        test_name = f"Pattern confirmation test for {source}"
        self.start_test(test_name)
        
        try:
            # Send the signal first
            if source == "tradingview":
                response = self.send_test_tradingview_signal(signal_data)
                if response.status_code != 200:
                    self.current_test.add_error(f"TradingView webhook returned status {response.status_code}: {response.text}")
            elif source == "alpaca":
                success = self.send_test_alpaca_signal(signal_data)
                if not success:
                    self.current_test.add_error("Failed to send Alpaca signal")
            elif source == "finnhub":
                success = self.send_test_finnhub_signal(signal_data)
                if not success:
                    self.current_test.add_error("Failed to send Finnhub signal")
            else:
                self.current_test.add_error(f"Unknown source: {source}")
                return self.end_test(False)
            
            # Wait a moment for signal processing
            time.sleep(2)
            
            # Then simulate pattern detection
            success = self.simulate_pattern_detection(pattern_data)
            if not success:
                self.current_test.add_error("Failed to simulate pattern detection")
            
            # Monitor signal flow
            metrics = self.monitor_signal_flow(duration_seconds)
            
            # Check if pattern confirmation occurred
            trades_generated = metrics.get("trade_signal_generated_count", 0)
            
            if trades_generated == 0:
                self.current_test.add_error("No trades were generated from pattern confirmation")
                return self.end_test(False)
            
            # End the test
            return self.end_test(len(self.current_test.errors) == 0)
            
        except Exception as e:
            self.current_test.add_error(f"Unexpected error in test: {str(e)}", e)
            return self.end_test(False)
        finally:
            # Stop the integration
            if not self.config.get("keep_running", False):
                self.integration.stop()
    
    def run_concurrent_signals_test(self, signals: List[Tuple[str, Dict[str, Any]]], 
                                 duration_seconds: int = 20) -> TestResult:
        """
        Run a test with multiple concurrent signals from different sources.
        
        Args:
            signals: List of (source, signal_data) tuples
            duration_seconds: Duration to monitor signal flow
            
        Returns:
            Test result
        """
        # Start the integration if not already started
        if not hasattr(self.integration, '_started') or not self.integration._started:
            self.integration.start()
        
        # Start the test
        test_name = "Concurrent signals test"
        self.start_test(test_name)
        
        try:
            # Use thread pool to send signals concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(signals)) as executor:
                futures = []
                
                for source, signal_data in signals:
                    if source == "tradingview":
                        future = executor.submit(self.send_test_tradingview_signal, signal_data)
                    elif source == "alpaca":
                        future = executor.submit(self.send_test_alpaca_signal, signal_data)
                    elif source == "finnhub":
                        future = executor.submit(self.send_test_finnhub_signal, signal_data)
                    else:
                        self.current_test.add_error(f"Unknown source: {source}")
                        continue
                    
                    futures.append((source, future))
                
                # Wait for all signals to be sent
                for source, future in futures:
                    try:
                        if source == "tradingview":
                            response = future.result()
                            if response.status_code != 200:
                                self.current_test.add_error(f"TradingView webhook returned status {response.status_code}: {response.text}")
                        else:
                            success = future.result()
                            if not success:
                                self.current_test.add_error(f"Failed to send {source} signal")
                    except Exception as e:
                        self.current_test.add_error(f"Error sending {source} signal: {str(e)}", e)
            
            # Monitor signal flow
            self.monitor_signal_flow(duration_seconds)
            
            # Check for errors
            success = len(self.current_test.errors) == 0
            
            # End the test
            return self.end_test(success)
            
        except Exception as e:
            self.current_test.add_error(f"Unexpected error in test: {str(e)}", e)
            return self.end_test(False)
        finally:
            # Stop the integration
            if not self.config.get("keep_running", False):
                self.integration.stop()
    
    def generate_report(self) -> str:
        """
        Generate a report of all test results.
        
        Returns:
            Text report
        """
        if not self.test_results:
            return "No tests have been run"
        
        lines = [
            "==================================",
            "     End-to-End Testing Report    ",
            "==================================",
            f"Tests Run: {len(self.test_results)}",
            f"Tests Passed: {sum(1 for t in self.test_results if t.success)}",
            f"Tests Failed: {sum(1 for t in self.test_results if not t.success)}",
            "==================================\n"
        ]
        
        for i, result in enumerate(self.test_results, 1):
            lines.append(f"Test #{i}: {result.test_name}")
            lines.append(f"Status: {'SUCCESS' if result.success else 'FAILURE'}")
            lines.append(f"Duration: {result.metrics.get('duration_seconds', 0):.2f} seconds")
            
            # Add key metrics
            metrics = [
                f"Signals Sent: {result.metrics.get('signal_sent_count', 0)}",
                f"Signals Received: {result.metrics.get('signal_received_count', 0)}",
                f"Trades Generated: {result.metrics.get('trade_signal_generated_count', 0)}",
                f"Reception Rate: {result.metrics.get('signal_reception_rate', 0):.2f}",
                f"Trade Generation Rate: {result.metrics.get('trade_generation_rate', 0):.2f}"
            ]
            
            lines.append("Metrics:")
            for metric in metrics:
                lines.append(f"  {metric}")
            
            # Add errors if any
            if result.errors:
                lines.append("Errors:")
                for error in result.errors:
                    lines.append(f"  - {error['message']}")
            
            lines.append("-" * 50 + "\n")
        
        return "\n".join(lines)
    
    def save_results(self, filename: str) -> None:
        """
        Save test results to a JSON file.
        
        Args:
            filename: Output filename
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_count": len(self.test_results),
            "tests_passed": sum(1 for t in self.test_results if t.success),
            "tests_failed": sum(1 for t in self.test_results if not t.success),
            "results": [t.to_dict() for t in self.test_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Test results saved to {filename}")


def main():
    """Run example tests."""
    # Example configuration
    config = {
        "webhook_port": 5000,
        "webhook_path": "webhook",
        "webhook_auth_token": None,
        "keep_running": False,
        
        "pattern_strategy_config": {
            "confidence_threshold": 0.7,
            "lookback_periods": 20,
            "confirmation_required": True
        }
    }
    
    # Create the tester
    tester = SignalTester(config)
    
    # Example TradingView signal
    tradingview_signal = {
        "symbol": "EURUSD",
        "action": "buy",
        "price": 1.0550,
        "timestamp": datetime.now().isoformat(),
        "timeframe": "1h",
        "strategy": "Test Strategy",
        "source": "tradingview"
    }
    
    # Example Alpaca signal
    alpaca_signal = {
        "symbol": "AAPL",
        "side": "buy",
        "status": "filled",
        "filled_avg_price": 175.50,
        "qty": 10,
        "filled_qty": 10,
        "id": "test-order-id",
        "client_order_id": "test-client-id",
        "type": "market"
    }
    
    # Example Finnhub signal
    finnhub_signal = {
        "s": "MSFT",  # Symbol
        "p": 280.75,  # Price
        "v": 100,     # Volume
        "t": int(datetime.now().timestamp() * 1000)  # Timestamp in milliseconds
    }
    
    # Example pattern data
    pattern_data = {
        "symbol": "EURUSD",
        "pattern_type": "pin_bar",
        "direction": "long",
        "confidence": 0.85,
        "price": 1.0550,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Run individual tests
        print("Running TradingView signal test...")
        result1 = tester.run_basic_test(tradingview_signal, "tradingview")
        print(result1.report())
        
        print("\nRunning Alpaca signal test...")
        result2 = tester.run_basic_test(alpaca_signal, "alpaca")
        print(result2.report())
        
        print("\nRunning Finnhub signal test...")
        result3 = tester.run_basic_test(finnhub_signal, "finnhub")
        print(result3.report())
        
        print("\nRunning pattern confirmation test...")
        result4 = tester.run_pattern_confirmation_test(tradingview_signal, pattern_data)
        print(result4.report())
        
        print("\nRunning concurrent signals test...")
        result5 = tester.run_concurrent_signals_test([
            ("tradingview", tradingview_signal),
            ("alpaca", alpaca_signal),
            ("finnhub", finnhub_signal)
        ])
        print(result5.report())
        
        # Generate and print overall report
        print("\n" + "=" * 60)
        print(tester.generate_report())
        
        # Save results to file
        tester.save_results("test_results.json")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nError in tests: {str(e)}")
    finally:
        print("\nTests completed.")


if __name__ == "__main__":
    main()
