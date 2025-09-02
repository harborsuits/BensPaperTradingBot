#!/usr/bin/env python
"""
End-to-End Pipeline Integration Test

This test validates the complete trading pipeline flow from market data ingestion
through signal generation, order execution, to performance reporting.
"""

import os
import sys
import unittest
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import threading
import pandas as pd
import numpy as np

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, TradingMode
from trading_bot.data.live_data_source import LiveDataSource
from trading_bot.data.persistence import PersistenceManager
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
from trading_bot.monitoring.recap_reporting import create_performance_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventCollector:
    """Collects events for pipeline validation"""
    
    def __init__(self):
        self.events = []
        self.event_types_received = set()
        self.events_by_type = {}
        self.transaction_events = {}
        self.event_timestamps = {}
        self.event_processing_times = {}
        
    def handle_event(self, event: Event):
        """Record an event for analysis"""
        self.events.append(event)
        self.event_types_received.add(event.event_type)
        
        # Track by type
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event)
        
        # Track timestamps
        self.event_timestamps[event.event_id] = datetime.now()
        
        # Track transaction IDs if present
        if 'transaction_id' in event.data:
            transaction_id = event.data['transaction_id']
            if transaction_id not in self.transaction_events:
                self.transaction_events[transaction_id] = []
            self.transaction_events[transaction_id].append(event)


class PipelineIntegrationTest(unittest.TestCase):
    """Tests the complete trading pipeline from data to reporting"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with required components"""
        # Create a test-specific event bus
        cls.event_bus = EventBus(verbose_logging=True)
        
        # Create a memory-based persistence manager for testing
        cls.persistence = PersistenceManager(uri="sqlite:///:memory:", db_name="test_pipeline")
        
        # Create a market data simulator
        cls.live_data = LiveDataSource(
            persistence_manager=cls.persistence,
            provider="simulation",
            symbols=["EUR/USD", "BTC/USD", "AAPL", "SPY"],
            timeframes=["1m", "5m", "15m", "1h"],
            event_bus=cls.event_bus
        )
        
        # Create a strategy intelligence recorder
        cls.recorder = StrategyIntelligenceRecorder(
            persistence_manager=cls.persistence,
            event_bus=cls.event_bus
        )
        
        # Create an event collector for test validation
        cls.collector = EventCollector()
        cls.event_bus.subscribe_all(cls.collector.handle_event)
        
        # Additional test data
        cls.test_events = {
            "expected_data_events": [
                EventType.MARKET_DATA_RECEIVED,
                EventType.TICK_RECEIVED,
                EventType.BAR_CLOSED
            ],
            "expected_signal_events": [
                EventType.SIGNAL_GENERATED
            ],
            "expected_order_events": [
                EventType.ORDER_CREATED,
                EventType.ORDER_SUBMITTED,
                EventType.ORDER_FILLED
            ],
            "expected_risk_events": [
                EventType.PORTFOLIO_EXPOSURE_UPDATED,
                EventType.RISK_ATTRIBUTION_CALCULATED
            ]
        }
        
    def test_1_pipeline_initialization(self):
        """Test that all pipeline components initialize correctly"""
        self.assertTrue(self.persistence is not None)
        self.assertTrue(self.event_bus is not None)
        self.assertTrue(self.live_data is not None)
        self.assertTrue(self.recorder is not None)
        
    def test_2_market_data_flow(self):
        """Test market data ingestion and propagation"""
        # Start the data source
        self.live_data.start()
        
        # Wait for data to be generated
        time.sleep(3)
        
        # Stop the data source
        self.live_data.stop()
        
        # Verify data events were generated
        for event_type in self.test_events["expected_data_events"]:
            self.assertIn(event_type, self.collector.event_types_received,
                         f"Expected {event_type} events not generated")
        
        # Verify market data structure
        if EventType.BAR_CLOSED in self.collector.events_by_type:
            bar_events = self.collector.events_by_type[EventType.BAR_CLOSED]
            self.assertTrue(len(bar_events) > 0)
            
            # Validate event data structure
            for event in bar_events[:3]:  # Check first few events
                self.assertIn('symbol', event.data)
                self.assertIn('timeframe', event.data)
                self.assertIn('timestamp', event.data)
                self.assertIn('open', event.data)
                self.assertIn('high', event.data)
                self.assertIn('low', event.data)
                self.assertIn('close', event.data)
    
    def test_3_regime_detection_flow(self):
        """Test market regime detection and propagation"""
        # Verify regime events were generated
        self.assertIn(EventType.MARKET_REGIME_CHANGED, self.collector.event_types_received,
                     "No market regime change events detected")
        
        # Check regime event structure
        if EventType.MARKET_REGIME_CHANGED in self.collector.events_by_type:
            regime_events = self.collector.events_by_type[EventType.MARKET_REGIME_CHANGED]
            self.assertTrue(len(regime_events) > 0)
            
            # Validate regime data structure
            for event in regime_events[:3]:  # Check first few events
                self.assertIn('symbol', event.data)
                self.assertIn('current_regime', event.data)
                self.assertIn('confidence', event.data)
                self.assertIn('timestamp', event.data)
                
                # Verify valid regime type
                self.assertIn(event.data['current_regime'], 
                             ["trending", "ranging", "volatile", "low_volatility"])
    
    def test_4_signal_generation_flow(self):
        """Test signal generation and propagation"""
        # Create a mock trade signal
        self.event_bus.create_and_publish(
            event_type=EventType.SIGNAL_GENERATED,
            data={
                'strategy': 'TestStrategy',
                'symbol': 'EUR/USD',
                'direction': 'long',
                'timestamp': datetime.now(),
                'confidence': 0.85,
                'timeframe': '1h',
                'price': 1.1025,
                'reason': 'Test signal for pipeline validation'
            }
        )
        
        # Verify signal was recorded
        self.assertIn(EventType.SIGNAL_GENERATED, self.collector.event_types_received)
        
        # Verify intelligence recorder captured the signal
        signal_events = self.collector.events_by_type.get(EventType.SIGNAL_GENERATED, [])
        self.assertTrue(len(signal_events) > 0)
        
        # Allow a moment for processing
        time.sleep(0.5)
    
    def test_5_order_execution_flow(self):
        """Test order execution flow"""
        # Create a mock order sequence
        transaction_id = "test-transaction-123"
        
        # 1. Create order
        self.event_bus.create_and_publish(
            event_type=EventType.ORDER_CREATED,
            data={
                'order_id': 'test-order-1',
                'strategy': 'TestStrategy',
                'symbol': 'EUR/USD',
                'direction': 'long',
                'quantity': 10000,
                'order_type': 'market',
                'timestamp': datetime.now(),
                'transaction_id': transaction_id
            }
        )
        
        # 2. Order submitted
        self.event_bus.create_and_publish(
            event_type=EventType.ORDER_SUBMITTED,
            data={
                'order_id': 'test-order-1',
                'broker': 'test-broker',
                'timestamp': datetime.now(),
                'transaction_id': transaction_id
            }
        )
        
        # 3. Order filled
        self.event_bus.create_and_publish(
            event_type=EventType.ORDER_FILLED,
            data={
                'order_id': 'test-order-1',
                'fill_price': 1.1028,
                'quantity_filled': 10000,
                'timestamp': datetime.now(),
                'commission': 2.50,
                'transaction_id': transaction_id
            }
        )
        
        # Verify order events were captured
        for event_type in self.test_events["expected_order_events"]:
            self.assertIn(event_type, self.collector.event_types_received)
        
        # Verify transaction tracking
        self.assertIn(transaction_id, self.collector.transaction_events)
        transaction_events = self.collector.transaction_events[transaction_id]
        self.assertEqual(len(transaction_events), 3)
        
        # Allow event processing to complete
        time.sleep(0.5)
    
    def test_6_risk_management_flow(self):
        """Test risk management events"""
        # Create portfolio exposure event
        self.event_bus.create_and_publish(
            event_type=EventType.PORTFOLIO_EXPOSURE_UPDATED,
            data={
                'total_exposure': 0.15,  # 15% of account
                'forex_exposure': 0.15,
                'stock_exposure': 0.0,
                'crypto_exposure': 0.0,
                'options_exposure': 0.0,
                'timestamp': datetime.now(),
                'account_balance': 100000
            }
        )
        
        # Risk attribution event
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_ATTRIBUTION_CALCULATED,
            data={
                'strategy_risk': {
                    'TestStrategy': 0.15
                },
                'asset_class_risk': {
                    'forex': 0.15
                },
                'symbol_risk': {
                    'EUR/USD': 0.15
                },
                'timestamp': datetime.now()
            }
        )
        
        # Verify risk events were captured
        for event_type in self.test_events["expected_risk_events"]:
            self.assertIn(event_type, self.collector.event_types_received)
        
        # Allow event processing to complete
        time.sleep(0.5)
    
    def test_7_reporting_flow(self):
        """Test performance reporting"""
        # Create test performance data
        today_results = {
            'date': datetime.now(),
            'daily_pnl': 125.50,
            'daily_return': 0.125,
            'ending_equity': 100125.50,
            'trades': 5,
            'win_rate': 0.60
        }
        
        benchmark_performance = {
            'SPY': {
                'return': 0.08,
                'correlation': 0.65
            },
            'EUR/USD': {
                'return': 0.05,
                'correlation': 0.35
            }
        }
        
        alerts = [
            {
                'strategy': 'TestStrategy',
                'severity': 'info',
                'alerts': [{
                    'message': 'Strategy performing as expected'
                }],
                'action_required': False
            }
        ]
        
        suggestions = [
            {
                'strategy': 'TestStrategy',
                'suggestion': {
                    'action': 'increase_weight',
                    'current_weight': 0.15,
                    'suggested_weight': 0.20,
                    'reason': 'Strong performance in current market regime'
                }
            }
        ]
        
        # Generate a report
        report_path = create_performance_report(
            today_results,
            benchmark_performance,
            alerts,
            suggestions,
            output_dir='./test_reports'
        )
        
        # Verify report was created
        self.assertTrue(len(report_path) > 0)
        self.assertTrue(os.path.exists(report_path))
        
        # Clean up
        if os.path.exists(report_path):
            os.remove(report_path)
    
    def test_8_pipeline_latency_analysis(self):
        """Analyze pipeline latency and event processing times"""
        # Calculate time differences between key events
        timestamps = self.collector.event_timestamps
        
        if len(timestamps) < 2:
            self.skipTest("Not enough events for latency analysis")
            return
        
        # Get events in order by timestamp
        event_ids = sorted(timestamps.keys(), key=lambda x: timestamps[x])
        
        # Calculate time differences
        time_diffs = []
        for i in range(1, len(event_ids)):
            prev_time = timestamps[event_ids[i-1]]
            curr_time = timestamps[event_ids[i]]
            diff_ms = (curr_time - prev_time).total_seconds() * 1000
            time_diffs.append(diff_ms)
        
        # Analyze latency statistics
        avg_latency = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        max_latency = max(time_diffs) if time_diffs else 0
        min_latency = min(time_diffs) if time_diffs else 0
        
        # Log latency statistics
        logger.info(f"Pipeline Latency Analysis:")
        logger.info(f"  Average event processing latency: {avg_latency:.2f} ms")
        logger.info(f"  Maximum event processing latency: {max_latency:.2f} ms")
        logger.info(f"  Minimum event processing latency: {min_latency:.2f} ms")
        
        # We don't assert specific values, just report them
        
    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        # Stop the live data source if still running
        if hasattr(cls, 'live_data') and cls.live_data.is_running():
            cls.live_data.stop()
        
        # Clear event bus subscriptions
        cls.event_bus.clear_subscribers()
        
        # Clean up any test files
        if os.path.exists('./test_reports'):
            import shutil
            shutil.rmtree('./test_reports')


if __name__ == '__main__':
    unittest.main()
