#!/usr/bin/env python
"""
Data Ingestion Test

This test validates the market data ingestion phase of the trading pipeline,
ensuring proper data flow from providers to the event bus.
"""

import unittest
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.live_data_source import LiveDataSource
from trading_bot.data.persistence import PersistenceManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventTracker:
    """Tracks events for test validation"""
    
    def __init__(self):
        self.events = []
        self.event_types_received = set()
        self.events_by_type = {}
        self.symbols_received = set()
        self.timeframes_received = set()
        
    def handle_event(self, event: Event):
        """Process and record an event"""
        self.events.append(event)
        self.event_types_received.add(event.event_type)
        
        # Track by type
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event)
        
        # Track symbols and timeframes for market data
        if event.event_type in [EventType.MARKET_DATA_RECEIVED, EventType.BAR_CLOSED]:
            if 'symbol' in event.data:
                self.symbols_received.add(event.data['symbol'])
            if 'timeframe' in event.data:
                self.timeframes_received.add(event.data['timeframe'])


class DataIngestionTest(unittest.TestCase):
    """Tests the market data ingestion phase"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        # Create test event bus
        cls.event_bus = EventBus()
        
        # Create memory-based persistence
        cls.persistence = PersistenceManager(uri="sqlite:///:memory:", db_name="test_data")
        
        # Create event tracker
        cls.tracker = EventTracker()
        cls.event_bus.subscribe_all(cls.tracker.handle_event)
        
        # Test configuration
        cls.test_symbols = ["EUR/USD", "BTC/USD", "AAPL", "SPY"]
        cls.test_timeframes = ["1m", "5m", "15m", "1h"]
        
    def test_1_simulation_data_provider(self):
        """Test data ingestion with simulation provider"""
        # Create simulation data source
        data_source = LiveDataSource(
            persistence_manager=self.persistence,
            provider="simulation",
            symbols=self.test_symbols,
            timeframes=self.test_timeframes,
            event_bus=self.event_bus
        )
        
        # Start data source
        data_source.start()
        
        # Wait for data events
        logger.info("Waiting for simulation data events...")
        max_wait = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check if we've received events for all symbols and timeframes
            if (all(symbol in self.tracker.symbols_received for symbol in self.test_symbols) and
                all(tf in self.tracker.timeframes_received for tf in self.test_timeframes)):
                break
            time.sleep(0.1)
        
        # Stop data source
        data_source.stop()
        
        # Validate results
        self.assertIn(EventType.MARKET_DATA_RECEIVED, self.tracker.event_types_received)
        self.assertIn(EventType.BAR_CLOSED, self.tracker.event_types_received)
        
        for symbol in self.test_symbols:
            self.assertIn(symbol, self.tracker.symbols_received)
        
        for timeframe in self.test_timeframes:
            self.assertIn(timeframe, self.tracker.timeframes_received)
        
        logger.info(f"Received {len(self.tracker.events)} data events")
        logger.info(f"Symbols received: {self.tracker.symbols_received}")
        logger.info(f"Timeframes received: {self.tracker.timeframes_received}")
    
    def test_2_historical_data_ingestion(self):
        """Test historical data ingestion"""
        # Clear previous events
        self.tracker.events = []
        self.tracker.events_by_type = {}
        
        # Define historical range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Create historical data request event
        historical_request = Event(
            event_type=EventType.HISTORICAL_DATA_REQUESTED,
            data={
                'symbol': 'AAPL',
                'timeframe': '1h',
                'start_date': start_date,
                'end_date': end_date
            }
        )
        
        # Publish request
        self.event_bus.publish(historical_request)
        
        # Wait for response
        logger.info("Waiting for historical data response...")
        time.sleep(2)
        
        # Check if historical data was received
        historical_events = [e for e in self.tracker.events 
                           if e.event_type == EventType.HISTORICAL_DATA_RECEIVED]
        
        self.assertTrue(len(historical_events) > 0, 
                       "No historical data received")
        
        if historical_events:
            # Verify the first historical data event
            hist_data = historical_events[0].data
            self.assertEqual(hist_data.get('symbol'), 'AAPL')
            self.assertEqual(hist_data.get('timeframe'), '1h')
            
            # Verify data structure
            self.assertIn('data', hist_data)
            if 'data' in hist_data and isinstance(hist_data['data'], list):
                logger.info(f"Received {len(hist_data['data'])} historical bars")
    
    def test_3_reconnection_handling(self):
        """Test data source reconnection handling"""
        # Create data source with reconnection settings
        data_source = LiveDataSource(
            persistence_manager=self.persistence,
            provider="simulation",
            symbols=["AAPL"],
            timeframes=["1m"],
            event_bus=self.event_bus,
            reconnect_attempts=3,
            reconnect_delay=1
        )
        
        # Start data source
        data_source.start()
        
        # Force disconnect simulation
        if hasattr(data_source, 'disconnect'):
            data_source.disconnect()
        elif hasattr(data_source, '_data_provider') and hasattr(data_source._data_provider, 'disconnect'):
            data_source._data_provider.disconnect()
        else:
            logger.warning("No disconnect method found, simulating by stopping/starting")
            data_source.stop()
            data_source.start()
        
        # Wait for reconnection
        logger.info("Waiting for reconnection...")
        time.sleep(3)
        
        # Check for events after reconnection
        initial_event_count = len(self.tracker.events)
        time.sleep(2)
        
        # Validate that new events are coming in after reconnection
        self.assertTrue(len(self.tracker.events) > initial_event_count,
                       "No new events after reconnection attempt")
        
        # Stop data source
        data_source.stop()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        cls.event_bus.clear_subscribers()


if __name__ == '__main__':
    unittest.main()
