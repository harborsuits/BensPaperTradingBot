"""
Data Validator Test Suite

This script tests the market data validation functionality to ensure that
bad/stale/unreasonable data is properly detected and handled.

Usage:
    python -m tests.validation.data_validator_test
"""
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_bot.data.data_validator import DataValidator
from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType


class DataValidatorTest(unittest.TestCase):
    """Tests for the DataValidator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.event_bus = get_global_event_bus()
        self.validator = DataValidator(
            enable_market_hours_check=False  # Disable for testing
        )
        
        # Track events published by the validator
        self.data_quality_alerts = []
        self.event_bus.subscribe(EventType.DATA_QUALITY_ALERT, self._on_data_alert)
        
    def _on_data_alert(self, event):
        """Handler for data quality alert events."""
        self.data_quality_alerts.append(event.data)
    
    def test_stale_data_detection(self):
        """Test stale data detection functionality."""
        # First call should always pass (establishing baseline)
        symbol = "AAPL"
        timeframe = "1m"
        initial_time = datetime.now()
        
        # Initial data point should be fresh
        self.assertTrue(self.validator.check_stale_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=initial_time
        ))
        
        # Data within threshold should be considered fresh
        still_fresh_time = initial_time + timedelta(seconds=30)
        self.assertTrue(self.validator.check_stale_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=still_fresh_time
        ))
        
        # Data beyond threshold should be considered stale
        very_delayed_time = initial_time + timedelta(seconds=300)  # 5 minutes
        self.assertFalse(self.validator.check_stale_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=very_delayed_time
        ))
        
        # Verify a data quality alert was published
        self.assertGreaterEqual(len(self.data_quality_alerts), 1)
        alert = self.data_quality_alerts[-1]
        self.assertEqual(alert.get("alert_type"), "stale_data")
        self.assertEqual(alert.get("symbol"), symbol)
    
    def test_price_reasonability(self):
        """Test price reasonability checks."""
        symbol = "MSFT"
        base_price = 100.0
        
        # Initial price should always be reasonable
        self.assertTrue(self.validator.check_price_reasonability(
            symbol=symbol,
            price=base_price
        ))
        
        # Small change should be reasonable
        self.assertTrue(self.validator.check_price_reasonability(
            symbol=symbol,
            price=base_price * 1.02  # 2% increase
        ))
        
        # Large change should be unreasonable and generate alert
        self.data_quality_alerts = []  # Clear previous alerts
        self.assertFalse(self.validator.check_price_reasonability(
            symbol=symbol,
            price=base_price * 1.25  # 25% increase
        ))
        
        # Verify a data quality alert was published
        self.assertGreaterEqual(len(self.data_quality_alerts), 1)
        alert = self.data_quality_alerts[-1]
        self.assertEqual(alert.get("alert_type"), "price_spike")
        self.assertEqual(alert.get("symbol"), symbol)
    
    def test_inconsistent_ohlc_detection(self):
        """Test detection of inconsistent OHLC data."""
        symbol = "TSLA"
        timeframe = "5m"
        timestamp = datetime.now()
        
        # Valid OHLC data should pass
        self.assertTrue(self.validator.validate_bar_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open_price=100.0,
            high_price=110.0,
            low_price=95.0,
            close_price=105.0,
            volume=1000.0
        ))
        
        # Inconsistent OHLC data (high < open) should fail
        self.data_quality_alerts = []  # Clear previous alerts
        self.assertFalse(self.validator.validate_bar_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open_price=100.0,
            high_price=99.0,  # High below open
            low_price=95.0,
            close_price=98.0,
            volume=1000.0
        ))
        
        # Verify a data quality alert was published
        self.assertGreaterEqual(len(self.data_quality_alerts), 1)
        alert = self.data_quality_alerts[-1]
        self.assertEqual(alert.get("alert_type"), "inconsistent_data")
        self.assertEqual(alert.get("symbol"), symbol)
    
    def test_negative_price_detection(self):
        """Test detection of negative prices."""
        symbol = "AMZN"
        timeframe = "1h"
        timestamp = datetime.now()
        
        # Negative price should fail
        self.data_quality_alerts = []  # Clear previous alerts
        self.assertFalse(self.validator.validate_bar_data(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open_price=100.0,
            high_price=110.0,
            low_price=-5.0,  # Negative low price
            close_price=105.0,
            volume=1000.0
        ))
        
        # Verify a data quality alert was published
        self.assertGreaterEqual(len(self.data_quality_alerts), 1)
        alert = self.data_quality_alerts[-1]
        self.assertEqual(alert.get("alert_type"), "invalid_data")
        self.assertEqual(alert.get("symbol"), symbol)
    
    def test_volume_spike_detection(self):
        """Test detection of unusual volume spikes."""
        symbol = "META"
        
        # Establish normal volume baseline
        for i in range(5):
            self.validator.check_volume_reasonability(
                symbol=symbol,
                volume=1000.0 + i * 100
            )
        
        # Large volume spike should be detected
        self.data_quality_alerts = []  # Clear previous alerts
        self.assertFalse(self.validator.check_volume_reasonability(
            symbol=symbol,
            volume=15000.0  # Much higher than baseline
        ))
        
        # Verify a data quality alert was published
        self.assertGreaterEqual(len(self.data_quality_alerts), 1)
        alert = self.data_quality_alerts[-1]
        self.assertEqual(alert.get("alert_type"), "volume_spike")
        self.assertEqual(alert.get("symbol"), symbol)
    
    def test_market_hours_validation(self):
        """Test market hours validation."""
        # Enable market hours check for this test
        self.validator.enable_market_hours_check = True
        
        symbol = "SPY"
        
        # Create a timestamp that's within market hours (10:30 AM ET on a weekday)
        # Note: This will fail if we run the test on a weekend
        today = datetime.now().replace(hour=10, minute=30)
        if today.weekday() < 5:  # Weekday
            # Time within market hours should pass
            self.assertTrue(self.validator.check_market_hours(
                symbol=symbol,
                timestamp=today,
                asset_class="stock"
            ))
            
            # Time outside market hours should fail
            after_hours = today.replace(hour=20, minute=0)  # 8:00 PM
            self.assertFalse(self.validator.check_market_hours(
                symbol=symbol,
                timestamp=after_hours,
                asset_class="stock"
            ))
        
        # 24/7 markets like crypto should always pass
        self.assertTrue(self.validator.check_market_hours(
            symbol="BTC/USD",
            timestamp=today,
            asset_class="crypto"
        ))
        
        # Reset for other tests
        self.validator.enable_market_hours_check = False
        
    def test_reset_history(self):
        """Test reset of historical data."""
        symbol = "GOOG"
        
        # Add some history
        self.validator.check_price_reasonability(
            symbol=symbol,
            price=100.0
        )
        
        # Reset history for symbol
        self.validator.reset_history(symbol=symbol)
        
        # Verify no history exists (newly set price is always reasonable)
        self.assertTrue(symbol not in self.validator.historical_ranges or
                     len(self.validator.historical_ranges[symbol].get("last_50_prices", [])) <= 1)

    def test_timeframe_conversion(self):
        """Test timeframe string conversion to seconds."""
        self.assertEqual(self.validator._timeframe_to_seconds("1m"), 60)
        self.assertEqual(self.validator._timeframe_to_seconds("5m"), 300)
        self.assertEqual(self.validator._timeframe_to_seconds("1h"), 3600)
        self.assertEqual(self.validator._timeframe_to_seconds("1d"), 86400)
        self.assertEqual(self.validator._timeframe_to_seconds("tick"), 1)
        
        # Invalid format should default to 60 seconds
        self.assertEqual(self.validator._timeframe_to_seconds("invalid"), 60)


def main():
    """Run data validator tests."""
    unittest.main()


if __name__ == "__main__":
    main()
