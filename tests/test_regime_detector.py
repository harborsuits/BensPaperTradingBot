"""
Unit tests for Market Regime Detector
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.analytics.market_regime.detector import MarketRegimeDetector, MarketRegimeType
from trading_bot.core.event_bus import EventBus

class TestMarketRegimeDetector(unittest.TestCase):
    """Test cases for MarketRegimeDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.event_bus = MagicMock(spec=EventBus)
        self.config = {
            "detection_interval_seconds": 1,
            "min_data_points": 50,
            "emit_events": True,
            "default_timeframes": ["1d", "4h", "1h"],
            "regime_history_limit": 10
        }
        
        # Create test data
        self.create_test_data()
        
        # Mock broker manager in detector
        self.broker_manager_patcher = patch('trading_bot.analytics.market_regime.detector.MultiBrokerManager')
        self.mock_broker_manager = self.broker_manager_patcher.start()
        
        # Create detector
        self.detector = MarketRegimeDetector(self.event_bus, self.config)
        
        # Mock the broker manager instance in the detector
        self.detector.broker_manager = MagicMock()
        
        # Mock detector methods
        self.detector._get_ohlcv_data = MagicMock(return_value=self.test_data)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.broker_manager_patcher.stop()
    
    def create_test_data(self):
        """Create test OHLCV data"""
        # Create a dataframe with 100 rows of data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # Sort by date ascending
        self.test_data.sort_index(inplace=True)
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.detector.config, self.config)
        self.assertEqual(self.detector.event_bus, self.event_bus)
        self.assertEqual(len(self.detector.tracked_symbols), 0)
    
    def test_add_symbol(self):
        """Test adding a symbol to track"""
        # Test adding a symbol
        self.detector.add_symbol("AAPL")
        self.assertIn("AAPL", self.detector.tracked_symbols)
        
        # Test adding timeframes
        self.detector.add_symbol("MSFT", timeframes=["1h", "5m"])
        self.assertIn("MSFT", self.detector.tracked_symbols)
        self.assertEqual(self.detector.symbol_timeframes["MSFT"], ["1h", "5m"])
    
    def test_remove_symbol(self):
        """Test removing a symbol"""
        # Add a symbol
        self.detector.add_symbol("AAPL")
        self.assertIn("AAPL", self.detector.tracked_symbols)
        
        # Remove it
        self.detector.remove_symbol("AAPL")
        self.assertNotIn("AAPL", self.detector.tracked_symbols)
    
    def test_detect_regime(self):
        """Test regime detection"""
        # Add a symbol
        self.detector.add_symbol("AAPL")
        
        # Mock features calculator and classifier
        self.detector.features_calculator.calculate_features = MagicMock(
            return_value={"trend_strength": 0.8, "volatility": 0.2}
        )
        self.detector.classifier.classify_regime = MagicMock(
            return_value=(MarketRegimeType.TRENDING_UP, 0.9)
        )
        
        # Detect regime
        regime_info = self.detector._detect_regime("AAPL", "1d")
        
        # Verify results
        self.assertEqual(regime_info["regime"], MarketRegimeType.TRENDING_UP)
        self.assertEqual(regime_info["confidence"], 0.9)
        
        # Verify event was emitted
        self.event_bus.emit.assert_called_with("market_regime_change", {
            'symbol': "AAPL", 
            'timeframe': "1d",
            'new_regime': MarketRegimeType.TRENDING_UP,
            'confidence': 0.9,
            'features': {"trend_strength": 0.8, "volatility": 0.2}
        })
    
    def test_get_current_regimes(self):
        """Test getting current regimes"""
        # Add a symbol
        self.detector.add_symbol("AAPL")
        
        # Setup test data
        self.detector.current_regimes["AAPL"] = {
            "1d": {"regime": MarketRegimeType.TRENDING_UP, "confidence": 0.9},
            "4h": {"regime": MarketRegimeType.RANGE_BOUND, "confidence": 0.7}
        }
        
        # Get current regimes
        regimes = self.detector.get_current_regimes("AAPL")
        
        # Verify results
        self.assertEqual(len(regimes), 2)
        self.assertEqual(regimes["1d"]["regime"], MarketRegimeType.TRENDING_UP)
        self.assertEqual(regimes["4h"]["regime"], MarketRegimeType.RANGE_BOUND)
    
    def test_get_regime_history(self):
        """Test getting regime history"""
        # Add a symbol
        self.detector.add_symbol("AAPL")
        
        # Setup test data
        self.detector.regime_history["AAPL"] = {
            "1d": [
                {"timestamp": datetime.now(), "regime": MarketRegimeType.TRENDING_UP, "confidence": 0.9},
                {"timestamp": datetime.now() - timedelta(days=1), "regime": MarketRegimeType.NORMAL, "confidence": 0.5}
            ]
        }
        
        # Get regime history
        history = self.detector.get_regime_history("AAPL", "1d")
        
        # Verify results
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["regime"], MarketRegimeType.TRENDING_UP)
        self.assertEqual(history[1]["regime"], MarketRegimeType.NORMAL)
    
    def test_detect_trend_direction(self):
        """Test trend direction detection"""
        # Create uptrend and downtrend data
        uptrend_data = self.test_data.copy()
        uptrend_data['close'] = np.linspace(90, 110, 100)  # Linearly increasing
        
        downtrend_data = self.test_data.copy()
        downtrend_data['close'] = np.linspace(110, 90, 100)  # Linearly decreasing
        
        # Test uptrend
        self.detector._get_ohlcv_data = MagicMock(return_value=uptrend_data)
        self.detector.features_calculator.calculate_features = MagicMock(
            return_value={"trend_strength": 0.8, "volatility": 0.2}
        )
        self.detector.classifier.classify_regime = MagicMock()
        
        self.detector._detect_regime("AAPL", "1d")
        
        # Verify classifier was called with uptrend features
        self.detector.classifier.classify_regime.assert_called_once()
        args, _ = self.detector.classifier.classify_regime.call_args
        self.assertTrue(args[0]["trend_strength"] > 0)
        
        # Reset mocks
        self.detector.classifier.classify_regime.reset_mock()
        
        # Test downtrend
        self.detector._get_ohlcv_data = MagicMock(return_value=downtrend_data)
        self.detector._detect_regime("AAPL", "1d")
        
        # Verify classifier was called with downtrend features
        self.detector.classifier.classify_regime.assert_called_once()
        args, _ = self.detector.classifier.classify_regime.call_args
        self.assertTrue(args[0]["trend_strength"] < 0)
    
    def test_auto_update_disabled(self):
        """Test auto update can be disabled"""
        # Create detector with auto_update set to False
        config = self.config.copy()
        config["auto_update"] = False
        detector = MarketRegimeDetector(self.event_bus, config)
        
        # Ensure _monitoring_thread is None
        self.assertIsNone(detector._monitoring_thread)
    
    @patch('threading.Thread')
    def test_auto_update_thread_start(self, mock_thread):
        """Test auto update thread is started"""
        # Create detector with auto_update set to True
        config = self.config.copy()
        config["auto_update"] = True
        
        # Create detector
        detector = MarketRegimeDetector(self.event_bus, config)
        
        # Verify thread was started
        mock_thread.assert_called_once()
    
    def test_shutdown(self):
        """Test shutdown"""
        # Create a stop event mock
        self.detector._stop_event = MagicMock()
        
        # Call shutdown
        self.detector.shutdown()
        
        # Verify stop event was set
        self.detector._stop_event.set.assert_called_once()

if __name__ == '__main__':
    unittest.main()
