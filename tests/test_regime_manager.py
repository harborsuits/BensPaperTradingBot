"""
Unit tests for Market Regime Manager
"""

import unittest
from unittest.mock import MagicMock, patch, call
import tempfile
import os
import json
from datetime import datetime, timedelta

from trading_bot.analytics.market_regime.integration import MarketRegimeManager
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.core.event_bus import EventBus, Event

class TestMarketRegimeManager(unittest.TestCase):
    """Test cases for MarketRegimeManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mocks
        self.event_bus = MagicMock(spec=EventBus)
        self.broker_manager = MagicMock()
        self.trade_accounting = MagicMock()
        self.capital_allocator = MagicMock()
        
        # Config
        self.config = {
            "monitoring_interval_seconds": 1,
            "update_interval_seconds": 10,
            "primary_timeframe": "1d",
            "detector": {
                "min_data_points": 10,
                "auto_update": False  # Disable auto-update for testing
            }
        }
        
        # Create detector patcher
        self.detector_patcher = patch('trading_bot.analytics.market_regime.integration.MarketRegimeDetector')
        self.parameter_optimizer_patcher = patch('trading_bot.analytics.market_regime.integration.ParameterOptimizer')
        self.performance_tracker_patcher = patch('trading_bot.analytics.market_regime.integration.RegimePerformanceTracker')
        self.strategy_selector_patcher = patch('trading_bot.analytics.market_regime.integration.StrategySelector')
        
        # Start patchers
        self.mock_detector = self.detector_patcher.start()
        self.mock_parameter_optimizer = self.parameter_optimizer_patcher.start()
        self.mock_performance_tracker = self.performance_tracker_patcher.start()
        self.mock_strategy_selector = self.strategy_selector_patcher.start()
        
        # Set up mock detector instance
        self.detector_instance = MagicMock()
        self.mock_detector.return_value = self.detector_instance
        
        # Set up mock parameter optimizer instance
        self.parameter_optimizer_instance = MagicMock()
        self.mock_parameter_optimizer.return_value = self.parameter_optimizer_instance
        
        # Set up mock performance tracker instance
        self.performance_tracker_instance = MagicMock()
        self.mock_performance_tracker.return_value = self.performance_tracker_instance
        
        # Set up mock strategy selector instance
        self.strategy_selector_instance = MagicMock()
        self.mock_strategy_selector.return_value = self.strategy_selector_instance
        
        # Create manager
        self.manager = MarketRegimeManager(
            self.event_bus, 
            self.broker_manager, 
            self.trade_accounting, 
            self.capital_allocator, 
            self.config
        )
        
        # Set up test data
        self.create_test_data()
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Stop patchers
        self.detector_patcher.stop()
        self.parameter_optimizer_patcher.stop()
        self.performance_tracker_patcher.stop()
        self.strategy_selector_patcher.stop()
    
    def create_test_data(self):
        """Create test data"""
        self.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Create regime data
        self.test_regimes = {
            "AAPL": {
                "1d": {"regime": MarketRegimeType.TRENDING_UP, "confidence": 0.9},
                "4h": {"regime": MarketRegimeType.RANGE_BOUND, "confidence": 0.7},
                "1h": {"regime": MarketRegimeType.NORMAL, "confidence": 0.8}
            },
            "MSFT": {
                "1d": {"regime": MarketRegimeType.TRENDING_DOWN, "confidence": 0.85},
                "4h": {"regime": MarketRegimeType.TRENDING_DOWN, "confidence": 0.75},
                "1h": {"regime": MarketRegimeType.VOLATILE, "confidence": 0.6}
            },
            "GOOGL": {
                "1d": {"regime": MarketRegimeType.RANGE_BOUND, "confidence": 0.8},
                "4h": {"regime": MarketRegimeType.VOLATILE, "confidence": 0.65},
                "1h": {"regime": MarketRegimeType.NORMAL, "confidence": 0.7}
            }
        }
        
        # Create selected strategies data
        self.test_selected_strategies = {
            "AAPL": [
                {"strategy_id": "trend_follow", "score": 0.8, "weight": 0.6},
                {"strategy_id": "momentum", "score": 0.6, "weight": 0.4}
            ],
            "MSFT": [
                {"strategy_id": "counter_trend", "score": 0.75, "weight": 0.7},
                {"strategy_id": "mean_reversion", "score": 0.5, "weight": 0.3}
            ],
            "GOOGL": [
                {"strategy_id": "range_breakout", "score": 0.7, "weight": 0.5},
                {"strategy_id": "volatility_capture", "score": 0.6, "weight": 0.5}
            ]
        }
        
        # Create test trade data
        self.test_trade = {
            "trade_id": "12345",
            "strategy_id": "trend_follow",
            "symbol": "AAPL",
            "entry_price": 150.0,
            "exit_price": 160.0,
            "realized_pnl": 1000.0,
            "return_pct": 6.67,
            "entry_time": datetime.now() - timedelta(days=1),
            "exit_time": datetime.now(),
            "position_size": 100,
            "side": "long"
        }
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.manager.config, self.config)
        self.assertEqual(self.manager.event_bus, self.event_bus)
        self.assertEqual(self.manager.broker_manager, self.broker_manager)
        self.assertEqual(self.manager.trade_accounting, self.trade_accounting)
        self.assertEqual(self.manager.capital_allocator, self.capital_allocator)
        
        # Verify components were created
        self.mock_detector.assert_called_once()
        self.mock_parameter_optimizer.assert_called_once()
        self.mock_performance_tracker.assert_called_once()
        self.mock_strategy_selector.assert_called_once()
    
    def test_register_event_handlers(self):
        """Test event handler registration"""
        # Verify event handlers were registered
        self.event_bus.register.assert_any_call("market_regime_change", self.manager._handle_regime_change)
        self.event_bus.register.assert_any_call("trade_closed", self.manager._handle_trade_closed)
        self.event_bus.register.assert_any_call("price_update", self.manager._handle_price_update)
    
    def test_initialize(self):
        """Test manager initialization"""
        # Setup detector
        self.detector_instance.add_symbol = MagicMock()
        
        # Initialize with symbols
        result = self.manager.initialize(self.test_symbols)
        
        # Verify result
        self.assertTrue(result)
        self.assertTrue(self.manager.initialized)
        
        # Verify symbols were added to tracked symbols
        for symbol in self.test_symbols:
            self.assertIn(symbol, self.manager.tracked_symbols)
        
        # Verify symbols were added to detector
        for symbol in self.test_symbols:
            self.detector_instance.add_symbol.assert_any_call(symbol)
    
    @patch('threading.Thread')
    def test_start_monitoring(self, mock_thread):
        """Test starting the monitoring thread"""
        # Call _start_monitoring
        self.manager._start_monitoring()
        
        # Verify thread was started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        
        # Verify monitoring state
        self.assertTrue(self.manager.monitoring_active)
    
    def test_log_regime_transition(self):
        """Test logging regime transitions"""
        symbol = "AAPL"
        timeframe = "1d"
        old_regime = MarketRegimeType.NORMAL
        new_regime = MarketRegimeType.TRENDING_UP
        confidence = 0.9
        
        # Log transition
        self.manager._log_regime_transition(symbol, timeframe, old_regime, new_regime, confidence)
        
        # Verify history was updated
        self.assertIn(symbol, self.manager.regime_history)
        self.assertIn(timeframe, self.manager.regime_history[symbol])
        self.assertEqual(len(self.manager.regime_history[symbol][timeframe]), 1)
        
        # Verify transition record
        transition = self.manager.regime_history[symbol][timeframe][0]
        self.assertEqual(transition["old_regime"], old_regime.value)
        self.assertEqual(transition["new_regime"], new_regime.value)
        self.assertEqual(transition["confidence"], confidence)
    
    def test_handle_regime_change(self):
        """Test handling regime change events"""
        # Create event data
        event_data = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "new_regime": MarketRegimeType.TRENDING_UP.value,
            "confidence": 0.9
        }
        
        # Create event
        event = MagicMock(spec=Event)
        event.data = event_data
        
        # Mock _update_symbol_strategies
        self.manager._update_symbol_strategies = MagicMock()
        
        # Handle event
        self.manager._handle_regime_change(event)
        
        # Verify active regimes were updated
        self.assertIn("AAPL", self.manager.active_regimes)
        self.assertIn("1d", self.manager.active_regimes["AAPL"])
        self.assertEqual(self.manager.active_regimes["AAPL"]["1d"], MarketRegimeType.TRENDING_UP)
        
        # Verify strategies were updated (since 1d is the primary timeframe)
        self.manager._update_symbol_strategies.assert_called_once_with("AAPL")
    
    def test_handle_trade_closed(self):
        """Test handling trade closed events"""
        # Setup active regimes
        self.manager.active_regimes = {
            "AAPL": {"1d": MarketRegimeType.TRENDING_UP}
        }
        
        # Create event data
        event_data = {
            "trade": self.test_trade
        }
        
        # Create event
        event = MagicMock(spec=Event)
        event.data = event_data
        
        # Handle event
        self.manager._handle_trade_closed(event)
        
        # Verify performance tracker was updated
        self.performance_tracker_instance.update_performance.assert_called_once()
        
        # Check arguments
        args, kwargs = self.performance_tracker_instance.update_performance.call_args
        self.assertEqual(args[0], "trend_follow")  # strategy_id
        self.assertEqual(args[1], MarketRegimeType.TRENDING_UP)  # regime_type
        self.assertEqual(args[3], "AAPL")  # symbol
        self.assertEqual(args[4], "1d")  # timeframe
        
        # Metrics should include profit_loss, win, win_rate, profit_factor, and returns
        metrics = args[2]
        self.assertEqual(metrics["profit_loss"], 1000.0)
        self.assertEqual(metrics["win"], 1.0)  # It's a winning trade
        self.assertIn("returns", metrics)
    
    def test_update_symbol_strategies(self):
        """Test updating symbol strategies"""
        symbol = "AAPL"
        
        # Setup detector to return regimes
        self.detector_instance.get_current_regimes.return_value = self.test_regimes["AAPL"]
        
        # Setup strategy selector
        self.strategy_selector_instance.get_preferred_timeframe.return_value = "1h"
        self.strategy_selector_instance.select_strategies.return_value = self.test_selected_strategies["AAPL"]
        
        # Update strategies
        self.manager._update_symbol_strategies(symbol)
        
        # Verify detector was called
        self.detector_instance.get_current_regimes.assert_called_once_with(symbol)
        
        # Verify preferred timeframe was retrieved
        self.strategy_selector_instance.get_preferred_timeframe.assert_called_once_with(
            symbol, MarketRegimeType.TRENDING_UP, default_timeframe="1d"
        )
        
        # Verify strategies were selected
        self.strategy_selector_instance.select_strategies.assert_called_once_with(
            symbol, MarketRegimeType.TRENDING_UP, "1h"
        )
        
        # Verify event was emitted
        self.event_bus.emit.assert_called_once_with(
            "strategy_selection_update", 
            {
                'symbol': symbol,
                'regime_type': MarketRegimeType.TRENDING_UP.value,
                'timeframe': "1h",
                'strategies': self.test_selected_strategies["AAPL"],
                'timestamp': unittest.mock.ANY
            }
        )
    
    def test_update_capital_allocation(self):
        """Test updating capital allocation"""
        # Setup active strategies and weights
        self.strategy_selector_instance.get_active_strategies.side_effect = lambda symbol: {
            "AAPL": ["trend_follow", "momentum"],
            "MSFT": ["counter_trend", "mean_reversion"],
            "GOOGL": ["range_breakout", "volatility_capture"]
        }[symbol]
        
        self.strategy_selector_instance.get_strategy_weight.side_effect = lambda symbol, strategy_id: {
            ("AAPL", "trend_follow"): 0.6,
            ("AAPL", "momentum"): 0.4,
            ("MSFT", "counter_trend"): 0.7,
            ("MSFT", "mean_reversion"): 0.3,
            ("GOOGL", "range_breakout"): 0.5,
            ("GOOGL", "volatility_capture"): 0.5
        }[(symbol, strategy_id)]
        
        # Add symbols to tracked symbols
        self.manager.tracked_symbols = set(self.test_symbols)
        
        # Update allocation
        self.manager._update_capital_allocation()
        
        # Verify capital allocator was called
        self.capital_allocator.update_regime_allocations.assert_called_once()
        
        # Check allocations
        allocations = self.capital_allocator.update_regime_allocations.call_args[0][0]
        
        # Should have entries for all symbols
        for symbol in self.test_symbols:
            self.assertIn(symbol, allocations)
            self.assertIn("strategies", allocations[symbol])
            self.assertEqual(len(allocations[symbol]["strategies"]), 2)  # Each symbol has 2 strategies
    
    def test_get_parameter_set(self):
        """Test getting optimized parameters"""
        symbol = "AAPL"
        strategy_id = "trend_follow"
        timeframe = "1d"
        
        # Setup detector to return regimes
        self.detector_instance.get_current_regimes.return_value = self.test_regimes["AAPL"]
        
        # Setup parameter optimizer
        test_params = {"stop_loss": 2.0, "take_profit": 4.0, "entry_threshold": 0.5}
        self.parameter_optimizer_instance.get_optimal_parameters.return_value = test_params
        
        # Get parameters
        result = self.manager.get_parameter_set(strategy_id, symbol, timeframe)
        
        # Verify result
        self.assertEqual(result, test_params)
        
        # Verify parameter optimizer was called
        self.parameter_optimizer_instance.get_optimal_parameters.assert_called_once_with(
            strategy_id, MarketRegimeType.TRENDING_UP, symbol, timeframe, 0.9
        )
    
    def test_get_regime_performance_summary(self):
        """Test getting regime performance summary"""
        # Setup tracked symbols and active regimes
        self.manager.tracked_symbols = set(self.test_symbols)
        self.manager.active_regimes = {
            "AAPL": {"1d": MarketRegimeType.TRENDING_UP},
            "MSFT": {"1d": MarketRegimeType.TRENDING_DOWN},
            "GOOGL": {"1d": MarketRegimeType.RANGE_BOUND}
        }
        
        # Setup strategy selector
        self.strategy_selector_instance.get_active_strategies.side_effect = lambda symbol: {
            "AAPL": ["trend_follow", "momentum"],
            "MSFT": ["counter_trend", "mean_reversion"],
            "GOOGL": ["range_breakout", "volatility_capture"]
        }[symbol]
        
        # Setup performance tracker
        self.performance_tracker_instance.get_best_strategies_for_regime.return_value = [
            ("trend_follow", 0.8),
            ("momentum", 0.7),
            ("volatility_capture", 0.6)
        ]
        
        self.performance_tracker_instance.get_regime_performance_summary.return_value = {
            MarketRegimeType.TRENDING_UP: {"sample_size": 20, "metrics": {}, "drift": {"drift_detected": False}}
        }
        
        # Get summary
        result = self.manager.get_regime_performance_summary()
        
        # Verify result structure
        self.assertIn("symbols", result)
        self.assertIn("regimes", result)
        self.assertIn("strategies", result)
        
        # Verify symbols section
        for symbol in self.test_symbols:
            self.assertIn(symbol, result["symbols"])
            self.assertIn("regimes", result["symbols"][symbol])
            self.assertIn("active_strategies", result["symbols"][symbol])
        
        # Verify regimes section
        for regime in [MarketRegimeType.TRENDING_UP.value, MarketRegimeType.TRENDING_DOWN.value, MarketRegimeType.RANGE_BOUND.value]:
            self.assertIn(regime, result["regimes"])
            self.assertIn("symbols_count", result["regimes"][regime])
            self.assertIn("top_strategies", result["regimes"][regime])
    
    def test_shutdown(self):
        """Test manager shutdown"""
        # Setup
        self.manager._stop_monitoring = MagicMock()
        
        # Call shutdown
        self.manager.shutdown()
        
        # Verify monitoring was stopped
        self.manager._stop_monitoring.assert_called_once()
        
        # Verify detector was shutdown
        self.detector_instance.shutdown.assert_called_once()

if __name__ == '__main__':
    unittest.main()
