"""
Unit tests for Market Regime System Integration
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus
from trading_bot.analytics.market_regime.bootstrap import (
    initialize_regime_system_with_defaults, DEFAULT_CONFIG
)
from trading_bot.strategy.regime_strategy_adapter import RegimeStrategyAdapter
from trading_bot.analytics.market_regime.detector import MarketRegimeType

class TestRegimeSystemIntegration(unittest.TestCase):
    """Test cases for Market Regime System integration with BensBot"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mocks
        self.event_bus = MagicMock(spec=EventBus)
        self.broker_manager = MagicMock()
        self.trade_accounting = MagicMock()
        self.capital_allocator = MagicMock()
        
        # Create mock main system
        self.main_system = MagicMock()
        self.main_system.event_bus = self.event_bus
        self.main_system.broker_manager = self.broker_manager
        self.main_system.trade_accounting = self.trade_accounting
        self.main_system.capital_allocator = self.capital_allocator
        
        # Create temp directory for configs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create minimal config
        self.config = DEFAULT_CONFIG.copy()
        self.config["watched_symbols"] = ["AAPL", "MSFT"]
        self.config["detector"]["auto_update"] = False  # Disable auto-update for testing
        
        # Save config to file
        self.config_path = os.path.join(self.temp_dir.name, "regime_config.json")
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        
        # Patch initialization functions
        self.initialize_patcher = patch('trading_bot.analytics.market_regime.bootstrap.initialize_regime_system')
        self.mock_initialize = self.initialize_patcher.start()
        
        # Create mock regime manager
        self.regime_manager = MagicMock()
        self.mock_initialize.return_value = self.regime_manager
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.initialize_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_initialize_with_defaults(self):
        """Test initialization with default settings"""
        # Call initialize
        manager = initialize_regime_system_with_defaults(
            self.event_bus, 
            self.broker_manager, 
            self.trade_accounting, 
            self.capital_allocator,
            self.config_path
        )
        
        # Verify initialize_regime_system was called
        self.mock_initialize.assert_called_once()
        
        # Verify parameters were registered
        self.regime_manager.parameter_optimizer.update_optimal_parameters.assert_called()
        
        # Verify strategy configs were registered
        self.regime_manager.strategy_selector.register_strategy.assert_called()
        
        # Verify timeframe mappings were set
        self.regime_manager.strategy_selector.set_timeframe_mapping.assert_called()
    
    def test_strategy_adapter_integration(self):
        """Test strategy adapter integration"""
        # Create strategy adapter
        adapter = RegimeStrategyAdapter(self.event_bus, self.regime_manager)
        
        # Create mock strategy
        strategy = MagicMock()
        strategy.symbol = "AAPL"
        strategy.timeframe = "1d"
        
        # Mock parameters
        params = {
            "stop_loss": 2.0,
            "take_profit": 4.0,
            "entry_threshold": 0.5
        }
        
        # Register strategy
        adapter.register_strategy("test_strategy", strategy, params)
        
        # Verify strategy was registered
        self.assertIn("test_strategy", adapter.adapted_strategies)
        self.assertEqual(adapter.adapted_strategies["test_strategy"]["instance"], strategy)
        
        # Verify original parameters were stored
        self.assertIn("test_strategy", adapter.original_parameters)
        self.assertEqual(adapter.original_parameters["test_strategy"]["stop_loss"], 2.0)
        
        # Test regime change event
        # Set up regime manager to return regime info
        regime_info = {
            "1d": {
                "regime": MarketRegimeType.TRENDING_UP,
                "confidence": 0.9
            }
        }
        self.regime_manager.detector.get_current_regimes.return_value = regime_info
        
        # Set up regime manager to return parameters
        optimized_params = {
            "stop_loss": 3.0,  # Different from original
            "take_profit": 6.0,
            "entry_threshold": 0.4
        }
        self.regime_manager.get_parameter_set.return_value = optimized_params
        
        # Create regime change event
        event = MagicMock()
        event.data = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "new_regime": MarketRegimeType.TRENDING_UP.value,
            "confidence": 0.9
        }
        
        # Handle event
        adapter._handle_regime_change(event)
        
        # Verify parameters were updated
        strategy.update_parameters.assert_called_once_with(optimized_params)
        
        # Test reset
        adapter.reset_strategy_parameters("test_strategy")
        
        # Verify original parameters were restored
        strategy.update_parameters.assert_called_with(params)
    
    def test_system_integrations(self):
        """Test system integrations module"""
        # Import after mocking
        from trading_bot.core.system_integrations import initialize_integrations, shutdown_integrations
        
        # Create config
        config = {
            "enable_market_regime_system": True,
            "market_regime_config_path": self.config_path
        }
        
        # Patch setup_market_regime_system
        with patch('trading_bot.core.system_integrations.setup_market_regime_system') as mock_setup:
            mock_setup.return_value = self.regime_manager
            
            # Initialize integrations
            initialize_integrations(self.main_system, config)
            
            # Verify setup was called
            mock_setup.assert_called_once_with(self.main_system, self.config_path)
            
            # Verify manager was registered
            self.main_system.register_component.assert_called_once_with(
                "market_regime_manager", self.regime_manager
            )
        
        # Test shutdown
        self.main_system.market_regime_manager = self.regime_manager
        
        # Shutdown
        shutdown_integrations(self.main_system)
        
        # Verify manager was shutdown
        self.regime_manager.shutdown.assert_called_once()

if __name__ == '__main__':
    unittest.main()
