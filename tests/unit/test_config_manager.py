"""
Unit tests for the configuration manager.

Tests loading, validation, and environment variable overrides
for the config_manager module.
"""

import os
import json
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

from trading_bot.core.config_manager import (
    load_config,
    ConfigModel,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)

class TestConfigManager(unittest.TestCase):
    """Test cases for the configuration manager"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Valid minimal config
        self.valid_config = {
            "enable_market_regime_system": True,
            "market_regime_config_path": "config/market_regime_config.json",
            "watched_symbols": ["AAPL", "MSFT"],
            "data_dir": "data",
            "trading_hours": {
                "start": "09:30",
                "end": "16:00",
                "timezone": "America/New_York"
            },
            "initial_capital": 10000,
            "risk_per_trade": 0.02,
            "max_open_positions": 5
        }
        
        # Create a valid config file
        self.valid_config_path = Path(self.temp_dir.name) / "valid_config.json"
        with open(self.valid_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Invalid config (missing required field)
        self.invalid_config = self.valid_config.copy()
        del self.invalid_config["risk_per_trade"]
        
        # Create an invalid config file
        self.invalid_config_path = Path(self.temp_dir.name) / "invalid_config.json"
        with open(self.invalid_config_path, 'w') as f:
            json.dump(self.invalid_config, f)
        
        # Invalid JSON file
        self.bad_json_path = Path(self.temp_dir.name) / "bad_json.json"
        with open(self.bad_json_path, 'w') as f:
            f.write("{This is not valid JSON}")
        
        # Nonexistent file
        self.nonexistent_path = Path(self.temp_dir.name) / "nonexistent.json"
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file"""
        config = load_config(str(self.valid_config_path))
        
        # Check that config is a ConfigModel
        self.assertIsInstance(config, ConfigModel)
        
        # Check some values
        self.assertEqual(config.initial_capital, 10000)
        self.assertEqual(config.risk_per_trade, 0.02)
        self.assertEqual(config.max_open_positions, 5)
        self.assertEqual(config.watched_symbols, ["AAPL", "MSFT"])
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file"""
        with self.assertRaises(ConfigFileNotFoundError):
            load_config(str(self.nonexistent_path))
    
    def test_load_invalid_json(self):
        """Test loading an invalid JSON file"""
        with self.assertRaises(ConfigParseError):
            load_config(str(self.bad_json_path))
    
    def test_load_invalid_config(self):
        """Test loading a config that fails validation"""
        with self.assertRaises(ConfigValidationError):
            load_config(str(self.invalid_config_path))
    
    @patch.dict(os.environ, {"BENBOT_INITIAL_CAPITAL": "20000"})
    def test_env_var_override_simple(self):
        """Test environment variable override for a simple value"""
        config = load_config(str(self.valid_config_path))
        
        # Check that the value was overridden
        self.assertEqual(config.initial_capital, 20000)
    
    @patch.dict(os.environ, {
        "BENBOT_RISK_PER_TRADE": "0.05",
        "BENBOT_MAX_OPEN_POSITIONS": "10",
        "BENBOT_TRADING_HOURS_START": "10:00"
    })
    def test_env_var_override_multiple(self):
        """Test multiple environment variable overrides"""
        config = load_config(str(self.valid_config_path))
        
        # Check that values were overridden
        self.assertEqual(config.risk_per_trade, 0.05)
        self.assertEqual(config.max_open_positions, 10)
        self.assertEqual(config.trading_hours.start, "10:00")
    
    @patch.dict(os.environ, {"BENBOT_WATCHED_SYMBOLS": "SPY,QQQ,AAPL"})
    def test_env_var_override_list(self):
        """Test environment variable override for a list value"""
        config = load_config(str(self.valid_config_path))
        
        # Check that the list was overridden and split correctly
        self.assertEqual(config.watched_symbols, ["SPY", "QQQ", "AAPL"])
    
    @patch.dict(os.environ, {"BENBOT_LOG_LEVEL": "DEBUG"})
    def test_env_var_override_enum(self):
        """Test environment variable override for an enum value"""
        config = load_config(str(self.valid_config_path))
        
        # Check that the enum was overridden
        self.assertEqual(config.log_level, "DEBUG")
    
    def test_validation_error_message(self):
        """Test that validation errors provide clear messages"""
        # Create a config with an invalid value
        bad_value_config = self.valid_config.copy()
        bad_value_config["risk_per_trade"] = 2.0  # Too high, should be < 1
        
        bad_value_path = Path(self.temp_dir.name) / "bad_value.json"
        with open(bad_value_path, 'w') as f:
            json.dump(bad_value_config, f)
        
        # Attempt to load and check for helpful error message
        try:
            load_config(str(bad_value_path))
            self.fail("Should have raised ConfigValidationError")
        except ConfigValidationError as e:
            error_msg = str(e)
            self.assertIn("risk_per_trade", error_msg)
            self.assertIn("2.0", error_msg)


# Run the tests if this file is executed directly
if __name__ == "__main__":
    unittest.main()
