"""
Unit tests for the simple configuration manager.

Tests loading, validation, and environment variable overrides
for the simple_config module.
"""

import os
import json
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

from trading_bot.core.simple_config import (
    load_config,
    apply_env_overrides,
    validate_config,
    get_nested_value,
    parse_time,
    to_bool,
    to_int,
    to_float,
    to_list,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)

class TestSimpleConfig(unittest.TestCase):
    """Test cases for the simple configuration manager"""
    
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
            "max_open_positions": 5,
            "log_level": "INFO",
            "system_safeguards": {
                "circuit_breakers": {
                    "max_drawdown_percent": 10,
                    "max_daily_loss_percent": 3,
                    "consecutive_loss_count": 3
                }
            }
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
        
        # Check some values
        self.assertEqual(config["initial_capital"], 10000)
        self.assertEqual(config["risk_per_trade"], 0.02)
        self.assertEqual(config["max_open_positions"], 5)
        self.assertEqual(config["watched_symbols"], ["AAPL", "MSFT"])
    
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
    
    def test_validate_config(self):
        """Test configuration validation"""
        # Should not raise an exception
        validate_config(self.valid_config)
        
        # Test invalid enable_market_regime_system
        bad_config = self.valid_config.copy()
        bad_config["enable_market_regime_system"] = "not_a_boolean"
        with self.assertRaises(ConfigValidationError):
            validate_config(bad_config)
        
        # Test invalid risk_per_trade
        bad_config = self.valid_config.copy()
        bad_config["risk_per_trade"] = 2.0  # Must be < 1.0
        with self.assertRaises(ConfigValidationError):
            validate_config(bad_config)
        
        # Test invalid trading_hours
        bad_config = self.valid_config.copy()
        bad_config["trading_hours"]["start"] = "invalid_time"
        with self.assertRaises(ConfigValidationError):
            validate_config(bad_config)
    
    def test_get_nested_value(self):
        """Test getting nested values from a config dictionary"""
        config = self.valid_config
        
        # Test direct key
        self.assertEqual(get_nested_value(config, "initial_capital"), 10000)
        
        # Test nested key
        self.assertEqual(get_nested_value(config, "trading_hours.start"), "09:30")
        self.assertEqual(get_nested_value(config, "trading_hours.timezone"), "America/New_York")
        
        # Test deep nested key
        self.assertEqual(get_nested_value(config, "system_safeguards.circuit_breakers.max_drawdown_percent"), 10)
        
        # Test missing key with default
        self.assertEqual(get_nested_value(config, "nonexistent", "default"), "default")
        self.assertEqual(get_nested_value(config, "trading_hours.nonexistent", 123), 123)
    
    def test_parse_time(self):
        """Test parsing time strings to datetime.time objects"""
        t = parse_time("09:30")
        self.assertEqual(t.hour, 9)
        self.assertEqual(t.minute, 30)
        
        t = parse_time("23:45")
        self.assertEqual(t.hour, 23)
        self.assertEqual(t.minute, 45)
    
    def test_type_conversion_functions(self):
        """Test the type conversion helper functions"""
        # to_bool
        self.assertTrue(to_bool(True))
        self.assertTrue(to_bool("true"))
        self.assertTrue(to_bool("True"))
        self.assertTrue(to_bool("yes"))
        self.assertTrue(to_bool("1"))
        self.assertFalse(to_bool(False))
        self.assertFalse(to_bool("false"))
        self.assertFalse(to_bool("no"))
        self.assertFalse(to_bool("0"))
        
        # to_int
        self.assertEqual(to_int("42"), 42)
        self.assertEqual(to_int(42), 42)
        with self.assertRaises(ValueError):
            to_int("5", min_value=10)
        with self.assertRaises(ValueError):
            to_int("20", max_value=10)
        
        # to_float
        self.assertEqual(to_float("3.14"), 3.14)
        self.assertEqual(to_float(3.14), 3.14)
        with self.assertRaises(ValueError):
            to_float("5.0", min_value=10.0)
        with self.assertRaises(ValueError):
            to_float("20.0", max_value=10.0)
        
        # to_list
        self.assertEqual(to_list("a,b,c"), ["a", "b", "c"])
        self.assertEqual(to_list(["a", "b", "c"]), ["a", "b", "c"])
        self.assertEqual(to_list("1,2,3", item_type=int), [1, 2, 3])
    
    @patch.dict(os.environ, {"BENBOT_INITIAL_CAPITAL": "20000"})
    def test_env_var_override_simple(self):
        """Test environment variable override for a simple value"""
        config = self.valid_config.copy()
        apply_env_overrides(config)
        
        # Check that the value was overridden
        self.assertEqual(config["initial_capital"], 20000)
    
    @patch.dict(os.environ, {
        "BENBOT_RISK_PER_TRADE": "0.05",
        "BENBOT_MAX_OPEN_POSITIONS": "10",
        "BENBOT_TRADING_HOURS_START": "10:00"
    })
    def test_env_var_override_multiple(self):
        """Test multiple environment variable overrides"""
        config = self.valid_config.copy()
        apply_env_overrides(config)
        
        # Check that values were overridden
        self.assertEqual(config["risk_per_trade"], 0.05)
        self.assertEqual(config["max_open_positions"], 10)
        self.assertEqual(config["trading_hours"]["start"], "10:00")
    
    @patch.dict(os.environ, {"BENBOT_WATCHED_SYMBOLS": "SPY,QQQ,AAPL"})
    def test_env_var_override_list(self):
        """Test environment variable override for a list value"""
        config = self.valid_config.copy()
        apply_env_overrides(config)
        
        # Check that the list was overridden and split correctly
        self.assertEqual(config["watched_symbols"], ["SPY", "QQQ", "AAPL"])
    
    @patch.dict(os.environ, {"BENBOT_SYSTEM_SAFEGUARDS_CIRCUIT_BREAKERS_MAX_DRAWDOWN_PERCENT": "15"})
    def test_env_var_override_nested(self):
        """Test environment variable override for a nested value"""
        config = self.valid_config.copy()
        apply_env_overrides(config)
        
        # Check that the nested value was overridden
        self.assertEqual(config["system_safeguards"]["circuit_breakers"]["max_drawdown_percent"], 15.0)


# Run tests if this script is executed directly
if __name__ == "__main__":
    unittest.main()
