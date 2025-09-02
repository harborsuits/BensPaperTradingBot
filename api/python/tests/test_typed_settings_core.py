#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core tests for the typed settings system.

This test module focuses specifically on testing the typed settings functionality
without importing all dependent modules.
"""

import os
import json
import yaml
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Import the typed settings module
from trading_bot.config.typed_settings import (
    TradingBotSettings,
    RiskSettings,
    BrokerSettings,
    APISettings,
    BacktestSettings,
    load_config,
    save_config
)

# Test directories
TEST_DIR = Path(tempfile.mkdtemp())
CONFIG_DIR = TEST_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

# ----- Test Fixtures -----

@pytest.fixture
def yaml_config_file():
    """Create a test YAML configuration file."""
    config_path = CONFIG_DIR / "config.yaml"
    
    config = {
        "broker": {
            "name": "tradier",
            "api_key": "test_key_yaml",
            "account_id": "test_account",
            "sandbox": True
        },
        "risk": {
            "max_position_pct": 0.05,
            "max_risk_pct": 0.01,
            "max_open_trades": 5,
            "max_portfolio_risk": 0.20
        },
        "backtest": {
            "default_symbols": ["AAPL", "MSFT", "GOOG"],
            "initial_capital": 100000.0
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8080,
            "api_keys": {
                "market_intelligence_api": ["test_api_key_1", "test_api_key_2"]
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def json_config_file():
    """Create a test JSON configuration file."""
    config_path = CONFIG_DIR / "config.json"
    
    config = {
        "broker": {
            "name": "tradier",
            "api_key": "test_key_json",
            "account_id": "test_account",
            "sandbox": True
        },
        "risk": {
            "max_position_pct": 0.03,
            "max_risk_pct": 0.005,
            "max_open_trades": 3,
            "max_portfolio_risk": 0.15
        },
        "backtest": {
            "default_symbols": ["SPY", "QQQ", "IWM"],
            "initial_capital": 50000.0
        },
        "api": {
            "host": "0.0.0.0",
            "port": 5000
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


@pytest.fixture
def minimal_config_file():
    """Create a minimal config with only required fields."""
    config_path = CONFIG_DIR / "minimal_config.yaml"
    
    config = {
        "broker": {
            "api_key": "minimal_test_key",
            "account_id": "minimal_account"
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def env_vars():
    """Setup environment variables for testing."""
    old_environ = os.environ.copy()
    
    # Set test environment variables
    os.environ["TRADIER_API_KEY"] = "env_var_api_key"
    os.environ["TRADIER_ACCOUNT_ID"] = "env_var_account_id"
    os.environ["TELEGRAM_TOKEN"] = "env_var_telegram_token"
    os.environ["TELEGRAM_CHAT_ID"] = "env_var_chat_id"
    os.environ["MAX_RISK_PCT"] = "0.02"
    os.environ["INITIAL_CAPITAL"] = "75000"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(old_environ)


# ----- Tests for Configuration Loading -----

def test_load_yaml_config(yaml_config_file):
    """Test loading a YAML configuration file."""
    # Convert Path to string
    settings = load_config(str(yaml_config_file))
    
    assert isinstance(settings, TradingBotSettings)
    assert settings.broker.api_key == "test_key_yaml"
    assert settings.risk.max_position_pct == 0.05
    assert len(settings.backtest.default_symbols) == 3
    assert "AAPL" in settings.backtest.default_symbols
    assert settings.api.port == 8080


def test_load_json_config(json_config_file):
    """Test loading a JSON configuration file."""
    # Convert Path to string
    settings = load_config(str(json_config_file))
    
    assert isinstance(settings, TradingBotSettings)
    assert settings.broker.api_key == "test_key_json"
    assert settings.risk.max_position_pct == 0.03
    assert len(settings.backtest.default_symbols) == 3
    assert "SPY" in settings.backtest.default_symbols
    assert settings.api.port == 5000


def test_minimal_config(minimal_config_file):
    """Test loading a minimal configuration with defaults for missing fields."""
    # Convert Path to string
    settings = load_config(str(minimal_config_file))
    
    assert isinstance(settings, TradingBotSettings)
    assert settings.broker.api_key == "minimal_test_key"
    assert settings.broker.account_id == "minimal_account"
    
    # Check defaults were set for missing fields
    assert settings.broker.name == "tradier"  # Default value
    assert settings.risk.max_position_pct == 0.05  # Default value
    assert settings.backtest.initial_capital == 100000.0  # Default value
    assert settings.api.port == 8000  # Default value


def test_env_var_override(env_vars, minimal_config_file):
    """Test environment variables overriding config file values."""
    # Convert Path to string
    settings = load_config(str(minimal_config_file))
    
    # Environment variables should take precedence
    assert settings.broker.api_key == "env_var_api_key"
    assert settings.broker.account_id == "env_var_account_id"


def test_config_validation():
    """Test configuration validation."""
    # Invalid risk percentage (outside 0-1 range)
    with pytest.raises(ValueError):
        RiskSettings(max_position_pct=1.5)
    
    # Invalid port number
    with pytest.raises(ValueError):
        APISettings(port=70000)


# ----- Tests for Configuration Saving -----

def test_save_config():
    """Test saving configuration to file."""
    # Create a config with all required fields
    settings = TradingBotSettings(
        broker=BrokerSettings(
            api_key="test_save_key",
            account_id="test_save_account"
        ),
        risk=RiskSettings(
            max_position_pct=0.1,
            max_risk_pct=0.02
        ),
        backtest=BacktestSettings(
            default_symbols=["AAPL", "MSFT"],
            initial_capital=150000.0
        )
    )
    
    # Save to YAML
    yaml_path = CONFIG_DIR / "saved_config.yaml"
    result = save_config(settings, str(yaml_path), format="yaml")
    assert result is True
    assert yaml_path.exists()
    
    # Ensure environment variables are set for loading
    os.environ["TRADIER_API_KEY"] = "test_save_key"
    os.environ["TRADIER_ACCOUNT_ID"] = "test_save_account"
    
    # Load it back and check values
    loaded = load_config(str(yaml_path))
    assert loaded.broker.account_id == "test_save_account"
    assert loaded.risk.max_position_pct == 0.1
    assert "AAPL" in loaded.backtest.default_symbols
    
    # Save to JSON
    json_path = CONFIG_DIR / "saved_config.json"
    result = save_config(settings, str(json_path), format="json")
    assert result is True
    assert json_path.exists()
    
    # Load JSON back and check values
    loaded = load_config(str(json_path))
    assert loaded.broker.account_id == "test_save_account"
    assert loaded.risk.max_position_pct == 0.1
    assert "AAPL" in loaded.backtest.default_symbols


# ----- Tests for combining configs -----

def test_combine_configs():
    """Test combining configs from different sources."""
    # Base config with some settings
    base_settings = TradingBotSettings(
        broker=BrokerSettings(
            api_key="base_key",
            account_id="base_account"
        ),
        risk=RiskSettings(
            max_position_pct=0.05
        )
    )
    
    # Update with new risk settings
    risk_override = RiskSettings(
        max_position_pct=0.10,
        max_risk_pct=0.02
    )
    
    # Combine configs
    combined = TradingBotSettings(
        **{**base_settings.dict(), "risk": risk_override.dict()}
    )
    
    assert combined.broker.api_key == "base_key"
    assert combined.risk.max_position_pct == 0.10
    assert combined.risk.max_risk_pct == 0.02


# Clean up test directory after all tests
def teardown_module():
    """Clean up test files after tests complete."""
    import shutil
    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
