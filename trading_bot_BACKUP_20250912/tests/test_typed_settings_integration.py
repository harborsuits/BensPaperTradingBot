#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end integration tests for the typed settings system.

These tests verify that all core modules correctly use the typed settings system
with proper validation, fallback mechanisms, and environment variable integration.
"""

import os
import json
import yaml
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the typed settings module
from trading_bot.config.typed_settings import (
    TradingBotSettings,
    RiskSettings,
    BrokerSettings,
    BacktestSettings,
    APISettings,
    NotificationSettings,
    DataSourceSettings,
    OrchestratorSettings,
    load_config,
    save_config
)

# Import core modules that use typed settings
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.brokers.trade_executor import TradeExecutor
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.core.main_orchestrator import MainOrchestrator
from trading_bot.backtesting.unified_backtester import UnifiedBacktester
from trading_bot.auth.service import AuthService

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
        },
        "notifications": {
            "enable_notifications": True,
            "telegram_token": "test_telegram_token",
            "telegram_chat_id": "test_chat_id"
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
            "port": 5000,
            "api_keys": {
                "market_intelligence_api": ["test_json_key"]
            }
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
    settings = load_config(yaml_config_file)
    
    assert isinstance(settings, TradingBotSettings)
    assert settings.broker.api_key == "test_key_yaml"
    assert settings.risk.max_position_pct == 0.05
    assert len(settings.backtest.default_symbols) == 3
    assert "AAPL" in settings.backtest.default_symbols
    assert settings.api.port == 8080
    assert settings.notifications.telegram_token == "test_telegram_token"


def test_load_json_config(json_config_file):
    """Test loading a JSON configuration file."""
    settings = load_config(json_config_file)
    
    assert isinstance(settings, TradingBotSettings)
    assert settings.broker.api_key == "test_key_json"
    assert settings.risk.max_position_pct == 0.03
    assert len(settings.backtest.default_symbols) == 3
    assert "SPY" in settings.backtest.default_symbols
    assert settings.api.port == 5000


def test_minimal_config(minimal_config_file):
    """Test loading a minimal configuration with defaults for missing fields."""
    settings = load_config(minimal_config_file)
    
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
    settings = load_config(minimal_config_file)
    
    # Environment variables should take precedence
    assert settings.broker.api_key == "env_var_api_key"
    assert settings.broker.account_id == "env_var_account_id"
    
    # Make sure float/int conversion works
    assert settings.risk.max_risk_pct == 0.02
    assert settings.backtest.initial_capital == 75000.0


def test_config_validation():
    """Test configuration validation."""
    # Invalid risk percentage (outside 0-1 range)
    with pytest.raises(ValueError):
        RiskSettings(max_position_pct=1.5)
    
    # Invalid port number
    with pytest.raises(ValueError):
        APISettings(port=70000)
    
    # Invalid date format
    with pytest.raises(ValueError):
        BacktestSettings(default_start_date="01/01/2023")


# ----- Tests for Module Integration -----

def test_risk_manager_with_typed_settings(yaml_config_file):
    """Test that RiskManager correctly uses typed settings."""
    settings = load_config(yaml_config_file)
    
    # Initialize with typed settings
    risk_manager = RiskManager(settings=settings.risk)
    
    # Verify settings were loaded correctly
    assert risk_manager.max_position_pct == 0.05
    assert risk_manager.max_risk_pct == 0.01
    assert risk_manager.max_open_trades == 5
    
    # Test with minimal settings
    minimal_settings = RiskSettings(max_position_pct=0.10)
    risk_manager = RiskManager(settings=minimal_settings)
    assert risk_manager.max_position_pct == 0.10


def test_trade_executor_with_typed_settings(yaml_config_file):
    """Test that TradeExecutor correctly uses typed settings."""
    settings = load_config(yaml_config_file)
    
    # Mock broker client
    mock_client = MagicMock()
    
    # Initialize with typed settings
    executor = TradeExecutor(
        client=mock_client,
        settings=settings.broker
    )
    
    # Verify settings were loaded correctly
    assert executor.broker_name == "tradier"
    assert executor.api_key == "test_key_yaml"
    assert executor.account_id == "test_account"
    assert executor.sandbox_mode is True


def test_unified_backtester_with_typed_settings(yaml_config_file):
    """Test that UnifiedBacktester correctly uses typed settings."""
    settings = load_config(yaml_config_file)
    
    # Initialize with typed settings
    backtester = UnifiedBacktester(settings=settings.backtest)
    
    # Verify settings were loaded correctly
    assert backtester.initial_capital == 100000.0
    assert len(backtester.symbols) == 3 if backtester.symbols else 0
    
    # Test risk settings integration if available
    if hasattr(backtester, "risk_settings") and backtester.risk_settings:
        assert backtester.risk_settings.max_position_pct == 0.05


def test_main_orchestrator_with_typed_settings(yaml_config_file):
    """Test that MainOrchestrator correctly uses typed settings."""
    settings = load_config(yaml_config_file)
    
    # Initialize with typed settings (with mocks to avoid network calls)
    with patch('trading_bot.core.main_orchestrator.RiskManager'), \
         patch('trading_bot.core.main_orchestrator.DataManager'), \
         patch('trading_bot.core.main_orchestrator.StrategyFactory'), \
         patch('trading_bot.core.main_orchestrator.OrderManager'):
             
        orchestrator = MainOrchestrator(settings=settings)
        
        # Verify settings were passed to orchestrator
        assert orchestrator.settings is settings
        assert orchestrator.config_path is None  # We used settings object directly


def test_api_services_with_typed_settings(yaml_config_file):
    """Test that API services correctly use typed settings."""
    settings = load_config(yaml_config_file)
    
    # Test loading API keys from settings
    assert "market_intelligence_api" in settings.api.api_keys
    assert len(settings.api.api_keys["market_intelligence_api"]) == 2
    assert settings.api.api_keys["market_intelligence_api"][0] == "test_api_key_1"


# ----- Tests for Fallback Mechanisms -----

def test_risk_manager_fallback():
    """Test RiskManager fallback to legacy config when typed settings unavailable."""
    # Legacy config dictionary
    legacy_config = {
        "max_position_size_pct": 0.07,
        "max_risk_pct": 0.015,
        "max_open_trades": 10
    }
    
    # Initialize with legacy config
    risk_manager = RiskManager(config=legacy_config)
    
    # Verify legacy config was used
    assert risk_manager.max_position_pct == 0.07
    assert risk_manager.max_risk_pct == 0.015
    assert risk_manager.max_open_trades == 10


def test_backtester_fallback():
    """Test UnifiedBacktester fallback to parameters when typed settings unavailable."""
    # Initialize with direct parameters
    backtester = UnifiedBacktester(
        initial_capital=200000.0,
        strategies=["strategy1", "strategy2"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Verify parameters were used
    assert backtester.initial_capital == 200000.0
    assert len(backtester.strategies) == 2 if hasattr(backtester, "strategies") else 0
    assert backtester.start_date == "2023-01-01" if hasattr(backtester, "start_date") else None


# ----- Tests for Environment Variable Integration -----

def test_env_var_for_sensitive_data(env_vars):
    """Test loading sensitive data from environment variables."""
    # Create minimal config without sensitive data
    config_data = {"broker": {"name": "tradier"}}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as temp:
        yaml.dump(config_data, temp)
        temp.flush()
        
        # Load config - should get sensitive data from env vars
        settings = load_config(temp.name)
        
        # Check that env vars were used
        assert settings.broker.api_key == "env_var_api_key"
        assert settings.broker.account_id == "env_var_account_id"
        
        # Check if notification settings were loaded from env vars
        if hasattr(settings, 'notifications'):
            assert settings.notifications.telegram_token == "env_var_telegram_token"
            assert settings.notifications.telegram_chat_id == "env_var_chat_id"


# ----- Tests for Risk Enforcement -----

def test_risk_enforcement_integration(yaml_config_file):
    """Test risk enforcement across trading execution paths."""
    settings = load_config(yaml_config_file)
    
    # Create risk manager with typed settings
    risk_manager = RiskManager(settings=settings.risk)
    
    # Mock order with excessive position size
    order_data = {
        "symbol": "AAPL",
        "position_size_pct": 0.10,  # Above the 0.05 limit
        "order_type": "market",
        "side": "buy"
    }
    
    # Check that risk check fails
    risk_check_result = risk_manager.check_risk(order_data)
    assert risk_check_result["pass"] is False
    assert "position_size" in risk_check_result["details"]
    
    # Update to acceptable position size
    order_data["position_size_pct"] = 0.04
    risk_check_result = risk_manager.check_risk(order_data)
    assert risk_check_result["pass"] is True


# Clean up test directory after all tests
def teardown_module():
    """Clean up test files after tests complete."""
    import shutil
    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
