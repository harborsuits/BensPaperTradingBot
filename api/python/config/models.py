#!/usr/bin/env python3
"""
Configuration Models

This module defines Pydantic models for configuration validation,
providing type checking and schema validation for all BensBot settings.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
import os

from pydantic import BaseModel, Field, validator, root_validator, SecretStr


class ConfigVersion(str, Enum):
    """Configuration schema version"""
    V1_0 = "1.0"
    V1_1 = "1.1"  # Current version


class LogLevel(str, Enum):
    """Logging level options"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MongoDBConfig(BaseModel):
    """MongoDB connection configuration"""
    uri: str = Field(..., description="MongoDB connection URI")
    database: str = Field(..., description="Database name")
    max_pool_size: int = Field(20, description="Maximum connection pool size")
    timeout_ms: int = Field(5000, description="Connection timeout in milliseconds")
    retry_writes: bool = Field(True, description="Whether to retry write operations")
    retry_reads: bool = Field(True, description="Whether to retry read operations")
    
    @validator('uri')
    def validate_uri(cls, v):
        """Validate MongoDB URI and allow environment variable substitution"""
        if v.startswith('env:'):
            env_var = v.split(':', 1)[1]
            if env_var in os.environ:
                return os.environ[env_var]
            else:
                raise ValueError(f"Environment variable {env_var} not found")
        return v


class RedisConfig(BaseModel):
    """Redis connection configuration"""
    host: str = Field(..., description="Redis host")
    port: int = Field(6379, description="Redis port")
    db: int = Field(0, description="Redis database number")
    timeout: float = Field(5.0, description="Connection timeout in seconds")
    decode_responses: bool = Field(True, description="Whether to decode byte responses to strings")
    key_prefix: str = Field("bensbot:", description="Prefix for all keys")
    
    @validator('host')
    def validate_host(cls, v):
        """Validate Redis host and allow environment variable substitution"""
        if v.startswith('env:'):
            env_var = v.split(':', 1)[1]
            if env_var in os.environ:
                return os.environ[env_var]
            else:
                raise ValueError(f"Environment variable {env_var} not found")
        return v


class PersistenceConfig(BaseModel):
    """Persistence layer configuration"""
    mongodb: MongoDBConfig
    redis: RedisConfig
    recovery: Dict[str, bool] = Field(
        default_factory=lambda: {
            "recover_on_startup": True,
            "recover_open_orders": True,
            "recover_positions": True,
            "recover_pnl": True,
        },
        description="Recovery options"
    )
    sync: Dict[str, Any] = Field(
        default_factory=lambda: {
            "periodic_sync_enabled": True,
            "sync_interval_seconds": 3600,
        },
        description="Synchronization options"
    )


class BrokerCredentials(BaseModel):
    """Broker API credentials"""
    api_key: SecretStr = Field(..., description="API key for broker")
    api_secret: Optional[SecretStr] = Field(None, description="API secret (if required)")
    account_id: Optional[str] = Field(None, description="Account ID (if required)")
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional broker-specific parameters"
    )
    
    @validator('api_key', 'api_secret')
    def validate_credentials(cls, v):
        """Validate credentials and allow environment variable substitution"""
        if isinstance(v, SecretStr) and v.get_secret_value().startswith('env:'):
            env_var = v.get_secret_value().split(':', 1)[1]
            if env_var in os.environ:
                return SecretStr(os.environ[env_var])
            else:
                raise ValueError(f"Environment variable {env_var} not found")
        return v


class BrokerAccount(BaseModel):
    """Broker account information container used across broker adapters."""
    broker_id: str = Field(..., description="Identifier of the broker implementation (e.g., 'tradier', 'etrade').")
    account_id: Optional[str] = Field(None, description="Account identifier used by the broker's API, if different from account number.")
    account_number: Optional[str] = Field(None, description="Human-readable account number.")
    account_type: Optional[str] = Field(None, description="Type of account (e.g., 'cash', 'margin').")
    buying_power: Optional[float] = Field(None, description="Buying power available in the account.")
    cash_balance: Optional[float] = Field(None, description="Available cash balance.")
    equity: Optional[float] = Field(None, description="Total account equity.")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional broker-specific parameters not covered by the standard fields.")

    class Config:
        arbitrary_types_allowed = True


class BrokerConfig(BaseModel):
    """Configuration for a broker"""
    id: str = Field(..., description="Unique identifier for the broker")
    name: str = Field(..., description="Display name for the broker")
    type: str = Field(..., description="Broker type (e.g., 'tradier', 'alpaca', 'interactive_brokers')")
    credentials: BrokerCredentials
    enabled: bool = Field(True, description="Whether this broker is enabled")
    sandbox_mode: bool = Field(True, description="Use sandbox/paper trading mode")
    timeout_seconds: int = Field(30, description="API call timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts for API calls")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for broker notifications")


class AssetRoutingRule(BaseModel):
    """Asset routing rule configuration"""
    asset_type: str = Field(..., description="Asset type (e.g., 'stock', 'option', 'forex')")
    symbols: Optional[List[str]] = Field(None, description="List of symbols this rule applies to")
    market: Optional[str] = Field(None, description="Market (e.g., 'US', 'EU', 'CRYPTO')")
    broker_id: str = Field(..., description="Broker ID to route to")
    priority: int = Field(1, description="Routing priority (lower number = higher priority)")


class BrokerManagerConfig(BaseModel):
    """Broker manager configuration"""
    brokers: List[BrokerConfig] = Field(..., description="List of broker configurations")
    asset_routing: List[AssetRoutingRule] = Field(
        default_factory=list,
        description="Asset routing rules"
    )
    failover_enabled: bool = Field(True, description="Enable broker failover")
    metrics_enabled: bool = Field(True, description="Enable broker performance metrics")
    quote_cache_ttl_seconds: int = Field(5, description="Quote cache time-to-live in seconds")


class RiskManagerConfig(BaseModel):
    """Risk management configuration"""
    max_drawdown_pct: float = Field(5.0, description="Maximum drawdown percentage before circuit breaker triggers")
    volatility_threshold: float = Field(2.5, description="Volatility threshold for circuit breaker")
    cooldown_minutes: int = Field(60, description="Circuit breaker cooldown period in minutes")
    margin_call_threshold: float = Field(0.25, description="Margin call threshold")
    margin_warning_threshold: float = Field(0.35, description="Margin warning threshold")
    max_leverage: float = Field(2.0, description="Maximum allowed leverage")
    position_size_limit_pct: float = Field(
        5.0, 
        description="Maximum position size as percentage of portfolio"
    )
    max_correlated_positions: int = Field(
        3,
        description="Maximum number of highly correlated positions allowed"
    )


class StrategyParameter(BaseModel):
    """Strategy parameter definition"""
    name: str
    type: str = Field(..., description="Parameter type (e.g., 'float', 'int', 'str', 'bool')")
    default: Any
    min_value: Optional[float] = Field(None, description="Minimum value (for numeric parameters)")
    max_value: Optional[float] = Field(None, description="Maximum value (for numeric parameters)")
    choices: Optional[List[Any]] = Field(None, description="Valid choices for this parameter")
    description: str


class StrategyConfig(BaseModel):
    """Trading strategy configuration"""
    id: str = Field(..., description="Unique identifier for the strategy")
    name: str = Field(..., description="Display name for the strategy")
    type: str = Field(..., description="Strategy type (e.g., 'momentum', 'mean_reversion')")
    enabled: bool = Field(True, description="Whether this strategy is enabled")
    assets: List[str] = Field(..., description="Target assets for this strategy")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    schedule: Optional[Dict[str, Any]] = Field(
        None, 
        description="Schedule configuration (when to run)"
    )
    risk_constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Risk constraints specific to this strategy"
    )


class StrategyManagerConfig(BaseModel):
    """Strategy manager configuration"""
    strategies: List[StrategyConfig] = Field(..., description="List of strategy configurations")
    rotation_enabled: bool = Field(False, description="Enable strategy rotation")
    rotation_interval_hours: Optional[int] = Field(
        None, 
        description="Rotation interval in hours (if rotation enabled)"
    )
    concurrent_strategies_limit: int = Field(
        5, 
        description="Maximum number of strategies to run concurrently"
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    file_path: Optional[str] = Field(None, description="Log file path")
    rotation: str = Field("1 day", description="Log rotation interval")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    console_logging: bool = Field(True, description="Enable console logging")


class NotificationConfig(BaseModel):
    """Notification configuration"""
    email: Optional[Dict[str, Any]] = Field(None, description="Email notification settings")
    slack: Optional[Dict[str, Any]] = Field(None, description="Slack notification settings")
    telegram: Optional[Dict[str, Any]] = Field(None, description="Telegram notification settings")
    sms: Optional[Dict[str, Any]] = Field(None, description="SMS notification settings")
    notification_level: LogLevel = Field(
        LogLevel.WARNING, 
        description="Minimum level to trigger notifications"
    )


class DashboardConfig(BaseModel):
    """Dashboard configuration"""
    enabled: bool = Field(True, description="Enable dashboard")
    port: int = Field(8501, description="Dashboard port")
    host: str = Field("0.0.0.0", description="Dashboard host")
    refresh_interval_seconds: int = Field(5, description="Dashboard refresh interval")
    authentication_required: bool = Field(False, description="Require authentication")
    username: Optional[str] = Field(None, description="Dashboard username if auth required")
    password: Optional[SecretStr] = Field(None, description="Dashboard password if auth required")
    theme: str = Field("light", description="Dashboard theme")


class BacktestingConfig(BaseModel):
    """Backtesting configuration"""
    data_source: str = Field("csv", description="Data source (csv, database, api)")
    data_path: Optional[str] = Field(None, description="Path to data files")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    initial_capital: float = Field(10000.0, description="Initial capital for backtest")
    commission_model: str = Field("percentage", description="Commission model to use")
    commission_percentage: float = Field(0.001, description="Commission percentage")


class BotConfig(BaseModel):
    """Root configuration model for BensBot"""
    version: ConfigVersion = Field(ConfigVersion.V1_1, description="Configuration schema version")
    environment: str = Field("development", description="Environment (development, staging, production)")
    persistence: PersistenceConfig
    broker_manager: BrokerManagerConfig
    risk_manager: RiskManagerConfig
    strategy_manager: StrategyManagerConfig
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="Notification configuration"
    )
    dashboard: DashboardConfig = Field(
        default_factory=DashboardConfig,
        description="Dashboard configuration"
    )
    backtesting: Optional[BacktestingConfig] = Field(
        None,
        description="Backtesting configuration (optional)"
    )
    
    @root_validator
    def check_compatibility(cls, values):
        """Ensure configuration compatibility"""
        version = values.get('version')
        if version == ConfigVersion.V1_0:
            # Apply legacy compatibility adjustments if needed
            pass
        return values

try:
    __all__.extend(["BrokerCredentials", "BrokerAccount"])
except NameError:
    __all__ = ["BrokerCredentials", "BrokerAccount"]
