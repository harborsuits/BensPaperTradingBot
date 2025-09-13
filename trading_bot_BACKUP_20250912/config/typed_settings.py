"""
Typed configuration system using Pydantic models.

This module provides a standardized way to load and validate configuration
from various sources (YAML, JSON, environment variables) with type checking,
validation, and default values.
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
try:
    # Attempt to import model_validator for Pydantic v2
    from pydantic import model_validator
except ImportError:
    # Fall back to root_validator for Pydantic v1
    from pydantic import root_validator as model_validator

class BrokerSettings(BaseModel):
    """Broker-specific configuration settings."""
    name: str = "tradier"
    api_key: str
    account_id: str
    sandbox: bool = True
    base_url: Optional[str] = None
    timeout_seconds: int = 15
    max_retries: int = 3
    
    # For live trading environments
    paper_trading: bool = True
    
    @validator('api_key', allow_reuse=True)
    def api_key_must_not_be_empty(cls, v):
        if v == "":
            raise ValueError('API key cannot be empty string')
        return v

class RiskSettings(BaseModel):
    """Risk management configuration."""
    max_position_pct: float = 0.05
    max_risk_pct: float = 0.01
    max_portfolio_risk: float = 0.20
    max_correlated_positions: int = 3
    max_sector_allocation: float = 0.30
    max_open_trades: int = 5
    correlation_threshold: float = 0.7
    enable_portfolio_stop_loss: bool = True
    portfolio_stop_loss_pct: float = 0.05
    enable_position_stop_loss: bool = True
    
    @validator('max_position_pct', 'max_risk_pct', 'max_portfolio_risk', 'max_sector_allocation')
    def validate_percentages(cls, v, values, **kwargs):
        if v is not None and (v < 0 or v > 1):
            field_name = kwargs.get('field').name if 'field' in kwargs else 'Percentage value'
            raise ValueError(f'{field_name} must be between 0 and 1')
        return v

class DataSourceSettings(BaseModel):
    """Data provider configuration."""
    provider: str = "tradier"  # Default to broker's API for data
    use_websocket: bool = False
    cache_expiry_seconds: int = 10
    max_cache_items: int = 1000
    historical_source: str = "alpha_vantage"
    api_keys: Dict[str, str] = Field(default_factory=dict)
    
    @validator('cache_expiry_seconds')
    def validate_cache_expiry(cls, v):
        if v < 1:
            raise ValueError('Cache expiry must be at least 1 second')
        return v

class NotificationSettings(BaseModel):
    """Notification service configuration."""
    enable_notifications: bool = True
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    slack_webhook: Optional[str] = None
    email_to: Optional[str] = None
    email_from: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: Optional[int] = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    
    notification_levels: List[str] = ["critical", "error", "warning", "info"]
    
    @model_validator(mode='after')
    def check_notification_channels(self):
        """Ensure at least one notification channel is properly configured"""
        telegram_token = self.telegram_token
        telegram_chat_id = self.telegram_chat_id
        slack_webhook = self.slack_webhook
        email_to = self.email_to
        email_from = self.email_from
        email_smtp_server = self.email_smtp_server
        email_username = self.email_username
        email_password = self.email_password
        enable_notifications = self.enable_notifications
        
        has_telegram = telegram_token and telegram_chat_id if (telegram_token and telegram_chat_id) else False
        has_slack = bool(slack_webhook)
        has_email = all([
            email_to,
            email_from,
            email_smtp_server,
            email_username,
            email_password
        ]) if (email_to and email_from and email_smtp_server and email_username and email_password) else False
        
        if enable_notifications and not any([has_telegram, has_slack, has_email]):
            import warnings
            warnings.warn("No notification channels configured. Notifications are enabled but won't be sent.")
        
        return self

class OrchestratorSettings(BaseModel):
    """Autonomous orchestrator configuration."""
    step_interval_seconds: int = 30
    enabled_strategies: List[str] = Field(default_factory=list)
    trading_hours_only: bool = True
    market_hours_start: str = "09:30"
    market_hours_end: str = "16:00"
    timezone: str = "America/New_York"
    
    @validator('step_interval_seconds')
    def validate_step_interval(cls, v):
        if v < 1:
            raise ValueError('Step interval must be at least 1 second')
        return v

class BacktestSettings(BaseModel):
    """Backtesting configuration."""
    default_symbols: List[str] = Field(default_factory=list)
    default_start_date: str = (datetime.now().replace(year=datetime.now().year-1)).strftime("%Y-%m-%d")
    default_end_date: str = datetime.now().strftime("%Y-%m-%d")
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.0005
    data_source: str = "local"  # local, alpha_vantage, etc.
    
    @validator('default_start_date', 'default_end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in format YYYY-MM-DD")
        return v

class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file_path: Optional[str] = "./logs/trading_bot.log"
    max_size_mb: int = 10
    backup_count: int = 5
    console_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()

class UISettings(BaseModel):
    """User interface settings."""
    theme: str = "dark"
    refresh_interval_seconds: int = 5
    default_page: str = "dashboard"
    chart_style: str = "candle"
    show_indicators: bool = True
    locale: str = "en-US"
    timezone_display: str = "local"

class APISettings(BaseModel):
    """API configuration settings."""
    enable_api: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_keys: Dict[str, List[str]] = Field(default_factory=dict)
    rate_limit_requests: int = 100
    rate_limit_period_seconds: int = 60
    require_authentication: bool = True
    token_expiry_minutes: int = 60
    
    @validator('port')
    def validate_port(cls, v):
        if not (0 <= v <= 65535):
            raise ValueError('Port must be between 0 and 65535')
        return v

class TradingBotSettings(BaseModel):
    """Master configuration for the trading bot system."""
    broker: BrokerSettings
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSourceSettings = Field(default_factory=DataSourceSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ui: UISettings = Field(default_factory=UISettings)
    api: APISettings = Field(default_factory=APISettings)
    
    environment: str = "development"
    version: str = "1.0.0"
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'testing', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    class Config:
        """Pydantic config settings"""
        extra = "ignore"  # Allow extra fields but ignore them
        validate_assignment = True  # Validate values when assigned after model creation


def load_config(config_path: Optional[str] = None) -> TradingBotSettings:
    """
    Load configuration from the specified YAML/JSON file, with fallbacks to environment variables.
    
    Args:
        config_path: Path to the config file (YAML or JSON)
        
    Returns:
        Validated TradingBotSettings instance
    """
    config_data = {}
    
    # Priority order for config files
    if not config_path:
        potential_paths = [
            os.environ.get("TRADING_CONFIG_PATH"),
            "./config.yaml",
            "./config.yml",
            "./config.json",
            "./trading_bot/config/config.yaml",
            "./trading_bot/config/config.json",
        ]
        
        for path in potential_paths:
            if path and os.path.exists(path):
                config_path = path
                break
    
    # Load from file if available
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_data = json.load(f)
    
    # Override with environment variables if available
    # Environment variables take precedence over file configs
    if os.environ.get("TRADIER_API_KEY"):
        if "broker" not in config_data:
            config_data["broker"] = {}
        config_data["broker"]["api_key"] = os.environ.get("TRADIER_API_KEY")
    
    if os.environ.get("TRADIER_ACCOUNT_ID"):
        if "broker" not in config_data:
            config_data["broker"] = {}
        config_data["broker"]["account_id"] = os.environ.get("TRADIER_ACCOUNT_ID")
    
    if os.environ.get("TRADIER_SANDBOX"):
        if "broker" not in config_data:
            config_data["broker"] = {}
        config_data["broker"]["sandbox"] = os.environ.get("TRADIER_SANDBOX").lower() == "true"
    
    # Add other environment variable overrides as needed
    
    # Create validated settings object
    try:
        settings = TradingBotSettings(**config_data)
        return settings
    except Exception as e:
        # Log the validation error and fall back to defaults where possible
        print(f"Config validation error: {str(e)}")
        
        # At minimum, we need broker settings
        if "broker" not in config_data or not config_data["broker"].get("api_key"):
            raise ValueError(
                "Broker API key must be provided in config file or TRADIER_API_KEY environment variable"
            )
        
        # Try again with just the minimum validated broker settings
        minimal_config = {
            "broker": {
                "api_key": config_data["broker"]["api_key"],
                "account_id": config_data["broker"].get("account_id", ""),
                "sandbox": config_data["broker"].get("sandbox", True)
            }
        }
        
        return TradingBotSettings(**minimal_config)


def save_config(settings: TradingBotSettings, config_path: str, format: str = "yaml") -> bool:
    """
    Save configuration to a file.
    
    Args:
        settings: TradingBotSettings instance
        config_path: Path to save the config to
        format: Format to save in ("yaml" or "json")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert settings to dict, excluding any secrets
        config_dict = settings.dict(exclude={"broker": {"api_key"}})
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Write to file in the specified format
        with open(config_path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(config_dict, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving config: {str(e)}")
        return False


def convert_config(input_path: str, output_path: str, output_format: str = "yaml") -> bool:
    """
    Convert between config formats.
    
    Args:
        input_path: Path to the input config file
        output_path: Path to save the output config to
        output_format: Format to save in ("yaml" or "json")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load and validate
        settings = load_config(input_path)
        
        # Save in new format
        return save_config(settings, output_path, output_format)
    except Exception as e:
        print(f"Error converting config: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Simple command line argument parsing
    if len(sys.argv) > 1:
        if sys.argv[1] == "convert" and len(sys.argv) >= 4:
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            format_type = "yaml" if len(sys.argv) < 5 else sys.argv[4]
            success = convert_config(input_file, output_file, format_type)
            print(f"Config conversion {'successful' if success else 'failed'}")
        else:
            print("Usage: python typed_settings.py convert input_config output_config [yaml|json]")
    else:
        # Just load and print config
        try:
            settings = load_config()
            print(f"Loaded configuration for environment: {settings.environment}")
            print(f"Broker: {settings.broker.name} ({'sandbox' if settings.broker.sandbox else 'live'})")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
