# Typed Settings System

The BensBot Trading System uses a comprehensive typed settings system based on Pydantic models to provide robust configuration management, validation, and integration across all components.

## Overview

The typed settings system provides several key benefits:

- **Type validation** - Prevents configuration errors that could lead to runtime failures
- **Default values** - Sensible defaults reduce configuration burden
- **Environment variable integration** - Sensitive data can be securely set via environment variables
- **Documentation** - Field descriptions and validation rules are part of the code
- **IDE support** - Autocompletion and type hints in modern editors

## Core Settings Models

The system is built around a hierarchy of Pydantic models:

```
TradingBotSettings
├── BrokerSettings
├── RiskSettings
├── DataSourceSettings
├── NotificationSettings
├── StrategySettings
├── BacktestSettings
├── LoggingSettings 
├── UISettings
└── APISettings
```

### TradingBotSettings

The master configuration model that contains all subsystem settings.

```python
class TradingBotSettings(BaseModel):
    """Master configuration for the trading bot system."""
    broker: BrokerSettings
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSourceSettings = Field(default_factory=DataSourceSettings)
    notification: NotificationSettings = Field(default_factory=NotificationSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ui: UISettings = Field(default_factory=UISettings)
    api: APISettings = Field(default_factory=APISettings)
    environment: str = "development"
    version: str = "1.0.0"
```

### Other Key Models

#### BrokerSettings

```python
class BrokerSettings(BaseModel):
    """Broker configuration."""
    name: str = "tradier"
    api_key: str
    account_id: str
    sandbox: bool = True

    @validator('name')
    def validate_broker_name(cls, v):
        valid_brokers = ["tradier", "alpaca"]
        if v not in valid_brokers:
            raise ValueError(f"Broker must be one of {valid_brokers}")
        return v
```

#### RiskSettings

```python
class RiskSettings(BaseModel):
    """Risk management configuration."""
    max_position_pct: float = 0.05  # Max position size as percentage of portfolio
    max_risk_pct: float = 0.01  # Max risk per trade
    max_portfolio_risk: float = 0.25  # Max total portfolio risk
    max_correlated_positions: int = 3
    max_sector_allocation: float = 0.30
    max_open_trades: int = 5
    correlation_threshold: float = 0.7
    enable_portfolio_stop_loss: bool = True
    portfolio_stop_loss_pct: float = 0.05
    enable_position_stop_loss: bool = True

    @validator('max_position_pct', 'max_risk_pct', 'max_portfolio_risk', 
              'max_sector_allocation', 'portfolio_stop_loss_pct')
    def validate_percentage(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v
```

## Loading Configuration

Configuration can be loaded from YAML or JSON files and overridden by environment variables:

```python
from trading_bot.config.typed_settings import load_config

# Load from default locations
settings = load_config()

# Or specify a config file path
settings = load_config("/path/to/config.yaml")

# Access settings in a type-safe way
api_key = settings.broker.api_key
max_risk = settings.risk.max_risk_pct
```

## Configuration Validation

All configurations are validated when loaded:

```python
# This will raise a validation error
invalid_config = {
    "risk": {
        "max_position_pct": 1.5  # Invalid: must be between 0-1
    }
}
```

## Integration with Components

All major components in the trading system have been updated to use the typed settings:

- **Broker Adapters**: Load API keys and account details
- **Risk Management**: Enforce position sizes and portfolio limits
- **Strategy Framework**: Configure strategy parameters
- **Orchestrator**: Coordinate system-wide settings
- **Backtesting Engine**: Set test parameters and defaults
- **API Layer**: Configure endpoints, rate limits, and security

## Migration Tool

A migration tool is provided to help convert legacy configurations to the new typed settings format:

```bash
python scripts/migrate_config.py old_config.json config.yaml
```
