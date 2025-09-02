# BensBot Migration Guide

This guide will help you migrate from the legacy BensBot system to the new unified package structure with persistence layer and containerized deployment.

## Table of Contents

1. [Overview of Changes](#overview-of-changes)
2. [Config File Migration](#config-file-migration)
3. [Script Changes](#script-changes)
4. [Database Migration](#database-migration)
5. [Deployment Migration](#deployment-migration)
6. [Breaking Changes & API Differences](#breaking-changes--api-differences)
7. [Troubleshooting](#troubleshooting)

## Overview of Changes

The BensBot trading system has undergone significant architectural changes to improve reliability, maintainability, and deployability:

| Feature | Legacy System | New System |
| ------- | ------------- | ---------- |
| Package Structure | Fragmented scripts | Unified `trading_bot` package |
| Configuration | Multiple JSON files | Single YAML/JSON with Pydantic validation |
| Persistence | File-based only | MongoDB + Redis with automatic recovery |
| Deployment | Manual process | Docker Compose with containerization |
| Risk Management | Basic | Advanced circuit breakers and margin monitoring |
| Dashboard | Streamlit only | Streamlit + FastAPI |

## Config File Migration

### Automatic Migration

The easiest way to migrate your configuration files is to use the built-in migration utility:

```bash
# Generate a migration report without making changes
python -m trading_bot.config.migrate_configs --report

# Migrate all legacy configs to a new unified config
python -m trading_bot.config.migrate_configs --base-dir ./config --output ./config/config.yaml
```

### Manual Migration

If you prefer to migrate manually, here's how to map old config files to the new structure:

#### 1. `broker_config.json` → `broker_manager` section

**Old:**
```json
{
  "max_retries": 3,
  "retry_delay": 5,
  "brokers": {
    "tradier_main": {
      "api_key": "YOUR_API_KEY",
      "account_id": "YOUR_ACCOUNT_ID",
      "enabled": true
    }
  }
}
```

**New:**
```yaml
broker_manager:
  brokers:
    - id: "tradier_main"
      name: "Tradier Main"
      type: "tradier"
      enabled: true
      timeout_seconds: 30
      retry_attempts: 3
      credentials:
        api_key: "env:TRADIER_API_KEY"
        account_id: "env:TRADIER_ACCOUNT_ID"
  failover_enabled: true
  metrics_enabled: true
```

#### 2. `risk_management.json` → `risk_manager` section

**Old:**
```json
{
  "max_drawdown_pct": 5.0,
  "volatility_threshold": 2.5,
  "margin_thresholds": {
    "warning": 0.35,
    "call": 0.25
  }
}
```

**New:**
```yaml
risk_manager:
  max_drawdown_pct: 5.0
  volatility_threshold: 2.5
  cooldown_minutes: 60
  margin_call_threshold: 0.25
  margin_warning_threshold: 0.35
  max_leverage: 2.0
  position_size_limit_pct: 5.0
```

#### 3. New `persistence` section

This is a new configuration section that didn't exist in the legacy system:

```yaml
persistence:
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "bensbot_trading"
    max_pool_size: 20
    timeout_ms: 5000
  redis:
    host: "localhost"
    port: 6379
    db: 0
    timeout: 5.0
    decode_responses: true
    key_prefix: "bensbot:"
  recovery:
    recover_on_startup: true
    recover_open_orders: true
    recover_positions: true
    recover_pnl: true
  sync:
    periodic_sync_enabled: true
    sync_interval_seconds: 3600
```

## Script Changes

### Entry Point Changes

The main entry point has changed from various scripts to a standardized approach:

| Legacy | New |
| ------ | --- |
| `python app.py` | `python -m trading_bot.run_bot` |
| `python trading_bot/main.py` | `python -m trading_bot.run_bot` |
| `python run_bot.py` | `python -m trading_bot.run_bot` |

### Dashboard

The dashboard is now launched separately from the main trading engine:

```bash
# Legacy
python -m trading_bot.main --mode dashboard

# New
python -m trading_bot.dashboard.app
# or
python -m trading_bot.dashboard.api  # FastAPI version
```

### API Changes

If you were directly importing from modules, some imports have changed:

| Legacy | New |
| ------ | --- |
| `from trading_bot.brokers.adapter import BrokerAdapter` | `from trading_bot.brokers.broker_interface import BrokerInterface` |
| `from trading_bot.core.strategy import Strategy` | `from trading_bot.core.strategy_interface import StrategyInterface` |
| `from trading_bot.analytics.market_regime` | `from trading_bot.analytics.market_regime.detector import MarketRegimeDetector` |

## Database Migration

The new system uses MongoDB and Redis for persistence. There's no automatic migration path from file-based storage, but here's how to set up and import data:

### Setting Up MongoDB and Redis

```bash
# With Docker
docker-compose up -d mongodb redis

# Or locally on macOS with Homebrew
brew services start mongodb-community
brew services start redis
```

### Importing Historical Data

For importing historical trade data:

```python
from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.persistence.order_repository import OrderRepository
from trading_bot.persistence.fill_repository import FillRepository
import pandas as pd

# Initialize connection manager
conn_manager = ConnectionManager(config.persistence)

# Create repositories
order_repo = OrderRepository(conn_manager)
fill_repo = FillRepository(conn_manager)

# Import from CSV
trades_df = pd.read_csv('legacy_trades.csv')
for _, row in trades_df.iterrows():
    # Convert row to Order object and save
    order = Order(
        id=row['order_id'],
        symbol=row['symbol'],
        side=row['side'],
        quantity=row['quantity'],
        order_type=row['order_type'],
        status=row['status'],
        # Add other fields as needed
    )
    order_repo.save(order)
```

## Deployment Migration

### Local Development

```bash
# Install package in development mode
pip install -e .

# Start the bot
python -m trading_bot.run_bot --config config/config.yaml
```

### Docker Deployment

The recommended approach for production is Docker Compose:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

## Breaking Changes & API Differences

### Events System

The event system has been enhanced with new event types and a more formal structure. If you were using custom event handlers, you'll need to update them:

**Legacy:**
```python
def handle_event(event):
    if event.type == "order_placed":
        # Handle order
```

**New:**
```python
from trading_bot.core.events import EventType, OrderPlacedEvent

def handle_order_placed(event: OrderPlacedEvent):
    # Type-safe handling
    order_id = event.order.id
```

### Risk Management

The risk management system now has automatic circuit breakers and forced de-leveraging. If you had custom risk controls, they may need to be adapted to work with the new system.

### Position Management

Position sizing logic is now handled by the `AdaptivePositionManager` which adjusts to market conditions. Previous fixed sizing methods may not be compatible.

## Troubleshooting

### Common Migration Issues

1. **Configuration Not Found**
   
   Error: `ConfigFileNotFoundError: Config file not found at path: ...`
   
   Solution: Make sure to create a unified config file:
   ```bash
   python -m trading_bot.config.migrate_configs --base-dir ./config --output ./config/config.yaml
   ```

2. **MongoDB Connection Errors**
   
   Error: `MongoConnectionError: Failed to connect to MongoDB at ...`
   
   Solution: Make sure MongoDB is running and accessible:
   ```bash
   # Check MongoDB status with Docker
   docker-compose ps mongodb
   
   # Or locally on macOS
   brew services list | grep mongodb
   ```

3. **Missing Module Errors**
   
   Error: `ImportError: No module named 'trading_bot.persistence'`
   
   Solution: Make sure you have installed the package:
   ```bash
   pip install -e .
   ```

4. **Legacy Code References**
   
   Error: Various import or attribute errors
   
   Solution: Check for imports from deprecated modules and update them according to the API Changes section above.

### Getting Help

If you encounter issues during migration, you can:

1. Check the [`docs/`](../docs/) directory for detailed documentation
2. Examine the source code in the [`trading_bot/`](../trading_bot/) package
3. Run the bot with debug logging: `LOG_LEVEL=DEBUG python -m trading_bot.run_bot`
4. Use the migration report for guidance: `python -m trading_bot.config.migrate_configs --report`

## Next Steps

After successfully migrating to the new system, we recommend:

1. Test thoroughly in paper trading mode before enabling live trading
2. Explore the new dashboard capabilities for monitoring
3. Review the risk management settings to ensure they match your risk tolerance
4. Set up proper backup procedures for MongoDB data

For any further questions or assistance, please open an issue in the repository.
