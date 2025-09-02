# Secure Authentication System for Trading Bot

This document provides instructions for setting up, testing, and using the secure authentication system and dashboard for your multi-broker trading platform.

## Overview

The secure authentication system provides:

1. **Encrypted Credential Storage**: Securely stores broker API keys and tokens
2. **Audit Logging**: Records all trading operations for compliance and debugging
3. **Event-Based Integration**: Automatically logs events via the event bus
4. **Dashboard Interface**: Manage credentials and analyze audit logs through a UI

## Setup Instructions

### Prerequisites

Install the required dependencies:

```bash
pip install streamlit pandas plotly cryptography pyyaml
```

### Configuration

1. Create a broker configuration file (if you haven't already):

```bash
cp config/broker_config.json.example config/broker_config.json
```

2. Set the master password for encrypting credentials:

```bash
export TRADING_BOT_MASTER_PASSWORD="your_secure_password"
```

3. Update the broker configuration with your specific settings:

```json
{
  "credential_store": {
    "type": "encrypted",
    "path": "data/credentials.enc"
  },
  "audit_log": {
    "enabled": true,
    "type": "sqlite",
    "path": "data/trading_audit.db"
  },
  "brokers": {
    "tradier": {
      "enabled": true,
      "api_key": "YOUR_TRADIER_API_KEY",
      "account_id": "YOUR_TRADIER_ACCOUNT_ID",
      "sandbox": true,
      "primary": true
    },
    "alpaca": {
      "enabled": true,
      "api_key": "YOUR_ALPACA_API_KEY",
      "api_secret": "YOUR_ALPACA_API_SECRET",
      "paper_trading": true
    }
  }
}
```

## Running the System

### 1. Initialize the Trading Bot with Secure Authentication

Use the setup script to initialize the trading bot with secure authentication:

```bash
python setup_trading_bot.py --config config/broker_config.json
```

This will:
- Create the credential store with your broker credentials
- Initialize the audit log system
- Connect the audit log to the event bus
- Test connections to your brokers

### 2. Launch the Security Dashboard

The dashboard provides a user interface for managing credentials and viewing audit logs:

```bash
streamlit run dashboard/secure_dashboard.py
```

This will open a browser window with the dashboard interface, where you can:
- View and manage broker credentials
- Search and analyze the audit log
- Monitor system status

### 3. Testing the Complete System

To test the entire system with sample data:

```bash
python test_dashboard_system.py
```

This script:
- Creates a test configuration with sample brokers
- Initializes the authentication system
- Generates sample events to populate the audit log
- Launches the dashboard for exploration

## Integration with Your Trading Bot

### Initializing in Your Main Application

Add the following code to your main application to initialize the authentication system:

```python
from trading_bot.brokers.auth_manager import initialize_auth_system, load_config
from trading_bot.core.event_bus import get_global_event_bus

# Load configuration
config = load_config("config/broker_config.json")

# Initialize event bus
event_bus = get_global_event_bus()

# Initialize auth system (credential store, audit log, and audit listener)
credential_store, audit_log, audit_listener = initialize_auth_system(config)

# Now you can create your broker manager with the credential store
from trading_bot.brokers.broker_factory import create_broker_manager
broker_manager = create_broker_manager(config)
```

### Publishing Events to the Audit Log

When using the event bus, events are automatically logged to the audit log:

```python
from trading_bot.core.constants import EventType

# Events published to the event bus are automatically logged
event_bus.create_and_publish(
    EventType.ORDER_SUBMITTED,
    {
        "order_id": "12345",
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.0,
        "broker_id": "tradier"
    },
    "order_manager"
)
```

### Manual Logging (if needed)

You can also log directly to the audit log if necessary:

```python
from trading_bot.brokers.trade_audit_log import AuditEventType

# Log directly to the audit log
audit_log.log_event(
    AuditEventType.ORDER_SUBMITTED,
    {
        "order_id": "12345",
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.0
    },
    broker_id="tradier"
)
```

## Security Best Practices

1. **Master Password**: Always set the master password via environment variable, not in code
2. **Encrypted Storage**: Use encrypted storage in production, YAML only for development
3. **Regular Audits**: Regularly review the audit log for unusual activity
4. **Access Control**: Limit access to the dashboard to authorized personnel
5. **Key Rotation**: Regularly rotate broker API keys and update them in the credential store

## Troubleshooting

### Credential Store Issues

- **Failed to decrypt credentials**: Ensure the master password is correctly set
- **Cannot find credential file**: Check the path in your configuration

### Audit Log Issues

- **Database locked**: Ensure no other process is accessing the SQLite database
- **Missing events**: Check that the audit log listener is registered with the event bus

### Dashboard Issues

- **Dashboard won't start**: Ensure Streamlit is installed (`pip install streamlit`)
- **Cannot connect to brokers**: Verify your broker credentials in the dashboard

## Additional Resources

- [Event Bus Documentation](./EVENT_BUS.md)
- [Audit Log Integration](./AUDIT_LOG_INTEGRATION.md)
- [Multi-Broker Manager](./MULTI_BROKER_MANAGER.md)

## Running Tests

To verify the authentication system is working correctly:

```bash
# Run the integration tests
python -m unittest tests/test_auth_system_integration.py

# Run the credential store tests
python -m unittest tests/test_credential_store.py

# Run the trade audit log tests
python -m unittest tests/test_trade_audit_log.py

# Run the broker integration tests
python -m unittest tests/test_broker_integration.py
```
