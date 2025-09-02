# Audit Log Integration

This document explains how the audit log system has been integrated with the event bus to provide comprehensive logging of all trading operations and events.

## Overview

The audit log integration connects the event-driven architecture of the trading bot to the persistent audit logging system. This ensures that all significant events, including order submissions, fills, cancellations, broker operations, and system status changes, are automatically recorded for compliance, debugging, and analysis.

## Components

### 1. Audit Log Listener (`audit_log_listener.py`)

The `AuditLogListener` class acts as a bridge between the event bus and the audit log:

- Subscribes to relevant event types on the event bus
- Translates event bus events to audit log events
- Extracts relevant details from events and logs them to the audit log
- Provides configurable mapping between event types

### 2. Event Bus Integration

The event bus system (`event_bus.py`) now works with the audit log listener to ensure all events are recorded:

- Trading events published to the event bus are automatically logged
- Custom event mappings ensure proper categorization of events
- No modifications to existing event publication required

### 3. Authentication Manager Integration

The authentication manager (`auth_manager.py`) now includes:

- Methods to create and register audit log listeners
- Integration functions to set up the complete authentication and audit system
- Helper functions for initializing all components at once

## Event Mapping

The following event types are automatically mapped from the event bus to the audit log:

| Event Bus Event Type         | Audit Log Event Type     |
|------------------------------|--------------------------|
| ORDER_CREATED                | ORDER_SUBMITTED          |
| ORDER_SUBMITTED              | ORDER_SUBMITTED          |
| ORDER_FILLED                 | ORDER_FILLED             |
| ORDER_CANCELLED              | ORDER_CANCELLED          |
| ORDER_REJECTED               | ORDER_REJECTED           |
| TRADE_EXECUTED               | ORDER_FILLED             |
| TRADE_CLOSED                 | POSITION_CLOSED          |
| STRATEGY_STARTED             | SYSTEM_ERROR             |
| STRATEGY_STOPPED             | SYSTEM_ERROR             |
| SIGNAL_GENERATED             | STRATEGY_SIGNAL          |
| SYSTEM_STARTED               | SYSTEM_ERROR             |
| SYSTEM_STOPPED               | SYSTEM_ERROR             |
| ERROR_OCCURRED               | SYSTEM_ERROR             |
| RISK_LIMIT_REACHED           | RISK_LIMIT_BREACH        |
| DRAWDOWN_ALERT               | RISK_LIMIT_BREACH        |
| POSITION_SIZE_CALCULATED     | POSITION_UPDATED         |
| HEALTH_STATUS_CHANGED        | BROKER_OPERATION         |
| MODE_CHANGED                 | CONFIG_CHANGE            |

## Usage

### 1. Basic Setup

The easiest way to set up the audit log integration is to use the `initialize_auth_system` function:

```python
from trading_bot.brokers.auth_manager import initialize_auth_system

# Load your configuration
config = load_config("config/broker_config.json")

# Initialize the complete auth system
credential_store, audit_log, audit_listener = initialize_auth_system(config)
```

### 2. Manual Setup

If you need more control over the setup process:

```python
from trading_bot.brokers.auth_manager import create_credential_store, create_audit_log, create_audit_log_listener
from trading_bot.core.event_bus import get_global_event_bus

# Create audit log
audit_log = create_audit_log(config)

# Get the global event bus
event_bus = get_global_event_bus()

# Create and register the audit log listener
audit_listener = create_audit_log_listener(audit_log, event_bus)
```

### 3. Generating Events

Once the audit log listener is registered, any events published to the event bus will be automatically logged:

```python
from trading_bot.core.constants import EventType
from trading_bot.core.event_bus import get_global_event_bus

# Get the global event bus
event_bus = get_global_event_bus()

# Publish an event - this will be automatically logged
event_bus.create_and_publish(
    EventType.ORDER_SUBMITTED,
    {
        "order_id": "12345",
        "symbol": "EURUSD",
        "quantity": 10000,
        "price": 1.1234,
        "broker_id": "tradier"
    },
    "order_manager"
)
```

### 4. Direct Logging

You can also log directly to the audit log if needed:

```python
from trading_bot.brokers.trade_audit_log import AuditEventType

# Log directly to the audit log
audit_log.log_event(
    AuditEventType.ORDER_SUBMITTED,
    {
        "order_id": "12345",
        "symbol": "EURUSD",
        "quantity": 10000,
        "price": 1.1234
    },
    broker_id="tradier"
)
```

## Configuration

The audit log is configured in the broker configuration file:

```json
{
  "audit_log": {
    "enabled": true,
    "type": "sqlite",  // or "json"
    "path": "data/trading_audit.db"
  }
}
```

## Query Examples

After events are logged, you can query the audit log for analysis:

```python
# Get all order events for a specific broker
order_events = audit_log.query_events(
    event_types=[AuditEventType.ORDER_SUBMITTED, AuditEventType.ORDER_FILLED],
    broker_id="tradier",
    start_time=datetime(2025, 4, 1),
    end_time=datetime(2025, 4, 29)
)

# Get all risk limit breach events
risk_events = audit_log.query_events(
    event_types=[AuditEventType.RISK_LIMIT_BREACH],
    start_time=datetime(2025, 4, 1)
)

# Get events for a specific order
order_history = audit_log.get_order_history("12345")
```
