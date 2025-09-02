# Feature Flag System

The feature flag system provides a robust mechanism for selectively enabling or disabling trading features without requiring a full deployment or restart of the trading bot. This is particularly useful for managing risk during volatile market conditions or for gradually rolling out new features.

## Key Features

- **Remote Management**: Control features via Telegram commands
- **Persistence**: Flag states survive bot restarts
- **Auto-Rollback**: Automatically disable experimental features after a specified time
- **Dependent Flags**: Configure dependencies between features
- **Confirmation Requirements**: Require explicit confirmation for critical changes
- **Metrics Collection**: Track flag usage and change history
- **Rich Dashboard**: Visualize feature flag status

## Usage

### Basic Usage

To check if a feature is enabled:

```python
from trading_bot.feature_flags import get_feature_flag_service

service = get_feature_flag_service()

if service.is_enabled("my_feature"):
    # Execute feature-specific code
    pass
```

### Creating a New Flag

```python
from trading_bot.feature_flags import get_feature_flag_service, FlagCategory

service = get_feature_flag_service()

success, message = service.create_flag(
    id="high_volatility_protection",
    name="High Volatility Protection",
    description="Reduces position sizes during high market volatility",
    category=FlagCategory.RISK,
    default=True,
    requires_confirmation=True
)
```

### Managing Flags

```python
# Enable a flag
service.set_flag(
    flag_id="experimental_strategy",
    enabled=True,
    changed_by="admin",
    reason="Testing in QA environment"
)

# Disable a flag
service.set_flag(
    flag_id="experimental_strategy",
    enabled=False,
    changed_by="system",
    reason="Excessive drawdown detected"
)

# Reset a flag to its default state
service.reset_flag("experimental_strategy")
```

### Flag Categories

Flags are organized into categories:

- `STRATEGY`: Trading strategies
- `RISK`: Risk management features
- `MONITORING`: Monitoring and alerting
- `NOTIFICATION`: Notification settings
- `DATA`: Data sources and providers
- `EXECUTION`: Order execution features
- `EXPERIMENTAL`: Experimental features
- `SYSTEM`: Core system features

### Receiving Flag Change Notifications

```python
from trading_bot.feature_flags import FlagChangeEvent

def on_flag_change(event: FlagChangeEvent):
    print(f"Flag '{event.flag_id}' changed to {event.enabled} by {event.changed_by}")
    
service = get_feature_flag_service()
service.register_callback(on_flag_change)
```

## Telegram Integration

The feature flag system integrates with the Telegram bot to allow for remote management. The following commands are available:

- `/flags` - List all feature flags
- `/flags <category>` - List flags in a specific category
- `/flag <id>` - Show details for a specific flag
- `/enable <id> [reason]` - Enable a flag
- `/disable <id> [reason]` - Disable a flag
- `/toggle <id> [reason]` - Toggle a flag
- `/create_flag <id> <name> <category> <description>` - Create a new flag
- `/flag_history <id>` - Show change history for a flag

## Dashboard Integration

Use the `FeatureFlagDashboard` class to display feature flag information in the trading bot's dashboard:

```python
from trading_bot.feature_flags.dashboard import FeatureFlagDashboard
from rich.console import Console

console = Console()
dashboard = FeatureFlagDashboard(console)

# Display summary of all flags
dashboard.display_flag_summary()

# Display detailed information by category
dashboard.display_flags_by_category()

# Display details for a specific flag
dashboard.display_flag_details("risk_limits")

# Display recent changes
dashboard.display_recent_changes()
```

## Example

See `trading_bot/examples/feature_flag_demo.py` for a complete example of using the feature flag system.

## Data Storage

Feature flag data is stored in JSON format in the `data/feature_flags/flags.json` file. A backup is created before each write to prevent data loss.

## Best Practices

1. **Use Meaningful IDs**: Choose descriptive IDs that clearly indicate the feature's purpose
2. **Set Appropriate Categories**: Organize flags into logical categories
3. **Provide Good Descriptions**: Write clear descriptions that explain what the flag does
4. **Consider Dependencies**: Define dependencies between flags to prevent inconsistent states
5. **Use Rollback for Experiments**: Set auto-rollback timers for experimental features
6. **Require Confirmation**: Set `requires_confirmation=True` for critical risk features 