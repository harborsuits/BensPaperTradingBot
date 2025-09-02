# Strategy Management System

A sophisticated, adaptive strategy management system that dynamically allocates capital across trading strategies based on market conditions, strategy performance, and continuous learning.

## Components

The system consists of three main components:

1. **Dynamic Strategy Rotator**: Adjusts strategy allocations based on market conditions and performance.
2. **Context Decision Integration**: Integrates unified market context (including news sentiment) with strategy decisions.
3. **Continuous Learning System**: Analyzes strategy performance over time and adapts parameters and allocations.

## Features

- **Regime-aware strategy allocation**: Allocates more capital to strategies that perform well in the current market regime
- **Risk-based adjustments**: Automatically reduces exposure to riskier strategies during drawdowns
- **Change constraints**: Prevents excessive allocation shifts that could increase trading costs
- **News impact analysis**: Processes news sentiment and impact on trading decisions
- **Continuous learning**: Adapts strategy parameters based on historical performance
- **Performance tracking**: Records detailed performance metrics across different market regimes
- **Parameter optimization**: Identifies optimal parameter values for each strategy in different market conditions

## Usage

```python
from trading_bot.strategy_management import CoreContext, MarketContext
from trading_bot.strategy_management.factory import create_strategy_management_system
from your_project.prioritizer import YourStrategyPrioritizer
from your_project.unified_context import YourUnifiedContextManager

# Create the core components
core_context = CoreContext()
strategy_prioritizer = YourStrategyPrioritizer()
unified_context = YourUnifiedContextManager()

# Create the strategy management system
system = create_strategy_management_system(
    core_context=core_context,
    strategy_prioritizer=strategy_prioritizer,
    unified_context_manager=unified_context,
    config_path="config/strategy_management.json"
)

# Access the components
rotator = system["rotator"]
integration = system["integration"]
learning = system["learning"]

# Start the system
async def start_system():
    # Start the context integration (which handles automatic rotation)
    await integration.start()
    
    # You can manually trigger strategy rotation if needed
    rotation_result = rotator.rotate_strategies(force=True)
    
    # Force a learning cycle
    learning_result = learning.force_learning_run()
    
    # Get learning insights
    report = learning.get_learning_report()
    
    print(f"Learning insights: {report}")
```

## Configuration

The system is highly configurable through a JSON configuration file. Example:

```json
{
  "rotator": {
    "rotation_frequency_days": 7,
    "min_change_threshold": 5.0,
    "force_on_regime_change": true,
    "max_allocation_change": 15.0,
    "drawdown_allocation_reduction": 0.5,
    "max_drawdown_threshold": 10.0
  },
  "integration": {
    "auto_rotation_enabled": true,
    "auto_rotation_interval_days": 7,
    "respond_to_unified_signals": true,
    "unified_signal_threshold": 0.7,
    "risk_triggered_rotation": true,
    "risk_threshold": "elevated"
  },
  "learning": {
    "learning_frequency_days": 14,
    "min_data_points": 30,
    "max_history_days": 365,
    "learning_rate": 0.2,
    "regime_weight": 0.7,
    "data_dir": "data/learning"
  }
}
```

## Dependencies

- Python 3.7+
- numpy
- pandas
- filelock

## Implementation Details

### Dynamic Strategy Rotator

The rotator implements the following algorithm:

1. Prioritize strategies based on current market context
2. Apply risk-based adjustments to allocations based on drawdowns
3. Apply change constraints to limit allocation shifts
4. Verify significant changes exceed the minimum threshold
5. Update strategy allocations and record history

### Context Decision Integration

Provides bidirectional integration between the unified market context and strategy decisions:

1. Triggers rotation on significant market regime changes
2. Processes news sentiment impact on market context
3. Responds to significant signal changes
4. Adjusts allocations based on risk levels

### Continuous Learning System

Analyzes historical performance to optimize strategy parameters:

1. Tracks performance by market regime
2. Analyzes parameter sensitivity 
3. Generates appropriate allocation adjustments
4. Applies adjustments with appropriate learning rate
5. Maintains strategy profiles across different regimes 