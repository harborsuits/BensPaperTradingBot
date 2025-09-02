# Risk Integration Interface

This document defines the API that the UI and other components will use to interact with the risk management system for autonomous trading strategies. It builds upon our established event-driven architecture and follows the principle of standardizing interfaces.

## Overview

The Risk Integration Interface connects the autonomous trading engine with risk management capabilities, ensuring all strategy deployments adhere to proper risk controls. This interface enables:

1. Risk-aware strategy deployment
2. Dynamic position sizing
3. Circuit breaker functionality
4. Risk allocation management
5. Risk monitoring and reporting

## Core Components

The risk integration system consists of several key components that work together:

1. **AutonomousRiskManager** (`risk_integration.py`) - The core bridge between autonomous engine and risk controls
2. **RiskEventTracker** (`risk_event_handlers.py`) - Monitors and tracks risk-related events
3. **StrategyDeploymentPipeline** (`strategy_deployment_pipeline.py`) - Standardized workflow for deploying strategies

## API Reference

### 1. Strategy Deployment

#### Deploy a Strategy with Risk Controls

```python
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.risk.risk_manager import RiskLevel, StopLossType

# Get the risk manager singleton
risk_manager = get_autonomous_risk_manager()

# Deploy a strategy with risk controls
success = risk_manager.deploy_strategy(
    strategy_id="strategy_123",
    allocation_percentage=5.0,         # Capital allocation (0-100%)
    risk_level=RiskLevel.MEDIUM,       # Risk tolerance level
    stop_loss_type=StopLossType.VOLATILITY  # Stop loss methodology
)
```

#### Using the Deployment Pipeline

```python
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline

# Get the deployment pipeline
pipeline = get_deployment_pipeline()

# Deploy strategy with standardized workflow
success, deployment_id = pipeline.deploy_strategy(
    strategy_id="strategy_123",
    allocation_percentage=5.0,
    risk_level=RiskLevel.MEDIUM,
    stop_loss_type=StopLossType.VOLATILITY,
    metadata={"source": "UI", "user": "admin"}
)

if success:
    print(f"Strategy deployed with ID: {deployment_id}")
else:
    print(f"Deployment failed: {deployment_id}")  # Error message
```

### 2. Managing Deployed Strategies

#### Pause a Strategy

```python
# Using the risk manager
risk_manager.pause_strategy(
    strategy_id="strategy_123",
    reason="Manual pause by user"
)

# Or using the deployment pipeline
pipeline.pause_deployment(
    deployment_id="deploy_123",
    reason="Manual pause by user"
)
```

#### Resume a Strategy

```python
# Using the risk manager
risk_manager.resume_strategy(strategy_id="strategy_123")

# Or using the deployment pipeline
pipeline.resume_deployment(deployment_id="deploy_123")
```

#### Stop a Strategy Permanently

```python
pipeline.stop_deployment(
    deployment_id="deploy_123",
    reason="Strategy no longer needed"
)
```

### 3. Risk Monitoring

#### Get Risk Report

```python
# Comprehensive risk report across all strategies
report = risk_manager.get_risk_report()

# For deployment-specific reporting
pipeline_summary = pipeline.get_deployment_summary()
```

#### Monitor Risk Events

```python
from trading_bot.event_system.risk_event_handlers import get_risk_event_tracker

# Get the risk event tracker
risk_tracker = get_risk_event_tracker()

# Get current risk status
status = risk_tracker.get_current_risk_status()

# Get risk alerts
alerts = risk_tracker.get_risk_alerts(limit=10)

# Get risk statistics for last week
stats = risk_tracker.get_risk_statistics(time_period="1w")
```

### 4. Risk Calculation

#### Calculate Position Size

```python
# Calculate risk-adjusted position size
position_size = risk_manager.calculate_position_size(
    strategy_id="strategy_123",
    symbol="AAPL",
    entry_price=150.0,
    stop_price=145.0,
    market_data={"price": 150.0}
)
```

#### Check Circuit Breakers

```python
# Check if circuit breakers are triggered
should_halt, reasons = risk_manager.check_circuit_breakers(market_data)

if should_halt:
    print(f"Trading halted: {reasons}")
```

### 5. Event System Integration

The risk integration system fully integrates with the event system. These are the key events:

#### Risk-related Events

- `STRATEGY_DEPLOYED_WITH_RISK` - Strategy deployed with risk controls
- `CIRCUIT_BREAKER_TRIGGERED` - Risk thresholds exceeded, trading paused
- `RISK_LEVEL_CHANGED` - System risk level changed
- `RISK_ALERT` - Risk warning issued
- `RISK_METRICS_UPDATED` - Risk metrics have been updated

#### Deployment Events

- `STRATEGY_DEPLOYMENT_COMPLETED` - Deployment process completed
- `STRATEGY_DEPLOYMENT_FAILED` - Deployment process failed
- `STRATEGY_DEPLOYMENT_PAUSED` - Deployment paused
- `STRATEGY_DEPLOYMENT_RESUMED` - Deployment resumed
- `STRATEGY_DEPLOYMENT_STOPPED` - Deployment permanently stopped
- `DEPLOYMENT_CIRCUIT_BREAKER` - Circuit breaker affecting deployments

### 6. UI Data Structures

These are the key data structures used for UI integration:

#### Risk Report Structure

```json
{
  "timestamp": "2025-04-25T12:30:45",
  "overall_metrics": {
    "current_drawdown_pct": 2.5,
    "daily_profit_loss_pct": -1.2,
    "total_portfolio_risk": 45.0,
    "open_positions": 5
  },
  "strategy_metrics": {
    "strategy_123": {
      "allocation": 5.0,
      "status": "active",
      "risk_metrics": {
        "current_drawdown": 1.5,
        "daily_profit_loss": 250.0,
        "position_count": 1
      },
      "performance": {
        "profit_loss": 750.0,
        "win_rate": 65.0,
        "trades": 15
      }
    }
  },
  "correlations": {
    "strategy_123": {
      "strategy_456": 0.2
    }
  },
  "circuit_breakers": {
    "portfolio_drawdown": 15.0,
    "strategy_drawdown": 25.0,
    "daily_loss": 5.0,
    "trade_frequency": 20
  },
  "total_strategies": 3,
  "active_strategies": 2
}
```

#### Deployment Status Structure

```json
{
  "deployment_id": "deploy_123",
  "strategy_id": "strategy_123",
  "status": "active",
  "config": {
    "allocation_percentage": 5.0,
    "risk_level": "MEDIUM",
    "stop_loss_type": "VOLATILITY"
  },
  "risk_params": {
    "allocation_percentage": 5.0,
    "risk_level": "MEDIUM",
    "stop_loss_type": "VOLATILITY"
  },
  "deploy_time": "2025-04-25T10:15:30",
  "last_update_time": "2025-04-25T12:30:45",
  "performance": {
    "trades": 10,
    "profit_loss": 500.0,
    "win_rate": 70.0,
    "max_drawdown": 2.5,
    "last_updated": "2025-04-25T12:30:45"
  },
  "status_history": [
    {
      "status": "pending",
      "timestamp": "2025-04-25T10:15:30",
      "reason": "Initial deployment"
    },
    {
      "status": "active",
      "timestamp": "2025-04-25T10:15:35",
      "reason": "Successfully deployed with risk controls"
    }
  ]
}
```

## Integration with Existing Components

The risk integration system integrates with these existing components:

1. **Event System** - Uses the established event bus for communication
2. **Autonomous Engine** - Connects to the engine for strategy deployment
3. **Risk Manager** - Leverages the core risk management capabilities
4. **Component Registry** - Ensures all strategies use a standardized interface

## Workflow Examples

### End-to-End Deployment Workflow

1. UI selects a strategy from optimized candidates
2. UI calls `pipeline.deploy_strategy()` with risk parameters
3. Deployment pipeline creates deployment record and calls risk manager
4. Risk manager applies risk controls and deploys strategy
5. Events are emitted for UI to track deployment progress
6. UI updates to show deployed strategy with risk metrics

### Circuit Breaker Workflow

1. Risk conditions deteriorate beyond thresholds
2. Circuit breaker is triggered via `risk_manager.check_circuit_breakers()`
3. `CIRCUIT_BREAKER_TRIGGERED` event is emitted
4. Risk event tracker records the circuit breaker event
5. Affected strategies are automatically paused
6. UI displays alert and updates strategy statuses

## Error Handling

All API methods follow consistent error handling patterns:

1. Success cases return `True` or the relevant data
2. Failure cases return `False` or appropriate error messages
3. All errors are logged with proper context
4. Events are emitted for critical errors

## UI Requirements

The UI should implement these capabilities:

1. Display current risk status and metrics
2. Show deployed strategies with risk parameters
3. Allow manual strategy pause/resume/stop actions
4. Display circuit breaker status and history
5. Provide risk allocation visualization
6. Show strategy correlation heatmap

## Version Control and Compatibility

This interface is designed to be backward compatible with existing components. All new development should use this standardized interface rather than accessing the risk manager directly.
