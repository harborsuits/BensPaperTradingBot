# Risk Management System

This document outlines the risk management features implemented in the trading system, including circuit breakers, dynamic position sizing, emergency risk controls, and stress testing capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Circuit Breaker System](#circuit-breaker-system)
3. [Volatility-Based Position Sizing](#volatility-based-position-sizing)
4. [Market Regime Adjustments](#market-regime-adjustments)
5. [Emergency Risk Controls](#emergency-risk-controls)
6. [Stress Testing](#stress-testing)
7. [Risk Scenario Simulations](#risk-scenario-simulations)
8. [Risk Monitoring Dashboard](#risk-monitoring-dashboard)
9. [Configuration](#configuration)

## Overview

The risk management system provides multiple layers of protection against adverse market conditions and trading system errors. It continuously monitors portfolio performance, market volatility, and correlation changes to dynamically adjust trading behavior and limit potential losses.

## Circuit Breaker System

The circuit breaker system temporarily restricts trading activity when predefined risk thresholds are breached.

### Activation Triggers

Circuit breakers can be activated by the following conditions:

1. **Drawdown-Based Triggers**:
   - Daily drawdown exceeding threshold (e.g., -3%)
   - Weekly drawdown exceeding threshold (e.g., -5%)
   - Monthly drawdown exceeding threshold (e.g., -10%)

2. **Volatility-Based Triggers**:
   - Realized volatility exceeding threshold (e.g., 25% annualized)
   - Implied volatility (VIX) spike above threshold

3. **Correlation-Based Triggers**:
   - Strategy correlation exceeding threshold (e.g., 0.8)
   - Indicates diversification breakdown

### Circuit Breaker Levels

The system implements tiered circuit breaker levels:

- **Level 1**: Mild restrictions (15% maximum allocation change)
- **Level 2**: Moderate restrictions (10% maximum allocation change)
- **Level 3**: Severe restrictions (5% maximum allocation change)

### Duration and Deactivation

- Circuit breakers remain active for a configurable duration based on severity level
- Default durations: Level 1 (1 day), Level 2 (3 days), Level 3 (5 days)
- Automatic deactivation once duration expires
- Manual override available for premature deactivation

### Usage Example

```python
# Check circuit breaker status
cb_status = risk_manager.check_circuit_breakers(current_date)

if cb_status["active"]:
    # Apply circuit breaker limits to allocation changes
    new_allocations = risk_manager._apply_circuit_breaker_limits(
        target_allocations, cb_status
    )
```

## Volatility-Based Position Sizing

The system dynamically adjusts position sizes based on current market volatility conditions.

### Volatility Scaling

- Positions are scaled inversely to volatility
- Low volatility periods allow larger positions
- High volatility periods enforce smaller positions

### Position Sizing Algorithm

```
if volatility <= 0.3:  # Low volatility
    adjustment = 1.1   # Allow 10% larger positions
elif volatility <= 0.5:  # Normal volatility
    adjustment = 1.0   # No adjustment
elif volatility <= 0.7:  # Elevated volatility
    adjustment = 0.9 - ((volatility - 0.5) * 0.4)  # Reduce by 10-30%
else:  # High volatility
    adjustment = 0.7 - ((volatility - 0.7) * 1.0)  # Reduce by 40-70%
    adjustment = max(0.3, adjustment)  # Floor at 30%
```

### Maximum Position Limits

- Hard limit on maximum position size (default: 50% of portfolio)
- Prevents overconcentration in any single strategy

## Market Regime Adjustments

Position sizing is further adjusted based on the current market regime.

### Regime-Based Multipliers

- **Bull market**: 1.05× (slightly more aggressive)
- **Bear market**: 0.8× (more conservative)
- **Volatile regime**: 0.7× (much more conservative)
- **Sideways market**: 0.9× (somewhat conservative)
- **Neutral regime**: 1.0× (no adjustment)

### Example Calculation

```python
# Calculate volatility adjustment
volatility = market_context.get('volatility', 0.5)
regime = market_context.get('market_regime', 'neutral')

# Base adjustment from volatility
adjustment = calculate_volatility_adjustment(volatility)

# Apply regime multiplier
regime_multipliers = {
    'bullish': 1.05, 'bearish': 0.8, 'volatile': 0.7,
    'sideways': 0.9, 'neutral': 1.0
}
adjustment *= regime_multipliers.get(regime, 1.0)
```

## Emergency Risk Controls

The system can detect anomalous conditions and apply emergency risk reduction measures.

### High-Risk Strategy Detection

- Monitors strategy-specific metrics:
  - Recent volatility (e.g., > 30% annualized)
  - Recent drawdown (e.g., > 10%)

### Emergency Risk Response

When severe anomalies are detected:

1. Identify high-risk strategies
2. Apply emergency allocation reductions (50-80% depending on severity)
3. Increase cash allocation correspondingly
4. Record emergency action in debug data for later analysis

### Risk Reduction Calculation

```python
# For high-risk strategies
severity = max(0.5, min(0.8, abs(max_drawdown) * 4))  # Scale by drawdown
new_allocation = current_allocation * (1 - severity)
```

## Stress Testing

The system can perform stress tests to assess portfolio resilience under adverse conditions.

### Stress Test Scenarios

- **Market crash**: Large negative returns across strategies with increased volatility
- **Volatility spike**: Moderate negative returns with significantly elevated volatility
- **Correlation breakdown**: Strategies become highly correlated with negative returns

### Risk Assessment

Stress tests generate:

- Projected returns under each scenario
- Projected maximum drawdown
- Projected volatility
- Risk level assessment (low, medium, high, extreme)

### Position Adjustments Based on Stress Tests

```python
# Adjust position sizing based on stress test results
risk_level = stress_results.get('risk_level', 'medium')
risk_adjustments = {
    'low': 1.05,    # Slightly more aggressive
    'medium': 1.0,  # No change
    'high': 0.75,   # Reduce positions by 25%
    'extreme': 0.5  # Reduce positions by 50%
}
adjustment = risk_adjustments.get(risk_level, 1.0)
```

## Risk Scenario Simulations

The system can simulate various risk scenarios to test risk management effectiveness.

### Scenario Types

1. **Market Crash**
   - Parameters: Large, persistent negative returns (e.g., -1.5% daily)
   - Elevated volatility (2.5× normal)
   - "Bearish" market regime

2. **Volatility Spike**
   - Parameters: Moderate negative returns
   - Significantly elevated volatility (3× normal)
   - "Volatile" market regime

3. **Correlation Breakdown**
   - Parameters: Moderate negative returns
   - Elevated volatility (2× normal)
   - High correlation between strategies (0.9)
   - "Bearish" market regime

### Analyzing Risk Responses

The system tracks and analyzes:

- Number of anomalies detected
- Circuit breaker activations
- Emergency risk control actions
- Allocation changes during stress periods vs. normal periods
- Effectiveness of volatility-based position sizing

### Effectiveness Score Calculation

The system generates an overall effectiveness score based on:

- Performance preservation during stress periods
- Responsiveness (appropriate number of risk events triggered)
- Position sizing effectiveness during high volatility

## Risk Monitoring Dashboard

The system includes a dashboard for monitoring risk metrics in real-time.

### Key Dashboard Components

- Current volatility and drawdown metrics
- Circuit breaker status and history
- Strategy-specific risk metrics
- VaR and CVaR metrics
- Stress test results
- Anomaly alerts
- Emergency action log

## Configuration

### Sample Risk Manager Configuration

```json
{
  "circuit_breakers": {
    "drawdown": {
      "daily": {"threshold": -0.03, "level": 1},
      "weekly": {"threshold": -0.05, "level": 2},
      "monthly": {"threshold": -0.10, "level": 3}
    },
    "volatility": {
      "threshold": 0.25,
      "level": 2
    },
    "correlation": {
      "threshold": 0.80,
      "level": 2
    },
    "duration": {
      "level1": 1,
      "level2": 3,
      "level3": 5
    }
  },
  "position_sizing": {
    "max_position": 0.50,
    "volatility_scaling": true,
    "target_volatility": 0.15
  },
  "anomaly_detection": {
    "z_score_threshold": 3.0,
    "window_size": 20
  }
}
```

### Configuration Parameter Descriptions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `circuit_breakers.drawdown.daily.threshold` | Daily drawdown threshold for Level 1 circuit breaker | -0.03 |
| `circuit_breakers.drawdown.weekly.threshold` | Weekly drawdown threshold for Level 2 circuit breaker | -0.05 |
| `circuit_breakers.drawdown.monthly.threshold` | Monthly drawdown threshold for Level 3 circuit breaker | -0.10 |
| `circuit_breakers.volatility.threshold` | Annualized volatility threshold | 0.25 |
| `circuit_breakers.correlation.threshold` | Strategy correlation threshold | 0.80 |
| `circuit_breakers.duration.level1` | Duration in days for Level 1 circuit breaker | 1 |
| `circuit_breakers.duration.level2` | Duration in days for Level 2 circuit breaker | 3 |
| `circuit_breakers.duration.level3` | Duration in days for Level 3 circuit breaker | 5 |
| `position_sizing.max_position` | Maximum position size as fraction of portfolio | 0.50 |
| `position_sizing.volatility_scaling` | Enable volatility scaling for position sizing | true |
| `position_sizing.target_volatility` | Target annualized volatility for scaling | 0.15 |
| `anomaly_detection.z_score_threshold` | Z-score threshold for anomaly detection | 3.0 |
| `anomaly_detection.window_size` | Window size for calculating statistics | 20 | 