# Risk Management

The BensBot Trading System's risk management framework provides robust protection against excessive losses by enforcing limits at multiple levels.

## Core Principles

The risk management system is built on several key principles:

1. **Pre-trade Risk Checks** - Validate all trades before execution
2. **Portfolio-level Protection** - Monitor and limit aggregate risk
3. **Position-level Controls** - Set limits for individual positions
4. **Automated Stop-Loss** - React to adverse market movements
5. **Correlated Position Management** - Prevent over-concentration

## RiskSettings Configuration

The risk management system is configured through the `RiskSettings` class in the typed settings system:

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
```

## Key Risk Parameters

### Position Sizing

- **max_position_pct**: Maximum size of any single position relative to portfolio value
- **max_risk_pct**: Maximum amount of portfolio value that can be risked on a single trade

### Portfolio Risk

- **max_portfolio_risk**: Maximum total risk allowed across all positions
- **max_open_trades**: Maximum number of concurrent open positions
- **portfolio_stop_loss_pct**: Drawdown threshold for portfolio-wide stop-loss

### Concentration Risk

- **max_correlated_positions**: Maximum number of highly correlated positions
- **correlation_threshold**: Correlation coefficient threshold to consider positions related
- **max_sector_allocation**: Maximum allocation to any single sector

## Risk Check Implementation

The risk management module performs the following checks:

### Pre-Trade Risk Checks

```python
def check_trade(self, symbol, side, quantity, price):
    """Validate a trade before execution."""
    
    # Get position value
    position_value = quantity * price
    
    # Calculate position size as percentage of portfolio
    position_pct = position_value / self.portfolio_value
    
    # Check maximum position size
    if position_pct > self.settings.max_position_pct:
        raise RiskViolationError(f"Position size ({position_pct:.2%}) exceeds maximum allowed ({self.settings.max_position_pct:.2%})")
    
    # Check current risk level
    current_risk = self.calculate_portfolio_risk()
    trade_risk = self.calculate_trade_risk(symbol, quantity, price)
    
    # Check maximum risk per trade
    if trade_risk > self.settings.max_risk_pct:
        raise RiskViolationError(f"Trade risk ({trade_risk:.2%}) exceeds maximum allowed ({self.settings.max_risk_pct:.2%})")
    
    # Check maximum portfolio risk
    if current_risk + trade_risk > self.settings.max_portfolio_risk:
        raise RiskViolationError(f"Portfolio risk would exceed maximum ({self.settings.max_portfolio_risk:.2%})")
```

### Correlation Checks

```python
def check_correlation(self, symbol):
    """Check if adding a position would violate correlation limits."""
    
    # Get current positions
    positions = self.get_current_positions()
    
    # Get correlated positions
    correlated = self.find_correlated_positions(symbol, positions)
    
    # Check maximum correlated positions
    if len(correlated) >= self.settings.max_correlated_positions:
        return False, f"Adding {symbol} would exceed maximum correlated positions limit"
    
    return True, ""
```

## Environment Variable Configuration

Risk parameters can be configured via environment variables:

```bash
export MAX_RISK_PCT="0.01"
export MAX_POSITION_PCT="0.05"
export MAX_PORTFOLIO_RISK="0.25"
export MAX_OPEN_TRADES="5"
export PORTFOLIO_STOP_LOSS_PCT="0.05"
```

## Integration with Trading Components

The risk management system integrates with:

- **Broker Interface**: Enforces position sizing before order execution
- **Strategy Framework**: Provides risk-adjusted position sizing
- **Orchestrator**: Monitors portfolio-level risk metrics
- **Stop-Loss Service**: Triggers exit orders when risk thresholds are breached
- **Backtesting Engine**: Simulates risk management rules on historical data

## Backtest Validation

Risk management rules are validated during backtesting to ensure they work as expected during live trading. The backtesting engine enforces all risk constraints to provide realistic performance estimates.
