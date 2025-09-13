# Butterfly Spread Strategy

## Overview
The Butterfly Spread Strategy is designed to capitalize on low-volatility, range-bound market conditions by creating a four-legged options position. This strategy maximizes profit when the underlying asset stays near your chosen strike price at expiration, while limiting risk through defined maximum loss.

## Strategy Philosophy
This strategy sells narrowly-spaced wings and hedges with outer wings to maximize premium capture when the underlying stays near your chosen strike at expiration. It's ideal for markets you expect to trade in a tight range with low volatility.

## Components

### 1. Strategy Structure
- **Core Files**:
  - `butterfly_spread.py`: Main strategy implementation
  - `butterfly_utils.py`: Helper functions for option pricing and spread construction
  - `butterfly_spread_config.py`: Configurable parameters

### 2. Underlying & Option Universe
- Highly liquid large-cap stocks or ETFs with tight option spreads (e.g., SPY, QQQ, AAPL)
- Monthly option expirations (25-45 DTE) for balanced theta decay
- Holding period of 2-6 weeks, aligned with expiration cycles

### 3. Selection Criteria
- Underlying trading in a tight range over past 10-20 days (low realized volatility)
- Moderate to low IV rank (20-50%) for reasonable premium without extreme skew
- Liquidity filters: Underlying ADV ≥ 1M; option OI ≥ 500; bid-ask spread ≤ 0.10%

### 4. Spread Construction
- Center strike (sell): ATM or nearest strike to current price (delta ~0.45-0.55)
- Inner wings (buy): One strike above and below center
- Outer wings (buy): One additional strike beyond each inner wing
- Structure can be adjusted for debit or credit based on your risk profile

## Usage Examples

### Basic Setup

```python
from trading_bot.strategies.butterfly_spread import ButterflySpreadStrategy
from trading_bot.config.butterfly_spread_config import get_butterfly_config

# Initialize with default parameters
butterfly_strategy = ButterflySpreadStrategy(
    name="SPY_Butterfly",
    parameters=get_butterfly_config()
)

# Generate signals
signals = butterfly_strategy.generate_signals(market_data)
```

### Custom Configuration

```python
# Custom parameters
custom_params = {
    "center_strike_delta": 0.50,  # At-the-money
    "inner_wing_width": 2,        # 2 strikes width for inner wings
    "outer_wing_width": 3,        # 3 strikes width for outer wings
    "min_days_to_expiration": 30,
    "max_days_to_expiration": 40,
    "profit_take_pct": 75         # Take profit at 75% of max potential
}

# Initialize with custom parameters
butterfly_strategy = ButterflySpreadStrategy(
    name="QQQ_Wide_Butterfly",
    parameters=custom_params
)
```

### Creating and Analyzing a Butterfly Spread

```python
from trading_bot.utils.butterfly_utils import ButterflySpread, price_butterfly_spread
from datetime import datetime, timedelta

# Create a butterfly spread
butterfly = ButterflySpread(
    symbol="SPY",
    expiration=datetime.now() + timedelta(days=35),
    center_strike=450.0,
    inner_wings_width=5.0,
    outer_wings_width=5.0,
    quantity=1,
    option_type="call"
)

# Price the butterfly spread
pricing = price_butterfly_spread(
    butterfly=butterfly,
    underlying_price=450.0,
    days_to_expiration=35,
    volatility=0.18
)

print(f"Butterfly price: ${pricing['price']:.2f}")
print(f"Max profit: ${pricing['max_profit']:.2f}")
print(f"Max loss: ${pricing['max_loss']:.2f}")
print(f"Breakeven points: ${pricing['breakeven_points'][0]:.2f} and ${pricing['breakeven_points'][1]:.2f}")
```

### Position Management

```python
from trading_bot.utils.butterfly_utils import should_adjust_butterfly

# Check if adjustment is needed
current_price = 452.50
days_remaining = 15
should_adjust, adjustment_type, details = should_adjust_butterfly(
    butterfly=butterfly,
    current_price=current_price,
    days_to_expiration=days_remaining
)

if should_adjust:
    print(f"Adjustment recommended: {adjustment_type}")
    print(f"Reason: {details['reason']}")
    print(f"Action: {details['action']}")
```

## Risk Management

### Position Sizing
- Risk per butterfly: ≤ 1% of account equity
- Maximum concurrent butterflies: 3-5 across uncorrelated underlyings
- Margin requirement should not exceed 5% of equity per position

### Exit Rules
- **Profit target**: Close at 50-75% of max potential gain
- **Time-based exit**: Close or roll at 7-10 DTE to avoid pin risk
- **Stop-loss**: Close if value declines to > 150% of paid debit

## Performance Metrics
The strategy tracks the following metrics:
- Win rate (profitable closes vs max loss)
- Average return per butterfly cycle
- Maximum drawdown on butterfly P&L
- Theta decay capture vs theoretical model
- Sensitivity to underlying drift

## Configuration Options
See `butterfly_spread_config.py` for all available configuration options including:
- Selection criteria thresholds
- Spread construction parameters
- Expiration windows
- Exit and management rules
- Position sizing and risk controls
- Advanced options for experienced users

## Notes
- This strategy performs best in low-volatility environments
- Consider wider wings in higher volatility environments
- Monthly review and parameter adjustment recommended based on market conditions
- The strategy includes dynamic re-centering and adjustment capabilities
- Optional ML overlay can help refine strike selection based on past performance 