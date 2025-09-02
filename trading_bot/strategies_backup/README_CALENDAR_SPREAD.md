# Calendar Spread Strategy

This README provides an overview of the Calendar Spread Strategy implementation in the trading bot framework.

## Overview

The Calendar Spread Strategy exploits the accelerated time decay of front-month options versus longer-dated options by selling short-dated premium and hedging with a longer-dated long-leg at the same strike. This approach captures net theta decay while keeping directional exposure minimal.

## Implementation Structure

The strategy is implemented across several files:

1. `trading_bot/strategies/calendar_spread.py` - Core strategy class implementation
2. `trading_bot/config/calendar_spread_config.yaml` - Configuration parameters
3. `trading_bot/utils/calendar_spread_utils.py` - Utility functions for the strategy

## Blueprint Sections

The strategy follows a 10-section blueprint:

### 1. Strategy Philosophy
Exploiting accelerated time decay by selling front-month options and buying longer-dated options at the same strike.

### 2. Underlying & Option Universe & Timeframe
- **Underlyings:** Highly liquid stocks, ETFs, or indices (e.g., SPY, QQQ, AAPL)
- **Short-leg expiry:** 7-21 DTE (captures steepest theta)
- **Long-leg expiry:** 45-90 DTE (provides vega hedge and directional flexibility)
- **Holding period:** Roll or close front-leg ~5-7 DTE before expiry; hold long-leg until 20-30 DTE remain

### 3. Selection Criteria for Underlying
- **Volatility regime:** IV_rank 30-60% to ensure front-month premium is rich but long-leg isn't too expensive
- **Trend neutrality:** Underlying trading within a defined range or gentle trend to avoid large directional gamma risk
- **Liquidity filters:**
  - Underlying ADV ≥ 500K shares
  - Both legs' OI ≥ 1K contracts
  - Bid-ask spreads ≤ 0.15%

### 4. Spread Construction
- **Strike:** ATM or nearest strike to spot for maximum front-leg theta
- **Net debit:** Small net debit (≤ 1% of equity per spread)
- **Ratio:** 1:1 leg count to keep payoff symmetric

### 5. Expiration & Roll Timing
- **Entry window:** Establish when front leg has 14-21 DTE and long leg has ≥ 45 DTE
- **Roll policies:**
  - **Time-based:** Roll front-leg when it hits 5-7 DTE
  - **Event-based:** Roll early if underlying moves strongly or IV spikes/crashes

### 6. Entry Execution
- **Combo order:** Place both legs simultaneously to lock net debit
- **Limit pricing:** Target theoretical calendar value with minimal slippage
- **Fallback:** Leg in long-leg first if combo order not supported

### 7. Exit & Adjustment Rules
- **Profit-take:** Close when P&L ≥ 50-75% of max theoretical calendar value
- **Time-based exit:** Close or roll front-leg at 5-7 DTE
- **Stop-loss:** If net debit loss > 1× initial debit, close to preserve capital

### 8. Position Sizing & Risk Controls
- **Risk per spread:** ≤ 1% of equity
- **Max concurrent calendars:** 3-5 across diversified underlyings
- **Margin buffer:** Ensure margin requirement ≤ 10% equity per spread

### 9. Backtesting & Performance Metrics
- **Backtest window:** ≥ 3 years of daily option-chain data
- **Key metrics:** Theta capture ratio, win rate, max drawdown, roll frequency and cost, net ROI

### 10. Continuous Optimization
- **Monthly review:** Re-optimize DTE bands and strike offsets
- **IV-adaptive entries:** Shift DTE window when IV_rank deviates
- **Strike biasing:** Tilt strikes when justified

## Configuration Parameters

The strategy is highly configurable through the `calendar_spread_config.yaml` file. Key parameter groups include:
- Strategy philosophy parameters
- Universe and timeframe parameters
- Selection criteria parameters
- Spread construction parameters
- Expiration and roll parameters
- Entry execution parameters
- Exit and adjustment parameters
- Position sizing and risk control parameters
- Backtesting parameters
- Optimization parameters

## Usage

To use the Calendar Spread Strategy:

1. Create an instance of `CalendarSpreadStrategy` with appropriate parameters
2. Load market data including option chains for supported underlyings
3. Call `generate_signals()` to get trading signals
4. Execute the signals through your broker/execution layer

Example:

```python
from trading_bot.strategies.calendar_spread import CalendarSpreadStrategy
from trading_bot.utils.calendar_spread_utils import load_calendar_spread_config, flatten_config_to_parameters

# Load configuration
config = load_calendar_spread_config('trading_bot/config/calendar_spread_config.yaml')
parameters = flatten_config_to_parameters(config)

# Create strategy instance
strategy = CalendarSpreadStrategy(name='CalendarSpread', parameters=parameters)

# Generate signals (assuming you have market_data with option chains)
signals = strategy.generate_signals(market_data)

# Process signals
for symbol, signal in signals.items():
    print(f"Calendar spread signal for {symbol}: {signal.signal_type} at {signal.price}")
```

## Implementation Status

This is a blueprint implementation with TODO items marked throughout the code. The actual implementation of several critical components is pending:

- IV rank calculation
- Theoretical option pricing models
- Strike selection logic
- Calendar spread value calculation
- Roll timing implementation
- Performance metrics calculation

These components should be implemented according to the TODOs in the code before deploying the strategy in a live environment. 