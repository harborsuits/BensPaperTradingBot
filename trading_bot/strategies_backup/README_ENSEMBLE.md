# Strategy Ensemble Framework

The Strategy Ensemble framework provides a way to combine multiple trading strategies into a single cohesive strategy. This approach can help improve robustness, reduce overfitting risk, and potentially enhance risk-adjusted returns by leveraging the strengths of different strategies.

## Overview

The framework consists of two main classes:

1. **StrategyEnsemble**: A composite strategy that combines signals from multiple strategies with various weighting methods
2. **DynamicEnsemble**: An extension of StrategyEnsemble that can dynamically adjust component strategies based on market conditions and performance

## Features

- **Multiple Weighting Methods**: 
  - Equal weighting
  - Custom weights
  - Performance-based weighting
  - Volatility-based weighting (inverse volatility)
  - Regime-based weighting
  - Adaptive weighting (performance + correlation adjustment)

- **Correlation Management**: The framework can reduce weights of highly correlated strategies to improve diversification

- **Dynamic Strategy Selection**: The DynamicEnsemble can activate or deactivate strategies based on their recent performance

- **Flexible Architecture**: Works with any strategy that inherits from the base Strategy class

## Usage

### Basic Usage

```python
from trading_bot.strategies.strategy_ensemble import StrategyEnsemble, WeightingMethod
from trading_bot.strategies.macro_trend_strategy import MacroTrendStrategy
from trading_bot.strategies.your_custom_strategy import YourCustomStrategy

# Create individual strategies
trend_strategy = MacroTrendStrategy(
    symbols=["SPY", "QQQ", "IWM"],
    trend_method="macd",
    rebalance_frequency=20
)

custom_strategy = YourCustomStrategy(
    symbols=["SPY", "QQQ", "IWM"],
    # your strategy parameters
)

# Create ensemble with equal weighting
ensemble = StrategyEnsemble(
    strategies=[trend_strategy, custom_strategy],
    weighting_method=WeightingMethod.EQUAL
)

# Generate signals using the ensemble
signals = ensemble.generate_signals(market_data)
```

### Advanced Usage

```python
from trading_bot.strategies.strategy_ensemble import DynamicEnsemble, WeightingMethod

# Create a dynamic ensemble with performance-based weighting
dynamic_ensemble = DynamicEnsemble(
    strategies=[trend_strategy, momentum_strategy, mean_reversion_strategy],
    weighting_method=WeightingMethod.PERFORMANCE,
    performance_window=60,  # Days to look back for calculating performance
    rebalance_frequency=20,  # How often to update weights
    correlation_threshold=0.7,  # Threshold for correlation adjustment
    min_weight=0.1,  # Minimum weight per strategy
    max_weight=0.6,  # Maximum weight per strategy
    min_active_strategies=1,  # Minimum number of active strategies
    max_active_strategies=3,  # Maximum number of active strategies
    activation_threshold=0.2,  # Performance threshold for activation
    deactivation_threshold=-0.1  # Performance threshold for deactivation
)

# Generate signals and the ensemble will internally handle
# strategy selection and weight updates
signals = dynamic_ensemble.generate_signals(market_data)
```

## Weighting Methods

### Equal Weighting
All strategies receive equal weight.

```python
ensemble = StrategyEnsemble(
    strategies=[strategy1, strategy2, strategy3],
    weighting_method=WeightingMethod.EQUAL
)
```

### Custom Weighting
Assign custom weights to each strategy.

```python
ensemble = StrategyEnsemble(
    strategies=[strategy1, strategy2, strategy3],
    weighting_method=WeightingMethod.CUSTOM,
    strategy_weights={"Strategy1": 0.5, "Strategy2": 0.3, "Strategy3": 0.2}
)
```

### Performance-Based Weighting
Weights are assigned proportionally to the risk-adjusted performance of each strategy.

```python
ensemble = StrategyEnsemble(
    strategies=[strategy1, strategy2, strategy3],
    weighting_method=WeightingMethod.PERFORMANCE,
    performance_window=60  # Period to calculate performance
)
```

### Volatility-Based Weighting
Weights are inversely proportional to the volatility of each strategy's returns.

```python
ensemble = StrategyEnsemble(
    strategies=[strategy1, strategy2, strategy3],
    weighting_method=WeightingMethod.VOLATILITY
)
```

### Adaptive Weighting
Combines performance-based weighting with correlation adjustments to improve diversification.

```python
ensemble = StrategyEnsemble(
    strategies=[strategy1, strategy2, strategy3],
    weighting_method=WeightingMethod.ADAPTIVE,
    correlation_threshold=0.7  # Threshold above which to adjust weights
)
```

## Dynamic Strategy Selection

The `DynamicEnsemble` class extends the functionality of `StrategyEnsemble` by dynamically activating or deactivating strategies based on their performance.

```python
dynamic_ensemble = DynamicEnsemble(
    strategies=[strategy1, strategy2, strategy3, strategy4, strategy5],
    weighting_method=WeightingMethod.PERFORMANCE,
    min_active_strategies=2,  # At least 2 strategies must be active
    max_active_strategies=4,  # No more than 4 strategies can be active
    activation_threshold=0.2,  # Sharpe ratio required for activation
    deactivation_threshold=-0.1  # Sharpe ratio below which to deactivate
)
```

## Advanced Features

### Adding and Removing Strategies

```python
# Add a new strategy with optional weight
ensemble.add_strategy(new_strategy, name="New Strategy", weight=0.3)

# Remove a strategy by name
ensemble.remove_strategy("Strategy1")
```

### Changing Weighting Method

```python
# Change weighting method at runtime
ensemble.set_weighting_method(WeightingMethod.VOLATILITY)
```

### Adjusting Individual Weights

```python
# Set weight for a specific strategy
ensemble.set_strategy_weight("Strategy1", 0.4)
```

### Retrieving Current Weights

```python
# Get current strategy weights
weights = ensemble.get_strategy_weights()
print(weights)
```

## Implementation Details

The ensemble strategy implements the following methods:

1. **generate_signals**: Combines signals from component strategies using the current weighting scheme
2. **calculate_position_size**: Calculates position sizes based on combined signals
3. **_update_weights**: Updates weights based on the selected weighting method
4. **_calculate_strategy_returns**: Calculates return series for each strategy for performance evaluation

For the `DynamicEnsemble`, additional methods include:

1. **_update_active_strategies**: Updates the list of active strategies based on performance
2. **get_active_strategies**: Returns the list of currently active strategies

## Benefits of Using the Ensemble Framework

1. **Robustness**: Combining multiple strategies can reduce the impact of individual strategy failure
2. **Diversification**: Using strategies with different approaches helps capture opportunities in various market conditions
3. **Reduced Overfitting**: The ensemble approach can help reduce overfitting risk associated with individual strategies
4. **Adaptability**: Dynamic ensembles can adapt to changing market conditions by adjusting weights or active strategies

## Testing

A comprehensive testing script is available at `trading_bot/strategy_ensemble_test.py` which demonstrates the usage of various ensemble approaches and compares their performance.

To run the tests:

```bash
python -m trading_bot.strategy_ensemble_test
```

## Future Enhancements

1. Machine learning-based weight allocation
2. Multi-timeframe ensembles 
3. Return attribution analysis for ensemble components
4. Optimization of ensemble parameters
5. Integration with market regime detection for more sophisticated regime-based allocation 