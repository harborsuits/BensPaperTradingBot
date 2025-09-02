# Parameter Optimization Framework

A comprehensive framework for optimizing trading strategy parameters across different market regimes.

## Features

- **Regime-Aware Optimization**: Optimize parameters across different market regimes (bull, bear, high volatility, etc.)
- **Multiple Search Methods**: Grid search, random search, Bayesian optimization, and genetic algorithms
- **Walk-Forward Testing**: Prevent overfitting with walk-forward validation
- **Advanced Metrics**: Evaluate parameters based on Sharpe ratio, Sortino ratio, regime stability, and more
- **Parallel Processing**: Accelerate optimization with multi-core processing
- **Comprehensive Reporting**: Generate detailed visualizations and reports

## Usage

### 1. Define Parameter Space

```python
from trading_bot.optimization import ParameterSpace

# Define parameter space
param_space = ParameterSpace()
param_space.add_integer_parameter("short_ma", 5, 50, 20, "Short moving average window")
param_space.add_integer_parameter("long_ma", 20, 200, 50, "Long moving average window")
param_space.add_range_parameter("stop_loss", 0.01, 0.1, 0.05, "Stop loss percentage")
param_space.add_boolean_parameter("use_trailing_stop", False, "Whether to use trailing stop")
```

### 2. Choose Search Method

```python
from trading_bot.optimization import GridSearch, RandomSearch, BayesianOptimization, GeneticAlgorithm

# Grid search
search_method = GridSearch(param_space, num_points=5)

# Random search
# search_method = RandomSearch(param_space, num_iterations=100)

# Bayesian optimization
# search_method = BayesianOptimization(param_space, num_iterations=50)

# Genetic algorithm
# search_method = GeneticAlgorithm(param_space, population_size=20, generations=5)
```

### 3. Create Optimizer

```python
from trading_bot.optimization import ParameterOptimizer, OptimizationMetric, RegimeWeight, WalkForwardMethod

optimizer = ParameterOptimizer(
    parameter_space=param_space,
    search_method=search_method,
    objective_metric=OptimizationMetric.SHARPE_RATIO,
    weight_strategy=RegimeWeight.EQUAL,
    is_maximizing=True,
    use_walk_forward=True,
    walk_forward_method=WalkForwardMethod.ROLLING,
    train_size=200,
    test_size=50,
    output_dir="optimization_results",
    verbose=True
)
```

### 4. Set Up Regime Detection

```python
from trading_bot.backtesting.order_book_simulator import MarketRegimeDetector

# Create regime detector
regime_detector = MarketRegimeDetector()
optimizer.set_regime_detector(regime_detector)
```

### 5. Define Strategy Evaluator

```python
def evaluate_strategy(params, indices):
    """
    Evaluate strategy with given parameters
    
    Args:
        params: Parameter values
        indices: Data indices to use
        
    Returns:
        Dictionary of performance metrics
    """
    # Extract parameters
    short_ma = params["short_ma"]
    long_ma = params["long_ma"]
    stop_loss = params["stop_loss"]
    use_trailing_stop = params["use_trailing_stop"]
    
    # Implement strategy logic using the parameters
    # ...
    
    # Return metrics
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown
    }
```

### 6. Run Optimization

```python
import pandas as pd

# Load price data
prices = pd.read_csv("prices.csv")["close"]

# Run optimization
result = optimizer.optimize(
    strategy_evaluator=evaluate_strategy,
    prices=prices,
    max_evaluations=100
)

# Print results
print(f"Best parameters: {result['best_params']}")
print(f"Best objective: {result['best_objective']}")
```

### 7. Visualize Results

```python
# Plot walk-forward results
optimizer.plot_walk_forward_results(result)

# Plot regime performance
optimizer.plot_regime_performance(result)
```

## Advanced Usage

### Customizing Regime Weights

You can customize how different market regimes are weighted in the optimization process:

```python
from trading_bot.backtesting.order_book_simulator import MarketRegime

# Custom regime weights
regime_weights = {
    MarketRegime.BULL: 0.3,    # Prioritize bull markets
    MarketRegime.BEAR: 0.2,    # Lower weight for bear markets
    MarketRegime.HIGH_VOL: 0.3, # Prioritize high volatility
    MarketRegime.LOW_VOL: 0.1,  # Lower weight for low volatility
    MarketRegime.SIDEWAYS: 0.1  # Lower weight for sideways markets
}

optimizer = ParameterOptimizer(
    # ... other parameters
    regime_weights=regime_weights,
    weight_strategy=RegimeWeight.CUSTOM
)
```

### Custom Objective Metrics

You can define custom optimization objectives beyond the built-in metrics:

```python
from trading_bot.optimization import OptimizationMetric

# Define a custom profit-to-loss ratio objective
def custom_pl_ratio(strategy_results):
    total_profit = sum([r for r in strategy_results if r > 0])
    total_loss = abs(sum([r for r in strategy_results if r < 0]))
    return total_profit / total_loss if total_loss > 0 else 0

# Use the custom metric
optimizer = ParameterOptimizer(
    # ... other parameters
    objective_metric=OptimizationMetric.CUSTOM,
    custom_objective_func=custom_pl_ratio
)
```

## Examples

See the `examples.py` file for complete usage examples, including:

1. Moving average crossover optimization
2. Volatility-based position sizing optimization

## Integration with BacktestCircuitBreakerManager

The framework integrates with the existing risk management components:

```python
from trading_bot.backtesting.order_book_simulator import (
    OrderBookSimulator, BacktestCircuitBreakerManager, CircuitBreaker
)

# Set up simulator and risk manager
simulator = OrderBookSimulator(symbols=["SPY"])
circuit_breaker = CircuitBreaker()
risk_manager = BacktestCircuitBreakerManager(simulator, circuit_breaker)

# Use risk manager in strategy evaluation
def evaluate_with_risk_management(params, indices):
    # Set up risk manager with parameters
    circuit_breaker.drawdown_thresholds = {
        CircuitBreakerLevel.LEVEL_1: params["cb_threshold_1"],
        CircuitBreakerLevel.LEVEL_2: params["cb_threshold_2"],
        CircuitBreakerLevel.LEVEL_3: params["cb_threshold_3"]
    }
    
    # Run backtest with risk management
    # ...
    
    return metrics
``` 