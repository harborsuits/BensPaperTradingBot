# Strategy System

The BensBot Trading System features a modular, extensible strategy framework that supports a wide range of trading approaches from simple technical indicators to advanced ML-driven models.

## Strategy Architecture

The strategy system is organized around these core components:

### Strategy Interface

All strategies implement a common interface:

```python
class Strategy(ABC):
    """Base strategy interface that all strategies must implement."""
    
    @abstractmethod
    def generate_signals(self, data):
        """Generate trading signals from input data."""
        pass
    
    @abstractmethod
    def get_parameters(self):
        """Return strategy parameters for serialization."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters):
        """Set strategy parameters from deserialized data."""
        pass
```

### Strategy Factory

Strategies are instantiated through a factory pattern for consistency and easy registration:

```python
class StrategyFactory:
    """Factory for creating strategy instances."""
    
    _strategies = {}
    
    @classmethod
    def register(cls, strategy_type, strategy_class):
        """Register a strategy class."""
        cls._strategies[strategy_type] = strategy_class
    
    @classmethod
    def create(cls, strategy_type, **kwargs):
        """Create a strategy instance."""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy = cls._strategies[strategy_type](**kwargs)
        
        # Optionally wrap with notification decorator
        if kwargs.get("enable_notifications", False):
            strategy = NotificationStrategyDecorator(strategy)
        
        return strategy
```

## Included Strategies

The system includes several built-in strategies:

### Technical Indicator Strategies

- **Moving Average Crossover**: Trade on SMA/EMA crossovers
- **RSI Strategy**: Trade based on oversold/overbought conditions
- **MACD Strategy**: Trade on MACD signal line crossovers
- **Bollinger Bands**: Trade on price touching bands
- **Momentum Strategy**: Trade based on momentum indicators

### Advanced Strategies

- **Pairs Trading**: Statistical arbitrage between correlated assets
- **Mean Reversion**: Trade reversions to statistical means
- **Trend Following**: Identify and follow market trends
- **Volatility Breakout**: Trade breakouts of key levels
- **Dual Momentum**: Combine absolute and relative momentum signals

### ML-Enhanced Strategies

- **Random Forest Classifier**: Predict movement based on technical features
- **LSTM Network**: Deep learning for time series prediction
- **Ensemble Strategy**: Combine multiple strategies with voting
- **Sentiment Analysis**: Trade based on news sentiment

## Strategy Configuration

Strategies are configured through the typed settings system:

```python
class StrategySettings(BaseModel):
    """Strategy configuration."""
    default_strategies: List[str] = Field(default_factory=lambda: ["momentum", "macd"])
    strategy_weights: Dict[str, float] = Field(default_factory=dict)
    strategy_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    enable_strategy_rotation: bool = True
    rotation_frequency: str = "weekly"  # daily, weekly, monthly
    rotation_criteria: str = "sharpe"  # sharpe, returns, drawdown
    lookback_periods: int = 90  # days for rotation analysis
```

## Strategy Rotation

The system supports automated strategy rotation:

1. **Selection Pool**: Define a set of candidate strategies
2. **Performance Metrics**: Rank strategies on Sharpe ratio, returns, or other metrics
3. **Allocation Weights**: Dynamically adjust allocations based on performance
4. **Rebalance Timing**: Configure when to reevaluate and rotate strategies

Example rotation configuration:

```yaml
strategy:
  enable_strategy_rotation: true
  rotation_frequency: "weekly"
  rotation_criteria: "sharpe"
  lookback_periods: 90
  default_strategies:
    - momentum
    - macd
    - rsi
    - bb_reversal
    - trend_following
  strategy_weights:
    momentum: 0.2
    macd: 0.2
    rsi: 0.2
    bb_reversal: 0.2
    trend_following: 0.2
```

## Strategy Development

### Creating New Strategies

To create a new strategy:

1. Create a new Python file in `trading_bot/strategies/`
2. Implement the `Strategy` interface
3. Register the strategy with the factory

Example:

```python
from trading_bot.strategies.strategy_base import Strategy
from trading_bot.strategies.strategy_factory import StrategyFactory

class MyCustomStrategy(Strategy):
    """Custom trading strategy."""
    
    def __init__(self, parameter1=default1, parameter2=default2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    def generate_signals(self, data):
        # Strategy logic here
        signals = {}
        # ...
        return signals
    
    def get_parameters(self):
        return {
            "parameter1": self.parameter1,
            "parameter2": self.parameter2
        }
    
    def set_parameters(self, parameters):
        self.parameter1 = parameters.get("parameter1", self.parameter1)
        self.parameter2 = parameters.get("parameter2", self.parameter2)

# Register the strategy
StrategyFactory.register("my_custom_strategy", MyCustomStrategy)
```

### Strategy Testing

All strategies should be validated through:

1. **Unit Tests**: Test signal generation logic
2. **Backtests**: Evaluate performance on historical data
3. **Forward Testing**: Run in paper trading mode before live deployment
