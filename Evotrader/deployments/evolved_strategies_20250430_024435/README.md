# Evolved Trading Strategies

Deployed on: 2025-04-30 02:44:35

Total strategies: 8

## Strategy Files

### RSIStrategy

- [evolved_rsi_1.py](evolved_rsi_1.py): Robustness: 0.914, Return: 142.48%, Win Rate: 69.4%
- [evolved_rsi_2.py](evolved_rsi_2.py): Robustness: 0.901, Return: 137.53%, Win Rate: 70.6%
- [evolved_rsi_3.py](evolved_rsi_3.py): Robustness: 0.875, Return: 128.71%, Win Rate: 71.4%

### MovingAverageCrossoverStrategy

- [evolved_movingaveragecrossover_1.py](evolved_movingaveragecrossover_1.py): Robustness: 0.977, Return: 161.46%, Win Rate: 70.4%
- [evolved_movingaveragecrossover_2.py](evolved_movingaveragecrossover_2.py): Robustness: 0.968, Return: 151.22%, Win Rate: 68.7%
- [evolved_movingaveragecrossover_3.py](evolved_movingaveragecrossover_3.py): Robustness: 0.966, Return: 145.25%, Win Rate: 67.8%

### BollingerBandsStrategy

- [evolved_bollingerbands_1.py](evolved_bollingerbands_1.py): Robustness: 0.699, Return: 69.53%, Win Rate: 43.3%
- [evolved_bollingerbands_2.py](evolved_bollingerbands_2.py): Robustness: 0.699, Return: 65.26%, Win Rate: 42.2%

## Usage Instructions

Each strategy file is self-contained and can be used independently:

```python
from evolved_strategy_file import EvolvedStrategyClass

# Initialize strategy
strategy = EvolvedStrategyClass()

# Calculate signal from market data
signal = strategy.calculate_signal(market_data)

# Use signal information
if signal['signal'] == 'buy':
    # Execute buy order
    pass
elif signal['signal'] == 'sell':
    # Execute sell order
    pass
```

## Integration

These strategies can be integrated into the Evotrader platform by:

1. Copying the strategy files to your project directory
2. Importing the strategy classes where needed
3. Using the `calculate_signal` method to generate trading signals
