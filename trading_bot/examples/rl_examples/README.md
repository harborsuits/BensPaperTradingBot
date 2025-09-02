# Reinforcement Learning for Trading Bot

This directory contains examples and documentation for using reinforcement learning (RL) strategies in the trading bot system.

## Overview

The reinforcement learning components integrate with the existing `StrategyRotator` to provide adaptive trading strategies that can learn from market data and adjust to changing market conditions.

Key components include:
- RL-based trading strategies (`DQNStrategy`, `PPOStrategy`, and `MetaLearningStrategy`)
- Simulated trading environment for training RL models
- Integration with market regime detection
- Adaptive weight optimization

## Getting Started

### Prerequisites

Make sure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

### Running the Example

To run a basic example of the RL-based trading system:

```bash
python trading_bot/examples/rl_examples/rl_strategy_example.py
```

This example:
1. Generates synthetic market data with different regimes
2. Trains the RL strategies on this data
3. Simulates trading with the traditional and RL strategies
4. Plots the results and shows strategy weights

## Components

### 1. RL Strategies

- **RLStrategy**: Base class for RL strategies, implementing core Q-learning functionality
- **DQNStrategy**: Deep Q-Network implementation for trading
- **PPOStrategy**: Proximal Policy Optimization strategy (placeholder for future implementation)
- **MetaLearningStrategy**: Strategy that adapts to different market regimes (placeholder for future implementation)

### 2. Trading Environment

The `TradingEnvironment` class provides an OpenAI Gym-compatible environment for training RL agents, with:
- Realistic market simulation with price, volume, and trading costs
- Reward function based on profit and loss
- Observation space including price history and account information

### 3. Integration with StrategyRotator

The `StrategyRotator` has been enhanced to:
- Create and manage RL strategies
- Train strategies on historical data
- Adjust strategy weights based on market regimes
- Send market regime events to meta-learning strategies

## Configuration

RL strategies can be configured in the `StrategyRotator` config:

```python
{
    "strategy_configs": {
        "DQNStrategy": {
            "window_size": 20,
            "batch_size": 32,
            "gamma": 0.99,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "eps_decay": 0.995,
            "learning_rate": 0.001,
            "memory_capacity": 10000
        },
        "MetaLearningStrategy": {
            "window_size": 30,
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 0.0005,
            "memory_capacity": 20000
        }
    },
    "enable_rl_strategies": true,
    "rl_training": {
        "enabled": true,
        "train_frequency": 604800,  # 1 week in seconds
        "train_epochs": 100,
        "min_training_data": 1000,
        "save_model_interval": 10
    }
}
```

## Advanced Usage

### Custom RL Strategies

You can create custom RL strategies by extending the `RLStrategy` class:

```python
from trading_bot.strategy.rl_strategy import RLStrategy

class MyCustomRLStrategy(RLStrategy):
    def __init__(self, name, config=None):
        super().__init__(name, config or {})
        # Custom initialization
        
    def _initialize_models(self):
        # Custom model architecture
        pass
        
    # Override other methods as needed
```

### Training on Live Data

To train RL strategies on live market data:

```python
from trading_bot.strategy.strategy_rotator import StrategyRotator
import pandas as pd

# Load market data
market_data = pd.read_csv("market_data.csv")

# Create rotator
rotator = StrategyRotator()

# Train RL strategies
rewards = rotator.train_rl_strategies(market_data)
```

## Future Work

- Implement PPO strategy for continuous action spaces
- Complete meta-learning strategy with regime-specific models
- Add multi-asset portfolio optimization
- Implement hierarchical reinforcement learning
- Add model explainability tools 