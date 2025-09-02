# BensBot-EvoTrader Integration

This package provides a bridge to integrate EvoTrader's evolutionary trading strategy system with BensBot. The integration enables autonomous strategy generation, variant mutation and evolution, generational memory tracking, and performance-based strategy promotion.

## Features

- **Strategy Adaptation**: Connect BensBot strategies to EvoTrader's evolutionary framework
- **Autonomous Strategy Generation**: Create new strategies through mutation and crossover
- **Evolution Engine**: Battle-test strategies and evolve them over generations
- **Generational Memory**: Track and log strategy performance across generations
- **Strategy Promotion**: Promote strategies to production based on statistical performance

## Integration Architecture

```
BensBot <-> EvoTrader Bridge <-> EvoTrader Core
```

The bridge components include:
- `strategy_adapter.py`: Adapts BensBot strategies to EvoTrader framework
- `evolution_manager.py`: Handles strategy evolution and generational tracking
- `performance_tracker.py`: Stores performance metrics in a persistent database
- `testing_framework.py`: Provides simulation environment for strategy testing
- `ab_testing.py`: Implements A/B testing for comparing strategies
- `main.py`: Main interface for using the bridge

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Adapt a BensBot strategy:
```python
from benbot.evotrader_bridge.main import EvoTraderBridge

# Create bridge
bridge = EvoTraderBridge()

# Adapt existing BensBot strategy
benbot_strategy = MyBensBotStrategy()
adapted_strategy = bridge.adapt_benbot_strategy(benbot_strategy)

# Register with evolution system
bridge.register_strategy(adapted_strategy)
```

3. Evolve strategies:
```python
# Evolve for 5 generations
evolution_stats = bridge.evolve_strategies(generations=5)
```

4. Get best strategies:
```python
# Get top 3 strategies from latest generation
best_strategies = bridge.get_best_strategies(count=3)
```

5. Promote strategies to production:
```python
# Compare best evolved strategy against original
original = bridge.get_best_strategies(generation=0)[0]
evolved = bridge.get_best_strategies()[0]

comparison = bridge.compare_strategies(original, evolved)

# Check if it meets promotion criteria
if bridge.should_promote_strategy(comparison):
    bridge.promote_strategy(evolved)
```

## Command-Line Interface

The bridge also provides a command-line interface:

```bash
# Evolve strategies for 5 generations
python -m benbot.evotrader_bridge.main --evolve 5

# Evaluate current strategies
python -m benbot.evotrader_bridge.main --evaluate

# Promote best strategies if they meet criteria
python -m benbot.evotrader_bridge.main --promote
```

## Configuration

The bridge can be configured through a JSON file:

```bash
python -m benbot.evotrader_bridge.main --config config.json
```

Example configuration file:
```json
{
  "output_dir": "evolution_results",
  "evolution": {
    "selection_percentage": 0.3,
    "mutation_rate": 0.2,
    "crossover_rate": 0.3,
    "population_size": 100
  },
  "simulation": {
    "data_source": "historical",
    "symbols": ["BTC/USD", "ETH/USD"],
    "timeframe": "1h"
  },
  "ab_testing": {
    "test_count": 10,
    "significance_level": 0.05
  },
  "strategy_promotion": {
    "min_improvement": 5.0,
    "required_significance": true
  }
}
```

## Integration with BensBot

To fully integrate with BensBot, update your main trading system to load promoted strategies:

```python
from benbot.evotrader_bridge.main import EvoTraderBridge

def load_promoted_strategies():
    bridge = EvoTraderBridge()
    promotion_dir = os.path.join(bridge.output_dir, "promoted_strategies")
    
    promoted_strategies = []
    if os.path.exists(promotion_dir):
        for file in os.listdir(promotion_dir):
            if file.endswith(".json") and not file.endswith("_metadata.json"):
                strategy_path = os.path.join(promotion_dir, file)
                strategy = BensBotStrategyAdapter.load_from_file(strategy_path)
                if strategy and strategy.benbot_strategy:
                    promoted_strategies.append(strategy.benbot_strategy)
    
    return promoted_strategies
```

This function can be called from your BensBot initialization code to dynamically load strategies that have been promoted through the evolutionary process.
