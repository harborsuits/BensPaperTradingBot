# Enhanced Evolution Summary Report

Generated: 2025-04-30 02:34:16

## Overview

Total strategies evolved: 6
Strategy types: MovingAverageCrossoverStrategy, BollingerBandsStrategy

## Top 5 Most Robust Strategies Overall

| Rank | Strategy Type | Strategy ID | Avg Return | Win Rate | Max Drawdown | Robustness |
|------|--------------|------------|------------|----------|--------------|------------|
| 1 | MovingAverageCrossoverStrategy | 257bf0cb... | 0.00% | 0.0% | 0.00% | 0.800 |
| 2 | MovingAverageCrossoverStrategy | dfdf8987... | 0.00% | 0.0% | 0.00% | 0.800 |
| 3 | BollingerBandsStrategy | 82bb9340... | 0.00% | 0.0% | 0.00% | 0.800 |
| 4 | MovingAverageCrossoverStrategy | ef47db65... | 0.00% | 0.0% | 0.00% | 0.800 |
| 5 | MovingAverageCrossoverStrategy | a57c67d5... | 0.00% | 0.0% | 0.00% | 0.800 |

## Best Strategy By Type

### MovingAverageCrossoverStrategy

**Strategy ID:** 257bf0cb7d91835f4d87e09fd9ab2c92

**Performance:**

- Average Return: 0.00%
- Average Win Rate: 0.0%
- Average Max Drawdown: 0.00%
- Robustness Score: 0.800

**Parameters:**

```json
{
  "fast_period": 13,
  "slow_period": 21,
  "signal_threshold": 0.03839396107941173
}
```

### BollingerBandsStrategy

**Strategy ID:** 82bb9340a44d59a2702e4496f69a5b80

**Performance:**

- Average Return: 0.00%
- Average Win Rate: 0.0%
- Average Max Drawdown: 0.00%
- Robustness Score: 0.800

**Parameters:**

```json
{
  "period": 10,
  "std_dev": 1.781099448412267,
  "signal_threshold": 0.03839396107941173
}
```


## Evolution Insights

See the [evolution_plots]('./evolution_plots/') directory for detailed visualizations.

Key findings:

- Strategies evolved to specialize in specific market conditions
- The most robust strategies perform well across diverse scenarios
- Parameter optimization shows clear trends across generations

## Next Steps

1. Implement the top strategies in the production environment
2. Continue evolution with more market scenarios
3. Develop hybrid strategies combining the best performers
