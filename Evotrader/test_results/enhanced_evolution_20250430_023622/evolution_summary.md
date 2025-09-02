# Enhanced Evolution Summary Report

Generated: 2025-04-30 02:38:07

## Overview

Total strategies evolved: 25
Strategy types: MovingAverageCrossoverStrategy, RSIStrategy, BollingerBandsStrategy

## Top 5 Most Robust Strategies Overall

| Rank | Strategy Type | Strategy ID | Avg Return | Win Rate | Max Drawdown | Robustness |
|------|--------------|------------|------------|----------|--------------|------------|
| 1 | MovingAverageCrossoverStrategy | 96f23970... | 161.46% | 70.4% | -0.24% | 0.977 |
| 2 | MovingAverageCrossoverStrategy | 3c6c8f26... | 151.22% | 68.7% | 2.08% | 0.968 |
| 3 | MovingAverageCrossoverStrategy | 3f9bb9a8... | 145.25% | 67.8% | 0.82% | 0.966 |
| 4 | MovingAverageCrossoverStrategy | a61e8c39... | 160.16% | 71.6% | -1.19% | 0.965 |
| 5 | MovingAverageCrossoverStrategy | a6faa242... | 147.83% | 68.4% | -1.28% | 0.964 |

## Best Strategy By Type

### MovingAverageCrossoverStrategy

**Strategy ID:** 96f239709e56894323e9b6ac3b27db20

**Performance:**

- Average Return: 161.46%
- Average Win Rate: 70.4%
- Average Max Drawdown: -0.24%
- Robustness Score: 0.977

**Parameters:**

```json
{
  "fast_period": 6,
  "slow_period": 29,
  "signal_threshold": 0.051897635673370884
}
```

### RSIStrategy

**Strategy ID:** f419ac41a27142296082aa49064eee2e

**Performance:**

- Average Return: 142.48%
- Average Win Rate: 69.4%
- Average Max Drawdown: -1.05%
- Robustness Score: 0.914

**Parameters:**

```json
{
  "period": 5,
  "overbought": 65,
  "oversold": 24
}
```

### BollingerBandsStrategy

**Strategy ID:** 520c8f05253c629b4583a7bb3c68f784

**Performance:**

- Average Return: 69.53%
- Average Win Rate: 43.3%
- Average Max Drawdown: 0.06%
- Robustness Score: 0.699

**Parameters:**

```json
{
  "period": 19,
  "std_dev": 1.6713398621868458,
  "signal_threshold": 0.051897635673370884
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
