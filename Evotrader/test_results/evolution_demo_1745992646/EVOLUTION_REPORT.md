# EvoTrader Evolution Demonstration Report

**Generated:** 2025-04-30 01:57:26

## Evolution Results

### Strategy Distribution

| Generation | BollingerBands | IronCondor | MovingAverageCrossover | RSIStrategy | VerticalSpread |
| --- | --- | --- | --- | --- | --- |
| 0 | 6 | 1 | 4 | 4 | 5 |
| 1 | 6 | 1 | 5 | 3 | 5 |
| 2 | 14 | 0 | 3 | 1 | 2 |
| 3 | 12 | 0 | 7 | 0 | 1 |
| 4 | 10 | 0 | 10 | 0 | 0 |
| 5 | 8 | 0 | 12 | 0 | 0 |
| 6 | 7 | 0 | 13 | 0 | 0 |
| 7 | 6 | 0 | 14 | 0 | 0 |
| 8 | 10 | 0 | 10 | 0 | 0 |

### Fitness Progression

| Generation | Average Fitness | Maximum Fitness |
| --- | --- | --- |
| 0 | 0.5482 | 0.8546 |
| 1 | 0.5630 | 0.7323 |
| 2 | 0.5398 | 0.7873 |
| 3 | 0.6514 | 0.8665 |
| 4 | 0.6492 | 0.8808 |
| 5 | 0.6478 | 0.8978 |
| 6 | 0.6617 | 0.8860 |
| 7 | 0.6587 | 0.9316 |
| 8 | 0.6966 | 0.9325 |

### Best Strategies by Generation

#### Generation 0: RSIStrategy

- **ID:** RSIStrategy_43665
- **Fitness:** 0.8546
- **Parameters:**
  - rsi_period: 13
  - overbought: 67
  - oversold: 29
- **Performance Metrics:**
  - fitness: 0.85
  - profit: 85.46
  - win_rate: 0.73
  - drawdown: 4.36

#### Generation 1: MovingAverageCrossover

- **ID:** MovingAverageCrossover_81148
- **Fitness:** 0.7323
- **Parameters:**
  - fast_period: 18
  - slow_period: 159
  - signal_threshold: 0.036732508685630647
- **Performance Metrics:**
  - fitness: 0.73
  - profit: 73.23
  - win_rate: 0.67
  - drawdown: 8.03

#### Generation 2: BollingerBands

- **ID:** BollingerBands_76394
- **Fitness:** 0.7873
- **Parameters:**
  - period: 32
  - std_dev: 2.346877408258103
  - signal_threshold: 0.07072247289613334
- **Performance Metrics:**
  - fitness: 0.79
  - profit: 78.73
  - win_rate: 0.69
  - drawdown: 6.38

#### Generation 3: MovingAverageCrossover

- **ID:** MovingAverageCrossover_42487
- **Fitness:** 0.8665
- **Parameters:**
  - fast_period: 18
  - slow_period: 159
  - signal_threshold: 0.036732508685630647
- **Performance Metrics:**
  - fitness: 0.87
  - profit: 86.65
  - win_rate: 0.73
  - drawdown: 4.00

#### Generation 4: BollingerBands

- **ID:** BollingerBands_26534
- **Fitness:** 0.8808
- **Parameters:**
  - period: 32
  - std_dev: 2.346877408258103
  - signal_threshold: 0.07072247289613334
- **Performance Metrics:**
  - fitness: 0.88
  - profit: 88.08
  - win_rate: 0.74
  - drawdown: 3.57

#### Generation 5: MovingAverageCrossover

- **ID:** MovingAverageCrossover_93007
- **Fitness:** 0.8978
- **Parameters:**
  - fast_period: 18
  - slow_period: 159
  - signal_threshold: 0.036732508685630647
- **Performance Metrics:**
  - fitness: 0.90
  - profit: 89.78
  - win_rate: 0.75
  - drawdown: 3.06

#### Generation 6: MovingAverageCrossover

- **ID:** MovingAverageCrossover_21764
- **Fitness:** 0.8860
- **Parameters:**
  - fast_period: 18
  - slow_period: 159
  - signal_threshold: 0.036732508685630647
- **Performance Metrics:**
  - fitness: 0.89
  - profit: 88.60
  - win_rate: 0.74
  - drawdown: 3.42

#### Generation 7: MovingAverageCrossover

- **ID:** MovingAverageCrossover_94094
- **Fitness:** 0.9316
- **Parameters:**
  - fast_period: 18
  - slow_period: 159
  - signal_threshold: 0.036732508685630647
- **Performance Metrics:**
  - fitness: 0.93
  - profit: 93.16
  - win_rate: 0.77
  - drawdown: 2.05

#### Generation 8: MovingAverageCrossover

- **ID:** MovingAverageCrossover_23142
- **Fitness:** 0.9325
- **Parameters:**
  - fast_period: 16
  - slow_period: 159
  - signal_threshold: 0.037545440475656784
- **Performance Metrics:**
  - fitness: 0.93
  - profit: 93.25
  - win_rate: 0.77
  - drawdown: 2.03

## Evolutionary Insights

- **Fitness Improvement:** 9.11%
- **Dominant Strategy Type:** MovingAverageCrossover
- **Parameter Evolution:**
  - **fast_period:** 18, 18, 18, 18, 18, 16
  - **signal_threshold:** 0.0367, 0.0367, 0.0367, 0.0367, 0.0367, 0.0375
  - **slow_period:** 159, 159, 159, 159, 159, 159

## Conclusion

The evolutionary process successfully improved strategy performance over generations.

## Recommended Parameters

Based on the evolution results, the following parameters are recommended for MovingAverageCrossover:

```python
parameters = {
    "fast_period": 16,
    "slow_period": 159,
    "signal_threshold": 0.0375,
}
```

This configuration achieved a fitness score of 0.9325 with a profit of 93.25% and a win rate of 0.77%.
