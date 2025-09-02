# EvoTrader Evolution Demonstration Report

**Generated:** 2025-04-30 01:42:04

## Evolution Results

### Strategy Distribution

| Generation | BollingerBands | IronCondor | MovingAverageCrossover | RSIStrategy | VerticalSpread |
| --- | --- | --- | --- | --- | --- |
| 0 | 10 | 2 | 3 | 3 | 2 |
| 1 | 14 | 1 | 1 | 3 | 1 |
| 2 | 17 | 0 | 1 | 2 | 0 |
| 3 | 18 | 0 | 1 | 1 | 0 |
| 4 | 20 | 0 | 0 | 0 | 0 |
| 5 | 20 | 0 | 0 | 0 | 0 |
| 6 | 20 | 0 | 0 | 0 | 0 |
| 7 | 20 | 0 | 0 | 0 | 0 |
| 8 | 20 | 0 | 0 | 0 | 0 |
| 9 | 20 | 0 | 0 | 0 | 0 |
| 10 | 20 | 0 | 0 | 0 | 0 |

### Fitness Progression

| Generation | Average Fitness | Maximum Fitness |
| --- | --- | --- |
| 0 | 0.5816 | 0.8516 |
| 1 | 0.5793 | 0.8356 |
| 2 | 0.5924 | 0.8680 |
| 3 | 0.6285 | 0.9061 |
| 4 | 0.6505 | 0.8348 |
| 5 | 0.6320 | 0.8779 |
| 6 | 0.7532 | 0.9500 |
| 7 | 0.7581 | 0.9500 |
| 8 | 0.7732 | 0.9500 |
| 9 | 0.7358 | 0.9500 |
| 10 | 0.7851 | 0.9500 |

### Best Strategies by Generation

#### Generation 0: RSIStrategy

- **ID:** RSIStrategy_36169
- **Fitness:** 0.8516
- **Parameters:**
  - rsi_period: 16
  - overbought: 76
  - oversold: 25
- **Performance Metrics:**
  - fitness: 0.85
  - profit: 85.16
  - win_rate: 0.73
  - drawdown: 4.45

#### Generation 1: BollingerBands

- **ID:** BollingerBands_49681
- **Fitness:** 0.8356
- **Parameters:**
  - period: 33
  - std_dev: 2.1240400897608644
  - signal_threshold: 0.09609482276095486
- **Performance Metrics:**
  - fitness: 0.84
  - profit: 83.56
  - win_rate: 0.72
  - drawdown: 4.93

#### Generation 2: BollingerBands

- **ID:** BollingerBands_95309
- **Fitness:** 0.8680
- **Parameters:**
  - period: 33
  - std_dev: 2.1240400897608644
  - signal_threshold: 0.09609482276095486
- **Performance Metrics:**
  - fitness: 0.87
  - profit: 86.80
  - win_rate: 0.73
  - drawdown: 3.96

#### Generation 3: BollingerBands

- **ID:** BollingerBands_94978
- **Fitness:** 0.9061
- **Parameters:**
  - period: 19
  - std_dev: 2.1240400897608644
  - signal_threshold: 0.061471349077509736
- **Performance Metrics:**
  - fitness: 0.91
  - profit: 90.61
  - win_rate: 0.75
  - drawdown: 2.82

#### Generation 4: BollingerBands

- **ID:** BollingerBands_70925
- **Fitness:** 0.8348
- **Parameters:**
  - period: 19
  - std_dev: 2.720299494966625
  - signal_threshold: 0.061471349077509736
- **Performance Metrics:**
  - fitness: 0.83
  - profit: 83.48
  - win_rate: 0.72
  - drawdown: 4.96

#### Generation 5: BollingerBands

- **ID:** BollingerBands_65210
- **Fitness:** 0.8779
- **Parameters:**
  - period: 18
  - std_dev: 2.1240400897608644
  - signal_threshold: 0.026891379210606643
- **Performance Metrics:**
  - fitness: 0.88
  - profit: 87.79
  - win_rate: 0.74
  - drawdown: 3.66

#### Generation 6: BollingerBands

- **ID:** BollingerBands_80211
- **Fitness:** 0.9500
- **Parameters:**
  - period: 18
  - std_dev: 2.1240400897608644
  - signal_threshold: 0.026891379210606643
- **Performance Metrics:**
  - fitness: 0.95
  - profit: 95.00
  - win_rate: 0.77
  - drawdown: 1.50

#### Generation 7: BollingerBands

- **ID:** BollingerBands_50557
- **Fitness:** 0.9500
- **Parameters:**
  - period: 16
  - std_dev: 2.720299494966625
  - signal_threshold: 0.061471349077509736
- **Performance Metrics:**
  - fitness: 0.95
  - profit: 95.00
  - win_rate: 0.77
  - drawdown: 1.50

#### Generation 8: BollingerBands

- **ID:** BollingerBands_81364
- **Fitness:** 0.9500
- **Parameters:**
  - period: 24
  - std_dev: 2.2125893717113176
  - signal_threshold: 0.10024461257930477
- **Performance Metrics:**
  - fitness: 0.95
  - profit: 95.00
  - win_rate: 0.77
  - drawdown: 1.50

#### Generation 9: BollingerBands

- **ID:** BollingerBands_45129
- **Fitness:** 0.9500
- **Parameters:**
  - period: 24
  - std_dev: 2.2125893717113176
  - signal_threshold: 0.10024461257930477
- **Performance Metrics:**
  - fitness: 0.95
  - profit: 95.00
  - win_rate: 0.77
  - drawdown: 1.50

#### Generation 10: BollingerBands

- **ID:** BollingerBands_71292
- **Fitness:** 0.9500
- **Parameters:**
  - period: 18
  - std_dev: 2.1240400897608644
  - signal_threshold: 0.02564549886959707
- **Performance Metrics:**
  - fitness: 0.95
  - profit: 95.00
  - win_rate: 0.77
  - drawdown: 1.50

## Evolutionary Insights

- **Fitness Improvement:** 11.55%
- **Dominant Strategy Type:** BollingerBands
- **Parameter Evolution:**
  - **period:** 33, 33, 19, 19, 18, 18, 16, 24, 24, 18
  - **signal_threshold:** 0.0961, 0.0961, 0.0615, 0.0615, 0.0269, 0.0269, 0.0615, 0.1002, 0.1002, 0.0256
  - **std_dev:** 2.1240, 2.1240, 2.1240, 2.7203, 2.1240, 2.1240, 2.7203, 2.2126, 2.2126, 2.1240

## Conclusion

The evolutionary process successfully improved strategy performance over generations.

## Recommended Parameters

Based on the evolution results, the following parameters are recommended for BollingerBands:

```python
parameters = {
    "period": 18,
    "std_dev": 2.1240,
    "signal_threshold": 0.0256,
}
```

This configuration achieved a fitness score of 0.9500 with a profit of 95.00% and a win rate of 0.77%.
