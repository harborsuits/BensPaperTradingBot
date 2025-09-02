#!/usr/bin/env python3
"""
Evolved Trading Strategy: MovingAverageCrossoverStrategy
Strategy ID: 96f239709e56894323e9b6ac3b27db20
Generated: 2025-04-30 02:44:35

Performance Metrics:
- Average Return: 161.46%
- Win Rate: 70.4%
- Max Drawdown: -0.24%
- Robustness Score: 0.977
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class EvolvedMovingAverageCrossover:
    """
    Evolved trading strategy based on MovingAverageCrossoverStrategy.
    This strategy was optimized through evolutionary algorithms against diverse market conditions.
    """
    
    def __init__(self):
        """Initialize the evolved strategy with optimized parameters"""
        self.name = "EvolvedMovingAverageCrossover"
        self.parameters = {
        "fast_period": 6,
        "slow_period": 29,
        "signal_threshold": 0.051897635673370884
}
        
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on moving average crossover.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["slow_period"]:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate fast and slow moving averages
        fast_ma = close_prices.rolling(window=self.parameters["fast_period"]).mean()
        slow_ma = close_prices.rolling(window=self.parameters["slow_period"]).mean()
        
        # Check for NaN values at the end
        if pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
            return {"signal": "none", "confidence": 0}
        
        # Get current values
        current_fast = float(fast_ma.iloc[-1])
        current_slow = float(slow_ma.iloc[-1])
        
        # Get previous values (for crossover detection)
        if len(fast_ma) > 1 and len(slow_ma) > 1:
            prev_fast = float(fast_ma.iloc[-2])
            prev_slow = float(slow_ma.iloc[-2])
        else:
            prev_fast, prev_slow = current_fast, current_slow
        
        # Calculate percentage difference
        diff_pct = abs(current_fast - current_slow) / current_slow
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Detect crossover and check threshold
        if diff_pct >= self.parameters["signal_threshold"]:
            if current_fast > current_slow and prev_fast <= prev_slow:
                # Bullish crossover
                signal = "buy"
                confidence = min(1.0, diff_pct * 10)
            elif current_fast < current_slow and prev_fast >= prev_slow:
                # Bearish crossover
                signal = "sell"
                confidence = min(1.0, diff_pct * 10)
        
        # Check for strong trends (not just crossover)
        elif current_fast > current_slow and diff_pct >= self.parameters["signal_threshold"] / 2:
            signal = "buy"
            confidence = min(0.7, diff_pct * 5)  # Lower confidence for trend following
        elif current_fast < current_slow and diff_pct >= self.parameters["signal_threshold"] / 2:
            signal = "sell"
            confidence = min(0.7, diff_pct * 5)  # Lower confidence for trend following
        
        return {
            "signal": signal,
            "confidence": confidence,
            "fast_ma": current_fast,
            "slow_ma": current_slow,
            "diff_pct": diff_pct
        }


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt
    
    # Get some test data
    symbol = "SPY"
    data = yf.download(symbol, start="2022-01-01")
    
    # Initialize strategy
    strategy = EvolvedMovingAverageCrossover()
    
    # Apply strategy to data
    signals = []
    for i in range(100, len(data)):
        chunk = data.iloc[:i+1]
        signal = strategy.calculate_signal(chunk)
        signals.append(signal["signal"])
    
    # Show results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(signals):], data['Close'][-len(signals):])
    
    # Mark buy signals
    buy_days = [day for i, day in enumerate(data.index[-len(signals):]) if signals[i] == "buy"]
    buy_prices = [data.loc[day, 'Close'] for day in buy_days]
    plt.scatter(buy_days, buy_prices, marker='^', color='green', s=100)
    
    # Mark sell signals
    sell_days = [day for i, day in enumerate(data.index[-len(signals):]) if signals[i] == "sell"]
    sell_prices = [data.loc[day, 'Close'] for day in sell_days]
    plt.scatter(sell_days, sell_prices, marker='v', color='red', s=100)
    
    plt.title(f"{strategy.name} Signals for {symbol}")
    plt.tight_layout()
    plt.show()
