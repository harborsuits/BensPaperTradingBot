#!/usr/bin/env python3
"""
Evolved Trading Strategy: RSIStrategy
Strategy ID: f419ac41a27142296082aa49064eee2e
Generated: 2025-04-30 02:44:35

Performance Metrics:
- Average Return: 142.48%
- Win Rate: 69.4%
- Max Drawdown: -1.05%
- Robustness Score: 0.914
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class EvolvedRSI:
    """
    Evolved trading strategy based on RSIStrategy.
    This strategy was optimized through evolutionary algorithms against diverse market conditions.
    """
    
    def __init__(self):
        """Initialize the evolved strategy with optimized parameters"""
        self.name = "EvolvedRSI"
        self.parameters = {
        "period": 5,
        "overbought": 65,
        "oversold": 24
}
        
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on Relative Strength Index (RSI).
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["period"] + 1:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Convert to positive values
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.parameters["period"]).mean()
        avg_loss = loss.rolling(window=self.parameters["period"]).mean()
        
        # Check for NaN values at the end
        if pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]):
            return {"signal": "none", "confidence": 0}
        
        # Calculate RSI
        if float(avg_loss.iloc[-1]) == 0:
            rsi = 100
        else:
            rs = float(avg_gain.iloc[-1]) / float(avg_loss.iloc[-1])
            rsi = 100 - (100 / (1 + rs))
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Generate signal logic
        if rsi <= self.parameters["oversold"]:
            # RSI in oversold territory
            signal = "buy"
            # Higher confidence when RSI is lower
            confidence = min(1.0, (self.parameters["oversold"] - rsi) / self.parameters["oversold"] * 2)
        elif rsi >= self.parameters["overbought"]:
            # RSI in overbought territory
            signal = "sell"
            # Higher confidence when RSI is higher
            confidence = min(1.0, (rsi - self.parameters["overbought"]) / (100 - self.parameters["overbought"]) * 2)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "rsi": rsi,
            "overbought_level": self.parameters["overbought"],
            "oversold_level": self.parameters["oversold"]
        }


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt
    
    # Get some test data
    symbol = "SPY"
    data = yf.download(symbol, start="2022-01-01")
    
    # Initialize strategy
    strategy = EvolvedRSI()
    
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
