#!/usr/bin/env python3
"""
Evolved Trading Strategy: BollingerBandsStrategy
Strategy ID: 520c8f05253c629b4583a7bb3c68f784
Generated: 2025-04-30 02:44:35

Performance Metrics:
- Average Return: 69.53%
- Win Rate: 43.3%
- Max Drawdown: 0.06%
- Robustness Score: 0.699
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class EvolvedBollingerBands:
    """
    Evolved trading strategy based on BollingerBandsStrategy.
    This strategy was optimized through evolutionary algorithms against diverse market conditions.
    """
    
    def __init__(self):
        """Initialize the evolved strategy with optimized parameters"""
        self.name = "EvolvedBollingerBands"
        self.parameters = {
        "period": 19,
        "std_dev": 1.6713398621868458,
        "signal_threshold": 0.051897635673370884
}
        
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on Bollinger Bands.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["period"]:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate Bollinger Bands
        rolling_mean = close_prices.rolling(window=self.parameters["period"]).mean()
        rolling_std = close_prices.rolling(window=self.parameters["period"]).std()
        
        upper_band = rolling_mean + (rolling_std * self.parameters["std_dev"])
        lower_band = rolling_mean - (rolling_std * self.parameters["std_dev"])
        
        # Check for NaN values at the end
        if pd.isna(rolling_mean.iloc[-1]) or pd.isna(rolling_std.iloc[-1]):
            return {"signal": "none", "confidence": 0}
        
        # Get current values
        current_price = float(close_prices.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_middle = float(rolling_mean.iloc[-1])
        
        # Calculate band width (volatility measure)
        band_width = (current_upper - current_lower) / current_middle
        
        # Calculate how far price is into the band (normalized position)
        if current_upper != current_lower:  # Avoid division by zero
            normalized_position = (current_price - current_lower) / (current_upper - current_lower)
        else:
            normalized_position = 0.5
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Calculate thresholds
        lower_threshold = current_lower + (band_width * self.parameters["signal_threshold"])
        upper_threshold = current_upper - (band_width * self.parameters["signal_threshold"])
        
        # Generate signal logic
        if current_price <= lower_threshold:
            # Price at or below lower band
            signal = "buy"
            # Higher confidence when price is further below the band
            confidence = min(1.0, (lower_threshold - current_price) / current_lower * 5 + 0.5)
            if confidence < 0:
                confidence = 0.5  # Ensure positive confidence
        elif current_price >= upper_threshold:
            # Price at or above upper band
            signal = "sell"
            # Higher confidence when price is further above the band
            confidence = min(1.0, (current_price - upper_threshold) / current_upper * 5 + 0.5)
            if confidence < 0:
                confidence = 0.5  # Ensure positive confidence
                
        # Add trend filter
        if signal == "buy" and len(rolling_mean) >= 3:
            # Check if moving average is falling
            if rolling_mean.iloc[-1] < rolling_mean.iloc[-3]:
                confidence *= 0.7  # Reduce confidence in counter-trend signal
        elif signal == "sell" and len(rolling_mean) >= 3:
            # Check if moving average is rising
            if rolling_mean.iloc[-1] > rolling_mean.iloc[-3]:
                confidence *= 0.7  # Reduce confidence in counter-trend signal
        
        return {
            "signal": signal,
            "confidence": confidence,
            "current_price": current_price,
            "upper_band": current_upper,
            "lower_band": current_lower,
            "middle_band": current_middle,
            "band_width": band_width,
            "normalized_position": normalized_position
        }


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt
    
    # Get some test data
    symbol = "SPY"
    data = yf.download(symbol, start="2022-01-01")
    
    # Initialize strategy
    strategy = EvolvedBollingerBands()
    
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
