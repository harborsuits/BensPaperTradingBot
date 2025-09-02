"""
Test strategies for EvoTrader integration with BensBot.

This module provides sample strategy implementations for testing the evolutionary
process, including basic directional strategies as well as option strategies like
vertical spreads and iron condors.
"""

# Add EvoTrader to Python path
import evotrader_path

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


class StrategyBase:
    """Base class for all test strategies."""
    
    def __init__(self, strategy_id: str = None, parameters: Dict[str, Any] = None):
        """
        Initialize the strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy
            parameters: Strategy parameters dictionary
        """
        self.strategy_id = strategy_id or f"strategy_{random.randint(1000, 9999)}"
        self.parameters = parameters or {}
        self.initialize()
    
    def initialize(self):
        """Initialize strategy with default parameters if needed."""
        pass
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trading signal from market data.
        
        Args:
            market_data: Current market data snapshot
            
        Returns:
            Signal dictionary
        """
        raise NotImplementedError("Subclasses must implement calculate_signal")


class MovingAverageCrossover(StrategyBase):
    """Moving average crossover strategy."""
    
    def initialize(self):
        """Initialize strategy with default parameters."""
        self.parameters.setdefault("fast_period", 10)
        self.parameters.setdefault("slow_period", 30)
        self.parameters.setdefault("symbol", "BTC/USD")
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal based on moving average crossover."""
        symbol = self.parameters["symbol"]
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        if symbol not in market_data:
            return {"signal": "none"}
        
        # Get price history
        symbol_data = market_data[symbol]
        prices = symbol_data.get("close_history", [])
        
        if len(prices) < slow_period:
            return {"signal": "none"}
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-fast_period:])
        slow_ma = np.mean(prices[-slow_period:])
        
        # Previous moving averages
        prev_prices = prices[:-1]
        prev_fast_ma = np.mean(prev_prices[-fast_period:]) if len(prev_prices) >= fast_period else None
        prev_slow_ma = np.mean(prev_prices[-slow_period:]) if len(prev_prices) >= slow_period else None
        
        # Determine signal
        signal = "none"
        
        if prev_fast_ma is not None and prev_slow_ma is not None:
            # Check if we have a crossover
            if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
                signal = "buy"
            elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
                signal = "sell"
        
        return {
            "signal": signal,
            "symbol": symbol,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma
        }


class RSIStrategy(StrategyBase):
    """Relative Strength Index (RSI) strategy."""
    
    def initialize(self):
        """Initialize strategy with default parameters."""
        self.parameters.setdefault("rsi_period", 14)
        self.parameters.setdefault("overbought", 70)
        self.parameters.setdefault("oversold", 30)
        self.parameters.setdefault("symbol", "BTC/USD")
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal based on RSI indicator."""
        symbol = self.parameters["symbol"]
        rsi_period = self.parameters["rsi_period"]
        overbought = self.parameters["overbought"]
        oversold = self.parameters["oversold"]
        
        if symbol not in market_data:
            return {"signal": "none"}
        
        # Get price history
        symbol_data = market_data[symbol]
        prices = symbol_data.get("close_history", [])
        
        if len(prices) < rsi_period + 1:
            return {"signal": "none"}
        
        # Calculate RSI
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, change) for change in price_changes]
        losses = [max(0, -change) for change in price_changes]
        
        avg_gain = np.mean(gains[-rsi_period:])
        avg_loss = np.mean(losses[-rsi_period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Determine signal
        signal = "none"
        
        if rsi < oversold:
            signal = "buy"
        elif rsi > overbought:
            signal = "sell"
        
        return {
            "signal": signal,
            "symbol": symbol,
            "rsi": rsi
        }


class BollingerBands(StrategyBase):
    """Bollinger Bands strategy."""
    
    def initialize(self):
        """Initialize strategy with default parameters."""
        self.parameters.setdefault("bb_period", 20)
        self.parameters.setdefault("bb_std", 2.0)
        self.parameters.setdefault("symbol", "BTC/USD")
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal based on Bollinger Bands."""
        symbol = self.parameters["symbol"]
        bb_period = self.parameters["bb_period"]
        bb_std = self.parameters["bb_std"]
        
        if symbol not in market_data:
            return {"signal": "none"}
        
        # Get price history
        symbol_data = market_data[symbol]
        prices = symbol_data.get("close_history", [])
        
        if len(prices) < bb_period:
            return {"signal": "none"}
        
        # Calculate Bollinger Bands
        ma = np.mean(prices[-bb_period:])
        std = np.std(prices[-bb_period:])
        
        upper_band = ma + (bb_std * std)
        lower_band = ma - (bb_std * std)
        
        # Get current price
        current_price = prices[-1]
        
        # Determine signal
        signal = "none"
        
        if current_price < lower_band:
            signal = "buy"
        elif current_price > upper_band:
            signal = "sell"
        
        return {
            "signal": signal,
            "symbol": symbol,
            "upper_band": upper_band,
            "middle_band": ma,
            "lower_band": lower_band
        }


class VerticalSpread(StrategyBase):
    """
    Vertical spread options strategy.
    
    A vertical spread involves buying and selling options of the same type (calls or puts)
    and expiration date but different strike prices.
    """
    
    def initialize(self):
        """Initialize strategy with default parameters."""
        self.parameters.setdefault("spread_type", "bull_call")  # bull_call, bear_call, bull_put, bear_put
        self.parameters.setdefault("strike_width", 5)  # Width between strikes
        self.parameters.setdefault("ma_period", 20)  # For trend detection
        self.parameters.setdefault("volatility_threshold", 0.2)  # IV threshold
        self.parameters.setdefault("days_to_expiry", 30)  # Target days to expiration
        self.parameters.setdefault("take_profit_pct", 50)  # Take profit at % of max profit
        self.parameters.setdefault("stop_loss_pct", 200)  # Stop loss at % of max profit
        self.parameters.setdefault("symbol", "BTC/USD")
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal for vertical spread strategy."""
        symbol = self.parameters["symbol"]
        spread_type = self.parameters["spread_type"]
        ma_period = self.parameters["ma_period"]
        volatility_threshold = self.parameters["volatility_threshold"]
        
        if symbol not in market_data:
            return {"signal": "none"}
        
        # Get price history
        symbol_data = market_data[symbol]
        prices = symbol_data.get("close_history", [])
        
        if len(prices) < ma_period:
            return {"signal": "none"}
        
        # Check trend
        ma = np.mean(prices[-ma_period:])
        current_price = prices[-1]
        trend = "up" if current_price > ma else "down"
        
        # Estimate volatility (simplified)
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Determine signal based on spread type and market conditions
        signal = "none"
        
        if spread_type == "bull_call" and trend == "up" and volatility < volatility_threshold:
            # Bull call spread when trend is up and volatility is low
            signal = "buy"
        elif spread_type == "bear_call" and trend == "down" and volatility < volatility_threshold:
            # Bear call spread when trend is down and volatility is low
            signal = "buy"
        elif spread_type == "bull_put" and trend == "up" and volatility > volatility_threshold:
            # Bull put spread when trend is up and volatility is high
            signal = "buy"
        elif spread_type == "bear_put" and trend == "down" and volatility > volatility_threshold:
            # Bear put spread when trend is down and volatility is high
            signal = "buy"
        
        # Convert "buy" to specific spread type for clearer signals
        if signal == "buy":
            signal = spread_type
        
        return {
            "signal": signal,
            "symbol": symbol,
            "trend": trend,
            "volatility": volatility,
            "strategy_type": "vertical_spread"
        }


class IronCondor(StrategyBase):
    """
    Iron Condor options strategy.
    
    An iron condor involves selling an out-of-the-money put spread and an
    out-of-the-money call spread on the same underlying asset and expiration date.
    """
    
    def initialize(self):
        """Initialize strategy with default parameters."""
        self.parameters.setdefault("call_spread_width", 5)  # Width of call spread
        self.parameters.setdefault("put_spread_width", 5)  # Width of put spread
        self.parameters.setdefault("call_distance", 10)  # % OTM for call spread
        self.parameters.setdefault("put_distance", 10)  # % OTM for put spread
        self.parameters.setdefault("volatility_min", 0.15)  # Min IV threshold
        self.parameters.setdefault("volatility_max", 0.40)  # Max IV threshold
        self.parameters.setdefault("days_to_expiry", 45)  # Target days to expiration
        self.parameters.setdefault("take_profit_pct", 50)  # Take profit at % of max profit
        self.parameters.setdefault("stop_loss_pct", 200)  # Stop loss at % of max profit
        self.parameters.setdefault("symbol", "BTC/USD")
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal for iron condor strategy."""
        symbol = self.parameters["symbol"]
        volatility_min = self.parameters["volatility_min"]
        volatility_max = self.parameters["volatility_max"]
        
        if symbol not in market_data:
            return {"signal": "none"}
        
        # Get price history
        symbol_data = market_data[symbol]
        prices = symbol_data.get("close_history", [])
        
        if len(prices) < 20:  # Need at least 20 days for volatility calc
            return {"signal": "none"}
        
        # Check if we're in a range-bound market (simplified)
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Check trend strength (simplified)
        ma20 = np.mean(prices[-20:])
        ma50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma20
        trend_strength = abs(ma20 / ma50 - 1)
        
        # Determine signal
        signal = "none"
        
        # Iron condor is best when volatility is moderate and market is range-bound
        if volatility_min <= volatility <= volatility_max and trend_strength < 0.05:
            signal = "iron_condor"
        
        return {
            "signal": signal,
            "symbol": symbol,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "strategy_type": "iron_condor"
        }


# Additional strategy implementations can be added here

def create_test_strategies(count: int = 10) -> List[StrategyBase]:
    """
    Create a diverse set of test strategies with varied parameters.
    
    Args:
        count: Number of strategies to create
        
    Returns:
        List of strategy instances
    """
    strategies = []
    
    # Create a mix of different strategy types
    strategy_types = [
        MovingAverageCrossover,
        RSIStrategy,
        BollingerBands,
        VerticalSpread,
        IronCondor
    ]
    
    for i in range(count):
        # Select strategy type
        strategy_class = random.choice(strategy_types)
        
        # Create strategy with randomized parameters
        if strategy_class == MovingAverageCrossover:
            strategy = MovingAverageCrossover(
                parameters={
                    "fast_period": random.randint(5, 20),
                    "slow_period": random.randint(21, 50)
                }
            )
        elif strategy_class == RSIStrategy:
            strategy = RSIStrategy(
                parameters={
                    "rsi_period": random.randint(7, 21),
                    "overbought": random.randint(65, 80),
                    "oversold": random.randint(20, 35)
                }
            )
        elif strategy_class == BollingerBands:
            strategy = BollingerBands(
                parameters={
                    "bb_period": random.randint(10, 30),
                    "bb_std": random.uniform(1.5, 3.0)
                }
            )
        elif strategy_class == VerticalSpread:
            strategy = VerticalSpread(
                parameters={
                    "spread_type": random.choice(["bull_call", "bear_call", "bull_put", "bear_put"]),
                    "strike_width": random.randint(3, 10),
                    "ma_period": random.randint(10, 50),
                    "volatility_threshold": random.uniform(0.15, 0.35),
                    "days_to_expiry": random.choice([14, 21, 30, 45, 60])
                }
            )
        elif strategy_class == IronCondor:
            strategy = IronCondor(
                parameters={
                    "call_spread_width": random.randint(3, 10),
                    "put_spread_width": random.randint(3, 10),
                    "call_distance": random.randint(5, 20),
                    "put_distance": random.randint(5, 20),
                    "volatility_min": random.uniform(0.1, 0.2),
                    "volatility_max": random.uniform(0.3, 0.5)
                }
            )
        
        strategies.append(strategy)
    
    return strategies
