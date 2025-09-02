"""
Optimized Bollinger Bands Strategy

This strategy implements the optimized Bollinger Bands parameters
discovered through evolutionary testing.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from evotrader.core.strategy import Strategy
from evotrader.utils.robust_indicators import safe_bollinger_bands


class OptimizedBollingerBandsStrategy(Strategy):
    """
    Bollinger Bands strategy with parameters optimized through evolution.
    
    This strategy:
    1. Calculates Bollinger Bands using the evolved optimal parameters
    2. Generates buy signals when price touches the lower band
    3. Generates sell signals when price touches the upper band
    4. Adjusts position sizing based on band penetration depth
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with evolved optimal parameters.
        
        Args:
            parameters: Optional parameter overrides
        """
        # Default parameters from evolutionary optimization
        default_params = {
            # Core Bollinger parameters from evolution
            "period": 18,  # Evolution found 18 optimal
            "std_dev": 2.12,  # Evolution refined to ~2.12
            "signal_threshold": 0.026,  # Evolution refined to ~0.026
            
            # Risk management parameters
            "position_size": 0.1,  # Default to 10% of available capital
            "stop_loss_pct": 0.05,  # 5% stop loss
            "take_profit_pct": 0.1,  # 10% take profit
            
            # Signal filtering
            "min_band_width": 0.02,  # Minimum band width as % of price
            "trend_filter": True,  # Enable trend filtering
            "trend_period": 50,  # Period for trend detection
        }
        
        # Override defaults with any provided parameters
        self.parameters = default_params.copy()
        if parameters:
            self.parameters.update(parameters)
            
        # Initialize state
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Required by Strategy interface
        self.strategy_id = f"OptimizedBollinger_{np.random.randint(10000, 99999)}"
        self.metadata = {
            "strategy_type": "OptimizedBollingerBandsStrategy",
            "description": "Evolutionarily optimized Bollinger Bands strategy",
            "version": "1.0.0",
            "source": "Evolution test results"
        }
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signals based on Bollinger Bands.
        
        Args:
            market_data: Market data with columns 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            Dict with signal information
        """
        if market_data is None or len(market_data) < self.parameters["period"]:
            return {"signal": "none", "confidence": 0, "reason": "insufficient_data"}
        
        # Extract price data
        close_prices = market_data["close"].values
        
        # Calculate Bollinger Bands
        upper, middle, lower = safe_bollinger_bands(
            close_prices, 
            period=self.parameters["period"], 
            std_dev=self.parameters["std_dev"]
        )
        
        if upper is None or middle is None or lower is None:
            return {"signal": "none", "confidence": 0, "reason": "indicator_error"}
        
        # Get the most recent values
        current_price = close_prices[-1]
        upper_band = upper[-1]
        middle_band = middle[-1]
        lower_band = lower[-1]
        
        # Calculate band width as percentage of price
        band_width = (upper_band - lower_band) / middle_band
        
        # Check if band width is sufficient
        if band_width < self.parameters["min_band_width"]:
            return {"signal": "none", "confidence": 0, "reason": "narrow_bands"}
        
        # Calculate the deviation from the middle band
        deviation = (current_price - middle_band) / middle_band
        
        # Calculate distance from bands
        upper_distance = (upper_band - current_price) / current_price
        lower_distance = (current_price - lower_band) / current_price
        
        # Check trend if filtering is enabled
        if self.parameters["trend_filter"]:
            # Simple moving average as trend indicator
            trend_period = min(self.parameters["trend_period"], len(close_prices))
            trend_sma = np.mean(close_prices[-trend_period:])
            trend_up = current_price > trend_sma
        else:
            trend_up = True  # Default to true if not filtering
        
        # Generate signals based on band penetration
        signal = "none"
        confidence = 0
        reason = "no_trigger"
        
        # Bollinger Band strategy logic
        if current_price <= lower_band + (band_width * self.parameters["signal_threshold"]):
            # Price is at or below lower band + threshold
            signal = "buy"
            confidence = min(1.0, lower_distance / (band_width * 0.5))
            reason = "price_at_lower_band"
            
        elif current_price >= upper_band - (band_width * self.parameters["signal_threshold"]):
            # Price is at or above upper band - threshold
            signal = "sell"
            confidence = min(1.0, upper_distance / (band_width * 0.5))
            reason = "price_at_upper_band"
            
        # Apply trend filter
        if self.parameters["trend_filter"]:
            if signal == "buy" and not trend_up:
                signal = "none"
                confidence = 0
                reason = "trend_filter_block_buy"
            elif signal == "sell" and trend_up:
                signal = "none"
                confidence = 0
                reason = "trend_filter_block_sell"
        
        # Calculate position size based on confidence
        position_size = self.parameters["position_size"] * confidence
        
        # Calculate stop loss and take profit levels
        if signal == "buy":
            stop_price = current_price * (1 - self.parameters["stop_loss_pct"])
            target_price = current_price * (1 + self.parameters["take_profit_pct"])
        elif signal == "sell":
            stop_price = current_price * (1 + self.parameters["stop_loss_pct"])
            target_price = current_price * (1 - self.parameters["take_profit_pct"])
        else:
            stop_price = 0
            target_price = 0
        
        # Return complete signal information
        return {
            "signal": signal,
            "confidence": confidence,
            "position_size": position_size,
            "reason": reason,
            "indicators": {
                "upper_band": float(upper_band),
                "middle_band": float(middle_band),
                "lower_band": float(lower_band),
                "band_width": float(band_width),
                "deviation": float(deviation),
                "price": float(current_price)
            },
            "risk_management": {
                "stop_loss": float(stop_price),
                "take_profit": float(target_price)
            }
        }
    
    def update_position(self, position: float, entry_price: float):
        """
        Update the strategy's position information.
        
        Args:
            position: Current position size (+ve for long, -ve for short)
            entry_price: Position entry price
        """
        self.position = position
        self.entry_price = entry_price
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the strategy parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Update the strategy parameters."""
        self.parameters.update(parameters)
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals for the strategy adapter interface.
        
        Args:
            market_data: Market data with OHLCV columns
            
        Returns:
            Signal dictionary
        """
        # This method satisfies the required interface for the BensBotStrategyAdapter
        return self.calculate_signal(market_data)
