#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating the use of StrategyRotator with different strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta

# Import from our new modular architecture
from trading_bot.strategy import (
    Strategy, 
    MomentumStrategy, 
    TrendFollowingStrategy, 
    MeanReversionStrategy,
    StrategyRotator
)
from trading_bot.common.market_types import MarketRegime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RotatorExample")

def generate_sample_market_data(
    days: int = 30, 
    initial_price: float = 100.0, 
    volatility: float = 0.01, 
    trend: float = 0.001
) -> Dict[str, Any]:
    """
    Generate sample market data for testing.
    
    Args:
        days: Number of days of data to generate
        initial_price: Starting price
        volatility: Daily volatility
        trend: Daily trend factor
        
    Returns:
        Dict with market data
    """
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate prices with random walk
    prices = [initial_price]
    for i in range(1, len(dates)):
        # Random component + trend component
        change = np.random.normal(0, volatility) + trend
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generate volume (loosely correlated with price changes)
    volumes = []
    for i in range(len(prices)):
        base_volume = 1000
        if i > 0:
            price_change = abs(prices[i] / prices[i-1] - 1)
            volume = base_volume * (1 + price_change * 10)
            volumes.append(int(volume))
        else:
            volumes.append(base_volume)
    
    # Create market data dictionary
    market_data = {
        "dates": dates,
        "prices": prices,
        "volume": volumes,
        "latest_price": prices[-1],
        "latest_volume": volumes[-1],
        "price_change_1d": prices[-1] / prices[-2] - 1 if len(prices) > 1 else 0,
        "price_change_7d": prices[-1] / prices[-8] - 1 if len(prices) > 7 else 0,
        "price_change_30d": prices[-1] / prices[-30] - 1 if len(prices) > 29 else 0,
    }
    
    return market_data

def simulate_performance(strategy_name: str, signal: float) -> float:
    """
    Simulate strategy performance based on signal.
    This is just a toy example - real performance would be based on actual trades.
    
    Args:
        strategy_name: Name of the strategy
        signal: Strategy signal (-1.0 to 1.0)
        
    Returns:
        float: Simulated performance metric
    """
    # Add some randomness plus a component based on signal strength
    base_performance = np.random.normal(0, 0.01)  # Random component
    signal_component = signal * 0.02  # Signal component
    
    # Different strategies might have different base performance
    strategy_component = {
        "MomentumStrategy": 0.01,
        "TrendFollowingStrategy": 0.005,
        "MeanReversionStrategy": 0.015
    }.get(strategy_name, 0)
    
    return base_performance + signal_component + strategy_component

def run_example():
    """Run the strategy rotator example."""
    logger.info("Starting Strategy Rotator Example")
    
    # Create a custom strategy
    class CustomStrategy(Strategy):
        """Custom strategy example"""
        def generate_signal(self, market_data: Dict[str, Any]) -> float:
            """Generate custom signal based on price and volume."""
            prices = market_data.get("prices", [])
            volumes = market_data.get("volume", [])
            
            if len(prices) < 5 or len(volumes) < 5:
                return 0.0
                
            # Price momentum
            price_momentum = prices[-1] / prices[-5] - 1
            
            # Volume momentum
            volume_momentum = volumes[-1] / volumes[-5] - 1
            
            # Combined signal
            signal = price_momentum * 5 + volume_momentum * 2
            signal = np.clip(signal, -1.0, 1.0)
            
            self.last_signal = signal
            self.last_update_time = datetime.now()
            
            return signal
    
    # Initialize rotator with default strategies
    rotator = StrategyRotator(regime_adaptation=True)
    
    # Add custom strategy
    custom_strategy = CustomStrategy("CustomStrategy", {"weight": 0.25})
    rotator.add_strategy(custom_strategy)
    
    # Generate sample market data
    bull_market = generate_sample_market_data(days=30, trend=0.003, volatility=0.01)
    bear_market = generate_sample_market_data(days=30, trend=-0.002, volatility=0.015)
    sideways_market = generate_sample_market_data(days=30, trend=0, volatility=0.005)
    
    # Test with different market regimes
    logger.info("Testing with Bull Market regime")
    rotator.update_market_regime(MarketRegime.BULL, confidence=0.8)
    bull_signals = rotator.generate_signals(bull_market)
    bull_combined = rotator.get_combined_signal()
    
    logger.info("Testing with Bear Market regime")
    rotator.update_market_regime(MarketRegime.BEAR, confidence=0.7)
    bear_signals = rotator.generate_signals(bear_market)
    bear_combined = rotator.get_combined_signal()
    
    logger.info("Testing with Sideways Market regime")
    rotator.update_market_regime(MarketRegime.SIDEWAYS, confidence=0.9)
    sideways_signals = rotator.generate_signals(sideways_market)
    sideways_combined = rotator.get_combined_signal()
    
    # Display results
    logger.info("\nCurrent Strategy Weights:")
    for name, weight in rotator.get_strategy_weights().items():
        logger.info(f"  {name}: {weight:.4f}")
    
    logger.info("\nSignal Results:")
    logger.info(f"  Bull Market Combined Signal: {bull_combined:.4f}")
    logger.info(f"  Bear Market Combined Signal: {bear_combined:.4f}")
    logger.info(f"  Sideways Market Combined Signal: {sideways_combined:.4f}")
    
    # Simulate performance updates
    logger.info("\nSimulating performance updates...")
    for _ in range(10):
        # Generate random market data
        market_data = generate_sample_market_data(days=10, 
                                                 trend=np.random.normal(0, 0.001), 
                                                 volatility=np.random.uniform(0.005, 0.02))
        
        # Generate signals
        signals = rotator.generate_signals(market_data)
        
        # Simulate performance for each strategy
        performance_data = {
            name: simulate_performance(name, signal)
            for name, signal in signals.items()
        }
        
        # Update performance
        rotator.update_strategy_performance(performance_data)
    
    # Display updated weights after performance adaptation
    logger.info("\nUpdated Strategy Weights (after performance adaptation):")
    for name, weight in rotator.get_strategy_weights().items():
        logger.info(f"  {name}: {weight:.4f}")
    
    # Get performance metrics
    performance_metrics = rotator.get_performance_metrics()
    logger.info("\nStrategy Performance Metrics:")
    for name, metrics in performance_metrics.items():
        logger.info(f"  {name}: {metrics['average_performance']:.4f}")
    
    # Save state
    rotator.save_state()
    logger.info("Strategy rotator state saved")

if __name__ == "__main__":
    run_example() 