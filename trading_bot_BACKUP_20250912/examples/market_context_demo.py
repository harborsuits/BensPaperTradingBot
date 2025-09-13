#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Context Fetcher Demo - Demonstrates how the MarketContextFetcher works
with the strategy rotator system to adapt trading strategies based on market regimes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our components
from trading_bot.market.market_context_fetcher import MarketContextFetcher, MarketRegime, MarketRegimeEvent
from trading_bot.strategies.integrated_strategy_rotator import IntegratedStrategyRotator
from trading_bot.strategies.momentum_strategy import MomentumStrategy
from trading_bot.strategies.trend_following_strategy import TrendFollowingStrategy
from trading_bot.strategies.mean_reversion_strategy import MeanReversionStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MarketContextDemo")

class MarketDataSimulator:
    """Simple simulator to generate market data for demonstration purposes."""
    
    def __init__(self, symbols=None, start_price=100.0, volatility=0.01):
        """Initialize simulator with given symbols."""
        self.symbols = symbols or ["SPY", "QQQ", "IWM"]
        self.current_prices = {symbol: start_price for symbol in self.symbols}
        self.volatility = volatility
        self.base_drift = 0.0001
        self.current_regime = MarketRegime.UNKNOWN
        
        # Create regime periods for simulation
        self.regime_periods = [
            (0, 60, MarketRegime.BULL),       # Bullish start
            (60, 120, MarketRegime.HIGH_VOL), # High volatility
            (120, 180, MarketRegime.BEAR),    # Bear market
            (180, 240, MarketRegime.SIDEWAYS),# Sideways movement
            (240, 270, MarketRegime.CRISIS),  # Brief crisis
            (270, 330, MarketRegime.SIDEWAYS),# Recovery sideways
            (330, 400, MarketRegime.BULL)     # Bullish end
        ]
        
        self.current_step = 0
        self._update_regime()
    
    def _update_regime(self):
        """Update the current regime based on step."""
        for start, end, regime in self.regime_periods:
            if start <= self.current_step < end:
                self.current_regime = regime
                return
    
    def get_latest_data(self, symbol):
        """Get latest market data for a symbol."""
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        # Update step
        self.current_step += 1
        self._update_regime()
        
        # Adjust drift and volatility based on regime
        drift = self.base_drift
        vol = self.volatility
        
        if self.current_regime == MarketRegime.BULL:
            drift = 0.0005
            vol = 0.008
        elif self.current_regime == MarketRegime.BEAR:
            drift = -0.0004
            vol = 0.012
        elif self.current_regime == MarketRegime.SIDEWAYS:
            drift = 0.0001
            vol = 0.005
        elif self.current_regime == MarketRegime.HIGH_VOL:
            drift = 0.0
            vol = 0.02
        elif self.current_regime == MarketRegime.LOW_VOL:
            drift = 0.0002
            vol = 0.003
        elif self.current_regime == MarketRegime.CRISIS:
            drift = -0.002
            vol = 0.03
        
        # Generate price movement
        price_change = np.random.normal(drift, vol)
        self.current_prices[symbol] *= (1 + price_change)
        
        # Generate volume
        volume = np.random.lognormal(9, 0.5)  # log-normal around ~8000
        
        # Return data point
        return {
            "price": self.current_prices[symbol],
            "volume": volume,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_current_regime(self):
        """Get the current true regime for verification."""
        return self.current_regime

def on_regime_change(event):
    """Handler for regime change events."""
    logger.info(f"REGIME CHANGE DETECTED: {event}")
    
    # Here you would trigger strategy rotations or risk adjustments
    # based on the new regime

def run_demo():
    """Run the market context fetcher demonstration."""
    logger.info("Starting Market Context Fetcher Demo")
    
    # Create data directory if it doesn't exist
    os.makedirs("data/demo", exist_ok=True)
    
    # Initialize our market data simulator
    symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    data_simulator = MarketDataSimulator(symbols=symbols)
    
    # Initialize market context fetcher
    fetcher = MarketContextFetcher(
        symbols=symbols,
        data_provider=data_simulator,
        update_interval=2,  # Update every 2 seconds for demo
        data_dir="data/demo/market_context",
        debug_mode=True
    )
    
    # Add regime change listener
    fetcher.add_event_listener(on_regime_change)
    
    # Initialize strategies
    strategies = {
        "momentum": MomentumStrategy(),
        "trend_following": TrendFollowingStrategy(),
        "mean_reversion": MeanReversionStrategy()
    }
    
    # Initialize strategy rotator
    rotator = IntegratedStrategyRotator(
        strategies=list(strategies.keys()),
        data_dir="data/demo"
    )
    
    # Start market context fetcher
    fetcher.start()
    
    # Prepare for tracking
    detected_regimes = []
    actual_regimes = []
    timestamps = []
    allocations_history = []
    
    # Run simulation for 400 steps
    logger.info("Running simulation...")
    
    try:
        for step in range(400):
            # Track current time
            timestamps.append(datetime.now())
            
            # Sleep to simulate time passing
            time.sleep(0.1)
            
            # Every 30 steps, perform strategy rotation based on market regime
            if step % 30 == 0:
                # Get current regime from fetcher
                current_regime, confidence = fetcher.get_current_regime()
                
                # Get actual regime from simulator (for verification)
                actual_regime = data_simulator.get_current_regime()
                
                # Record regimes
                detected_regimes.append(current_regime)
                actual_regimes.append(actual_regime)
                
                logger.info(f"Step {step}: Detected regime: {current_regime.name}, "
                          f"Actual regime: {actual_regime.name}, "
                          f"Confidence: {confidence:.2f}")
                
                # Get latest metrics from fetcher
                metrics = fetcher.get_latest_metrics()
                
                # Create a dataframe with current metrics for the rotator
                market_data = pd.DataFrame({
                    'close': [metrics.get('short_term_return', 0.0)],
                    'volatility': [metrics.get('volatility_ratio', 1.0)],
                    'momentum': [metrics.get('avg_rsi', 50.0) / 100.0]
                })
                
                # Rotate strategies based on market regime
                rotation_result = rotator.rotate_strategies(market_data, force_rotation=True)
                
                # Record allocations
                allocations_history.append({
                    'step': step,
                    'regime': current_regime.name,
                    'allocations': rotation_result['new_allocations'].copy()
                })
                
                logger.info(f"New allocations: {rotation_result['new_allocations']}")
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    # Stop fetcher
    fetcher.stop()
    
    # Plot results
    plot_simulation_results(timestamps, detected_regimes, actual_regimes, allocations_history)
    
    logger.info("Market Context Fetcher Demo completed")

def plot_simulation_results(timestamps, detected_regimes, actual_regimes, allocations_history):
    """Plot the results of the simulation."""
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot regime detection accuracy
        actual_regime_nums = [regime.value for regime in actual_regimes]
        detected_regime_nums = [regime.value for regime in detected_regimes]
        
        # Convert to relative timestamps in seconds
        rel_timestamps = [(t - timestamps[0]).total_seconds() for t in timestamps[:len(actual_regimes)]]
        
        ax1.plot(rel_timestamps, actual_regime_nums, 'b-', label='Actual Regime')
        ax1.plot(rel_timestamps, detected_regime_nums, 'r--', label='Detected Regime')
        
        # Add regime labels
        ax1.set_yticks(list(range(7)))
        ax1.set_yticklabels([r.name for r in MarketRegime])
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Market Regime')
        ax1.set_title('Market Regime Detection Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot strategy allocations
        steps = [entry['step'] for entry in allocations_history]
        strategies = list(allocations_history[0]['allocations'].keys())
        
        data = {}
        for strategy in strategies:
            data[strategy] = [entry['allocations'][strategy] for entry in allocations_history]
        
        bottom = np.zeros(len(steps))
        
        for strategy in strategies:
            ax2.bar(steps, data[strategy], bottom=bottom, label=strategy)
            bottom += np.array(data[strategy])
        
        # Add regime markers
        for i, entry in enumerate(allocations_history):
            ax2.text(entry['step'], 105, entry['regime'], ha='center', fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Strategy Allocation (%)')
        ax2.set_title('Strategy Allocations Based on Market Regime')
        ax2.set_ylim(0, 115)  # Leave space for regime labels
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(strategies))
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('data/demo/market_context_results.png')
        plt.close()
        
        logger.info("Results plotted and saved to data/demo/market_context_results.png")
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

if __name__ == "__main__":
    run_demo() 