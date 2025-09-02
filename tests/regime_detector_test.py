#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Detector Test

This script demonstrates how the Contextual Data Provider and 
Market Regime Detector work together to identify market regimes.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RegimeDetectorTest")

# Import necessary components
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.contextual_data_provider import ContextualDataProvider
from trading_bot.analysis.market_regime_detector import MarketRegimeDetector

# Mock persistence for testing
class MockPersistence:
    def __init__(self):
        self.stored_data = {}
        
    def save_system_state(self, component_id, state):
        self.stored_data[component_id] = state
        return True
            
    def load_system_state(self, component_id):
        return self.stored_data.get(component_id, {})

def generate_test_data(symbols, days=100):
    """Generate synthetic test data for multiple symbols."""
    data = {}
    now = datetime.now()
    
    for symbol in symbols:
        # Create a DataFrame with random but somewhat realistic market data
        periods = days * 24  # hourly data
        
        dates = [now - timedelta(hours=i) for i in range(periods)]
        dates.reverse()
        
        # Starting price differs by symbol
        if symbol == "EURUSD":
            base_price = 1.12
        elif symbol == "GBPUSD":
            base_price = 1.30
        elif symbol == "USDJPY":
            base_price = 108.5
        else:  # Default
            base_price = 1.0
        
        # Generate price data with some randomness but general patterns
        close_prices = [base_price]
        for i in range(1, periods):
            # Create regimes:
            # - First 1/4: ranging
            # - Second 1/4: trending up
            # - Third 1/4: ranging with higher volatility
            # - Last 1/4: trending down
            
            quarter = periods // 4
            
            if i < quarter:  
                # Ranging
                rnd = np.random.normal(0, 0.0004)
                momentum = 0
                mean_reversion = -0.1 * (close_prices[i-1] - base_price) / base_price
            elif i < 2 * quarter:  
                # Trending up
                rnd = np.random.normal(0, 0.0005)
                momentum = 0.0002
                mean_reversion = 0
            elif i < 3 * quarter:  
                # Ranging higher volatility
                rnd = np.random.normal(0, 0.0008)
                momentum = 0
                mean_reversion = -0.05 * (close_prices[i-1] - (base_price * 1.02)) / (base_price * 1.02)
            else:  
                # Trending down
                rnd = np.random.normal(0, 0.0005)
                momentum = -0.0002
                mean_reversion = 0
                
            close_prices.append(close_prices[i-1] * (1 + rnd + momentum + mean_reversion))
        
        # Calculate high, low, open prices
        high_prices = [price * (1 + abs(np.random.normal(0, 0.0015))) for price in close_prices]
        low_prices = [price * (1 - abs(np.random.normal(0, 0.0015))) for price in close_prices]
        open_prices = [low + (high - low) * np.random.random() for high, low in zip(high_prices, low_prices)]
        
        # Create volume data
        volumes = [int(np.random.normal(10000, 2500)) for _ in range(periods)]
        
        # Assemble DataFrame
        df = pd.DataFrame({
            'datetime': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        df.set_index('datetime', inplace=True)
        data[symbol] = df
        
        # Save to CSV for reuse
        if not os.path.exists('data/market_data'):
            os.makedirs('data/market_data', exist_ok=True)
            
        df.to_csv(f'data/market_data/{symbol}_1h.csv')
        
    return data, symbols

def run_regime_detection_test():
    """Test the market regime detector with synthetic data."""
    # Initialize event bus
    event_bus = EventBus()
    
    # Initialize mock persistence
    persistence = MockPersistence()
    
    # Initialize contextual components
    data_provider = ContextualDataProvider(event_bus, persistence, {'csv_data_path': 'data/market_data'})
    regime_detector = MarketRegimeDetector(event_bus, persistence)
    
    # List of regimes for visualization
    detected_regimes = {}
    
    # Subscribe to regime change events
    def handle_regime_change(event):
        symbol = event.data.get('symbol')
        regime = event.data.get('regime')
        confidence = event.data.get('confidence', 0)
        timestamp = event.timestamp
        
        if symbol not in detected_regimes:
            detected_regimes[symbol] = []
            
        detected_regimes[symbol].append({
            'timestamp': timestamp,
            'regime': regime,
            'confidence': confidence
        })
        
        logger.info(f"Regime change: {symbol} -> {regime} (confidence: {confidence:.2f})")
    
    event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, handle_regime_change)
    
    # Test symbols
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Generate or load test data
    if not os.path.exists(f'data/market_data/{symbols[0]}_1h.csv'):
        logger.info("Generating test data...")
        market_data, _ = generate_test_data(symbols)
    else:
        logger.info("Using existing test data...")
        market_data = {}
        for symbol in symbols:
            file_path = f'data/market_data/{symbol}_1h.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                market_data[symbol] = df
    
    # Run regime detection for each symbol
    for symbol in symbols:
        if symbol in market_data:
            logger.info(f"Detecting regimes for {symbol}...")
            
            # Send data to provider
            for chunk_idx in range(0, len(market_data[symbol]), 24):  # Process in daily chunks
                chunk = market_data[symbol].iloc[chunk_idx:chunk_idx+24]
                
                # This would normally be handled by the data provider's _load_from_csv method
                # But for testing, we'll simulate the data response directly
                event_bus.publish(Event(
                    event_type=EventType.DATA_RESPONSE,
                    data={
                        'request_type': 'market_data',
                        'symbol': symbol,
                        'timeframe': '1h',
                        'source': 'test',
                        'data': chunk
                    }
                ))
                
                # Small delay to allow events to process
                time.sleep(0.02)
    
    # Wait for all events to process
    time.sleep(1)
    
    # Visualize the results
    plt.figure(figsize=(15, 10))
    
    for i, symbol in enumerate(symbols):
        ax = plt.subplot(len(symbols), 1, i+1)
        
        # Plot price
        if symbol in market_data:
            df = market_data[symbol]
            ax.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
            
            # Add regime overlay
            if symbol in detected_regimes:
                regimes = detected_regimes[symbol]
                
                # Define colors for different regimes
                colors = {
                    'trending_up': 'green',
                    'trending_down': 'red',
                    'ranging': 'gray',
                    'breakout': 'purple',
                    'reversal': 'orange',
                    'volatility_compression': 'blue',
                    'volatility_expansion': 'yellow',
                    'unknown': 'lightgray'
                }
                
                # Plot regime background
                prev_time = df.index[0]
                prev_regime = 'unknown'
                
                for r in regimes:
                    # Fill from previous time to this regime change
                    current_time = r['timestamp'] if isinstance(r['timestamp'], datetime) else datetime.fromisoformat(r['timestamp'])
                    color = colors.get(prev_regime, 'lightgray')
                    
                    # Add colored background for the regime period
                    ax.axvspan(prev_time, current_time, alpha=0.2, color=color)
                    
                    # Update for next span
                    prev_time = current_time
                    prev_regime = r['regime']
                
                # Fill the last period to the end
                if prev_time < df.index[-1]:
                    color = colors.get(prev_regime, 'lightgray')
                    ax.axvspan(prev_time, df.index[-1], alpha=0.2, color=color)
                
                # Add legend for regimes
                handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.2) for regime, color in colors.items()]
                labels = list(colors.keys())
                ax.legend(handles, labels, loc='upper right')
        
        ax.set_title(f"{symbol} with Detected Regimes")
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('regime_detection_results.png')
    plt.close()
    
    logger.info(f"Results saved to regime_detection_results.png")
    return detected_regimes

if __name__ == "__main__":
    run_regime_detection_test()
