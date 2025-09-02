#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Market Regime Detector Test

This script demonstrates how the Contextual Data Provider and 
Market Regime Detector work together to identify market regimes.
(No visualization dependencies required)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleRegimeTest")

# Mock classes for testing without dependencies
class Event:
    """Mock Event class for demonstration"""
    def __init__(self, event_type, data):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()
        
class EventType:
    """Mock EventType constants"""
    MARKET_REGIME_CHANGE = "market_regime_change"
    VOLATILITY_UPDATE = "volatility_update"
    DATA_RESPONSE = "data_response"
    DATA_REQUEST = "data_request"
    
class EventBus:
    """Simplified EventBus for demonstration"""
    def __init__(self):
        self.subscribers = {}
        self.event_history = []
        
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event):
        logger.debug(f"Publishing event: {event.event_type}")
        self.event_history.append(event)
        
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                callback(event)

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
        logger.info(f"Generated and saved test data for {symbol}")
        
    return data, symbols

class SimpleMarketRegimeDetector:
    """Simplified Market Regime Detector for testing."""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.current_regimes = {}
        
        # Subscribe to data events
        self.event_bus.subscribe(EventType.DATA_RESPONSE, self.handle_data_response)
    
    def handle_data_response(self, event):
        """Handle data response events."""
        request_type = event.data.get('request_type')
        
        # We only process market data responses
        if request_type == 'market_data':
            symbol = event.data.get('symbol')
            timeframe = event.data.get('timeframe', '1h')
            
            # Extract market data
            market_data = event.data.get('data')
            
            if market_data is not None and isinstance(market_data, pd.DataFrame):
                # Detect regime
                regime_data = self.detect_regime(symbol, timeframe, market_data)
                
                # Log the detected regime
                logger.info(f"Detected regime for {symbol}: {regime_data['regime']} " +
                          f"(confidence: {regime_data['confidence']:.2f})")
    
    def detect_regime(self, symbol, timeframe, market_data):
        """
        Detect the current market regime for a symbol and timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: Data timeframe
            market_data: DataFrame with OHLCV data
            
        Returns:
            Regime detection results
        """
        key = f"{symbol}_{timeframe}"
        
        try:
            # Apply indicators for regime detection
            data = market_data.copy()
            
            # Make sure we have the required columns
            if 'close' not in data.columns:
                return {'regime': 'unknown', 'confidence': 0.5}
            
            # Calculate EMAs
            data['ema20'] = data['close'].ewm(span=20).mean()
            data['ema50'] = data['close'].ewm(span=50).mean()
            
            # Simple volatility
            if len(data) > 20:
                data['volatility'] = data['close'].pct_change().rolling(20).std()
            else:
                data['volatility'] = 0
            
            # Get latest values
            if len(data) < 2:
                return {'regime': 'unknown', 'confidence': 0.5}
                
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Determine regime
            regime = 'unknown'
            confidence = 0.5
            volatility_state = 'medium'
            
            # Trending determination
            if latest['ema20'] > latest['ema50'] and latest['ema20'] > prev['ema20']:
                # Trending up
                regime = 'trending_up'
                confidence = 0.7
            elif latest['ema20'] < latest['ema50'] and latest['ema20'] < prev['ema20']:
                # Trending down
                regime = 'trending_down'
                confidence = 0.7
            else:
                # Ranging or unknown
                regime = 'ranging'
                confidence = 0.6
            
            # Volatility state
            if 'volatility' in latest and latest['volatility'] > 0:
                if len(data) > 50:
                    # Use historical percentiles
                    vol_series = data['volatility'].dropna()
                    vol_low = np.percentile(vol_series, 25)
                    vol_high = np.percentile(vol_series, 75)
                    
                    if latest['volatility'] < vol_low:
                        volatility_state = 'low'
                    elif latest['volatility'] > vol_high:
                        volatility_state = 'high'
                        
                        # Check for breakout
                        if latest['close'] > latest['ema20'] * 1.01 or latest['close'] < latest['ema20'] * 0.99:
                            regime = 'breakout'
                            confidence = 0.75
            
            # Previous regime
            previous_regime = self.current_regimes.get(key, 'unknown')
            
            # If regime changed, publish event
            if regime != previous_regime:
                self.event_bus.publish(Event(
                    event_type=EventType.MARKET_REGIME_CHANGE,
                    data={
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'regime': regime,
                        'previous_regime': previous_regime,
                        'confidence': confidence
                    }
                ))
                
                logger.info(f"Regime change for {symbol}: {previous_regime} -> {regime}")
            
            # Update current regime
            self.current_regimes[key] = regime
            
            # Return detection results
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility_state': volatility_state
            }
            
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.5}

def run_simple_regime_test():
    """Run a simple regime detection test."""
    # Initialize event bus
    event_bus = EventBus()
    
    # Initialize mock persistence
    persistence = MockPersistence()
    
    # Initialize detector
    regime_detector = SimpleMarketRegimeDetector(event_bus)
    
    # List of regimes for tracking
    detected_regimes = {}
    
    # Subscribe to regime change events
    def handle_regime_change(event):
        symbol = event.data.get('symbol')
        regime = event.data.get('regime')
        confidence = event.data.get('confidence', 0)
        
        if symbol not in detected_regimes:
            detected_regimes[symbol] = []
            
        detected_regimes[symbol].append({
            'timestamp': event.timestamp,
            'regime': regime,
            'confidence': confidence
        })
        
        logger.info(f"Caught regime change: {symbol} -> {regime} (confidence: {confidence:.2f})")
    
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
                logger.info(f"Loaded data for {symbol}: {len(df)} rows")
    
    # Run regime detection for each symbol
    for symbol in symbols:
        if symbol in market_data:
            logger.info(f"Detecting regimes for {symbol}...")
            
            # Send data to detector
            for chunk_idx in range(0, len(market_data[symbol]), 24):  # Process in daily chunks
                chunk = market_data[symbol].iloc[chunk_idx:chunk_idx+24]
                
                # Simulate the data response
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
    
    # Summarize the results
    logger.info("\n=== REGIME DETECTION SUMMARY ===")
    
    for symbol in symbols:
        if symbol in detected_regimes:
            regimes = detected_regimes[symbol]
            
            logger.info(f"\n{symbol} Regime Changes ({len(regimes)}):")
            for i, r in enumerate(regimes):
                logger.info(f"{i+1}. {r['regime']} (confidence: {r['confidence']:.2f})")
    
    return detected_regimes

if __name__ == "__main__":
    run_simple_regime_test()
