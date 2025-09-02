#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence-Adjusted Position Sizing Example

This example demonstrates how to use the indicator-sentiment integrator and 
confidence-adjusted position sizing together for optimal trade sizing.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.persistence import PersistenceManager

from trading_bot.ai_analysis.indicator_sentiment_integrator import IndicatorSentimentIntegrator
from trading_bot.strategies.forex.confidence_adjusted_position_sizing import ConfidenceAdjustedPositionSizing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the confidence-adjusted position sizing example."""
    # Initialize components
    event_bus = EventBus()
    
    # Initialize persistence manager
    persistence_dir = os.path.join(os.path.dirname(__file__), '../state')
    os.makedirs(persistence_dir, exist_ok=True)
    persistence = PersistenceManager(persistence_dir)
    
    # Initialize indicator-sentiment integrator
    integrator_config = {
        'state_dir': persistence_dir,
        'news_sentiment_weight': 0.4,
        'social_sentiment_weight': 0.3,
        'market_sentiment_weight': 0.3,
        'stale_data_seconds': 3600,  # 1 hour
        'integration_interval_seconds': 5.0,
        'min_data_points': 3
    }
    
    integrator = IndicatorSentimentIntegrator(
        event_bus=event_bus,
        indicator_weight=0.6,  # Technical indicators get 60% weight
        sentiment_weight=0.4,  # Sentiment gets 40% weight
        config=integrator_config,
        max_cache_size=1000,
        cache_expiry_seconds=3600
    )
    
    # Initialize confidence-adjusted position sizing
    position_sizing_config = {
        'max_risk_per_trade_percent': 1.0,     # 1% risk per trade
        'min_position_size': 0.01,             # Minimum position size (micro lot)
        'max_position_size': 5.0,              # Maximum position size
        'use_confidence_adjustment': True,     # Enable confidence adjustment
        'min_confidence_threshold': 0.4,       # Minimum confidence to trade
        'high_confidence_threshold': 0.7,      # Threshold for high confidence
        'signal_agreement_bonus': 0.3,         # Stronger agreement bonus
        'signal_disagreement_penalty': 0.5     # Stronger disagreement penalty
    }
    
    position_sizer = ConfidenceAdjustedPositionSizing(position_sizing_config)
    
    # Subscribe to relevant events
    event_bus.subscribe(EventType.TRADE_SIGNAL_RECEIVED, lambda event: handle_trade_signal(
        event, integrator, position_sizer
    ))
    
    # Run simulation
    run_simulation(event_bus)
    
    logger.info("Example completed.")

def handle_trade_signal(event: Event, 
                       integrator: IndicatorSentimentIntegrator, 
                       position_sizer: ConfidenceAdjustedPositionSizing):
    """
    Handle trade signals and apply confidence-adjusted position sizing.
    
    Args:
        event: Trade signal event
        integrator: Indicator-sentiment integrator
        position_sizer: Confidence-adjusted position sizer
    """
    data = event.data
    symbol = data.get('symbol')
    direction = data.get('signal_type', 'UNKNOWN')
    entry_price = data.get('price', 0.0)
    
    # Extract stop loss in pips (convert from price to pips)
    stop_loss_price = data.get('stop_loss', 0.0)
    stop_loss_pips = convert_price_to_pips(symbol, entry_price, stop_loss_price)
    
    logger.info(f"Processing trade signal for {symbol}: {direction} at {entry_price}, SL: {stop_loss_pips} pips")
    
    # Get integrated analysis
    integrated_data = integrator.get_integrated_analysis(symbol)
    
    if not integrated_data:
        logger.warning(f"No integrated data available for {symbol}")
        return
    
    logger.info(f"Integrated data for {symbol}: score={integrated_data.get('integrated_score', 0)}, "
                f"confidence={integrated_data.get('confidence', 0)}")
    
    # Calculate position size with confidence adjustment
    account_balance = 10000  # Example account balance
    position_size, adjustment_details = position_sizer.calculate_position_size_with_confidence(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss_pips=stop_loss_pips,
        account_balance=account_balance,
        integrated_data=integrated_data
    )
    
    logger.info(f"Position sizing for {symbol}: {adjustment_details}")
    
    # Calculate risk amount
    risk_amount = position_sizer.calculate_risk_amount(
        position_size=position_size,
        stop_loss_pips=stop_loss_pips,
        symbol=symbol,
        entry_price=entry_price
    )
    
    # Log trade details with confidence metrics
    if position_size > 0:
        logger.info(f"TRADE: {symbol} {direction} {position_size:.2f} lots at {entry_price}")
        logger.info(f"       Risk: ${risk_amount:.2f} ({(risk_amount/account_balance)*100:.2f}% of account)")
        logger.info(f"       Confidence: {integrated_data.get('confidence', 0):.2f}, "
                    f"Adjustment Factor: {adjustment_details.get('adjustment_factor', 1.0):.2f}")
        
        if adjustment_details.get('signal_agreement', False):
            logger.info("       Signal Agreement Bonus Applied: Indicators and sentiment agree")
        elif adjustment_details.get('signal_disagreement', False):
            logger.info("       Signal Disagreement Penalty Applied: Indicators and sentiment disagree")
    else:
        logger.info(f"NO TRADE: {symbol} {direction} - Insufficient confidence or other restrictions")

def run_simulation(event_bus: EventBus):
    """
    Run a simulation to demonstrate the confidence-adjusted position sizing.
    
    Args:
        event_bus: Event bus for publishing events
    """
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
    
    # First publish some technical indicator updates
    for symbol in symbols:
        # Simulated technical indicators
        indicators = generate_technical_indicators(symbol)
        
        event_bus.publish(
            EventType.TECHNICAL_INDICATORS_UPDATED,
            {
                'symbol': symbol,
                'indicators': indicators,
                'timestamp': datetime.now(),
                'timeframe': '1h'
            }
        )
        logger.debug(f"Published technical indicators for {symbol}")
    
    # Then publish some sentiment updates
    for symbol in symbols:
        # Simulated sentiment data
        sentiment = generate_sentiment_data(symbol)
        
        # Publish news sentiment
        event_bus.publish(
            EventType.NEWS_SENTIMENT_UPDATED,
            {
                'symbol': symbol,
                'sentiment': sentiment['news_sentiment'],
                'timestamp': datetime.now()
            }
        )
        
        # Publish social sentiment
        event_bus.publish(
            EventType.SOCIAL_SENTIMENT_UPDATED,
            {
                'symbol': symbol,
                'sentiment': sentiment['social_sentiment'],
                'timestamp': datetime.now()
            }
        )
        
        # Publish market sentiment
        event_bus.publish(
            EventType.MARKET_SENTIMENT_UPDATED,
            {
                'symbol': symbol,
                'sentiment': sentiment['market_sentiment'],
                'timestamp': datetime.now()
            }
        )
        
        logger.debug(f"Published sentiment data for {symbol}")
    
    # Give the integrator a moment to process
    import time
    time.sleep(0.5)
    
    # Now generate trade signals for testing
    for symbol in symbols:
        # Simulated trade signal
        signal_type = 'BUY' if symbol in ['EUR/USD', 'GBP/USD'] else 'SELL'
        price = 1.2500 if 'EUR' in symbol else 1.5000 if 'GBP' in symbol else 110.50 if 'JPY' in symbol else 0.7500
        
        # Create stop loss price
        stop_pips = 50 if symbol in ['EUR/USD', 'GBP/USD', 'AUD/USD'] else 70  # Higher for JPY
        stop_loss = price - (stop_pips * 0.0001) if signal_type == 'BUY' else price + (stop_pips * 0.0001)
        if 'JPY' in symbol:
            stop_loss = price - (stop_pips * 0.01) if signal_type == 'BUY' else price + (stop_pips * 0.01)
        
        # Take profit at 2x the stop distance
        take_profit = price + (2 * (price - stop_loss)) if signal_type == 'BUY' else price - (2 * (stop_loss - price))
        
        # Publish trade signal
        event_bus.publish(
            EventType.TRADE_SIGNAL_RECEIVED,
            {
                'symbol': symbol,
                'signal_type': signal_type,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now(),
                'strategy': 'ExampleStrategy'
            }
        )
        
        logger.debug(f"Published trade signal for {symbol}: {signal_type} at {price}")

def generate_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Generate simulated technical indicators for a symbol."""
    import random
    
    # Different values based on symbol to create variety
    if symbol == 'EUR/USD':
        rsi = 65  # Bullish RSI
        macd = 0.002  # Positive MACD
        macd_signal = 0.001
        ema_fast = 1.2520
        ema_slow = 1.2490
    elif symbol == 'GBP/USD':
        rsi = 58  # Mildly bullish RSI
        macd = 0.001  # Small positive MACD
        macd_signal = 0.0005
        ema_fast = 1.5020
        ema_slow = 1.5010
    elif symbol == 'USD/JPY':
        rsi = 32  # Bearish RSI
        macd = -0.05  # Negative MACD
        macd_signal = -0.03
        ema_fast = 110.40
        ema_slow = 110.60
    else:  # AUD/USD
        rsi = 42  # Mildly bearish RSI
        macd = -0.001  # Small negative MACD
        macd_signal = -0.0005
        ema_fast = 0.7480
        ema_slow = 0.7510
        
    # Add some randomness to avoid exact same values
    rsi += random.uniform(-5, 5)
    macd += random.uniform(-0.0005, 0.0005)
    macd_signal += random.uniform(-0.0002, 0.0002)
    
    # Make sure RSI stays in range
    rsi = max(1, min(99, rsi))
    
    # Create indicators dictionary
    indicators = {
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd - macd_signal,
        'ema_20': ema_fast,
        'ema_50': ema_slow,
        'ema_crossover': 1 if ema_fast > ema_slow else -1 if ema_fast < ema_slow else 0,
        'volatility': random.uniform(0.5, 1.5),
        'atr': random.uniform(0.0050, 0.0150) if 'JPY' not in symbol else random.uniform(0.05, 0.15)
    }
    
    return indicators

def generate_sentiment_data(symbol: str) -> Dict[str, Dict[str, Any]]:
    """Generate simulated sentiment data for a symbol."""
    import random
    
    # Different values based on symbol to create variety
    if symbol == 'EUR/USD':
        news_score = 0.4  # Positive news
        social_score = 0.3  # Positive social
        market_score = 0.2  # Positive market
    elif symbol == 'GBP/USD':
        news_score = 0.1  # Slightly positive news
        social_score = 0.2  # Positive social
        market_score = -0.1  # Slightly negative market
    elif symbol == 'USD/JPY':
        news_score = -0.3  # Negative news
        social_score = -0.2  # Negative social
        market_score = -0.4  # Very negative market
    else:  # AUD/USD
        news_score = -0.2  # Negative news
        social_score = 0.1  # Slightly positive social
        market_score = -0.1  # Slightly negative market
        
    # Add some randomness
    news_score += random.uniform(-0.1, 0.1)
    social_score += random.uniform(-0.1, 0.1)
    market_score += random.uniform(-0.1, 0.1)
    
    # Ensure values stay in range [-1, 1]
    news_score = max(-1, min(1, news_score))
    social_score = max(-1, min(1, social_score))
    market_score = max(-1, min(1, market_score))
    
    # Create sentiment dictionary
    sentiment = {
        'news_sentiment': {
            'score': news_score,
            'confidence': random.uniform(0.6, 0.9)
        },
        'social_sentiment': {
            'score': social_score,
            'confidence': random.uniform(0.5, 0.8)
        },
        'market_sentiment': {
            'score': market_score,
            'confidence': random.uniform(0.7, 0.9)
        }
    }
    
    return sentiment

def convert_price_to_pips(symbol: str, entry_price: float, stop_loss_price: float) -> float:
    """Convert price difference to pips for a forex pair."""
    price_diff = abs(entry_price - stop_loss_price)
    
    # For JPY pairs, 1 pip = 0.01, for others 1 pip = 0.0001
    pip_value = 0.01 if 'JPY' in symbol else 0.0001
    
    # Convert to pips
    pips = price_diff / pip_value
    
    return pips

if __name__ == "__main__":
    main()
