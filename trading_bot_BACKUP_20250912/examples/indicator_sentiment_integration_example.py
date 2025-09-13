#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator and Sentiment Integration Example

This example demonstrates how to use the IndicatorSentimentIntegrator with
the indicator strategy framework to ensure the LLM evaluator and AI analysis
components have access to both technical indicators and sentiment data.
"""

import logging
import time
import os
from datetime import datetime
from typing import Dict, Any

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.adaptive_scheduler import AdaptiveScheduler, SymbolTier
from trading_bot.core.adaptive_scheduler_factory import create_scheduler
from trading_bot.core.state_manager import StateManager
from trading_bot.core.recovery_controller import RecoveryController

from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

from trading_bot.strategies.indicator.indicator_strategy_factory import IndicatorStrategyFactory
from trading_bot.strategies.indicator.indicator_data_provider import IndicatorDataProvider

from trading_bot.ai_analysis.indicator_sentiment_integrator import IndicatorSentimentIntegrator
from trading_bot.ai_analysis.llm_trade_evaluator import LLMTradeEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory for state files
STATE_DIR = os.path.join(os.path.dirname(__file__), '../state')
os.makedirs(STATE_DIR, exist_ok=True)

def main():
    """Main function to run the indicator sentiment integration example."""
    logger.info("Starting Indicator Sentiment Integration Example")
    
    # Set up core components
    setup_components()


def setup_components():
    """Set up the trading system components."""
    # Initialize event bus (central communication channel)
    event_bus = EventBus()
    
    # Initialize broker manager
    broker_manager = MultiBrokerManager()
    
    # Initialize state manager for persistence
    state_manager = StateManager(
        state_dir=STATE_DIR,
        snapshot_interval_seconds=300  # 5 minutes
    )
    
    # Initialize recovery controller
    recovery_controller = RecoveryController(
        event_bus=event_bus,
        state_dir=STATE_DIR
    )
    recovery_controller.register_state_manager(state_manager)
    
    # Initialize adaptive scheduler
    scheduler = create_scheduler(
        broker_manager=broker_manager,
        event_bus=event_bus
    )
    
    # Initialize LLM trade evaluator
    llm_evaluator = LLMTradeEvaluator(
        model="gpt-4", 
        use_mock=True  # Set to False for production with valid API key
    )
    
    # Initialize indicator strategy factory and load strategies
    strategies_config_dir = os.path.join(os.path.dirname(__file__), '../config/strategies')
    strategy_factory = IndicatorStrategyFactory(
        broker_manager=broker_manager,
        config_dir=strategies_config_dir
    )
    strategies = strategy_factory.load_all_strategies()
    logger.info(f"Loaded {len(strategies)} indicator strategies")
    
    # Get active symbols from loaded strategies
    active_symbols = list({strategy.symbol for strategy in strategies.values() if strategy.symbol})
    logger.info(f"Active symbols: {', '.join(active_symbols)}")
    
    # Initialize the Indicator Sentiment Integrator
    integrator_config = {
        'state_dir': STATE_DIR,
        'news_sentiment_weight': 0.4,
        'social_sentiment_weight': 0.3,
        'market_sentiment_weight': 0.3,
        'stale_data_seconds': 3600,  # 1 hour
        'integration_interval_seconds': 5.0,
        'min_data_points': 3
    }
    
    integrator = IndicatorSentimentIntegrator(
        event_bus=event_bus,
        llm_evaluator=llm_evaluator,
        indicator_weight=0.6,  # Technical indicators get 60% weight
        sentiment_weight=0.4,  # Sentiment gets 40% weight
        config=integrator_config,
        max_cache_size=1000,
        cache_expiry_seconds=3600
    )
    
    # Register components with recovery controller
    recovery_controller.register_component(
        "broker_manager",
        broker_manager,
        health_check_method="get_health_status",
        restart_method="restart"
    )
    
    recovery_controller.register_component(
        "scheduler",
        scheduler,
        health_check_method="get_health_status",
        restart_method="restart"
    )
    
    recovery_controller.register_component(
        "integrator",
        integrator, 
        health_check_method="handle_status_request",
        restart_method="handle_recovery_request"
    )
    
    # Set up example tasks
    setup_example_tasks(
        event_bus=event_bus, 
        scheduler=scheduler, 
        active_symbols=active_symbols
    )
    
    # Start the system
    start_system(
        event_bus=event_bus, 
        scheduler=scheduler, 
        active_symbols=active_symbols
    )


def setup_example_tasks(event_bus: EventBus, scheduler: AdaptiveScheduler, active_symbols: list):
    """Set up example tasks for the scheduler."""
    # Example task to simulate indicator updates
    scheduler.register_task(
        name="simulate_indicator_updates",
        task_function=lambda: simulate_indicator_updates(event_bus, active_symbols),
        interval_seconds=15,  # Every 15 seconds
        priority=3
    )
    
    # Example task to simulate sentiment updates
    scheduler.register_task(
        name="simulate_sentiment_updates",
        task_function=lambda: simulate_sentiment_updates(event_bus, active_symbols),
        interval_seconds=30,  # Every 30 seconds
        priority=2
    )
    
    # Example task to simulate trade signals
    scheduler.register_task(
        name="simulate_trade_signals",
        task_function=lambda: simulate_trade_signals(event_bus, active_symbols),
        interval_seconds=60,  # Every minute
        priority=1
    )
    
    # System health check task
    scheduler.register_task(
        name="system_health_check",
        task_function=lambda: event_bus.publish(EventType.SYSTEM_STATUS_REQUEST, {
            "timestamp": datetime.now()
        }),
        interval_seconds=120,  # Every 2 minutes
        priority=3
    )


def simulate_indicator_updates(event_bus: EventBus, symbols: list):
    """Simulate technical indicator updates for testing."""
    import random
    import numpy as np
    
    for symbol in symbols:
        try:
            # Generate fake indicator data
            rsi = random.uniform(30, 70)
            if random.random() < 0.2:  # 20% chance of extreme values
                rsi = random.uniform(10, 30) if random.random() < 0.5 else random.uniform(70, 90)
                
            # Generate MACD data
            macd = random.uniform(-2, 2)
            macd_signal = random.uniform(-2, 2)
            macd_hist = macd - macd_signal
            
            # Generate moving average data
            close_price = random.uniform(90, 110)
            sma_20 = close_price * (1 + random.uniform(-0.05, 0.05))
            sma_50 = close_price * (1 + random.uniform(-0.08, 0.08))
            
            # Generate Bollinger Bands
            bb_middle = close_price
            bb_std = close_price * 0.02
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Create indicator data event
            indicators = {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'close': close_price,
                'adx': random.uniform(10, 40)
            }
            
            # Publish indicator update event
            event_bus.publish(
                EventType.TECHNICAL_INDICATORS_UPDATED,
                {
                    'symbol': symbol,
                    'indicators': indicators,
                    'timestamp': datetime.now(),
                    'timeframe': '1h',
                    'source': 'simulated_data'
                }
            )
            logger.debug(f"Published indicator update for {symbol}")
            
        except Exception as e:
            logger.error(f"Error simulating indicator update for {symbol}: {str(e)}")


def simulate_sentiment_updates(event_bus: EventBus, symbols: list):
    """Simulate sentiment data updates for testing."""
    import random
    
    sentiment_types = [
        EventType.NEWS_SENTIMENT_UPDATED,
        EventType.SOCIAL_SENTIMENT_UPDATED,
        EventType.MARKET_SENTIMENT_UPDATED
    ]
    
    for symbol in symbols:
        try:
            # Choose a random sentiment type to update
            event_type = random.choice(sentiment_types)
            
            # Generate sentiment score (-1.0 to 1.0)
            # Slightly biased toward small values near zero
            sentiment_score = random.uniform(-1, 1)
            if abs(sentiment_score) < 0.3:  # 30% chance of stronger sentiment
                sentiment_score = sentiment_score * 2  # Double the score to make it stronger
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clip to -1.0 to 1.0
            
            # Create sentiment data based on type
            sentiment_data = {
                'score': sentiment_score,
                'confidence': random.uniform(0.5, 0.9)
            }
            
            # Add some sample data based on sentiment type
            if event_type == EventType.NEWS_SENTIMENT_UPDATED:
                sentiment_data['articles'] = [{
                    'title': f"Sample News for {symbol}",
                    'summary': f"This is a simulated news article about {symbol}.",
                    'source': random.choice(['Bloomberg', 'CNBC', 'Reuters', 'WSJ']),
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'sentiment': sentiment_score
                }]
            elif event_type == EventType.SOCIAL_SENTIMENT_UPDATED:
                sentiment_data['mentions'] = random.randint(10, 1000)
                sentiment_data['sources'] = ['twitter', 'reddit', 'stocktwits']
            elif event_type == EventType.MARKET_SENTIMENT_UPDATED:
                sentiment_data['market_indicators'] = {
                    'vix': random.uniform(15, 35),
                    'put_call_ratio': random.uniform(0.7, 1.3),
                    'advance_decline_ratio': random.uniform(0.5, 1.5)
                }
            
            # Publish sentiment update event
            event_bus.publish(
                event_type,
                {
                    'symbol': symbol,
                    'sentiment': sentiment_data,
                    'timestamp': datetime.now(),
                    'source': 'simulated_data'
                }
            )
            logger.debug(f"Published {event_type} for {symbol}")
            
        except Exception as e:
            logger.error(f"Error simulating sentiment update for {symbol}: {str(e)}")


def simulate_trade_signals(event_bus: EventBus, symbols: list):
    """Simulate trade signals for testing."""
    import random
    
    # Only generate signals for some symbols (not all at once)
    signal_symbols = random.sample(symbols, min(2, len(symbols)))
    
    for symbol in signal_symbols:
        try:
            # Random signal type (buy or sell)
            signal_type = random.choice(['BUY', 'SELL'])
            
            # Generate signal data
            price = random.uniform(90, 110)
            
            # Create signal event
            signal_data = {
                'symbol': symbol,
                'signal_type': signal_type,
                'price': price,
                'timestamp': datetime.now(),
                'strategy': 'Simulated_Strategy',
                'timeframe': '1h',
                'stop_loss': price * (0.95 if signal_type == 'BUY' else 1.05),
                'take_profit': price * (1.10 if signal_type == 'BUY' else 0.90)
            }
            
            # Publish trade signal event
            event_bus.publish(
                EventType.TRADE_SIGNAL_RECEIVED,
                signal_data
            )
            logger.debug(f"Published trade signal for {symbol}: {signal_type} at {price:.2f}")
            
        except Exception as e:
            logger.error(f"Error simulating trade signal for {symbol}: {str(e)}")


def start_system(event_bus: EventBus, scheduler: AdaptiveScheduler, active_symbols: list):
    """Start the system and run until interrupted."""
    try:
        # Set priority for active symbols
        for symbol in active_symbols:
            scheduler.add_symbol(symbol, SymbolTier.TIER_1)
        
        # Publish system start event
        event_bus.publish(
            EventType.SYSTEM_START,
            {
                'timestamp': datetime.now(),
                'active_symbols': active_symbols
            }
        )
        
        # Start the scheduler
        scheduler.start()
        
        logger.info("System started. Press Ctrl+C to exit.")
        
        # Run until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Publish system shutdown event
        event_bus.publish(
            EventType.SYSTEM_SHUTDOWN,
            {'timestamp': datetime.now()}
        )
        
        # Stop the scheduler
        scheduler.stop()
        
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()
