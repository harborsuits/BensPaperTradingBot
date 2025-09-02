#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Integration Runner

This script brings together all the components of the contextual integration system:
1. Data Provider
2. Market Regime Detector
3. Decision Scoring
4. Strategy Intelligence
5. Feedback Loop
6. Adaptive Position Sizing

It demonstrates how these components work together to create a highly
contextually-aware trading system.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import argparse
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContextualRunner")

# Import core components
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.decision_scoring import DecisionScorer
from trading_bot.core.smart_strategy_selector import SmartStrategySelector
from trading_bot.core.adaptive_position_manager import AdaptivePositionManager
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
from trading_bot.core.feedback_loop import StrategyFeedbackLoop
from trading_bot.core.contextual_integration import ContextualIntegrationManager

# Import data and analysis components
from trading_bot.data.contextual_data_provider import ContextualDataProvider
from trading_bot.analysis.market_regime_detector import MarketRegimeDetector
from trading_bot.strategies.forex.base.pip_based_position_sizing import PipBasedPositionSizing

# Mock persistence for testing
class MockPersistence:
    def __init__(self):
        self.stored_data = {}
        
    def save_system_state(self, component_id, state):
        self.stored_data[component_id] = state
        return True
            
    def load_system_state(self, component_id):
        return self.stored_data.get(component_id, {})

def initialize_components(config=None):
    """
    Initialize all system components.
    
    Args:
        config: Configuration options
        
    Returns:
        Dictionary with initialized components
    """
    config = config or {}
    
    # Initialize event bus
    event_bus = EventBus()
    
    # Initialize persistence
    persistence = MockPersistence()
    
    # Initialize contextual components
    data_provider = ContextualDataProvider(event_bus, persistence, 
                                          {'csv_data_path': config.get('data_path', 'data/market_data')})
    
    regime_detector = MarketRegimeDetector(event_bus, persistence)
    
    decision_scorer = DecisionScorer()
    
    strategy_intelligence = StrategyIntelligenceRecorder(event_bus, persistence)
    
    feedback_loop = StrategyFeedbackLoop(event_bus, decision_scorer, strategy_intelligence)
    
    # Initialize position sizing
    position_sizer = PipBasedPositionSizing()
    position_manager = AdaptivePositionManager(event_bus, position_sizer)
    
    # Initialize strategy selector
    strategy_selector = SmartStrategySelector(event_bus, persistence)
    
    # Register some example strategies
    test_strategies = [
        {
            "strategy_id": "strategy_001",
            "strategy_name": "TrendFollowingStrategy",
            "asset_class": "forex",
            "timeframe": "1h",
            "type": "trend_following",
            "preferred_regime": "trending_up",
            "indicators": ["ema", "macd", "rsi"],
            "performance_stats": {
                "win_rate": 0.58,
                "profit_factor": 1.75,
                "regime_performance": {
                    "trending_up": 0.82,
                    "trending_down": 0.65,
                    "ranging": 0.31,
                    "breakout": 0.45,
                    "unknown": 0.51
                }
            }
        },
        {
            "strategy_id": "strategy_002",
            "strategy_name": "MeanReversionStrategy",
            "asset_class": "forex",
            "timeframe": "1h",
            "type": "mean_reversion",
            "preferred_regime": "ranging",
            "indicators": ["bollinger_bands", "rsi", "stochastic"],
            "performance_stats": {
                "win_rate": 0.61,
                "profit_factor": 1.62,
                "regime_performance": {
                    "trending_up": 0.40,
                    "trending_down": 0.38,
                    "ranging": 0.79,
                    "breakout": 0.25,
                    "unknown": 0.52
                }
            }
        },
        {
            "strategy_id": "strategy_003",
            "strategy_name": "BreakoutStrategy",
            "asset_class": "forex",
            "timeframe": "1h",
            "type": "breakout",
            "preferred_regime": "breakout",
            "indicators": ["atr", "donchian", "volume_profile"],
            "performance_stats": {
                "win_rate": 0.54,
                "profit_factor": 1.85,
                "regime_performance": {
                    "trending_up": 0.55,
                    "trending_down": 0.58,
                    "ranging": 0.32,
                    "breakout": 0.88,
                    "unknown": 0.50
                }
            }
        },
        {
            "strategy_id": "strategy_004",
            "strategy_name": "DowntrendStrategy",
            "asset_class": "forex",
            "timeframe": "1h",
            "type": "trend_following",
            "preferred_regime": "trending_down",
            "indicators": ["ema", "rsi", "adx"],
            "performance_stats": {
                "win_rate": 0.56,
                "profit_factor": 1.68,
                "regime_performance": {
                    "trending_up": 0.42,
                    "trending_down": 0.81,
                    "ranging": 0.35,
                    "breakout": 0.48,
                    "unknown": 0.49
                }
            }
        },
    ]
    
    for strategy in test_strategies:
        strategy_selector.register_strategy(strategy)
    
    # Initialize the contextual integration manager
    contextual_manager = ContextualIntegrationManager(
        event_bus=event_bus,
        persistence=persistence,
        strategy_intelligence=strategy_intelligence,
        decision_scorer=decision_scorer,
        strategy_selector=strategy_selector,
        position_manager=position_manager,
        position_sizer=position_sizer
    )
    
    # Return all components
    return {
        "event_bus": event_bus,
        "persistence": persistence,
        "data_provider": data_provider,
        "regime_detector": regime_detector,
        "decision_scorer": decision_scorer,
        "strategy_intelligence": strategy_intelligence,
        "feedback_loop": feedback_loop,
        "position_sizer": position_sizer,
        "position_manager": position_manager,
        "strategy_selector": strategy_selector,
        "contextual_manager": contextual_manager
    }

def load_market_data(symbols, config=None):
    """
    Load market data for testing.
    
    Args:
        symbols: List of symbols to load
        config: Configuration options
        
    Returns:
        Dictionary of market data
    """
    config = config or {}
    market_data = {}
    
    # Create data directory if it doesn't exist
    data_path = config.get('data_path', 'data/market_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    
    # Load data from CSV files
    for symbol in symbols:
        file_path = os.path.join(data_path, f"{symbol}_1h.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            market_data[symbol] = df
    
    # Check if we have data
    if not market_data:
        logger.warning("No market data found. Run regime_detector_test.py first to generate data.")
    
    return market_data

def run_context_integration_example(components, market_data, symbols, config=None):
    """
    Run a simple example of the contextual integration system.
    
    Args:
        components: Dictionary of initialized components
        market_data: Dictionary of market data
        symbols: List of symbols to process
        config: Configuration options
        
    Returns:
        Results dictionary
    """
    config = config or {}
    
    # Extract components
    event_bus = components["event_bus"]
    contextual_manager = components["contextual_manager"]
    data_provider = components["data_provider"]
    regime_detector = components["regime_detector"]
    
    # Events listener to track what happens
    events_log = {
        EventType.MARKET_REGIME_CHANGE: [],
        EventType.VOLATILITY_UPDATE: [],
        EventType.SIGNAL_GENERATED: [],
        EventType.TRADE_EXECUTED: [],
        EventType.TRADE_CLOSED: [],
        EventType.STRATEGY_SELECTED: [],
        EventType.CONTEXT_UPDATED: []
    }
    
    def event_listener(event):
        if event.event_type in events_log:
            events_log[event.event_type].append({
                'timestamp': datetime.now(),
                'data': event.data
            })
    
    # Subscribe to all interesting events
    for event_type in events_log.keys():
        event_bus.subscribe(event_type, event_listener)
    
    # Process each symbol
    for symbol in symbols:
        if symbol not in market_data:
            logger.warning(f"No data for {symbol}, skipping")
            continue
        
        logger.info(f"Processing {symbol} with contextual integration...")
        
        # Get data
        df = market_data[symbol]
        
        # Process in chunks to simulate streaming data
        chunk_size = 24  # One day at a time
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            if len(chunk) < 2:  # Need at least 2 bars for meaningful processing
                continue
            
            # Process this chunk of data
            logger.info(f"Processing data chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}")
            
            # 1. Provide data to data provider
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
            
            # 2. This triggers regime detection automatically via event handlers
            
            # 3. Every 6 hours, make a trading decision
            for j in range(0, len(chunk), 6):
                if j + 1 >= len(chunk):
                    continue
                
                # Current bar
                current_bar = chunk.iloc[j]
                current_time = chunk.index[j]
                
                # Get the current regime
                context = contextual_manager.get_current_context()
                market_regime = context.get('market_regime', 'unknown')
                volatility_state = context.get('volatility_state', 'medium')
                
                logger.info(f"Current regime: {market_regime}, Volatility: {volatility_state}")
                
                # Make trading decisions
                current_price = current_bar['close']
                
                # 1. Select optimal strategy
                selected_strategy = contextual_manager.select_optimal_strategy(
                    symbol=symbol,
                    market_data={
                        'symbol': symbol,
                        'current_price': current_price,
                        'timestamp': current_time
                    }
                )
                
                if selected_strategy:
                    strategy_id = selected_strategy.get('strategy_id')
                    strategy_name = selected_strategy.get('strategy_name')
                    strategy_score = selected_strategy.get('score', 0)
                    
                    logger.info(f"Selected strategy: {strategy_name} (score: {strategy_score:.2f})")
                    
                    # 2. Calculate position size
                    position_result = contextual_manager.calculate_position_size(
                        symbol=symbol,
                        account_balance=context.get('account_balance', 5000.0),
                        entry_price=current_price,
                        stop_loss_pips=20,
                        asset_class="forex"
                    )
                    
                    position_size = position_result.get('position_size', 0)
                    risk_amount = position_result.get('risk_amount', 0)
                    risk_percentage = position_result.get('risk_percentage', 0)
                    
                    logger.info(f"Position sizing: {position_size:.4f} lots, " +
                              f"risking ${risk_amount:.2f} ({risk_percentage:.2f}%)")
                    
                    # 3. Generate signal
                    signal_id = f"{symbol}_{current_time.strftime('%Y%m%d%H%M%S')}_{strategy_id}"
                    
                    # Determine direction based on regime
                    if market_regime == 'trending_up':
                        direction = 'buy'
                    elif market_regime == 'trending_down':
                        direction = 'sell'
                    else:
                        # Alternate between buy and sell
                        direction = 'buy' if j % 2 == 0 else 'sell'
                    
                    # Publish signal event
                    event_bus.publish(Event(
                        event_type=EventType.SIGNAL_GENERATED,
                        data={
                            'signal_id': signal_id,
                            'strategy_id': strategy_id,
                            'strategy_name': strategy_name,
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': current_price,
                            'confidence': strategy_score,
                            'timestamp': current_time
                        }
                    ))
                    
                    # 4. Simulate trade execution
                    trade_id = f"trade_{signal_id}"
                    
                    event_bus.publish(Event(
                        event_type=EventType.TRADE_EXECUTED,
                        data={
                            'trade_id': trade_id,
                            'signal_id': signal_id,
                            'strategy_id': strategy_id,
                            'strategy_name': strategy_name,
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': current_price,
                            'position_size': position_size,
                            'risk_amount': risk_amount,
                            'stop_loss_pips': 20,
                            'take_profit_pips': 40,
                            'timestamp': current_time
                        }
                    ))
                    
                    # 5. Simulate trade outcome after a few bars
                    outcome_time = chunk.index[min(j + 4, len(chunk) - 1)]
                    outcome_price = chunk.iloc[min(j + 4, len(chunk) - 1)]['close']
                    
                    # Calculate PnL
                    if direction == 'buy':
                        price_change = outcome_price - current_price
                    else:  # sell
                        price_change = current_price - outcome_price
                    
                    # Simplified PnL calculation
                    pip_value = 0.0001
                    pips = price_change / pip_value
                    pip_value_amount = risk_amount / 20  # risk per pip
                    pnl = pips * pip_value_amount
                    
                    # Determine outcome
                    outcome = 'win' if pnl > 0 else 'loss'
                    
                    # Adjust account balance
                    new_balance = context.get('account_balance', 5000.0) + pnl
                    
                    # Publish account update
                    event_bus.publish(Event(
                        event_type=EventType.ACCOUNT_BALANCE_UPDATE,
                        data={
                            'balance': new_balance,
                            'previous_balance': context.get('account_balance', 5000.0),
                            'initial_balance': 5000.0,
                            'timestamp': outcome_time
                        }
                    ))
                    
                    # Publish trade closed event
                    event_bus.publish(Event(
                        event_type=EventType.TRADE_CLOSED,
                        data={
                            'trade_id': trade_id,
                            'signal_id': signal_id,
                            'strategy_id': strategy_id,
                            'strategy_name': strategy_name,
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': current_price,
                            'exit_price': outcome_price,
                            'entry_time': current_time,
                            'exit_time': outcome_time,
                            'position_size': position_size,
                            'pnl': pnl,
                            'outcome': outcome,
                            'market_regime': market_regime,
                            'volatility_state': volatility_state,
                            'timestamp': outcome_time
                        }
                    ))
                    
                    logger.info(f"Trade outcome: {outcome.upper()} with PnL ${pnl:.2f}")
            
            # Small delay to allow events to process
            time.sleep(0.1)
    
    # Generate summary
    summary = {
        'regime_changes': len(events_log[EventType.MARKET_REGIME_CHANGE]),
        'volatility_updates': len(events_log[EventType.VOLATILITY_UPDATE]),
        'signals_generated': len(events_log[EventType.SIGNAL_GENERATED]),
        'trades_executed': len(events_log[EventType.TRADE_EXECUTED]),
        'trades_closed': len(events_log[EventType.TRADE_CLOSED]),
        'context_updates': len(events_log[EventType.CONTEXT_UPDATED]),
        'events_log': events_log
    }
    
    # Print summary
    logger.info("\n=== CONTEXTUAL INTEGRATION SUMMARY ===")
    logger.info(f"Regime Changes: {summary['regime_changes']}")
    logger.info(f"Volatility Updates: {summary['volatility_updates']}")
    logger.info(f"Signals Generated: {summary['signals_generated']}")
    logger.info(f"Trades Executed: {summary['trades_executed']}")
    logger.info(f"Trades Closed: {summary['trades_closed']}")
    logger.info(f"Context Updates: {summary['context_updates']}")
    
    # Calculate trade statistics
    if events_log[EventType.TRADE_CLOSED]:
        wins = sum(1 for t in events_log[EventType.TRADE_CLOSED] if t['data']['outcome'] == 'win')
        losses = len(events_log[EventType.TRADE_CLOSED]) - wins
        win_rate = (wins / len(events_log[EventType.TRADE_CLOSED])) * 100 if events_log[EventType.TRADE_CLOSED] else 0
        total_pnl = sum(t['data']['pnl'] for t in events_log[EventType.TRADE_CLOSED])
        
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        
        # Performance by regime
        regime_performance = {}
        for trade in events_log[EventType.TRADE_CLOSED]:
            regime = trade['data'].get('market_regime', 'unknown')
            if regime not in regime_performance:
                regime_performance[regime] = {'trades': 0, 'wins': 0, 'pnl': 0}
            
            regime_performance[regime]['trades'] += 1
            if trade['data']['outcome'] == 'win':
                regime_performance[regime]['wins'] += 1
            regime_performance[regime]['pnl'] += trade['data']['pnl']
        
        logger.info("\nPerformance by Market Regime:")
        for regime, perf in regime_performance.items():
            regime_win_rate = (perf['wins'] / perf['trades']) * 100 if perf['trades'] > 0 else 0
            logger.info(f"{regime}: {perf['trades']} trades, {regime_win_rate:.2f}% win rate, ${perf['pnl']:.2f} PnL")
        
        # Add trade statistics to summary
        summary.update({
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'regime_performance': regime_performance
        })
    
    return summary

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Contextual Integration Runner")
    parser.add_argument('--symbols', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY'],
                        help='Symbols to process')
    parser.add_argument('--data-path', default='data/market_data',
                        help='Path to market data directory')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path
    }
    
    # Initialize components
    logger.info("Initializing contextual integration components...")
    components = initialize_components(config)
    
    # Load market data
    logger.info(f"Loading market data for {args.symbols}...")
    market_data = load_market_data(args.symbols, config)
    
    if not market_data:
        logger.error("No market data found. Please run regime_detector_test.py first.")
        return 1
    
    # Run example
    logger.info("Running contextual integration example...")
    results = run_context_integration_example(components, market_data, args.symbols, config)
    
    logger.info("Contextual integration example complete!")
    return 0

if __name__ == "__main__":
    main()
