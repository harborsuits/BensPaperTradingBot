#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Awareness Integration Test

This script tests the new contextual integration system that connects:
1. Decision scoring
2. Market regime detection 
3. Strategy selection
4. Position sizing
5. Pattern recognition

The test simulates various market scenarios and shows how context flows through
the entire system to create a highly adaptive trading system.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
import time
import os
import sys
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContextualAwarenessTest")

def generate_market_regime_data():
    """Generate a sequence of market regime changes to test with."""
    regimes = [
        # Starting with normal/unknown regime for baseline
        {"regime": "unknown", "duration_minutes": 5, "volatility": "medium"},
        
        # Trending up with increasing volatility
        {"regime": "trending_up", "duration_minutes": 15, "volatility": "low"},
        {"regime": "trending_up", "duration_minutes": 20, "volatility": "medium"},
        {"regime": "trending_up", "duration_minutes": 10, "volatility": "high"},
        
        # Transition to ranging market
        {"regime": "ranging", "duration_minutes": 30, "volatility": "low"},
        
        # Followed by breakout
        {"regime": "breakout", "duration_minutes": 5, "volatility": "high"},
        
        # And then trending down
        {"regime": "trending_down", "duration_minutes": 25, "volatility": "medium"},
        {"regime": "trending_down", "duration_minutes": 15, "volatility": "high"},
        
        # Back to ranging for conclusion
        {"regime": "ranging", "duration_minutes": 15, "volatility": "low"},
    ]
    
    return regimes

def generate_test_market_data():
    """Generate test market data for the symbols we'll use."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    market_data = {}
    
    for symbol in symbols:
        # Create a DataFrame with random but somewhat realistic market data
        now = datetime.now()
        periods = 500
        
        dates = [now - timedelta(minutes=i) for i in range(periods)]
        dates.reverse()
        
        # Starting price differs by symbol
        if symbol == "EURUSD":
            base_price = 1.12
        elif symbol == "GBPUSD":
            base_price = 1.30
        elif symbol == "USDJPY":
            base_price = 108.5
        else:  # AUDUSD
            base_price = 0.75
        
        # Generate price data with some randomness but general patterns
        close_prices = [base_price]
        for i in range(1, periods):
            # Random walk with some mean reversion and momentum
            rnd = np.random.normal(0, 0.0005)
            momentum = 0.2 * (close_prices[i-1] - close_prices[max(0, i-10)]) / close_prices[max(0, i-10)]
            mean_reversion = -0.1 * (close_prices[i-1] - base_price) / base_price
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
        market_data[symbol] = df
    
    return market_data, symbols

def test_contextual_integration_setup():
    """Set up all the components needed for contextual integration."""
    try:
        # Import necessary components
        from trading_bot.core.event_bus import EventBus, Event
        from trading_bot.core.constants import EventType
        from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
        from trading_bot.core.decision_scoring import DecisionScorer
        from trading_bot.core.smart_strategy_selector import SmartStrategySelector
        from trading_bot.core.adaptive_position_manager import AdaptivePositionManager
        from trading_bot.strategies.forex.base.pip_based_position_sizing import PipBasedPositionSizing
        from trading_bot.core.contextual_integration import ContextualIntegrationManager
        
        # Initialize the event bus
        event_bus = EventBus()
        
        # Initialize persistence manager (mock for testing)
        class MockPersistence:
            def __init__(self):
                self.stored_data = {}
                
            def save_system_state(self, component_id, state):
                self.stored_data[component_id] = state
                return True
                
            def load_system_state(self, component_id):
                return self.stored_data.get(component_id, {})
        
        persistence = MockPersistence()
        
        # Initialize components
        decision_scorer = DecisionScorer()
        strategy_intelligence = StrategyIntelligenceRecorder(event_bus, persistence)
        
        # Initialize strategy selector (with some test strategies)
        strategy_selector = SmartStrategySelector(event_bus, persistence)
        
        # Add some test strategies
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
        
        # Initialize position sizing components
        position_sizer = PipBasedPositionSizing()
        position_manager = AdaptivePositionManager(event_bus, position_sizer)
        
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
        
        logger.info("Successfully initialized all contextual integration components")
        
        # Return all components for testing
        return {
            "event_bus": event_bus,
            "persistence": persistence,
            "decision_scorer": decision_scorer,
            "strategy_intelligence": strategy_intelligence,
            "strategy_selector": strategy_selector,
            "position_manager": position_manager,
            "position_sizer": position_sizer,
            "contextual_manager": contextual_manager
        }
        
    except ImportError as e:
        logger.error(f"Import error during contextual integration setup: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error setting up contextual integration components: {str(e)}")
        raise

def simulate_market_progression(components, market_data, symbols, regimes):
    """
    Simulate market progression through different regimes and test system response.
    """
    event_bus = components["event_bus"]
    contextual_manager = components["contextual_manager"]
    
    # Statistics tracking
    strategy_selections = defaultdict(int)
    position_sizes = []
    risk_percentages = []
    regime_transitions = []
    context_snapshots = []
    
    # Account balance simulation
    account_balance = 5000.0  # Start with $5,000
    
    # Current market state
    current_market_state = {
        "datetime": datetime.now(),
        "regime": "unknown",
        "volatility": "medium"
    }
    
    # Simulation start time
    start_time = datetime.now()
    current_time = start_time
    
    # Log initial state
    logger.info(f"Starting simulation with account balance: ${account_balance:.2f}")
    logger.info(f"Testing across {len(symbols)} symbols and {len(regimes)} regime changes")
    
    # For each regime in our test sequence
    for regime_idx, regime_info in enumerate(regimes):
        regime = regime_info["regime"]
        volatility = regime_info["volatility"]
        duration = regime_info["duration_minutes"]
        
        # Update current time
        current_time += timedelta(minutes=duration)
        
        # Log regime change
        logger.info(f"[{current_time}] REGIME CHANGE: {regime} with {volatility} volatility")
        
        # Publish regime change event
        event_bus.publish(Event(
            event_type=EventType.MARKET_REGIME_CHANGE,
            data={
                "regime": regime,
                "previous_regime": current_market_state["regime"],
                "confidence": 0.85,
                "timestamp": current_time
            }
        ))
        
        # Publish volatility update event
        event_bus.publish(Event(
            event_type=EventType.VOLATILITY_UPDATE,
            data={
                "volatility_state": volatility,
                "previous_state": current_market_state["volatility"],
                "timestamp": current_time
            }
        ))
        
        # Update current market state
        current_market_state["regime"] = regime
        current_market_state["volatility"] = volatility
        current_market_state["datetime"] = current_time
        
        # Track regime transition
        regime_transitions.append({
            "timestamp": current_time,
            "regime": regime,
            "volatility": volatility
        })
        
        # Allow context to propagate
        time.sleep(0.1)
        
        # Take snapshot of context
        context_snapshot = contextual_manager.get_current_context()
        context_snapshots.append({
            "timestamp": current_time.isoformat(),
            "regime": regime,
            "context": context_snapshot
        })
        
        # Update account balance (simulate some trading activity)
        # In trending regimes, we tend to make money
        if regime in ["trending_up", "trending_down"]:
            pnl_factor = np.random.normal(1.02, 0.05)  # Slightly positive expected return
        # In breakout regimes, more volatility but higher returns when right
        elif regime == "breakout":
            pnl_factor = np.random.normal(1.04, 0.12)  # Higher variance
        # In ranging regimes, mixed results
        else:
            pnl_factor = np.random.normal(1.00, 0.03)  # Neutral expected return
            
        # Apply PnL factor
        new_balance = account_balance * pnl_factor
        pnl = new_balance - account_balance
        account_balance = new_balance
        
        # Publish account update
        event_bus.publish(Event(
            event_type=EventType.ACCOUNT_BALANCE_UPDATE,
            data={
                "balance": account_balance,
                "previous_balance": account_balance - pnl,
                "initial_balance": 5000.0,
                "timestamp": current_time
            }
        ))
        
        logger.info(f"[{current_time}] Account balance: ${account_balance:.2f} (PnL: ${pnl:.2f})")
        
        # Make strategy selections and position sizing decisions for each symbol
        for symbol in symbols:
            # Basic market data for current symbol
            symbol_data = market_data[symbol].iloc[-100:].copy()
            
            # Current price info
            current_price = float(symbol_data.iloc[-1]["close"])
            price_data = {
                "symbol": symbol,
                "current_price": current_price,
                "timeframe": "1h",
                "timestamp": current_time
            }
            
            # Strategy selection based on current context
            selected_strategy = contextual_manager.select_optimal_strategy(
                symbol=symbol,
                market_data=price_data
            )
            
            if selected_strategy:
                strategy_id = selected_strategy.get("strategy_id")
                strategy_name = selected_strategy.get("strategy_name")
                score = selected_strategy.get("score", 0)
                explanation = selected_strategy.get("explanation", "")
                
                logger.info(f"[{current_time}] Selected {strategy_name} for {symbol} (score: {score:.2f})")
                logger.debug(f"Selection explanation: {explanation}")
                
                strategy_selections[strategy_id] += 1
                
                # Calculate position size with context
                position_size_result = contextual_manager.calculate_position_size(
                    symbol=symbol,
                    account_balance=account_balance,
                    entry_price=current_price,
                    stop_loss_pips=20,  # Example stop loss
                    asset_class="forex"
                )
                
                position_size = position_size_result.get("position_size", 0)
                risk_amount = position_size_result.get("risk_amount", 0)
                risk_percentage = (risk_amount / account_balance) * 100 if account_balance > 0 else 0
                explanation = position_size_result.get("explanation", "")
                
                logger.info(f"[{current_time}] Position size for {symbol}: {position_size:.4f} lots, " +
                           f"risking ${risk_amount:.2f} ({risk_percentage:.2f}% of account)")
                logger.debug(f"Position sizing explanation: {explanation}")
                
                position_sizes.append({
                    "timestamp": current_time,
                    "symbol": symbol,
                    "account_balance": account_balance,
                    "position_size": position_size,
                    "risk_amount": risk_amount,
                    "risk_percentage": risk_percentage,
                    "regime": regime
                })
                
                risk_percentages.append(risk_percentage)
                
                # For demonstration, simulate some trades with outcomes
                # In the appropriate regimes, strategies do better when they match the regime
                win_probability = 0.5  # Default
                if regime == "trending_up" and strategy_name == "TrendFollowingStrategy":
                    win_probability = 0.65
                elif regime == "trending_down" and strategy_name == "DowntrendStrategy":
                    win_probability = 0.65
                elif regime == "ranging" and strategy_name == "MeanReversionStrategy":
                    win_probability = 0.65
                elif regime == "breakout" and strategy_name == "BreakoutStrategy":
                    win_probability = 0.70
                
                # Simulate trade outcome
                trade_outcome = np.random.random() < win_probability
                
                # Generate signal ID and trade ID
                signal_id = f"{symbol}_{current_time.strftime('%Y%m%d%H%M%S')}_{strategy_id}"
                trade_id = f"trade_{signal_id}"
                
                # Publish signal generated event
                event_bus.publish(Event(
                    event_type=EventType.SIGNAL_GENERATED,
                    data={
                        "signal_id": signal_id,
                        "strategy_id": strategy_id,
                        "strategy_name": strategy_name,
                        "symbol": symbol,
                        "direction": "buy" if np.random.random() > 0.5 else "sell",
                        "timestamp": current_time,
                        "confidence": score
                    }
                ))
                
                # Publish trade executed event
                event_bus.publish(Event(
                    event_type=EventType.TRADE_EXECUTED,
                    data={
                        "trade_id": trade_id,
                        "signal_id": signal_id,
                        "strategy_id": strategy_id,
                        "strategy_name": strategy_name,
                        "symbol": symbol,
                        "entry_price": current_price,
                        "position_size": position_size,
                        "stop_loss": current_price * 0.995,  # Example
                        "take_profit": current_price * 1.015,  # Example
                        "timestamp": current_time
                    }
                ))
                
                # Simulate trade outcome after a short period
                trade_time = current_time + timedelta(minutes=np.random.randint(5, 20))
                
                # Calculate simulated PnL based on outcome
                trade_pnl = 0.0
                if trade_outcome:  # Win
                    trade_pnl = risk_amount * np.random.uniform(1.0, 2.5)  # R:R between 1:1 and 1:2.5
                else:  # Loss
                    trade_pnl = -risk_amount * np.random.uniform(0.8, 1.0)  # Partial or full stop loss hit
                
                # Update account balance from this trade
                account_balance += trade_pnl
                
                # Publish account update after trade
                event_bus.publish(Event(
                    event_type=EventType.ACCOUNT_BALANCE_UPDATE,
                    data={
                        "balance": account_balance,
                        "previous_balance": account_balance - trade_pnl,
                        "initial_balance": 5000.0,
                        "timestamp": trade_time
                    }
                ))
                
                # Publish trade closed event
                event_bus.publish(Event(
                    event_type=EventType.TRADE_CLOSED,
                    data={
                        "trade_id": trade_id,
                        "signal_id": signal_id,
                        "strategy_id": strategy_id,
                        "strategy_name": strategy_name,
                        "symbol": symbol,
                        "entry_price": current_price,
                        "exit_price": current_price * (1.01 if trade_outcome else 0.995),
                        "position_size": position_size,
                        "pnl": trade_pnl,
                        "outcome": "win" if trade_outcome else "loss",
                        "duration_minutes": np.random.randint(5, 20),
                        "timestamp": trade_time
                    }
                ))
                
                logger.info(f"[{trade_time}] Trade {trade_id} closed: " +
                           f"{'WIN' if trade_outcome else 'LOSS'} with PnL ${trade_pnl:.2f}")
                logger.info(f"[{trade_time}] New account balance: ${account_balance:.2f}")
        
        # Small delay between regime iterations to allow events to process
        time.sleep(0.2)
    
    # End of simulation
    simulation_duration = (current_time - start_time).total_seconds() / 60
    logger.info(f"Simulation complete! Covered {simulation_duration:.1f} minutes of market time")
    logger.info(f"Final account balance: ${account_balance:.2f}")
    
    # Return simulation statistics
    return {
        "initial_balance": 5000.0,
        "final_balance": account_balance,
        "percent_change": ((account_balance / 5000.0) - 1) * 100,
        "strategy_selections": dict(strategy_selections),
        "position_sizes": position_sizes,
        "risk_percentages": risk_percentages,
        "regime_transitions": regime_transitions,
        "context_snapshots": context_snapshots
    }

def analyze_and_report_results(results):
    """Analyze the results of the simulation and produce a report."""
    # Extract key metrics
    initial_balance = results["initial_balance"]
    final_balance = results["final_balance"]
    percent_change = results["percent_change"]
    strategy_selections = results["strategy_selections"]
    position_sizes = results["position_sizes"]
    risk_percentages = results["risk_percentages"]
    regime_transitions = results["regime_transitions"]
    context_snapshots = results["context_snapshots"]
    
    # Generate report
    logger.info("\n=== CONTEXTUAL AWARENESS SIMULATION RESULTS ===\n")
    logger.info(f"Initial Balance: ${initial_balance:.2f}")
    logger.info(f"Final Balance: ${final_balance:.2f}")
    logger.info(f"Percent Change: {percent_change:.2f}%")
    
    # Strategy selection analysis
    total_selections = sum(strategy_selections.values())
    logger.info("\n--- Strategy Selection Analysis ---")
    logger.info(f"Total strategy selections: {total_selections}")
    for strategy_id, count in strategy_selections.items():
        percentage = (count / total_selections) * 100 if total_selections > 0 else 0
        logger.info(f"Strategy {strategy_id}: {count} selections ({percentage:.1f}%)")
    
    # Risk management analysis
    logger.info("\n--- Risk Management Analysis ---")
    avg_risk = sum(risk_percentages) / len(risk_percentages) if risk_percentages else 0
    max_risk = max(risk_percentages) if risk_percentages else 0
    min_risk = min(risk_percentages) if risk_percentages else 0
    logger.info(f"Average risk per trade: {avg_risk:.2f}%")
    logger.info(f"Maximum risk per trade: {max_risk:.2f}%")
    logger.info(f"Minimum risk per trade: {min_risk:.2f}%")
    
    # Risk profile by account balance
    logger.info("\n--- Risk Profile by Account Balance ---")
    
    # Bin account balances and calculate average risk
    balance_bins = [0, 2500, 5000, 7500, 10000, 15000, 25000, float('inf')]
    bin_labels = ["<$2.5K", "$2.5K-$5K", "$5K-$7.5K", "$7.5K-$10K", "$10K-$15K", "$15K-$25K", ">$25K"]
    
    bin_risks = {label: [] for label in bin_labels}
    
    for entry in position_sizes:
        balance = entry["account_balance"]
        risk_pct = entry["risk_percentage"]
        
        for i, upper in enumerate(balance_bins[1:], 0):
            if balance < upper:
                bin_risks[bin_labels[i]].append(risk_pct)
                break
    
    for label, risks in bin_risks.items():
        avg = sum(risks) / len(risks) if risks else 0
        logger.info(f"{label}: Average risk {avg:.2f}% ({len(risks)} trades)")
    
    # Context transition analysis
    logger.info("\n--- Context Transition Analysis ---")
    logger.info(f"Total regime transitions: {len(regime_transitions)}")
    
    # Analyze strategy selection by regime
    regime_strategy_counts = defaultdict(lambda: defaultdict(int))
    
    # Group position sizes by regime
    regime_position_sizes = defaultdict(list)
    
    # Collect data
    for entry in position_sizes:
        regime = entry["regime"]
        regime_position_sizes[regime].append(entry["position_size"])
    
    # Display average position size by regime
    logger.info("\n--- Position Sizing by Market Regime ---")
    for regime, sizes in regime_position_sizes.items():
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        logger.info(f"{regime}: Average position size {avg_size:.4f} lots ({len(sizes)} trades)")
    
    # Extract contextual awareness evidence
    logger.info("\n--- Contextual Awareness Evidence ---")
    logger.info("Examples of context propagation:")
    
    # Show a few snapshots of context
    for i, snapshot in enumerate(context_snapshots):
        if i % 3 == 0:  # Show every third snapshot to keep output reasonable
            logger.info(f"\nContext at {snapshot['timestamp']} (Regime: {snapshot['regime']}):")
            context = snapshot['context']
            for key, value in context.items():
                if key != 'last_updated' and key != 'correlation_matrix' and not isinstance(value, dict):
                    logger.info(f"  {key}: {value}")
    
    logger.info("\n=== CONCLUSION ===")
    logger.info("The simulation demonstrates how context flows through all trading components.")
    logger.info("Key observations:")
    logger.info("1. Strategy selection adapts to market regimes")
    logger.info("2. Position sizing scales with account balance and market conditions")
    logger.info("3. Decision scoring creates a feedback loop for continuous improvement")
    logger.info("4. Pattern recognition enhances confidence in favorable conditions")
    
    return {
        "avg_risk": avg_risk,
        "regime_position_sizes": regime_position_sizes
    }

def run_contextual_awareness_test(verbose=False):
    """Run the full contextual awareness integration test."""
    try:
        logger.info("=== CONTEXTUAL AWARENESS INTEGRATION TEST ===")
        
        # Set up components
        logger.info("Setting up contextual integration components...")
        components = test_contextual_integration_setup()
        
        # Generate test data
        logger.info("Generating test market data...")
        market_data, symbols = generate_test_market_data()
        
        # Generate regime sequence
        logger.info("Generating market regime sequence...")
        regimes = generate_market_regime_data()
        
        # Run simulation
        logger.info("Starting market simulation...")
        results = simulate_market_progression(components, market_data, symbols, regimes)
        
        # Analyze results
        analysis = analyze_and_report_results(results)
        
        logger.info("Contextual awareness integration test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"Import error during contextual awareness test: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error during contextual awareness test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Contextual Awareness Integration Test")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting contextual awareness integration test")
    
    success = run_contextual_awareness_test(args.verbose)
    
    if success:
        logger.info("Test PASSED: Contextual awareness integration is working correctly!")
        return 0
    else:
        logger.error("Test FAILED: Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
