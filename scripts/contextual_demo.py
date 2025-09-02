#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Integration Demo

This is a simplified version that demonstrates the core functionality
of the contextual integration system without external dependencies.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContextDemo")

#----------------
# Mock Classes to simulate our trading environment
#----------------

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
    ACCOUNT_BALANCE_UPDATE = "account_balance_update"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"
    PATTERN_DETECTED = "pattern_detected"
    CONTEXT_UPDATED = "context_updated"
    DECISION_SCORED = "decision_scored"
    PATTERN_CONFIDENCE = "pattern_confidence"
    
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

class RiskProfile:
    """Mock Risk Profile constants"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SECURITY_FOCUSED = "security_focused"  # Our custom profile
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"

#----------------
# Simplified Contextual Integration Manager
#----------------

class ContextualIntegrationManager:
    """
    Simplified version of the Contextual Integration Manager for demonstration
    """
    
    def __init__(self, event_bus):
        """Initialize with just the event bus for demonstration"""
        self.event_bus = event_bus
        
        # Current context state
        self.current_context = {
            'market_regime': 'unknown',
            'volatility_state': 'medium',
            'performance_state': 'neutral',
            'drawdown_percentage': 0.0,
            'account_balance': 5000.0,
            'account_multiplier': 1.0,
            'last_updated': datetime.now()
        }
        
        # Available strategies for demonstration
        self.available_strategies = [
            {
                "strategy_id": "trend_following",
                "strategy_name": "Trend Following Strategy",
                "preferred_regime": "trending_up"
            },
            {
                "strategy_id": "mean_reversion",
                "strategy_name": "Mean Reversion Strategy",
                "preferred_regime": "ranging"
            },
            {
                "strategy_id": "breakout",
                "strategy_name": "Breakout Strategy",
                "preferred_regime": "breakout"
            },
            {
                "strategy_id": "downtrend",
                "strategy_name": "Downtrend Strategy",
                "preferred_regime": "trending_down"
            }
        ]
        
        # Register for events
        self._subscribe_to_events()
        
        logger.info("Contextual Integration Manager initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events from the event bus."""
        # Market state events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self.handle_market_regime_change)
        self.event_bus.subscribe(EventType.VOLATILITY_UPDATE, self.handle_volatility_update)
        
        # Performance events
        self.event_bus.subscribe(EventType.ACCOUNT_BALANCE_UPDATE, self.handle_account_update)
        
        # Trading events
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self.handle_signal_generated)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        
        # Pattern events
        self.event_bus.subscribe(EventType.PATTERN_DETECTED, self.handle_pattern_detected)
        
        logger.info("Subscribed to relevant events")
    
    def handle_market_regime_change(self, event):
        """Handle market regime change events."""
        regime = event.data.get('regime', 'unknown')
        self.current_context['market_regime'] = regime
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Market regime updated to: {regime}")
    
    def handle_volatility_update(self, event):
        """Handle volatility update events."""
        volatility = event.data.get('volatility_state', 'medium')
        self.current_context['volatility_state'] = volatility
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Volatility state updated to: {volatility}")
    
    def handle_account_update(self, event):
        """Handle account balance update events."""
        account_balance = event.data.get('balance', 0.0)
        previous_balance = event.data.get('previous_balance', 0.0)
        
        # Store balance
        self.current_context['account_balance'] = account_balance
        
        # Calculate account multiplier (for progressive risk scaling)
        initial_balance = 5000.0  # Demo hard-coded value
        if initial_balance > 0:
            account_multiplier = account_balance / initial_balance
            self.current_context['account_multiplier'] = account_multiplier
        
        # Calculate performance state based on recent change
        if previous_balance > 0:
            percent_change = (account_balance - previous_balance) / previous_balance
            
            if percent_change > 0.05:
                performance_state = 'strong_up'
            elif percent_change > 0.01:
                performance_state = 'up'
            elif percent_change < -0.05:
                performance_state = 'strong_down'
            elif percent_change < -0.01:
                performance_state = 'down'
            else:
                performance_state = 'neutral'
                
            self.current_context['performance_state'] = performance_state
        
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Account balance updated to: ${account_balance:.2f}")
    
    def handle_signal_generated(self, event):
        """Handle signal generation events."""
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        symbol = event.data.get('symbol')
        
        if not all([signal_id, strategy_id, symbol]):
            logger.warning("Incomplete signal data received")
            return
            
        # Enrich the signal with current context
        enriched_signal = event.data.copy()
        enriched_signal.update({
            'market_regime': self.current_context.get('market_regime', 'unknown'),
            'volatility_state': self.current_context.get('volatility_state', 'medium'),
            'performance_state': self.current_context.get('performance_state', 'neutral'),
            'drawdown_percentage': self.current_context.get('drawdown_percentage', 0.0),
        })
        
        # Publish enriched signal
        self.event_bus.publish(Event(
            event_type=EventType.CONTEXT_UPDATED,
            data=enriched_signal
        ))
        
        logger.info(f"Signal {signal_id} enriched with contextual data")
    
    def handle_trade_closed(self, event):
        """Handle trade closed events."""
        trade_id = event.data.get('trade_id')
        pnl = event.data.get('pnl', 0.0)
        
        # For demonstration, we'll just log the outcome
        outcome = "win" if pnl > 0 else "loss"
        logger.info(f"Trade {trade_id} closed with outcome: {outcome}, PnL: ${pnl:.2f}")
        
        # In full implementation, this would update strategy performance metrics
        # and feed back into the decision scoring system
    
    def handle_pattern_detected(self, event):
        """Handle pattern detection events."""
        pattern_type = event.data.get('pattern_type')
        symbol = event.data.get('symbol')
        confidence = event.data.get('confidence', 0.0)
        
        logger.info(f"Pattern {pattern_type} detected for {symbol} with confidence {confidence:.2f}")
        
        # This would normally update pattern confidence in the context
    
    def _propagate_context_update(self):
        """Propagate context updates."""
        # Publish context update event
        self.event_bus.publish(Event(
            event_type=EventType.CONTEXT_UPDATED,
            data=self.current_context
        ))
    
    def calculate_position_size(self, symbol, account_balance, entry_price, stop_loss_pips=20):
        """Calculate position size based on current context and account balance."""
        # Demo implementation for calculating position size
        
        # Get risk percentage based on account balance
        risk_percentage = self._get_risk_percentage(account_balance)
        
        # Adjust risk based on volatility
        volatility = self.current_context.get('volatility_state', 'medium')
        if volatility == 'high':
            risk_percentage *= 0.7  # Reduce risk in high volatility
        elif volatility == 'low':
            risk_percentage *= 1.1  # Slightly increase risk in low volatility
            
        # Calculate risk amount
        risk_amount = account_balance * (risk_percentage / 100)
        
        # Calculate pip value (simplified for demo)
        pip_value = 0.0001  # For most forex pairs
        
        # Calculate position size based on stop loss
        position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
        
        # Generate explanation
        explanation = (
            f"Position size of {position_size:.4f} lots calculated using "
            f"{risk_percentage:.2f}% account risk (${risk_amount:.2f}) "
            f"with account balance ${account_balance:.2f}. "
            f"Adjusted for {volatility} volatility in "
            f"{self.current_context.get('market_regime', 'unknown')} market regime."
        )
        
        return {
            "position_size": position_size,
            "risk_amount": risk_amount,
            "risk_percentage": risk_percentage,
            "explanation": explanation
        }
    
    def _get_risk_percentage(self, account_balance):
        """
        Implements the progressive risk scaling strategy:
        - Up to $500: risk 95% of account
        - $1,000: 90%
        - $2,500: 85%
        - $5,000: 80%
        - $7,000: 75%
        - $10,000: 65%
        - $15,000: 55%
        - $20,000: 45%
        - $24,999: 35%
        - $25,000 (PDT threshold): drop to 15%
        - $35,000: 12%
        - $50,000: 10%
        - $100,000: 8%
        - $250,000: 6%
        - $500,000: 4%
        - $1M: 2%
        """
        # Dictionary of balance thresholds and corresponding risk percentages
        risk_schedule = {
            500: 95,
            1000: 90,
            2500: 85,
            5000: 80,
            7000: 75,
            10000: 65,
            15000: 55,
            20000: 45,
            24999: 35,
            25000: 15,  # Sharp drop at PDT threshold
            35000: 12,
            50000: 10,
            100000: 8,
            250000: 6,
            500000: 4,
            1000000: 2
        }
        
        # Find the appropriate risk percentage based on the account balance
        risk_percentage = 2  # Default to lowest risk
        
        for threshold in sorted(risk_schedule.keys()):
            if account_balance <= threshold:
                risk_percentage = risk_schedule[threshold]
                break
        
        return risk_percentage
    
    def select_optimal_strategy(self, symbol, market_data=None):
        """
        Select the optimal strategy for current conditions.
        Simplified implementation for demonstration.
        """
        # Current market regime
        regime = self.current_context.get('market_regime', 'unknown')
        
        # Find strategies that match the current regime
        matching_strategies = []
        for strategy in self.available_strategies:
            if strategy['preferred_regime'] == regime:
                matching_strategies.append(strategy)
        
        # If no exact matches, return a random strategy
        if not matching_strategies:
            selected = random.choice(self.available_strategies)
            selected = selected.copy()
            selected['score'] = 0.5
            selected['explanation'] = f"No strategy perfect for {regime} regime, selected as fallback."
            return selected
        
        # Select the best matching strategy
        selected = random.choice(matching_strategies)
        selected = selected.copy()
        selected['score'] = 0.8  # High confidence for matching regime
        selected['explanation'] = (
            f"Selected {selected['strategy_name']} because it is optimal for "
            f"current {regime} market regime with "
            f"{self.current_context.get('volatility_state', 'medium')} volatility."
        )
        
        return selected
    
    def get_current_context(self):
        """Get the current context state."""
        return self.current_context.copy()

#----------------
# Demo Simulation
#----------------

def simulate_market_changes(manager):
    """Simulate different market scenarios to demonstrate contextual awareness."""
    
    # Create event bus for simulation
    event_bus = manager.event_bus
    
    # Starting account balance
    account_balance = 5000.0
    
    # Sequence of market regimes to simulate
    regimes = [
        {"regime": "trending_up", "volatility": "low", "duration": 2},
        {"regime": "trending_up", "volatility": "medium", "duration": 2},
        {"regime": "ranging", "volatility": "low", "duration": 2},
        {"regime": "breakout", "volatility": "high", "duration": 2},
        {"regime": "trending_down", "volatility": "medium", "duration": 2},
    ]
    
    # Test symbols
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Simulation loop
    for step, regime_info in enumerate(regimes):
        logger.info(f"\n====== SIMULATION STEP {step+1}: {regime_info['regime']} regime ======\n")
        
        # Publish regime change event
        event_bus.publish(Event(
            event_type=EventType.MARKET_REGIME_CHANGE,
            data={
                "regime": regime_info["regime"],
                "previous_regime": manager.current_context.get('market_regime', 'unknown'),
                "confidence": 0.85
            }
        ))
        
        # Publish volatility update
        event_bus.publish(Event(
            event_type=EventType.VOLATILITY_UPDATE,
            data={
                "volatility_state": regime_info["volatility"],
                "previous_state": manager.current_context.get('volatility_state', 'medium')
            }
        ))
        
        time.sleep(0.5)  # Small delay to allow events to process
        
        # Make some trades with our different symbols
        for symbol in symbols:
            logger.info(f"\n--- Trading {symbol} in {regime_info['regime']} regime ---")
            
            # Select strategy based on current context
            strategy = manager.select_optimal_strategy(symbol)
            logger.info(f"Selected strategy: {strategy['strategy_name']} (score: {strategy['score']:.2f})")
            logger.info(f"Selection explanation: {strategy['explanation']}")
            
            # Calculate position size
            position_result = manager.calculate_position_size(
                symbol=symbol,
                account_balance=account_balance,
                entry_price=1.0,  # Mock price
                stop_loss_pips=20
            )
            
            position_size = position_result["position_size"]
            risk_amount = position_result["risk_amount"]
            risk_percentage = position_result["risk_percentage"]
            
            logger.info(f"Position size: {position_size:.4f} lots, risking ${risk_amount:.2f} ({risk_percentage:.2f}%)")
            logger.info(f"Position sizing explanation: {position_result['explanation']}")
            
            # Simulate trade outcome (random win/loss but biased by regime/strategy match)
            win_probability = 0.5  # Base win probability
            
            # Strategy matched to regime has better outcomes
            if strategy['preferred_regime'] == regime_info['regime']:
                win_probability = 0.7
            
            trade_outcome = random.random() < win_probability
            
            # Calculate PnL
            if trade_outcome:  # Win
                pnl = risk_amount * random.uniform(1.0, 2.0)
            else:  # Loss
                pnl = -risk_amount
            
            # Update account balance
            account_balance += pnl
            
            # Create trade ID
            trade_id = f"{symbol}_{step}_{strategy['strategy_id']}"
            
            # Publish trade closed event
            event_bus.publish(Event(
                event_type=EventType.TRADE_CLOSED,
                data={
                    "trade_id": trade_id,
                    "signal_id": f"signal_{trade_id}",
                    "strategy_id": strategy['strategy_id'],
                    "strategy_name": strategy['strategy_name'],
                    "symbol": symbol,
                    "position_size": position_size,
                    "pnl": pnl,
                    "outcome": "win" if trade_outcome else "loss"
                }
            ))
            
            logger.info(f"Trade {trade_id} result: {'WIN' if trade_outcome else 'LOSS'}, PnL: ${pnl:.2f}")
            
            # Update account balance
            event_bus.publish(Event(
                event_type=EventType.ACCOUNT_BALANCE_UPDATE,
                data={
                    "balance": account_balance,
                    "previous_balance": account_balance - pnl,
                    "initial_balance": 5000.0
                }
            ))
            
            logger.info(f"Account balance: ${account_balance:.2f}")
        
        # Sleep to simulate passage of time
        time.sleep(regime_info["duration"])
    
    # Show final account balance
    logger.info(f"\n====== SIMULATION COMPLETE ======")
    logger.info(f"Starting balance: $5,000.00")
    logger.info(f"Final balance: ${account_balance:.2f}")
    logger.info(f"Profit/Loss: ${account_balance - 5000:.2f} ({(account_balance/5000 - 1)*100:.2f}%)")
    
    # Show final context state
    logger.info("\nFinal Context State:")
    context = manager.get_current_context()
    for key, value in context.items():
        if key != 'last_updated' and not isinstance(value, dict):
            logger.info(f"  {key}: {value}")
    
    return account_balance

#----------------
# Main Demo
#----------------

def main():
    """Main demonstration function."""
    logger.info("Starting Contextual Integration Demonstration")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create contextual manager
    contextual_manager = ContextualIntegrationManager(event_bus)
    
    # Run simulation
    final_balance = simulate_market_changes(contextual_manager)
    
    logger.info("\nDemonstration complete!")
    logger.info("\nKey features demonstrated:")
    logger.info("1. Contextual awareness across market regimes")
    logger.info("2. Strategy selection adaptive to market conditions")
    logger.info("3. Progressive risk scaling based on account balance")
    logger.info("4. Position sizing adjusted for volatility")
    logger.info("5. Event-driven architecture for real-time adaptation")
    
    return 0 if final_balance > 5000 else 1

if __name__ == "__main__":
    main()
