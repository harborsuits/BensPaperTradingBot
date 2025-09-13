#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contextual Integration Manager

This module connects all the core components of the trading system to ensure
maximum contextual awareness and adaptive behavior across all decisions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, MarketRegime
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
from trading_bot.core.decision_scoring import DecisionScorer
from trading_bot.core.smart_strategy_selector import SmartStrategySelector
from trading_bot.core.adaptive_position_manager import AdaptivePositionManager
from trading_bot.strategies.forex.base.pip_based_position_sizing import PipBasedPositionSizing, RiskProfile
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger(__name__)

class ContextualIntegrationManager:
    """
    Connects all adaptive components to maximize contextual awareness.
    
    This manager ensures that:
    1. Market regime awareness flows to all components
    2. Decision scoring feeds back to strategy selection
    3. Pattern recognition impacts confidence scores
    4. Position sizing reflects both account status and market conditions
    5. All components communicate via standardized events
    """
    
    def __init__(self, 
                event_bus: EventBus,
                persistence: PersistenceManager,
                strategy_intelligence: StrategyIntelligenceRecorder,
                decision_scorer: DecisionScorer,
                strategy_selector: SmartStrategySelector,
                position_manager: AdaptivePositionManager,
                position_sizer: PipBasedPositionSizing):
        """
        Initialize the integration manager.
        
        Args:
            event_bus: Central event bus for communication
            persistence: Persistence manager for state
            strategy_intelligence: Strategy intelligence recorder
            decision_scorer: Decision scoring system
            strategy_selector: Smart strategy selector
            position_manager: Adaptive position manager
            position_sizer: Position sizing system
        """
        self.event_bus = event_bus
        self.persistence = persistence
        self.strategy_intelligence = strategy_intelligence
        self.decision_scorer = decision_scorer
        self.strategy_selector = strategy_selector
        self.position_manager = position_manager
        self.position_sizer = position_sizer
        
        # Current context state
        self.current_context = {
            'market_regime': 'unknown',
            'volatility_state': 'medium',
            'correlation_state': 'low',
            'performance_state': 'neutral',
            'drawdown_percentage': 0.0,
            'account_multiplier': 1.0,
            'last_updated': datetime.now()
        }
        
        # Register for events
        self._subscribe_to_events()
        
        # Set priority levels for critical events
        self._configure_event_priorities()
        
        logger.info("Contextual Integration Manager initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events from the event bus."""
        # Market state events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self.handle_market_regime_change)
        self.event_bus.subscribe(EventType.VOLATILITY_UPDATE, self.handle_volatility_update)
        self.event_bus.subscribe(EventType.CORRELATION_UPDATE, self.handle_correlation_update)
        
        # Performance events
        self.event_bus.subscribe(EventType.PORTFOLIO_UPDATE, self.handle_portfolio_update)
        self.event_bus.subscribe(EventType.ACCOUNT_BALANCE_UPDATE, self.handle_account_update)
        
        # Trading events
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self.handle_signal_generated)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.handle_trade_executed)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        
        # Pattern events
        self.event_bus.subscribe(EventType.PATTERN_DETECTED, self.handle_pattern_detected)
        
        logger.info("Subscribed to relevant events")
    
    def _configure_event_priorities(self):
        """Configure priority levels for different event types."""
        # Critical events - processed first
        self.event_bus.set_event_priority(EventType.RISK_ALERT, 0)
        self.event_bus.set_event_priority(EventType.DRAWDOWN_THRESHOLD, 0)
        self.event_bus.set_event_priority(EventType.ERROR, 0)
        
        # High priority events
        self.event_bus.set_event_priority(EventType.MARKET_REGIME_CHANGE, 1)
        self.event_bus.set_event_priority(EventType.VOLATILITY_UPDATE, 1)
        self.event_bus.set_event_priority(EventType.SIGNAL_GENERATED, 1)
        
        # Medium priority events
        self.event_bus.set_event_priority(EventType.CORRELATION_UPDATE, 2)
        self.event_bus.set_event_priority(EventType.TRADE_EXECUTED, 2)
        self.event_bus.set_event_priority(EventType.TRADE_CLOSED, 2)
        
        # Lower priority events
        self.event_bus.set_event_priority(EventType.PATTERN_DETECTED, 3)
        self.event_bus.set_event_priority(EventType.ACCOUNT_BALANCE_UPDATE, 3)
        
        logger.info("Configured event priorities")
    
    def handle_market_regime_change(self, event: Event):
        """Handle market regime change events."""
        regime = event.data.get('regime', 'unknown')
        self.current_context['market_regime'] = regime
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Market regime updated to: {regime}")
    
    def handle_volatility_update(self, event: Event):
        """Handle volatility update events."""
        volatility = event.data.get('volatility_state', 'medium')
        self.current_context['volatility_state'] = volatility
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Volatility state updated to: {volatility}")
    
    def handle_correlation_update(self, event: Event):
        """Handle correlation update events."""
        correlation = event.data.get('correlation_state', 'low')
        self.current_context['correlation_state'] = correlation
        
        # Get correlation matrix if available
        correlation_matrix = event.data.get('correlation_matrix')
        if correlation_matrix:
            self.current_context['correlation_matrix'] = correlation_matrix
            
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Correlation state updated to: {correlation}")
    
    def handle_portfolio_update(self, event: Event):
        """Handle portfolio update events."""
        # Extract drawdown information
        drawdown = event.data.get('drawdown_percentage', 0.0)
        self.current_context['drawdown_percentage'] = drawdown
        
        # Extract current positions
        current_positions = event.data.get('positions', [])
        if current_positions:
            self.current_context['current_positions'] = current_positions
            
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Portfolio updated, drawdown: {drawdown:.2f}%")
    
    def handle_account_update(self, event: Event):
        """Handle account balance update events."""
        account_balance = event.data.get('balance', 0.0)
        previous_balance = event.data.get('previous_balance', 0.0)
        initial_balance = event.data.get('initial_balance', 500.0)
        
        # Calculate account multiplier (for progressive risk scaling)
        if initial_balance > 0:
            account_multiplier = account_balance / initial_balance
            self.current_context['account_multiplier'] = account_multiplier
            
            # Update position sizing initial balance reference
            if hasattr(self.position_sizer, 'initial_capital'):
                self.position_sizer.initial_capital = initial_balance
        
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
            
        self.current_context['account_balance'] = account_balance
        self.current_context['last_updated'] = datetime.now()
        
        # Update the context for other components
        self._propagate_context_update()
        
        logger.info(f"Account balance updated to: ${account_balance:.2f}")
    
    def handle_signal_generated(self, event: Event):
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
            'correlation_state': self.current_context.get('correlation_state', 'low'),
            'performance_state': self.current_context.get('performance_state', 'neutral'),
            'drawdown_percentage': self.current_context.get('drawdown_percentage', 0.0),
        })
        
        # Add confidence boost from pattern recognition if applicable
        pattern_boost = self.current_context.get('pattern_confidence', {}).get(symbol, 0.0)
        if pattern_boost > 0:
            base_confidence = enriched_signal.get('confidence', 0.5)
            enriched_signal['confidence'] = min(0.95, base_confidence + (pattern_boost * 0.2))
            enriched_signal['pattern_boost'] = pattern_boost
            
        # Publish enriched signal
        self.event_bus.publish(Event(
            event_type=EventType.SIGNAL_ENRICHED,
            data=enriched_signal
        ))
        
        logger.info(f"Signal {signal_id} enriched with contextual data")
    
    def handle_trade_executed(self, event: Event):
        """Handle trade execution events."""
        trade_id = event.data.get('trade_id')
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        
        if not all([trade_id, signal_id, strategy_id]):
            logger.warning("Incomplete trade execution data received")
            return
            
        # Record the decision execution in the scoring system
        self.decision_scorer.update_decision_execution(
            signal_id=signal_id,
            execution_data=event.data
        )
        
        logger.info(f"Trade execution {trade_id} recorded for signal {signal_id}")
    
    def handle_trade_closed(self, event: Event):
        """Handle trade closed events."""
        trade_id = event.data.get('trade_id')
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        pnl = event.data.get('pnl', 0.0)
        
        if not all([trade_id, signal_id, strategy_id]):
            logger.warning("Incomplete trade closed data received")
            return
            
        # Enrich the trade outcome with context
        enriched_outcome = event.data.copy()
        enriched_outcome.update({
            'market_regime': self.current_context.get('market_regime', 'unknown'),
            'volatility_state': self.current_context.get('volatility_state', 'medium'),
        })
        
        # Close and score the decision
        score, explanation = self.decision_scorer.close_and_score_decision(
            signal_id=signal_id,
            outcome_data=enriched_outcome
        )
        
        # Update strategy performance in the selector
        self.strategy_selector.update_strategy_performance(enriched_outcome)
        
        # Publish decision score event
        self.event_bus.publish(Event(
            event_type=EventType.DECISION_SCORED,
            data={
                'signal_id': signal_id,
                'strategy_id': strategy_id,
                'trade_id': trade_id,
                'score': score,
                'explanation': explanation,
                'pnl': pnl,
                'market_regime': self.current_context.get('market_regime', 'unknown'),
                'volatility_state': self.current_context.get('volatility_state', 'medium'),
            }
        ))
        
        logger.info(f"Trade {trade_id} closed and scored: {score:.2f}")
    
    def handle_pattern_detected(self, event: Event):
        """Handle pattern detection events."""
        pattern_type = event.data.get('pattern_type')
        symbol = event.data.get('symbol')
        confidence = event.data.get('confidence', 0.0)
        
        if not all([pattern_type, symbol]) or confidence <= 0:
            return
            
        # Store pattern confidence for later use
        if 'pattern_confidence' not in self.current_context:
            self.current_context['pattern_confidence'] = {}
            
        self.current_context['pattern_confidence'][symbol] = confidence
        
        # If we have pattern confidence threshold crossed, publish enhanced event
        if confidence > 0.7:  # High confidence threshold
            # Find matching successful patterns from history
            successful_patterns = self.decision_scorer.find_successful_patterns(
                regime=self.current_context.get('market_regime', 'unknown'),
                min_occurrences=3,
                min_success_rate=0.6
            )
            
            matching_patterns = [p for p in successful_patterns 
                               if p.get('pattern_type') == pattern_type]
            
            if matching_patterns:
                best_match = max(matching_patterns, key=lambda p: p.get('success_rate', 0))
                
                # Publish pattern confidence event
                self.event_bus.publish(Event(
                    event_type=EventType.PATTERN_CONFIDENCE,
                    data={
                        'pattern_type': pattern_type,
                        'symbol': symbol,
                        'confidence': confidence,
                        'historical_success_rate': best_match.get('success_rate', 0),
                        'historical_occurrences': best_match.get('occurrences', 0),
                        'market_regime': self.current_context.get('market_regime', 'unknown'),
                        'volatility_state': self.current_context.get('volatility_state', 'medium')
                    }
                ))
                
                logger.info(f"High confidence pattern {pattern_type} detected for {symbol}: {confidence:.2f}")
    
    def _propagate_context_update(self):
        """Propagate context updates to all components that need them."""
        # Update the strategy selector context
        self.strategy_selector.update_context(self.current_context)
        
        # Update the position manager context
        self.position_manager.update_context(self.current_context)
        
        # Publish context update event
        self.event_bus.publish(Event(
            event_type=EventType.CONTEXT_UPDATED,
            data=self.current_context
        ))
    
    def calculate_position_size(self, **kwargs) -> Dict[str, Any]:
        """
        Calculate optimal position size with all available context.
        
        This method combines all contextual information to calculate the ideal
        position size based on the current market regime, account status, and
        risk profile.
        
        Args:
            **kwargs: Position sizing parameters
            
        Returns:
            Position sizing result with explanations
        """
        # Ensure we have all required context
        symbol = kwargs.get('symbol')
        account_balance = kwargs.get('account_balance')
        
        if not all([symbol, account_balance]):
            logger.error("Missing required parameters for position sizing")
            return {}
        
        # Add current context to parameters
        params = kwargs.copy()
        
        # Set appropriate risk profile based on context
        risk_profile = self._determine_risk_profile()
        params['risk_profile'] = risk_profile
        
        # Include market regime and volatility state
        params['market_regime'] = self.current_context.get('market_regime', 'unknown')
        params['volatility_state'] = self.current_context.get('volatility_state', 'medium')
        
        # Include correlation data if available
        if 'correlation_matrix' in self.current_context:
            params['correlation_matrix'] = self.current_context['correlation_matrix']
            
        if 'current_positions' in self.current_context:
            params['current_positions'] = self.current_context['current_positions']
            
        # Include drawdown percentage
        params['current_drawdown'] = self.current_context.get('drawdown_percentage', 0.0)
        
        # Calculate position size using the adaptive position sizer
        result = self.position_sizer.calculate_adaptive_position_size(**params)
        
        # Track the decision for context
        self.current_context['last_position_size'] = {
            'symbol': symbol,
            'position_size': result.get('position_size', 0),
            'lot_size': result.get('lot_size', 0),
            'risk_amount': result.get('risk_amount', 0),
            'timestamp': datetime.now()
        }
        
        return result
    
    def _determine_risk_profile(self) -> RiskProfile:
        """
        Determine the appropriate risk profile based on current context.
        
        Returns:
            RiskProfile enum representing the optimal risk approach
        """
        # Default to security-focused (our custom approach)
        risk_profile = RiskProfile.SECURITY_FOCUSED
        
        # Market regime specific overrides
        regime = self.current_context.get('market_regime', 'unknown')
        volatility = self.current_context.get('volatility_state', 'medium')
        
        # In very high volatility, be more conservative regardless of other factors
        if volatility == 'high':
            return RiskProfile.CONSERVATIVE
            
        # In strong trends, consider trending profile
        if regime in ['trending_up', 'trending_down']:
            risk_profile = RiskProfile.TREND_FOLLOWING
            
        # In ranging markets, consider mean reversion profile
        elif regime == 'ranging':
            risk_profile = RiskProfile.MEAN_REVERSION
            
        # In breakout conditions, more aggressive
        elif regime == 'breakout':
            risk_profile = RiskProfile.AGGRESSIVE
            
        # Default to security focused
        return risk_profile
    
    def select_optimal_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the optimal strategy for current conditions with full context awareness.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            Selected strategy with explanations
        """
        # Ensure we have market regime in the context
        context = self.current_context.copy()
        
        # Add the specific symbol
        context['symbol'] = symbol
        
        # Merge in any market data
        if market_data:
            context.update(market_data)
            
        # Use the smart strategy selector with bandit algorithm
        selected = self.strategy_selector.select_strategy(
            symbol=symbol,
            market_data=market_data,
            additional_context=context,
            use_bandit_selection=True
        )
        
        if selected:
            # Track the selection for context
            self.current_context['last_strategy_selection'] = {
                'symbol': symbol,
                'strategy_id': selected.get('strategy_id'),
                'strategy_name': selected.get('strategy_name'),
                'score': selected.get('score'),
                'timestamp': datetime.now()
            }
            
        return selected
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current context state."""
        return self.current_context.copy()
    
    def load_state(self) -> bool:
        """Load state from persistence."""
        try:
            if self.persistence:
                state = self.persistence.load_system_state('contextual_integration')
                if state:
                    self.current_context.update(state)
                    logger.info("Loaded contextual integration state from persistence")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def save_state(self) -> bool:
        """Save state to persistence."""
        try:
            if self.persistence:
                self.persistence.save_system_state('contextual_integration', self.current_context)
                logger.info("Saved contextual integration state to persistence")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return False
