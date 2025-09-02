#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Feedback Loop

This module connects the decision scoring system with the strategy intelligence recorder
to create a continuous learning feedback loop that improves strategy selection and
execution over time.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.core.decision_scoring import DecisionScorer
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder

logger = logging.getLogger(__name__)

class StrategyFeedbackLoop:
    """
    Creates a feedback loop between trade outcomes, decision scoring,
    and future strategy selection to enable continuous improvement.
    """
    
    def __init__(self, 
                event_bus: EventBus,
                decision_scorer: DecisionScorer,
                strategy_intelligence: StrategyIntelligenceRecorder,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy feedback loop.
        
        Args:
            event_bus: Central event bus for communication
            decision_scorer: Decision scoring system
            strategy_intelligence: Strategy intelligence recorder
            config: Configuration options
        """
        self.event_bus = event_bus
        self.decision_scorer = decision_scorer
        self.strategy_intelligence = strategy_intelligence
        self.config = config or {}
        
        # Configuration
        self.min_trades_for_pattern = self.config.get('min_trades_for_pattern', 5)
        self.min_success_rate = self.config.get('min_success_rate', 0.6)
        self.pattern_recognition_interval = self.config.get('pattern_recognition_interval', 3600)
        self.learning_rate = self.config.get('learning_rate', 0.2)
        
        # Tracking data
        self.strategy_performance = {}
        self.identified_patterns = []
        self.regime_specific_insights = {}
        
        # Performance by market condition
        self.regime_performance = {
            'trending_up': {},
            'trending_down': {},
            'ranging': {},
            'breakout': {},
            'reversal': {},
            'volatility_compression': {},
            'volatility_expansion': {},
            'unknown': {}
        }
        
        # Last update timestamps
        self._last_pattern_update = datetime.min
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info("Strategy Feedback Loop initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        # Trade events
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        self.event_bus.subscribe(EventType.DECISION_SCORED, self.handle_decision_scored)
        
        # Market regime events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self.handle_regime_change)
        
        # Strategy events
        self.event_bus.subscribe(EventType.STRATEGY_SELECTED, self.handle_strategy_selected)
        
        logger.info("Subscribed to trade and strategy events")
    
    def handle_trade_closed(self, event: Event):
        """
        Handle trade closed events to update the feedback loop.
        
        Args:
            event: Trade closed event
        """
        trade_id = event.data.get('trade_id')
        strategy_id = event.data.get('strategy_id')
        symbol = event.data.get('symbol')
        pnl = event.data.get('pnl', 0.0)
        outcome = event.data.get('outcome', 'unknown')
        signal_id = event.data.get('signal_id')
        
        # Check for required data
        if not all([trade_id, strategy_id, symbol, signal_id]):
            logger.warning(f"Incomplete trade data for feedback: {trade_id}")
            return
        
        # Calculate win/loss status
        is_win = pnl > 0
        
        # Get additional context if available
        market_regime = event.data.get('market_regime', 'unknown')
        
        # Update strategy performance tracking
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'regimes': {r: {'trades': 0, 'wins': 0, 'pnl': 0.0} for r in self.regime_performance.keys()}
            }
        
        # Update trade count and PnL
        self.strategy_performance[strategy_id]['trades'] += 1
        self.strategy_performance[strategy_id]['pnl'] += pnl
        
        # Update win/loss statistics
        if is_win:
            self.strategy_performance[strategy_id]['wins'] += 1
            
            # Update average win size
            prev_wins = self.strategy_performance[strategy_id]['wins'] - 1
            prev_avg_win = self.strategy_performance[strategy_id]['avg_win']
            self.strategy_performance[strategy_id]['avg_win'] = (prev_avg_win * prev_wins + pnl) / self.strategy_performance[strategy_id]['wins']
        else:
            self.strategy_performance[strategy_id]['losses'] += 1
            
            # Update average loss size
            prev_losses = self.strategy_performance[strategy_id]['losses'] - 1
            prev_avg_loss = self.strategy_performance[strategy_id]['avg_loss'] if prev_losses > 0 else 0
            self.strategy_performance[strategy_id]['avg_loss'] = (prev_avg_loss * prev_losses + abs(pnl)) / self.strategy_performance[strategy_id]['losses']
        
        # Update regime-specific performance
        if market_regime in self.strategy_performance[strategy_id]['regimes']:
            self.strategy_performance[strategy_id]['regimes'][market_regime]['trades'] += 1
            self.strategy_performance[strategy_id]['regimes'][market_regime]['pnl'] += pnl
            if is_win:
                self.strategy_performance[strategy_id]['regimes'][market_regime]['wins'] += 1
        
        # Use the decision scorer to score this outcome
        outcome_data = event.data.copy()
        outcome_data.update({
            'market_regime': market_regime,
            'is_win': is_win,
            'pnl': pnl
        })
        
        # Close and score the decision
        score, explanation = self.decision_scorer.close_and_score_decision(
            signal_id=signal_id,
            outcome_data=outcome_data
        )
        
        logger.info(f"Trade {trade_id} scored: {score:.2f} - {explanation}")
        
        # Update pattern recognition if enough time has passed
        current_time = datetime.now()
        if (current_time - self._last_pattern_update).total_seconds() > self.pattern_recognition_interval:
            self._update_pattern_recognition()
            self._last_pattern_update = current_time
    
    def handle_decision_scored(self, event: Event):
        """
        Handle decision scored events to update the feedback loop.
        
        Args:
            event: Decision scored event
        """
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        score = event.data.get('score', 0.0)
        explanation = event.data.get('explanation', '')
        market_regime = event.data.get('market_regime', 'unknown')
        
        # Add to regime-specific insights
        if market_regime not in self.regime_specific_insights:
            self.regime_specific_insights[market_regime] = []
        
        # Add the insight
        self.regime_specific_insights[market_regime].append({
            'signal_id': signal_id,
            'strategy_id': strategy_id,
            'score': score,
            'explanation': explanation,
            'timestamp': datetime.now()
        })
        
        # Keep insights at a reasonable size
        max_insights = 100
        if len(self.regime_specific_insights[market_regime]) > max_insights:
            self.regime_specific_insights[market_regime] = self.regime_specific_insights[market_regime][-max_insights:]
    
    def handle_regime_change(self, event: Event):
        """
        Handle market regime change events to update strategy preferences.
        
        Args:
            event: Market regime change event
        """
        regime = event.data.get('regime', 'unknown')
        symbol = event.data.get('symbol', 'UNKNOWN')
        
        # Log the regime change
        logger.info(f"Market regime change for {symbol}: {regime}")
        
        # Identify top strategies for this regime
        top_strategies = self._get_top_strategies_for_regime(regime, 3)
        
        if top_strategies:
            # Publish best strategies for regime event
            self.event_bus.publish(Event(
                event_type=EventType.BEST_STRATEGIES_FOR_REGIME,
                data={
                    'regime': regime,
                    'symbol': symbol,
                    'strategies': top_strategies,
                    'explanation': f"These strategies have performed best in {regime} conditions based on historical analysis."
                }
            ))
            
            logger.info(f"Published top {len(top_strategies)} strategies for {regime} regime")
    
    def handle_strategy_selected(self, event: Event):
        """
        Handle strategy selected events to track selection accuracy.
        
        Args:
            event: Strategy selected event
        """
        strategy_id = event.data.get('strategy_id')
        symbol = event.data.get('symbol')
        market_regime = event.data.get('market_regime', 'unknown')
        
        # This will be used to later evaluate if the selection was good
        # by comparing to the trade outcome
        selection_data = {
            'strategy_id': strategy_id,
            'symbol': symbol,
            'market_regime': market_regime,
            'timestamp': datetime.now()
        }
        
        # We could store this for later analysis if needed
        pass
    
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Performance metrics
        """
        if strategy_id not in self.strategy_performance:
            return {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'regimes': {}
            }
        
        # Get performance data
        perf = self.strategy_performance[strategy_id]
        
        # Calculate derived metrics
        trades = perf['trades']
        wins = perf['wins']
        losses = perf['losses']
        win_rate = wins / trades if trades > 0 else 0.0
        profit_factor = (perf['avg_win'] * wins) / (perf['avg_loss'] * losses) if losses > 0 and perf['avg_loss'] > 0 else 0.0
        
        # Calculate regime-specific metrics
        regime_metrics = {}
        for regime, data in perf['regimes'].items():
            if data['trades'] > 0:
                regime_metrics[regime] = {
                    'trades': data['trades'],
                    'wins': data['wins'],
                    'win_rate': data['wins'] / data['trades'] if data['trades'] > 0 else 0.0,
                    'pnl': data['pnl'],
                    'avg_trade': data['pnl'] / data['trades']
                }
        
        # Return comprehensive metrics
        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'pnl': perf['pnl'],
            'avg_win': perf['avg_win'],
            'avg_loss': perf['avg_loss'],
            'profit_factor': profit_factor,
            'regimes': regime_metrics
        }
    
    def get_patterns_for_regime(self, regime: str) -> List[Dict[str, Any]]:
        """
        Get identified patterns for a specific market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            List of patterns
        """
        # Filter patterns by regime
        return [p for p in self.identified_patterns if p.get('regime') == regime]
    
    def get_top_insights_for_regime(self, regime: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top insights for a specific market regime.
        
        Args:
            regime: Market regime
            limit: Maximum number of insights to return
            
        Returns:
            List of insights
        """
        if regime not in self.regime_specific_insights:
            return []
        
        # Sort insights by score
        sorted_insights = sorted(
            self.regime_specific_insights[regime],
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        return sorted_insights[:limit]
    
    def _update_pattern_recognition(self):
        """Update pattern recognition based on decision scores."""
        # Use the decision scorer to find successful patterns
        for regime in self.regime_performance.keys():
            patterns = self.decision_scorer.find_successful_patterns(
                regime=regime,
                min_occurrences=self.min_trades_for_pattern,
                min_success_rate=self.min_success_rate
            )
            
            if patterns:
                # Add to identified patterns
                for pattern in patterns:
                    # Check if this pattern is already identified
                    pattern_key = f"{pattern.get('pattern_type')}_{regime}"
                    existing = [p for p in self.identified_patterns if
                               p.get('pattern_type') == pattern.get('pattern_type') and
                               p.get('regime') == regime]
                    
                    if existing:
                        # Update existing pattern
                        existing[0].update({
                            'success_rate': pattern.get('success_rate', 0),
                            'occurrences': pattern.get('occurrences', 0),
                            'avg_pnl': pattern.get('avg_pnl', 0),
                            'last_updated': datetime.now()
                        })
                    else:
                        # Add new pattern
                        pattern_data = pattern.copy()
                        pattern_data.update({
                            'regime': regime,
                            'first_identified': datetime.now(),
                            'last_updated': datetime.now()
                        })
                        self.identified_patterns.append(pattern_data)
                
                # Publish pattern insights event
                self.event_bus.publish(Event(
                    event_type=EventType.PATTERN_INSIGHTS,
                    data={
                        'regime': regime,
                        'patterns': patterns,
                        'count': len(patterns)
                    }
                ))
                
                logger.info(f"Identified {len(patterns)} patterns for {regime} regime")
    
    def _get_top_strategies_for_regime(self, regime: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get top performing strategies for a specific regime.
        
        Args:
            regime: Market regime
            limit: Maximum number of strategies to return
            
        Returns:
            List of top strategies
        """
        strategies = []
        
        # Find strategies with data for this regime
        for strategy_id, perf in self.strategy_performance.items():
            if regime in perf['regimes'] and perf['regimes'][regime]['trades'] >= 5:
                regime_data = perf['regimes'][regime]
                
                # Calculate metrics
                win_rate = regime_data['wins'] / regime_data['trades'] if regime_data['trades'] > 0 else 0
                avg_trade = regime_data['pnl'] / regime_data['trades'] if regime_data['trades'] > 0 else 0
                
                # Add strategy with key metrics
                strategies.append({
                    'strategy_id': strategy_id,
                    'win_rate': win_rate,
                    'trades': regime_data['trades'],
                    'pnl': regime_data['pnl'],
                    'avg_trade': avg_trade,
                    'score': (win_rate * 0.5) + (avg_trade / 100 * 0.5)  # Simple scoring formula
                })
        
        # Sort by score
        sorted_strategies = sorted(strategies, key=lambda x: x['score'], reverse=True)
        
        return sorted_strategies[:limit]
