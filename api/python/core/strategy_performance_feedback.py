#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Performance Feedback Loop

This module records strategy performance in different market conditions
and automatically adjusts strategy weights based on historical results.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os

from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class StrategyPerformanceFeedback:
    """
    Strategy Performance Feedback Loop
    
    This class tracks strategy performance and adapts strategy selection:
    - Records which strategies perform best in specific market conditions
    - Builds a performance history database for strategies by regime
    - Automatically adjusts strategy weights based on historical performance
    - Provides performance analytics and insights
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Performance tracking
        'min_trades_for_significance': 10,  # Minimum trades to consider performance significant
        'performance_lookback_days': 90,    # Days to look back for performance
        'regime_history_size': 50,          # Number of regime periods to store
        
        # Performance metrics weights
        'win_rate_weight': 0.3,             # Importance of win rate
        'profit_factor_weight': 0.3,        # Importance of profit factor
        'avg_return_weight': 0.2,           # Importance of average return
        'max_drawdown_weight': 0.2,         # Importance of maximum drawdown
        
        # Adaptation parameters
        'adaptation_speed': 0.2,            # How quickly to adapt weights (0-1)
        'min_strategy_weight': 0.1,         # Minimum weight for any strategy
        'max_strategy_weight': 0.5,         # Maximum weight for any strategy
        
        # Performance thresholds
        'excellent_score_threshold': 0.8,   # Threshold for excellent performance score
        'good_score_threshold': 0.6,        # Threshold for good performance score
        'poor_score_threshold': 0.4,        # Threshold for poor performance score
        
        # Data storage
        'performance_data_path': 'data/strategy_performance',  # Path to save performance data
        'save_interval_minutes': 60          # How often to save performance data
    }
    
    def __init__(self, event_bus: EventBus, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Strategy Performance Feedback Loop.
        
        Args:
            event_bus: System event bus
            parameters: Custom parameters to override defaults
        """
        self.event_bus = event_bus
        
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
            
        # Initialize performance tracking
        self.performance_history = {
            'strategies': {},      # Strategy performance by name
            'asset_classes': {},   # Asset class performance
            'regimes': {},         # Market regime performance history
            'current_regime': 'neutral'  # Default regime
        }
        
        # Strategy weights by regime
        self.strategy_weights = {
            'bullish': {},
            'bearish': {},
            'volatile': {},
            'neutral': {}
        }
        
        # Current active trades
        self.active_trades = {}
        
        # Last save time
        self.last_save_time = datetime.now()
        
        # Ensure data directory exists
        os.makedirs(self.parameters['performance_data_path'], exist_ok=True)
        
        # Load existing performance data if available
        self._load_performance_data()
        
        # Register for events
        self._register_events()
        
        logger.info("Strategy Performance Feedback Loop initialized")
        
    def _register_events(self):
        """Register for events of interest."""
        self.event_bus.subscribe(EventType.TRADE_OPENED, self._on_trade_opened)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        self.event_bus.subscribe(EventType.STRATEGY_SELECTION_REQUESTED, self._on_strategy_selection_requested)
        self.event_bus.subscribe(EventType.CONTEXT_ANALYSIS_COMPLETED, self._on_context_analysis)
    
    def _on_trade_opened(self, event: Event):
        """
        Handle trade opened events.
        
        Args:
            event: Trade opened event
        """
        # Extract trade data
        if 'trade' not in event.data:
            return
            
        trade = event.data['trade']
        
        # Ensure required fields
        required_fields = ['id', 'symbol', 'strategy', 'asset_class', 'entry_time', 'entry_price']
        if not all(field in trade for field in required_fields):
            logger.warning(f"Trade missing required fields: {trade}")
            return
            
        # Add to active trades
        self.active_trades[trade['id']] = {
            'trade': trade,
            'market_regime': self.performance_history['current_regime'],
            'entry_time': datetime.now() if 'entry_time' not in trade else trade['entry_time']
        }
    
    def _on_trade_closed(self, event: Event):
        """
        Handle trade closed events.
        
        Args:
            event: Trade closed event
        """
        # Extract trade data
        if 'trade' not in event.data:
            return
            
        trade = event.data['trade']
        
        # Ensure required fields
        required_fields = ['id', 'symbol', 'strategy', 'asset_class', 'exit_time', 'exit_price', 'profit_loss']
        if not all(field in trade for field in required_fields):
            logger.warning(f"Trade missing required fields: {trade}")
            return
            
        # Check if we have this trade in active trades
        if trade['id'] not in self.active_trades:
            # Try to reconstruct missing data
            self.active_trades[trade['id']] = {
                'trade': trade,
                'market_regime': self.performance_history['current_regime'],
                'entry_time': datetime.now() - timedelta(hours=1) if 'entry_time' not in trade else trade['entry_time']
            }
            
        # Get active trade data
        active_trade = self.active_trades[trade['id']]
        
        # Record trade performance
        self._record_trade_performance(trade, active_trade['market_regime'])
        
        # Remove from active trades
        del self.active_trades[trade['id']]
        
        # Save performance data periodically
        if (datetime.now() - self.last_save_time).total_seconds() / 60 >= self.parameters['save_interval_minutes']:
            self._save_performance_data()
            self.last_save_time = datetime.now()
    
    def _on_strategy_selection_requested(self, event: Event):
        """
        Handle strategy selection requested events.
        
        Args:
            event: Strategy selection requested event
        """
        # Current market regime
        current_regime = self.performance_history['current_regime']
        
        # Update strategy weights based on performance
        self._update_strategy_weights()
        
        # Get weights for current regime
        regime_weights = self.strategy_weights.get(current_regime, {})
        
        # Publish strategy weights event
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_PERFORMANCE_WEIGHTS,
            data={
                'strategy_weights': regime_weights,
                'market_regime': current_regime,
                'performance_data': self._get_regime_performance_summary(current_regime),
                'timestamp': datetime.now()
            }
        ))
    
    def _on_context_analysis(self, event: Event):
        """
        Handle context analysis events.
        
        Args:
            event: Context analysis event
        """
        # Update market context
        if 'context' in event.data:
            context = event.data['context']
            
            # Update market regime
            if 'market_regime' in context:
                new_regime = context['market_regime']
                old_regime = self.performance_history['current_regime']
                
                # Only update if regime changed
                if new_regime != old_regime:
                    self.performance_history['current_regime'] = new_regime
                    
                    # Add regime change to history
                    if 'regime_changes' not in self.performance_history:
                        self.performance_history['regime_changes'] = []
                        
                    self.performance_history['regime_changes'].append({
                        'from': old_regime,
                        'to': new_regime,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Limit history size
                    if len(self.performance_history['regime_changes']) > self.parameters['regime_history_size']:
                        self.performance_history['regime_changes'] = \
                            self.performance_history['regime_changes'][-self.parameters['regime_history_size']:]
                    
                    # Log regime change
                    logger.info(f"Market regime changed from {old_regime} to {new_regime}")
    
    def _record_trade_performance(self, trade: Dict[str, Any], market_regime: str):
        """
        Record performance of a completed trade.
        
        Args:
            trade: Completed trade data
            market_regime: Market regime during trade
        """
        # Extract key data
        strategy_name = trade['strategy']
        asset_class = trade['asset_class']
        profit_loss = trade['profit_loss']
        
        # Initialize strategy if not exists
        if strategy_name not in self.performance_history['strategies']:
            self.performance_history['strategies'][strategy_name] = {
                'overall': self._create_empty_performance_record(),
                'by_regime': {regime: self._create_empty_performance_record() 
                             for regime in self.strategy_weights.keys()},
                'by_asset_class': {}
            }
        
        # Initialize asset class for strategy if not exists
        strategy_data = self.performance_history['strategies'][strategy_name]
        if asset_class not in strategy_data['by_asset_class']:
            strategy_data['by_asset_class'][asset_class] = {
                'overall': self._create_empty_performance_record(),
                'by_regime': {regime: self._create_empty_performance_record() 
                             for regime in self.strategy_weights.keys()}
            }
        
        # Initialize asset class if not exists
        if asset_class not in self.performance_history['asset_classes']:
            self.performance_history['asset_classes'][asset_class] = {
                'overall': self._create_empty_performance_record(),
                'by_regime': {regime: self._create_empty_performance_record() 
                             for regime in self.strategy_weights.keys()}
            }
        
        # Initialize regime if not exists
        if market_regime not in self.performance_history['regimes']:
            self.performance_history['regimes'][market_regime] = {
                'overall': self._create_empty_performance_record(),
                'by_asset_class': {}
            }
        
        # Initialize asset class for regime if not exists
        regime_data = self.performance_history['regimes'][market_regime]
        if asset_class not in regime_data['by_asset_class']:
            regime_data['by_asset_class'][asset_class] = self._create_empty_performance_record()
        
        # Update all relevant performance records
        
        # 1. Update strategy overall performance
        self._update_performance_record(strategy_data['overall'], profit_loss)
        
        # 2. Update strategy performance for this regime
        self._update_performance_record(strategy_data['by_regime'][market_regime], profit_loss)
        
        # 3. Update strategy performance for this asset class
        self._update_performance_record(strategy_data['by_asset_class'][asset_class]['overall'], profit_loss)
        
        # 4. Update strategy performance for this asset class and regime
        self._update_performance_record(
            strategy_data['by_asset_class'][asset_class]['by_regime'][market_regime], 
            profit_loss
        )
        
        # 5. Update asset class overall performance
        self._update_performance_record(self.performance_history['asset_classes'][asset_class]['overall'], profit_loss)
        
        # 6. Update asset class performance for this regime
        self._update_performance_record(
            self.performance_history['asset_classes'][asset_class]['by_regime'][market_regime], 
            profit_loss
        )
        
        # 7. Update regime overall performance
        self._update_performance_record(self.performance_history['regimes'][market_regime]['overall'], profit_loss)
        
        # 8. Update regime performance for this asset class
        self._update_performance_record(
            self.performance_history['regimes'][market_regime]['by_asset_class'][asset_class], 
            profit_loss
        )
        
        # Log performance update
        logger.debug(f"Recorded {profit_loss:.2f} P/L for {strategy_name} ({asset_class}) in {market_regime} regime")
    
    def _create_empty_performance_record(self) -> Dict[str, Any]:
        """
        Create an empty performance record structure.
        
        Returns:
            Empty performance record
        """
        return {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'trade_history': [],  # Recent trade P/L for equity curve
            'last_updated': datetime.now().isoformat(),
            'performance_score': 0.0  # Normalized score (0-1)
        }
    
    def _update_performance_record(self, record: Dict[str, Any], profit_loss: float):
        """
        Update a performance record with a new trade result.
        
        Args:
            record: Performance record to update
            profit_loss: Profit/loss from the trade
        """
        # Increment trade count
        record['trades'] += 1
        
        # Categorize and update counters
        if profit_loss > 0.001:  # Small epsilon to handle floating point
            record['wins'] += 1
            record['total_profit'] += profit_loss
            record['largest_win'] = max(record['largest_win'], profit_loss)
            record['current_consecutive_wins'] += 1
            record['current_consecutive_losses'] = 0
        elif profit_loss < -0.001:
            record['losses'] += 1
            record['total_loss'] += abs(profit_loss)
            record['largest_loss'] = max(record['largest_loss'], abs(profit_loss))
            record['current_consecutive_losses'] += 1
            record['current_consecutive_wins'] = 0
        else:
            record['breakeven'] += 1
            # Reset consecutive counts for breakeven trades
            record['current_consecutive_wins'] = 0
            record['current_consecutive_losses'] = 0
        
        # Update max consecutive counters
        record['max_consecutive_wins'] = max(record['max_consecutive_wins'], record['current_consecutive_wins'])
        record['max_consecutive_losses'] = max(record['max_consecutive_losses'], record['current_consecutive_losses'])
        
        # Update averages
        if record['wins'] > 0:
            record['average_win'] = record['total_profit'] / record['wins']
        if record['losses'] > 0:
            record['average_loss'] = record['total_loss'] / record['losses']
        
        # Update profit factor and win rate
        if record['total_loss'] > 0:
            record['profit_factor'] = record['total_profit'] / record['total_loss']
        elif record['total_profit'] > 0:  # No losses but has profits
            record['profit_factor'] = record['total_profit']  # Large number to indicate perfect profit factor
        else:
            record['profit_factor'] = 0.0  # No profits
            
        if record['trades'] > 0:
            record['win_rate'] = record['wins'] / record['trades']
        
        # Add to trade history
        record['trade_history'].append(profit_loss)
        
        # Limit history size
        if len(record['trade_history']) > 100:  # Keep last 100 trades
            record['trade_history'] = record['trade_history'][-100:]
        
        # Update timestamp
        record['last_updated'] = datetime.now().isoformat()
        
        # Calculate performance score
        record['performance_score'] = self._calculate_performance_score(record)
    
    def _calculate_performance_score(self, record: Dict[str, Any]) -> float:
        """
        Calculate normalized performance score (0-1) for a performance record.
        
        Args:
            record: Performance record
            
        Returns:
            Normalized performance score
        """
        # If not enough trades, return neutral score
        if record['trades'] < self.parameters['min_trades_for_significance']:
            return 0.5
        
        # Calculate component scores
        
        # Win rate score (0-1)
        win_rate_score = min(1.0, record['win_rate'] * 1.5)  # 0% = 0, 67%+ = 1.0
        
        # Profit factor score (0-1)
        if record['profit_factor'] <= 0:
            profit_factor_score = 0.0
        elif record['profit_factor'] >= 3.0:
            profit_factor_score = 1.0
        else:
            profit_factor_score = record['profit_factor'] / 3.0
        
        # Average return score
        # Calculate average trade return
        avg_return = (record['total_profit'] - record['total_loss']) / record['trades']
        
        # Normalize based on asset class typical returns
        # For now, using a simple approach - could be enhanced with asset-specific normalizations
        if avg_return <= 0:
            avg_return_score = 0.0
        elif avg_return >= 1.0:
            avg_return_score = 1.0
        else:
            avg_return_score = avg_return
        
        # Maximum drawdown score
        # Calculate simple drawdown estimate from trade history
        if record['trade_history']:
            # Calculate a running equity curve
            equity = [0]
            for pl in record['trade_history']:
                equity.append(equity[-1] + pl)
            
            # Find maximum drawdown
            max_drawdown = 0
            peak = equity[0]
            for value in equity[1:]:
                if value > peak:
                    peak = value
                drawdown = peak - value
                max_drawdown = max(max_drawdown, drawdown)
            
            # Normalize drawdown score
            if max_drawdown <= 0:
                max_drawdown_score = 1.0
            else:
                # Lower drawdown is better
                max_drawdown_score = max(0, 1.0 - (max_drawdown / (peak + 0.0001) * 2))
        else:
            max_drawdown_score = 0.5  # Default score
        
        # Combine scores with weights
        combined_score = (
            win_rate_score * self.parameters['win_rate_weight'] +
            profit_factor_score * self.parameters['profit_factor_weight'] +
            avg_return_score * self.parameters['avg_return_weight'] +
            max_drawdown_score * self.parameters['max_drawdown_weight']
        )
        
        return combined_score
