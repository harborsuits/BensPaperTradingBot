"""
Strategy Pattern Discovery Module

This module is responsible for discovering, classifying, and tracking specialized trading patterns
that work only for specific symbols, timeframes, or market conditions.

Key capabilities:
1. Pattern identification - Identifies repeatable patterns through ML and statistical analysis
2. Pattern classification - Categorizes patterns by type, symbol, timeframe, conditions
3. Pattern storage - Maintains a database of discovered patterns
4. Pattern application - Recommends when to apply specific patterns
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

from trading_bot.core.event_bus import EventBus, Event, EventType
from trading_bot.core.persistence import PersistenceManager
from trading_bot.ai_analysis.ml_pattern_analyzer import MLPatternAnalyzer  # Would need to be implemented

logger = logging.getLogger(__name__)

class StrategyPatternDiscovery:
    """
    Discovers and catalogs specialized trading patterns that work in specific conditions.
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 persistence_manager: Optional[PersistenceManager] = None,
                 state_dir: Optional[str] = None):
        """
        Initialize the Strategy Pattern Discovery system.
        
        Args:
            event_bus: Event bus for system communication
            persistence_manager: Optional persistence manager for state
            state_dir: Directory for state storage (created if not provided)
        """
        self.event_bus = event_bus
        self.persistence_manager = persistence_manager
        
        # Set up state directory
        if not state_dir:
            state_dir = os.path.join(os.path.dirname(__file__), '../../state/patterns')
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        
        # Pattern databases
        self.symbol_specific_patterns = {}  # Symbol-specific patterns
        self.seasonal_patterns = {}  # Time/season-specific patterns
        self.event_driven_patterns = {}  # Event-driven patterns
        self.market_condition_patterns = {}  # Market condition specific patterns
        
        # Pattern metadata
        self.pattern_performance = defaultdict(list)  # Tracks performance of each pattern
        self.pattern_confidence = {}  # Confidence scores for each pattern
        self.last_applied_patterns = {}  # When patterns were last applied
        
        # Pattern discovery tracking
        self.potential_patterns = []  # Patterns under investigation
        self.discovery_thresholds = {
            'min_occurrences': 3,  # Minimum occurrences to confirm a pattern
            'min_win_rate': 0.65,  # Minimum win rate to consider a valid pattern
            'min_profit_factor': 1.5,  # Minimum profit factor
            'significance_level': 0.05,  # Statistical significance level
        }
        
        logger.info("Strategy Pattern Discovery initialized")
        
    def register_event_handlers(self):
        """Register all event handlers for pattern discovery."""
        # Trade and performance events
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
        self.event_bus.subscribe(EventType.TRADE_OPENED, self.handle_trade_opened)
        
        # Market data events
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self.handle_market_data)
        self.event_bus.subscribe(EventType.SENTIMENT_DATA_UPDATED, self.handle_sentiment_data)
        
        # Event tracking
        self.event_bus.subscribe(EventType.ECONOMIC_CALENDAR_EVENT, self.handle_economic_event)
        self.event_bus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.handle_earnings_event)
        
        # System events
        self.event_bus.subscribe(EventType.SYSTEM_START, self.handle_system_start)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self.handle_system_shutdown)
        
        logger.info("Event handlers registered for Strategy Pattern Discovery")
        
    def handle_trade_closed(self, event: Event):
        """
        Process closed trades to find patterns in successful and unsuccessful trades.
        This is a key source for pattern discovery.
        """
        trade_data = event.data
        
        # Extract key trade data
        symbol = trade_data.get('symbol')
        profit_pips = trade_data.get('profit_pips')
        profit_percent = trade_data.get('profit_percent')
        strategy = trade_data.get('strategy')
        entry_time = trade_data.get('entry_time')
        exit_time = trade_data.get('exit_time')
        market_conditions = trade_data.get('market_conditions', {})
        
        # Skip if missing essential data
        if not all([symbol, strategy, entry_time, exit_time]):
            return
            
        # Record for pattern analysis
        self._analyze_trade_for_patterns(trade_data)
        
        # Update performance of applied patterns
        applied_pattern_ids = trade_data.get('applied_pattern_ids', [])
        for pattern_id in applied_pattern_ids:
            if pattern_id in self.pattern_performance:
                self.pattern_performance[pattern_id].append({
                    'timestamp': exit_time,
                    'profit_pips': profit_pips,
                    'profit_percent': profit_percent,
                    'success': profit_pips > 0 if profit_pips is not None else False
                })
                
                # Update pattern confidence based on recent performance
                self._update_pattern_confidence(pattern_id)
                
        # Periodically check for new patterns
        if len(self.potential_patterns) >= 50:  # Arbitrary threshold
            self._evaluate_potential_patterns()
            
    def handle_trade_opened(self, event: Event):
        """Record when patterns are applied to new trades."""
        trade_data = event.data
        applied_pattern_ids = trade_data.get('applied_pattern_ids', [])
        
        for pattern_id in applied_pattern_ids:
            self.last_applied_patterns[pattern_id] = datetime.now()
            
    def handle_market_data(self, event: Event):
        """Process market data for pattern discovery."""
        data = event.data
        symbol = data.get('symbol')
        
        # Skip if we don't have enough data
        if not symbol:
            return
            
        # Check for seasonal patterns
        self._check_seasonal_patterns(data)
        
        # Check for market condition patterns
        self._check_market_condition_patterns(data)
            
    def handle_sentiment_data(self, event: Event):
        """Process sentiment data for pattern discovery."""
        data = event.data
        symbol = data.get('symbol')
        sentiment = data.get('sentiment')
        
        # Skip if we don't have enough data
        if not symbol or sentiment is None:
            return
            
        # Check for sentiment-driven patterns
        self._check_sentiment_patterns(data)
        
    def handle_economic_event(self, event: Event):
        """Track economic events for event-driven pattern discovery."""
        data = event.data
        event_name = data.get('event_name')
        impact = data.get('impact')
        
        # Skip low-impact events
        if impact == 'low':
            return
            
        # Record event for pattern analysis
        self._record_economic_event(data)
        
    def handle_earnings_event(self, event: Event):
        """Track earnings announcements for event-driven pattern discovery."""
        data = event.data
        symbol = data.get('symbol')
        
        # Record earnings event for pattern analysis
        self._record_earnings_event(data)
        
    def handle_system_start(self, event: Event):
        """Load patterns from persistent storage."""
        self._load_patterns()
        logger.info("Loaded strategy patterns from storage")
        
    def handle_system_shutdown(self, event: Event):
        """Save patterns to persistent storage."""
        self._save_patterns()
        logger.info("Saved strategy patterns to storage")
        
    def get_applicable_patterns(self, symbol: str, timeframe: str, 
                               current_market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find patterns that are applicable to the current trading conditions.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            current_market_conditions: Dictionary of current market conditions
            
        Returns:
            List of applicable patterns with their confidence scores
        """
        applicable_patterns = []
        
        # Check symbol-specific patterns
        symbol_patterns = self.symbol_specific_patterns.get(symbol, [])
        for pattern in symbol_patterns:
            if self._is_pattern_applicable(pattern, symbol, timeframe, current_market_conditions):
                applicable_patterns.append({
                    'pattern_id': pattern.get('id'),
                    'pattern_name': pattern.get('name'),
                    'pattern_type': 'symbol_specific',
                    'confidence': self.pattern_confidence.get(pattern.get('id'), 0.5),
                    'description': pattern.get('description'),
                    'parameters': pattern.get('parameters', {})
                })
                
        # Check seasonal patterns
        current_date = datetime.now()
        month = current_date.month
        day = current_date.day
        weekday = current_date.weekday()
        
        for pattern_id, pattern in self.seasonal_patterns.items():
            # Check if the pattern applies to current date
            if self._is_seasonal_pattern_applicable(pattern, month, day, weekday, symbol):
                applicable_patterns.append({
                    'pattern_id': pattern_id,
                    'pattern_name': pattern.get('name'),
                    'pattern_type': 'seasonal',
                    'confidence': self.pattern_confidence.get(pattern_id, 0.5),
                    'description': pattern.get('description'),
                    'parameters': pattern.get('parameters', {})
                })
                
        # Check event-driven patterns
        for pattern_id, pattern in self.event_driven_patterns.items():
            if self._is_event_pattern_applicable(pattern, symbol, current_market_conditions):
                applicable_patterns.append({
                    'pattern_id': pattern_id,
                    'pattern_name': pattern.get('name'),
                    'pattern_type': 'event_driven',
                    'confidence': self.pattern_confidence.get(pattern_id, 0.5),
                    'description': pattern.get('description'),
                    'parameters': pattern.get('parameters', {})
                })
                
        # Check market condition patterns
        for pattern_id, pattern in self.market_condition_patterns.items():
            if self._is_market_condition_pattern_applicable(pattern, current_market_conditions):
                applicable_patterns.append({
                    'pattern_id': pattern_id,
                    'pattern_name': pattern.get('name'),
                    'pattern_type': 'market_condition',
                    'confidence': self.pattern_confidence.get(pattern_id, 0.5),
                    'description': pattern.get('description'),
                    'parameters': pattern.get('parameters', {})
                })
                
        # Sort by confidence
        applicable_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return applicable_patterns
        
    def create_custom_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """
        Manually create a custom pattern.
        
        Args:
            pattern_data: Pattern definition and parameters
            
        Returns:
            Pattern ID
        """
        pattern_type = pattern_data.get('pattern_type')
        pattern_id = f"{pattern_type}_{int(datetime.now().timestamp())}"
        
        # Add metadata
        pattern_data['id'] = pattern_id
        pattern_data['created_at'] = datetime.now().isoformat()
        pattern_data['created_by'] = 'manual'
        pattern_data['total_applications'] = 0
        pattern_data['successful_applications'] = 0
        
        # Store the pattern in the appropriate database
        if pattern_type == 'symbol_specific':
            symbol = pattern_data.get('symbol')
            if symbol not in self.symbol_specific_patterns:
                self.symbol_specific_patterns[symbol] = []
            self.symbol_specific_patterns[symbol].append(pattern_data)
        elif pattern_type == 'seasonal':
            self.seasonal_patterns[pattern_id] = pattern_data
        elif pattern_type == 'event_driven':
            self.event_driven_patterns[pattern_id] = pattern_data
        elif pattern_type == 'market_condition':
            self.market_condition_patterns[pattern_id] = pattern_data
            
        # Initialize confidence
        self.pattern_confidence[pattern_id] = pattern_data.get('initial_confidence', 0.5)
        
        # Save patterns
        self._save_patterns()
        
        logger.info(f"Created new custom pattern: {pattern_data.get('name')} (ID: {pattern_id})")
        return pattern_id
        
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove a pattern from the database.
        
        Args:
            pattern_id: ID of the pattern to remove
            
        Returns:
            True if the pattern was removed, False otherwise
        """
        # Check each pattern database
        for symbol, patterns in self.symbol_specific_patterns.items():
            for i, pattern in enumerate(patterns):
                if pattern.get('id') == pattern_id:
                    self.symbol_specific_patterns[symbol].pop(i)
                    if pattern_id in self.pattern_confidence:
                        del self.pattern_confidence[pattern_id]
                    if pattern_id in self.pattern_performance:
                        del self.pattern_performance[pattern_id]
                    logger.info(f"Removed pattern {pattern_id} from symbol-specific patterns")
                    self._save_patterns()
                    return True
                    
        # Check seasonal patterns
        if pattern_id in self.seasonal_patterns:
            del self.seasonal_patterns[pattern_id]
            if pattern_id in self.pattern_confidence:
                del self.pattern_confidence[pattern_id]
            if pattern_id in self.pattern_performance:
                del self.pattern_performance[pattern_id]
            logger.info(f"Removed pattern {pattern_id} from seasonal patterns")
            self._save_patterns()
            return True
            
        # Check event-driven patterns
        if pattern_id in self.event_driven_patterns:
            del self.event_driven_patterns[pattern_id]
            if pattern_id in self.pattern_confidence:
                del self.pattern_confidence[pattern_id]
            if pattern_id in self.pattern_performance:
                del self.pattern_performance[pattern_id]
            logger.info(f"Removed pattern {pattern_id} from event-driven patterns")
            self._save_patterns()
            return True
            
        # Check market condition patterns
        if pattern_id in self.market_condition_patterns:
            del self.market_condition_patterns[pattern_id]
            if pattern_id in self.pattern_confidence:
                del self.pattern_confidence[pattern_id]
            if pattern_id in self.pattern_performance:
                del self.pattern_performance[pattern_id]
            logger.info(f"Removed pattern {pattern_id} from market condition patterns")
            self._save_patterns()
            return True
            
        return False
        
    def get_pattern_performance(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Dictionary with performance statistics
        """
        if pattern_id not in self.pattern_performance:
            return {'error': 'Pattern not found'}
            
        performances = self.pattern_performance[pattern_id]
        
        # Calculate performance metrics
        total_trades = len(performances)
        if total_trades == 0:
            return {
                'pattern_id': pattern_id,
                'total_trades': 0,
                'win_rate': None,
                'avg_profit': None,
                'profit_factor': None,
                'confidence': self.pattern_confidence.get(pattern_id, 0.5)
            }
            
        winning_trades = sum(1 for p in performances if p.get('success', False))
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics if we have profit data
        profits = [p.get('profit_pips', 0) for p in performances if p.get('profit_pips') is not None]
        if profits:
            avg_profit = sum(profits) / len(profits)
            winning_profits = [p for p in profits if p > 0]
            losing_profits = [abs(p) for p in profits if p < 0]
            
            profit_factor = sum(winning_profits) / sum(losing_profits) if sum(losing_profits) > 0 else float('inf')
        else:
            avg_profit = None
            profit_factor = None
            
        return {
            'pattern_id': pattern_id,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'profit_factor': profit_factor,
            'confidence': self.pattern_confidence.get(pattern_id, 0.5),
            'last_applied': self.last_applied_patterns.get(pattern_id)
        }
        
    def _analyze_trade_for_patterns(self, trade_data: Dict[str, Any]):
        """Analyze a completed trade for potential patterns."""
        # This would analyze various aspects of the trade and market conditions
        # to detect potential patterns
        
        # Add to potential patterns for further analysis
        self.potential_patterns.append(trade_data)
        
    def _evaluate_potential_patterns(self):
        """Evaluate potential patterns to confirm or reject them."""
        # This would use statistical methods to evaluate if potential patterns
        # are statistically significant
        
        # For now, just log that we're evaluating
        logger.info(f"Evaluating {len(self.potential_patterns)} potential patterns")
        
        # Here we would implement the actual pattern discovery algorithms
        # This is a placeholder for the actual implementation
        
        # Clear potential patterns after evaluation
        self.potential_patterns = []
        
    def _update_pattern_confidence(self, pattern_id: str):
        """Update confidence score for a pattern based on recent performance."""
        if pattern_id not in self.pattern_performance:
            return
            
        performances = self.pattern_performance[pattern_id]
        
        # Get only the last N applications (e.g., 10) to focus on recent performance
        recent_performances = performances[-10:] if len(performances) > 10 else performances
        
        if not recent_performances:
            return
            
        # Calculate win rate from recent performances
        recent_wins = sum(1 for p in recent_performances if p.get('success', False))
        recent_win_rate = recent_wins / len(recent_performances)
        
        # Update confidence based on recent win rate
        # This is a simplistic approach - could be much more sophisticated
        self.pattern_confidence[pattern_id] = 0.7 * recent_win_rate + 0.3 * self.pattern_confidence.get(pattern_id, 0.5)
        
    def _is_pattern_applicable(self, pattern: Dict[str, Any], symbol: str, 
                              timeframe: str, market_conditions: Dict[str, Any]) -> bool:
        """Check if a pattern is applicable to the current conditions."""
        # This is a placeholder - actual implementation would be more sophisticated
        pattern_symbol = pattern.get('symbol')
        pattern_timeframe = pattern.get('timeframe')
        
        # Basic symbol and timeframe matching
        if pattern_symbol != symbol:
            return False
            
        if pattern_timeframe and pattern_timeframe != timeframe:
            return False
            
        # Further condition checks would go here
        
        return True
        
    def _is_seasonal_pattern_applicable(self, pattern: Dict[str, Any], 
                                      month: int, day: int, weekday: int, symbol: str) -> bool:
        """Check if a seasonal pattern applies to the current date."""
        # Match symbol if specified
        if 'symbols' in pattern and symbol not in pattern.get('symbols', []):
            return False
            
        # Check month applicability
        if 'months' in pattern and month not in pattern.get('months', []):
            return False
            
        # Check day of month applicability
        if 'days' in pattern and day not in pattern.get('days', []):
            return False
            
        # Check day of week applicability
        if 'weekdays' in pattern and weekday not in pattern.get('weekdays', []):
            return False
            
        return True
        
    def _is_event_pattern_applicable(self, pattern: Dict[str, Any], 
                                   symbol: str, market_conditions: Dict[str, Any]) -> bool:
        """Check if an event-driven pattern applies to current conditions."""
        # Match symbol if specified
        if 'symbols' in pattern and symbol not in pattern.get('symbols', []):
            return False
            
        # Check if the relevant event is active
        event_type = pattern.get('event_type')
        if event_type not in market_conditions.get('active_events', []):
            return False
            
        # Further condition checks would go here
        
        return True
        
    def _is_market_condition_pattern_applicable(self, pattern: Dict[str, Any],
                                              market_conditions: Dict[str, Any]) -> bool:
        """Check if a market condition pattern applies to current conditions."""
        # This would check various market conditions like volatility, trend, etc.
        # For now, just return a simple check
        
        required_conditions = pattern.get('required_conditions', {})
        
        for condition, value in required_conditions.items():
            if condition not in market_conditions:
                return False
                
            if market_conditions[condition] != value:
                return False
                
        return True
        
    def _check_seasonal_patterns(self, data: Dict[str, Any]):
        """Check for seasonal patterns in market data."""
        # This would implement algorithms to detect seasonal patterns
        # For now, just a placeholder
        pass
        
    def _check_market_condition_patterns(self, data: Dict[str, Any]):
        """Check for market condition patterns in market data."""
        # This would implement algorithms to detect market condition patterns
        # For now, just a placeholder
        pass
        
    def _check_sentiment_patterns(self, data: Dict[str, Any]):
        """Check for sentiment-driven patterns in sentiment data."""
        # This would implement algorithms to detect sentiment patterns
        # For now, just a placeholder
        pass
        
    def _record_economic_event(self, data: Dict[str, Any]):
        """Record an economic event for pattern analysis."""
        # This would store economic events for later analysis
        # For now, just a placeholder
        pass
        
    def _record_earnings_event(self, data: Dict[str, Any]):
        """Record an earnings event for pattern analysis."""
        # This would store earnings events for later analysis
        # For now, just a placeholder
        pass
        
    def _load_patterns(self):
        """Load patterns from persistent storage."""
        pattern_file = os.path.join(self.state_dir, 'patterns.json')
        
        if not os.path.exists(pattern_file):
            return
            
        try:
            with open(pattern_file, 'r') as f:
                patterns_data = json.load(f)
                
            self.symbol_specific_patterns = patterns_data.get('symbol_specific_patterns', {})
            self.seasonal_patterns = patterns_data.get('seasonal_patterns', {})
            self.event_driven_patterns = patterns_data.get('event_driven_patterns', {})
            self.market_condition_patterns = patterns_data.get('market_condition_patterns', {})
            self.pattern_confidence = patterns_data.get('pattern_confidence', {})
            
            # Convert string timestamps back to datetime objects for last applied
            last_applied = patterns_data.get('last_applied_patterns', {})
            self.last_applied_patterns = {}
            for pattern_id, timestamp_str in last_applied.items():
                try:
                    self.last_applied_patterns[pattern_id] = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    self.last_applied_patterns[pattern_id] = datetime.now()
                    
            # Load performance data
            if os.path.exists(os.path.join(self.state_dir, 'pattern_performance.json')):
                with open(os.path.join(self.state_dir, 'pattern_performance.json'), 'r') as f:
                    self.pattern_performance = defaultdict(list, json.load(f))
                    
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            
    def _save_patterns(self):
        """Save patterns to persistent storage."""
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            
            # Convert datetime objects to strings for serialization
            last_applied_serializable = {}
            for pattern_id, timestamp in self.last_applied_patterns.items():
                last_applied_serializable[pattern_id] = timestamp.isoformat()
                
            patterns_data = {
                'symbol_specific_patterns': self.symbol_specific_patterns,
                'seasonal_patterns': self.seasonal_patterns,
                'event_driven_patterns': self.event_driven_patterns,
                'market_condition_patterns': self.market_condition_patterns,
                'pattern_confidence': self.pattern_confidence,
                'last_applied_patterns': last_applied_serializable,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.state_dir, 'patterns.json'), 'w') as f:
                json.dump(patterns_data, f, indent=2)
                
            # Save performance data separately as it can be large
            with open(os.path.join(self.state_dir, 'pattern_performance.json'), 'w') as f:
                json.dump(dict(self.pattern_performance), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
