#!/usr/bin/env python3
"""
Forex Smart Session Analysis - Enhanced Session Intelligence Module

This module provides advanced session awareness capabilities:
- Dynamic session strength detection
- Session transition intelligence
- Adaptive session windows
- Session personality profiling
"""

import os
import sys
import yaml
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_smart_session')


class SmartSessionAnalyzer:
    """
    Enhanced session intelligence for Forex trading.
    Provides advanced analytics and insights about trading sessions.
    """
    
    def __init__(self, session_performance_tracker=None, config: Dict[str, Any] = None):
        """
        Initialize the smart session analyzer.
        
        Args:
            session_performance_tracker: Session performance tracker instance
            config: Configuration dictionary
        """
        self.session_performance_tracker = session_performance_tracker
        self.config = config or {}
        
        # Default session times in UTC
        self.session_windows = {
            'Sydney': {'start': datetime.time(20, 0), 'end': datetime.time(6, 0)},
            'Tokyo': {'start': datetime.time(23, 0), 'end': datetime.time(8, 0)},
            'London': {'start': datetime.time(7, 0), 'end': datetime.time(16, 0)},
            'NewYork': {'start': datetime.time(12, 0), 'end': datetime.time(21, 0)}
        }
        
        # Update session windows from config if available
        if 'session_windows' in self.config:
            for session, times in self.config['session_windows'].items():
                self.session_windows[session] = times
        
        # Initialize session personality profiles
        self.session_personalities = {
            'Sydney': {'range_respecting': 0.6, 'trending': 0.3, 'volatility': 0.4},
            'Tokyo': {'range_respecting': 0.7, 'trending': 0.2, 'volatility': 0.3},
            'London': {'range_respecting': 0.5, 'trending': 0.5, 'volatility': 0.6},
            'NewYork': {'range_respecting': 0.3, 'trending': 0.7, 'volatility': 0.7},
            'London-NewYork': {'range_respecting': 0.2, 'trending': 0.8, 'volatility': 0.9}
        }
        
        # Session volatility history for dynamic detection
        self.session_volatility_history = {}
        
        # Load historic volatility data if available
        self._load_volatility_history()
        
        logger.info("Smart Session Analyzer initialized")
    
    def _load_volatility_history(self) -> None:
        """Load historical volatility data from file if available."""
        volatility_file = self.config.get('volatility_history_file', 'session_volatility_history.json')
        
        if os.path.exists(volatility_file):
            try:
                with open(volatility_file, 'r') as f:
                    self.session_volatility_history = json.load(f)
                logger.info(f"Loaded volatility history from {volatility_file}")
            except Exception as e:
                logger.error(f"Error loading volatility history: {e}")
    
    def save_volatility_history(self) -> None:
        """Save volatility history to file."""
        volatility_file = self.config.get('volatility_history_file', 'session_volatility_history.json')
        
        try:
            with open(volatility_file, 'w') as f:
                json.dump(self.session_volatility_history, f, indent=2)
            logger.info(f"Saved volatility history to {volatility_file}")
        except Exception as e:
            logger.error(f"Error saving volatility history: {e}")
    
    def detect_session_strength(self, session: str, pair: str, 
                               current_data: pd.DataFrame = None, 
                               lookback_days: int = 30) -> float:
        """
        Calculate the relative strength of a trading session for the current pair.
        
        Args:
            session: Session name ('London', 'NewYork', etc.)
            pair: Currency pair ('EURUSD', 'GBPJPY', etc.)
            current_data: Current market data (optional)
            lookback_days: Number of days to look back for historical comparison
            
        Returns:
            Relative session strength (>1.0 means stronger than average)
        """
        # Get historical volatility for this session and pair
        historical_volatility = self._get_historical_volatility(session, pair, lookback_days)
        
        # Get current volatility for the session
        current_volatility = self._get_current_volatility(session, pair, current_data)
        
        # Calculate relative strength
        if historical_volatility > 0:
            relative_strength = current_volatility / historical_volatility
        else:
            relative_strength = 1.0
        
        logger.debug(f"Session strength for {session} on {pair}: {relative_strength:.2f}")
        return relative_strength
    
    def _get_historical_volatility(self, session: str, pair: str, lookback_days: int) -> float:
        """
        Get historical volatility for a session and pair.
        
        Args:
            session: Session name
            pair: Currency pair
            lookback_days: Number of days to look back
            
        Returns:
            Average historical volatility
        """
        # Use session_performance_tracker if available
        if self.session_performance_tracker:
            try:
                session_stats = self.session_performance_tracker.get_session_statistics(
                    session=session, 
                    pair=pair,
                    days=lookback_days
                )
                return session_stats.get('average_volatility', 0.0)
            except Exception as e:
                logger.error(f"Error getting session statistics: {e}")
        
        # Fallback to internal history
        session_pair_key = f"{session}_{pair}"
        if session_pair_key in self.session_volatility_history:
            volatility_history = self.session_volatility_history[session_pair_key]
            if volatility_history:
                # Calculate average of last N days
                recent_history = volatility_history[-lookback_days:]
                return sum(recent_history) / len(recent_history) if recent_history else 0.0
        
        # Fallback to typical values if no data available
        typical_volatility = {
            'Sydney': 0.3,
            'Tokyo': 0.4,
            'London': 0.6,
            'NewYork': 0.7,
            'London-NewYork': 0.8  # Overlap period
        }
        return typical_volatility.get(session, 0.5)
    
    def _get_current_volatility(self, session: str, pair: str, data: pd.DataFrame = None) -> float:
        """
        Calculate current volatility for a session and pair.
        
        Args:
            session: Session name
            pair: Currency pair
            data: Market data (optional)
            
        Returns:
            Current volatility
        """
        # If data is provided, calculate from it
        if data is not None and len(data) > 0:
            return self._calculate_volatility_from_data(data)
        
        # Use session_performance_tracker if available
        if self.session_performance_tracker:
            try:
                current_stats = self.session_performance_tracker.get_current_session_stats(
                    session=session, 
                    pair=pair
                )
                return current_stats.get('current_volatility', 0.0)
            except Exception as e:
                logger.error(f"Error getting current session stats: {e}")
        
        # Default to recent history or fallback value
        session_pair_key = f"{session}_{pair}"
        if session_pair_key in self.session_volatility_history:
            volatility_history = self.session_volatility_history[session_pair_key]
            if volatility_history:
                return volatility_history[-1]  # Most recent value
        
        # Fallback to average value
        return 0.5
    
    def _calculate_volatility_from_data(self, data: pd.DataFrame) -> float:
        """
        Calculate volatility from OHLC data.
        
        Args:
            data: OHLC data
            
        Returns:
            Volatility value
        """
        if len(data) < 2:
            return 0.0
        
        if 'high' in data.columns and 'low' in data.columns:
            # Use high-low range for volatility
            high_low_range = (data['high'] - data['low']).mean()
            return high_low_range
        elif 'close' in data.columns:
            # Use standard deviation of returns
            returns = data['close'].pct_change().dropna()
            return returns.std()
        
        return 0.0
    
    def update_volatility_history(self, session: str, pair: str, volatility: float) -> None:
        """
        Update volatility history for a session and pair.
        
        Args:
            session: Session name
            pair: Currency pair
            volatility: Volatility value to record
        """
        session_pair_key = f"{session}_{pair}"
        
        if session_pair_key not in self.session_volatility_history:
            self.session_volatility_history[session_pair_key] = []
        
        # Append new value
        self.session_volatility_history[session_pair_key].append(volatility)
        
        # Trim history to last 90 days
        max_history = 90
        if len(self.session_volatility_history[session_pair_key]) > max_history:
            self.session_volatility_history[session_pair_key] = self.session_volatility_history[session_pair_key][-max_history:]
    
    def get_best_session_for_pair(self, pair: str, strategy_id: Optional[str] = None) -> str:
        """
        Determine the best session for trading a specific pair.
        
        Args:
            pair: Currency pair
            strategy_id: Strategy ID (optional)
            
        Returns:
            Best session name
        """
        # Use session_performance_tracker if available and strategy_id provided
        if self.session_performance_tracker and strategy_id:
            try:
                best_session = self.session_performance_tracker.get_best_session_for_strategy(
                    strategy_id=strategy_id, 
                    pair=pair
                )
                if best_session:
                    return best_session
            except Exception as e:
                logger.error(f"Error getting best session for strategy: {e}")
        
        # Fallback to general pair analysis
        pair_base = pair[:3]
        pair_quote = pair[3:]
        
        # General rules for which session is best based on currencies involved
        if pair_base in ['AUD', 'NZD'] or pair_quote in ['AUD', 'NZD']:
            return 'Sydney'
        elif pair_base in ['JPY', 'CNH', 'SGD'] or pair_quote in ['JPY', 'CNH', 'SGD']:
            return 'Tokyo'
        elif pair_base in ['GBP', 'EUR', 'CHF'] or pair_quote in ['GBP', 'EUR', 'CHF']:
            return 'London'
        elif pair_base in ['USD', 'CAD', 'MXN'] or pair_quote in ['USD', 'CAD', 'MXN']:
            return 'NewYork'
        
        # Default to London-NewYork overlap for highest liquidity
        return 'London-NewYork'
    
    def get_session_transition_times(self, from_session: str, to_session: str, 
                                    as_datetime: bool = False, 
                                    reference_date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Get detailed information about session transition times.
        
        Args:
            from_session: Starting session
            to_session: Ending session
            as_datetime: Return as datetime objects
            reference_date: Reference date for datetime objects
            
        Returns:
            Dictionary with transition information
        """
        if from_session not in self.session_windows or to_session not in self.session_windows:
            return {}
        
        from_end = self.session_windows[from_session]['end']
        to_start = self.session_windows[to_session]['start']
        
        # Calculate overlap or gap
        overlap_start = max(self.session_windows[from_session]['start'], to_start)
        overlap_end = min(from_end, self.session_windows[to_session]['end'])
        
        has_overlap = not (from_end < to_start or self.session_windows[to_session]['end'] < self.session_windows[from_session]['start'])
        
        if has_overlap:
            transition_type = "overlap"
            transition_start = overlap_start
            transition_end = overlap_end
        else:
            transition_type = "gap"
            transition_start = from_end
            transition_end = to_start
        
        # Convert to datetime if requested
        if as_datetime and reference_date:
            transition_start = datetime.datetime.combine(reference_date, transition_start)
            transition_end = datetime.datetime.combine(reference_date, transition_end)
            
            # Handle day crossover
            if transition_end < transition_start:
                transition_end = transition_end + datetime.timedelta(days=1)
        
        return {
            "transition_type": transition_type,
            "transition_start": transition_start,
            "transition_end": transition_end,
            "duration_minutes": ((transition_end.hour * 60 + transition_end.minute) - 
                               (transition_start.hour * 60 + transition_start.minute)) % (24 * 60)
        }
    
    def analyze_session_personality(self, session: str, pair: str, 
                                   data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Analyze the "personality" of a trading session for a specific pair.
        
        Args:
            session: Session name
            pair: Currency pair
            data: Historical data (optional)
            
        Returns:
            Dictionary with personality traits
        """
        # If data is provided, calculate personality from it
        if data is not None and len(data) > 20:  # Need enough data points
            return self._calculate_personality_from_data(data)
        
        # Use pre-defined personalities or blend with performance data
        base_personality = self.session_personalities.get(session, {
            'range_respecting': 0.5, 
            'trending': 0.5, 
            'volatility': 0.5
        })
        
        # Use session_performance_tracker if available to refine personality
        if self.session_performance_tracker:
            try:
                session_stats = self.session_performance_tracker.get_session_statistics(
                    session=session, 
                    pair=pair,
                    days=30  # Last 30 days
                )
                
                # Blend with performance data if available
                if session_stats:
                    # Higher win rate on trend strategies suggests trending personality
                    if 'trend_win_rate' in session_stats and 'range_win_rate' in session_stats:
                        trend_strength = session_stats['trend_win_rate']
                        range_strength = session_stats['range_win_rate']
                        
                        # Normalize to sum to 1.0
                        total = trend_strength + range_strength
                        if total > 0:
                            base_personality['trending'] = trend_strength / total
                            base_personality['range_respecting'] = range_strength / total
                    
                    # Volatility from performance data
                    if 'average_volatility' in session_stats:
                        # Normalize to 0-1 range (assuming 0-1.5 range of volatility)
                        base_personality['volatility'] = min(session_stats['average_volatility'] / 1.5, 1.0)
            except Exception as e:
                logger.error(f"Error getting session statistics for personality: {e}")
        
        return base_personality
    
    def _calculate_personality_from_data(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate session personality metrics from historical data.
        
        Args:
            data: OHLC data
            
        Returns:
            Dictionary with personality traits
        """
        personality = {
            'range_respecting': 0.5,
            'trending': 0.5,
            'volatility': 0.5
        }
        
        if len(data) < 20:
            return personality
        
        try:
            # Calculate basic metrics
            if 'close' in data.columns:
                # Trend strength: autocorrelation of returns (higher = more trending)
                returns = data['close'].pct_change().dropna()
                autocorr = returns.autocorr(lag=1)
                
                # Range analysis: ratio of close-to-close vs high-to-low
                if 'high' in data.columns and 'low' in data.columns:
                    cc_volatility = returns.std()
                    hl_ranges = (data['high'] - data['low']) / data['low']
                    hl_volatility = hl_ranges.mean()
                    
                    # Range respecting = higher hl_volatility relative to cc_volatility
                    # Trending = higher cc_volatility
                    if hl_volatility > 0:
                        range_trend_ratio = cc_volatility / hl_volatility
                        
                        # Convert to 0-1 scale (higher value = more trending)
                        trending = min(max(range_trend_ratio, 0), 1)
                        personality['trending'] = trending
                        personality['range_respecting'] = 1.0 - trending
                
                # Adjust trending from autocorrelation
                if not np.isnan(autocorr):
                    # Convert -1 to +1 scale to 0-1 scale
                    trending_factor = (autocorr + 1) / 2
                    # Blend with existing trending calculation
                    personality['trending'] = (personality['trending'] + trending_factor) / 2
                    personality['range_respecting'] = 1.0 - personality['trending']
                
                # Volatility from standard deviation of returns
                volatility = returns.std() * np.sqrt(252)  # Annualized
                # Normalize to 0-1 range (assuming 0-30% annualized is the range)
                personality['volatility'] = min(volatility / 0.3, 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating personality from data: {e}")
        
        return personality
    
    def get_session_optimal_times(self, session: str, pair: str, 
                                 strategy_type: Optional[str] = None) -> Dict[str, datetime.time]:
        """
        Get optimal trading times within a session based on historical performance.
        
        Args:
            session: Session name
            pair: Currency pair
            strategy_type: Type of strategy (trend, range, etc.)
            
        Returns:
            Dictionary with optimal entry and exit times
        """
        # Get base session times
        if session not in self.session_windows:
            return {}
        
        base_start = self.session_windows[session]['start']
        base_end = self.session_windows[session]['end']
        
        # Default optimal times (first third for entry, last third for exit)
        session_minutes = ((base_end.hour * 60 + base_end.minute) - 
                        (base_start.hour * 60 + base_start.minute)) % (24 * 60)
        
        third_duration = session_minutes // 3
        
        default_entry = datetime.time(
            (base_start.hour + (third_duration // 60)) % 24,
            (base_start.minute + (third_duration % 60)) % 60
        )
        
        default_exit = datetime.time(
            (base_start.hour + (2 * third_duration // 60)) % 24,
            (base_start.minute + (2 * third_duration % 60)) % 60
        )
        
        # Adjust based on strategy type
        if strategy_type == "trend":
            # For trend strategies, better to enter early in session
            optimal_entry = base_start
            optimal_exit = default_exit
        elif strategy_type == "range":
            # For range strategies, better to enter mid-session
            optimal_entry = default_entry
            optimal_exit = default_exit
        elif strategy_type == "breakout":
            # For breakout strategies, better around session open
            minutes_after_open = 30
            optimal_entry = datetime.time(
                (base_start.hour + (minutes_after_open // 60)) % 24,
                (base_start.minute + (minutes_after_open % 60)) % 60
            )
            optimal_exit = default_exit
        else:
            # Default times
            optimal_entry = default_entry
            optimal_exit = default_exit
        
        # Use session_performance_tracker if available for better optimization
        if self.session_performance_tracker:
            try:
                optimal_times = self.session_performance_tracker.get_optimal_session_times(
                    session=session,
                    pair=pair,
                    strategy_type=strategy_type
                )
                
                if optimal_times and 'optimal_entry' in optimal_times and 'optimal_exit' in optimal_times:
                    return optimal_times
            except Exception as e:
                logger.error(f"Error getting optimal session times: {e}")
        
        return {
            "optimal_entry": optimal_entry,
            "optimal_exit": optimal_exit
        }
    
    def is_in_session_transition(self, timestamp: datetime.datetime, 
                               from_session: str, to_session: str,
                               buffer_minutes: int = 15) -> bool:
        """
        Check if the timestamp is in a session transition period.
        
        Args:
            timestamp: Timestamp to check
            from_session: Starting session
            to_session: Ending session
            buffer_minutes: Additional minutes to add around transition
            
        Returns:
            True if in transition period, False otherwise
        """
        # Convert timestamp to UTC time
        if timestamp.tzinfo is None:
            utc_timestamp = pytz.UTC.localize(timestamp)
        else:
            utc_timestamp = timestamp.astimezone(pytz.UTC)
        
        # Get transition times
        transition_info = self.get_session_transition_times(
            from_session=from_session,
            to_session=to_session,
            as_datetime=True,
            reference_date=utc_timestamp.date()
        )
        
        if not transition_info:
            return False
        
        # Add buffer
        buffer = datetime.timedelta(minutes=buffer_minutes)
        start_with_buffer = transition_info["transition_start"] - buffer
        end_with_buffer = transition_info["transition_end"] + buffer
        
        # Check if timestamp is within transition period (with buffer)
        return start_with_buffer <= utc_timestamp <= end_with_buffer


# Test function
def test_smart_session():
    """Test the smart session analyzer functionality."""
    analyzer = SmartSessionAnalyzer()
    
    # Test session strength
    strength = analyzer.detect_session_strength('London', 'EURUSD')
    print(f"London session strength for EURUSD: {strength:.2f}")
    
    # Test session personality
    personality = analyzer.analyze_session_personality('NewYork', 'GBPUSD')
    print(f"NewYork session personality for GBPUSD: {personality}")
    
    # Test optimal times
    optimal_times = analyzer.get_session_optimal_times('London-NewYork', 'EURUSD', 'trend')
    print(f"Optimal times for trend strategy in London-NewYork overlap: {optimal_times}")
    
    # Test transition detection
    now = datetime.datetime.now()
    is_transition = analyzer.is_in_session_transition(now, 'London', 'NewYork')
    print(f"Current time is in London-NewYork transition: {is_transition}")
    
    return "Smart session tests completed"


if __name__ == "__main__":
    test_smart_session()
