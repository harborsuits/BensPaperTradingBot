#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Strategy Selector

This module extends the base forex strategy selector with risk profiling,
trader preferences, and strategy combination capabilities.
"""

import logging
import json
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector
from trading_bot.strategies.strategy_template import MarketRegime, TimeFrame
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.strategy_selection.risk_profile_manager import RiskProfileManager, RiskToleranceLevel
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class TradingTimeOptimality(Enum):
    """Enum representing how optimal current trading time is."""
    OPTIMAL = 3
    GOOD = 2
    SUBOPTIMAL = 1
    POOR = 0


class EnhancedStrategySelector:
    """
    Enhanced strategy selector that considers risk profile, 
    trader preferences, and optimal trading times to select strategies.
    """
    
    def __init__(self, 
                 base_selector: Optional[ForexStrategySelector] = None,
                 risk_profile_manager: Optional[RiskProfileManager] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the enhanced strategy selector.
        
        Args:
            base_selector: Base forex strategy selector
            risk_profile_manager: Risk profile manager
            config_path: Path to configuration file
        """
        # Initialize base selector if not provided
        self.base_selector = base_selector or ForexStrategySelector()
        
        # Initialize risk profile manager if not provided
        self.risk_profile_manager = risk_profile_manager or RiskProfileManager()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Strategy factory for creating strategy instances
        self.strategy_factory = StrategyFactory()
        
        # Event bus for publishing selection events
        self.event_bus = EventBus()
        
        # Historical performance cache
        self.strategy_performance = {}
        
        # Trading time performance data
        self.time_performance = self._initialize_time_performance()
        
        # Trading conditions history
        self.conditions_history = []
        
        # Trader preferences
        self.preferences = self._initialize_preferences()
        
        logger.info("Enhanced Strategy Selector initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "strategy_combination": {
                "enabled": True,
                "max_strategies": 3,
                "min_compatibility_score": 0.5,
                "diversification_factor": 0.7
            },
            "time_optimization": {
                "enabled": True,
                "session_weight": 0.4,
                "volatility_weight": 0.3,
                "historical_performance_weight": 0.3,
                "update_interval_days": 7
            },
            "performance_tracking": {
                "lookback_periods": {
                    "short_term": 30,  # days
                    "medium_term": 90,
                    "long_term": 365
                },
                "metrics": ["sharpe_ratio", "win_rate", "profit_factor", "average_trade_pips"]
            },
            "strategy_weights": {
                "regime_compatibility": 0.35,
                "historical_performance": 0.25,
                "risk_profile_match": 0.20,
                "trader_preference": 0.20
            }
        }
        
        # If config path provided, attempt to load it
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Deep merge with default config
                    self._deep_merge(default_config, loaded_config)
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _initialize_time_performance(self) -> Dict[str, Any]:
        """
        Initialize trading time performance data.
        
        Returns:
            Dictionary with time performance data
        """
        # Create a 24x7 grid for day-hour performance
        hours = list(range(24))
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Initialize with neutral values
        time_grid = {}
        for day in days:
            time_grid[day] = {}
            for hour in hours:
                time_grid[day][hour] = {
                    "win_rate": 0.5,
                    "avg_pips": 0.0,
                    "trade_count": 0,
                    "volatility": "medium",
                    "optimality": TradingTimeOptimality.SUBOPTIMAL.name
                }
        
        # Time periods optimality (pre-configured based on common knowledge)
        session_optimality = {
            # Asian session
            ("Monday", list(range(0, 8))): TradingTimeOptimality.GOOD,
            ("Tuesday", list(range(0, 8))): TradingTimeOptimality.GOOD,
            ("Wednesday", list(range(0, 8))): TradingTimeOptimality.GOOD,
            ("Thursday", list(range(0, 8))): TradingTimeOptimality.GOOD,
            ("Friday", list(range(0, 8))): TradingTimeOptimality.GOOD,
            
            # London open (optimal)
            ("Monday", list(range(8, 12))): TradingTimeOptimality.OPTIMAL,
            ("Tuesday", list(range(8, 12))): TradingTimeOptimality.OPTIMAL,
            ("Wednesday", list(range(8, 12))): TradingTimeOptimality.OPTIMAL,
            ("Thursday", list(range(8, 12))): TradingTimeOptimality.OPTIMAL,
            ("Friday", list(range(8, 12))): TradingTimeOptimality.OPTIMAL,
            
            # London/NY overlap (optimal)
            ("Monday", list(range(13, 17))): TradingTimeOptimality.OPTIMAL,
            ("Tuesday", list(range(13, 17))): TradingTimeOptimality.OPTIMAL,
            ("Wednesday", list(range(13, 17))): TradingTimeOptimality.OPTIMAL,
            ("Thursday", list(range(13, 17))): TradingTimeOptimality.OPTIMAL,
            ("Friday", list(range(13, 17))): TradingTimeOptimality.OPTIMAL,
            
            # NY afternoon (good)
            ("Monday", list(range(17, 21))): TradingTimeOptimality.GOOD,
            ("Tuesday", list(range(17, 21))): TradingTimeOptimality.GOOD,
            ("Wednesday", list(range(17, 21))): TradingTimeOptimality.GOOD,
            ("Thursday", list(range(17, 21))): TradingTimeOptimality.GOOD,
            ("Friday", list(range(17, 21))): TradingTimeOptimality.GOOD,
            
            # Late NY/Early Asian (suboptimal)
            ("Monday", list(range(21, 24))): TradingTimeOptimality.SUBOPTIMAL,
            ("Tuesday", list(range(21, 24))): TradingTimeOptimality.SUBOPTIMAL,
            ("Wednesday", list(range(21, 24))): TradingTimeOptimality.SUBOPTIMAL,
            ("Thursday", list(range(21, 24))): TradingTimeOptimality.SUBOPTIMAL,
            
            # Friday close to Sunday (poor)
            ("Friday", list(range(21, 24))): TradingTimeOptimality.POOR,
            ("Saturday", list(range(0, 24))): TradingTimeOptimality.POOR,
            ("Sunday", list(range(0, 21))): TradingTimeOptimality.POOR,
            
            # Sunday pre-Asian (suboptimal)
            ("Sunday", list(range(21, 24))): TradingTimeOptimality.SUBOPTIMAL
        }
        
        # Apply pre-configured optimality
        for (day, hours_list), optimality in session_optimality.items():
            for hour in hours_list:
                if day in time_grid and hour in time_grid[day]:
                    time_grid[day][hour]["optimality"] = optimality.name
        
        return {
            "time_grid": time_grid,
            "last_updated": datetime.now().isoformat(),
            "data_points": 0
        }
    
    def _initialize_preferences(self) -> Dict[str, Any]:
        """
        Initialize trader preferences.
        
        Returns:
            Dictionary with trader preferences
        """
        return {
            "preferred_timeframes": [TimeFrame.HOURS_1.name, TimeFrame.HOURS_4.name, TimeFrame.DAILY.name],
            "preferred_strategies": [],
            "excluded_strategies": [],
            "preferred_currency_pairs": [],
            "trading_hours": {
                "start": "08:00",
                "end": "17:00",
                "timezone": "UTC"
            },
            "weekend_trading": False,
            "news_trading": {
                "enabled": True,
                "impact_threshold": "medium"  # minimum news impact to consider
            }
        }
    
    def set_risk_profile(self, profile_name: str) -> bool:
        """
        Set the active risk profile for strategy selection.
        
        Args:
            profile_name: Name of the risk profile to load
            
        Returns:
            True if profile was loaded successfully, False otherwise
        """
        profile = self.risk_profile_manager.load_profile(profile_name)
        return profile is not None
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update trader preferences.
        
        Args:
            preferences: Dictionary with updated preferences
        """
        # Deep merge preferences
        self._deep_merge(self.preferences, preferences)
        logger.info("Updated trader preferences")
    
    def evaluate_trading_time(self, current_time: Optional[datetime] = None) -> TradingTimeOptimality:
        """
        Evaluate how optimal the current time is for trading.
        
        Args:
            current_time: Current time (if None, use system time)
            
        Returns:
            Trading time optimality enum
        """
        # Use current time if not provided
        if current_time is None:
            current_time = datetime.now()
        
        # Get day of week and hour
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = day_names[current_time.weekday()]
        hour = current_time.hour
        
        # Check time grid
        if day in self.time_performance["time_grid"] and hour in self.time_performance["time_grid"][day]:
            optimality_str = self.time_performance["time_grid"][day][hour]["optimality"]
            try:
                return TradingTimeOptimality[optimality_str]
            except KeyError:
                logger.warning(f"Unknown optimality value: {optimality_str}")
        
        # Default to suboptimal if not found
        return TradingTimeOptimality.SUBOPTIMAL
    
    def should_trade_now(self, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Determine if trading should be done at the current time.
        
        Args:
            current_time: Current time (if None, use system time)
            
        Returns:
            Tuple of (should_trade, reason)
        """
        # Use current time if not provided
        if current_time is None:
            current_time = datetime.now()
        
        # Check if weekend and weekend trading is disabled
        if current_time.weekday() >= 5 and not self.preferences["weekend_trading"]:  # 5,6 = Sat,Sun
            return False, "Weekend trading is disabled"
        
        # Check if within trading hours
        try:
            start_hour, start_minute = map(int, self.preferences["trading_hours"]["start"].split(":"))
            end_hour, end_minute = map(int, self.preferences["trading_hours"]["end"].split(":"))
            
            start_time = current_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            end_time = current_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
            
            if not (start_time <= current_time <= end_time):
                return False, "Outside of preferred trading hours"
        except (ValueError, KeyError):
            logger.warning("Invalid trading hours format, ignoring time restriction")
        
        # Check trading time optimality
        optimality = self.evaluate_trading_time(current_time)
        if optimality == TradingTimeOptimality.POOR:
            return False, "Poor trading time (low liquidity/high spread expected)"
        
        return True, "Trading conditions acceptable"
    
    def select_optimal_strategy(self, data: Dict[str, pd.DataFrame], 
                              current_time: Optional[datetime] = None,
                              symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Select the optimal strategy combination based on current market conditions,
        risk profile, and trader preferences.
        
        Args:
            data: Dictionary of market data frames
            current_time: Current time (if None, use system time)
            symbols: List of symbols to consider (if None, use all available)
            
        Returns:
            Dictionary with selected strategies and metadata
        """
        # Use current time if not provided
        if current_time is None:
            current_time = datetime.now()
        
        # Use available symbols if not specified
        if symbols is None and data:
            symbols = list(data.keys())
        elif symbols is None:
            symbols = []
        
        # Filter to preferred currency pairs if set
        if self.preferences["preferred_currency_pairs"]:
            preferred_pairs = set(self.preferences["preferred_currency_pairs"])
            symbols = [s for s in symbols if s in preferred_pairs]
        
        # Check if we should trade now
        should_trade, reason = self.should_trade_now(current_time)
        if not should_trade:
            logger.info(f"Not trading: {reason}")
            return {
                "should_trade": False,
                "reason": reason,
                "strategies": [],
                "timestamp": current_time.isoformat()
            }
        
        # Get market regime from base selector
        market_regime = self.base_selector._detect_market_regime(data)
        
        # Score all available strategies
        strategy_scores = self._score_strategies(data, market_regime, current_time, symbols)
        
        # Select optimal strategy combination
        if self.config["strategy_combination"]["enabled"]:
            selected_strategies = self._select_strategy_combination(strategy_scores, market_regime)
        else:
            # Just pick the highest scoring strategy
            top_strategy = max(strategy_scores.items(), key=lambda x: x[1]["total_score"])
            selected_strategies = [{"strategy_type": top_strategy[0], **top_strategy[1]}]
        
        # Create event for strategy selection
        event_data = {
            "timestamp": current_time.isoformat(),
            "market_regime": market_regime.name,
            "selected_strategies": [s["strategy_type"] for s in selected_strategies],
            "trading_time_optimality": self.evaluate_trading_time(current_time).name
        }
        
        event = Event(
            event_type=EventType.STRATEGY_SELECTED,
            source=self.__class__.__name__,
            data=event_data
        )
        self.event_bus.publish(event)
        
        # Return selection results
        return {
            "should_trade": True,
            "market_regime": market_regime.name,
            "strategies": selected_strategies,
            "timestamp": current_time.isoformat(),
            "trading_time_optimality": self.evaluate_trading_time(current_time).name
        }
    
    def _score_strategies(self, data: Dict[str, pd.DataFrame], 
                          market_regime: MarketRegime,
                          current_time: datetime,
                          symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Score all available strategies based on multiple factors.
        
        Args:
            data: Dictionary of market data
            market_regime: Current market regime
            current_time: Current time
            symbols: List of symbols to consider
            
        Returns:
            Dictionary mapping strategy types to scores and metadata
        """
        # Get all available strategy types
        available_strategies = self.strategy_factory.get_available_strategies()
        
        # Filter out excluded strategies
        if self.preferences["excluded_strategies"]:
            available_strategies = [s for s in available_strategies if s not in self.preferences["excluded_strategies"]]
        
        # Initialize scores dictionary
        strategy_scores = {}
        
        # Weights for different scoring factors
        weights = self.config["strategy_weights"]
        
        # For each strategy type
        for strategy_type in available_strategies:
            # Skip non-forex strategies
            if not strategy_type.lower().startswith("forex"):
                continue
                
            # Initialize strategy instance
            strategy = self.strategy_factory.create_strategy(strategy_type)
            
            if not strategy:
                continue
            
            # 1. Score based on regime compatibility
            regime_score = strategy.get_regime_compatibility_score(market_regime)
            
            # 2. Score based on historical performance
            performance_score = self._get_performance_score(strategy_type, market_regime)
            
            # 3. Score based on risk profile match
            risk_score = self._get_risk_profile_score(strategy_type)
            
            # 4. Score based on trader preferences
            preference_score = self._get_preference_score(strategy_type)
            
            # Calculate weighted total score
            total_score = (
                regime_score * weights["regime_compatibility"] +
                performance_score * weights["historical_performance"] +
                risk_score * weights["risk_profile_match"] +
                preference_score * weights["trader_preference"]
            )
            
            # Store scores and metadata
            strategy_scores[strategy_type] = {
                "total_score": total_score,
                "regime_score": regime_score,
                "performance_score": performance_score,
                "risk_score": risk_score,
                "preference_score": preference_score,
                "market_regime": market_regime.name,
                "timestamp": current_time.isoformat()
            }
        
        return strategy_scores
    
    def _get_performance_score(self, strategy_type: str, market_regime: MarketRegime) -> float:
        """
        Calculate score based on historical performance.
        
        Args:
            strategy_type: Strategy type
            market_regime: Current market regime
            
        Returns:
            Performance score (0-1)
        """
        # Default score if no historical data
        if strategy_type not in self.strategy_performance:
            return 0.5
        
        # Get performance data
        performance = self.strategy_performance[strategy_type]
        
        # Check for regime-specific performance
        if "regime_performance" in performance and market_regime.name in performance["regime_performance"]:
            regime_perf = performance["regime_performance"][market_regime.name]
            
            # Calculate score based on key metrics
            score = 0.0
            count = 0
            
            if "win_rate" in regime_perf:
                score += min(1.0, regime_perf["win_rate"] / 0.6)  # normalize: 60% win rate -> 1.0
                count += 1
                
            if "profit_factor" in regime_perf:
                score += min(1.0, regime_perf["profit_factor"] / 2.0)  # normalize: 2.0 PF -> 1.0
                count += 1
                
            if "sharpe_ratio" in regime_perf:
                score += min(1.0, regime_perf["sharpe_ratio"] / 1.5)  # normalize: 1.5 Sharpe -> 1.0
                count += 1
            
            # Return average if we have data
            if count > 0:
                return score / count
        
        # Default moderate score if no regime-specific performance
        return 0.5
    
    def _get_risk_profile_score(self, strategy_type: str) -> float:
        """
        Calculate score based on risk profile compatibility.
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            Risk profile score (0-1)
        """
        # Default score if no risk profile is loaded
        if not self.risk_profile_manager.current_profile:
            return 0.5
        
        # Use risk profile manager to calculate compatibility
        # Convert from forex_X to just X for the compatibility check
        base_type = strategy_type.lower().replace("forex_", "")
        return self.risk_profile_manager.get_strategy_compatibility(base_type)
    
    def _get_preference_score(self, strategy_type: str) -> float:
        """
        Calculate score based on trader preferences.
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            Preference score (0-1)
        """
        # Default score
        score = 0.5
        
        # Boost score for preferred strategies
        if strategy_type in self.preferences["preferred_strategies"]:
            score = 0.9
        
        # Lower score for strategies using non-preferred timeframes
        # Note: This would need to reference strategy metadata in a real implementation
        # For now, we're using a simplified approach based on strategy type
        if strategy_type.lower() == "forex_scalping" and TimeFrame.MINUTES_5.name not in self.preferences["preferred_timeframes"]:
            score -= 0.2
        elif strategy_type.lower() == "forex_day_trading" and TimeFrame.HOURS_1.name not in self.preferences["preferred_timeframes"]:
            score -= 0.2
        elif strategy_type.lower() == "forex_swing" and TimeFrame.HOURS_4.name not in self.preferences["preferred_timeframes"]:
            score -= 0.2
        elif strategy_type.lower() == "forex_position" and TimeFrame.DAILY.name not in self.preferences["preferred_timeframes"]:
            score -= 0.2
            
        # Cap the score
        return max(0.1, min(1.0, score))
    
    def _select_strategy_combination(self, strategy_scores: Dict[str, Dict[str, Any]], 
                                   market_regime: MarketRegime) -> List[Dict[str, Any]]:
        """
        Select optimal combination of strategies based on scores and diversification.
        
        Args:
            strategy_scores: Dictionary of strategy scores
            market_regime: Current market regime
            
        Returns:
            List of selected strategies with metadata
        """
        # Sort strategies by total score
        sorted_strategies = sorted(strategy_scores.items(), 
                                   key=lambda x: x[1]["total_score"], 
                                   reverse=True)
        
        # Get configuration parameters
        max_strategies = self.config["strategy_combination"]["max_strategies"]
        min_score = self.config["strategy_combination"]["min_compatibility_score"]
        diversification_factor = self.config["strategy_combination"]["diversification_factor"]
        
        # Select strategies
        selected = []
        selected_types = set()  # For tracking strategy types
        
        # Always include the top strategy if it meets minimum score
        if sorted_strategies and sorted_strategies[0][1]["total_score"] >= min_score:
            strategy_type = sorted_strategies[0][0]
            selected.append({"strategy_type": strategy_type, **sorted_strategies[0][1]})
            
            # Extract general strategy type (e.g., "trend_following" from "forex_trend_following")
            base_type = strategy_type.lower().replace("forex_", "")
            selected_types.add(base_type)
        
        # Consider additional strategies
        for strategy_type, scores in sorted_strategies[1:]:
            # Stop if we've reached the maximum
            if len(selected) >= max_strategies:
                break
                
            # Skip if below minimum score
            if scores["total_score"] < min_score:
                continue
                
            # Extract general strategy type
            base_type = strategy_type.lower().replace("forex_", "")
            
            # Check for strategy type diversity
            if base_type in selected_types and diversification_factor > 0.5:
                # Skip if we already have this type and diversity is important
                continue
                
            # Add strategy to selection
            selected.append({"strategy_type": strategy_type, **scores})
            selected_types.add(base_type)
        
        # If risk profile prefers diversification, ensure we have at least 2 strategies
        if self.risk_profile_manager.current_profile and \
           hasattr(self.risk_profile_manager.current_profile, 'diversification_preference') and \
           self.risk_profile_manager.current_profile.diversification_preference > 0.7 and \
           len(selected) < 2 and len(sorted_strategies) >= 2:
            
            # Try to add one more strategy for diversification
            for strategy_type, scores in sorted_strategies:
                if {"strategy_type": strategy_type} not in selected and scores["total_score"] >= min_score * 0.8:
                    selected.append({"strategy_type": strategy_type, **scores})
                    break
        
        # Normalize allocation weights based on scores
        total_score = sum(s["total_score"] for s in selected)
        if total_score > 0:
            for strategy in selected:
                strategy["allocation_weight"] = strategy["total_score"] / total_score
        elif selected:  # If all scores are 0 but we have selections
            equal_weight = 1.0 / len(selected)
            for strategy in selected:
                strategy["allocation_weight"] = equal_weight
        
        return selected
    
    def update_time_performance(self, performance_data: Dict[str, Any]) -> None:
        """
        Update trading time performance data with recent results.
        
        Args:
            performance_data: Dictionary with performance data
        """
        # Validate input
        if 'trades' not in performance_data or not performance_data['trades']:
            logger.warning("No trade data provided for time performance update")
            return
            
        trades = performance_data['trades']
        updated_count = 0
        
        # Process each trade
        for trade in trades:
            # Skip if missing required data
            if 'entry_time' not in trade or 'exit_time' not in trade or 'profit_pips' not in trade:
                continue
                
            try:
                # Parse timestamps
                entry_time = datetime.fromisoformat(trade['entry_time'])
                
                # Get day and hour
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day = day_names[entry_time.weekday()]
                hour = entry_time.hour
                
                # Update time grid
                if day in self.time_performance["time_grid"] and hour in self.time_performance["time_grid"][day]:
                    grid_cell = self.time_performance["time_grid"][day][hour]
                    
                    # Update win rate
                    win_count = grid_cell.get("win_count", 0)
                    trade_count = grid_cell.get("trade_count", 0)
                    
                    if trade['profit_pips'] > 0:
                        win_count += 1
                    
                    trade_count += 1
                    
                    grid_cell["win_count"] = win_count
                    grid_cell["trade_count"] = trade_count
                    grid_cell["win_rate"] = win_count / trade_count if trade_count > 0 else 0.5
                    
                    # Update average pips
                    total_pips = grid_cell.get("total_pips", 0) + trade['profit_pips']
                    grid_cell["total_pips"] = total_pips
                    grid_cell["avg_pips"] = total_pips / trade_count if trade_count > 0 else 0
                    
                    # Update optimality based on performance
                    self._update_time_optimality(day, hour)
                    
                    updated_count += 1
            except Exception as e:
                logger.error(f"Error updating time performance: {str(e)}")
        
        # Update metadata
        self.time_performance["last_updated"] = datetime.now().isoformat()
        self.time_performance["data_points"] += updated_count
        
        logger.info(f"Updated trading time performance with {updated_count} trades")
    
    def _update_time_optimality(self, day: str, hour: int) -> None:
        """
        Update the optimality rating for a time slot based on performance.
        
        Args:
            day: Day of week
            hour: Hour (0-23)
        """
        if day not in self.time_performance["time_grid"] or hour not in self.time_performance["time_grid"][day]:
            return
            
        grid_cell = self.time_performance["time_grid"][day][hour]
        
        # Need minimum trades for reliable data
        if grid_cell.get("trade_count", 0) < 10:
            return
            
        # Calculate optimality based on win rate and average pips
        win_rate = grid_cell.get("win_rate", 0.5)
        avg_pips = grid_cell.get("avg_pips", 0)
        
        # Define thresholds
        if win_rate > 0.6 and avg_pips > 5:  # Excellent performance
            grid_cell["optimality"] = TradingTimeOptimality.OPTIMAL.name
        elif win_rate > 0.5 and avg_pips > 2:  # Good performance
            grid_cell["optimality"] = TradingTimeOptimality.GOOD.name
        elif win_rate < 0.4 and avg_pips < 0:  # Poor performance
            grid_cell["optimality"] = TradingTimeOptimality.POOR.name
        else:  # Average performance
            grid_cell["optimality"] = TradingTimeOptimality.SUBOPTIMAL.name
    
    def update_strategy_performance(self, performance_data: Dict[str, Any]) -> None:
        """
        Update strategy historical performance data.
        
        Args:
            performance_data: Dictionary with performance data
        """
        # Validate input
        if 'strategy_type' not in performance_data or 'results' not in performance_data:
            logger.warning("Invalid performance data format")
            return
            
        strategy_type = performance_data['strategy_type']
        results = performance_data['results']
        
        # Initialize if not exists
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = {
                "overall": {},
                "regime_performance": {},
                "last_updated": datetime.now().isoformat()
            }
        
        # Update overall performance
        if 'overall' in results:
            self.strategy_performance[strategy_type]["overall"] = results['overall']
            
        # Update regime-specific performance
        if 'regime_performance' in results:
            for regime, metrics in results['regime_performance'].items():
                self.strategy_performance[strategy_type]["regime_performance"][regime] = metrics
        
        # Update last updated timestamp
        self.strategy_performance[strategy_type]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Updated performance data for {strategy_type}")
    
    def export_strategy_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a map of strategies to market regimes and risk profiles.
        Used for helping traders understand when to use each strategy.
        
        Returns:
            Dictionary mapping market regimes to recommended strategies
        """
        # Initialize map structure
        strategy_map = {regime.name: [] for regime in MarketRegime}
        
        # Get all available strategies
        available_strategies = self.strategy_factory.get_available_strategies()
        forex_strategies = [s for s in available_strategies if s.lower().startswith("forex")]
        
        # For each strategy, calculate compatibility with each regime
        for strategy_type in forex_strategies:
            # Create strategy instance
            strategy = self.strategy_factory.create_strategy(strategy_type)
            if not strategy:
                continue
                
            # Get risk profile compatibility (if available)
            risk_score = self._get_risk_profile_score(strategy_type)
            
            # For each market regime
            for regime in MarketRegime:
                # Get regime compatibility score
                regime_score = strategy.get_regime_compatibility_score(regime)
                
                # Only include if reasonably compatible
                if regime_score >= 0.5:
                    strategy_info = {
                        "strategy_type": strategy_type,
                        "compatibility_score": regime_score,
                        "risk_compatibility": risk_score
                    }
                    
                    strategy_map[regime.name].append(strategy_info)
        
        # Sort strategies by compatibility score within each regime
        for regime in strategy_map:
            strategy_map[regime] = sorted(strategy_map[regime], 
                                          key=lambda x: x["compatibility_score"], 
                                          reverse=True)
        
        return strategy_map
    
    def save_state(self, file_path: str) -> bool:
        """
        Save the current state of the enhanced strategy selector.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state = {
                "time_performance": self.time_performance,
                "strategy_performance": self.strategy_performance,
                "preferences": self.preferences,
                "config": self.config,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved strategy selector state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy selector state: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Load a previously saved state.
        
        Args:
            file_path: Path to load the state from
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(file_path):
            logger.warning(f"State file does not exist: {file_path}")
            return False
            
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            if "time_performance" in state:
                self.time_performance = state["time_performance"]
                
            if "strategy_performance" in state:
                self.strategy_performance = state["strategy_performance"]
                
            if "preferences" in state:
                self.preferences = state["preferences"]
                
            if "config" in state:
                self._deep_merge(self.config, state["config"])
                
            logger.info(f"Loaded strategy selector state from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading strategy selector state: {str(e)}")
            return False
