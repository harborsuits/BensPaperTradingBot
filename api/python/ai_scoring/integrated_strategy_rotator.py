#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IntegratedStrategyRotator - Enhanced strategy rotation system that combines AI-driven 
rotation with performance-based optimization, risk management, and dynamic constraint handling.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from trading_bot.ai_scoring.strategy_rotator import StrategyRotator
from trading_bot.ai_scoring.regime_aware_strategy_prioritizer import RegimeAwareStrategyPrioritizer
from trading_bot.utils.market_context_fetcher import MarketContextFetcher
from trading_bot.utils.telegram_notifier import TelegramNotifier
from trading_bot.utils.trade_journal import TradeJournal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegratedStrategyRotator")

class AllocationConstraintManager:
    """
    Manages the constraints for strategy allocations based on various factors 
    including risk metrics, performance history, and market regimes.
    """
    
    def __init__(self, strategies: List[str], base_constraints: Dict[str, Dict[str, float]] = None):
        """
        Initialize the AllocationConstraintManager.
        
        Args:
            strategies: List of strategy names to manage constraints for
            base_constraints: Base constraints configuration (optional)
        """
        self.strategies = strategies
        
        # Set up default constraints if not provided
        if base_constraints is None:
            self.constraints = {
                "min_allocation": {strategy: 1.0 for strategy in strategies},
                "max_allocation": {strategy: 40.0 for strategy in strategies},
                "max_change": {strategy: 15.0 for strategy in strategies}
            }
        else:
            self.constraints = base_constraints
            
        # Ensure all strategies have constraints
        self._ensure_all_strategies_have_constraints()
        
        logger.info(f"AllocationConstraintManager initialized with {len(strategies)} strategies")
    
    def _ensure_all_strategies_have_constraints(self):
        """Ensure all strategies have constraints defined."""
        for constraint_type in self.constraints:
            for strategy in self.strategies:
                if strategy not in self.constraints[constraint_type]:
                    # Set default values based on constraint type
                    if constraint_type == "min_allocation":
                        self.constraints[constraint_type][strategy] = 1.0
                    elif constraint_type == "max_allocation":
                        self.constraints[constraint_type][strategy] = 40.0
                    elif constraint_type == "max_change":
                        self.constraints[constraint_type][strategy] = 15.0
    
    def update_constraints_based_on_performance(self, 
                                             performance_metrics: Dict[str, Dict[str, float]]):
        """
        Update allocation constraints based on strategy performance metrics.
        
        Args:
            performance_metrics: Dict mapping strategies to their performance metrics
        """
        for strategy in self.strategies:
            if strategy in performance_metrics:
                metrics = performance_metrics[strategy]
                
                # Get key performance indicators
                sharpe = metrics.get("sharpe_ratio", 0.0)
                drawdown = metrics.get("max_drawdown", 0.0)
                volatility = metrics.get("volatility", 0.0)
                
                # Adjust max allocation based on Sharpe ratio and drawdown
                if sharpe > 1.5 and drawdown < 0.15:
                    # Increase max allocation for high Sharpe, low drawdown strategies
                    self.constraints["max_allocation"][strategy] = min(
                        self.constraints["max_allocation"][strategy] * 1.2,  # 20% increase
                        50.0  # Cap at 50%
                    )
                elif sharpe < 0.5 or drawdown > 0.25:
                    # Decrease max allocation for low Sharpe or high drawdown strategies
                    self.constraints["max_allocation"][strategy] = max(
                        self.constraints["max_allocation"][strategy] * 0.8,  # 20% decrease
                        self.constraints["min_allocation"][strategy] * 1.5  # Keep above min
                    )
                
                # Adjust max change based on volatility
                if volatility > 0.2:  # High volatility
                    # Reduce max change for high volatility strategies
                    self.constraints["max_change"][strategy] = max(
                        self.constraints["max_change"][strategy] * 0.8,  # 20% decrease
                        5.0  # Minimum max change
                    )
                elif volatility < 0.1:  # Low volatility
                    # Allow more change for low volatility strategies
                    self.constraints["max_change"][strategy] = min(
                        self.constraints["max_change"][strategy] * 1.2,  # 20% increase
                        25.0  # Cap at 25%
                    )
        
        logger.info("Updated allocation constraints based on performance metrics")
    
    def update_constraints_based_on_regime(self, regime: str, regime_confidence: float = 0.7):
        """
        Update allocation constraints based on the detected market regime.
        
        Args:
            regime: The detected market regime
            regime_confidence: Confidence level in the regime detection (0-1)
        """
        # Only make significant adjustments if confidence is high
        if regime_confidence < 0.6:
            logger.info(f"Skipping regime-based constraint updates due to low confidence ({regime_confidence:.2f})")
            return
        
        # Define regime-specific constraint adjustments
        regime_adjustments = {
            "bullish": {
                "momentum": {"max_allocation": 50.0, "min_allocation": 5.0},
                "trend_following": {"max_allocation": 45.0, "min_allocation": 5.0},
                "volatility_breakout": {"max_allocation": 15.0}
            },
            "bearish": {
                "momentum": {"max_allocation": 15.0},
                "trend_following": {"max_allocation": 40.0, "min_allocation": 5.0},
                "volatility_breakout": {"max_allocation": 30.0, "min_allocation": 5.0}
            },
            "volatile": {
                "momentum": {"max_allocation": 10.0, "max_change": 10.0},
                "volatility_breakout": {"max_allocation": 50.0, "min_allocation": 10.0},
                "option_spreads": {"max_allocation": 40.0, "min_allocation": 5.0}
            },
            "sideways": {
                "mean_reversion": {"max_allocation": 50.0, "min_allocation": 10.0},
                "option_spreads": {"max_allocation": 35.0, "min_allocation": 5.0},
                "trend_following": {"max_allocation": 20.0}
            }
        }
        
        # Apply regime-specific adjustments if available
        if regime in regime_adjustments:
            for strategy, adjustments in regime_adjustments[regime].items():
                if strategy in self.strategies:
                    for constraint_type, value in adjustments.items():
                        if constraint_type in self.constraints and strategy in self.constraints[constraint_type]:
                            self.constraints[constraint_type][strategy] = value
            
            logger.info(f"Updated allocation constraints based on {regime} regime")
    
    def get_constraints(self) -> Dict[str, Dict[str, float]]:
        """Get the current allocation constraints."""
        return self.constraints
    
    def set_constraint(self, 
                      constraint_type: str, 
                      strategy: str, 
                      value: float) -> None:
        """
        Set a specific constraint value for a strategy.
        
        Args:
            constraint_type: Type of constraint ('min_allocation', 'max_allocation', or 'max_change')
            strategy: Strategy name
            value: Constraint value
        """
        if constraint_type in self.constraints and strategy in self.strategies:
            self.constraints[constraint_type][strategy] = value
            logger.info(f"Set {constraint_type} for {strategy} to {value}")


class PerformanceOptimizer:
    """
    Optimizes strategy allocations based on historical performance metrics
    and risk-adjusted return calculations.
    """
    
    def __init__(self, 
                strategies: List[str],
                lookback_days: int = 90,
                risk_weight: float = 0.5):
        """
        Initialize the PerformanceOptimizer.
        
        Args:
            strategies: List of strategy names
            lookback_days: Number of days to look back for performance data
            risk_weight: Weight given to risk metrics in optimization (0-1)
        """
        self.strategies = strategies
        self.lookback_days = lookback_days
        self.risk_weight = risk_weight
        
        logger.info(f"PerformanceOptimizer initialized with {len(strategies)} strategies "
                  f"and {lookback_days} days lookback")
    
    def get_performance_based_allocations(self, 
                                        performance_data: Dict[str, Dict[str, float]],
                                        current_allocations: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate suggested allocations based on historical performance metrics.
        
        Args:
            performance_data: Dict of performance metrics for each strategy
            current_allocations: Current strategy allocations (optional)
            
        Returns:
            Dict of suggested allocations for each strategy
        """
        if not performance_data:
            logger.warning("No performance data provided for optimization")
            return {}
            
        # Calculate performance scores
        performance_scores = self._calculate_performance_scores(performance_data)
        
        # Calculate suggested allocations based on scores
        suggested_allocations = self._calculate_allocations_from_scores(performance_scores)
        
        # If current allocations provided, blend with suggested allocations
        if current_allocations:
            blended_allocations = self._blend_allocations(
                current_allocations, suggested_allocations, blend_ratio=0.3
            )
            return blended_allocations
        
        return suggested_allocations
    
    def _calculate_performance_scores(self, 
                                    performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate performance scores for each strategy based on risk-adjusted metrics.
        
        Args:
            performance_data: Dict of performance metrics for each strategy
            
        Returns:
            Dict of performance scores for each strategy
        """
        scores = {}
        
        for strategy in self.strategies:
            if strategy not in performance_data:
                scores[strategy] = 0.0
                continue
                
            metrics = performance_data[strategy]
            
            # Extract key metrics (with reasonable defaults if missing)
            returns = metrics.get("returns", 0.0)
            sharpe = metrics.get("sharpe_ratio", 0.0)
            sortino = metrics.get("sortino_ratio", 0.0)
            max_dd = metrics.get("max_drawdown", 1.0)  # Default to 100% drawdown if missing
            volatility = metrics.get("volatility", 1.0)
            profit_factor = metrics.get("profit_factor", 1.0)
            
            # Calculate return score (higher is better)
            return_score = max(min(returns * 5, 10), -5)  # Scale returns and cap
            
            # Calculate risk score (lower risk is better)
            risk_score = (
                ((1 - max_dd) * 2) +  # Lower drawdown is better
                min(sharpe, 3) +  # Higher Sharpe is better
                min(sortino, 3) +  # Higher Sortino is better
                (1 / (volatility + 0.05)) +  # Lower volatility is better
                min(profit_factor, 3)  # Higher profit factor is better
            ) / 5  # Average the components
            
            # Combine return and risk scores with risk_weight
            overall_score = return_score * (1 - self.risk_weight) + risk_score * self.risk_weight
            
            # Apply a sigmoid-like function to ensure scores are positive and reasonable
            scores[strategy] = max(1 / (1 + np.exp(-overall_score)), 0.01)
        
        return scores
    
    def _calculate_allocations_from_scores(self, 
                                         performance_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate suggested allocations based on performance scores.
        
        Args:
            performance_scores: Dict of performance scores for each strategy
            
        Returns:
            Dict of suggested allocations for each strategy
        """
        # Normalize scores to sum to 100%
        total_score = sum(performance_scores.values())
        
        if total_score > 0:
            allocations = {
                strategy: (score / total_score) * 100 
                for strategy, score in performance_scores.items()
            }
        else:
            # Equal allocation if no positive scores
            equal_allocation = 100 / len(self.strategies)
            allocations = {strategy: equal_allocation for strategy in self.strategies}
        
        return allocations
    
    def _blend_allocations(self,
                          current: Dict[str, float],
                          suggested: Dict[str, float],
                          blend_ratio: float = 0.5) -> Dict[str, float]:
        """
        Blend current allocations with suggested allocations.
        
        Args:
            current: Current strategy allocations
            suggested: Suggested strategy allocations
            blend_ratio: How much weight to give to suggested allocations (0-1)
            
        Returns:
            Dict of blended allocations for each strategy
        """
        blended = {}
        
        for strategy in self.strategies:
            current_alloc = float(current.get(strategy, 0.0))
            suggested_alloc = float(suggested.get(strategy, 0.0))
            
            # Blend the allocations
            blended[strategy] = current_alloc * (1 - blend_ratio) + suggested_alloc * blend_ratio
        
        # Normalize to ensure allocations sum to 100%
        total = sum(blended.values())
        if total > 0:
            blended = {k: (v / total) * 100 for k, v in blended.items()}
        
        return blended


class IntegratedStrategyRotator:
    """
    Enhanced strategy rotation system that combines AI-driven rotation with 
    performance-based optimization, risk management, and dynamic constraint handling.
    """
    
    def __init__(self, 
                strategies: List[str],
                initial_allocations: Dict[str, float] = None,
                portfolio_value: float = 100000.0,
                use_mock: bool = False,
                config_path: str = None,
                state_path: str = None):
        """
        Initialize the IntegratedStrategyRotator.
        
        Args:
            strategies: List of strategy names
            initial_allocations: Initial allocation percentages (default to equal allocation)
            portfolio_value: Total portfolio value
            use_mock: Whether to use mock responses
            config_path: Path to configuration file
            state_path: Path to state file for saving/loading allocations
        """
        self.strategies = strategies
        self.portfolio_value = portfolio_value
        self.use_mock = use_mock
        
        # Set default paths if not provided
        if config_path is None:
            config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "strategy_rotator_config.json")
        self.config_path = config_path
        
        if state_path is None:
            state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(state_dir, exist_ok=True)
            state_path = os.path.join(state_dir, "strategy_rotator_state.json")
        self.state_path = state_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self._init_components()
        
        # Set initial allocations (equal if not provided)
        if initial_allocations is None:
            equal_allocation = 100.0 / len(strategies)
            initial_allocations = {strategy: equal_allocation for strategy in strategies}
        
        # Ensure allocations include all strategies
        for strategy in strategies:
            if strategy not in initial_allocations:
                initial_allocations[strategy] = 0.0
        
        # Initialize current allocations (load from state or use initial)
        state = self._load_state()
        self.current_allocations = state.get("current_allocations", initial_allocations)
        self._normalize_allocations()
        
        # Initialize rotation history
        self.rotation_history = state.get("rotation_history", [])
        
        logger.info(f"IntegratedStrategyRotator initialized with {len(strategies)} strategies")
        if use_mock:
            logger.info("Using mock responses for strategy prioritization")
    
    def _init_components(self):
        """Initialize component modules needed for rotation."""
        # Strategy Prioritizer: Enhanced with regime awareness
        self.strategy_prioritizer = RegimeAwareStrategyPrioritizer(
            strategies=self.strategies,
            use_mock=self.use_mock,
            regime_lookback_days=self.config.get("regime_lookback_days", 60)
        )
        
        # Constraint Manager: Manages allocation constraints
        self.constraint_manager = AllocationConstraintManager(
            strategies=self.strategies,
            base_constraints=self.config.get("allocation_constraints", None)
        )
        
        # Performance Optimizer: Optimizes based on historical performance
        self.performance_optimizer = PerformanceOptimizer(
            strategies=self.strategies,
            lookback_days=self.config.get("performance_lookback_days", 90),
            risk_weight=self.config.get("risk_weight", 0.5)
        )
        
        # Market Context Fetcher: Gets market context
        self.market_fetcher = MarketContextFetcher()
        
        # Trade Journal: Records trades and performance
        self.journal = TradeJournal(
            db_path=self.config.get("journal_db_path"),
            strategies=self.strategies
        )
        
        # Notifier: Sends notifications about rotations
        self.notifier = TelegramNotifier(
            token=self.config.get("telegram_token", os.getenv("TELEGRAM_BOT_TOKEN")),
            chat_id=self.config.get("telegram_chat_id", os.getenv("TELEGRAM_CHAT_ID"))
        )
    
    def _normalize_allocations(self) -> None:
        """Ensure current allocations sum to 100%."""
        total = sum(float(v) for v in self.current_allocations.values())
        if total > 0:
            self.current_allocations = {
                k: (float(v) / total) * 100 for k, v in self.current_allocations.items()
            }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "rotation_interval_days": 7,
            "min_allocation_percent": 1.0,
            "max_allocation_percent": 40.0,
            "max_allocation_change_percent": 15.0,
            "journal_db_path": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "data", 
                "trade_journal.db"
            ),
            "regime_lookback_days": 60,
            "performance_lookback_days": 90,
            "risk_weight": 0.5,
            "use_performance_optimization": True,
            "optimization_blend_ratio": 0.3,
            "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
            "send_notifications": True
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Update with any missing defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {str(e)}")
        
        return default_config
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or return empty state."""
        empty_state = {
            "current_allocations": {},
            "rotation_history": [],
            "last_rotation_date": None
        }
        
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state from {self.state_path}: {str(e)}")
        
        return empty_state
    
    def _save_state(self) -> None:
        """Save current state to file."""
        state = {
            "current_allocations": self.current_allocations,
            "rotation_history": self.rotation_history,
            "last_rotation_date": datetime.now().isoformat()
        }
        
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"State saved to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving state to {self.state_path}: {str(e)}")
    
    def get_strategy_performance(self, 
                               lookback_days: int = None) -> Dict[str, Dict[str, float]]:
        """
        Get recent performance metrics for each strategy.
        
        Args:
            lookback_days: Number of days to look back (default to config value)
            
        Returns:
            Dict mapping strategies to their performance metrics
        """
        if lookback_days is None:
            lookback_days = self.config.get("performance_lookback_days", 90)
        
        start_date = datetime.now() - timedelta(days=lookback_days)
        
        performance_data = {}
        for strategy in self.strategies:
            # Get strategy performance from journal
            metrics = self.journal.get_strategy_metrics(
                strategy_name=strategy,
                start_date=start_date
            )
            
            if metrics:
                performance_data[strategy] = metrics
            else:
                # Default metrics if no data
                performance_data[strategy] = {
                    "returns": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                    "profit_factor": 1.0
                }
        
        return performance_data
    
    def is_rotation_due(self) -> bool:
        """
        Check if it's time to perform a strategy rotation.
        
        Returns:
            Boolean indicating if rotation is due
        """
        state = self._load_state()
        last_rotation = state.get("last_rotation_date")
        
        # If no rotation has been done yet, it's due
        if last_rotation is None:
            return True
        
        # Parse date from ISO format
        try:
            last_rotation_date = datetime.fromisoformat(last_rotation)
            days_since_rotation = (datetime.now() - last_rotation_date).days
            rotation_interval = self.config.get("rotation_interval_days", 7)
            
            return days_since_rotation >= rotation_interval
        except Exception as e:
            logger.error(f"Error parsing last rotation date: {str(e)}")
            return True
    
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update the total portfolio value.
        
        Args:
            new_value: New portfolio value
        """
        self.portfolio_value = new_value
        logger.info(f"Portfolio value updated to ${new_value:.2f}")
    
    def get_current_allocations(self) -> Dict[str, Dict[str, float]]:
        """
        Get current strategy allocations with dollar amounts.
        
        Returns:
            Dict with percentage and dollar allocations
        """
        result = {"percent": {}, "dollars": {}}
        
        for strategy, allocation in self.current_allocations.items():
            result["percent"][strategy] = allocation
            result["dollars"][strategy] = (allocation / 100) * self.portfolio_value
        
        return result
    
    def rotate_strategies(self, 
                         market_context: Optional[Dict[str, Any]] = None,
                         force_rotation: bool = False) -> Dict[str, Any]:
        """
        Perform strategy rotation based on market conditions and strategy performance.
        
        Args:
            market_context: Optional market context data (fetched if not provided)
            force_rotation: Whether to force rotation regardless of schedule
            
        Returns:
            Dict containing rotation results
        """
        # Check if rotation is due unless forcing
        if not force_rotation and not self.is_rotation_due():
            logger.info("Strategy rotation not due yet")
            return {
                "rotated": False,
                "message": "Strategy rotation not due yet",
                "current_allocations": self.get_current_allocations()
            }
        
        # Get market context if not provided
        if market_context is None:
            market_context = self.market_fetcher.get_market_context()
        
        # Get regime classification
        regime_data = self.strategy_prioritizer.get_regime_classification()
        primary_regime = regime_data["primary_regime"]
        regime_confidence = regime_data["confidence_scores"][primary_regime]
        
        # Update constraints based on regime
        self.constraint_manager.update_constraints_based_on_regime(
            regime=primary_regime,
            regime_confidence=regime_confidence
        )
        
        # Get performance data
        performance_data = self.get_strategy_performance()
        
        # Update constraints based on performance
        self.constraint_manager.update_constraints_based_on_performance(performance_data)
        
        # Get target allocations from different sources
        # 1. AI-based prioritization
        ai_prioritization = self.strategy_prioritizer.prioritize_strategies(market_context)
        ai_allocations = {
            strategy: float(allocation) 
            for strategy, allocation in ai_allocations_data.items()
        } if (ai_allocations_data := ai_prioritization.get("allocations", {})) else {}
        
        # 2. Performance-based optimization (if enabled)
        if self.config.get("use_performance_optimization", True):
            performance_allocations = self.performance_optimizer.get_performance_based_allocations(
                performance_data=performance_data,
                current_allocations=self.current_allocations
            )
            
            # Blend AI and performance allocations
            blend_ratio = self.config.get("optimization_blend_ratio", 0.3)
            target_allocations = {}
            
            for strategy in self.strategies:
                ai_alloc = ai_allocations.get(strategy, 0.0)
                perf_alloc = performance_allocations.get(strategy, 0.0)
                
                # Weighted average of AI and performance allocations
                target_allocations[strategy] = ai_alloc * (1 - blend_ratio) + perf_alloc * blend_ratio
        else:
            # Use AI allocations only
            target_allocations = ai_allocations
        
        # Ensure target allocations sum to 100%
        target_total = sum(float(v) for v in target_allocations.values())
        if target_total > 0:
            target_allocations = {
                k: (float(v) / target_total) * 100 for k, v in target_allocations.items()
            }
        
        # Apply allocation constraints
        constrained_allocations = self._apply_allocation_constraints(
            current_allocations=self.current_allocations,
            target_allocations=target_allocations
        )
        
        # Calculate allocation changes
        allocation_changes = {}
        for strategy in self.strategies:
            current = float(self.current_allocations.get(strategy, 0.0))
            target = float(constrained_allocations.get(strategy, 0.0))
            allocation_changes[strategy] = target - current
        
        # Create summary of market context
        market_summary = self._get_market_context_summary(market_context, regime_data)
        
        # Record rotation in history
        rotation_entry = {
            "date": datetime.now().isoformat(),
            "regime": primary_regime,
            "regime_confidence": regime_confidence,
            "previous_allocations": self.current_allocations.copy(),
            "new_allocations": constrained_allocations.copy(),
            "allocation_changes": allocation_changes,
            "market_context": market_summary,
            "ai_reasoning": ai_prioritization.get("reasoning", "No reasoning provided"),
            "performance_metrics": {
                k: {m: v[m] for m in ["returns", "sharpe_ratio", "max_drawdown"] 
                   if m in v}
                for k, v in performance_data.items()
            }
        }
        
        self.rotation_history.append(rotation_entry)
        
        # Update current allocations
        self.current_allocations = constrained_allocations
        
        # Save state
        self._save_state()
        
        # Send notification if enabled
        if self.config.get("send_notifications", True):
            self._send_rotation_notification(rotation_entry)
        
        # Return rotation results
        return {
            "rotated": True,
            "previous_allocations": rotation_entry["previous_allocations"],
            "new_allocations": rotation_entry["new_allocations"],
            "allocation_changes": rotation_entry["allocation_changes"],
            "regime": primary_regime,
            "regime_confidence": regime_confidence,
            "ai_reasoning": ai_prioritization.get("reasoning", "No reasoning provided"),
            "dollar_allocations": {
                strategy: (allocation / 100) * self.portfolio_value
                for strategy, allocation in constrained_allocations.items()
            }
        }
    
    def _apply_allocation_constraints(self,
                                    current_allocations: Dict[str, float],
                                    target_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Apply allocation constraints to ensure smooth transitions and respect limits.
        
        Args:
            current_allocations: Current strategy allocations
            target_allocations: Target strategy allocations
            
        Returns:
            Dict of constrained allocations
        """
        constraints = self.constraint_manager.get_constraints()
        constrained_allocations = {}
        
        # First pass: Apply min/max allocation and max change constraints
        for strategy in self.strategies:
            current_alloc = float(current_allocations.get(strategy, 0.0))
            target_alloc = float(target_allocations.get(strategy, 0.0))
            
            # Get constraints for this strategy
            min_alloc = constraints["min_allocation"].get(strategy, 1.0)
            max_alloc = constraints["max_allocation"].get(strategy, 40.0)
            max_change = constraints["max_change"].get(strategy, 15.0)
            
            # Apply max change constraint
            change = target_alloc - current_alloc
            if abs(change) > max_change:
                # Limit the change to max_change
                change = max_change if change > 0 else -max_change
                constrained_alloc = current_alloc + change
            else:
                constrained_alloc = target_alloc
            
            # Apply min/max allocation constraints
            constrained_alloc = max(min_alloc, min(constrained_alloc, max_alloc))
            
            constrained_allocations[strategy] = constrained_alloc
        
        # Second pass: Normalize to ensure allocations sum to 100%
        total_allocation = sum(constrained_allocations.values())
        
        if total_allocation != 100.0:
            # Calculate adjustment needed for each allocation
            adjustment_factor = 100.0 / total_allocation
            
            # Apply adjustment while respecting constraints
            adjusted_allocations = {}
            for strategy, allocation in constrained_allocations.items():
                adjusted_allocation = allocation * adjustment_factor
                min_alloc = constraints["min_allocation"].get(strategy, 1.0)
                
                # Ensure minimum allocation is maintained
                adjusted_allocations[strategy] = max(adjusted_allocation, min_alloc)
            
            # Final normalization to exactly 100%
            total = sum(adjusted_allocations.values())
            normalized_allocations = {
                k: (v / total) * 100 for k, v in adjusted_allocations.items()
            }
            
            return normalized_allocations
        
        return constrained_allocations
    
    def _get_market_context_summary(self, 
                                  market_context: Dict[str, Any],
                                  regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of market context for recording in rotation history.
        
        Args:
            market_context: Full market context data
            regime_data: Market regime classification data
            
        Returns:
            Dict containing summary of key market indicators
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime_data["primary_regime"],
            "secondary_characteristics": regime_data["secondary_characteristics"],
            "regime_confidence": regime_data["confidence_scores"][regime_data["primary_regime"]]
        }
        
        # Extract key metrics from market context
        try:
            summary["vix"] = market_context.get("vix", {}).get("current_value", 0.0)
            summary["trend_strength"] = market_context.get("trend_strength", {}).get("value", 0.0)
            summary["market_breadth"] = market_context.get("market_breadth", {}).get("value", 0.0)
            
            # Get top sectors
            sector_data = market_context.get("sector_performance", {})
            if sector_data:
                # Convert values to float and sort
                try:
                    sorted_sectors = sorted(
                        [(k, float(v)) for k, v in sector_data.items()], 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    top_sectors = {k: v for k, v in sorted_sectors[:3]}
                    worst_sectors = {k: v for k, v in sorted_sectors[-3:]}
                    
                    summary["top_sectors"] = top_sectors
                    summary["worst_sectors"] = worst_sectors
                except (ValueError, TypeError):
                    logger.warning("Could not sort sector performance due to non-numeric values")
                    summary["top_sectors"] = {}
                    summary["worst_sectors"] = {}
            
            # Get economic indicators
            summary["economic_indicators"] = market_context.get("economic_indicators", {})
            
        except Exception as e:
            logger.error(f"Error creating market context summary: {str(e)}")
        
        return summary
    
    def _send_rotation_notification(self, rotation_entry: Dict[str, Any]) -> None:
        """
        Send a notification about strategy rotation via TelegramNotifier.
        
        Args:
            rotation_entry: Dict containing rotation details
        """
        if not self.notifier.is_configured():
            logger.warning("Notifier not configured, skipping notification")
            return
        
        try:
            # Format notification message
            message = f"🔄 *Strategy Rotation ({rotation_entry['date']})*\n\n"
            message += f"*Market Regime:* {rotation_entry['regime']} "
            message += f"(Confidence: {rotation_entry['regime_confidence']:.2f})\n\n"
            
            # Add allocation changes
            message += "*Allocation Changes:*\n"
            for strategy, change in rotation_entry["allocation_changes"].items():
                emoji = "🔼" if change > 0 else "🔽" if change < 0 else "-"
                message += f"{emoji} {strategy}: {rotation_entry['previous_allocations'][strategy]:.1f}% → "
                message += f"{rotation_entry['new_allocations'][strategy]:.1f}% "
                message += f"({change:+.1f}%)\n"
            
            # Add dollar allocations
            message += "\n*Dollar Allocations:*\n"
            for strategy, allocation in rotation_entry["new_allocations"].items():
                dollars = (allocation / 100) * self.portfolio_value
                message += f"• {strategy}: ${dollars:.2f}\n"
            
            # Send notification
            self.notifier.send_message(message)
            logger.info("Rotation notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending rotation notification: {str(e)}")
    
    def get_rotation_history(self, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent rotation history entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of rotation history entries
        """
        # Return most recent entries first
        return list(reversed(self.rotation_history[-limit:]))
    
    def export_allocations_to_csv(self, 
                                file_path: str = None) -> str:
        """
        Export current allocations to a CSV file.
        
        Args:
            file_path: Path to save CSV file (optional)
            
        Returns:
            Path to saved CSV file
        """
        if file_path is None:
            export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(
                export_dir, 
                f"strategy_allocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        
        try:
            # Create dataframe from allocations
            df = pd.DataFrame({
                'Strategy': list(self.current_allocations.keys()),
                'Allocation (%)': list(self.current_allocations.values()),
                'Allocation ($)': [
                    (alloc / 100) * self.portfolio_value 
                    for alloc in self.current_allocations.values()
                ]
            })
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Allocations exported to {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting allocations to CSV: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # List of strategies
    strategies = [
        "momentum", 
        "trend_following", 
        "breakout_swing", 
        "mean_reversion", 
        "volatility_breakout", 
        "option_spreads"
    ]
    
    # Initial allocations
    initial_allocations = {
        "momentum": 20.0,
        "trend_following": 25.0,
        "breakout_swing": 20.0,
        "mean_reversion": 15.0,
        "volatility_breakout": 10.0,
        "option_spreads": 10.0
    }
    
    # Initialize rotator
    rotator = IntegratedStrategyRotator(
        strategies=strategies,
        initial_allocations=initial_allocations,
        portfolio_value=100000.0,
        use_mock=True  # Use mock for testing
    )
    
    # Perform rotation
    rotation_result = rotator.rotate_strategies(force_rotation=True)
    
    # Display results
    print("\nIntegrated Strategy Rotation Results:")
    print(f"Market Regime: {rotation_result['regime']} (Confidence: {rotation_result['regime_confidence']:.2f})")
    
    print("\nNew Allocations:")
    for strategy, allocation in rotation_result["new_allocations"].items():
        dollars = rotation_result["dollar_allocations"][strategy]
        print(f"{strategy}: {allocation:.1f}% (${dollars:.2f})")
    
    print("\nAllocation Changes:")
    for strategy, change in rotation_result["allocation_changes"].items():
        direction = "↑" if change > 0 else "↓" if change < 0 else "-"
        print(f"{strategy}: {direction} {change:+.1f}%")
    
    print("\nAI Reasoning:")
    print(rotation_result["ai_reasoning"]) 