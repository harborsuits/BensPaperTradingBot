#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated strategy rotator that combines AI-powered strategy evaluation with
enhanced allocation controls for optimal capital distribution.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import json
from datetime import datetime

from trading_bot.ai_scoring.strategy_rotator import StrategyRotator as AIStrategyRotator

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedRotator:
    """
    Optimized strategy rotator with improved allocation controls.
    """
    
    def __init__(self, min_allocation: float = 0.05, max_allocation: float = 0.4):
        """
        Initialize the enhanced strategy rotator.
        
        Args:
            min_allocation: Minimum allocation for any active strategy (0.05 = 5%)
            max_allocation: Maximum allocation for any strategy (0.4 = 40%)
        """
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        logger.info(f"Enhanced rotator initialized with min={min_allocation*100}%, max={max_allocation*100}%")
    
    def allocate(self, strategy_scores: Dict[str, float], constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Allocate capital based on strategy scores.
        
        Args:
            strategy_scores: Dictionary mapping strategy names to their scores
            constraints: Optional constraints on allocation
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        if not strategy_scores:
            return {}
        
        # Start with scores as initial weights
        weights = strategy_scores.copy()
        
        # Ensure all weights are positive
        weights = {k: max(0, v) for k, v in weights.items()}
        
        # Handle constraints
        if constraints:
            weights = self._apply_constraints(weights, constraints)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # If all weights are 0, use equal weighting
            equal_weight = 1.0 / len(weights)
            weights = {k: equal_weight for k in weights.keys()}
        
        # Apply min and max allocation constraints
        weights = self._apply_min_max(weights)
        
        # Convert to percentages
        allocation = {k: v * 100 for k, v in weights.items()}
        
        return allocation
    
    def _apply_constraints(self, weights: Dict[str, float], constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply specific constraints to weights.
        
        Args:
            weights: Dictionary of weights
            constraints: Dictionary of constraints
            
        Returns:
            Adjusted weights
        """
        adjusted_weights = weights.copy()
        
        # Apply disabled strategies
        if 'disabled' in constraints:
            for strategy in constraints['disabled']:
                if strategy in adjusted_weights:
                    adjusted_weights[strategy] = 0
                    logger.info(f"Disabling strategy: {strategy}")
        
        # Apply minimum weights
        if 'min_weights' in constraints:
            for strategy, min_weight in constraints['min_weights'].items():
                if strategy in adjusted_weights:
                    if adjusted_weights[strategy] < min_weight:
                        logger.info(f"Enforcing minimum weight for {strategy}: {min_weight:.2f}")
                        adjusted_weights[strategy] = min_weight
        
        # Apply maximum weights
        if 'max_weights' in constraints:
            for strategy, max_weight in constraints['max_weights'].items():
                if strategy in adjusted_weights:
                    if adjusted_weights[strategy] > max_weight:
                        logger.info(f"Enforcing maximum weight for {strategy}: {max_weight:.2f}")
                        adjusted_weights[strategy] = max_weight
        
        return adjusted_weights
    
    def _apply_min_max(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum and maximum allocation constraints.
        
        Args:
            weights: Dictionary of normalized weights
            
        Returns:
            Adjusted weights
        """
        adjusted_weights = {}
        
        # Identify active strategies (weight > 0)
        active_strategies = {k: v for k, v in weights.items() if v > 0}
        if not active_strategies:
            return weights
        
        # Check if min allocation would exceed 100%
        total_min = len(active_strategies) * self.min_allocation
        if total_min >= 1.0:
            # Distribute equally
            equal_weight = 1.0 / len(active_strategies)
            logger.warning(f"Min allocations would exceed 100%, using equal weighting: {equal_weight:.2f}")
            return {k: (equal_weight if k in active_strategies else 0) for k in weights.keys()}
        
        # Adjust for minimum allocation
        remaining = 1.0
        for strategy in list(active_strategies.keys()):
            if weights[strategy] < self.min_allocation:
                adjusted_weights[strategy] = self.min_allocation
                remaining -= self.min_allocation
                logger.debug(f"Setting {strategy} to minimum: {self.min_allocation:.2f}")
            else:
                adjusted_weights[strategy] = weights[strategy]
                remaining -= weights[strategy]
        
        # If any weight exceeds max_allocation, cap it
        over_max = {k: v for k, v in adjusted_weights.items() if v > self.max_allocation}
        if over_max:
            excess = sum(v - self.max_allocation for v in over_max.values())
            
            # Cap those exceeding max
            for strategy in over_max:
                logger.debug(f"Capping {strategy} to maximum: {self.max_allocation:.2f}")
                adjusted_weights[strategy] = self.max_allocation
            
            # Distribute excess to others proportionally
            under_max = {k: v for k, v in adjusted_weights.items() 
                        if v < self.max_allocation and k in active_strategies}
            if under_max:
                under_max_total = sum(under_max.values())
                if under_max_total > 0:
                    for strategy in under_max:
                        proportion = under_max[strategy] / under_max_total
                        adjusted_weights[strategy] += excess * proportion
        
        # Add inactive strategies back
        for strategy in weights:
            if strategy not in adjusted_weights:
                adjusted_weights[strategy] = 0
        
        # Normalize to ensure sum is 1.0
        total = sum(adjusted_weights.values())
        if total > 0:
            return {k: v / total for k, v in adjusted_weights.items()}
        else:
            return weights


class IntegratedStrategyRotator:
    """
    Combines the AI-powered strategy rotator with enhanced allocation controls
    from the new EnhancedRotator implementation.
    """
    
    def __init__(
        self,
        ai_rotator: AIStrategyRotator,
        min_allocation: float = 0.05,
        max_allocation: float = 0.4
    ):
        """
        Initialize the integrated strategy rotator.
        
        Args:
            ai_rotator: AI-powered strategy rotator instance
            min_allocation: Minimum allocation for any active strategy
            max_allocation: Maximum allocation for any strategy
        """
        self.ai_rotator = ai_rotator
        self.enhanced_rotator = EnhancedRotator(
            min_allocation=min_allocation,
            max_allocation=max_allocation
        )
        
        # Copy configuration from AI rotator
        self.strategies = ai_rotator.strategies
        self.portfolio_value = ai_rotator.portfolio_value
        self.current_allocations = ai_rotator.current_allocations
        
        # Initialize history
        self.allocation_history = []
        
        logger.info(f"Integrated strategy rotator initialized with {len(self.strategies)} strategies")
    
    def rotate_strategies(self, market_context: Optional[Dict[str, Any]] = None, 
                         force_rotation: bool = False) -> Dict[str, Any]:
        """
        Perform strategy rotation using the AI rotator for scoring and
        enhanced rotator for allocation optimization.
        
        Args:
            market_context: Market context data
            force_rotation: Force rotation even if not due
            
        Returns:
            Rotation results
        """
        # First, check if rotation is due using the AI rotator's logic
        if not force_rotation and not self.ai_rotator.is_rotation_due():
            return {
                'status': 'skipped',
                'message': 'Rotation not due yet',
                'days_to_next_rotation': self.ai_rotator.rotation_frequency_days - 
                    (datetime.now() - self.ai_rotator.last_rotation_date).days
                    if self.ai_rotator.last_rotation_date else 0
            }
        
        # Use AI rotator to get initial strategy scores and target allocations
        ai_rotation_result = self.ai_rotator.rotate_strategies(
            market_context, 
            force_rotation=True  # We already checked if due
        )
        
        if ai_rotation_result['status'] != 'success':
            return ai_rotation_result
        
        # Extract target allocations from AI rotation
        target_allocations = ai_rotation_result['results']['target_allocations']
        
        # Convert percentages to scores for enhanced rotator
        strategy_scores = {
            strategy: allocation / 100.0  # Convert percentage to fraction
            for strategy, allocation in target_allocations.items()
        }
        
        # Apply constraints based on market context
        constraints = self._derive_constraints(market_context)
        
        # Use enhanced rotator to optimize allocations
        optimized_allocations = self.enhanced_rotator.allocate(
            strategy_scores,
            constraints=constraints
        )
        
        # Update AI rotator allocations with optimized values
        updated_result = ai_rotation_result.copy()
        updated_result['results']['constrained_allocations'] = optimized_allocations
        
        # Create log of allocation differences
        allocation_diffs = {}
        for strategy in self.strategies:
            original = target_allocations.get(strategy, 0.0)
            optimized = optimized_allocations.get(strategy, 0.0)
            if abs(original - optimized) > 0.5:  # Only log meaningful differences
                allocation_diffs[strategy] = {
                    'original': original,
                    'optimized': optimized,
                    'difference': optimized - original
                }
        
        if allocation_diffs:
            logger.info("Guardrail adjustments applied:")
            for strategy, diffs in allocation_diffs.items():
                logger.info(f"  {strategy}: {diffs['original']:.1f}% → {diffs['optimized']:.1f}% ({diffs['difference']:+.1f}%)")
        
        # Update current allocations
        self.current_allocations = optimized_allocations
        self.ai_rotator.current_allocations = optimized_allocations
        
        # Calculate dollar values
        dollar_values = {}
        for strategy, allocation in optimized_allocations.items():
            dollar_values[strategy] = (allocation / 100.0) * self.portfolio_value
        
        updated_result['results']['dollar_values'] = dollar_values
        updated_result['results']['guardrail_adjustments'] = allocation_diffs
        
        # Track history
        self.allocation_history.append({
            'timestamp': datetime.now().isoformat(),
            'original_allocations': target_allocations,
            'optimized_allocations': optimized_allocations,
            'constraints': constraints
        })
        
        # Save state using AI rotator's method
        self.ai_rotator._save_state()
        
        return updated_result
    
    def _derive_constraints(self, market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Derive allocation constraints based on market context.
        
        Args:
            market_context: Market context data
            
        Returns:
            Constraints dictionary
        """
        constraints = {}
        
        if not market_context:
            return constraints
        
        # Extract regime information if available
        regime = None
        if 'regime' in market_context:
            if isinstance(market_context['regime'], dict):
                regime = market_context['regime'].get('primary_regime')
            elif isinstance(market_context['regime'], str):
                regime = market_context['regime']
        
        # Get portfolio drawdown if available in AI rotator
        portfolio_drawdown = getattr(self.ai_rotator, 'portfolio_drawdown', 0.0)
        
        # Apply regime-specific constraints
        if regime:
            # Strategies to disable in certain regimes
            disabled = []
            
            # Minimum weights for certain strategies
            min_weights = {}
            
            # Maximum weights for certain strategies
            max_weights = {}
            
            # Apply strategy risk levels if available
            strategy_risk_levels = getattr(self.ai_rotator, 'strategy_risk_levels', {})
            
            if regime == 'volatile':
                # In volatile regimes, limit high-risk strategies
                for strategy in self.strategies:
                    if strategy in strategy_risk_levels and strategy_risk_levels[strategy] == 'high':
                        max_weights[strategy] = 0.15  # 15% max for high-risk in volatile regimes
            
            elif regime == 'bearish':
                # In bearish regimes, ensure defensive strategies have minimum allocation
                for strategy in self.strategies:
                    if strategy in strategy_risk_levels and strategy_risk_levels[strategy] == 'low':
                        min_weights[strategy] = 0.1  # 10% min for low-risk in bearish regimes
            
            # Add constraints if any were set
            if disabled:
                constraints['disabled'] = disabled
            if min_weights:
                constraints['min_weights'] = min_weights
            if max_weights:
                constraints['max_weights'] = max_weights
        
        # Apply drawdown-based constraints
        if portfolio_drawdown > 10.0:
            # Severe drawdown - further limit high-risk strategies
            max_weights = constraints.get('max_weights', {})
            for strategy in self.strategies:
                if hasattr(self.ai_rotator, 'strategy_risk_levels') and strategy in self.ai_rotator.strategy_risk_levels:
                    if self.ai_rotator.strategy_risk_levels[strategy] == 'high':
                        # 10% max for high-risk in severe drawdowns
                        max_weights[strategy] = 0.1
            constraints['max_weights'] = max_weights
            
            # Ensure minimum allocation to low-risk strategies
            min_weights = constraints.get('min_weights', {})
            for strategy in self.strategies:
                if hasattr(self.ai_rotator, 'strategy_risk_levels') and strategy in self.ai_rotator.strategy_risk_levels:
                    if self.ai_rotator.strategy_risk_levels[strategy] == 'low':
                        # 15% min for low-risk in severe drawdowns
                        min_weights[strategy] = 0.15
            constraints['min_weights'] = min_weights
                
        return constraints
    
    def get_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations.
        
        Returns:
            Dictionary with current allocations
        """
        return self.current_allocations
    
    def get_dollar_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations in dollar amounts.
        
        Returns:
            Dictionary with current dollar allocations
        """
        return {
            strategy: (allocation / 100.0) * self.portfolio_value
            for strategy, allocation in self.current_allocations.items()
        }
    
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update the total portfolio value.
        
        Args:
            new_value: New portfolio value
        """
        self.portfolio_value = new_value
        self.ai_rotator.update_portfolio_value(new_value)
    
    def is_rotation_due(self) -> bool:
        """
        Check if it's time to perform a strategy rotation.
        
        Returns:
            True if rotation is due, False otherwise
        """
        return self.ai_rotator.is_rotation_due()
    
    def manual_adjust_allocation(self, strategy: str, new_allocation: float, normalize: bool = True) -> Dict[str, Any]:
        """
        Manually adjust the allocation for a specific strategy.
        
        Args:
            strategy: Strategy name
            new_allocation: New allocation percentage
            normalize: Whether to normalize all allocations after adjustment
            
        Returns:
            Dictionary with adjustment results
        """
        return self.ai_rotator.manual_adjust_allocation(strategy, new_allocation, normalize)
    
    def get_rotation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of strategy rotations.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of rotation records, most recent first
        """
        return self.ai_rotator.get_rotation_history(limit)
    
    def reset_to_equal_allocation(self) -> Dict[str, Any]:
        """
        Reset allocations to equal distribution across all strategies.
        
        Returns:
            Dictionary with reset results
        """
        return self.ai_rotator.reset_to_equal_allocation() 