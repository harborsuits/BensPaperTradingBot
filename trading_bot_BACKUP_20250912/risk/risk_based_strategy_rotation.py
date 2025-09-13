#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk-Based Strategy Rotation

This module implements automatic strategy rotation based on detected risk factors.
It analyzes risk insights from the risk management engine and automatically
adjusts strategy allocations to optimize for the current risk environment.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set

from trading_bot.core.event_bus import Event, EventBus, get_global_event_bus
from trading_bot.core.constants import EventType
from trading_bot.core.strategy_manager import StrategyPerformanceManager

logger = logging.getLogger(__name__)

class RiskBasedStrategyRotation:
    """
    Risk-based strategy rotation system that automatically adjusts strategy
    allocations based on detected risk factors.
    """
    
    def __init__(
            self, 
            strategy_manager: StrategyPerformanceManager, 
            event_bus: Optional[EventBus] = None,
            config: Optional[Dict[str, Any]] = None
        ):
        """
        Initialize the risk-based strategy rotation system.
        
        Args:
            strategy_manager: The strategy manager to interact with
            event_bus: The event bus to subscribe to (defaults to global)
            config: Configuration parameters
        """
        self.strategy_manager = strategy_manager
        self.event_bus = event_bus or get_global_event_bus()
        self.config = config or {}
        
        # Default configuration with sensible defaults
        self.max_strategies = self.config.get('max_active_strategies', 5)
        self.risk_factor_weights = self.config.get('risk_factor_weights', {
            'market_beta': 0.2,
            'sector_exposure': 0.2,
            'volatility': 0.3,
            'correlation': 0.2,
            'liquidity': 0.1,
            # Add more factors as needed
        })
        
        # State variables
        self.market_regime = "normal"
        self.current_risk_factors = {}
        self.strategy_risk_profiles = {}  # Strategy ID -> risk profile
        self.pending_rotations = False
        
        # Register event handlers
        self.register_event_handlers()
        
        logger.info("Risk-based strategy rotation system initialized")
    
    def register_event_handlers(self):
        """Register for relevant events from the event bus."""
        # Listen for risk events
        self.event_bus.subscribe(
            EventType.RISK_ATTRIBUTION_CALCULATED, 
            self._on_risk_attribution
        )
        self.event_bus.subscribe(
            EventType.CORRELATION_RISK_ALERT, 
            self._on_correlation_alert
        )
        self.event_bus.subscribe(
            EventType.DRAWDOWN_THRESHOLD_EXCEEDED, 
            self._on_drawdown_alert
        )
        self.event_bus.subscribe(
            EventType.MARKET_REGIME_CHANGED, 
            self._on_market_regime_change
        )
        
        # Subscribe to portfolio events
        self.event_bus.subscribe(
            EventType.PORTFOLIO_EXPOSURE_UPDATED, 
            self._on_portfolio_exposure_update
        )
        
        logger.info("Risk-based strategy rotation event handlers registered")
    
    def _on_risk_attribution(self, event: Event):
        """
        Handle risk attribution events.
        
        Args:
            event: The risk attribution event
        """
        if not event.data or 'risk_factors' not in event.data:
            return
            
        # Update current risk factors
        self.current_risk_factors = event.data['risk_factors']
        
        # Trigger strategy evaluation based on new risk insights
        self.evaluate_strategy_risk_compatibility()
    
    def _on_correlation_alert(self, event: Event):
        """
        Handle correlation risk alert events.
        
        Args:
            event: The correlation alert event
        """
        if not event.data:
            return
            
        # Update correlation risk factor
        symbols = event.data.get('symbols', [])
        correlation = event.data.get('correlation', 0)
        
        # If high correlation is detected, trigger a strategy rotation
        if correlation > 0.7 and len(symbols) >= 2:
            self.evaluate_strategy_risk_compatibility(
                focused_factors=['correlation']
            )
    
    def _on_drawdown_alert(self, event: Event):
        """
        Handle drawdown threshold exceeded events.
        
        Args:
            event: The drawdown alert event
        """
        if not event.data:
            return
            
        # Check drawdown severity
        severity = event.data.get('severity', 0)
        
        # For higher severity drawdowns, trigger defensive strategy rotation
        if severity >= 2:
            self.execute_defensive_rotation()
    
    def _on_market_regime_change(self, event: Event):
        """
        Handle market regime change events.
        
        Args:
            event: The market regime change event
        """
        if not event.data or 'current_regime' not in event.data:
            return
            
        # Update market regime
        self.market_regime = event.data['current_regime']
        
        # Significant regime changes should trigger a strategy rotation
        logger.info(f"Market regime changed to {self.market_regime}, evaluating strategies")
        self.evaluate_strategy_risk_compatibility()
    
    def _on_portfolio_exposure_update(self, event: Event):
        """
        Handle portfolio exposure update events.
        
        Args:
            event: The portfolio exposure update event
        """
        if not event.data:
            return
            
        # Check if action is required based on risk exposure
        total_risk = event.data.get('total_risk', 0)
        max_risk = event.data.get('max_risk', 0.05)
        
        # If portfolio risk exceeds 80% of max risk, consider rotation
        if total_risk > (max_risk * 0.8):
            self.evaluate_strategy_risk_compatibility(
                focused_factors=['volatility', 'market_beta']
            )
    
    def evaluate_strategy_risk_compatibility(self, focused_factors: List[str] = None):
        """
        Evaluate all strategies for compatibility with current risk factors.
        
        Args:
            focused_factors: Optional list of specific risk factors to focus on
        """
        logger.info("Evaluating strategies for risk compatibility")
        
        # Get all strategies
        all_strategies = self.strategy_manager.get_all_strategies()
        if not all_strategies:
            logger.warning("No strategies available for evaluation")
            return
        
        # Calculate risk compatibility scores for each strategy
        strategy_scores = []
        
        for strategy_id, strategy in all_strategies.items():
            # Get strategy risk profile - this should be calculated or stored somewhere
            risk_profile = self._get_strategy_risk_profile(strategy_id)
            
            # Calculate compatibility score
            score = self._calculate_risk_compatibility_score(
                risk_profile, 
                focused_factors=focused_factors
            )
            
            strategy_scores.append((strategy_id, score))
        
        # Sort strategies by score (highest is best)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine if rotation is needed
        current_active = set(self.strategy_manager.get_active_strategies().keys())
        top_strategies = set([s[0] for s in strategy_scores[:self.max_strategies]])
        
        if current_active != top_strategies:
            self._execute_strategy_rotation(strategy_scores, current_active)
            
            # Publish rotation event
            self.event_bus.create_and_publish(
                event_type=EventType.STRATEGY_ALLOCATION_CHANGED,
                data={
                    'reason': 'risk_based_rotation',
                    'risk_factors': self.current_risk_factors,
                    'market_regime': self.market_regime,
                    'timestamp': datetime.now().isoformat(),
                    'strategy_changes': {
                        'promoted': list(top_strategies - current_active),
                        'demoted': list(current_active - top_strategies)
                    }
                },
                source="risk_based_rotation"
            )
    
    def _get_strategy_risk_profile(self, strategy_id: str) -> Dict[str, float]:
        """
        Get or calculate the risk profile for a strategy.
        
        Args:
            strategy_id: The strategy ID
        
        Returns:
            The risk profile as a dictionary of risk factors
        """
        # If we have a cached profile, use it
        if strategy_id in self.strategy_risk_profiles:
            return self.strategy_risk_profiles[strategy_id]
        
        # Otherwise, calculate it
        # This would typically be based on historical performance and strategy metadata
        
        strategy = self.strategy_manager.get_strategy(strategy_id)
        if not strategy:
            return {}
        
        # Get strategy metadata for risk profile calculation
        metadata = getattr(strategy, 'metadata', {})
        
        # Build a risk profile - this is simplified and should be expanded
        risk_profile = {
            'market_beta': metadata.get('market_beta', 0.5),
            'volatility': metadata.get('volatility', 0.5),
            'correlation': metadata.get('correlation_risk', 0.5),
            'sector_exposure': metadata.get('sector_bias', 0.5),
            'liquidity': metadata.get('liquidity_risk', 0.5),
            # These could also come from historical analysis
        }
        
        # Map strategy type to typical risk factors
        strategy_type = metadata.get('strategy_type', '').lower()
        
        if 'momentum' in strategy_type:
            risk_profile.update({
                'market_beta': 0.8,
                'volatility': 0.7
            })
        elif 'value' in strategy_type:
            risk_profile.update({
                'market_beta': 0.5,
                'volatility': 0.4
            })
        elif 'mean_reversion' in strategy_type:
            risk_profile.update({
                'market_beta': 0.3,
                'volatility': 0.6
            })
        
        # Cache the profile
        self.strategy_risk_profiles[strategy_id] = risk_profile
        
        return risk_profile
    
    def _calculate_risk_compatibility_score(
            self, 
            risk_profile: Dict[str, float], 
            focused_factors: List[str] = None
        ) -> float:
        """
        Calculate how compatible a strategy is with current risk factors.
        
        Args:
            risk_profile: The strategy's risk profile
            focused_factors: Optional list of specific risk factors to focus on
        
        Returns:
            Compatibility score (0-100, higher is better)
        """
        if not risk_profile or not self.current_risk_factors:
            return 50.0  # Default middle score
        
        # If we're focusing on specific factors, adjust weights
        weights = self.risk_factor_weights.copy()
        
        if focused_factors:
            # Increase weight of focused factors
            remaining_weight = 0.4  # 40% for non-focused factors
            focused_weight = (1.0 - remaining_weight) / len(focused_factors)
            
            for factor in weights:
                if factor in focused_factors:
                    weights[factor] = focused_weight
                else:
                    weights[factor] = remaining_weight / (len(weights) - len(focused_factors))
        
        # Calculate score based on market regime and risk factors
        score = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            # Skip if factor not in profile
            if factor not in risk_profile:
                continue
                
            # Get current risk level for this factor
            current_risk = self.current_risk_factors.get(factor, 0.5)
            profile_risk = risk_profile.get(factor, 0.5)
            
            # Calculate factor compatibility based on market regime
            if self.market_regime == "volatile":
                # In volatile regimes, prefer lower risk strategies
                factor_score = 100 * (1.0 - profile_risk)
            elif self.market_regime == "trending":
                # In trending regimes, allow more risk
                factor_score = 100 * (0.5 + (profile_risk * 0.5))
            elif self.market_regime == "mean_reverting":
                # In mean-reverting regimes, prefer mean-reversion strategies
                # This is simplified - should be based on actual strategy characteristics
                factor_score = 100 * (1.0 - abs(profile_risk - 0.5))
            else:  # normal regime
                # In normal regimes, balanced approach
                factor_score = 100 * (1.0 - abs(profile_risk - current_risk))
            
            # Add to weighted score
            score += factor_score * weight
            total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 50.0
            
        return final_score
    
    def _execute_strategy_rotation(
            self, 
            strategy_scores: List[Tuple[str, float]], 
            current_active: Set[str]
        ):
        """
        Execute the strategy rotation based on scores.
        
        Args:
            strategy_scores: List of (strategy_id, score) tuples
            current_active: Set of currently active strategy IDs
        """
        # Get top strategies by score
        top_strategies = [s[0] for s in strategy_scores[:self.max_strategies]]
        
        # Determine which strategies to promote/demote
        to_promote = [s for s in top_strategies if s not in current_active]
        to_demote = [s for s in current_active if s not in top_strategies]
        
        # Log the rotation plan
        logger.info(f"Risk-based rotation plan:")
        logger.info(f"  Promoting: {to_promote}")
        logger.info(f"  Demoting: {to_demote}")
        
        # Execute the rotation
        for strategy_id in to_promote:
            logger.info(f"Promoting strategy {strategy_id} based on risk compatibility")
            self.strategy_manager.promote_strategy(
                strategy_id, 
                reason="risk_based_rotation"
            )
        
        for strategy_id in to_demote:
            logger.info(f"Demoting strategy {strategy_id} based on risk compatibility")
            self.strategy_manager.demote_strategy(
                strategy_id, 
                reason="risk_based_rotation"
            )
    
    def execute_defensive_rotation(self):
        """
        Execute a defensive strategy rotation during significant drawdowns.
        This is a more aggressive risk reduction approach.
        """
        logger.warning("Executing defensive strategy rotation due to significant drawdown")
        
        # Get all strategies
        all_strategies = self.strategy_manager.get_all_strategies()
        if not all_strategies:
            logger.warning("No strategies available for defensive rotation")
            return
        
        # Calculate defensive scores - prioritize low volatility and low beta
        strategy_scores = []
        
        for strategy_id, strategy in all_strategies.items():
            risk_profile = self._get_strategy_risk_profile(strategy_id)
            
            # For defensive rotation, we prioritize low volatility and low beta
            volatility = risk_profile.get('volatility', 0.5)
            market_beta = risk_profile.get('market_beta', 0.5)
            
            # Lower is better for defensive rotation
            defensive_score = 100 * (1.0 - ((volatility * 0.6) + (market_beta * 0.4)))
            
            strategy_scores.append((strategy_id, defensive_score))
        
        # Sort strategies by score (highest is best)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine if rotation is needed
        current_active = set(self.strategy_manager.get_active_strategies().keys())
        
        # In defensive mode, we might reduce the number of active strategies
        defensive_max = max(1, self.max_strategies // 2)  # At least 1 strategy
        top_strategies = set([s[0] for s in strategy_scores[:defensive_max]])
        
        if current_active != top_strategies:
            self._execute_strategy_rotation(strategy_scores, current_active)
            
            # Publish defensive rotation event
            self.event_bus.create_and_publish(
                event_type=EventType.STRATEGY_ALLOCATION_CHANGED,
                data={
                    'reason': 'defensive_rotation',
                    'risk_factors': self.current_risk_factors,
                    'market_regime': self.market_regime,
                    'timestamp': datetime.now().isoformat(),
                    'strategy_changes': {
                        'promoted': list(top_strategies - current_active),
                        'demoted': list(current_active - top_strategies),
                        'defensive_mode': True
                    }
                },
                source="risk_based_rotation"
            )
            
            logger.warning(f"Defensive rotation completed, {len(top_strategies)} strategies now active")
    
    def update_strategy_risk_profile(self, strategy_id: str, risk_profile: Dict[str, float]):
        """
        Update the risk profile for a specific strategy.
        
        Args:
            strategy_id: The strategy ID
            risk_profile: The updated risk profile
        """
        self.strategy_risk_profiles[strategy_id] = risk_profile
