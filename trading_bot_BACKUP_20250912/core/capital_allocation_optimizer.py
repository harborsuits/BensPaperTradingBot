#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capital Allocation Optimizer

This module implements intelligent capital allocation across multiple asset classes,
optimizing position sizes based on opportunity quality rather than fixed percentages.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math

from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies.factory.strategy_template import Signal
from trading_bot.core.cross_asset_opportunity_ranker import OpportunityScore

logger = logging.getLogger(__name__)

class CapitalAllocationOptimizer:
    """
    Capital Allocation Optimizer
    
    This class optimizes capital allocation across different asset classes:
    - Dynamically adjusts position sizes based on opportunity quality
    - Balances risk across the portfolio
    - Ensures diversification while capitalizing on exceptional opportunities
    - Adapts to changing market conditions
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Base allocation limits
        'max_portfolio_risk': 0.03,         # Maximum portfolio risk (3%)
        'max_position_risk': 0.015,         # Maximum single position risk (1.5%)
        'min_position_size_percent': 0.005, # Minimum position size (0.5%)
        
        # Opportunity-based scaling
        'base_opportunity_score': 65.0,     # Base score for standard allocation
        'exceptional_opportunity_score': 85.0,  # Score for maximum allocation
        'opportunity_scaling_factor': 1.5,  # How much to scale allocations by score
        
        # Asset class limits (% of portfolio)
        'max_asset_class_allocation': {
            'forex': 0.40,    # Max 40% in forex
            'stock': 0.60,    # Max 60% in stocks
            'options': 0.25,  # Max 25% in options
            'crypto': 0.30,   # Max 30% in crypto
        },
        
        # Asset class risk adjustments
        'asset_class_risk_factor': {
            'forex': 0.8,     # Forex risk multiplier
            'stock': 1.0,     # Stock risk multiplier (baseline)
            'options': 1.5,   # Options risk multiplier
            'crypto': 1.3,    # Crypto risk multiplier
        },
        
        # Regime-based adjustments
        'regime_allocation_adjustments': {
            'bullish': {
                'stock': 1.2,     # Increase stock allocation in bullish regime
                'options': 1.1,   # Slight increase for options
                'forex': 0.9,     # Slight decrease for forex
                'crypto': 1.1     # Slight increase for crypto
            },
            'bearish': {
                'stock': 0.7,     # Decrease stock allocation in bearish regime
                'options': 0.8,   # Decrease options allocation
                'forex': 1.2,     # Increase forex allocation
                'crypto': 0.8     # Decrease crypto allocation
            },
            'volatile': {
                'stock': 0.8,     # Decrease stock allocation in volatile regime
                'options': 0.7,   # Decrease options allocation
                'forex': 1.1,     # Slight increase for forex
                'crypto': 0.9     # Slight decrease for crypto
            },
            'neutral': {
                'stock': 1.0,     # Standard allocation in neutral regime
                'options': 1.0,   # Standard allocation
                'forex': 1.0,     # Standard allocation
                'crypto': 1.0     # Standard allocation
            }
        }
    }
    
    def __init__(self, event_bus: EventBus, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Capital Allocation Optimizer.
        
        Args:
            event_bus: System event bus
            parameters: Custom parameters to override defaults
        """
        self.event_bus = event_bus
        
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Portfolio state tracking
        self.portfolio_state = {
            'total_equity': 10000.0,  # Default starting equity
            'free_margin': 10000.0,   # Available margin
            'current_allocations': {
                'forex': 0.0,
                'stock': 0.0,
                'options': 0.0,
                'crypto': 0.0
            },
            'positions': [],
            'margin_used': 0.0,
            'total_risk': 0.0
        }
        
        # Market context
        self.market_context = {
            'regime': 'neutral',
            'vix': 15.0
        }
        
        # Register for events
        self._register_events()
        
        logger.info("Capital Allocation Optimizer initialized")
    
    def _register_events(self):
        """Register for events of interest."""
        self.event_bus.subscribe(EventType.OPPORTUNITIES_RANKED, self._on_opportunities_ranked)
        self.event_bus.subscribe(EventType.PORTFOLIO_UPDATED, self._on_portfolio_updated)
        self.event_bus.subscribe(EventType.CONTEXT_ANALYSIS_COMPLETED, self._on_context_analysis)
        self.event_bus.subscribe(EventType.ACCOUNT_BALANCE_UPDATED, self._on_account_balance_updated)
    
    def _on_opportunities_ranked(self, event: Event):
        """
        Handle opportunities ranked events.
        
        Args:
            event: Opportunities ranked event
        """
        # Extract opportunities
        if 'opportunities' not in event.data:
            return
            
        opportunities = event.data['opportunities']
        
        # Get top opportunities
        top_opportunities = opportunities.get('top_opportunities', [])
        if not top_opportunities:
            return
            
        # Calculate optimal allocations
        allocations = self._calculate_optimal_allocations(top_opportunities)
        
        # Publish allocations event
        self.event_bus.publish(Event(
            event_type=EventType.CAPITAL_ALLOCATIONS_UPDATED,
            data={
                'allocations': allocations,
                'timestamp': datetime.now()
            }
        ))
    
    def _on_portfolio_updated(self, event: Event):
        """
        Handle portfolio updated events.
        
        Args:
            event: Portfolio updated event
        """
        # Update portfolio state
        if 'portfolio' in event.data:
            portfolio = event.data['portfolio']
            
            # Update total equity
            if 'equity' in portfolio:
                self.portfolio_state['total_equity'] = portfolio['equity']
                
            # Update free margin
            if 'free_margin' in portfolio:
                self.portfolio_state['free_margin'] = portfolio['free_margin']
                
            # Update current allocations
            if 'allocations' in portfolio:
                self.portfolio_state['current_allocations'] = portfolio['allocations']
                
            # Update positions
            if 'positions' in portfolio:
                self.portfolio_state['positions'] = portfolio['positions']
                
            # Update margin used
            if 'margin_used' in portfolio:
                self.portfolio_state['margin_used'] = portfolio['margin_used']
                
            # Update total risk
            if 'total_risk' in portfolio:
                self.portfolio_state['total_risk'] = portfolio['total_risk']
    
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
                self.market_context['regime'] = context['market_regime']
                
            # Update VIX
            if 'vix' in context:
                self.market_context['vix'] = context['vix']
    
    def _on_account_balance_updated(self, event: Event):
        """
        Handle account balance updated events.
        
        Args:
            event: Account balance updated event
        """
        # Update account balance
        if 'balance' in event.data:
            self.portfolio_state['total_equity'] = event.data['balance']
            
        # Update free margin
        if 'free_margin' in event.data:
            self.portfolio_state['free_margin'] = event.data['free_margin']
    
    def _calculate_optimal_allocations(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate optimal allocations for a set of opportunities.
        
        Args:
            opportunities: List of opportunity data
            
        Returns:
            Dictionary of allocation decisions
        """
        # Convert raw opportunity data to normalized format
        normalized_opportunities = []
        for opp in opportunities:
            normalized_opportunities.append({
                'symbol': opp['symbol'],
                'asset_class': opp['asset_class'].lower(),
                'signal_type': opp['signal_type'],
                'score': opp['score'],
                'risk_adjusted_return': opp.get('risk_adjusted_return', 1.5),
                'expected_return': opp.get('expected_return', 1.0),
                'time_horizon': opp.get('time_horizon', 'medium')
            })
        
        # Total equity available
        total_equity = self.portfolio_state['total_equity']
        free_margin = self.portfolio_state['free_margin']
        available_capital = min(total_equity, free_margin)
        
        # Get current regime
        regime = self.market_context['regime']
        
        # Calculate score-based position sizes
        positions = []
        total_allocated = 0.0
        asset_class_allocations = {
            'forex': 0.0,
            'stock': 0.0,
            'options': 0.0,
            'crypto': 0.0
        }
        
        # Track opportunity scores by asset class
        asset_class_scores = {}
        for opp in normalized_opportunities:
            asset_class = opp['asset_class']
            if asset_class not in asset_class_scores:
                asset_class_scores[asset_class] = []
            asset_class_scores[asset_class].append(opp['score'])
        
        # Calculate average score by asset class
        avg_scores = {}
        for asset_class, scores in asset_class_scores.items():
            if scores:
                avg_scores[asset_class] = sum(scores) / len(scores)
            else:
                avg_scores[asset_class] = 0.0
        
        # Determine base allocation percentages by asset class
        base_allocations = self._calculate_base_allocations(avg_scores, regime)
        
        # Allocate capital to each opportunity
        remaining_opportunities = normalized_opportunities.copy()
        
        # First pass: allocate to exceptional opportunities
        exceptional_opportunities = [
            opp for opp in remaining_opportunities
            if opp['score'] >= self.parameters['exceptional_opportunity_score']
        ]
        
        for opp in exceptional_opportunities:
            # Calculate position size for exceptional opportunity
            position_size = self._calculate_position_size(
                opp, total_equity, available_capital, base_allocations, asset_class_allocations
            )
            
            # Add to positions
            if position_size > 0:
                positions.append({
                    'symbol': opp['symbol'],
                    'asset_class': opp['asset_class'],
                    'signal_type': opp['signal_type'],
                    'allocation': position_size,
                    'allocation_percent': position_size / total_equity,
                    'score': opp['score'],
                    'priority': 'exceptional'
                })
                
                # Update tracking
                total_allocated += position_size
                asset_class_allocations[opp['asset_class']] += position_size
                
                # Remove from remaining opportunities
                remaining_opportunities.remove(opp)
        
        # Update available capital
        available_capital -= total_allocated
        
        # Second pass: allocate to high-quality opportunities
        high_quality_opportunities = [
            opp for opp in remaining_opportunities
            if opp['score'] >= self.parameters['base_opportunity_score']
        ]
        
        for opp in high_quality_opportunities:
            # Calculate position size
            position_size = self._calculate_position_size(
                opp, total_equity, available_capital, base_allocations, asset_class_allocations
            )
            
            # Add to positions
            if position_size > 0:
                positions.append({
                    'symbol': opp['symbol'],
                    'asset_class': opp['asset_class'],
                    'signal_type': opp['signal_type'],
                    'allocation': position_size,
                    'allocation_percent': position_size / total_equity,
                    'score': opp['score'],
                    'priority': 'high'
                })
                
                # Update tracking
                total_allocated += position_size
                asset_class_allocations[opp['asset_class']] += position_size
                
                # Remove from remaining opportunities
                remaining_opportunities.remove(opp)
                
                # Update available capital
                available_capital -= position_size
                
                # Stop if we've allocated all available capital
                if available_capital <= 0:
                    break
        
        # Calculate allocation percentages
        asset_class_percentages = {
            asset: allocation / total_equity
            for asset, allocation in asset_class_allocations.items()
        }
        
        # Sort positions by allocation (largest first)
        positions.sort(key=lambda x: x['allocation'], reverse=True)
        
        # Return allocation decisions
        return {
            'positions': positions,
            'total_allocated': total_allocated,
            'total_allocated_percent': total_allocated / total_equity,
            'asset_class_allocations': asset_class_allocations,
            'asset_class_percentages': asset_class_percentages,
            'remaining_capital': available_capital,
            'timestamp': datetime.now()
        }
    
    def _calculate_base_allocations(self, 
                                  avg_scores: Dict[str, float], 
                                  regime: str) -> Dict[str, float]:
        """
        Calculate base allocation percentages by asset class.
        
        Args:
            avg_scores: Average opportunity scores by asset class
            regime: Current market regime
            
        Returns:
            Dictionary of base allocation percentages by asset class
        """
        # Get regime adjustments
        regime_adjustments = self.parameters['regime_allocation_adjustments'].get(
            regime, self.parameters['regime_allocation_adjustments']['neutral']
        )
        
        # Start with maximum allocations
        base_allocations = self.parameters['max_asset_class_allocation'].copy()
        
        # Apply regime adjustments
        for asset_class, adjustment in regime_adjustments.items():
            if asset_class in base_allocations:
                base_allocations[asset_class] *= adjustment
        
        # Apply score-based scaling
        total_score = sum(avg_scores.values())
        if total_score > 0:
            # Calculate score-weighted allocations
            weighted_allocations = {}
            for asset_class, score in avg_scores.items():
                weight = score / total_score
                weighted_allocations[asset_class] = base_allocations.get(asset_class, 0.0) * weight
                
            # Normalize to ensure we don't exceed 100%
            total_allocation = sum(weighted_allocations.values())
            if total_allocation > 0:
                for asset_class in weighted_allocations:
                    weighted_allocations[asset_class] /= total_allocation
                    
                return weighted_allocations
        
        # If no scores or zero total, return adjusted base allocations
        # Normalize to ensure we don't exceed 100%
        total_allocation = sum(base_allocations.values())
        if total_allocation > 0:
            for asset_class in base_allocations:
                base_allocations[asset_class] /= total_allocation
                
        return base_allocations
    
    def _calculate_position_size(self, 
                               opportunity: Dict[str, Any],
                               total_equity: float,
                               available_capital: float,
                               base_allocations: Dict[str, float],
                               current_allocations: Dict[str, float]) -> float:
        """
        Calculate position size for an opportunity.
        
        Args:
            opportunity: Opportunity data
            total_equity: Total portfolio equity
            available_capital: Available capital
            base_allocations: Base allocation percentages
            current_allocations: Current allocation amounts
            
        Returns:
            Position size in account currency
        """
        asset_class = opportunity['asset_class']
        score = opportunity['score']
        
        # Get base score for comparison
        base_score = self.parameters['base_opportunity_score']
        exceptional_score = self.parameters['exceptional_opportunity_score']
        
        # Calculate score-based scaling factor
        if score >= exceptional_score:
            # Exceptional opportunity gets maximum allocation
            scaling_factor = self.parameters['opportunity_scaling_factor']
        else:
            # Linear scaling between base and exceptional
            scaling_factor = 1.0 + (score - base_score) / (exceptional_score - base_score) * \
                            (self.parameters['opportunity_scaling_factor'] - 1.0)
            
        # Ensure scaling factor is in reasonable range
        scaling_factor = max(0.5, min(scaling_factor, self.parameters['opportunity_scaling_factor']))
        
        # Get maximum allocation for this asset class
        max_asset_allocation = self.parameters['max_asset_class_allocation'].get(asset_class, 0.3) * total_equity
        
        # Check current allocation for this asset class
        current_asset_allocation = current_allocations.get(asset_class, 0.0)
        
        # Calculate remaining available for this asset class
        remaining_asset_allocation = max(0, max_asset_allocation - current_asset_allocation)
        
        # Calculate base position size based on risk
        max_position_risk = self.parameters['max_position_risk']
        
        # Adjust risk based on asset class
        risk_factor = self.parameters['asset_class_risk_factor'].get(asset_class, 1.0)
        adjusted_max_risk = max_position_risk * risk_factor
        
        # Calculate base position size as percentage of portfolio
        base_position_size = adjusted_max_risk * total_equity
        
        # Apply score-based scaling
        scaled_position_size = base_position_size * scaling_factor
        
        # Ensure we don't exceed remaining allocation for this asset class
        position_size = min(scaled_position_size, remaining_asset_allocation)
        
        # Ensure we don't exceed available capital
        position_size = min(position_size, available_capital)
        
        # Ensure position size meets minimum threshold
        min_size = self.parameters['min_position_size_percent'] * total_equity
        if position_size < min_size:
            position_size = 0.0  # Don't allocate if below minimum
            
        return position_size
    
    def calculate_position_size_for_signal(self, 
                                         signal: Signal, 
                                         opportunity_score: float) -> float:
        """
        Calculate optimal position size for a specific signal.
        
        Args:
            signal: Trading signal
            opportunity_score: Opportunity score
            
        Returns:
            Position size in account currency
        """
        # Extract signal information
        asset_class = signal.asset_class.lower()
        symbol = signal.symbol
        
        # Get total equity
        total_equity = self.portfolio_state['total_equity']
        
        # Create opportunity dict
        opportunity = {
            'symbol': symbol,
            'asset_class': asset_class,
            'signal_type': str(signal.signal_type),
            'score': opportunity_score,
            'risk_adjusted_return': getattr(signal, 'risk_reward_ratio', 1.5),
            'expected_return': getattr(signal, 'expected_return', 1.0),
            'time_horizon': getattr(signal, 'time_horizon', 'medium')
        }
        
        # Get current allocations
        current_allocations = {
            asset: allocation
            for asset, allocation in self.portfolio_state['current_allocations'].items()
        }
        
        # Calculate base allocations
        avg_scores = {asset_class: opportunity_score}
        base_allocations = self._calculate_base_allocations(
            avg_scores, self.market_context['regime']
        )
        
        # Calculate position size
        return self._calculate_position_size(
            opportunity=opportunity,
            total_equity=total_equity,
            available_capital=self.portfolio_state['free_margin'],
            base_allocations=base_allocations,
            current_allocations=current_allocations
        )

# Add custom event types
EventType.CAPITAL_ALLOCATIONS_UPDATED = "CAPITAL_ALLOCATIONS_UPDATED"
EventType.ACCOUNT_BALANCE_UPDATED = "ACCOUNT_BALANCE_UPDATED"
