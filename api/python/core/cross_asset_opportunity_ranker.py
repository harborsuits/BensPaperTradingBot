#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Asset Opportunity Ranker

This module provides a comprehensive framework for ranking trading opportunities
across different asset classes, ensuring that capital is always allocated to
the highest quality opportunities regardless of asset class origin.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

class OpportunityScore:
    """Container for opportunity scoring information"""
    def __init__(self, 
                 signal: Signal, 
                 score: float,
                 expected_return: float,
                 risk_adjusted_return: float,
                 time_horizon: str,
                 confidence: float):
        self.signal = signal
        self.score = score  # Overall score (0-100)
        self.expected_return = expected_return  # Expected return (%)
        self.risk_adjusted_return = risk_adjusted_return  # Expected return / risk
        self.time_horizon = time_horizon  # Expected trade duration
        self.confidence = confidence  # Confidence in prediction (0-1)
        self.creation_time = datetime.now()
    
    def __repr__(self):
        return (f"OpportunityScore({self.signal.symbol}, {self.signal.signal_type}, "
                f"score={self.score:.2f}, RAR={self.risk_adjusted_return:.2f})")


class CrossAssetOpportunityRanker:
    """
    Cross-Asset Opportunity Ranker
    
    This class analyzes and ranks trading opportunities across different asset classes:
    - Normalizes returns and risk metrics for fair comparison
    - Considers correlation and portfolio impact
    - Adjusts for market regime alignment
    - Creates a unified ranking regardless of asset origin
    - Dynamically adapts to changing market conditions
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Scoring weights
        'quality_score_weight': 0.25,      # Signal quality importance
        'rar_weight': 0.30,                # Risk-adjusted return importance
        'regime_alignment_weight': 0.20,   # Market regime alignment importance
        'liquidity_weight': 0.15,          # Liquidity importance
        'portfolio_impact_weight': 0.10,   # Portfolio diversification importance
        
        # Asset class baseline adjustments
        'forex_baseline_volatility': 0.8,  # Forex volatility vs standard
        'stock_baseline_volatility': 1.0,  # Stock volatility (standard)
        'options_baseline_volatility': 1.5, # Options volatility vs standard
        'crypto_baseline_volatility': 1.7,  # Crypto volatility vs standard
        
        # Opportunity thresholds
        'min_opportunity_score': 65.0,     # Minimum score to consider (0-100)
        'exceptional_opportunity_score': 85.0,  # Score considered exceptional
        
        # Time decay settings for opportunity scores
        'score_half_life_minutes': {
            'ultra_short': 15,    # Ultra short-term opportunities
            'short': 60,          # Short-term opportunities
            'medium': 240,        # Medium-term opportunities
            'long': 1440,         # Long-term opportunities
        },
        
        # Maximum opportunities to track per asset class
        'max_opportunities_per_asset': 10
    }
    
    def __init__(self, event_bus: EventBus, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Cross-Asset Opportunity Ranker.
        
        Args:
            event_bus: System event bus
            parameters: Custom parameters to override defaults
        """
        self.event_bus = event_bus
        
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Initialize opportunity tracking
        self.opportunities = {
            'forex': [],
            'stock': [],
            'options': [],
            'crypto': [],
            'all': []  # Combined and ranked list across all assets
        }
        
        # Market context tracking
        self.market_context = {
            'regime': 'unknown',
            'vix': 15.0,
            'asset_volatilities': {
                'forex': self.parameters['forex_baseline_volatility'],
                'stock': self.parameters['stock_baseline_volatility'],
                'options': self.parameters['options_baseline_volatility'],
                'crypto': self.parameters['crypto_baseline_volatility']
            }
        }
        
        # Portfolio state tracking
        self.portfolio_state = {
            'asset_allocations': {
                'forex': 0.0,
                'stock': 0.0,
                'options': 0.0,
                'crypto': 0.0
            },
            'positions': [],
            'correlations': {}
        }
        
        # Register for events
        self._register_events()
        
        logger.info("Cross-Asset Opportunity Ranker initialized")
    
    def _register_events(self):
        """Register for events of interest."""
        self.event_bus.subscribe(EventType.SIGNAL_ENHANCED, self._on_signal_enhanced)
        self.event_bus.subscribe(EventType.CONTEXT_ANALYSIS_COMPLETED, self._on_context_analysis)
        self.event_bus.subscribe(EventType.PORTFOLIO_UPDATED, self._on_portfolio_updated)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
    
    def _on_signal_enhanced(self, event: Event):
        """
        Handle enhanced signal events.
        
        Args:
            event: Enhanced signal event
        """
        # Skip invalid signals
        if not event.data.get('valid', False):
            return
            
        # Extract enhanced signal
        signal = event.data.get('enhanced_signal')
        if not signal:
            return
            
        # Calculate opportunity score
        opportunity = self._calculate_opportunity_score(signal)
        
        # Check if score meets minimum threshold
        if opportunity.score < self.parameters['min_opportunity_score']:
            logger.debug(f"Signal for {signal.symbol} scored below threshold: {opportunity.score:.2f}")
            return
            
        # Add to appropriate asset class list
        asset_class = signal.asset_class.lower()
        if asset_class not in self.opportunities:
            asset_class = 'stock'  # Default to stock if unknown
            
        # Add opportunity to tracking
        self._add_opportunity(opportunity, asset_class)
        
        # Re-rank all opportunities
        self._rank_all_opportunities()
        
        # Publish ranked opportunities event
        self._publish_opportunity_rankings()
    
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
                
            # Update asset class volatilities if available
            if 'asset_volatilities' in context:
                for asset, volatility in context['asset_volatilities'].items():
                    if asset in self.market_context['asset_volatilities']:
                        self.market_context['asset_volatilities'][asset] = volatility
        
        # Re-rank opportunities with new context
        self._rank_all_opportunities()
        
        # Publish new rankings
        self._publish_opportunity_rankings()
    
    def _on_portfolio_updated(self, event: Event):
        """
        Handle portfolio updated events.
        
        Args:
            event: Portfolio updated event
        """
        # Update portfolio state
        if 'portfolio' in event.data:
            portfolio = event.data['portfolio']
            
            # Update asset allocations
            if 'allocations' in portfolio:
                self.portfolio_state['asset_allocations'] = portfolio['allocations']
                
            # Update positions
            if 'positions' in portfolio:
                self.portfolio_state['positions'] = portfolio['positions']
                
            # Update correlations
            if 'correlations' in portfolio:
                self.portfolio_state['correlations'] = portfolio['correlations']
        
        # Re-rank with new portfolio context
        self._rank_all_opportunities()
    
    def _on_trade_executed(self, event: Event):
        """
        Handle trade executed events.
        
        Args:
            event: Trade executed event
        """
        # Extract trade details
        if 'trade' in event.data:
            trade = event.data['trade']
            symbol = trade.get('symbol')
            
            # Remove the opportunity for this symbol if it exists
            self._remove_opportunity_by_symbol(symbol)
            
            # Re-rank remaining opportunities
            self._rank_all_opportunities()
            
            # Publish updated rankings
            self._publish_opportunity_rankings()
    
    def _calculate_opportunity_score(self, signal: Signal) -> OpportunityScore:
        """
        Calculate opportunity score for a signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Opportunity score
        """
        # Extract signal metadata
        metadata = signal.metadata or {}
        asset_class = signal.asset_class.lower()
        
        # Default values if metadata is missing
        quality_score = metadata.get('quality_score', 0.5)
        
        # Extract risk/reward values
        if hasattr(signal, 'risk_reward_ratio') and signal.risk_reward_ratio:
            risk_reward = signal.risk_reward_ratio
        else:
            # Default risk/reward based on signal type and asset class
            risk_reward = self._estimate_risk_reward(signal)
        
        # Extract expected return
        if hasattr(signal, 'expected_return') and signal.expected_return:
            expected_return = signal.expected_return
        else:
            # Estimate expected return based on signal type and asset class
            expected_return = self._estimate_expected_return(signal)
        
        # Extract confidence
        if hasattr(signal, 'confidence') and signal.confidence:
            confidence = signal.confidence
        else:
            # Default confidence based on quality score
            confidence = quality_score * 0.8
        
        # Calculate risk-adjusted return
        if hasattr(signal, 'stop_loss') and signal.stop_loss and hasattr(signal, 'take_profit') and signal.take_profit:
            # Calculate from actual stop/target levels
            if signal.signal_type in [SignalType.BUY, SignalType.BREAKOUT_LONG]:
                risk = abs(signal.price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.price)
            else:
                risk = abs(signal.stop_loss - signal.price)
                reward = abs(signal.price - signal.take_profit)
            
            # Calculate risk-adjusted return
            risk_adjusted_return = reward / max(risk, 0.0001)
        else:
            # Use the risk_reward value
            risk_adjusted_return = risk_reward
        
        # Determine time horizon
        time_horizon = self._determine_time_horizon(signal)
        
        # Calculate base score components
        quality_component = quality_score * 100 * self.parameters['quality_score_weight']
        rar_component = min(risk_adjusted_return * 25, 100) * self.parameters['rar_weight']
        
        # Calculate regime alignment
        regime_alignment = self._calculate_regime_alignment(signal)
        regime_component = regime_alignment * 100 * self.parameters['regime_alignment_weight']
        
        # Calculate liquidity score
        liquidity_score = metadata.get('liquidity', {}).get('liquidity_score', 0.5)
        liquidity_component = liquidity_score * 100 * self.parameters['liquidity_weight']
        
        # Calculate portfolio impact
        portfolio_impact = self._calculate_portfolio_impact(signal)
        portfolio_component = portfolio_impact * 100 * self.parameters['portfolio_impact_weight']
        
        # Sum all components for final score
        final_score = (quality_component + rar_component + regime_component + 
                      liquidity_component + portfolio_component)
        
        # Create and return opportunity score
        return OpportunityScore(
            signal=signal,
            score=final_score,
            expected_return=expected_return,
            risk_adjusted_return=risk_adjusted_return,
            time_horizon=time_horizon,
            confidence=confidence
        )
    
    def _estimate_risk_reward(self, signal: Signal) -> float:
        """
        Estimate risk/reward ratio for a signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Estimated risk/reward ratio
        """
        # Default values by signal type
        if signal.signal_type in [SignalType.BREAKOUT_LONG, SignalType.BREAKOUT_SHORT]:
            base_rr = 2.0  # Higher for breakouts
        else:
            base_rr = 1.5  # Standard for other signals
            
        # Adjust by asset class
        asset_class = signal.asset_class.lower()
        if asset_class == 'options':
            # Options often have defined risk/reward by structure
            if hasattr(signal, 'strategy_subtype') and signal.strategy_subtype:
                if 'butterfly' in signal.strategy_subtype.lower():
                    return 3.0
                elif 'iron' in signal.strategy_subtype.lower():
                    return 2.0
                else:
                    return 1.5
        elif asset_class == 'forex':
            # Forex usually has tighter ranges
            return base_rr * 0.8
        elif asset_class == 'crypto':
            # Crypto can have larger moves
            return base_rr * 1.2
            
        return base_rr
    
    def _estimate_expected_return(self, signal: Signal) -> float:
        """
        Estimate expected percentage return for a signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Estimated expected return percentage
        """
        # Base expected returns by asset class (daily %)
        base_returns = {
            'forex': 0.5,    # 0.5% typical forex return
            'stock': 1.0,    # 1% typical stock return
            'options': 5.0,  # 5% typical options return
            'crypto': 3.0    # 3% typical crypto return
        }
        
        asset_class = signal.asset_class.lower()
        if asset_class not in base_returns:
            asset_class = 'stock'  # Default
            
        # Get base return for this asset class
        base_return = base_returns[asset_class]
        
        # Adjust by signal type
        if signal.signal_type in [SignalType.BREAKOUT_LONG, SignalType.BREAKOUT_SHORT]:
            # Breakouts typically have higher potential
            return base_return * 1.5
        elif signal.signal_type in [SignalType.REVERSAL_LONG, SignalType.REVERSAL_SHORT]:
            # Reversals can be powerful
            return base_return * 1.3
            
        # Adjust by time horizon
        time_horizon = self._determine_time_horizon(signal)
        if time_horizon == 'ultra_short':
            return base_return * 0.5  # Lower return for very short term
        elif time_horizon == 'short':
            return base_return  # Standard return
        elif time_horizon == 'medium':
            return base_return * 2  # Higher for medium term
        elif time_horizon == 'long':
            return base_return * 3  # Higher for long term
            
        return base_return
    
    def _determine_time_horizon(self, signal: Signal) -> str:
        """
        Determine time horizon for a signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Time horizon category (ultra_short, short, medium, long)
        """
        # Extract from signal if available
        if hasattr(signal, 'time_horizon') and signal.time_horizon:
            return signal.time_horizon
            
        # Determine based on signal type and metadata
        metadata = signal.metadata or {}
        
        # Check strategy type
        if hasattr(signal, 'strategy_name'):
            strategy_name = signal.strategy_name.lower()
            if 'scalp' in strategy_name:
                return 'ultra_short'
            elif 'swing' in strategy_name:
                return 'medium'
            elif 'position' in strategy_name or 'trend' in strategy_name:
                return 'long'
        
        # Default by asset class
        asset_class = signal.asset_class.lower()
        if asset_class == 'forex':
            return 'short'  # Default to short for forex
        elif asset_class == 'options':
            # Most option strategies are shorter term
            return 'medium'
        elif asset_class == 'crypto':
            return 'short'  # Default to short for crypto
            
        return 'medium'  # Default
    
    def _calculate_regime_alignment(self, signal: Signal) -> float:
        """
        Calculate how well a signal aligns with current market regime.
        
        Args:
            signal: Trading signal
            
        Returns:
            Alignment score (0-1)
        """
        # Get current market regime
        market_regime = self.market_context['regime']
        
        # Default alignment is moderate
        alignment = 0.5
        
        # Check signal type vs regime
        if signal.signal_type in [SignalType.BUY, SignalType.BREAKOUT_LONG]:
            # Long signals
            if market_regime == 'bullish':
                alignment = 0.9  # Strongly aligned
            elif market_regime == 'bearish':
                alignment = 0.2  # Poorly aligned
            elif market_regime == 'volatile':
                alignment = 0.5  # Moderately aligned
            elif market_regime == 'neutral':
                alignment = 0.6  # Somewhat aligned
        elif signal.signal_type in [SignalType.SELL, SignalType.BREAKOUT_SHORT]:
            # Short signals
            if market_regime == 'bearish':
                alignment = 0.9  # Strongly aligned
            elif market_regime == 'bullish':
                alignment = 0.2  # Poorly aligned
            elif market_regime == 'volatile':
                alignment = 0.5  # Moderately aligned
            elif market_regime == 'neutral':
                alignment = 0.6  # Somewhat aligned
        
        # For volatility-based signals
        if signal.signal_type in [SignalType.BREAKOUT_LONG, SignalType.BREAKOUT_SHORT]:
            if market_regime == 'volatile':
                alignment = 0.9  # Breakouts work well in volatile regimes
        
        # For mean-reversion signals
        if hasattr(signal, 'strategy_subtype') and 'reversion' in signal.strategy_subtype.lower():
            if market_regime == 'neutral':
                alignment = 0.9  # Mean reversion works well in neutral markets
                
        return alignment
    
    def _calculate_portfolio_impact(self, signal: Signal) -> float:
        """
        Calculate how positively a signal impacts portfolio diversification.
        
        Args:
            signal: Trading signal
            
        Returns:
            Portfolio impact score (0-1)
        """
        # Default impact is neutral
        impact = 0.5
        
        # If no positions, any new position improves diversification
        if not self.portfolio_state['positions']:
            return 0.8  # Good impact
            
        # Check for concentration in this asset class
        asset_class = signal.asset_class.lower()
        current_allocation = self.portfolio_state['asset_allocations'].get(asset_class, 0)
        
        # Lower score for already concentrated asset classes
        if current_allocation > 0.4:  # If >40% allocation
            impact -= 0.3  # Significant reduction
        elif current_allocation > 0.2:  # If >20% allocation
            impact -= 0.15  # Moderate reduction
            
        # Check correlation with existing positions
        symbol = signal.symbol
        symbol_correlations = self.portfolio_state['correlations'].get(symbol, {})
        
        # If we have correlation data
        if symbol_correlations:
            # Calculate average correlation with existing positions
            correlations = []
            for position in self.portfolio_state['positions']:
                position_symbol = position.get('symbol')
                if position_symbol in symbol_correlations:
                    correlations.append(abs(symbol_correlations[position_symbol]))
            
            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                
                # Adjust impact based on correlation
                # Lower correlation is better for diversification
                if avg_correlation > 0.7:
                    impact -= 0.3  # Highly correlated - bad
                elif avg_correlation < 0.3:
                    impact += 0.3  # Low correlation - good
                    
        return max(0.1, min(impact, 1.0))  # Ensure in 0.1-1.0 range
    
    def _add_opportunity(self, opportunity: OpportunityScore, asset_class: str):
        """
        Add an opportunity to tracking.
        
        Args:
            opportunity: Opportunity score
            asset_class: Asset class
        """
        # Ensure asset class exists
        if asset_class not in self.opportunities:
            self.opportunities[asset_class] = []
            
        # Check if we already have an opportunity for this symbol
        symbol = opportunity.signal.symbol
        existing_indices = [i for i, op in enumerate(self.opportunities[asset_class]) 
                            if op.signal.symbol == symbol]
        
        if existing_indices:
            # Replace existing opportunity
            self.opportunities[asset_class][existing_indices[0]] = opportunity
        else:
            # Add new opportunity
            self.opportunities[asset_class].append(opportunity)
            
        # Enforce maximum opportunities per asset class
        max_opportunities = self.parameters['max_opportunities_per_asset']
        if len(self.opportunities[asset_class]) > max_opportunities:
            # Sort by score and keep top opportunities
            self.opportunities[asset_class].sort(key=lambda x: x.score, reverse=True)
            self.opportunities[asset_class] = self.opportunities[asset_class][:max_opportunities]
    
    def _remove_opportunity_by_symbol(self, symbol: str):
        """
        Remove an opportunity by symbol.
        
        Args:
            symbol: Symbol to remove
        """
        # Remove from asset-specific lists
        for asset_class in self.opportunities:
            if asset_class == 'all':
                continue  # Skip combined list
                
            self.opportunities[asset_class] = [
                op for op in self.opportunities[asset_class]
                if op.signal.symbol != symbol
            ]
            
        # Remove from combined list
        self.opportunities['all'] = [
            op for op in self.opportunities['all']
            if op.signal.symbol != symbol
        ]
    
    def _apply_time_decay(self):
        """Apply time decay to opportunity scores based on age."""
        current_time = datetime.now()
        
        # Process each asset class
        for asset_class in self.opportunities:
            if asset_class == 'all':
                continue  # Skip combined list
                
            # Update each opportunity
            for opportunity in self.opportunities[asset_class]:
                # Calculate age in minutes
                age_minutes = (current_time - opportunity.creation_time).total_seconds() / 60
                
                # Get half-life for this time horizon
                half_life = self.parameters['score_half_life_minutes'].get(
                    opportunity.time_horizon, 60)  # Default 60 minutes
                
                # Calculate decay factor
                decay_factor = 2 ** (-age_minutes / half_life)
                
                # Apply decay to score
                original_score = opportunity.score
                opportunity.score = original_score * decay_factor
    
    def _rank_all_opportunities(self):
        """Rank all opportunities across asset classes."""
        # Apply time decay to account for age
        self._apply_time_decay()
        
        # Create combined list
        all_opportunities = []
        for asset_class in self.opportunities:
            if asset_class == 'all':
                continue  # Skip the combined list itself
                
            all_opportunities.extend(self.opportunities[asset_class])
            
        # Apply normalization
        self._normalize_opportunity_scores(all_opportunities)
        
        # Sort by score
        all_opportunities.sort(key=lambda x: x.score, reverse=True)
        
        # Update combined list
        self.opportunities['all'] = all_opportunities

    def _normalize_opportunity_scores(self, opportunities: List[OpportunityScore]):
        """
        Normalize scores to account for different asset class characteristics.
        
        Args:
            opportunities: List of opportunities to normalize
        """
        if not opportunities:
            return
            
        # Group by asset class
        by_asset = {}
        for op in opportunities:
            asset_class = op.signal.asset_class.lower()
            if asset_class not in by_asset:
                by_asset[asset_class] = []
            by_asset[asset_class].append(op)
            
        # Calculate adjustments based on market context
        adjustments = {}
        for asset_class, volatility in self.market_context['asset_volatilities'].items():
            # Calculate normalization factor
            # Higher volatility assets need more reward to compensate for risk
            base_vol = self.parameters['stock_baseline_volatility']  # Reference level
            if volatility > base_vol:
                # Penalize higher volatility somewhat
                adjustments[asset_class] = base_vol / volatility
            else:
                # Boost lower volatility somewhat
                adjustments[asset_class] = 1.0 + (base_vol - volatility) * 0.1
                
        # Apply adjustments
        for asset_class, ops in by_asset.items():
            if asset_class in adjustments:
                factor = adjustments[asset_class]
                for op in ops:
                    op.score = op.score * factor

    def _publish_opportunity_rankings(self):
        """Publish opportunity rankings event."""
        # Create event with top opportunities
        top_overall = self.opportunities['all'][:10] if self.opportunities['all'] else []
        
        # Create summary data
        summary = {
            'top_opportunities': [
                {
                    'symbol': op.signal.symbol,
                    'asset_class': op.signal.asset_class,
                    'signal_type': str(op.signal.signal_type),
                    'score': op.score,
                    'risk_adjusted_return': op.risk_adjusted_return,
                    'expected_return': op.expected_return,
                    'time_horizon': op.time_horizon
                }
                for op in top_overall
            ],
            'opportunity_counts': {
                asset: len(ops) for asset, ops in self.opportunities.items()
                if asset != 'all'
            },
            'best_asset_class': self._determine_best_asset_class(),
            'exceptional_opportunities': [
                {
                    'symbol': op.signal.symbol,
                    'asset_class': op.signal.asset_class,
                    'score': op.score
                }
                for op in top_overall
                if op.score >= self.parameters['exceptional_opportunity_score']
            ]
        }
        
        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.OPPORTUNITIES_RANKED,
            data={
                'opportunities': summary,
                'timestamp': datetime.now()
            }
        ))
        
        # Log top opportunities
        if top_overall:
            logger.info(f"Top opportunity: {top_overall[0]}")
            logger.info(f"Found {len(top_overall)} ranked opportunities across all asset classes")
            
            # Log exceptional opportunities
            exceptional = [op for op in top_overall 
                          if op.score >= self.parameters['exceptional_opportunity_score']]
            if exceptional:
                logger.info(f"Found {len(exceptional)} EXCEPTIONAL opportunities!")
    
    def _determine_best_asset_class(self) -> str:
        """
        Determine which asset class currently has the best opportunities.
        
        Returns:
            Best asset class
        """
        best_asset = 'stock'  # Default
        best_avg_score = 0
        
        # Calculate average score for each asset class
        for asset, ops in self.opportunities.items():
            if asset == 'all' or not ops:
                continue
                
            # Calculate average of top 3 opportunities
            top_ops = sorted(ops, key=lambda x: x.score, reverse=True)[:3]
            if top_ops:
                avg_score = sum(op.score for op in top_ops) / len(top_ops)
                
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_asset = asset
                    
        return best_asset
    
    def get_top_opportunities(self, count: int = 10) -> List[OpportunityScore]:
        """
        Get top opportunities across all asset classes.
        
        Args:
            count: Number of opportunities to return
            
        Returns:
            List of top opportunities
        """
        return self.opportunities['all'][:count]
    
    def get_top_opportunities_by_asset(self, 
                                     asset_class: str, 
                                     count: int = 5) -> List[OpportunityScore]:
        """
        Get top opportunities for a specific asset class.
        
        Args:
            asset_class: Asset class
            count: Number of opportunities to return
            
        Returns:
            List of top opportunities for the asset class
        """
        if asset_class not in self.opportunities:
            return []
            
        return self.opportunities[asset_class][:count]
    
    def get_exceptional_opportunities(self) -> List[OpportunityScore]:
        """
        Get exceptional opportunities across all asset classes.
        
        Returns:
            List of exceptional opportunities
        """
        threshold = self.parameters['exceptional_opportunity_score']
        return [op for op in self.opportunities['all'] 
               if op.score >= threshold]

# Add custom event types
EventType.OPPORTUNITIES_RANKED = "OPPORTUNITIES_RANKED"
EventType.PORTFOLIO_UPDATED = "PORTFOLIO_UPDATED"
