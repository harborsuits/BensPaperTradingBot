#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence-Adjusted Position Sizing for Forex

This module extends the pip-based position sizing system with confidence
adjustments from the indicator-sentiment integrator to dynamically size
positions based on both risk parameters and signal quality.
"""

import logging
import math
from typing import Dict, Optional, Any, Union, Tuple

from trading_bot.strategies.forex.base.pip_based_position_sizing import PipBasedPositionSizing

logger = logging.getLogger(__name__)

class ConfidenceAdjustedPositionSizing(PipBasedPositionSizing):
    """
    Confidence-Adjusted Position Sizing
    
    Extends PipBasedPositionSizing to incorporate confidence metrics from
    indicator-sentiment integration, adjusting position sizes based on
    the quality and agreement of technical and sentiment signals.
    """
    
    # Default confidence parameters - can be overridden via constructor
    DEFAULT_CONFIDENCE_PARAMS = {
        'use_confidence_adjustment': True,     # Whether to use confidence adjustments
        'min_confidence_threshold': 0.3,       # Minimum confidence to trade at all (0-1)
        'high_confidence_threshold': 0.7,      # Threshold for high confidence (0-1)
        'baseline_confidence': 0.5,            # Baseline confidence value
        'low_confidence_factor': 0.5,          # Position size multiplier for low confidence
        'high_confidence_factor': 1.5,         # Position size multiplier for high confidence
        'signal_agreement_bonus': 0.2,         # Bonus factor when indicators & sentiment agree
        'signal_disagreement_penalty': 0.4,    # Penalty factor when indicators & sentiment disagree
        'min_integrated_score': 0.2,           # Minimum integrated score magnitude to consider
        'score_scale_factor': 0.3              # How much to scale by integrated score magnitude
    }
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Confidence-Adjusted Position Sizing.
        
        Args:
            parameters: Custom parameters to override defaults
        """
        # Initialize base class
        super().__init__(parameters)
        
        # Add confidence parameters
        confidence_params = self.DEFAULT_CONFIDENCE_PARAMS.copy()
        if parameters:
            # Extract confidence parameters from the provided parameters
            for key, value in parameters.items():
                if key in confidence_params:
                    confidence_params[key] = value
                    
        self.confidence_params = confidence_params
        logger.info(f"Confidence-Adjusted Position Sizing initialized with parameters: {self.confidence_params}")
    
    def calculate_position_size_with_confidence(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss_pips: float, 
                              account_balance: float,
                              integrated_data: Dict[str, Any],
                              account_currency: str = 'USD',
                              pair_exchange_rate: Optional[float] = None,
                              volatility_factor: Optional[float] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size with adjustments based on confidence metrics.
        
        Args:
            symbol: Forex pair symbol (e.g., 'EUR/USD')
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            account_balance: Account balance
            integrated_data: Integrated indicator and sentiment data
            account_currency: Account currency (default: USD)
            pair_exchange_rate: Exchange rate for cross-currency calculations
            volatility_factor: Optional volatility adjustment
            
        Returns:
            Tuple containing:
                - Adjusted position size in lots
                - Dictionary with details about the confidence adjustment
        """
        # Calculate base position size first
        base_position_size = self.calculate_position_size(
            symbol, 
            entry_price, 
            stop_loss_pips, 
            account_balance,
            account_currency,
            pair_exchange_rate,
            volatility_factor
        )
        
        # If confidence adjustment is disabled or no integrated data, return base size
        if not self.confidence_params['use_confidence_adjustment'] or not integrated_data:
            return base_position_size, {
                'base_position_size': base_position_size,
                'adjusted_position_size': base_position_size,
                'adjustment_factor': 1.0,
                'reason': 'No confidence adjustment applied'
            }
        
        # Extract key metrics from integrated data
        confidence = integrated_data.get('confidence', self.confidence_params['baseline_confidence'])
        integrated_score = integrated_data.get('integrated_score', 0)
        indicator_contribution = integrated_data.get('indicator_contribution', 0)
        sentiment_contribution = integrated_data.get('sentiment_contribution', 0)
        
        # Early check - if confidence is below minimum threshold, consider not trading
        if confidence < self.confidence_params['min_confidence_threshold']:
            return 0.0, {
                'base_position_size': base_position_size,
                'adjusted_position_size': 0.0,
                'adjustment_factor': 0.0,
                'reason': f'Confidence ({confidence:.2f}) below minimum threshold ({self.confidence_params["min_confidence_threshold"]})'
            }
            
        # Start with base adjustment factor based on confidence
        if confidence >= self.confidence_params['high_confidence_threshold']:
            # High confidence
            adjustment_factor = 1.0 + ((confidence - self.confidence_params['high_confidence_threshold']) / 
                                      (1.0 - self.confidence_params['high_confidence_threshold']) * 
                                      (self.confidence_params['high_confidence_factor'] - 1.0))
            confidence_level = 'high'
        elif confidence <= self.confidence_params['baseline_confidence']:
            # Low confidence
            adjustment_factor = self.confidence_params['low_confidence_factor'] + ((confidence - self.confidence_params['min_confidence_threshold']) / 
                                                                                 (self.confidence_params['baseline_confidence'] - self.confidence_params['min_confidence_threshold']) * 
                                                                                 (1.0 - self.confidence_params['low_confidence_factor']))
            confidence_level = 'low'
        else:
            # Medium confidence (baseline to high threshold)
            adjustment_factor = 1.0
            confidence_level = 'medium'
            
        # Adjust based on signal agreement/disagreement
        signal_agreement = False
        signal_disagreement = False
        
        # Check if technical and sentiment are both meaningful and if they agree
        indicator_significant = abs(indicator_contribution) >= self.confidence_params['min_integrated_score']
        sentiment_significant = abs(sentiment_contribution) >= self.confidence_params['min_integrated_score']
        
        if indicator_significant and sentiment_significant:
            if (indicator_contribution > 0 and sentiment_contribution > 0) or \
               (indicator_contribution < 0 and sentiment_contribution < 0):
                # Both indicators and sentiment agree on direction
                adjustment_factor *= (1.0 + self.confidence_params['signal_agreement_bonus'])
                signal_agreement = True
            elif (indicator_contribution > 0 and sentiment_contribution < 0) or \
                 (indicator_contribution < 0 and sentiment_contribution > 0):
                # Indicators and sentiment disagree on direction
                adjustment_factor *= (1.0 - self.confidence_params['signal_disagreement_penalty'])
                signal_disagreement = True
        
        # Adjust based on the magnitude of the integrated score
        if abs(integrated_score) >= self.confidence_params['min_integrated_score']:
            # Stronger signal = larger position
            score_adjustment = 1.0 + (abs(integrated_score) - self.confidence_params['min_integrated_score']) * self.confidence_params['score_scale_factor']
            adjustment_factor *= score_adjustment
        
        # Ensure we don't exceed reasonable bounds
        adjustment_factor = max(0.1, min(adjustment_factor, 2.0))
        
        # Apply adjustment to base position size
        adjusted_position_size = base_position_size * adjustment_factor
        
        # Round to the appropriate increment
        adjusted_position_size = self._round_lot_size(adjusted_position_size, self.parameters['position_size_rounding'])
        
        # Cap at the maximum allowed position size
        adjusted_position_size = min(adjusted_position_size, self.parameters['max_position_size'])
        
        # Prepare detailed response
        adjustment_details = {
            'base_position_size': base_position_size,
            'adjusted_position_size': adjusted_position_size,
            'adjustment_factor': adjustment_factor,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'integrated_score': integrated_score,
            'signal_agreement': signal_agreement,
            'signal_disagreement': signal_disagreement,
            'indicator_contribution': indicator_contribution,
            'sentiment_contribution': sentiment_contribution
        }
        
        logger.info(f"Position size adjusted for {symbol}: {base_position_size:.2f} → {adjusted_position_size:.2f} (factor: {adjustment_factor:.2f})")
        return adjusted_position_size, adjustment_details
    
    def calculate_optimal_risk_percent(self, integrated_data: Dict[str, Any]) -> float:
        """
        Calculate optimal risk percentage based on confidence metrics.
        
        Args:
            integrated_data: Integrated indicator and sentiment data
            
        Returns:
            Optimized risk percentage
        """
        # Default risk percentage
        base_risk_percent = self.parameters['max_risk_per_trade_percent']
        
        # If no integrated data or confidence adjustment disabled, return base risk
        if not integrated_data or not self.confidence_params['use_confidence_adjustment']:
            return base_risk_percent
            
        # Extract confidence
        confidence = integrated_data.get('confidence', self.confidence_params['baseline_confidence'])
        
        # Scale risk based on confidence
        if confidence >= self.confidence_params['high_confidence_threshold']:
            # High confidence - allow full risk
            risk_factor = 1.0
        else:
            # Scale risk linearly from 50% to 100% of max risk as confidence increases
            min_risk_factor = 0.5
            risk_factor = min_risk_factor + (confidence / self.confidence_params['high_confidence_threshold']) * (1.0 - min_risk_factor)
            
        return base_risk_percent * risk_factor
