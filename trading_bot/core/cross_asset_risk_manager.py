#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Asset Risk Manager

This module implements cross-asset risk management to prevent excessive
concentration in correlated assets across different classes.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class CrossAssetRiskManager:
    """
    Cross-Asset Risk Manager
    
    This class manages risk across different asset classes:
    - Prevents excessive concentration in correlated assets
    - Identifies hidden correlations (e.g., USD exposure across forex and crypto)
    - Monitors and limits portfolio-level risk metrics
    - Provides risk-based position size adjustments
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Risk limits
        'max_portfolio_var': 0.015,   # Maximum portfolio variance (1.5%)
        'max_currency_exposure': 0.4,  # Maximum exposure to single currency (40%)
        'max_sector_exposure': 0.3,    # Maximum exposure to single sector (30%)
        'max_correlation_cluster': 0.25,  # Maximum exposure to highly correlated assets (25%)
        
        # Correlation thresholds
        'high_correlation_threshold': 0.7,  # Threshold for high correlation
        'medium_correlation_threshold': 0.4,  # Threshold for medium correlation
        
        # Risk adjustment factors
        'high_correlation_penalty': 0.7,  # Position size multiplier for high correlation
        'medium_correlation_penalty': 0.85,  # Position size multiplier for medium correlation
        'safe_correlation_bonus': 1.1,  # Position size multiplier for negative correlation
        
        # VIX-based adjustments
        'high_vix_threshold': 25.0,  # VIX threshold for high volatility
        'low_vix_threshold': 15.0,   # VIX threshold for low volatility
        'high_vix_adjustment': 0.8,  # Position size multiplier for high VIX
        'low_vix_adjustment': 1.1    # Position size multiplier for low VIX
    }
    
    def __init__(self, event_bus: EventBus, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Cross-Asset Risk Manager.
        
        Args:
            event_bus: System event bus
            parameters: Custom parameters to override defaults
        """
        self.event_bus = event_bus
        
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
            
        # Initialize state
        self.portfolio_state = {
            'positions': [],
            'exposures': {
                'currency': {},  # Currency exposures across assets
                'sector': {},    # Sector exposures (stocks, crypto)
                'factor': {}     # Factor exposures (value, growth, etc.)
            },
            'correlations': {},  # Correlation matrix
            'risk_metrics': {
                'portfolio_var': 0.0,
                'var_95': 0.0,
                'max_drawdown': 0.0
            }
        }
        
        # Market context
        self.market_context = {
            'vix': 15.0,
            'regime': 'neutral'
        }
        
        # Register for events
        self._register_events()
        
        logger.info("Cross-Asset Risk Manager initialized")
        
    def _register_events(self):
        """Register for events of interest."""
        self.event_bus.subscribe(EventType.CAPITAL_ALLOCATIONS_UPDATED, self._on_allocations_updated)
        self.event_bus.subscribe(EventType.POSITION_OPENED, self._on_position_changed)
        self.event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_changed)
        self.event_bus.subscribe(EventType.POSITION_MODIFIED, self._on_position_changed)
        self.event_bus.subscribe(EventType.PORTFOLIO_UPDATED, self._on_portfolio_updated)
        self.event_bus.subscribe(EventType.CONTEXT_ANALYSIS_COMPLETED, self._on_context_analysis)
    
    def _on_allocations_updated(self, event: Event):
        """
        Handle capital allocations updated events.
        
        Args:
            event: Capital allocations updated event
        """
        # Extract allocations
        if 'allocations' not in event.data:
            return
            
        allocations = event.data['allocations']
        
        # Get positions
        positions = allocations.get('positions', [])
        
        # Apply risk adjustments
        adjusted_positions = self._apply_risk_adjustments(positions)
        
        # Publish risk-adjusted allocations
        self.event_bus.publish(Event(
            event_type=EventType.RISK_ADJUSTED_ALLOCATIONS,
            data={
                'original_allocations': allocations,
                'risk_adjusted_positions': adjusted_positions,
                'risk_metrics': self.portfolio_state['risk_metrics'],
                'timestamp': datetime.now()
            }
        ))
    
    def _on_position_changed(self, event: Event):
        """
        Handle position changed events.
        
        Args:
            event: Position changed event
        """
        # Recalculate exposures
        self._calculate_exposures()
        
        # Recalculate risk metrics
        self._calculate_risk_metrics()
        
        # Publish updated risk metrics
        self._publish_risk_metrics()
    
    def _on_portfolio_updated(self, event: Event):
        """
        Handle portfolio updated events.
        
        Args:
            event: Portfolio updated event
        """
        # Update portfolio state
        if 'portfolio' in event.data:
            portfolio = event.data['portfolio']
            
            # Update positions
            if 'positions' in portfolio:
                self.portfolio_state['positions'] = portfolio['positions']
                
            # Update correlations if available
            if 'correlations' in portfolio:
                self.portfolio_state['correlations'] = portfolio['correlations']
                
        # Recalculate exposures
        self._calculate_exposures()
        
        # Recalculate risk metrics
        self._calculate_risk_metrics()
        
        # Publish updated risk metrics
        self._publish_risk_metrics()
    
    def _on_context_analysis(self, event: Event):
        """
        Handle context analysis events.
        
        Args:
            event: Context analysis event
        """
        # Update market context
        if 'context' in event.data:
            context = event.data['context']
            
            # Update VIX
            if 'vix' in context:
                self.market_context['vix'] = context['vix']
                
            # Update market regime
            if 'market_regime' in context:
                self.market_context['regime'] = context['market_regime']
