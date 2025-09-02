#!/usr/bin/env python3
"""
Forex Smart Integration - Connects all smart modules to the main EvoTrader system

This module integrates all the smart enhancements:
- Smart Session Analysis
- Smart Pip Analytics
- Smart News Processing
- Smart Compliance Monitoring
- Smart BenBot Integration
"""

import os
import sys
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import datetime
from pathlib import Path

# Import smart modules
from forex_smart_session import SmartSessionAnalyzer
from forex_smart_pips import SmartPipAnalyzer
from forex_smart_news import SmartNewsAnalyzer
from forex_smart_compliance import SmartComplianceMonitor
from forex_smart_benbot import SmartBenBotConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_smart_integration')


class ForexSmartIntegration:
    """
    Integrates all smart modules and connects them to the main EvoTrader system.
    """
    
    def __init__(self, evotrader=None, config: Dict[str, Any] = None):
        """
        Initialize the smart integration.
        
        Args:
            evotrader: Main EvoTrader instance
            config: Configuration dictionary
        """
        self.evotrader = evotrader
        self.config = config or {}
        
        # Initialize smart components
        self._initialize_smart_components()
        
        logger.info("Forex Smart Integration initialized")
    
    def _initialize_smart_components(self):
        """Initialize all smart components."""
        # Get required objects from evotrader if available
        session_tracker = getattr(self.evotrader, 'session_performance_tracker', None)
        pair_manager = getattr(self.evotrader, 'pair_manager', None)
        news_guard = getattr(self.evotrader, 'news_guard', None)
        benbot_endpoint = self.config.get('benbot_endpoint', 'http://localhost:8080/benbot/api')
        
        # Initialize smart components
        self.smart_session = SmartSessionAnalyzer(session_tracker, self.config)
        self.smart_pips = SmartPipAnalyzer(pair_manager, self.config)
        self.smart_news = SmartNewsAnalyzer(news_guard, self.config)
        self.smart_compliance = SmartComplianceMonitor(self.config)
        self.smart_benbot = SmartBenBotConnector(benbot_endpoint, self.config)
        
        logger.info("All smart components initialized")
    
    def update_evotrader_methods(self):
        """
        Enhance EvoTrader methods with smart functionality.
        This replaces standard methods with enhanced versions.
        """
        if not self.evotrader:
            logger.warning("No EvoTrader instance provided, cannot update methods")
            return
        
        # Store original methods for fallback
        self.original_methods = {
            'check_session_optimal': getattr(self.evotrader, 'check_session_optimal', None),
            'calculate_pip_target': getattr(self.evotrader, 'calculate_pip_target', None),
            'check_news_safe': getattr(self.evotrader, 'check_news_safe', None),
            'calculate_position_size': getattr(self.evotrader, 'calculate_position_size', None),
            'consult_benbot': getattr(self.evotrader, 'consult_benbot', None)
        }
        
        # Replace with enhanced methods
        if hasattr(self.evotrader, 'check_session_optimal'):
            self.evotrader.check_session_optimal = self.enhanced_check_session_optimal
        
        if hasattr(self.evotrader, 'calculate_pip_target'):
            self.evotrader.calculate_pip_target = self.enhanced_calculate_pip_target
        
        if hasattr(self.evotrader, 'check_news_safe'):
            self.evotrader.check_news_safe = self.enhanced_check_news_safe
        
        if hasattr(self.evotrader, 'calculate_position_size'):
            self.evotrader.calculate_position_size = self.enhanced_calculate_position_size
        
        if hasattr(self.evotrader, 'consult_benbot'):
            self.evotrader.consult_benbot = self.enhanced_consult_benbot
        
        logger.info("Enhanced EvoTrader methods with smart functionality")
    
    def enhanced_check_session_optimal(self, pair, strategy_id=None, timestamp=None):
        """
        Enhanced version of session optimality check using smart session analysis.
        
        Args:
            pair: Currency pair
            strategy_id: Strategy ID
            timestamp: Timestamp to check (defaults to current time)
            
        Returns:
            Tuple of (is_optimal, reason)
        """
        # Get original result first
        original_is_optimal, original_reason = False, ""
        if self.original_methods['check_session_optimal']:
            original_is_optimal, original_reason = self.original_methods['check_session_optimal'](
                pair, strategy_id, timestamp)
        
        # Get current data
        timestamp = timestamp or datetime.datetime.now()
        
        # Get current session
        session = self._determine_current_session(timestamp)
        
        # Use smart session analyzer for enhanced check
        session_strength = self.smart_session.detect_session_strength(
            session, pair, current_data=None)
        
        session_personality = self.smart_session.analyze_session_personality(
            session, pair)
        
        # Make decision based on strength and personality
        if session_strength > 1.2:
            # Strong session, generally good for trading
            return True, f"Strong {session} session (strength: {session_strength:.2f})"
        elif session_strength < 0.8:
            # Weak session, generally poor for trading
            return False, f"Weak {session} session (strength: {session_strength:.2f})"
        elif session_personality and 'volatility' in session_personality:
            # Use personality for additional insight
            if session_personality['volatility'] > 0.7:
                return True, f"Volatile {session} session, good for breakout strategies"
            elif session_personality.get('trending', 0) > 0.7:
                return True, f"Trending {session} session, good for trend strategies"
            elif session_personality.get('range', 0) > 0.7:
                return True, f"Ranging {session} session, good for range strategies"
        
        # Fall back to original result if we can't make a clear determination
        return original_is_optimal, original_reason or f"Default {session} session assessment"
    
    def _determine_current_session(self, timestamp=None):
        """
        Determine current forex session based on time.
        
        Args:
            timestamp: Timestamp to check (defaults to current time)
            
        Returns:
            Session name
        """
        timestamp = timestamp or datetime.datetime.now()
        hour_utc = timestamp.hour if timestamp.tzinfo else (timestamp.hour - 4)  # Assuming EDT offset
        
        # Simple session determination
        if 7 <= hour_utc < 16:
            return "London"
        elif 12 <= hour_utc < 21:
            return "NewYork"
        elif 0 <= hour_utc < 9:
            return "Tokyo"
        elif 21 <= hour_utc < 24 or 0 <= hour_utc < 7:
            return "Sydney"
        else:
            return "Overlap"
    
    def enhanced_calculate_pip_target(self, pair, strategy_id=None):
        """
        Enhanced version of pip target calculation using volatility-based analytics.
        
        Args:
            pair: Currency pair
            strategy_id: Strategy ID
            
        Returns:
            Dictionary with pip targets
        """
        # Get original result first
        original_target = 20.0  # Default
        if self.original_methods['calculate_pip_target']:
            original_target = self.original_methods['calculate_pip_target'](pair, strategy_id)
        
        # Use smart pip analyzer for enhanced targets
        pip_targets = self.smart_pips.calculate_pip_target(pair, base_pips=original_target)
        
        return pip_targets
    
    def enhanced_check_news_safe(self, pair, timestamp=None, hours_ahead=24):
        """
        Enhanced version of news safety check using impact prediction.
        
        Args:
            pair: Currency pair
            timestamp: Timestamp to check (defaults to current time)
            hours_ahead: Hours ahead to check for news
            
        Returns:
            Tuple of (is_safe, reason)
        """
        # Get original result first
        original_is_safe, original_reason = True, ""
        if self.original_methods['check_news_safe']:
            original_is_safe, original_reason = self.original_methods['check_news_safe'](
                pair, timestamp, hours_ahead)
        
        # If news guard says unsafe, respect that
        if not original_is_safe:
            return False, original_reason
        
        # Get upcoming news events from news guard if available
        news_events = []
        news_guard = getattr(self.evotrader, 'news_guard', None)
        if news_guard:
            news_events = news_guard.get_upcoming_events(hours_ahead=hours_ahead)
        
        # If no events, safe to trade
        if not news_events:
            return True, "No upcoming news events"
        
        # Use smart news analyzer to predict impact
        high_impact_events = []
        for event in news_events:
            impact = self.smart_news.predict_news_impact(event, pair)
            if impact.get('pips', 0) > 20 and impact.get('confidence', 0) > 0.6:
                high_impact_events.append({
                    'event': event,
                    'impact': impact,
                    'time_until': self._calculate_hours_until(event.get('timestamp'))
                })
        
        # Sort by impact and time
        high_impact_events.sort(key=lambda x: (x['time_until'], -x['impact'].get('pips', 0)))
        
        # Check if we have high impact events soon
        if high_impact_events:
            nearest = high_impact_events[0]
            if nearest['time_until'] < 1:
                return False, f"High impact news in <1 hour: {nearest['event'].get('title')}"
            elif nearest['time_until'] < 3:
                return False, f"Medium-high impact news in {nearest['time_until']:.1f} hours"
        
        # Otherwise safe to trade
        return True, "No high impact news events detected"
    
    def _calculate_hours_until(self, timestamp):
        """Calculate hours until a timestamp."""
        if not timestamp:
            return 24  # Default to 24 hours if no timestamp
        
        try:
            now = datetime.datetime.now()
            if isinstance(timestamp, str):
                timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            diff = timestamp - now
            return max(0, diff.total_seconds() / 3600)
        except Exception:
            return 24  # Default on error
    
    def enhanced_calculate_position_size(self, equity, pair, strategy_id=None):
        """
        Enhanced version of position size calculation using adaptive risk budgeting.
        
        Args:
            equity: Account equity
            pair: Currency pair
            strategy_id: Strategy ID
            
        Returns:
            Position size in lots
        """
        # Get original result first
        original_size = 0.1  # Default
        if self.original_methods['calculate_position_size']:
            original_size = self.original_methods['calculate_position_size'](
                equity, pair, strategy_id)
        
        # Get current account state
        drawdown = getattr(self.evotrader, 'current_drawdown', 0.0)
        daily_pnl = getattr(self.evotrader, 'daily_pnl', 0.0)
        
        # Use smart compliance for enhanced position sizing
        self.smart_compliance.update_account_state(equity, daily_pnl)
        adjusted_size = self.smart_compliance.calculate_position_size(
            original_size, pair, strategy_id)
        
        return adjusted_size
    
    def enhanced_consult_benbot(self, action, data):
        """
        Enhanced version of BenBot consultation using confidence weighting.
        
        Args:
            action: Action type
            data: Consultation data
            
        Returns:
            BenBot response
        """
        # Use smart BenBot connector for enhanced consultation
        enhanced_response = self.smart_benbot.consult_benbot_with_confidence(action, data)
        
        return enhanced_response
    
    def run_smart_analysis(self, pair, analysis_type='all', strategy_id=None):
        """
        Run smart analysis on a pair or strategy.
        
        Args:
            pair: Currency pair to analyze
            analysis_type: Type of analysis ('session', 'pips', 'news', 'compliance', 'all')
            strategy_id: Strategy ID to analyze
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Session analysis
        if analysis_type in ['session', 'all']:
            current_session = self._determine_current_session()
            session_strength = self.smart_session.detect_session_strength(current_session, pair)
            session_personality = self.smart_session.analyze_session_personality(current_session, pair)
            
            results['session'] = {
                'current_session': current_session,
                'session_strength': session_strength,
                'session_personality': session_personality,
                'optimal_sessions': self.smart_session.get_optimal_sessions(pair, strategy_id)
            }
        
        # Pip analysis
        if analysis_type in ['pips', 'all']:
            pip_targets = self.smart_pips.calculate_pip_target(pair)
            risk_adjusted_metrics = self.smart_pips.calculate_risk_adjusted_metrics(pair)
            
            results['pips'] = {
                'pip_targets': pip_targets,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'correlation_adjusted_values': self.smart_pips.calculate_correlation_adjusted_pip_value(pair)
            }
        
        # News analysis
        if analysis_type in ['news', 'all']:
            # Get upcoming news events
            news_guard = getattr(self.evotrader, 'news_guard', None)
            news_events = []
            if news_guard:
                news_events = news_guard.get_upcoming_events(hours_ahead=24)
            
            predicted_impacts = []
            for event in news_events:
                impact = self.smart_news.predict_news_impact(event, pair)
                predicted_impacts.append({
                    'event': event.get('title', ''),
                    'time': event.get('timestamp', ''),
                    'impact': impact
                })
            
            results['news'] = {
                'upcoming_events': len(news_events),
                'predicted_impacts': predicted_impacts
            }
        
        # Compliance analysis
        if analysis_type in ['compliance', 'all']:
            # Get current account state from EvoTrader if available
            equity = 1000.0
            if hasattr(self.evotrader, 'account_equity'):
                equity = self.evotrader.account_equity
            
            drawdown = 0.0
            if hasattr(self.evotrader, 'current_drawdown'):
                drawdown = self.evotrader.current_drawdown
            
            daily_pnl = 0.0
            if hasattr(self.evotrader, 'daily_pnl'):
                daily_pnl = self.evotrader.daily_pnl
            
            # Update compliance monitor
            self.smart_compliance.update_account_state(equity, daily_pnl)
            
            # Get risk projection
            risk_projection = self.smart_compliance.project_drawdown_risk()
            
            # Check if trading is allowed
            is_trading_allowed, reason = self.smart_compliance.is_trading_allowed()
            
            results['compliance'] = {
                'current_state': {
                    'equity': equity,
                    'drawdown': drawdown,
                    'daily_pnl': daily_pnl
                },
                'risk_projection': risk_projection,
                'trading_allowed': is_trading_allowed,
                'reason': reason
            }
        
        return results


def run_cli():
    """CLI entry point for smart analysis."""
    parser = argparse.ArgumentParser(description='Run Forex Smart Analysis')
    
    parser.add_argument('--pair', type=str, required=True, help='Currency pair to analyze')
    parser.add_argument('--analysis-type', type=str, choices=['session', 'pips', 'news', 'compliance', 'all'],
                       default='all', help='Type of smart analysis to run')
    parser.add_argument('--strategy-id', type=str, help='Strategy ID to analyze')
    parser.add_argument('--config', type=str, default='forex_evotrader_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        config = {}
    
    # Create smart integration
    integration = ForexSmartIntegration(config=config)
    
    # Run analysis
    results = integration.run_smart_analysis(
        args.pair, args.analysis_type, args.strategy_id)
    
    # Print results
    import json
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    run_cli()
