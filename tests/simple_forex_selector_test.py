#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified test for the Forex Strategy Selector.

This standalone script tests the principles of:
1. Choosing a Forex Strategy based on market conditions
2. Knowing when to trade (time-awareness)
3. Understanding risk tolerance parameters

It avoids dependencies on the main trading_bot module structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
import json

# Define necessary enums here to avoid dependencies
class MarketRegime(Enum):
    BULL_TREND = 1
    BEAR_TREND = 2
    CONSOLIDATION = 3
    HIGH_VOLATILITY = 4
    LOW_VOLATILITY = 5
    UNKNOWN = 6

class ForexSession(Enum):
    SYDNEY = 1
    TOKYO = 2
    LONDON = 3
    NEWYORK = 4
    LONDON_NEWYORK_OVERLAP = 5

class RiskTolerance:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# =========================================================
# Strategy Scoring Functions - Simulating what's in our full implementation
# =========================================================

def trend_following_score(regime):
    """Return Forex Trend Following Strategy compatibility score"""
    scores = {
        MarketRegime.BULL_TREND: 0.85,
        MarketRegime.BEAR_TREND: 0.80,
        MarketRegime.CONSOLIDATION: 0.30,
        MarketRegime.HIGH_VOLATILITY: 0.65,
        MarketRegime.LOW_VOLATILITY: 0.40,
        MarketRegime.UNKNOWN: 0.50
    }
    return scores.get(regime, 0.50)

def range_trading_score(regime):
    """Return Forex Range Trading Strategy compatibility score"""
    scores = {
        MarketRegime.BULL_TREND: 0.30,
        MarketRegime.BEAR_TREND: 0.35,
        MarketRegime.CONSOLIDATION: 0.90,
        MarketRegime.HIGH_VOLATILITY: 0.50,
        MarketRegime.LOW_VOLATILITY: 0.85,
        MarketRegime.UNKNOWN: 0.50
    }
    return scores.get(regime, 0.50)

def breakout_score(regime):
    """Return Forex Breakout Strategy compatibility score"""
    scores = {
        MarketRegime.BULL_TREND: 0.65,
        MarketRegime.BEAR_TREND: 0.65,
        MarketRegime.CONSOLIDATION: 0.60,
        MarketRegime.HIGH_VOLATILITY: 0.90,
        MarketRegime.LOW_VOLATILITY: 0.40,
        MarketRegime.UNKNOWN: 0.55
    }
    return scores.get(regime, 0.55)

def momentum_score(regime):
    """Return Forex Momentum Strategy compatibility score"""
    scores = {
        MarketRegime.BULL_TREND: 0.80,
        MarketRegime.BEAR_TREND: 0.95,
        MarketRegime.CONSOLIDATION: 0.40,
        MarketRegime.HIGH_VOLATILITY: 0.85,
        MarketRegime.LOW_VOLATILITY: 0.30,
        MarketRegime.UNKNOWN: 0.60
    }
    return scores.get(regime, 0.60)

# =========================================================
# Simplified Strategy Selector
# =========================================================

class SimpleForexStrategySelector:
    """Simplified version of the forex strategy selector for testing"""
    
    def __init__(self, risk_tolerance=RiskTolerance.MEDIUM):
        self.risk_tolerance = risk_tolerance
        
        # Strategy compatibility database
        self.strategy_compatibility = {
            "forex_trend_following": trend_following_score,
            "forex_range_trading": range_trading_score,
            "forex_breakout": breakout_score,
            "forex_momentum": momentum_score
        }
        
        # Session preferences for strategies
        self.session_preferences = {
            ForexSession.SYDNEY: ["forex_range_trading", "forex_breakout"],
            ForexSession.TOKYO: ["forex_range_trading", "forex_momentum"],
            ForexSession.LONDON: ["forex_trend_following", "forex_momentum"],
            ForexSession.NEWYORK: ["forex_breakout", "forex_momentum"],
            ForexSession.LONDON_NEWYORK_OVERLAP: ["forex_breakout", "forex_trend_following"]
        }
        
        # Risk adjustments
        self.risk_adjustments = {
            RiskTolerance.LOW: {
                "position_size_multiplier": 0.7,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 1.2,
                "max_trades_per_session": 3,
                "max_risk_per_trade_pct": 1.0,
                "max_risk_per_day_pct": 3.0,
                "preferred_strategies": ["forex_range_trading", "forex_trend_following"]
            },
            RiskTolerance.MEDIUM: {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "max_trades_per_session": 5,
                "max_risk_per_trade_pct": 2.0,
                "max_risk_per_day_pct": 6.0, 
                "preferred_strategies": []
            },
            RiskTolerance.HIGH: {
                "position_size_multiplier": 1.3,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 0.8,
                "max_trades_per_session": 8,
                "max_risk_per_trade_pct": 3.0,
                "max_risk_per_day_pct": 10.0,
                "preferred_strategies": ["forex_breakout", "forex_momentum"]
            }
        }

    def select_strategy(self, market_regime, active_sessions, has_high_impact_news=False):
        """
        Select the optimal strategy based on market regime, sessions and risk tolerance
        
        Args:
            market_regime: Current market regime
            active_sessions: List of active trading sessions
            has_high_impact_news: Whether there's high impact news
            
        Returns:
            Tuple with (strategy_name, score, parameters)
        """
        # 1. Get base scores from market regime
        scores = {}
        for strategy_name, score_func in self.strategy_compatibility.items():
            scores[strategy_name] = score_func(market_regime)
            
        # 2. Apply session preferences
        for session in active_sessions:
            preferred = self.session_preferences.get(session, [])
            for strategy in preferred:
                if strategy in scores:
                    scores[strategy] += 0.1
        
        # 3. Apply risk tolerance adjustments
        risk_params = self.risk_adjustments.get(self.risk_tolerance)
        for strategy in risk_params.get("preferred_strategies", []):
            if strategy in scores:
                scores[strategy] += 0.15
                
        # 4. Apply news event adjustments if needed
        if has_high_impact_news:
            if "forex_range_trading" in scores:
                scores["forex_range_trading"] *= 0.6
            if "forex_breakout" in scores:
                scores["forex_breakout"] *= 1.5
            if "forex_trend_following" in scores:
                scores["forex_trend_following"] *= 0.8
                
        # 5. Normalize scores to keep them in the 0-1 range
        max_score = max(scores.values())
        if max_score > 1.0:
            scores = {k: v/max_score for k, v in scores.items()}
                
        # 6. Select the best strategy
        best_strategy = max(scores.items(), key=lambda x: x[1])
        strategy_name, score = best_strategy
        
        # 7. Get optimized parameters
        params = self.get_optimized_parameters(strategy_name, market_regime)
        
        return strategy_name, score, params
    
    def get_optimized_parameters(self, strategy_name, regime):
        """Get optimized parameters for the selected strategy"""
        # Base parameters
        params = {
            "pip_value": 0.0001,
            "atr_period": 14,
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 3.0
        }
        
        # Apply risk adjustments
        risk_params = self.risk_adjustments.get(self.risk_tolerance)
        params["stop_loss_atr_mult"] *= risk_params["stop_loss_multiplier"]
        params["take_profit_atr_mult"] *= risk_params["take_profit_multiplier"]
        
        # Strategy-specific adjustments
        if strategy_name == "forex_trend_following":
            params["fast_ma_period"] = 20 if regime == MarketRegime.HIGH_VOLATILITY else 10
            params["slow_ma_period"] = 50 if regime == MarketRegime.HIGH_VOLATILITY else 30
            params["adx_threshold"] = 25
            
        elif strategy_name == "forex_range_trading":
            params["bb_period"] = 20
            params["rsi_period"] = 14
            params["rsi_overbought"] = 70
            params["rsi_oversold"] = 30
            
        elif strategy_name == "forex_breakout":
            params["donchian_period"] = 20
            params["confirmation_candles"] = 3 if self.risk_tolerance == RiskTolerance.LOW else 1
            
        elif strategy_name == "forex_momentum":
            params["roc_period"] = 14
            params["rsi_period"] = 14
            params["adx_threshold"] = 20 if self.risk_tolerance == RiskTolerance.HIGH else 25
            
        return params
    
    def get_active_sessions(self, hour):
        """Get active forex sessions for a given hour (UTC)"""
        active_sessions = []
        
        # Simple hour-based session detection
        if hour >= 22 or hour < 7:
            active_sessions.append(ForexSession.SYDNEY)
            
        if hour >= 0 and hour < 9:
            active_sessions.append(ForexSession.TOKYO)
            
        if hour >= 8 and hour < 17:
            active_sessions.append(ForexSession.LONDON)
            
        if hour >= 13 and hour < 22:
            active_sessions.append(ForexSession.NEWYORK)
            
        if hour >= 13 and hour < 17:
            active_sessions.append(ForexSession.LONDON_NEWYORK_OVERLAP)
            
        return active_sessions

def test_strategy_selector():
    """Test the forex strategy selection mechanism"""
    print("\n" + "=" * 80)
    print("FOREX STRATEGY SELECTOR TEST")
    print("=" * 80)
    
    # Market regimes to test
    regimes = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.CONSOLIDATION,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY
    ]
    
    # Trading hours to test (UTC)
    hours = [2, 8, 14, 20]
    
    # Risk levels to test
    risk_levels = [RiskTolerance.LOW, RiskTolerance.MEDIUM, RiskTolerance.HIGH]
    
    print("\nStrategy Selection Results:")
    print("-" * 100)
    print(f"{'Risk Level':<10} {'Hour (UTC)':<12} {'Active Sessions':<30} {'Market Regime':<16} {'Selected Strategy':<20} {'Score':<10} {'Parameters'}")
    print("-" * 100)
    
    # Test all combinations
    for risk in risk_levels:
        selector = SimpleForexStrategySelector(risk_tolerance=risk)
        
        for hour in hours:
            active_sessions = selector.get_active_sessions(hour)
            session_names = [s.name for s in active_sessions]
            
            for regime in regimes:
                # Select strategy
                strategy, score, params = selector.select_strategy(
                    market_regime=regime,
                    active_sessions=active_sessions
                )
                
                # Display results
                risk_name = "LOW" if risk == RiskTolerance.LOW else "MEDIUM" if risk == RiskTolerance.MEDIUM else "HIGH"
                session_str = ", ".join(session_names) if session_names else "None"
                
                # Format parameters nicely
                param_str = f"SL:{params['stop_loss_atr_mult']:.2f}, TP:{params['take_profit_atr_mult']:.2f}"
                
                print(f"{risk_name:<10} {hour:02d}:00 UTC{'':<6} {session_str:<30} {regime.name:<16} {strategy:<20} {score:.2f}{'':<5} {param_str}")
    
    print("\n" + "=" * 80)
    
    # Test risk parameter differences
    print("\nRisk Parameter Comparison:")
    print("-" * 80)
    print(f"{'Parameter':<25} {'Low Risk':<15} {'Medium Risk':<15} {'High Risk':<15}")
    print("-" * 80)
    
    params_to_show = [
        'position_size_multiplier', 
        'stop_loss_multiplier', 
        'take_profit_multiplier', 
        'max_trades_per_session', 
        'max_risk_per_trade_pct', 
        'max_risk_per_day_pct'
    ]
    
    # Get parameters for each risk level
    low_params = SimpleForexStrategySelector(RiskTolerance.LOW).risk_adjustments[RiskTolerance.LOW]
    med_params = SimpleForexStrategySelector(RiskTolerance.MEDIUM).risk_adjustments[RiskTolerance.MEDIUM]
    high_params = SimpleForexStrategySelector(RiskTolerance.HIGH).risk_adjustments[RiskTolerance.HIGH]
    
    for param in params_to_show:
        print(f"{param:<25} {low_params[param]:<15} {med_params[param]:<15} {high_params[param]:<15}")
    
    # Show preferred strategies by risk level
    print("\nPreferred Strategies by Risk Tolerance:")
    print("-" * 80)
    
    low_preferred = ", ".join(low_params["preferred_strategies"]) if low_params["preferred_strategies"] else "None"
    med_preferred = ", ".join(med_params["preferred_strategies"]) if med_params["preferred_strategies"] else "None"
    high_preferred = ", ".join(high_params["preferred_strategies"]) if high_params["preferred_strategies"] else "None"
    
    print(f"{'Low Risk':<15}: {low_preferred}")
    print(f"{'Medium Risk':<15}: {med_preferred}")
    print(f"{'High Risk':<15}: {high_preferred}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    test_strategy_selector()
