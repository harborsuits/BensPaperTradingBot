"""
Macro Guidance Engine

This module provides a comprehensive framework for integrating macroeconomic events
and market regime analysis into trading decisions.
"""

import logging
import json
import os
import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from .macro_event_definitions import (
    MacroEvent, TradingBias, EventType, MarketImpact, EventImportance,
    create_event, StrategyAdjustment
)
from .sector_rotation_loader import SectorRotationLoader
from .seasonality_insights_loader import SeasonalityInsightsLoader

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Economic market regimes."""
    EXPANSION = "expansion"
    LATE_CYCLE = "late_cycle"
    CONTRACTION = "contraction"
    EARLY_RECOVERY = "early_recovery"
    UNKNOWN = "unknown"

class MacroGuidanceEngine:
    """
    Engine for processing macroeconomic events and regimes to guide trading decisions.
    Integrates with the broader trading system to adjust strategies based on macro conditions.
    """
    
    def __init__(self, config=None):
        """Initialize the macro guidance engine."""
        self.config = config or {}
        self.events = {}
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_probabilities = {
            MarketRegime.EXPANSION: 0.25,
            MarketRegime.LATE_CYCLE: 0.25,
            MarketRegime.CONTRACTION: 0.25,
            MarketRegime.EARLY_RECOVERY: 0.25
        }
        self.upcoming_events = []
        self.recent_events = []
        self.economic_indicators = {}
        
        # Load predefined macro events
        self._load_macro_events()
        
        # Load economic calendar
        self._load_economic_calendar()
        
        # Load sector rotation framework if specified in config
        self._load_sector_rotation_framework()
        
        # Load seasonality insights framework if specified in config
        self._load_seasonality_insights_framework()
        
        # Determine current market regime
        self._determine_current_regime()
        
        logger.info(f"Macro Guidance Engine initialized with {len(self.events)} events, current regime: {self.current_regime.value}")
    
    def _load_macro_events(self):
        """Load predefined macro economic events."""
        # In a production system, this might load from a database or API
        # For now, we'll just create instances of key events
        
        self.events = {
            "cpi": create_event("cpi"),
            # Add other events as needed
        }
    
    def _load_economic_calendar(self):
        """Load upcoming economic events from calendar."""
        # In a production system, this would fetch from an economic calendar API
        # For now, we'll use a simplified approach
        
        today = datetime.datetime.now().date()
        
        # Example of adding upcoming events
        self.upcoming_events = [
            {
                "event_type": "cpi",
                "date": today + datetime.timedelta(days=7),
                "time": "08:30 ET",
                "description": "US Consumer Price Index",
                "forecast": {
                    "headline_cpi": 3.1,
                    "core_cpi": 3.3
                }
            },
            # Additional events would be added here
        ]
    
    def _load_sector_rotation_framework(self):
        """Load sector rotation framework from configuration."""
        # Check if sector rotation path is specified in config
        sector_rotation_path = self.config.get("sector_rotation_path")
        
        if sector_rotation_path and os.path.exists(sector_rotation_path):
            try:
                # Load from file
                loader = SectorRotationLoader(sector_rotation_path)
                self.config["sector_rotation"] = loader.get_framework()
                logger.info(f"Loaded sector rotation framework from {sector_rotation_path}")
            except Exception as e:
                logger.error(f"Error loading sector rotation framework: {str(e)}")
        
        # Check if we already have framework data in config
        elif "sector_rotation" in self.config and isinstance(self.config["sector_rotation"], dict):
            try:
                # Validate the existing framework data
                loader = SectorRotationLoader()
                loader.load_from_data(self.config["sector_rotation"])
                logger.info("Validated existing sector rotation framework data in config")
            except Exception as e:
                logger.error(f"Error validating sector rotation framework in config: {str(e)}")
                # Remove invalid framework data
                self.config.pop("sector_rotation", None)
    
    def _load_seasonality_insights_framework(self):
        """Load seasonality insights framework from configuration."""
        # Check if seasonality insights path is specified in config
        seasonality_path = self.config.get("seasonality_insights_path")
        
        if seasonality_path and os.path.exists(seasonality_path):
            try:
                # Load from file
                loader = SeasonalityInsightsLoader(seasonality_path)
                self.config["seasonality_insights"] = loader.get_framework()
                logger.info(f"Loaded seasonality insights framework from {seasonality_path}")
            except Exception as e:
                logger.error(f"Error loading seasonality insights framework: {str(e)}")
        
        # Check if we already have framework data in config
        elif "seasonality_insights" in self.config and isinstance(self.config["seasonality_insights"], dict):
            try:
                # Validate the existing framework data
                loader = SeasonalityInsightsLoader()
                loader.load_from_data(self.config["seasonality_insights"])
                logger.info("Validated existing seasonality insights framework data in config")
            except Exception as e:
                logger.error(f"Error validating seasonality insights framework in config: {str(e)}")
                # Remove invalid framework data
                self.config.pop("seasonality_insights", None)
    
    def _determine_current_regime(self):
        """
        Determine the current market regime based on economic indicators.
        This is a simplified implementation - a real system would use 
        more sophisticated methods and multiple indicators.
        """
        # For demonstration purposes, we'll just use a fixed regime
        # In practice, this would analyze various economic indicators
        
        # This would be replaced with actual analysis of indicators like:
        # - GDP growth trends
        # - Unemployment data
        # - Yield curve shape
        # - Inflation trends
        # - Credit spreads
        
        # Simplified logic for demonstration
        # with random variation in the probabilities
        np.random.seed(int(datetime.datetime.now().timestamp()))
        
        # Base probabilities
        probs = np.array([0.40, 0.35, 0.15, 0.10])  # Expansion, Late Cycle, Contraction, Early Recovery
        
        # Add some randomness
        noise = np.random.normal(0, 0.05, 4)
        probs = probs + noise
        
        # Ensure probabilities are valid
        probs = np.maximum(probs, 0)
        probs = probs / probs.sum()
        
        # Update regime probabilities
        self.regime_probabilities = {
            MarketRegime.EXPANSION: probs[0],
            MarketRegime.LATE_CYCLE: probs[1],
            MarketRegime.CONTRACTION: probs[2],
            MarketRegime.EARLY_RECOVERY: probs[3]
        }
        
        # Set current regime to the highest probability
        self.current_regime = max(self.regime_probabilities, key=self.regime_probabilities.get)
    
    def process_economic_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an economic event and provide trading guidance.
        
        Args:
            event_type: Type of economic event (e.g., 'cpi', 'fomc')
            event_data: Actual data from the event
            
        Returns:
            Dict with analysis and strategy recommendations
        """
        if event_type not in self.events:
            logger.warning(f"Unknown event type: {event_type}")
            return {"status": "error", "message": f"Unknown event type: {event_type}"}
        
        event = self.events[event_type]
        
        # Determine the current scenario based on the event outcome
        scenario, bias = event.get_current_scenario(event_data)
        
        # Get strategy adjustments for the current scenario
        strategy_adjustment = event.get_strategy_adjustment("post_event", scenario)
        
        # Record this event in recent events
        self.recent_events.append({
            "event_type": event_type,
            "date": datetime.datetime.now().isoformat(),
            "data": event_data,
            "scenario": scenario.name,
            "bias": bias.value
        })
        
        # Return analysis and recommendations
        return {
            "status": "success",
            "event_type": event_type,
            "event_id": event.event_id,
            "importance": event.importance_level,
            "scenario": {
                "name": scenario.name,
                "description": scenario.description if hasattr(scenario, 'description') else None,
                "primary_catalyst": scenario.primary_catalyst,
                "market_response": {
                    "equities": scenario.typical_market_response.equities,
                    "bonds": scenario.typical_market_response.bonds,
                    "volatility": scenario.typical_market_response.volatility,
                    "duration": scenario.typical_market_response.duration_of_effect
                }
            },
            "trading_bias": bias.value,
            "strategy_recommendations": self._format_strategy_recommendations(strategy_adjustment),
            "current_regime": self.current_regime.value,
            "regime_impact": self._get_regime_specific_guidance(event_type, bias)
        }
    
    def _format_strategy_recommendations(self, adjustment: StrategyAdjustment) -> Dict[str, Any]:
        """Format strategy recommendations in a consistent structure."""
        if not adjustment:
            return {}
        
        recommendations = {}
        
        # Add equity strategy if available
        if adjustment.equity_strategy:
            recommendations["equity"] = {
                "primary_action": adjustment.equity_strategy.primary_action,
                "position_management": adjustment.equity_strategy.position_management,
                "sector_adjustments": adjustment.equity_strategy.sector_adjustments
            }
            if adjustment.equity_strategy.target_sectors:
                recommendations["equity"]["target_sectors"] = adjustment.equity_strategy.target_sectors
            if adjustment.equity_strategy.timing:
                recommendations["equity"]["timing"] = adjustment.equity_strategy.timing
        
        # Add options strategy if available
        if adjustment.options_strategy:
            recommendations["options"] = {
                "primary_approach": adjustment.options_strategy.primary_approach,
                "strategies": []
            }
            
            for strategy in adjustment.options_strategy.recommended_strategies:
                strategy_dict = {
                    "strategy": strategy.strategy,
                    "implementation": strategy.implementation
                }
                
                # Add optional fields if they exist
                for field in ["timeframe", "exit_plan", "benefit", "risk_adjustment", "strikes", "sizing"]:
                    if hasattr(strategy, field) and getattr(strategy, field):
                        strategy_dict[field] = getattr(strategy, field)
                
                recommendations["options"]["strategies"].append(strategy_dict)
        
        # Add other strategy types if available
        for strategy_type in ["futures_strategy", "fixed_income_strategy", "bond_strategy"]:
            if hasattr(adjustment, strategy_type) and getattr(adjustment, strategy_type):
                recommendations[strategy_type.replace("_strategy", "")] = getattr(adjustment, strategy_type)
        
        # Add risk management if available
        if hasattr(adjustment, "risk_management") and adjustment.risk_management:
            recommendations["risk_management"] = adjustment.risk_management
        
        return recommendations
    
    def _get_regime_specific_guidance(self, event_type: str, bias: TradingBias) -> Dict[str, Any]:
        """
        Get regime-specific guidance for the given event and bias.
        
        Args:
            event_type: Type of economic event
            bias: Trading bias from the event
            
        Returns:
            Dict with regime-specific adjustments
        """
        # This would contain regime-specific modifications to the general guidance
        # For example, a bullish CPI report might have different implications in 
        # an expansion regime vs. a contraction regime
        
        regime = self.current_regime
        
        # Simplified implementation - would be more sophisticated in practice
        if regime == MarketRegime.EXPANSION:
            return {
                "position_sizing_modifier": 1.0 if bias == TradingBias.BULLISH else 0.8,
                "timeframe_modifier": "standard",
                "emphasis": "focus on growth sectors if bullish"
            }
        elif regime == MarketRegime.LATE_CYCLE:
            return {
                "position_sizing_modifier": 0.8 if bias == TradingBias.BULLISH else 0.7,
                "timeframe_modifier": "shorter by 30%",
                "emphasis": "prioritize quality and defense even in bullish scenarios"
            }
        elif regime == MarketRegime.CONTRACTION:
            return {
                "position_sizing_modifier": 0.6 if bias == TradingBias.BULLISH else 0.5,
                "timeframe_modifier": "shorter by 50%",
                "emphasis": "focus on preservation, use rallies to reduce risk"
            }
        elif regime == MarketRegime.EARLY_RECOVERY:
            return {
                "position_sizing_modifier": 1.2 if bias == TradingBias.BULLISH else 0.9,
                "timeframe_modifier": "standard",
                "emphasis": "favor early-cycle sectors and small caps if bullish"
            }
        else:
            return {
                "position_sizing_modifier": 0.8,
                "timeframe_modifier": "standard",
                "emphasis": "balanced approach due to regime uncertainty"
            }
    
    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming economic events within the specified timeframe.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of upcoming events with recommendations
        """
        today = datetime.datetime.now().date()
        cutoff_date = today + datetime.timedelta(days=days_ahead)
        
        upcoming = []
        for event in self.upcoming_events:
            if event['date'] <= cutoff_date:
                event_type = event['event_type']
                if event_type in self.events:
                    macro_event = self.events[event_type]
                    
                    # Get pre-event strategy guidance
                    pre_event_strategy = macro_event.get_strategy_adjustment("pre_event", None)
                    
                    upcoming.append({
                        "event_type": event_type,
                        "date": event['date'].isoformat(),
                        "time": event['time'],
                        "description": event['description'],
                        "importance": macro_event.importance_level,
                        "forecast": event.get('forecast', {}),
                        "days_until": (event['date'] - today).days,
                        "pre_event_guidance": self._format_strategy_recommendations(pre_event_strategy)
                    })
        
        return upcoming
    
    def get_market_regime_guidance(self) -> Dict[str, Any]:
        """
        Get comprehensive guidance based on the current market regime.
        
        Returns:
            Dict with regime analysis and recommendations
        """
        regime = self.current_regime
        
        # Base guidance for each regime
        regime_guidance = {
            MarketRegime.EXPANSION: {
                "description": "Economic growth above trend, stable/declining unemployment, moderate inflation",
                "optimal_positioning": {
                    "equity_emphasis": {
                        "sectors": ["technology", "financials", "industrials", "consumer_discretionary"],
                        "factors": ["growth", "momentum", "quality"],
                        "market_cap": "balanced, slight small/mid-cap tilt"
                    },
                    "options_strategy": {
                        "primary_focus": "income generation with moderate protection",
                        "recommended": ["short_puts", "covered_calls", "call_spreads_on_strength"]
                    },
                    "risk_parameters": {
                        "position_sizing": "100% of standard",
                        "stop_placement": "standard technical levels",
                        "sector_concentration": "allow up to 25% in strongest sectors"
                    }
                }
            },
            MarketRegime.LATE_CYCLE: {
                "description": "Growth moderating, unemployment at cyclical lows, rising inflation, restrictive monetary policy",
                "optimal_positioning": {
                    "equity_emphasis": {
                        "sectors": ["healthcare", "utilities", "staples", "quality_tech"],
                        "factors": ["quality", "low_volatility", "dividend_growth"],
                        "market_cap": "large cap bias, reduce small caps"
                    },
                    "options_strategy": {
                        "primary_focus": "balanced income and protection",
                        "recommended": ["collar_strategy", "credit_spreads", "calendar_spreads"]
                    },
                    "risk_parameters": {
                        "position_sizing": "80-90% of standard",
                        "stop_placement": "tighten by 10-15%",
                        "sector_concentration": "maximum 20% in any sector"
                    }
                }
            },
            MarketRegime.CONTRACTION: {
                "description": "Negative or below-trend growth, rising unemployment, variable inflation, accommodative monetary policy",
                "optimal_positioning": {
                    "equity_emphasis": {
                        "sectors": ["utilities", "healthcare", "consumer_staples", "quality_dividend_payers"],
                        "factors": ["minimum_volatility", "quality", "dividend_yield"],
                        "market_cap": "strong large cap bias, minimal small cap"
                    },
                    "options_strategy": {
                        "primary_focus": "significant downside protection",
                        "recommended": ["protective_puts", "put_spreads", "vix_calls"]
                    },
                    "risk_parameters": {
                        "position_sizing": "60-70% of standard",
                        "stop_placement": "wider to accommodate volatility",
                        "sector_concentration": "maximum 15% in any sector"
                    }
                }
            },
            MarketRegime.EARLY_RECOVERY: {
                "description": "Growth resuming from low base, high but stabilizing unemployment, low inflation, accommodative policy",
                "optimal_positioning": {
                    "equity_emphasis": {
                        "sectors": ["consumer_discretionary", "financials", "industrials", "materials"],
                        "factors": ["value", "small_size", "high_beta"],
                        "market_cap": "strong small/mid cap tilt"
                    },
                    "options_strategy": {
                        "primary_focus": "capitalize on volatility normalization and upside",
                        "recommended": ["call_spreads", "put_selling", "ratio_spreads"]
                    },
                    "risk_parameters": {
                        "position_sizing": "90-100% of standard",
                        "stop_placement": "standard technical levels",
                        "sector_concentration": "allow up to 25% in recovery leaders"
                    }
                }
            },
            MarketRegime.UNKNOWN: {
                "description": "Regime unclear or in transition",
                "optimal_positioning": {
                    "equity_emphasis": {
                        "sectors": ["balanced_allocation", "quality_focus"],
                        "factors": ["quality", "balanced_growth_value"],
                        "market_cap": "neutral weighting"
                    },
                    "options_strategy": {
                        "primary_focus": "balanced protection and opportunity",
                        "recommended": ["defined_risk_strategies", "balanced_iron_condors"]
                    },
                    "risk_parameters": {
                        "position_sizing": "80% of standard",
                        "stop_placement": "standard technical levels",
                        "sector_concentration": "maximum 15% in any sector"
                    }
                }
            }
        }
        
        # Get guidance for current regime
        guidance = regime_guidance.get(regime, regime_guidance[MarketRegime.UNKNOWN])
        
        # Add probabilities and indicators
        return {
            "current_regime": regime.value,
            "description": guidance["description"],
            "regime_probabilities": {k.value: round(v, 2) for k, v in self.regime_probabilities.items()},
            "guidance": guidance["optimal_positioning"],
            "key_indicators": {
                # This would include actual economic indicators in a real implementation
                "yield_curve": "positive 50bp 10Y-2Y spread",
                "unemployment_trend": "stable at cycle lows",
                "leading_economic_index": "slight decline for 2 months",
                "inflation_trend": "moderating but elevated",
                "credit_spreads": "near historical lows but widening"
            },
            "transition_risk": self._calculate_regime_transition_risk()
        }
    
    def _calculate_regime_transition_risk(self) -> Dict[str, Any]:
        """
        Calculate the risk of regime transition based on current indicators.
        
        Returns:
            Dict with transition risk assessment
        """
        # Simplified implementation - would be more sophisticated in practice
        
        # Get the highest probability regime and its probability
        current_regime = self.current_regime
        current_prob = self.regime_probabilities[current_regime]
        
        # Calculate transition risk based on how concentrated the probabilities are
        # If probabilities are evenly distributed, transition risk is high
        # If one regime dominates, transition risk is low
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in self.regime_probabilities.values())
        max_entropy = -np.log(1/len(self.regime_probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        transition_risk = normalized_entropy
        
        # Determine most likely next regime
        regime_probs_without_current = {k: v for k, v in self.regime_probabilities.items() if k != current_regime}
        next_regime = max(regime_probs_without_current, key=regime_probs_without_current.get)
        
        return {
            "transition_risk_level": transition_risk,
            "risk_category": "high" if transition_risk > 0.7 else "medium" if transition_risk > 0.4 else "low",
            "most_likely_next_regime": next_regime.value,
            "next_regime_probability": round(regime_probs_without_current[next_regime], 2),
            "estimated_timeframe": "1-3 months" if transition_risk > 0.7 else "3-6 months" if transition_risk > 0.4 else "6+ months",
            "suggested_preparations": [
                "Begin adjusting sector allocations incrementally",
                "Increase quality factor emphasis across all positions",
                "Prepare watchlists for next regime's favored sectors"
            ] if transition_risk > 0.5 else ["Maintain current regime positioning with minor adjustments"]
        }
    
    def process_event_outcome(self, event_type: str, actual_data: Dict[str, Any], expected_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process the outcome of an economic event that has occurred.
        
        Args:
            event_type: Type of economic event
            actual_data: Actual reported data
            expected_data: Expected/forecast data for comparison
            
        Returns:
            Dict with analysis and strategy recommendations
        """
        if event_type not in self.events:
            logger.warning(f"Unknown event type: {event_type}")
            return {"status": "error", "message": f"Unknown event type: {event_type}"}
        
        # Combine actual and expected data
        event_data = {}
        if expected_data:
            for key, value in expected_data.items():
                event_data[f"forecast_{key}"] = value
        
        for key, value in actual_data.items():
            event_data[key] = value
        
        # Process the event
        return self.process_economic_event(event_type, event_data)
    
    def get_vix_spike_guidance(self, current_vix: float, vix_change_percent: float) -> Dict[str, Any]:
        """
        Get guidance for handling a VIX spike.
        
        Args:
            current_vix: Current VIX level
            vix_change_percent: Percentage change in VIX
            
        Returns:
            Dict with analysis and strategy recommendations
        """
        # Determine severity of VIX spike
        severity = "none"
        if current_vix > 40 or vix_change_percent > 40:
            severity = "severe"
        elif current_vix > 30 or vix_change_percent > 30:
            severity = "high"
        elif current_vix > 25 or vix_change_percent > 20:
            severity = "moderate"
        
        # No guidance needed if no significant spike
        if severity == "none":
            return {
                "status": "normal",
                "message": "No significant VIX spike detected",
                "vix_level": current_vix,
                "vix_change_percent": vix_change_percent
            }
        
        # Use the VIX spike event if available, otherwise use simplified guidance
        if "vix_spike" in self.events:
            event = self.events["vix_spike"]
            # This would use the actual event data and structure
            # For simplicity, we'll use a basic structure here
            
            # Determine scenario based on severity
            scenario_map = {
                "moderate": "moderate_vix_spike",
                "high": "severe_vix_spike",
                "severe": "severe_vix_spike"
            }
            
            # Get appropriate strategy adjustment
            strategy_adjustment = event.strategy_adjustments.get(scenario_map[severity])
            recommendations = self._format_strategy_recommendations(strategy_adjustment)
            
        else:
            # Simplified guidance if VIX spike event not defined
            if severity == "severe":
                recommendations = {
                    "equity": {
                        "primary_action": "significant defensive positioning",
                        "position_management": [
                            "reduce overall equity exposure by 30-50%",
                            "implement portfolio hedges using inverse ETFs",
                            "focus remaining exposure on quality and minimum volatility factors"
                        ],
                        "sector_adjustments": [
                            "increase cash position to 15-25% of portfolio",
                            "rotate toward defensive sectors and dividend payers",
                            "consider gold or Treasury positions for diversification"
                        ]
                    },
                    "options": {
                        "primary_approach": "focus on capital preservation and volatility exploitation",
                        "strategies": [
                            {
                                "strategy": "put_spreads_on_indices",
                                "implementation": "focus on broader market protection",
                                "sizing": "cover 50-75% of remaining equity exposure"
                            },
                            {
                                "strategy": "vix_call_spreads",
                                "implementation": "direct volatility exposure",
                                "timeframe": "focus on 1-2 month expirations"
                            }
                        ]
                    }
                }
            elif severity == "high":
                recommendations = {
                    "equity": {
                        "primary_action": "defensive positioning",
                        "position_management": [
                            "reduce overall equity exposure by 20-30%",
                            "hedge most vulnerable positions",
                            "focus on quality and stability"
                        ],
                        "sector_adjustments": [
                            "increase defensive sector allocation",
                            "reduce high beta exposure significantly"
                        ]
                    },
                    "options": {
                        "primary_approach": "implement protection while managing cost",
                        "strategies": [
                            {
                                "strategy": "put_spreads",
                                "implementation": "on indices and vulnerable sectors",
                                "sizing": "cover 40-60% of portfolio"
                            }
                        ]
                    }
                }
            else:  # moderate
                recommendations = {
                    "equity": {
                        "primary_action": "selective de-risking",
                        "position_management": [
                            "reduce highest beta positions by 20-30%",
                            "implement tighter stops on remaining positions"
                        ],
                        "sector_adjustments": [
                            "shift 10-20% toward defensive sectors",
                            "emphasize quality factors"
                        ]
                    },
                    "options": {
                        "primary_approach": "targeted protection",
                        "strategies": [
                            {
                                "strategy": "put_spread_instead_of_puts",
                                "implementation": "define risk in high volatility environment"
                            }
                        ]
                    }
                }
        
        return {
            "status": "vix_spike",
            "severity": severity,
            "vix_level": current_vix,
            "vix_change_percent": vix_change_percent,
            "recommendations": recommendations,
            "market_impact": {
                "expected_duration": "3-8 trading days" if severity == "moderate" else "7-15 trading days",
                "typical_sectors_outperform": ["utilities", "consumer_staples", "healthcare"],
                "typical_sectors_underperform": ["financials", "technology", "small_caps"]
            }
        }
    
    def get_yield_curve_guidance(self, 
                               ten_year_yield: float, 
                               two_year_yield: float,
                               days_inverted: int = 0) -> Dict[str, Any]:
        """
        Get guidance based on the state of the yield curve.
        
        Args:
            ten_year_yield: Current 10-year Treasury yield
            two_year_yield: Current 2-year Treasury yield
            days_inverted: Number of days the curve has been inverted
            
        Returns:
            Dict with analysis and guidance
        """
        # Calculate the spread
        spread = ten_year_yield - two_year_yield
        
        # Determine curve status
        if spread > 0.50:
            curve_status = "steep"
        elif spread > 0.25:
            curve_status = "normal"
        elif spread > 0:
            curve_status = "flattening"
        elif spread > -0.25:
            curve_status = "inverted"
        else:
            curve_status = "deeply_inverted"
        
        # Determine phase based on status and duration
        if curve_status in ["inverted", "deeply_inverted"]:
            if days_inverted < 30:
                phase = "initial_inversion"
            elif days_inverted < 90:
                phase = "persistent_inversion"
            else:
                phase = "deep_inversion_with_confirming_indicators"
        else:
            phase = "normal"
        
        # Use yield curve event if available, otherwise use simplified guidance
        if "yield_curve_inversion" in self.events:
            event = self.events["yield_curve_inversion"]
            # This would use the actual event data and structure
            
            # For simplicity, using basic structure here
            if phase != "normal":
                strategy_adjustment = event.strategy_adjustments.get(phase)
                recommendations = self._format_strategy_recommendations(strategy_adjustment)
            else:
                recommendations = {}
                
        else:
            # Simplified guidance if yield curve event not defined
            if phase == "initial_inversion":
                recommendations = {
                    "equity": {
                        "primary_action": "gradual defensive shift over months, not immediate",
                        "position_management": [
                            "begin reducing cyclical exposure by 5-10%",
                            "increase quality factor emphasis across sectors"
                        ],
                        "sector_adjustments": [
                            "maintain normal allocations initially but increase quality screening",
                            "begin building watchlists of defensive names for later rotation"
                        ]
                    },
                    "options": {
                        "primary_approach": "gradual implementation of longer-term protection",
                        "strategies": [
                            {
                                "strategy": "long-dated_puts",
                                "implementation": "on indices 9-15 months forward",
                                "sizing": "small initial position with plan to scale"
                            }
                        ]
                    }
                }
            elif phase == "persistent_inversion":
                recommendations = {
                    "equity": {
                        "primary_action": "more pronounced defensive positioning",
                        "position_management": [
                            "reduce overall equity exposure by 10-20%",
                            "increase cash position to 10-15% of portfolio"
                        ],
                        "sector_adjustments": [
                            "begin rotating 20-30% of portfolio toward defensive sectors",
                            "reduce small cap and high-beta exposure by 30-40%"
                        ]
                    },
                    "options": {
                        "primary_approach": "implement more comprehensive protection",
                        "strategies": [
                            {
                                "strategy": "collar_strategy",
                                "implementation": "for core equity holdings",
                                "benefit": "define downside while allowing some upside participation"
                            }
                        ]
                    }
                }
            elif phase == "deep_inversion_with_confirming_indicators":
                recommendations = {
                    "equity": {
                        "primary_action": "significant defensive positioning",
                        "position_management": [
                            "reduce overall equity exposure by 30-40%",
                            "increase cash position to 20-30% of portfolio"
                        ],
                        "sector_adjustments": [
                            "shift majority of remaining equity exposure to defensive sectors",
                            "focus on recession-resistant business models"
                        ]
                    },
                    "options": {
                        "primary_approach": "comprehensive downside protection",
                        "strategies": [
                            {
                                "strategy": "long_put_spreads",
                                "implementation": "on indices and vulnerable sectors",
                                "sizing": "cover 50-75% of remaining equity exposure"
                            }
                        ]
                    }
                }
            else:
                recommendations = {}
        
        return {
            "status": "yield_curve_analysis",
            "current_spread": spread,
            "curve_status": curve_status,
            "days_inverted": days_inverted,
            "phase": phase,
            "historical_context": {
                "typical_recession_lag": "6-24 months after initial inversion",
                "market_impact": "Equities often continue higher for 6-18 months after initial inversion"
            },
            "recommendations": recommendations
        }
    
    def update_economic_indicators(self, indicators: Dict[str, Any]):
        """
        Update stored economic indicators.
        
        Args:
            indicators: Dictionary of economic indicators and values
        """
        # Update stored indicators
        for key, value in indicators.items():
            self.economic_indicators[key] = value
        
        # Re-evaluate market regime with new data
        self._determine_current_regime()
        
        logger.info(f"Economic indicators updated, current regime: {self.current_regime.value}")
        
    def get_earnings_season_guidance(self, 
                                   current_phase: str,
                                   upcoming_reports: List[str] = None,
                                   recent_surprises: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get guidance for navigating earnings season.
        
        Args:
            current_phase: Current phase of earnings season 
                           ('pre', 'early', 'peak', 'late', 'post')
            upcoming_reports: List of tickers with upcoming reports
            recent_surprises: Dict mapping tickers to EPS surprise percentages
            
        Returns:
            Dict with analysis and recommendations
        """
        # Use earnings season event if available, otherwise use simplified guidance
        if "earnings_season" in self.events:
            event = self.events["earnings_season"]
            # This would use the actual event data and structure
            
            # Map phase to appropriate strategy adjustment
            phase_map = {
                "pre": "pre_release",
                "early": "post_release",  # Would depend on early results
                "peak": "post_release",
                "late": "post_release",
                "post": "post_release"
            }
            
            # Determine bias based on recent surprises
            bias = TradingBias.NEUTRAL
            if recent_surprises:
                avg_surprise = sum(recent_surprises.values()) / len(recent_surprises)
                if avg_surprise > 5:
                    bias = TradingBias.BULLISH
                elif avg_surprise < -5:
                    bias = TradingBias.BEARISH
            
            # Get appropriate scenario
            if bias == TradingBias.BULLISH:
                scenario = "bullish_scenarios"
            elif bias == TradingBias.BEARISH:
                scenario = "bearish_scenarios"
            else:
                scenario = "neutral_scenario"
            
            # Get strategy adjustment
            strategy_adjustment = event.get_strategy_adjustment(phase_map[current_phase], scenario)
            recommendations = self._format_strategy_recommendations(strategy_adjustment)
            
        else:
            # Simplified guidance if earnings event not defined
            if current_phase == "pre":
                recommendations = {
                    "equity": {
                        "primary_action": "reduce concentration risk before heavy reporting periods",
                        "position_management": [
                            "trim positions in companies reporting soon by 20-30%",
                            "prioritize trimming positions with elevated implied volatility"
                        ],
                        "sector_adjustments": [
                            "reduce exposure to sectors showing negative early reports",
                            "watch for emerging themes and rotate accordingly"
                        ]
                    },
                    "options": {
                        "primary_approach": "selective earnings volatility strategies",
                        "strategies": [
                            {
                                "strategy": "reduce_short_options_exposure",
                                "implementation": "particularly naked positions",
                                "timing": "7-10 days before expected heavy reporting periods"
                            }
                        ]
                    }
                }
            else:
                # For other phases, guidance would depend on results so far
                sentiment = "neutral"
                if recent_surprises:
                    avg_surprise = sum(recent_surprises.values()) / len(recent_surprises)
                    if avg_surprise > 5:
                        sentiment = "positive"
                    elif avg_surprise < -5:
                        sentiment = "negative"
                
                if sentiment == "positive":
                    recommendations = {
                        "equity": {
                            "primary_action": "focus on strongest reports with increasing guidance",
                            "position_management": [
                                "enter positions after post-earnings IV crush",
                                "focus on stocks beating on revenue and raising guidance"
                            ]
                        },
                        "options": {
                            "primary_approach": "capitalize on positive momentum",
                            "strategies": [
                                {
                                    "strategy": "call_spreads",
                                    "implementation": "on strongest sectors showing momentum"
                                }
                            ]
                        }
                    }
                elif sentiment == "negative":
                    recommendations = {
                        "equity": {
                            "primary_action": "identify structural weakness vs. temporary setbacks",
                            "position_management": [
                                "focus on stocks breaking key technical levels post-earnings",
                                "watch for multiple companies citing similar problems"
                            ]
                        },
                        "options": {
                            "primary_approach": "position for continued weakness",
                            "strategies": [
                                {
                                    "strategy": "put_spreads",
                                    "implementation": "on sectors showing consistent weakness"
                                }
                            ]
                        }
                    }
                else:
                    recommendations = {
                        "equity": {
                            "primary_action": "selective approach based on individual results",
                            "position_management": [
                                "focus on company-specific performance rather than broad themes",
                                "watch for emerging sector trends as more reports come in"
                            ]
                        },
                        "options": {
                            "primary_approach": "opportunistic, company-specific strategies",
                            "strategies": [
                                {
                                    "strategy": "post_earnings_premium_selling",
                                    "implementation": "after IV crush on quality names with predictable outcomes"
                                }
                            ]
                        }
                    }
        
        return {
            "status": "earnings_season_guidance",
            "current_phase": current_phase,
            "sentiment": "positive" if recent_surprises and sum(recent_surprises.values()) / len(recent_surprises) > 5 else 
                         "negative" if recent_surprises and sum(recent_surprises.values()) / len(recent_surprises) < -5 else 
                         "neutral",
            "upcoming_reports_count": len(upcoming_reports) if upcoming_reports else 0,
            "recommendations": recommendations,
            "specific_guidance": self._get_specific_earnings_guidance(upcoming_reports) if upcoming_reports else {}
        }
    
    def _get_specific_earnings_guidance(self, upcoming_reports: List[str]) -> Dict[str, Any]:
        """Generate guidance for specific earnings reports."""
        # Implementation would be more sophisticated in production
        return {
            "tickers": upcoming_reports,
            "high_volatility_expected": [ticker for ticker in upcoming_reports if ticker in ["AAPL", "AMZN", "NFLX"]],
            "approach": "Use options strategies with defined risk during these reports"
        }
        
    def get_sector_rotation_guidance(self, cycle_phase: str = None, ticker: str = None) -> Dict[str, Any]:
        """
        Get sector rotation guidance based on the current or specified economic cycle phase.
        
        Args:
            cycle_phase: Optional specific cycle phase to get guidance for
            ticker: Optional ticker to get specific guidance for
            
        Returns:
            Dict with sector rotation guidance
        """
        # Determine current cycle phase if not specified
        if not cycle_phase:
            cycle_determination = self.get_economic_cycle_determination()
            cycle_phase = cycle_determination["current_phase"]
        
        # Get sector rotation framework from config
        sector_rotation_framework = self.config.get("sector_rotation", {})
        
        # If sector rotation isn't configured, return default response
        if not sector_rotation_framework:
        return {
                "status": "not_configured",
                "message": "Sector rotation framework is not configured"
            }
        
        # Get guidance for the specified cycle phase
        phase_guidance = sector_rotation_framework.get(cycle_phase, {})
        
        # If no guidance for this phase, return error
        if not phase_guidance:
            return {
                "status": "invalid_phase",
                "message": f"No guidance available for cycle phase: {cycle_phase}",
                "available_phases": list(sector_rotation_framework.keys())
            }
        
        # Build response with the guidance
        response = {
            "cycle_phase": cycle_phase,
            "framework_version": sector_rotation_framework.get("framework_version", "1.0.0"),
            "last_updated": sector_rotation_framework.get("last_updated", "N/A"),
            "phase_description": phase_guidance.get("description", ""),
            "typical_duration": phase_guidance.get("typical_duration", ""),
            "macro_signals": phase_guidance.get("macro_signals", {}),
            "favored_sectors": phase_guidance.get("favored_sectors", {}),
            "strategies": phase_guidance.get("strategies", {}),
            "implementation_guidance": phase_guidance.get("implementation_guidance", {})
        }
        
        # If ticker provided, add ticker-specific guidance
        if ticker:
            ticker_sector = self._get_ticker_sector(ticker)
            response["ticker_guidance"] = self._get_ticker_rotation_guidance(
                ticker, 
                ticker_sector, 
                cycle_phase, 
                phase_guidance
            )
        
        # Add historical performance data if available
        if "historical_performance" in phase_guidance:
            response["historical_performance"] = phase_guidance["historical_performance"]
        
        # Add bot implementation specifics if available
        if "bot_implementation" in phase_guidance:
            response["bot_implementation"] = phase_guidance["bot_implementation"]
        
        return response
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """
        Get the sector for a ticker.
        This would typically use a more robust sector database.
        """
        # Simplified sector mapping for demo purposes
        sector_mapping = {
            # Technology
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "NVDA": "technology", "ADBE": "technology", "CRM": "technology",
            
            # Financial
            "JPM": "financials", "BAC": "financials", "WFC": "financials",
            "GS": "financials", "MS": "financials", "V": "financials",
            
            # Healthcare
            "JNJ": "healthcare", "PFE": "healthcare", "MRK": "healthcare",
            "UNH": "healthcare", "ABT": "healthcare", "MDT": "healthcare",
            
            # Consumer Discretionary
            "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary", 
            "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
            "NKE": "consumer_discretionary", "SBUX": "consumer_discretionary",
            
            # Consumer Staples
            "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
            "WMT": "consumer_staples", "COST": "consumer_staples", "CL": "consumer_staples",
            
            # Energy
            "XOM": "energy", "CVX": "energy", "COP": "energy",
            "SLB": "energy", "EOG": "energy", "PSX": "energy",
            
            # Utilities
            "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
            "D": "utilities", "AEP": "utilities", "EXC": "utilities",
            
            # Industrials
            "HON": "industrials", "UNP": "industrials", "BA": "industrials",
            "CAT": "industrials", "GE": "industrials", "MMM": "industrials",
            
            # Materials
            "LIN": "materials", "ECL": "materials", "APD": "materials",
            "SHW": "materials", "DD": "materials", "NEM": "materials",
            
            # Real Estate
            "AMT": "real_estate", "WELL": "real_estate", "SPG": "real_estate",
            "PSA": "real_estate", "O": "real_estate", "EQIX": "real_estate",
            
            # Communication Services
            "META": "communication_services", "GOOG": "communication_services", 
            "NFLX": "communication_services", "TMUS": "communication_services",
            "VZ": "communication_services", "T": "communication_services"
        }
        
        return sector_mapping.get(ticker.upper(), "unknown")
    
    def _get_ticker_rotation_guidance(self, ticker: str, sector: str, cycle_phase: str, phase_guidance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get sector rotation guidance specific to a ticker.
        
        Args:
            ticker: Ticker symbol
            sector: Sector of the ticker
            cycle_phase: Current economic cycle phase
            phase_guidance: Guidance for the current phase
            
        Returns:
            Dict with ticker-specific rotation guidance
        """
        # Default response
        response = {
            "ticker": ticker,
            "sector": sector,
            "sector_classification": "unknown"
        }
        
        # Get favored sectors
        favored_sectors = phase_guidance.get("favored_sectors", {})
        
        # Check if this sector is in primary favored sectors
        primary_sectors = [s["sector"] for s in favored_sectors.get("primary_sectors", [])]
        secondary_sectors = [s["sector"] for s in favored_sectors.get("secondary_sectors", [])]
        sectors_to_avoid = [s["sector"] for s in favored_sectors.get("sectors_to_avoid", [])]
        
        if sector in primary_sectors:
            response["sector_classification"] = "primary_favored"
            # Find the sector data
            for s in favored_sectors.get("primary_sectors", []):
                if s["sector"] == sector:
                    response["sector_data"] = s
                    break
        elif sector in secondary_sectors:
            response["sector_classification"] = "secondary_favored"
            # Find the sector data
            for s in favored_sectors.get("secondary_sectors", []):
                if s["sector"] == sector:
                    response["sector_data"] = s
                    break
        elif sector in sectors_to_avoid:
            response["sector_classification"] = "avoid"
            # Find the sector data
            for s in favored_sectors.get("sectors_to_avoid", []):
                if s["sector"] == sector:
                    response["sector_data"] = s
                    break
        else:
            response["sector_classification"] = "neutral"
        
        # Get strategies by sector
        strategies = phase_guidance.get("strategies", {})
        appropriate_equity_strategies = []
        appropriate_options_strategies = []
        
        # Find equity strategies appropriate for this sector
        for strategy in strategies.get("equity_strategies", []):
            focus = strategy.get("implementation", {}).get("primary_focus", "").lower()
            if sector in focus or response["sector_classification"] in focus:
                appropriate_equity_strategies.append(strategy)
        
        # Find options strategies appropriate for this sector
        for strategy in strategies.get("options_strategies", []):
            focus = strategy.get("implementation", {}).get("primary_focus", "").lower()
            if sector in focus or response["sector_classification"] in focus:
                appropriate_options_strategies.append(strategy)
        
        # Add strategies to response
        if appropriate_equity_strategies:
            response["recommended_equity_strategies"] = appropriate_equity_strategies
        
        if appropriate_options_strategies:
            response["recommended_options_strategies"] = appropriate_options_strategies
            
        # Check historical performance ranking
        historical_performance = phase_guidance.get("historical_performance", {})
        sector_ranking = historical_performance.get("sector_performance_ranking", [])
        
        for rank, sector_data in enumerate(sector_ranking):
            if sector_data.get("sector") == sector:
                response["historical_performance"] = {
                    "rank": rank + 1,
                    "total_sectors": len(sector_ranking),
                    "average_outperformance": sector_data.get("average_outperformance", "N/A")
                }
                break
        
        return response
    
    def get_economic_cycle_determination(self) -> Dict[str, Any]:
        """
        Determine the current economic cycle phase based on macro indicators.
        
        Returns:
            Dict with the current economic cycle determination
        """
        # In a production system, this would use real economic data and ML models
        # For now, we'll map market regime to economic cycle with a simplification
        
        # Get current market regime
        regime = self.current_regime
        
        # Map market regime to economic cycle
        cycle_mapping = {
            MarketRegime.EXPANSION: "early_expansion",
            MarketRegime.LATE_CYCLE: "late_cycle",
            MarketRegime.CONTRACTION: "recession",
            MarketRegime.EARLY_RECOVERY: "recovery",
            MarketRegime.UNKNOWN: "mid_cycle"  # Default to mid cycle if unknown
        }
        
        current_phase = cycle_mapping.get(regime, "mid_cycle")
        
        # Generate confidence scores for each phase
        # In production, this would be based on multiple indicators and ML model output
        confidences = {}
        
        # Base confidences on regime probabilities
        for regime_type, cycle_phase in cycle_mapping.items():
            # Get the probability for this regime
            prob = self.regime_probabilities.get(regime_type, 0)
            
            # Only include in confidences if probability > 0
            if prob > 0:
                confidences[cycle_phase] = prob
        
        # If we're missing any phases, add with low confidence
        for phase in ["early_expansion", "mid_cycle", "late_cycle", "recession", "recovery"]:
            if phase not in confidences:
                confidences[phase] = 0.05
        
        # Ensure confidences sum to 1
        total = sum(confidences.values())
        confidences = {k: v/total for k, v in confidences.items()}
        
        # Get supporting indicators
        supporting_indicators = self._get_cycle_supporting_indicators(current_phase)
        
        return {
            "current_phase": current_phase,
            "phase_confidences": confidences,
            "supporting_indicators": supporting_indicators,
            "market_regime": regime.value,
            "determination_timestamp": datetime.datetime.now().isoformat()
        }
    
    def _get_cycle_supporting_indicators(self, cycle_phase: str) -> List[Dict[str, Any]]:
        """
        Get indicators supporting the current cycle determination.
        
        Args:
            cycle_phase: Current economic cycle phase
            
        Returns:
            List of supporting indicators with descriptions
        """
        # In production, this would return actual indicator values
        # For demonstration, we'll return hypothetical indicators
        
        if cycle_phase == "early_expansion":
            return [
                {"indicator": "GDP growth", "value": "3.2%", "description": "Accelerating growth from low base"},
                {"indicator": "Unemployment", "value": "5.8%", "description": "Declining from peak"},
                {"indicator": "Yield curve", "value": "1.5%", "description": "Steepening from previously flat state"}
            ]
        elif cycle_phase == "mid_cycle":
            return [
                {"indicator": "GDP growth", "value": "2.5%", "description": "Stable growth at trend"},
                {"indicator": "Unemployment", "value": "4.5%", "description": "Low and stable"},
                {"indicator": "Capacity utilization", "value": "78%", "description": "Rising toward long-term averages"}
            ]
        elif cycle_phase == "late_cycle":
            return [
                {"indicator": "Inflation", "value": "3.5%", "description": "Rising above central bank target"},
                {"indicator": "Yield curve", "value": "0.3%", "description": "Flattening trend"},
                {"indicator": "Capacity utilization", "value": "82%", "description": "Near full capacity"}
            ]
        elif cycle_phase == "recession":
            return [
                {"indicator": "GDP growth", "value": "-1.2%", "description": "Negative for 2+ quarters"},
                {"indicator": "Unemployment", "value": "6.5%", "description": "Rising from cycle lows"},
                {"indicator": "Consumer sentiment", "value": "72", "description": "Sharp decline to recessionary levels"}
            ]
        elif cycle_phase == "recovery":
            return [
                {"indicator": "Leading indicators", "value": "+0.8%", "description": "Bottoming and turning positive"},
                {"indicator": "Credit spreads", "value": "4.2%", "description": "Narrowing from recessionary wides"},
                {"indicator": "Manufacturing PMI", "value": "48.5", "description": "Rising from low levels, approaching 50"}
            ]
        else:
            return []
    
    def update_sector_rotation_framework(self, framework: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the sector rotation framework with new data.
        
        Args:
            framework: New sector rotation framework data
            
        Returns:
            Dict with update results
        """
        try:
            # Validate framework using the loader
            loader = SectorRotationLoader()
            validated_framework = loader.load_from_data(framework)
            
            # Update the framework in config
            self.config["sector_rotation"] = validated_framework
            
            # If sector_rotation_path is configured, save to file
            sector_rotation_path = self.config.get("sector_rotation_path")
            if sector_rotation_path:
                loader.save_to_file(sector_rotation_path)
                logger.info(f"Saved updated sector rotation framework to {sector_rotation_path}")
            
            # Return success message with phase count
            phases = [key for key in validated_framework.keys() if key not in ["framework_version", "last_updated", "meta_data", "advanced_identification_framework", "multi_timeframe_application", "implementation_framework", "bot_specific_implementation"]]
            
            return {
                "status": "success",
                "phases_updated": len(phases),
                "phases": phases,
                "framework_version": validated_framework.get("framework_version", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error updating sector rotation framework: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_seasonality_guidance(self, ticker: str = None, specific_month: str = None) -> Dict[str, Any]:
        """
        Get seasonality-based guidance for the current month or a specific month.
        
        Args:
            ticker: Optional ticker symbol to get specific guidance for
            specific_month: Optional month name to get guidance for (default: current month)
            
        Returns:
            Dict with seasonality guidance
        """
        # Initialize loader if needed
        loader = None
        if "seasonality_insights" in self.config and isinstance(self.config["seasonality_insights"], dict):
            loader = SeasonalityInsightsLoader()
            loader.load_from_data(self.config["seasonality_insights"])
        else:
            seasonality_path = self.config.get("seasonality_insights_path")
            if seasonality_path and os.path.exists(seasonality_path):
                loader = SeasonalityInsightsLoader(seasonality_path)
        
        if not loader or not loader.get_framework():
            return {
                "status": "not_configured",
                "message": "Seasonality insights framework is not configured"
            }
        
        # Get current month if not specified
        if not specific_month:
            import datetime
            specific_month = datetime.datetime.now().strftime("%B")
        
        # Get monthly pattern
        monthly_pattern = loader.get_monthly_pattern(specific_month)
        
        if not monthly_pattern:
            return {
                "status": "invalid_month",
                "message": f"No guidance available for month: {specific_month}",
                "available_months": [pattern.get("month") for pattern in loader.get_framework().get("seasonality_insights", {}).get("monthly_patterns", [])]
            }
        
        # Get active recurring patterns
        active_recurring_patterns = loader.get_active_recurring_patterns()
        
        # Build response
        response = {
            "status": "success",
            "month": specific_month,
            "name": monthly_pattern.get("name", f"{specific_month} Seasonality"),
            "expected_bias": self._get_monthly_bias(monthly_pattern),
            "primary_asset_classes": monthly_pattern.get("primary_asset_classes", []),
            "specific_patterns": self._extract_specific_patterns(monthly_pattern),
            "trading_strategies": monthly_pattern.get("trading_strategies", {}),
            "market_dynamics": monthly_pattern.get("market_dynamics", {}),
            "risk_factors": monthly_pattern.get("risk_factors", {}),
            "active_recurring_patterns": [{
                "pattern": pattern.get("pattern"),
                "frequency": pattern.get("frequency"),
                "expected_bias": self._get_recurring_pattern_bias(pattern)
            } for pattern in active_recurring_patterns]
        }
        
        # Add framework version if available
        insights = loader.get_framework().get("seasonality_insights", {})
        if "framework_version" in insights:
            response["framework_version"] = insights["framework_version"]
        if "last_updated" in insights:
            response["last_updated"] = insights["last_updated"]
        
        # Add ticker-specific guidance if provided
        if ticker:
            response["ticker_guidance"] = self._get_ticker_seasonality_guidance(
                ticker, 
                monthly_pattern,
                active_recurring_patterns
            )
        
        # Add composite seasonality score if available
        if "seasonality_framework" in insights and "composite_seasonality_score" in insights["seasonality_framework"]:
            response["composite_score"] = self._calculate_composite_seasonality_score(
                monthly_pattern,
                active_recurring_patterns
            )
        
        return response
    
    def _get_monthly_bias(self, monthly_pattern: Dict[str, Any]) -> str:
        """Extract the overall expected bias for a monthly pattern."""
        # First check if we have an explicit bias at the top level
        if "expected_bias" in monthly_pattern:
            return monthly_pattern["expected_bias"]
        
        # Otherwise, try to extract from the primary asset classes
        primary_asset_classes = monthly_pattern.get("primary_asset_classes", [])
        for asset_class in primary_asset_classes:
            if asset_class.get("asset_class") == "Equities" and "expected_bias" in asset_class:
                return asset_class["expected_bias"]
        
        # Default
        return "Neutral"
    
    def _get_recurring_pattern_bias(self, recurring_pattern: Dict[str, Any]) -> str:
        """Extract the overall expected bias for a recurring pattern."""
        # Try to extract from the primary asset classes
        primary_asset_classes = recurring_pattern.get("primary_asset_classes", [])
        for asset_class in primary_asset_classes:
            if "expected_bias" in asset_class:
                return asset_class["expected_bias"]
        
        # Default
        return "Mixed"
    
    def _extract_specific_patterns(self, monthly_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract specific patterns from a monthly pattern."""
        patterns = []
        
        # Extract from primary asset classes
        primary_asset_classes = monthly_pattern.get("primary_asset_classes", [])
        for asset_class in primary_asset_classes:
            for pattern in asset_class.get("specific_patterns", []):
                patterns.append({
                    "pattern": pattern.get("pattern", ""),
                    "description": pattern.get("description", ""),
                    "timing": pattern.get("timing", ""),
                    "historical_reliability": pattern.get("historical_reliability", ""),
                    "asset_class": asset_class.get("asset_class", "")
                })
        
        return patterns
    
    def _get_ticker_seasonality_guidance(self, ticker: str, monthly_pattern: Dict[str, Any], recurring_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get seasonality guidance specific to a ticker.
        
        Args:
            ticker: Ticker symbol
            monthly_pattern: Current month's pattern data
            recurring_patterns: List of active recurring patterns
            
        Returns:
            Dict with ticker-specific seasonality guidance
        """
        # Get sector for the ticker
        sector = self._get_ticker_sector(ticker)
        
        # Get sector seasonal performance 
        sector_performance = self._get_sector_seasonal_performance(sector, monthly_pattern)
        
        # Get appropriate strategies
        equity_strategies = self._get_ticker_appropriate_strategies(ticker, sector, monthly_pattern, "equity_strategies")
        options_strategies = self._get_ticker_appropriate_strategies(ticker, sector, monthly_pattern, "options_strategies")
        
        # Combine with recurring pattern strategies
        for pattern in recurring_patterns:
            equity_pattern_strategies = self._get_ticker_appropriate_strategies(ticker, sector, pattern, "equity_strategies")
            options_pattern_strategies = self._get_ticker_appropriate_strategies(ticker, sector, pattern, "options_strategies")
            
            equity_strategies.extend(equity_pattern_strategies)
            options_strategies.extend(options_pattern_strategies)
        
        return {
            "ticker": ticker,
            "sector": sector,
            "sector_performance": sector_performance,
            "recommended_equity_strategies": equity_strategies,
            "recommended_options_strategies": options_strategies,
            "position_sizing_guidance": self._get_seasonal_position_sizing(ticker, sector, monthly_pattern, recurring_patterns)
        }
    
    def _get_sector_seasonal_performance(self, sector: str, monthly_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Get seasonal performance for a specific sector based on monthly pattern."""
        result = {
            "sector": sector,
            "relative_strength": "neutral"
        }
        
        # Check all asset classes for sector information
        for asset_class in monthly_pattern.get("primary_asset_classes", []):
            if asset_class.get("asset_class") == "Equities":
                historical = asset_class.get("historical_performance", {})
                
                # Check best performing sectors
                best_sectors = historical.get("best_performing_sectors", [])
                for i, best_sector in enumerate(best_sectors):
                    if sector.lower() in best_sector.lower():
                        result["relative_strength"] = "strong"
                        result["ranking"] = f"Top {i+1}"
                        break
                
                # Check worst performing sectors
                worst_sectors = historical.get("worst_performing_sectors", [])
                for i, worst_sector in enumerate(worst_sectors):
                    if sector.lower() in worst_sector.lower():
                        result["relative_strength"] = "weak"
                        result["ranking"] = f"Bottom {i+1}"
                        break
                
                # Add small vs large info if relevant
                if "small_vs_large" in historical:
                    result["small_vs_large"] = historical["small_vs_large"]
        
        return result
    
    def _get_ticker_appropriate_strategies(self, ticker: str, sector: str, pattern: Dict[str, Any], strategy_type: str) -> List[Dict[str, Any]]:
        """Get strategies appropriate for a specific ticker based on the pattern."""
        appropriate_strategies = []
        
        # Get strategies from the pattern
        strategies = pattern.get("trading_strategies", {}).get(strategy_type, [])
        
        for strategy in strategies:
            implementation = strategy.get("implementation", {})
            
            # Check if this strategy is appropriate for this ticker/sector
            target_securities = implementation.get("target_securities", "")
            selection_criteria = implementation.get("selection_criteria", "")
            
            # Simple check - this would be more sophisticated in practice
            if (ticker.upper() in target_securities.upper() or 
                sector.lower() in target_securities.lower() or
                "ETF" in target_securities and ticker.upper() in ["SPY", "QQQ", "IWM"] or
                "broad market" in target_securities.lower()):
                
                appropriate_strategies.append({
                    "pattern_name": pattern.get("name", pattern.get("pattern", "")),
                    "strategy": strategy.get("strategy", ""),
                    "implementation": implementation,
                    "is_recurring": "pattern" in pattern and "frequency" in pattern
                })
        
        return appropriate_strategies
    
    def _get_seasonal_position_sizing(self, ticker: str, sector: str, monthly_pattern: Dict[str, Any], recurring_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get position sizing guidance based on seasonality."""
        # Default position sizing is 1.0 (standard)
        position_sizing = 1.0
        rationale = []
        
        # Adjust based on monthly pattern
        monthly_bias = self._get_monthly_bias(monthly_pattern)
        if monthly_bias == "Bullish":
            position_sizing *= 1.1
            rationale.append(f"Bullish seasonal bias for {monthly_pattern.get('month')}")
        elif monthly_bias == "Bearish":
            position_sizing *= 0.9
            rationale.append(f"Bearish seasonal bias for {monthly_pattern.get('month')}")
            
        # Check sector performance
        sector_performance = self._get_sector_seasonal_performance(sector, monthly_pattern)
        if sector_performance["relative_strength"] == "strong":
            position_sizing *= 1.1
            rationale.append(f"{sector} tends to outperform in {monthly_pattern.get('month')}")
        elif sector_performance["relative_strength"] == "weak":
            position_sizing *= 0.9
            rationale.append(f"{sector} tends to underperform in {monthly_pattern.get('month')}")
            
        # Consider recurring patterns
        for pattern in recurring_patterns:
            pattern_bias = self._get_recurring_pattern_bias(pattern)
            pattern_name = pattern.get("pattern", "Unknown pattern")
            
            if pattern_bias == "Bullish" or pattern_bias == "Rising":
                position_sizing *= 1.05
                rationale.append(f"Bullish bias from {pattern_name}")
            elif pattern_bias == "Bearish" or pattern_bias == "Declining":
                position_sizing *= 0.95
                rationale.append(f"Bearish bias from {pattern_name}")
            elif pattern_bias == "Volatile":
                position_sizing *= 0.9
                rationale.append(f"Increased volatility expected from {pattern_name}")
        
        # Cap the adjustments within reasonable bounds
        position_sizing = max(0.6, min(1.3, position_sizing))
        
        return {
            "position_sizing_factor": round(position_sizing, 2),
            "rationale": rationale
        }
    
    def _calculate_composite_seasonality_score(self, monthly_pattern: Dict[str, Any], recurring_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate a composite seasonality score (0-100) based on current patterns."""
        # This would be more sophisticated in practice
        
        # Start with neutral score
        score = 50
        factors = []
        
        # Adjust for monthly pattern
        monthly_bias = self._get_monthly_bias(monthly_pattern)
        if monthly_bias == "Bullish":
            score += 15
            factors.append(f"Bullish bias for {monthly_pattern.get('month')}")
        elif monthly_bias == "Bearish":
            score -= 15
            factors.append(f"Bearish bias for {monthly_pattern.get('month')}")
        
        # Adjust for historical returns
        for asset_class in monthly_pattern.get("primary_asset_classes", []):
            if asset_class.get("asset_class") == "Equities":
                historical = asset_class.get("historical_performance", {})
                overall = historical.get("overall", "")
                
                if "average return" in overall:
                    try:
                        # Extract return value from string like "SPY average return: +1.7% (1950-2024)"
                        import re
                        match = re.search(r"([+-]\d+\.\d+)%", overall)
                        if match:
                            avg_return = float(match.group(1))
                            score += avg_return * 5  # Adjust score proportionally to return
                            factors.append(f"Historical average return: {avg_return}%")
                    except:
                        pass
        
        # Adjust for specific patterns
        specific_patterns = self._extract_specific_patterns(monthly_pattern)
        for pattern in specific_patterns:
            reliability = pattern.get("historical_reliability", "")
            
            try:
                # Extract percentage from string like "75% success rate over past 50 years"
                import re
                match = re.search(r"(\d+)%", reliability)
                if match:
                    reliability_pct = int(match.group(1))
                    
                    # Add more points for highly reliable patterns
                    if reliability_pct > 75:
                        score += 5
                        factors.append(f"Highly reliable pattern: {pattern.get('pattern')} ({reliability_pct}%)")
                    elif reliability_pct > 65:
                        score += 3
                        factors.append(f"Reliable pattern: {pattern.get('pattern')} ({reliability_pct}%)")
            except:
                pass
        
        # Adjust for recurring patterns
        for pattern in recurring_patterns:
            pattern_bias = self._get_recurring_pattern_bias(pattern)
            pattern_name = pattern.get("pattern", "")
            
            if pattern_bias == "Bullish":
                score += 7
                factors.append(f"Bullish recurring pattern: {pattern_name}")
            elif pattern_bias == "Bearish":
                score -= 7
                factors.append(f"Bearish recurring pattern: {pattern_name}")
            elif pattern_bias == "Volatile":
                score -= 3
                factors.append(f"Volatile recurring pattern: {pattern_name}")
        
        # Cap the score within 0-100 range
        score = max(0, min(100, score))
        
        # Determine interpretation
        interpretation = "Neutral"
        if score >= 90:
            interpretation = "Extremely Strong Seasonal Bias"
        elif score >= 75:
            interpretation = "Strong Seasonal Bias"
        elif score >= 60:
            interpretation = "Moderate Seasonal Bias"
        elif score <= 10:
            interpretation = "Extremely Weak Seasonal Bias"
        elif score <= 25:
            interpretation = "Weak Seasonal Bias"
        elif score <= 40:
            interpretation = "Slight Seasonal Headwind"
        
        return {
            "score": round(score),
            "interpretation": interpretation,
            "factors": factors
        }
        
    def update_seasonality_insights_framework(self, framework: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the seasonality insights framework with new data.
        
        Args:
            framework: New seasonality insights framework data
            
        Returns:
            Dict with update results
        """
        try:
            # Validate framework using the loader
            loader = SeasonalityInsightsLoader()
            validated_framework = loader.load_from_data(framework)
            
            # Update the framework in config
            self.config["seasonality_insights"] = validated_framework
            
            # If seasonality_insights_path is configured, save to file
            seasonality_path = self.config.get("seasonality_insights_path")
            if seasonality_path:
                loader.save_to_file(seasonality_path)
                logger.info(f"Saved updated seasonality insights framework to {seasonality_path}")
            
            # Return success message
            insights = validated_framework.get("seasonality_insights", {})
            monthly_patterns = insights.get("monthly_patterns", [])
            recurring_patterns = insights.get("recurring_patterns", [])
            
            return {
                "status": "success",
                "months_updated": len(monthly_patterns),
                "months": [pattern.get("month") for pattern in monthly_patterns],
                "recurring_patterns_updated": len(recurring_patterns),
                "recurring_patterns": [pattern.get("pattern") for pattern in recurring_patterns],
                "framework_version": insights.get("framework_version", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error updating seasonality insights framework: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
        } 