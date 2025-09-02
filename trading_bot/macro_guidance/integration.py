"""
Macro Guidance Integration Module

This module connects the macro guidance engine with the overall trading bot
framework, providing a unified interface for strategy recommendations.
"""

import logging
import datetime
from typing import Dict, List, Any, Optional

from .macro_engine import MacroGuidanceEngine
from .config import DEFAULT_MACRO_CONFIG

logger = logging.getLogger(__name__)

class MacroGuidanceIntegration:
    """
    Integration layer between the macro guidance engine and the main trading bot.
    This class provides methods to enhance trading decisions with macro insights.
    """
    
    def __init__(self, config=None, context_engine=None, advanced_integration=None):
        """
        Initialize the macro guidance integration.
        
        Args:
            config: Overall trading bot configuration
            context_engine: Reference to the context engine
            advanced_integration: Reference to the advanced integration layer
        """
        self.config = config or {}
        self.context_engine = context_engine
        self.advanced_integration = advanced_integration
        
        # Extract macro guidance specific config or use defaults
        macro_config = self.config.get("macro_guidance", DEFAULT_MACRO_CONFIG)
        
        # Initialize macro guidance engine
        self.macro_engine = MacroGuidanceEngine(macro_config)
        
        # Track if we're approaching important economic events
        self.upcoming_events = []
        self.update_upcoming_events()
        
        logger.info("Macro Guidance Integration initialized")
    
    def update_upcoming_events(self):
        """Update the list of upcoming economic events."""
        self.upcoming_events = self.macro_engine.get_upcoming_events(
            days_ahead=self.config.get("macro_guidance", {}).get("economic_calendar", {}).get("look_ahead_days", 14)
        )
        
        # Sort by date and importance
        self.upcoming_events.sort(key=lambda x: (x["date"], -x["importance"]))
        
        logger.info(f"Updated upcoming events: {len(self.upcoming_events)} events in the next 14 days")
    
    def is_approaching_major_event(self, days_threshold=2) -> bool:
        """
        Check if we're approaching a major economic event.
        
        Args:
            days_threshold: Number of days to consider "approaching"
            
        Returns:
            True if approaching major event, False otherwise
        """
        if not self.upcoming_events:
            self.update_upcoming_events()
        
        for event in self.upcoming_events:
            # Check if event is important (importance >= 8) and within threshold
            if event["importance"] >= 8 and event["days_until"] <= days_threshold:
                return True
        
        return False
    
    def get_pre_event_guidance(self, ticker=None) -> Dict[str, Any]:
        """
        Get guidance for approaching economic events.
        
        Args:
            ticker: Optional ticker symbol for stock-specific guidance
            
        Returns:
            Dict with pre-event guidance
        """
        if not self.upcoming_events:
            self.update_upcoming_events()
        
        # Filter to important upcoming events within 3 days
        imminent_events = [
            event for event in self.upcoming_events 
            if event["importance"] >= 8 and event["days_until"] <= 3
        ]
        
        if not imminent_events:
            return {
                "status": "normal",
                "message": "No major economic events in the next 3 days",
                "position_sizing_adjustment": 1.0,
                "recommendations": {}
            }
        
        # Get the most important imminent event
        most_important = max(imminent_events, key=lambda x: x["importance"])
        
        # Get specific guidance for this event
        guidance = most_important.get("pre_event_guidance", {})
        
        # Get position sizing adjustment based on event type
        event_type = most_important["event_type"]
        position_sizing_adjustment = self.config.get("macro_guidance", {}).get("position_sizing", {}).get("event_adjustments", {}).get(
            event_type, 
            self.config.get("macro_guidance", {}).get("position_sizing", {}).get("event_adjustments", {}).get("default", 0.8)
        )
        
        # Build response
        response = {
            "status": "pre_event",
            "event": {
                "type": most_important["event_type"],
                "description": most_important["description"],
                "date": most_important["date"],
                "time": most_important["time"],
                "days_until": most_important["days_until"]
            },
            "position_sizing_adjustment": position_sizing_adjustment,
            "recommendations": guidance
        }
        
        # Add ticker-specific guidance if provided
        if ticker:
            # Get sector for the ticker
            sector = self._get_ticker_sector(ticker)
            
            # Check sector sensitivity to this event type
            sensitivity = self._get_sector_sensitivity(sector, event_type)
            
            response["ticker_specific"] = {
                "ticker": ticker,
                "sector": sector,
                "sensitivity_to_event": sensitivity,
                "position_sizing_modifier": self._get_sensitivity_adjustment(sensitivity)
            }
        
        return response
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Get sector for a ticker symbol."""
        # Try to get from context engine if available
        if self.context_engine and hasattr(self.context_engine, "_get_symbol_sector"):
            return self.context_engine._get_symbol_sector(ticker)
        
        # Fallback to basic mapping (would be more comprehensive in practice)
        sector_mapping = {
            "AAPL": "technology",
            "MSFT": "technology",
            "GOOG": "technology",
            "AMZN": "consumer_discretionary",
            "META": "communication_services",
            "TSLA": "consumer_discretionary",
            "JPM": "financials",
            "BAC": "financials",
            "JNJ": "healthcare",
            "PFE": "healthcare",
            "XOM": "energy",
            "CVX": "energy",
            "PG": "consumer_staples",
            "KO": "consumer_staples",
            "HD": "consumer_discretionary"
        }
        
        return sector_mapping.get(ticker, "unknown")
    
    def _get_sector_sensitivity(self, sector: str, event_type: str) -> str:
        """Get sector sensitivity to an event type."""
        sector_sensitivities = self.config.get("macro_guidance", {}).get("sector_sensitivities", {}).get(sector, {})
        
        if event_type in sector_sensitivities.get("highly_sensitive_to", []):
            return "high"
        elif event_type in sector_sensitivities.get("moderately_sensitive_to", []):
            return "moderate"
        elif event_type in sector_sensitivities.get("low_sensitivity_to", []):
            return "low"
        else:
            return "unknown"
    
    def _get_sensitivity_adjustment(self, sensitivity: str) -> float:
        """Get position sizing adjustment based on sensitivity."""
        adjustments = {
            "high": 0.7,     # Reduce to 70% for high sensitivity
            "moderate": 0.85, # Reduce to 85% for moderate sensitivity
            "low": 0.95,     # Reduce to 95% for low sensitivity
            "unknown": 0.9   # Default reduction
        }
        
        return adjustments.get(sensitivity, 0.9)
    
    def enhance_trading_decision(self, 
                               ticker: str, 
                               base_recommendation: Dict[str, Any],
                               market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance a trading recommendation with macro guidance.
        
        Args:
            ticker: Ticker symbol
            base_recommendation: Original recommendation from the trading system
            market_data: Current market data
            
        Returns:
            Enhanced recommendation with macro guidance
        """
        # Clone the base recommendation
        enhanced = base_recommendation.copy()
        
        # Check if we're approaching a major event
        approaching_event = self.is_approaching_major_event()
        
        # Check VIX level if available
        vix_level = market_data.get("vix", 0) if market_data else 0
        vix_change = market_data.get("vix_change_percent", 0) if market_data else 0
        
        # Check yield curve if available
        ten_year = market_data.get("ten_year_yield", 0) if market_data else 0
        two_year = market_data.get("two_year_yield", 0) if market_data else 0
        
        # Get current market regime
        regime_guidance = self.macro_engine.get_market_regime_guidance()
        
        # Get sector rotation guidance for this ticker
        sector_guidance = self.macro_engine.get_sector_rotation_guidance(ticker=ticker)
        
        # Get seasonality guidance for this ticker
        seasonality_guidance = self.macro_engine.get_seasonality_guidance(ticker=ticker)
        
        # Adjust position sizing based on market conditions
        position_sizing_adjustment = 1.0
        
        # Adjust for approaching economic events
        if approaching_event:
            pre_event_guidance = self.get_pre_event_guidance(ticker)
            position_sizing_adjustment *= pre_event_guidance["position_sizing_adjustment"]
            
            # Add pre-event guidance to recommendation
            enhanced["macro_guidance"] = {
                "pre_event_guidance": pre_event_guidance,
                "position_sizing_adjustment": position_sizing_adjustment
            }
        
        # Adjust for VIX level
        if vix_level > 0:
            vix_adjustments = self.config.get("macro_guidance", {}).get("position_sizing", {}).get("vix_adjustments", {})
            if vix_level >= 40 and "above_40" in vix_adjustments:
                position_sizing_adjustment *= vix_adjustments["above_40"]
            elif vix_level >= 30 and "above_30" in vix_adjustments:
                position_sizing_adjustment *= vix_adjustments["above_30"]
            elif vix_level >= 25 and "above_25" in vix_adjustments:
                position_sizing_adjustment *= vix_adjustments["above_25"]
            elif vix_level >= 20 and "above_20" in vix_adjustments:
                position_sizing_adjustment *= vix_adjustments["above_20"]
            
            # Check for VIX spike
            if vix_change >= 20:
                vix_guidance = self.macro_engine.get_vix_spike_guidance(vix_level, vix_change)
                
                # Add VIX guidance to recommendation
                if "macro_guidance" not in enhanced:
                    enhanced["macro_guidance"] = {}
                
                enhanced["macro_guidance"]["vix_guidance"] = vix_guidance
                
                # Further adjust position sizing for VIX spike
                if vix_guidance["severity"] == "severe":
                    position_sizing_adjustment *= 0.5
                elif vix_guidance["severity"] == "high":
                    position_sizing_adjustment *= 0.7
                elif vix_guidance["severity"] == "moderate":
                    position_sizing_adjustment *= 0.8
        
        # Adjust for market regime
        regime_adjustments = self.config.get("macro_guidance", {}).get("position_sizing", {}).get("regime_adjustments", {})
        regime_adjustment = regime_adjustments.get(regime_guidance["current_regime"], 0.9)
        position_sizing_adjustment *= regime_adjustment
        
        # Add regime guidance to recommendation
        if "macro_guidance" not in enhanced:
            enhanced["macro_guidance"] = {}
        
        enhanced["macro_guidance"]["regime_guidance"] = {
            "current_regime": regime_guidance["current_regime"],
            "description": regime_guidance["description"],
            "position_sizing_adjustment": regime_adjustment
        }
        
        # Add yield curve guidance if we have the data
        if ten_year > 0 and two_year > 0:
            spread = ten_year - two_year
            # We'd need to track days_inverted in a real implementation
            days_inverted = 0
            if spread < 0:
                days_inverted = 30  # Placeholder
            
            yield_curve_guidance = self.macro_engine.get_yield_curve_guidance(
                ten_year_yield=ten_year,
                two_year_yield=two_year,
                days_inverted=days_inverted
            )
            
            enhanced["macro_guidance"]["yield_curve_guidance"] = {
                "spread": spread,
                "status": yield_curve_guidance["curve_status"],
                "phase": yield_curve_guidance["phase"]
            }
            
            # Adjust position sizing for inverted yield curve
            if yield_curve_guidance["phase"] == "deep_inversion_with_confirming_indicators":
                position_sizing_adjustment *= 0.7
            elif yield_curve_guidance["phase"] == "persistent_inversion":
                position_sizing_adjustment *= 0.8
            elif yield_curve_guidance["phase"] == "initial_inversion":
                position_sizing_adjustment *= 0.9
        
        # Add sector rotation guidance if available
        if "ticker_guidance" in sector_guidance:
            # Get the ticker-specific sector rotation guidance
            ticker_guidance = sector_guidance["ticker_guidance"]
            
            # Add to enhanced recommendation
            if "macro_guidance" not in enhanced:
                enhanced["macro_guidance"] = {}
            
            enhanced["macro_guidance"]["sector_rotation"] = {
                "sector": ticker_guidance["sector"],
                "classification": ticker_guidance["sector_classification"],
                "cycle_phase": sector_guidance["cycle_phase"]
            }
            
            # Add historical performance if available
            if "historical_performance" in ticker_guidance:
                enhanced["macro_guidance"]["sector_rotation"]["historical_performance"] = ticker_guidance["historical_performance"]
            
            # Add recommended strategies if available
            if "recommended_equity_strategies" in ticker_guidance or "recommended_options_strategies" in ticker_guidance:
                enhanced["macro_guidance"]["sector_rotation"]["recommended_strategies"] = {}
                
                if "recommended_equity_strategies" in ticker_guidance:
                    enhanced["macro_guidance"]["sector_rotation"]["recommended_strategies"]["equity"] = [
                        {"name": strategy["strategy"], "implementation": strategy["implementation"]}
                        for strategy in ticker_guidance["recommended_equity_strategies"]
                    ]
                
                if "recommended_options_strategies" in ticker_guidance:
                    enhanced["macro_guidance"]["sector_rotation"]["recommended_strategies"]["options"] = [
                        {"name": strategy["strategy"], "implementation": strategy["implementation"]}
                        for strategy in ticker_guidance["recommended_options_strategies"]
                    ]
            
            # Adjust position sizing based on sector classification
            sector_classification = ticker_guidance["sector_classification"]
            if sector_classification == "primary_favored":
                # Increase position sizing for primary favored sectors
                position_sizing_adjustment *= 1.2  # 20% increase
            elif sector_classification == "secondary_favored":
                # Slight increase for secondary favored sectors
                position_sizing_adjustment *= 1.1  # 10% increase
            elif sector_classification == "avoid":
                # Significant decrease for sectors to avoid
                position_sizing_adjustment *= 0.6  # 40% decrease
            elif sector_classification == "neutral":
                # No change for neutral sectors
                pass
        
        # Add seasonality guidance if available
        if seasonality_guidance.get("status") == "success" and "ticker_guidance" in seasonality_guidance:
            # Get the ticker-specific seasonality guidance
            seasonality_ticker_guidance = seasonality_guidance["ticker_guidance"]
            
            # Add to enhanced recommendation
            if "macro_guidance" not in enhanced:
                enhanced["macro_guidance"] = {}
            
            enhanced["macro_guidance"]["seasonality"] = {
                "month": seasonality_guidance["month"],
                "expected_bias": seasonality_guidance["expected_bias"],
                "sector_performance": seasonality_ticker_guidance["sector_performance"],
                "active_patterns": [p["pattern"] for p in seasonality_guidance.get("active_recurring_patterns", [])]
            }
            
            # Add recommended strategies if available
            if "recommended_equity_strategies" in seasonality_ticker_guidance or "recommended_options_strategies" in seasonality_ticker_guidance:
                enhanced["macro_guidance"]["seasonality"]["recommended_strategies"] = {}
                
                if "recommended_equity_strategies" in seasonality_ticker_guidance:
                    enhanced["macro_guidance"]["seasonality"]["recommended_strategies"]["equity"] = [
                        {"name": strategy["strategy"], "pattern": strategy["pattern_name"]}
                        for strategy in seasonality_ticker_guidance["recommended_equity_strategies"][:3]  # Limit to top 3
                    ]
                
                if "recommended_options_strategies" in seasonality_ticker_guidance:
                    enhanced["macro_guidance"]["seasonality"]["recommended_strategies"]["options"] = [
                        {"name": strategy["strategy"], "pattern": strategy["pattern_name"]}
                        for strategy in seasonality_ticker_guidance["recommended_options_strategies"][:3]  # Limit to top 3
                    ]
            
            # Add composite score if available
            if "composite_score" in seasonality_guidance:
                enhanced["macro_guidance"]["seasonality"]["composite_score"] = seasonality_guidance["composite_score"]
            
            # Adjust position sizing based on seasonality bias
            if "position_sizing_guidance" in seasonality_ticker_guidance:
                seasonal_position_sizing = seasonality_ticker_guidance["position_sizing_guidance"]["position_sizing_factor"]
                position_sizing_adjustment *= seasonal_position_sizing
                
                # Add rationale to the adjustment
                if "position_sizing_guidance" in seasonality_ticker_guidance and "rationale" in seasonality_ticker_guidance["position_sizing_guidance"]:
                    if "macro_guidance" not in enhanced:
                        enhanced["macro_guidance"] = {}
                    
                    if "seasonality" not in enhanced["macro_guidance"]:
                        enhanced["macro_guidance"]["seasonality"] = {}
                    
                    enhanced["macro_guidance"]["seasonality"]["position_sizing_rationale"] = seasonality_ticker_guidance["position_sizing_guidance"]["rationale"]
        
        # Apply final position sizing adjustment to recommendation
        enhanced["macro_adjusted_position_sizing"] = position_sizing_adjustment
        
        # If there's position sizing in the base recommendation, adjust it
        if "position_sizing" in enhanced:
            for key in enhanced["position_sizing"]:
                if isinstance(enhanced["position_sizing"][key], (int, float)):
                    enhanced["position_sizing"][key] = enhanced["position_sizing"][key] * position_sizing_adjustment
            
            enhanced["position_sizing"]["macro_adjusted"] = True
        
        # Add timeframe adjustment if needed
        timeframe_adjustment = self._get_timeframe_adjustment(
            approaching_event=approaching_event,
            vix_level=vix_level,
            current_regime=regime_guidance["current_regime"]
        )
        
        if timeframe_adjustment != "standard":
            if "macro_guidance" not in enhanced:
                enhanced["macro_guidance"] = {}
            
            enhanced["macro_guidance"]["timeframe_adjustment"] = timeframe_adjustment
        
        return enhanced
    
    def _get_timeframe_adjustment(self, approaching_event=False, vix_level=0, current_regime="unknown") -> str:
        """Determine appropriate timeframe adjustment based on market conditions."""
        # Start with standard timeframe
        adjustment = "standard"
        
        # Check for VIX-based adjustments
        if vix_level >= 40:
            adjustment = "shorten by 60%"
        elif vix_level >= 30:
            adjustment = "shorten by 40%"
        elif vix_level >= 25:
            adjustment = "shorten by 20%"
        
        # Regime-based adjustments
        if current_regime == "contraction":
            # If already adjusted for VIX, take the more conservative adjustment
            if adjustment == "standard":
                adjustment = "shorten by 30%"
            # Otherwise keep the VIX adjustment as it's likely more conservative
        elif current_regime == "late_cycle":
            if adjustment == "standard":
                adjustment = "shorten by 15%"
        
        # If approaching a major event, adjust timeframe if not already adjusted
        if approaching_event and adjustment == "standard":
            adjustment = "shorten by 20%"
        
        return adjustment
    
    def process_economic_event(self, event_type: str, actual_data: Dict[str, Any], expected_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a specific economic event and get trading guidance.
        
        Args:
            event_type: Type of economic event
            actual_data: Actual reported data
            expected_data: Expected/forecast data for comparison
            
        Returns:
            Dict with analysis and recommendations
        """
        # Process the event with the macro engine
        return self.macro_engine.process_event_outcome(event_type, actual_data, expected_data)
    
    def adjust_strategy_for_macro_conditions(self, 
                                          ticker: str, 
                                          strategy_name: str,
                                          strategy_params: Dict[str, Any],
                                          market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Adjust a trading strategy based on macro conditions.
        
        Args:
            ticker: Ticker symbol
            strategy_name: Name of the strategy
            strategy_params: Strategy parameters
            market_data: Current market data
            
        Returns:
            Adjusted strategy parameters
        """
        # Clone the strategy parameters
        adjusted_params = strategy_params.copy()
        
        # Get macro guidance
        macro_guidance = {}
        
        # Check if approaching major event
        if self.is_approaching_major_event():
            pre_event = self.get_pre_event_guidance(ticker)
            macro_guidance["pre_event"] = pre_event
        
        # Get current regime guidance
        regime_guidance = self.macro_engine.get_market_regime_guidance()
        macro_guidance["regime"] = {
            "current_regime": regime_guidance["current_regime"],
            "description": regime_guidance["description"]
        }
        
        # Get sector rotation guidance
        sector_guidance = self.macro_engine.get_sector_rotation_guidance(ticker=ticker)
        if "ticker_guidance" in sector_guidance:
            macro_guidance["sector_rotation"] = sector_guidance["ticker_guidance"]
        
        # Get seasonality guidance
        seasonality_guidance = self.macro_engine.get_seasonality_guidance(ticker=ticker)
        if seasonality_guidance.get("status") == "success" and "ticker_guidance" in seasonality_guidance:
            macro_guidance["seasonality"] = {
                "month": seasonality_guidance["month"],
                "expected_bias": seasonality_guidance["expected_bias"],
                "position_sizing_guidance": seasonality_guidance["ticker_guidance"]["position_sizing_guidance"]
            }
        
        # Check VIX level if available
        if market_data and "vix" in market_data:
            vix_level = market_data["vix"]
            vix_change = market_data.get("vix_change_percent", 0)
            
            if vix_change >= 20:
                vix_guidance = self.macro_engine.get_vix_spike_guidance(vix_level, vix_change)
                macro_guidance["vix_spike"] = vix_guidance
        
        # Check if we're in earnings season
        # This would be enhanced in a real implementation to actually check
        current_month = datetime.datetime.now().month
        if current_month in [1, 4, 7, 10]:  # Earnings months
            # Simplified determination of earnings phase
            current_day = datetime.datetime.now().day
            if current_day < 15:
                earnings_phase = "early"
            elif current_day < 25:
                earnings_phase = "peak"
            else:
                earnings_phase = "late"
                
            earnings_guidance = self.macro_engine.get_earnings_season_guidance(earnings_phase)
            macro_guidance["earnings_season"] = earnings_guidance
        
        # Apply adjustments based on guidance
        
        # 1. Adjust position size
        position_sizing_adjustment = 1.0
        
        if "pre_event" in macro_guidance:
            position_sizing_adjustment *= macro_guidance["pre_event"]["position_sizing_adjustment"]
        
        if "vix_spike" in macro_guidance:
            severity = macro_guidance["vix_spike"]["severity"]
            if severity == "severe":
                position_sizing_adjustment *= 0.5
            elif severity == "high":
                position_sizing_adjustment *= 0.7
            elif severity == "moderate":
                position_sizing_adjustment *= 0.8
        
        # Apply regime-based adjustment
        if "regime" in macro_guidance:
            regime = macro_guidance["regime"]["current_regime"]
            regime_adjustments = self.config.get("macro_guidance", {}).get("position_sizing", {}).get("regime_adjustments", {})
            regime_adjustment = regime_adjustments.get(regime, 0.9)
            position_sizing_adjustment *= regime_adjustment
        
        # Apply sector rotation adjustment
        if "sector_rotation" in macro_guidance:
            sector_classification = macro_guidance["sector_rotation"]["sector_classification"]
            if sector_classification == "primary_favored":
                # Increase position sizing for primary favored sectors
                position_sizing_adjustment *= 1.2  # 20% increase
            elif sector_classification == "secondary_favored":
                # Slight increase for secondary favored sectors
                position_sizing_adjustment *= 1.1  # 10% increase
            elif sector_classification == "avoid":
                # Significant decrease for sectors to avoid
                position_sizing_adjustment *= 0.6  # 40% decrease
        
        # Apply seasonality adjustment
        if "seasonality" in macro_guidance and "position_sizing_guidance" in macro_guidance["seasonality"]:
            seasonal_position_sizing = macro_guidance["seasonality"]["position_sizing_guidance"]["position_sizing_factor"]
            position_sizing_adjustment *= seasonal_position_sizing
        
        # Adjust quantity if present
        if "quantity" in adjusted_params and isinstance(adjusted_params["quantity"], (int, float)):
            original_quantity = adjusted_params["quantity"]
            adjusted_quantity = int(original_quantity * position_sizing_adjustment)
            # Ensure at least 1 if original was non-zero
            if original_quantity > 0 and adjusted_quantity < 1:
                adjusted_quantity = 1
            adjusted_params["quantity"] = adjusted_quantity
        
        # 2. Adjust risk parameters
        if "stop_loss" in adjusted_params and isinstance(adjusted_params["stop_loss"], (int, float)):
            # Widen stop loss during volatility or approaching events
            stop_adjustment = 1.0
            
            if "vix_spike" in macro_guidance:
                severity = macro_guidance["vix_spike"]["severity"]
                if severity == "severe":
                    stop_adjustment = 1.3  # 30% wider
                elif severity == "high":
                    stop_adjustment = 1.2  # 20% wider
                elif severity == "moderate":
                    stop_adjustment = 1.1  # 10% wider
            
            if "pre_event" in macro_guidance and stop_adjustment == 1.0:
                stop_adjustment = 1.15  # 15% wider before major events
            
            # Adjust for sector rotation
            if "sector_rotation" in macro_guidance:
                sector_classification = macro_guidance["sector_rotation"]["sector_classification"]
                if sector_classification == "avoid":
                    # Tighter stops for sectors to avoid
                    stop_adjustment = 0.9  # 10% tighter
            
            # Apply stop adjustment
            if stop_adjustment != 1.0:
                # Calculate adjustment based on direction (long vs short)
                is_long = strategy_name.lower() in ["long", "buy", "bull", "call"] or "action" in adjusted_params and adjusted_params["action"].lower() in ["buy", "buy_to_open"]
                
                current_price = market_data.get("close", 0) if market_data else 0
                original_stop = adjusted_params["stop_loss"]
                
                if is_long:
                    # For long positions, widen stop by moving it lower
                    if current_price > 0:
                        # Calculate as percentage from current price
                        original_distance = (current_price - original_stop) / current_price
                        new_distance = original_distance * stop_adjustment
                        adjusted_params["stop_loss"] = current_price * (1 - new_distance)
                    else:
                        # Simple adjustment if we don't have current price
                        adjusted_params["stop_loss"] = original_stop * (1 - (stop_adjustment - 1))
                else:
                    # For short positions, widen stop by moving it higher
                    if current_price > 0:
                        # Calculate as percentage from current price
                        original_distance = (original_stop - current_price) / current_price
                        new_distance = original_distance * stop_adjustment
                        adjusted_params["stop_loss"] = current_price * (1 + new_distance)
                    else:
                        # Simple adjustment if we don't have current price
                        adjusted_params["stop_loss"] = original_stop * (1 + (stop_adjustment - 1))
        
        # 3. Adjust profit targets
        if "take_profit" in adjusted_params and isinstance(adjusted_params["take_profit"], (int, float)):
            # During volatile periods or pre-event, consider more conservative profit targets
            profit_adjustment = 1.0
            
            if "vix_spike" in macro_guidance:
                severity = macro_guidance["vix_spike"]["severity"]
                if severity == "severe":
                    profit_adjustment = 0.7  # 30% closer target
                elif severity == "high":
                    profit_adjustment = 0.8  # 20% closer target
                elif severity == "moderate":
                    profit_adjustment = 0.9  # 10% closer target
            
            if "pre_event" in macro_guidance and profit_adjustment == 1.0:
                profit_adjustment = 0.9  # 10% closer target before major events
            
            # Adjust for sector rotation
            if "sector_rotation" in macro_guidance:
                sector_classification = macro_guidance["sector_rotation"]["sector_classification"]
                if sector_classification == "primary_favored":
                    # More ambitious targets for favored sectors
                    profit_adjustment = 1.2  # 20% further target
                elif sector_classification == "avoid":
                    # Conservative targets for sectors to avoid
                    profit_adjustment = 0.7  # 30% closer target
            
            # Apply profit target adjustment
            if profit_adjustment != 1.0:
                is_long = strategy_name.lower() in ["long", "buy", "bull", "call"] or "action" in adjusted_params and adjusted_params["action"].lower() in ["buy", "buy_to_open"]
                
                current_price = market_data.get("close", 0) if market_data else 0
                original_target = adjusted_params["take_profit"]
                
                if is_long:
                    # For long positions, adjust by bringing target lower
                    if current_price > 0:
                        # Calculate as percentage from current price
                        original_distance = (original_target - current_price) / current_price
                        new_distance = original_distance * profit_adjustment
                        adjusted_params["take_profit"] = current_price * (1 + new_distance)
                    else:
                        # Simple adjustment if we don't have current price
                        adjusted_params["take_profit"] = original_target * profit_adjustment
                else:
                    # For short positions, adjust by bringing target higher
                    if current_price > 0:
                        # Calculate as percentage from current price
                        original_distance = (current_price - original_target) / current_price
                        new_distance = original_distance * profit_adjustment
                        adjusted_params["take_profit"] = current_price * (1 - new_distance)
                    else:
                        # Simple adjustment if we don't have current price
                        adjusted_params["take_profit"] = original_target / profit_adjustment
        
        # Add recommended strategies from sector rotation if available
        if "sector_rotation" in macro_guidance and "recommended_strategies" in macro_guidance["sector_rotation"]:
            if "sector_rotation_strategies" not in adjusted_params:
                adjusted_params["sector_rotation_strategies"] = macro_guidance["sector_rotation"]["recommended_strategies"]
        
        # Add recommended strategies from seasonality if available
        if "seasonality" in macro_guidance and "ticker_guidance" in seasonality_guidance:
            ticker_guidance = seasonality_guidance["ticker_guidance"]
            if "recommended_equity_strategies" in ticker_guidance or "recommended_options_strategies" in ticker_guidance:
                if "seasonality_strategies" not in adjusted_params:
                    adjusted_params["seasonality_strategies"] = {}
                
                if "recommended_equity_strategies" in ticker_guidance:
                    adjusted_params["seasonality_strategies"]["equity"] = [
                        {
                            "strategy": strategy["strategy"],
                            "pattern": strategy["pattern_name"],
                            "implementation": strategy.get("implementation", {})
                        }
                        for strategy in ticker_guidance["recommended_equity_strategies"][:3]  # Limit to top 3
                    ]
                
                if "recommended_options_strategies" in ticker_guidance:
                    adjusted_params["seasonality_strategies"]["options"] = [
                        {
                            "strategy": strategy["strategy"],
                            "pattern": strategy["pattern_name"],
                            "implementation": strategy.get("implementation", {})
                        }
                        for strategy in ticker_guidance["recommended_options_strategies"][:3]  # Limit to top 3
                    ]
        
        # Add macro guidance to the adjusted parameters
        adjusted_params["macro_guidance"] = {
            "position_sizing_adjustment": position_sizing_adjustment,
            "guidance_summary": macro_guidance
        }
        
        return adjusted_params
    
    def get_seasonality_guidance(self, ticker: str = None, specific_month: str = None) -> Dict[str, Any]:
        """
        Get seasonality guidance for a specific ticker or month.
        
        Args:
            ticker: Optional ticker symbol to get specific guidance for
            specific_month: Optional month name to get guidance for (default: current month)
            
        Returns:
            Dict with seasonality guidance
        """
        # Get guidance from macro engine
        return self.macro_engine.get_seasonality_guidance(ticker=ticker, specific_month=specific_month) 