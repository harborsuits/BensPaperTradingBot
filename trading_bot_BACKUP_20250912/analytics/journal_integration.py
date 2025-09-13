"""
Trade Journal Integration

This module connects the Trade Journal System with various components of the trading bot
to automate data collection and enrich trade journal entries.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .trade_journal_system import TradeJournalSystem
from ..macro_guidance.macro_engine import MacroGuidanceEngine
from ..pattern_detection import PatternDetectionEngine

logger = logging.getLogger(__name__)

class JournalIntegration:
    """
    Integrates the Trade Journal System with various trading bot components.
    
    This class automates the collection of data from different parts of the trading bot
    and enriches trade journal entries with contextual information.
    """
    
    def __init__(self, journal_system: TradeJournalSystem):
        """
        Initialize the journal integration.
        
        Args:
            journal_system: The trade journal system instance
        """
        self.journal = journal_system
        self.macro_engine = None
        self.pattern_engine = None
        
        # Attempt to initialize optional components
        try:
            from ..macro_guidance.macro_engine import MacroGuidanceEngine
            self.macro_engine = MacroGuidanceEngine()
            logger.info("Macro Guidance Engine initialized for journal integration")
        except Exception as e:
            logger.warning(f"Could not initialize Macro Guidance Engine: {str(e)}")
        
        try:
            from ..pattern_detection import PatternDetectionEngine
            self.pattern_engine = PatternDetectionEngine()
            logger.info("Pattern Detection Engine initialized for journal integration")
        except Exception as e:
            logger.warning(f"Could not initialize Pattern Detection Engine: {str(e)}")
    
    def process_new_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Process a new trade and create a journal entry with enriched data.
        
        Args:
            trade_data: Basic trade data
            
        Returns:
            str: Trade ID
        """
        # Format the data for the journal template
        formatted_data = self.format_trade_data(trade_data)
        
        # Enrich with additional context if available
        enriched_data = self.enrich_trade_data(formatted_data)
        
        # Start a new trade journal entry
        trade_id = self.journal.start_new_trade(enriched_data)
        
        logger.info(f"Created journal entry for trade {trade_id}")
        return trade_id
    
    def format_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format trade data to match the journal template structure.
        
        Args:
            trade_data: Raw trade data
            
        Returns:
            Dict: Formatted trade data
        """
        # Extract ticker and format it consistently
        ticker = trade_data.get("symbol", trade_data.get("ticker", ""))
        
        # Determine asset class
        asset_type = trade_data.get("asset_type", "equity")
        if "option" in asset_type.lower() or "strike" in str(trade_data):
            asset_class = "options"
        elif "future" in asset_type.lower():
            asset_class = "futures"
        else:
            asset_class = asset_type
        
        # Determine position type
        action = trade_data.get("action", "").lower()
        if "sell" in action and not "close" in action:
            position_type = "short"
        else:
            position_type = "long"
        
        # Create timestamp if not provided
        if "timestamp" not in trade_data:
            timestamp = datetime.now().isoformat()
        else:
            timestamp = trade_data["timestamp"]
        
        # Format date and time
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date = dt.strftime("%Y-%m-%d")
            time = dt.strftime("%H:%M %p ET")  # Assuming ET, adjust as needed
        except:
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M %p ET")
        
        # Create formatted structure
        formatted = {
            "trade_metadata": {
                "date": date,
                "timestamp": timestamp,
                "ticker": ticker,
                "underlying_ticker": ticker.split()[0] if asset_class == "options" else ticker,
                "asset_class": asset_class,
                "position_type": position_type,
                "position_details": {
                    "primary_strategy": trade_data.get("strategy", "unknown"),
                    "strategy_variant": trade_data.get("strategy_variant", "")
                }
            },
            "execution_details": {
                "entry": {
                    "date": date,
                    "time": time,
                    "price": trade_data.get("price", 0),
                    "quantity": trade_data.get("quantity", 0),
                    "order_type": trade_data.get("order_type", "market"),
                    "commission_fees": trade_data.get("commission", 0)
                }
            }
        }
        
        # Add option-specific details if applicable
        if asset_class == "options" and "option_data" in trade_data:
            option_data = trade_data["option_data"]
            
            # Determine option type from the symbol or data
            option_symbol = ticker.upper()
            contract_type = ""
            
            if "CALL" in option_symbol or "C" in option_symbol.split()[-1]:
                contract_type = "call"
            elif "PUT" in option_symbol or "P" in option_symbol.split()[-1]:
                contract_type = "put"
            
            # Or use provided type
            if "type" in option_data:
                contract_type = option_data["type"].lower()
            
            # Format expiration date
            expiration_date = option_data.get("expiration_date", "")
            
            # Add to formatted data
            formatted["trade_metadata"]["position_details"]["option_specific"] = {
                "contract_type": contract_type,
                "expiration_date": expiration_date,
                "strike_price": option_data.get("strike", 0),
                "days_to_expiration_entry": option_data.get("days_to_expiration", 0),
                "implied_volatility_entry": option_data.get("implied_volatility", 0)
            }
            
            # Add greeks if available
            if "greeks" in option_data:
                greeks = option_data["greeks"]
                formatted["trade_metadata"]["position_details"]["option_specific"].update({
                    "delta_at_entry": greeks.get("delta", 0),
                    "theta_at_entry": greeks.get("theta", 0),
                    "vega_at_entry": greeks.get("vega", 0),
                    "gamma_at_entry": greeks.get("gamma", 0)
                })
        
        return formatted
    
    def enrich_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich trade data with additional context from other components.
        
        Args:
            trade_data: Formatted trade data
            
        Returns:
            Dict: Enriched trade data
        """
        # Clone the data to avoid modifying the original
        enriched = trade_data.copy()
        
        # Get ticker
        ticker = trade_data.get("trade_metadata", {}).get("ticker", "")
        if not ticker:
            return enriched
        
        # Add market context if macro engine is available
        if self.macro_engine:
            try:
                # Get the current market regime
                regime_guidance = self.macro_engine.get_market_regime_guidance()
                
                # Get sector rotation guidance
                sector_guidance = self.macro_engine.get_sector_rotation_guidance(ticker=ticker)
                
                # Get seasonality guidance
                seasonality_guidance = self.macro_engine.get_seasonality_guidance(ticker=ticker)
                
                # Add market context
                market_context = {
                    "market_regime": {
                        "primary_regime": regime_guidance.get("current_regime", "unknown"),
                        "description": regime_guidance.get("description", ""),
                        "market_phase": self._map_regime_to_phase(regime_guidance.get("current_regime", "unknown"))
                    },
                    "market_structure": {
                        "volatility_environment": {
                            "vix_level": 0  # This would be filled with actual data
                        }
                    },
                    "macro_factors": {
                        "fed_policy": regime_guidance.get("key_indicators", {})
                    }
                }
                
                # Add sector context if available
                if "ticker_guidance" in sector_guidance:
                    ticker_guidance = sector_guidance["ticker_guidance"]
                    market_context["sector_context"] = {
                        "sector_performance": {
                            "sector_ranking": ticker_guidance.get("historical_performance", {}).get("rank", 0)
                        },
                        "sector_rotation_phase": sector_guidance.get("cycle_phase", ""),
                        "sector_sentiment": self._determine_sector_sentiment(ticker_guidance)
                    }
                
                # Add seasonality factors if available
                if "ticker_guidance" in seasonality_guidance:
                    market_context["seasonal_factors"] = {
                        "monthly_pattern": seasonality_guidance.get("name", seasonality_guidance.get("month", "")),
                        "historical_edge": seasonality_guidance.get("composite_score", {}).get("score", 0) / 100 if isinstance(seasonality_guidance.get("composite_score", {}), dict) else 0
                    }
                
                # Add to enriched data
                if "market_context" not in enriched:
                    enriched["market_context"] = {}
                
                self._deep_update(enriched["market_context"], market_context)
                
            except Exception as e:
                logger.error(f"Error enriching trade with macro data: {str(e)}")
        
        # Add pattern detection if pattern engine is available
        if self.pattern_engine:
            try:
                # Get active patterns for this ticker
                patterns = self.pattern_engine.get_active_patterns(symbol=ticker)
                
                if patterns:
                    # Format for trade_rationale section
                    technical_factors = {
                        "pattern_recognized": patterns[0].get("pattern", "") if patterns else "",
                        "indicators": []
                    }
                    
                    # Add to enriched data
                    if "trade_rationale" not in enriched:
                        enriched["trade_rationale"] = {}
                    
                    if "entry_criteria" not in enriched["trade_rationale"]:
                        enriched["trade_rationale"]["entry_criteria"] = {}
                    
                    if "technical_factors" not in enriched["trade_rationale"]["entry_criteria"]:
                        enriched["trade_rationale"]["entry_criteria"]["technical_factors"] = {}
                    
                    self._deep_update(enriched["trade_rationale"]["entry_criteria"]["technical_factors"], technical_factors)
                    
            except Exception as e:
                logger.error(f"Error enriching trade with pattern data: {str(e)}")
        
        return enriched
    
    def process_trade_close(self, trade_id: str, exit_data: Dict[str, Any]) -> bool:
        """
        Process a trade closure and update the journal entry.
        
        Args:
            trade_id: Trade ID
            exit_data: Trade exit data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format exit data
        formatted_exit = self._format_exit_data(exit_data)
        
        # Add any additional context at exit time
        enriched_exit = self._enrich_exit_data(formatted_exit)
        
        # Close the trade in the journal
        success = self.journal.close_trade(trade_id, enriched_exit)
        
        # Add post-trade analysis
        if success:
            self._add_post_trade_analysis(trade_id)
        
        return success
    
    def _format_exit_data(self, exit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format exit data to match the journal template structure."""
        # Create timestamp if not provided
        if "timestamp" not in exit_data:
            timestamp = datetime.now().isoformat()
        else:
            timestamp = exit_data["timestamp"]
        
        # Format date and time
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date = dt.strftime("%Y-%m-%d")
            time = dt.strftime("%H:%M %p ET")  # Assuming ET, adjust as needed
        except:
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M %p ET")
        
        # Format exit data
        formatted = {
            "date": date,
            "time": time,
            "price": exit_data.get("price", 0),
            "quantity": exit_data.get("quantity", 0),
            "order_type": exit_data.get("order_type", "market"),
            "commission_fees": exit_data.get("commission", 0),
            "exit_reason": exit_data.get("exit_reason", "target_reached")
        }
        
        # Add exit condition if provided
        if "exit_condition" in exit_data:
            formatted["exit_condition"] = exit_data["exit_condition"]
        
        return formatted
    
    def _enrich_exit_data(self, exit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich exit data with additional context."""
        # This would add any additional context at exit time
        # For now, just return the original data
        return exit_data
    
    def _add_post_trade_analysis(self, trade_id: str) -> bool:
        """
        Add post-trade analysis to a closed trade.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get the complete trade data
        trade = self.journal.get_trade(trade_id)
        if not trade:
            return False
        
        # Prepare evaluation data
        evaluation = {
            "technical_execution": {
                "entry_timing_grade": "B",  # These would be calculated based on actual trade performance
                "exit_timing_grade": "B",
                "trade_management_grade": "B",
                "position_sizing_grade": "B",
                "technical_analysis_grade": "B",
                "overall_technical_grade": "B"
            },
            "psychological_execution": {
                "emotional_state": "neutral",  # Default value
                "decision_clarity": "medium",
                "patience_score": "medium",
                "discipline_to_plan": "moderate",
                "adaptation_appropriateness": "appropriate",
                "overall_psychological_grade": "B"
            },
            "process_adherence": {
                "pre_trade_checklist_completed": True,
                "followed_entry_rules": True,
                "followed_exit_rules": True,
                "followed_position_sizing_rules": True,
                "followed_risk_management_rules": True,
                "overall_process_grade": "B"
            }
        }
        
        # Add lessons learned - basic template
        lessons = {
            "key_observations": [
                "Trade executed according to plan"
            ],
            "what_worked_well": [],
            "what_needs_improvement": [],
            "follow_up_actions": [
                "Review similar setups for pattern consistency"
            ]
        }
        
        # Determine result and add appropriate lessons
        result = trade.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss", "")
        
        if result == "win":
            lessons["what_worked_well"].append("Entry criteria identified a profitable opportunity")
            lessons["what_worked_well"].append("Exit at target preserved profits")
        elif result == "loss":
            lessons["what_needs_improvement"].append("Review entry criteria for potential improvements")
            lessons["what_needs_improvement"].append("Evaluate stop placement methodology")
        
        # Add the evaluation and lessons
        success1 = self.journal.add_trade_evaluation(trade_id, evaluation)
        success2 = self.journal.add_lessons_learned(trade_id, lessons)
        
        return success1 and success2
    
    def _map_regime_to_phase(self, regime: str) -> str:
        """Map market regime to market phase."""
        regime_to_phase = {
            "expansion": "mid_bull",
            "late_cycle": "late_bull",
            "contraction": "mid_bear",
            "early_recovery": "early_bull",
            "unknown": "consolidation"
        }
        return regime_to_phase.get(regime.lower(), "consolidation")
    
    def _determine_sector_sentiment(self, ticker_guidance: Dict[str, Any]) -> str:
        """Determine sector sentiment based on ticker guidance."""
        classification = ticker_guidance.get("sector_classification", "")
        
        if classification == "primary_favored":
            return "bullish"
        elif classification == "secondary_favored":
            return "neutral"
        elif classification == "avoid":
            return "bearish"
        else:
            return "neutral"
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Dict: Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

# Convenience function to get an instance with default configuration
def get_journal_integration() -> JournalIntegration:
    """
    Get a journal integration instance with default configuration.
    
    Returns:
        JournalIntegration: Configured journal integration instance
    """
    journal = TradeJournalSystem()
    return JournalIntegration(journal) 