"""
Trade Executor Journal Integration

This module provides a seamless integration between the advanced TradeExecutor system
and the comprehensive TradeJournalSystem for complete trade lifecycle management.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Import the trade journal system components
from analytics.trade_journal_system import TradeJournalSystem
from analytics.journal_integration import get_journal_integration

# Import executor components (adjust imports as needed for your project structure)
# from trade_executor import TradeExecutor, TradeResult, OrderType, OrderSide, TradeType

class JournaledTradeExecutor:
    """
    Enhanced Trade Executor with integrated comprehensive journaling.
    
    This class wraps a TradeExecutor instance and adds sophisticated
    journaling capabilities for detailed trade analysis and review.
    """
    
    def __init__(self, executor, journal_dir="journal", auto_journal=True):
        """
        Initialize the journaled trade executor.
        
        Args:
            executor: Existing TradeExecutor instance
            journal_dir: Directory for journal data
            auto_journal: Whether to automatically journal all trades
        """
        self.executor = executor
        self.auto_journal = auto_journal
        
        # Initialize journal system
        self.journal_integration = get_journal_integration()
        self.journal = self.journal_integration.journal
        
        # Store original methods for wrapping
        self._original_record_trade = executor.record_trade
        self._original_route_trade = executor.route_trade
        self._original_exit_trade = executor.exit_trade
        
        # Override executor methods with journaling-enhanced versions
        if auto_journal:
            executor.record_trade = self._journaled_record_trade
            executor.route_trade = self._journaled_route_trade
            executor.exit_trade = self._journaled_exit_trade
        
        # Logger
        self.logger = logging.getLogger("JournaledTradeExecutor")
        self.logger.info("Journal integration enabled for TradeExecutor")
    
    def _journaled_record_trade(self, trade_id: str, **trade_details):
        """Enhanced record_trade method that uses the journal system."""
        # First call the original method for backward compatibility
        self._original_record_trade(trade_id, **trade_details)
        
        # Then record in the trade journal system if it's not an exit
        if trade_details.get("trade_type") != "exit":
            self._journal_entry(trade_id, trade_details)
    
    def _journaled_route_trade(self, trade_signal):
        """Enhanced route_trade method with journaling."""
        # Execute trade using original method
        result = self._original_route_trade(trade_signal)
        
        # If successful, ensure it's in our journal with enriched data
        if result.success and self.auto_journal:
            # Add market context data
            self._add_market_context(result.trade_id)
        
        return result
    
    def _journaled_exit_trade(self, trade_id, reason="manual"):
        """Enhanced exit_trade method with journaling."""
        # Execute exit using original method
        result = self._original_exit_trade(trade_id, reason)
        
        # If successful, record the exit in our journal
        if result.success and self.auto_journal:
            self._journal_exit(trade_id, result, reason)
            
            # Add execution evaluation
            self._add_execution_evaluation(trade_id, result)
        
        return result
    
    def _journal_entry(self, trade_id, trade_details):
        """Create journal entry for a trade."""
        try:
            # Format data for journal
            formatted_data = self._format_entry_data(trade_id, trade_details)
            
            # Use journal integration to process new trade with enriched context
            self.journal_integration.process_new_trade(formatted_data)
        except Exception as e:
            self.logger.error(f"Error journaling trade {trade_id}: {e}")
    
    def _journal_exit(self, trade_id, result, reason):
        """Record trade exit in journal."""
        try:
            # Get exit details from the result
            exit_data = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.datetime.now().strftime("%H:%M %p ET"),
                "price": result.details.get("exit_price", 0),
                "quantity": result.quantity,
                "exit_reason": reason,
                "commission_fees": 0  # Add if available in your broker data
            }
            
            # Close the trade in the journal system
            self.journal_integration.process_trade_close(trade_id, exit_data)
        except Exception as e:
            self.logger.error(f"Error journaling trade exit {trade_id}: {e}")
    
    def _format_entry_data(self, trade_id, trade_details):
        """Format trade data for journal entry."""
        # Get basic trade details
        symbol = trade_details.get("symbol", "")
        strategy = trade_details.get("strategy", "")
        direction = trade_details.get("direction", "")
        quantity = trade_details.get("quantity", 0)
        price = trade_details.get("price", 0)
        order_type = trade_details.get("order_type", "")
        trade_type = trade_details.get("trade_type", "equity")
        
        # Format for journal template
        formatted = {
            "trade_metadata": {
                "trade_id": trade_id,
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.datetime.now().isoformat(),
                "ticker": symbol,
                "underlying_ticker": symbol,
                "asset_class": trade_type,
                "position_type": "long" if "buy" in direction.lower() else "short",
                "position_details": {
                    "primary_strategy": strategy,
                    "strategy_variant": ""
                }
            },
            "execution_details": {
                "entry": {
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.datetime.now().strftime("%H:%M %p ET"),
                    "price": price,
                    "quantity": quantity,
                    "order_type": order_type,
                    "commission_fees": trade_details.get("commission", 0)
                }
            },
            "trade_rationale": {
                "entry_criteria": {
                    "strategy_signals": {
                        "primary_signal": strategy
                    }
                },
                "exit_criteria": {
                    "planned_target": {
                        "price": trade_details.get("profit_target", 0)
                    },
                    "planned_stop": {
                        "price": trade_details.get("stop_loss", 0)
                    }
                }
            }
        }
        
        # Add options specific details if applicable
        if trade_type == "options" and "option_details" in trade_details:
            option_details = trade_details["option_details"]
            formatted["trade_metadata"]["position_details"]["option_specific"] = {
                "contract_type": option_details.get("option_type", ""),
                "expiration_date": option_details.get("expiration", ""),
                "strike_price": option_details.get("strike", 0),
                "days_to_expiration_entry": self._calculate_days_to_expiration(option_details.get("expiration", ""))
            }
        
        # Add stop loss and profit target if available from risk manager
        if trade_id in self.executor.risk_manager.open_trades:
            trade = self.executor.risk_manager.open_trades[trade_id]
            if "stop_loss" in trade and "profit_target" in trade:
                if "exit_criteria" not in formatted["trade_rationale"]:
                    formatted["trade_rationale"]["exit_criteria"] = {}
                
                formatted["trade_rationale"]["exit_criteria"]["planned_stop"] = {
                    "price": trade["stop_loss"],
                    "justification": "Risk management stop"
                }
                
                formatted["trade_rationale"]["exit_criteria"]["planned_target"] = {
                    "price": trade["profit_target"],
                    "justification": "Risk-reward based target"
                }
        
        return formatted
    
    def _calculate_days_to_expiration(self, expiration_date):
        """Calculate days to expiration for options."""
        if not expiration_date or not isinstance(expiration_date, str):
            return 0
            
        try:
            # Parse the expiration date
            if '-' in expiration_date:
                # Format like '2023-01-20'
                exp_date = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
            else:
                # Try other formats
                return 0
                
            # Calculate days between now and expiration
            days = (exp_date - datetime.datetime.now().date()).days
            return max(0, days)
        except:
            return 0
    
    def _add_market_context(self, trade_id):
        """Add market context data to a journal entry."""
        try:
            # Get the trade from risk manager
            if trade_id not in self.executor.risk_manager.open_trades:
                return
                
            trade = self.executor.risk_manager.open_trades[trade_id]
            symbol = trade.get("symbol", "")
            
            # Build market context data
            market_context = {}
            
            # Get market regime from risk manager
            market_context["market_regime"] = {
                "primary_regime": self.executor.risk_manager.risk_profile.get("current_regime", "unknown")
            }
            
            # Get sector information if available
            if hasattr(self.executor.risk_manager, "symbol_sectors") and symbol in self.executor.risk_manager.symbol_sectors:
                sector = self.executor.risk_manager.symbol_sectors[symbol]
                market_context["sector_context"] = {
                    "sector_performance": {
                        "sector_ranking": 0  # Placeholder - would come from actual data
                    },
                    "sector_sentiment": "neutral"  # Default
                }
            
            # Add volatility data if available
            if hasattr(self.executor, "market_data"):
                market_data = self.executor.market_data.get_market_data(symbol)
                if market_data and market_data.volatility:
                    if "market_structure" not in market_context:
                        market_context["market_structure"] = {}
                    
                    market_context["market_structure"]["volatility_environment"] = {
                        "realized_volatility": market_data.volatility
                    }
                
                # Try to get VIX data
                try:
                    vix_data = self.executor.market_data.get_market_data("VIX")
                    if vix_data:
                        if "market_structure" not in market_context:
                            market_context["market_structure"] = {}
                        if "volatility_environment" not in market_context["market_structure"]:
                            market_context["market_structure"]["volatility_environment"] = {}
                            
                        market_context["market_structure"]["volatility_environment"]["vix_level"] = vix_data.current_price
                except:
                    pass
            
            # Add market context if we have data
            if market_context:
                self.journal.add_market_context(trade_id, market_context)
                
        except Exception as e:
            self.logger.error(f"Error adding market context for trade {trade_id}: {e}")
    
    def _add_execution_evaluation(self, trade_id, result):
        """Add execution evaluation to a completed trade."""
        try:
            # Check if the trade details are available
            if trade_id not in self.executor.risk_manager.open_trades:
                return
                
            trade = self.executor.risk_manager.open_trades.get(trade_id)
            if not trade:
                return
                
            # Get entry and exit details
            entry_price = trade.get("entry_price", 0)
            exit_price = result.details.get("exit_price", 0)
            target_price = trade.get("profit_target", 0)
            stop_price = trade.get("stop_loss", 0)
            
            # Determine execution grades based on circumstances
            entry_grade = "B"  # Default grade
            exit_grade = "B"  # Default grade
            
            # Check if hit target or stop
            reason = result.details.get("exit_reason", "manual")
            
            if reason == "profit_target" or (target_price > 0 and 
                                         ((exit_price >= target_price * 0.95 and trade.get("direction") == "buy") or
                                          (exit_price <= target_price * 1.05 and trade.get("direction") == "sell"))):
                exit_grade = "A"
            elif reason == "stop_loss" or (stop_price > 0 and 
                                        ((exit_price <= stop_price * 1.05 and trade.get("direction") == "buy") or
                                         (exit_price >= stop_price * 0.95 and trade.get("direction") == "sell"))):
                exit_grade = "C"
            
            # Create evaluation object
            evaluation = {
                "technical_execution": {
                    "entry_timing_grade": entry_grade,
                    "exit_timing_grade": exit_grade,
                    "trade_management_grade": "B",  # Default
                    "position_sizing_grade": "B",   # Default
                    "technical_analysis_grade": "B", # Default
                    "overall_technical_grade": self._calculate_overall_grade([entry_grade, exit_grade, "B", "B", "B"])
                },
                "psychological_execution": {
                    "emotional_state": "neutral",  # Default - could be user input
                    "decision_clarity": "medium",  # Default
                    "patience_score": "medium",    # Default
                    "discipline_to_plan": "moderate", # Default
                    "adaptation_appropriateness": "appropriate", # Default
                    "overall_psychological_grade": "B"  # Default
                },
                "process_adherence": {
                    "pre_trade_checklist_completed": True,
                    "followed_entry_rules": True,
                    "followed_exit_rules": True,
                    "followed_position_sizing_rules": True,
                    "followed_risk_management_rules": True,
                    "overall_process_grade": "A"  # Default
                }
            }
            
            # Add the evaluation to the journal
            self.journal.add_trade_evaluation(trade_id, evaluation)
            
            # Add basic lessons learned
            pnl = result.details.get("pnl", 0)
            success = pnl > 0
            
            lessons = {
                "key_observations": [
                    f"Trade {'succeeded' if success else 'failed'} with P&L of ${pnl:.2f}"
                ],
                "what_worked_well": [
                    "Strategy signal identification" if success else ""
                ] if success else [],
                "what_needs_improvement": [
                    "Entry timing could be improved" if not success else ""
                ] if not success else [],
                "follow_up_actions": [
                    "Review similar setups for pattern consistency"
                ]
            }
            
            self.journal.add_lessons_learned(trade_id, lessons)
            
        except Exception as e:
            self.logger.error(f"Error adding execution evaluation for trade {trade_id}: {e}")
    
    def _calculate_overall_grade(self, grades):
        """Calculate overall letter grade from individual grades."""
        # Convert letter grades to numeric values
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        
        # Filter out empty grades
        valid_grades = [g for g in grades if g in grade_values]
        
        if not valid_grades:
            return "B"  # Default
        
        # Calculate average
        avg = sum(grade_values[g] for g in valid_grades) / len(valid_grades)
        
        # Convert back to letter grade
        if avg >= 3.5:
            return "A"
        elif avg >= 2.5:
            return "B"
        elif avg >= 1.5:
            return "C"
        elif avg >= 0.5:
            return "D"
        else:
            return "F"
    
    def get_trade_analysis(self, trade_id):
        """Get comprehensive analysis for a specific trade."""
        # Get trade details from executor
        executor_trade = None
        if trade_id in self.executor.risk_manager.open_trades:
            executor_trade = self.executor.risk_manager.open_trades[trade_id]
        else:
            # Look in trade journal
            for entry in self.executor.trade_journal:
                if entry.get("trade_id") == trade_id:
                    executor_trade = entry
                    break
        
        # Get journal entry
        journal_trade = self.journal.get_trade(trade_id)
        
        # Combine data from both sources
        combined_analysis = {
            "trade_id": trade_id,
            "executor_data": executor_trade or {},
            "journal_data": journal_trade or {},
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add strategy details if available
        if executor_trade and "strategy" in executor_trade:
            strategy_name = executor_trade["strategy"]
            if hasattr(self.executor, "loader") and hasattr(self.executor.loader, "get_strategy"):
                strategy = self.executor.loader.get_strategy(strategy_name)
                if strategy:
                    combined_analysis["strategy_details"] = strategy
        
        return combined_analysis
    
    def get_performance_report(self, period="all", strategy=None):
        """Get comprehensive performance report from both systems."""
        # Get executor metrics
        executor_metrics = self.executor.get_performance_metrics(period, strategy)
        
        # Get journal metrics
        journal_analysis = {}
        filtered_trades = self.journal.get_all_trades(closed_only=True)
        
        # Filter by strategy if needed
        if strategy:
            filtered_trades = [t for t in filtered_trades 
                              if t.get("trade_metadata", {}).get("position_details", {}).get("primary_strategy") == strategy]
        
        # Get journal analysis
        journal_analysis = self.journal.analyze_trades(filtered_trades)
        
        # Combine metrics
        combined_report = {
            "period": period,
            "strategy": strategy,
            "executor_metrics": executor_metrics,
            "journal_metrics": journal_analysis,
            "generated_at": datetime.datetime.now().isoformat(),
            "account_balance": self.executor.account_balance,
            "open_trades_count": len(self.executor.risk_manager.open_trades)
        }
        
        return combined_report
        
    def export_journal_to_file(self, format="json", filepath=None):
        """Export the trade journal to a file."""
        return self.journal.export_data(format=format, file_path=filepath)


def apply_journal_to_executor(executor, journal_dir="journal"):
    """
    Factory function to easily add journaling to an existing TradeExecutor.
    
    Args:
        executor: Existing TradeExecutor instance
        journal_dir: Directory for journal data
        
    Returns:
        JournaledTradeExecutor instance
    """
    return JournaledTradeExecutor(executor, journal_dir) 