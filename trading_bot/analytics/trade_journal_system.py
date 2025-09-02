"""
Trade Journal System

This module provides a comprehensive trade journal system for tracking, analyzing,
and improving trading performance. It integrates with the existing TradeLogger and
MetricsTracker classes to provide enhanced journaling capabilities.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid

# Import existing analytics modules
from .trade_logger import TradeLogger
from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)

class TradeJournalSystem:
    """
    Comprehensive trade journal system for tracking, analyzing, and improving trading performance.
    
    This system extends the basic trade logging functionality with:
    - Detailed trade metadata capture
    - Market context recording
    - Performance analytics
    - Psychological tracking
    - Strategy evolution support
    - AI-assisted insights
    """
    
    def __init__(self, journal_dir: str = "journal", config_path: str = None):
        """
        Initialize the trade journal system.
        
        Args:
            journal_dir: Directory for storing journal data
            config_path: Optional path to configuration file
        """
        self.journal_dir = journal_dir
        self.trades_dir = os.path.join(journal_dir, "trades")
        self.analytics_dir = os.path.join(journal_dir, "analytics")
        self.templates_dir = os.path.join(journal_dir, "templates")
        
        # Ensure directories exist
        os.makedirs(self.journal_dir, exist_ok=True)
        os.makedirs(self.trades_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize existing analytics modules
        self.trade_logger = TradeLogger(os.path.join(journal_dir, "logs"))
        self.metrics_tracker = MetricsTracker(os.path.join(journal_dir, "logs"))
        
        # Initialize journal template
        self.journal_template = self.config.get("journal_template", {})
        if not self.journal_template:
            self._initialize_default_template()
        
        # Track current trade being journaled
        self.current_trade = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # If no config file or error loading, initialize with empty config
        return {"schema_version": "2.1.0", "journal_template": {}}
    
    def _initialize_default_template(self):
        """Initialize the default journal template."""
        template_path = os.path.join(self.templates_dir, "default_template.json")
        
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    self.journal_template = json.load(f)
                    return
            except Exception as e:
                logger.error(f"Error loading default template: {str(e)}")
        
        # Create basic template if none exists
        self.journal_template = {
            "trade_metadata": {
                "trade_id": "",
                "date": "",
                "timestamp": "",
                "ticker": "",
                "asset_class": "",
                "position_type": ""
            },
            "execution_details": {
                "entry": {
                    "date": "",
                    "time": "",
                    "price": 0,
                    "quantity": 0
                },
                "exit": {
                    "date": "",
                    "time": "",
                    "price": 0,
                    "quantity": 0,
                    "exit_reason": ""
                }
            },
            "performance_metrics": {
                "profit_loss": {
                    "net_pnl_dollars": 0,
                    "pnl_percent": 0,
                    "win_loss": ""
                }
            },
            "trade_rationale": {
                "entry_criteria": {
                    "strategy_signals": {
                        "primary_signal": ""
                    }
                },
                "exit_criteria": {
                    "planned_target": {
                        "price": 0
                    },
                    "planned_stop": {
                        "price": 0
                    }
                }
            },
            "market_context": {
                "market_regime": {
                    "primary_regime": ""
                }
            },
            "execution_evaluation": {
                "technical_execution": {
                    "overall_technical_grade": ""
                },
                "psychological_execution": {
                    "emotional_state": ""
                }
            },
            "lessons_and_improvements": {
                "key_observations": [],
                "what_worked_well": [],
                "what_needs_improvement": []
            }
        }
        
        # Save default template
        with open(template_path, 'w') as f:
            json.dump(self.journal_template, f, indent=2)
    
    def start_new_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Start journaling a new trade.
        
        Args:
            trade_data: Initial trade data
            
        Returns:
            str: Trade ID
        """
        # Generate a unique trade ID if not provided
        if "trade_id" not in trade_data:
            trade_id = f"{datetime.now().strftime('%Y-%m-%d')}-{trade_data.get('ticker', 'UNKNOWN')}-{str(uuid.uuid4())[:8]}"
            trade_data["trade_id"] = trade_id
        else:
            trade_id = trade_data["trade_id"]
        
        # Initialize new trade from template
        self.current_trade = self._create_trade_from_template()
        
        # Record initial trade data
        self._update_trade_with_data(self.current_trade, trade_data)
        
        # Log basic trade info to the trade logger
        self.trade_logger.log_trade({
            "timestamp": trade_data.get("timestamp", datetime.now().isoformat()),
            "order_id": trade_id,
            "strategy": trade_data.get("primary_strategy", ""),
            "symbol": trade_data.get("ticker", ""),
            "asset_type": trade_data.get("asset_class", "equity"),
            "action": trade_data.get("action", ""),
            "quantity": trade_data.get("quantity", 0),
            "price": trade_data.get("price", 0),
            "commission": trade_data.get("commission_fees", 0),
            "status": "open"
        })
        
        # Save the initial trade journal entry
        self._save_current_trade()
        
        logger.info(f"Started journaling trade {trade_id}")
        return trade_id
    
    def _create_trade_from_template(self) -> Dict[str, Any]:
        """Create a new trade object based on the template."""
        # Deep copy the template to avoid modifying it
        import copy
        return copy.deepcopy(self.journal_template)
    
    def _update_trade_with_data(self, trade_obj: Dict[str, Any], data: Dict[str, Any], prefix: str = ""):
        """
        Recursively update trade object with data.
        
        Args:
            trade_obj: Trade object to update
            data: Data to update with
            prefix: Current key prefix for nested objects
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Handle nested objects
            if isinstance(value, dict) and key in trade_obj and isinstance(trade_obj[key], dict):
                self._update_trade_with_data(trade_obj[key], value, full_key)
            # Handle lists specially to avoid overwriting entire lists when might want to append
            elif isinstance(value, list) and key in trade_obj and isinstance(trade_obj[key], list):
                # Special case for arrays that should be appended vs. replaced
                if key in ["key_observations", "what_worked_well", "what_needs_improvement",
                          "adjustment_to_strategy", "follow_up_actions"]:
                    trade_obj[key].extend(value)
                else:
                    trade_obj[key] = value
            # Direct update for simple values
            else:
                if key in trade_obj:
                    trade_obj[key] = value
                else:
                    logger.debug(f"Key {full_key} not found in trade template")
    
    def update_trade(self, trade_id: str, trade_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade with new data.
        
        Args:
            trade_id: ID of the trade to update
            trade_data: New trade data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # If this is the current trade being worked on
        if self.current_trade and self.current_trade.get("trade_metadata", {}).get("trade_id") == trade_id:
            self._update_trade_with_data(self.current_trade, trade_data)
            self._save_current_trade()
            return True
        
        # Otherwise, load the trade, update it, and save it
        trade_path = os.path.join(self.trades_dir, f"{trade_id}.json")
        if not os.path.exists(trade_path):
            logger.error(f"Trade {trade_id} not found")
            return False
        
        try:
            with open(trade_path, 'r') as f:
                trade_obj = json.load(f)
            
            self._update_trade_with_data(trade_obj, trade_data)
            
            with open(trade_path, 'w') as f:
                json.dump(trade_obj, f, indent=2)
            
            logger.info(f"Updated trade {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {str(e)}")
            return False
    
    def close_trade(self, trade_id: str, exit_data: Dict[str, Any]) -> bool:
        """
        Close a trade and complete its journal entry.
        
        Args:
            trade_id: ID of the trade to close
            exit_data: Exit data including price, date, time, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get the trade object
        if self.current_trade and self.current_trade.get("trade_metadata", {}).get("trade_id") == trade_id:
            trade_obj = self.current_trade
        else:
            trade_path = os.path.join(self.trades_dir, f"{trade_id}.json")
            if not os.path.exists(trade_path):
                logger.error(f"Trade {trade_id} not found")
                return False
            
            try:
                with open(trade_path, 'r') as f:
                    trade_obj = json.load(f)
            except Exception as e:
                logger.error(f"Error reading trade {trade_id}: {str(e)}")
                return False
        
        # Update exit details
        if "execution_details" in trade_obj and "exit" in trade_obj["execution_details"]:
            self._update_trade_with_data(trade_obj["execution_details"]["exit"], exit_data)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(trade_obj, exit_data)
        
        # Calculate trade duration
        self._calculate_trade_duration(trade_obj)
        
        # Update the trade logger with exit
        ticker = trade_obj.get("trade_metadata", {}).get("ticker", "")
        asset_class = trade_obj.get("trade_metadata", {}).get("asset_class", "equity")
        entry_price = trade_obj.get("execution_details", {}).get("entry", {}).get("price", 0)
        exit_price = exit_data.get("price", 0)
        quantity = trade_obj.get("execution_details", {}).get("entry", {}).get("quantity", 0)
        
        self.trade_logger.log_trade({
            "timestamp": exit_data.get("date", "") + "T" + exit_data.get("time", "").replace(" ", "") + ":00",
            "order_id": trade_id,
            "strategy": trade_obj.get("trade_metadata", {}).get("position_details", {}).get("primary_strategy", ""),
            "symbol": ticker,
            "asset_type": asset_class,
            "action": "sell" if asset_class == "equity" else "sell_to_close",
            "quantity": quantity,
            "price": exit_price,
            "commission": exit_data.get("commission_fees", 0),
            "status": "closed"
        })
        
        # Save the updated trade
        if self.current_trade and self.current_trade.get("trade_metadata", {}).get("trade_id") == trade_id:
            self.current_trade = trade_obj
            self._save_current_trade()
            self.current_trade = None  # Clear current trade since it's closed
        else:
            trade_path = os.path.join(self.trades_dir, f"{trade_id}.json")
            with open(trade_path, 'w') as f:
                json.dump(trade_obj, f, indent=2)
        
        # Update metrics
        self.metrics_tracker.update_metrics()
        
        logger.info(f"Closed trade {trade_id}")
        return True
    
    def _calculate_performance_metrics(self, trade_obj: Dict[str, Any], exit_data: Dict[str, Any]):
        """Calculate performance metrics for a trade."""
        try:
            # Get trade details
            entry_price = trade_obj.get("execution_details", {}).get("entry", {}).get("price", 0)
            exit_price = exit_data.get("price", 0)
            quantity = trade_obj.get("execution_details", {}).get("entry", {}).get("quantity", 0)
            position_type = trade_obj.get("trade_metadata", {}).get("position_type", "long")
            
            # Calculate P&L
            if position_type.lower() == "long":
                gross_pnl = (exit_price - entry_price) * quantity
            else:  # short
                gross_pnl = (entry_price - exit_price) * quantity
            
            # Calculate fees
            entry_commission = trade_obj.get("execution_details", {}).get("entry", {}).get("commission_fees", 0)
            exit_commission = exit_data.get("commission_fees", 0)
            total_fees = entry_commission + exit_commission
            
            # Net P&L
            net_pnl = gross_pnl - total_fees
            
            # P&L percent
            if position_type.lower() == "long":
                pnl_percent = ((exit_price / entry_price) - 1) * 100
            else:  # short
                pnl_percent = ((entry_price / exit_price) - 1) * 100
            
            # Win or loss
            win_loss = "win" if net_pnl > 0 else "loss" if net_pnl < 0 else "breakeven"
            
            # Max favorable/adverse excursion (placeholder - would need price data series)
            max_favorable = {
                "dollars": net_pnl if net_pnl > 0 else 0,
                "percent": pnl_percent if pnl_percent > 0 else 0,
                "date": exit_data.get("date", ""),
                "time": exit_data.get("time", "")
            }
            
            max_adverse = {
                "dollars": net_pnl if net_pnl < 0 else 0,
                "percent": pnl_percent if pnl_percent < 0 else 0,
                "date": exit_data.get("date", ""),
                "time": exit_data.get("time", "")
            }
            
            # Risk metrics
            planned_stop = trade_obj.get("trade_rationale", {}).get("exit_criteria", {}).get("planned_stop", {}).get("price", 0)
            initial_risk = 0
            risk_reward_planned = 0
            
            if planned_stop > 0:
                if position_type.lower() == "long":
                    initial_risk = (entry_price - planned_stop) * quantity
                else:  # short
                    initial_risk = (planned_stop - entry_price) * quantity
                
                if initial_risk > 0:
                    risk_reward_planned = abs(net_pnl / initial_risk)
            
            # Initial risk percent
            account_value = 10000  # Default or could be retrieved from config
            initial_risk_percent = (initial_risk / account_value) * 100 if account_value > 0 else 0
            
            # Update the trade object with calculated metrics
            if "performance_metrics" in trade_obj and "profit_loss" in trade_obj["performance_metrics"]:
                profit_loss = trade_obj["performance_metrics"]["profit_loss"]
                profit_loss["net_pnl_dollars"] = net_pnl
                profit_loss["gross_pnl_dollars"] = gross_pnl
                profit_loss["pnl_percent"] = pnl_percent
                profit_loss["win_loss"] = win_loss
            
            # Update position metrics if they exist
            if "performance_metrics" in trade_obj and "position_metrics" in trade_obj["performance_metrics"]:
                position_metrics = trade_obj["performance_metrics"]["position_metrics"]
                
                if "max_favorable_excursion" in position_metrics:
                    position_metrics["max_favorable_excursion"] = max_favorable
                
                if "max_adverse_excursion" in position_metrics:
                    position_metrics["max_adverse_excursion"] = max_adverse
            
            # Update risk metrics if they exist
            if "performance_metrics" in trade_obj and "risk_metrics" in trade_obj["performance_metrics"]:
                risk_metrics = trade_obj["performance_metrics"]["risk_metrics"]
                
                if "initial_risk_amount" in risk_metrics:
                    risk_metrics["initial_risk_amount"] = initial_risk
                
                if "initial_risk_percent" in risk_metrics:
                    risk_metrics["initial_risk_percent"] = initial_risk_percent
                
                if "risk_reward_planned" in risk_metrics:
                    risk_metrics["risk_reward_planned"] = risk_reward_planned
                
                if "risk_reward_actual" in risk_metrics and initial_risk > 0:
                    risk_metrics["risk_reward_actual"] = abs(net_pnl / initial_risk)
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
    
    def _calculate_trade_duration(self, trade_obj: Dict[str, Any]):
        """Calculate the duration of a trade."""
        try:
            entry_date_str = trade_obj.get("execution_details", {}).get("entry", {}).get("date", "")
            entry_time_str = trade_obj.get("execution_details", {}).get("entry", {}).get("time", "")
            exit_date_str = trade_obj.get("execution_details", {}).get("exit", {}).get("date", "")
            exit_time_str = trade_obj.get("execution_details", {}).get("exit", {}).get("time", "")
            
            if not entry_date_str or not exit_date_str:
                return
            
            # Parse dates and times
            try:
                entry_datetime = pd.to_datetime(f"{entry_date_str} {entry_time_str}")
                exit_datetime = pd.to_datetime(f"{exit_date_str} {exit_time_str}")
            except:
                return
            
            # Calculate duration
            duration = exit_datetime - entry_datetime
            
            # Duration in days, hours, minutes
            days = duration.days
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            # Trading sessions (simplified - assuming 1 session per day when market is open)
            trading_sessions = max(1, days)
            
            # Update trade duration in the trade object
            if "execution_details" in trade_obj and "trade_duration" in trade_obj["execution_details"]:
                trade_duration = trade_obj["execution_details"]["trade_duration"]
                trade_duration["days"] = days
                trade_duration["hours"] = hours
                trade_duration["minutes"] = minutes
                trade_duration["trading_sessions"] = trading_sessions
                
        except Exception as e:
            logger.error(f"Error calculating trade duration: {str(e)}")
    
    def _save_current_trade(self):
        """Save the current trade to disk."""
        if not self.current_trade:
            logger.warning("No current trade to save")
            return
        
        trade_id = self.current_trade.get("trade_metadata", {}).get("trade_id", "")
        if not trade_id:
            logger.error("Current trade has no ID")
            return
        
        trade_path = os.path.join(self.trades_dir, f"{trade_id}.json")
        
        try:
            with open(trade_path, 'w') as f:
                json.dump(self.current_trade, f, indent=2)
            logger.debug(f"Saved trade {trade_id}")
        except Exception as e:
            logger.error(f"Error saving trade {trade_id}: {str(e)}")
    
    def get_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Get a specific trade by ID.
        
        Args:
            trade_id: ID of the trade to retrieve
            
        Returns:
            dict: Trade data
        """
        if self.current_trade and self.current_trade.get("trade_metadata", {}).get("trade_id") == trade_id:
            return self.current_trade
        
        trade_path = os.path.join(self.trades_dir, f"{trade_id}.json")
        if not os.path.exists(trade_path):
            logger.error(f"Trade {trade_id} not found")
            return {}
        
        try:
            with open(trade_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading trade {trade_id}: {str(e)}")
            return {}
    
    def get_all_trades(self, days: int = None, closed_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all trades, optionally filtered.
        
        Args:
            days: If set, only return trades from the last n days
            closed_only: If True, only return closed trades
            
        Returns:
            list: List of trade data dictionaries
        """
        trades = []
        
        # Get all trade files
        for filename in os.listdir(self.trades_dir):
            if not filename.endswith(".json"):
                continue
            
            trade_path = os.path.join(self.trades_dir, filename)
            
            try:
                with open(trade_path, 'r') as f:
                    trade = json.load(f)
                
                # Filter by closed status if requested
                if closed_only:
                    exit_price = trade.get("execution_details", {}).get("exit", {}).get("price", 0)
                    if exit_price <= 0:
                        continue
                
                # Filter by date if requested
                if days:
                    trade_date_str = trade.get("trade_metadata", {}).get("date", "")
                    if not trade_date_str:
                        continue
                    
                    try:
                        trade_date = pd.to_datetime(trade_date_str)
                        cutoff_date = pd.to_datetime('today') - pd.Timedelta(days=days)
                        if trade_date < cutoff_date:
                            continue
                    except:
                        continue
                
                trades.append(trade)
                
            except Exception as e:
                logger.error(f"Error reading trade file {filename}: {str(e)}")
        
        return trades
    
    def analyze_trades(self, trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a set of trades to generate insights.
        
        Args:
            trades: List of trades to analyze, if None, analyze all trades
            
        Returns:
            dict: Analysis results
        """
        if trades is None:
            trades = self.get_all_trades(closed_only=True)
        
        if not trades:
            return {"error": "No trades available for analysis"}
        
        # Basic performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss") == "win")
        losing_trades = sum(1 for t in trades if t.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss") == "loss")
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate total P&L
        total_pnl = sum(t.get("performance_metrics", {}).get("profit_loss", {}).get("net_pnl_dollars", 0) for t in trades)
        
        # Average metrics
        avg_win = np.mean([t.get("performance_metrics", {}).get("profit_loss", {}).get("net_pnl_dollars", 0) 
                          for t in trades if t.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss") == "win"] or [0])
        
        avg_loss = np.mean([t.get("performance_metrics", {}).get("profit_loss", {}).get("net_pnl_dollars", 0) 
                           for t in trades if t.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss") == "loss"] or [0])
        
        # Analysis by strategy
        strategies = {}
        for trade in trades:
            strategy = trade.get("trade_metadata", {}).get("position_details", {}).get("primary_strategy", "unknown")
            
            if strategy not in strategies:
                strategies[strategy] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0
                }
            
            strategies[strategy]["trades"] += 1
            
            win_loss = trade.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss")
            if win_loss == "win":
                strategies[strategy]["wins"] += 1
            elif win_loss == "loss":
                strategies[strategy]["losses"] += 1
            
            pnl = trade.get("performance_metrics", {}).get("profit_loss", {}).get("net_pnl_dollars", 0)
            strategies[strategy]["total_pnl"] += pnl
        
        # Calculate win rates and average P&L for each strategy
        for strategy in strategies:
            total = strategies[strategy]["trades"]
            wins = strategies[strategy]["wins"]
            strategies[strategy]["win_rate"] = wins / total if total > 0 else 0
            strategies[strategy]["avg_pnl"] = strategies[strategy]["total_pnl"] / total if total > 0 else 0
        
        # Psychological analysis - emotional states and their impact
        emotional_states = {}
        for trade in trades:
            state = trade.get("execution_evaluation", {}).get("psychological_execution", {}).get("emotional_state", "unknown")
            
            if state not in emotional_states:
                emotional_states[state] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0
                }
            
            emotional_states[state]["trades"] += 1
            
            if trade.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss") == "win":
                emotional_states[state]["wins"] += 1
            
            pnl = trade.get("performance_metrics", {}).get("profit_loss", {}).get("net_pnl_dollars", 0)
            emotional_states[state]["total_pnl"] += pnl
        
        # Calculate win rates and average P&L for each emotional state
        for state in emotional_states:
            total = emotional_states[state]["trades"]
            wins = emotional_states[state]["wins"]
            emotional_states[state]["win_rate"] = wins / total if total > 0 else 0
            emotional_states[state]["avg_pnl"] = emotional_states[state]["total_pnl"] / total if total > 0 else 0
        
        # Return all analysis results
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "strategy_analysis": strategies,
            "emotional_analysis": emotional_states
        }
    
    def add_market_context(self, trade_id: str, market_data: Dict[str, Any]) -> bool:
        """
        Add market context to a trade.
        
        Args:
            trade_id: ID of the trade to update
            market_data: Market context data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First get the trade
        trade = self.get_trade(trade_id)
        if not trade:
            return False
        
        # Update the market context section
        if "market_context" in trade:
            self._update_trade_with_data(trade["market_context"], market_data)
            
            # Save the updated trade
            return self.update_trade(trade_id, {"market_context": trade["market_context"]})
        
        return False
    
    def add_trade_evaluation(self, trade_id: str, evaluation: Dict[str, Any]) -> bool:
        """
        Add execution evaluation to a trade.
        
        Args:
            trade_id: ID of the trade to update
            evaluation: Execution evaluation data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get the trade
        trade = self.get_trade(trade_id)
        if not trade:
            return False
        
        # Update the execution evaluation section
        if "execution_evaluation" in trade:
            self._update_trade_with_data(trade["execution_evaluation"], evaluation)
            
            # Save the updated trade
            return self.update_trade(trade_id, {"execution_evaluation": trade["execution_evaluation"]})
        
        return False
    
    def add_lessons_learned(self, trade_id: str, lessons: Dict[str, Any]) -> bool:
        """
        Add lessons learned to a trade.
        
        Args:
            trade_id: ID of the trade to update
            lessons: Lessons learned data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get the trade
        trade = self.get_trade(trade_id)
        if not trade:
            return False
        
        # Update the lessons and improvements section
        if "lessons_and_improvements" in trade:
            self._update_trade_with_data(trade["lessons_and_improvements"], lessons)
            
            # Save the updated trade
            return self.update_trade(trade_id, {"lessons_and_improvements": trade["lessons_and_improvements"]})
        
        return False
    
    def export_data(self, format: str = "json", file_path: str = None) -> Any:
        """
        Export journal data to various formats.
        
        Args:
            format: Output format ('json', 'csv', 'excel')
            file_path: Path to save the exported data
            
        Returns:
            Exported data or path where data was saved
        """
        # Get all trades
        trades = self.get_all_trades()
        
        if format.lower() == "json":
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(trades, f, indent=2)
                return file_path
            else:
                return trades
        
        elif format.lower() in ["csv", "excel"]:
            # Convert to DataFrame - flatten structure first
            flattened_trades = []
            for trade in trades:
                flat_trade = {}
                
                # Extract key fields (could be expanded to more fields as needed)
                flat_trade["trade_id"] = trade.get("trade_metadata", {}).get("trade_id", "")
                flat_trade["date"] = trade.get("trade_metadata", {}).get("date", "")
                flat_trade["ticker"] = trade.get("trade_metadata", {}).get("ticker", "")
                flat_trade["asset_class"] = trade.get("trade_metadata", {}).get("asset_class", "")
                flat_trade["position_type"] = trade.get("trade_metadata", {}).get("position_type", "")
                flat_trade["primary_strategy"] = trade.get("trade_metadata", {}).get("position_details", {}).get("primary_strategy", "")
                
                flat_trade["entry_price"] = trade.get("execution_details", {}).get("entry", {}).get("price", 0)
                flat_trade["entry_quantity"] = trade.get("execution_details", {}).get("entry", {}).get("quantity", 0)
                flat_trade["exit_price"] = trade.get("execution_details", {}).get("exit", {}).get("price", 0)
                flat_trade["exit_quantity"] = trade.get("execution_details", {}).get("exit", {}).get("quantity", 0)
                flat_trade["exit_reason"] = trade.get("execution_details", {}).get("exit", {}).get("exit_reason", "")
                
                flat_trade["net_pnl"] = trade.get("performance_metrics", {}).get("profit_loss", {}).get("net_pnl_dollars", 0)
                flat_trade["pnl_percent"] = trade.get("performance_metrics", {}).get("profit_loss", {}).get("pnl_percent", 0)
                flat_trade["win_loss"] = trade.get("performance_metrics", {}).get("profit_loss", {}).get("win_loss", "")
                
                flat_trade["market_regime"] = trade.get("market_context", {}).get("market_regime", {}).get("primary_regime", "")
                flat_trade["emotional_state"] = trade.get("execution_evaluation", {}).get("psychological_execution", {}).get("emotional_state", "")
                flat_trade["technical_grade"] = trade.get("execution_evaluation", {}).get("technical_execution", {}).get("overall_technical_grade", "")
                
                flattened_trades.append(flat_trade)
            
            df = pd.DataFrame(flattened_trades)
            
            if format.lower() == "csv":
                if file_path:
                    df.to_csv(file_path, index=False)
                    return file_path
                else:
                    return df.to_csv(index=False)
            
            elif format.lower() == "excel":
                if file_path:
                    df.to_excel(file_path, index=False)
                    return file_path
                else:
                    buffer = BytesIO()
                    df.to_excel(buffer, index=False)
                    buffer.seek(0)
                    return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}") 