import json
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime

class AssistantContext:
    """
    AssistantContext provides a bridge between the trading system's PortfolioStateManager 
    and the BenBot Assistant. 
    
    This class is responsible for formatting trading system data into a structure that can be 
    easily consumed by the BenBot Assistant, enabling personalized and context-aware responses 
    regarding the portfolio, strategies, performance, and system status.
    """
    
    def __init__(self, portfolio_state_manager=None):
        """
        Initialize the AssistantContext with an optional portfolio state manager.
        
        Args:
            portfolio_state_manager: A PortfolioStateManager instance (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        self.portfolio_state_manager = portfolio_state_manager
        self.last_update = None
        
        self.logger.info("AssistantContext initialized")
    
    def set_portfolio_state_manager(self, portfolio_state_manager):
        """
        Set or update the portfolio state manager.
        
        Args:
            portfolio_state_manager: A PortfolioStateManager instance
        """
        with self.lock:
            self.portfolio_state_manager = portfolio_state_manager
            self.last_update = datetime.now()
        
        self.logger.info("Portfolio state manager updated")
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context formatted for the assistant.
        
        Returns:
            Dict containing the formatted context for the assistant
        """
        with self.lock:
            if self.portfolio_state_manager is None:
                self.logger.warning("Portfolio state manager not initialized")
                return {
                    "portfolio_initialized": False,
                    "message": "Trading system not initialized yet. Please initialize the portfolio state manager first."
                }
            
            try:
                # Get raw state from the portfolio state manager
                raw_state = self.portfolio_state_manager.get_current_state()
                
                # Format the state for the assistant
                formatted_context = self._format_state_for_assistant(raw_state)
                
                # Update last update timestamp
                self.last_update = datetime.now()
                
                return formatted_context
            
            except Exception as e:
                self.logger.error(f"Error getting context: {e}")
                return {
                    "portfolio_initialized": True,
                    "error": True,
                    "message": f"Error retrieving portfolio state: {str(e)}"
                }
    
    def _format_state_for_assistant(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the raw portfolio state into a structured context for the assistant.
        
        Args:
            raw_state: Raw state from the portfolio state manager
            
        Returns:
            Dict containing formatted context
        """
        try:
            # Create formatted context
            context = {
                "portfolio_initialized": True,
                "last_update": raw_state["system"]["last_update"],
                "portfolio_summary": self._format_portfolio_summary(raw_state),
                "strategies": self._format_strategies(raw_state),
                "performance": self._format_performance(raw_state),
                "recent_activity": self._format_recent_activity(raw_state),
                "system_status": self._format_system_status(raw_state)
            }
            
            return context
        
        except Exception as e:
            self.logger.error(f"Error formatting state: {e}")
            return {
                "portfolio_initialized": True,
                "error": True,
                "message": f"Error formatting portfolio state: {str(e)}"
            }
    
    def _format_portfolio_summary(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the portfolio summary section.
        
        Args:
            raw_state: Raw state from the portfolio state manager
            
        Returns:
            Dict containing formatted portfolio summary
        """
        portfolio = raw_state["portfolio"]
        
        # Count active positions
        active_positions = len(portfolio["positions"])
        
        # Format positions for display
        formatted_positions = []
        for symbol, position in portfolio["positions"].items():
            formatted_positions.append({
                "symbol": symbol,
                "quantity": position["quantity"],
                "value": position["market_value"],
                "pnl": position["unrealized_pnl"],
                "pnl_percent": (position["unrealized_pnl"] / position["market_value"]) * 100 if position["market_value"] > 0 else 0
            })
        
        # Sort positions by value (descending)
        formatted_positions.sort(key=lambda x: x["value"], reverse=True)
        
        # Take top 5 positions
        top_positions = formatted_positions[:5]
        
        # Calculate allocation percentages
        total_value = portfolio["total_value"]
        for position in top_positions:
            position["allocation"] = (position["value"] / total_value) * 100 if total_value > 0 else 0
        
        return {
            "total_value": portfolio["total_value"],
            "cash": portfolio["cash"],
            "cash_allocation": (portfolio["cash"] / total_value) * 100 if total_value > 0 else 0,
            "daily_pnl": portfolio["daily_pnl"],
            "daily_pnl_percent": portfolio["daily_pnl_percent"],
            "overall_pnl": portfolio["overall_pnl"],
            "overall_pnl_percent": portfolio["overall_pnl_percent"],
            "active_positions": active_positions,
            "top_positions": top_positions
        }
    
    def _format_strategies(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the strategies section.
        
        Args:
            raw_state: Raw state from the portfolio state manager
            
        Returns:
            Dict containing formatted strategies information
        """
        allocations = raw_state["allocations"]
        
        # Format allocations for display
        formatted_allocations = []
        for strategy_name, data in allocations.items():
            strategy_info = {
                "name": strategy_name,
                "allocation": data["allocation"]
            }
            
            # Add performance metrics if available
            if "performance" in data and data["performance"]:
                strategy_info["performance"] = data["performance"]
            
            formatted_allocations.append(strategy_info)
        
        # Sort allocations by allocation percentage (descending)
        formatted_allocations.sort(key=lambda x: x["allocation"], reverse=True)
        
        return {
            "active_strategies": len(allocations),
            "allocations": formatted_allocations
        }
    
    def _format_performance(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the performance metrics section.
        
        Args:
            raw_state: Raw state from the portfolio state manager
            
        Returns:
            Dict containing formatted performance metrics
        """
        metrics = raw_state["metrics"]
        risk = raw_state["risk"]
        
        return {
            "returns": {
                "daily": metrics["daily_return"],
                "weekly": metrics["weekly_return"],
                "monthly": metrics["monthly_return"],
                "yearly": metrics["yearly_return"]
            },
            "risk_metrics": {
                "sharpe_ratio": metrics["sharpe_ratio"],
                "volatility": metrics["volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "var_95": risk["var_95"],
                "var_99": risk["var_99"]
            },
            "trading_metrics": {
                "win_rate": metrics["win_rate"]
            }
        }
    
    def _format_recent_activity(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the recent activity section.
        
        Args:
            raw_state: Raw state from the portfolio state manager
            
        Returns:
            Dict containing formatted recent activity
        """
        # Get the 5 most recent trades
        recent_trades = raw_state["trades"][-5:] if raw_state["trades"] else []
        
        # Get the 5 most recent signals
        recent_signals = raw_state["signals"][-5:] if raw_state["signals"] else []
        
        return {
            "trades": recent_trades,
            "signals": recent_signals
        }
    
    def _format_system_status(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the system status section.
        
        Args:
            raw_state: Raw state from the portfolio state manager
            
        Returns:
            Dict containing formatted system status
        """
        system = raw_state["system"]
        learning = raw_state["learning"]
        
        return {
            "status": system["status"],
            "market_hours": system["market_hours"],
            "last_update": system["last_update"],
            "learning": {
                "rl_training_active": learning["rl_training"],
                "pattern_learning_active": learning["pattern_learning"],
                "training_progress": (learning["current_episode"] / learning["total_episodes"]) * 100 if learning["total_episodes"] > 0 else 0,
                "current_episode": learning["current_episode"],
                "total_episodes": learning["total_episodes"],
                "current_reward": learning["current_reward"],
                "best_reward": learning["best_reward"],
                "last_model_update": learning["last_model_update"]
            }
        }
    
    def get_response_context(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze user query and return relevant context tailored to the query.
        
        This method analyzes the user's query to determine what context is most relevant
        and returns a focused subset of the full context.
        
        Args:
            user_query: The user's query string
            
        Returns:
            Dict containing context most relevant to the query
        """
        query = user_query.lower()
        
        # Get full context
        full_context = self.get_context()
        
        # Return error message if there's an error
        if "error" in full_context and full_context["error"]:
            return full_context
        
        # Return initialization message if portfolio not initialized
        if not full_context.get("portfolio_initialized", False):
            return full_context
        
        # Prepare base response context
        response_context = {
            "portfolio_initialized": True,
            "last_update": full_context["last_update"]
        }
        
        # Identify query type
        is_portfolio_query = any(term in query for term in ["portfolio", "position", "holding", "value", "balance", "pnl", "profit", "loss", "cash"])
        is_strategy_query = any(term in query for term in ["strategy", "allocation", "allocated", "invest", "approach"])
        is_performance_query = any(term in query for term in ["performance", "return", "metric", "sharpe", "drawdown", "volatility", "risk", "ratio", "win rate"])
        is_trading_query = any(term in query for term in ["trade", "buy", "sell", "signal", "order", "transaction", "entry", "exit"])
        is_system_query = any(term in query for term in ["system", "status", "learning", "training", "model", "rl", "pattern", "market hours", "running"])
        
        # Add relevant sections based on query type
        if is_portfolio_query or not any([is_strategy_query, is_performance_query, is_trading_query, is_system_query]):
            # If portfolio query or general query, include portfolio summary
            response_context["portfolio_summary"] = full_context["portfolio_summary"]
        
        if is_strategy_query:
            response_context["strategies"] = full_context["strategies"]
        
        if is_performance_query:
            response_context["performance"] = full_context["performance"]
        
        if is_trading_query:
            response_context["recent_activity"] = full_context["recent_activity"]
        
        if is_system_query:
            response_context["system_status"] = full_context["system_status"]
        
        return response_context
    
    def to_json(self) -> str:
        """
        Convert the current context to a JSON string.
        
        Returns:
            JSON string representation of the context
        """
        context = self.get_context()
        return json.dumps(context, indent=2) 