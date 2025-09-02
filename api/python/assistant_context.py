#!/usr/bin/env python3
"""
Assistant Context Processor for Trading Bot

This module processes user queries about trading portfolios
and classifies them into predefined types for generating
appropriate responses.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum, auto

from .portfolio_state import PortfolioStateManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enumeration of possible query types."""
    PORTFOLIO_OVERVIEW = auto()
    POSITION_DETAILS = auto()
    PERFORMANCE_METRICS = auto()
    RECENT_ACTIVITY = auto()
    STRATEGY_INFO = auto()
    SYSTEM_STATUS = auto()
    LEARNING_STATUS = auto()
    UNKNOWN = auto()

class AssistantContext:
    """
    Context processor for BenBot Assistant to handle trading portfolio queries.
    
    This class:
    1. Classifies user queries into predefined types
    2. Fetches relevant data from the PortfolioStateManager
    3. Formats responses for the assistant to deliver
    """
    
    def __init__(self, portfolio_state_manager: Optional[PortfolioStateManager] = None):
        """
        Initialize the assistant context processor.
        
        Args:
            portfolio_state_manager: Optional instance of PortfolioStateManager.
                                    If None, a new instance will be created.
        """
        # Initialize the portfolio state manager
        self.portfolio_state = portfolio_state_manager or PortfolioStateManager()
        logger.info("Assistant context initialized with portfolio state manager")
        
        # Define regex patterns for query classification
        self._query_patterns = {
            QueryType.PORTFOLIO_OVERVIEW: [
                r"(?i).*portfolio\s+(?:overview|summary).*",
                r"(?i).*show\s+(?:me\s+)?(?:my\s+)?portfolio.*",
                r"(?i).*what(?:'s| is)\s+(?:in\s+)?my\s+portfolio.*",
                r"(?i).*portfolio\s+(?:status|holdings|positions).*",
                r"(?i).*portfolio\s+composition.*",
                r"(?i).*asset\s+allocation.*"
            ],
            QueryType.POSITION_DETAILS: [
                r"(?i).*position(?:s)?\s+(?:for|in|on)\s+([A-Z]{1,5}).*",
                r"(?i).*show\s+(?:me\s+)?([A-Z]{1,5})\s+position.*",
                r"(?i).*how\s+much\s+([A-Z]{1,5})\s+(?:do\s+)?(?:I|we)\s+(?:own|have).*",
                r"(?i).*([A-Z]{1,5})\s+(?:position|holding).*"
            ],
            QueryType.PERFORMANCE_METRICS: [
                r"(?i).*performance\s+(?:metrics|data|stats|statistics).*",
                r"(?i).*portfolio\s+performance.*",
                r"(?i).*how\s+(?:is|was)\s+(?:the\s+)?(?:portfolio|trading)\s+(?:doing|performing).*",
                r"(?i).*(?:sharpe|drawdown|volatility|returns).*",
                r"(?i).*profit\s+and\s+loss.*",
                r"(?i).*(?:pnl|p&l).*"
            ],
            QueryType.RECENT_ACTIVITY: [
                r"(?i).*recent\s+(?:trades|activity|signals).*",
                r"(?i).*latest\s+(?:trades|activity|signals).*",
                r"(?i).*what\s+(?:trades|signals)\s+(?:have\s+been\s+)?(?:generated|executed).*",
                r"(?i).*trading\s+(?:activity|history).*"
            ],
            QueryType.STRATEGY_INFO: [
                r"(?i).*(?:active\s+)?strategies.*",
                r"(?i).*strategy\s+(?:allocation|performance).*",
                r"(?i).*which\s+strategies\s+(?:are\s+running|being\s+used).*",
                r"(?i).*how\s+(?:are|is)\s+(?:the\s+)?strategies?\s+(?:performing|doing).*"
            ],
            QueryType.SYSTEM_STATUS: [
                r"(?i).*system\s+(?:status|health).*",
                r"(?i).*(?:is|are)\s+(?:the\s+)?(?:market|markets)\s+open.*",
                r"(?i).*(?:data\s+providers|brokers|connections).*",
                r"(?i).*market\s+hours.*",
                r"(?i).*system\s+(?:connectivity|connection).*"
            ],
            QueryType.LEARNING_STATUS: [
                r"(?i).*(?:learning|training)\s+(?:status|progress).*",
                r"(?i).*model\s+(?:status|training).*",
                r"(?i).*(?:is|are)\s+(?:the\s+)?models?\s+(?:training|learning).*",
                r"(?i).*when\s+was\s+the\s+last\s+training.*",
                r"(?i).*ai\s+(?:learning|training).*"
            ]
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query about the trading portfolio.
        
        Args:
            query: The user's query text
            
        Returns:
            A dictionary with response data and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Classify the query
        query_type, params = self._classify_query(query)
        logger.info(f"Classified as: {query_type.name}, params: {params}")
        
        # Get response data based on query type
        response_data = self._get_response_data(query_type, params)
        
        # Format the response
        formatted_response = self._format_response(query_type, response_data)
        
        # Return the full response context
        return {
            "query": query,
            "query_type": query_type.name,
            "params": params,
            "response_data": response_data,
            "formatted_response": formatted_response
        }
    
    def _classify_query(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """
        Classify the query into one of the predefined types.
        
        Args:
            query: The user's query text
            
        Returns:
            A tuple of (QueryType, parameters dictionary)
        """
        # Default to unknown query type
        query_type = QueryType.UNKNOWN
        params = {}
        
        # Check each query type's patterns
        for qt, patterns in self._query_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, query)
                if match:
                    query_type = qt
                    
                    # Extract parameters from regex groups if any
                    if match.groups():
                        # For position details, the first group is usually the symbol
                        if qt == QueryType.POSITION_DETAILS and len(match.groups()) > 0:
                            params["symbol"] = match.group(1)
                    
                    return query_type, params
        
        return query_type, params
    
    def _get_response_data(self, query_type: QueryType, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the data needed to respond to the query.
        
        Args:
            query_type: The type of query
            params: Parameters extracted from the query
            
        Returns:
            A dictionary with the data needed for the response
        """
        # Get data based on query type
        if query_type == QueryType.PORTFOLIO_OVERVIEW:
            return {
                "summary": self.portfolio_state.get_portfolio_summary(),
                "positions": self.portfolio_state.get_positions()
            }
        
        elif query_type == QueryType.POSITION_DETAILS:
            symbol = params.get("symbol", "")
            position = self.portfolio_state.get_position(symbol) if symbol else None
            
            return {
                "symbol": symbol,
                "position": position
            }
        
        elif query_type == QueryType.PERFORMANCE_METRICS:
            return {
                "metrics": self.portfolio_state.get_metrics()
            }
        
        elif query_type == QueryType.RECENT_ACTIVITY:
            return {
                "activity": self.portfolio_state.get_recent_activity()
            }
        
        elif query_type == QueryType.STRATEGY_INFO:
            return {
                "strategies": self.portfolio_state.get_strategy_data()
            }
        
        elif query_type == QueryType.SYSTEM_STATUS:
            return {
                "system": self.portfolio_state.get_system_status()
            }
        
        elif query_type == QueryType.LEARNING_STATUS:
            return {
                "learning": self.portfolio_state.get_learning_status()
            }
        
        # Unknown query type
        return {
            "error": "Could not understand query"
        }
    
    def _format_response(self, query_type: QueryType, data: Dict[str, Any]) -> str:
        """
        Format the response data into a human-readable string.
        
        Args:
            query_type: The type of query
            data: The response data dictionary
            
        Returns:
            A formatted response string
        """
        if query_type == QueryType.PORTFOLIO_OVERVIEW:
            summary = data.get("summary", {})
            positions = data.get("positions", {})
            
            # Format the portfolio overview response
            response = [
                f"Portfolio Overview:",
                f"Total Value: ${summary.get('total_value', 0):.2f}",
                f"Cash Balance: ${summary.get('cash', 0):.2f}",
                f"Number of Positions: {len(positions)}",
                "",
                "Top Positions:"
            ]
            
            # Show top positions by value
            sorted_positions = sorted(
                positions.items(), 
                key=lambda x: x[1].get('market_value', 0), 
                reverse=True
            )
            
            for symbol, pos in sorted_positions[:5]:  # Show top 5
                response.append(
                    f"  {symbol}: {pos.get('quantity', 0)} shares, "
                    f"${pos.get('market_value', 0):.2f} "
                    f"({pos.get('unrealized_pnl_percent', 0):.2f}%)"
                )
            
            # Add asset allocation if available
            asset_allocation = summary.get("asset_allocation", {})
            if asset_allocation:
                response.append("")
                response.append("Asset Allocation:")
                for asset_class, percentage in asset_allocation.items():
                    response.append(f"  {asset_class}: {percentage:.2f}%")
            
            return "\n".join(response)
        
        elif query_type == QueryType.POSITION_DETAILS:
            symbol = data.get("symbol", "")
            position = data.get("position")
            
            if not position:
                return f"No position found for {symbol}."
            
            # Format the position details response
            response = [
                f"Position Details for {symbol}:",
                f"Quantity: {position.get('quantity', 0)} shares",
                f"Average Cost: ${position.get('avg_price', 0):.2f}",
                f"Current Price: ${position.get('current_price', 0):.2f}",
                f"Market Value: ${position.get('market_value', 0):.2f}",
                f"Unrealized P&L: ${position.get('unrealized_pnl', 0):.2f} ({position.get('unrealized_pnl_percent', 0):.2f}%)"
            ]
            
            # Add any additional position information
            additional_info = {k: v for k, v in position.items() 
                              if k not in ['symbol', 'quantity', 'avg_price', 'current_price', 
                                          'market_value', 'unrealized_pnl', 'unrealized_pnl_percent']}
            
            if additional_info:
                response.append("")
                response.append("Additional Information:")
                for key, value in additional_info.items():
                    # Format key from snake_case to Title Case
                    formatted_key = ' '.join(word.capitalize() for word in key.split('_'))
                    response.append(f"  {formatted_key}: {value}")
            
            return "\n".join(response)
        
        elif query_type == QueryType.PERFORMANCE_METRICS:
            metrics = data.get("metrics", {})
            
            # Format the performance metrics response
            response = [
                f"Performance Metrics:",
                f"Cumulative Return: {metrics.get('cumulative_return', 0):.2f}%",
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
                f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%",
                f"Volatility: {metrics.get('volatility', 0):.2f}%",
                f"Win Rate: {metrics.get('win_rate', 0):.2f}%",
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}"
            ]
            
            # Add recent daily returns if available
            recent_returns = metrics.get("recent_daily_returns", [])
            if recent_returns:
                response.append("")
                response.append("Recent Daily Returns:")
                for i, ret in enumerate(recent_returns[-5:]):  # Show last 5 days
                    response.append(f"  Day {len(recent_returns) - i}: {ret:.2f}%")
            
            return "\n".join(response)
        
        elif query_type == QueryType.RECENT_ACTIVITY:
            activity = data.get("activity", {})
            recent_trades = activity.get("recent_trades", [])
            recent_signals = activity.get("recent_signals", [])
            
            # Format the recent activity response
            response = ["Recent Trading Activity:"]
            
            if recent_trades:
                response.append("")
                response.append("Recent Trades:")
                for trade in recent_trades[:5]:  # Show last 5 trades
                    timestamp = trade.get("timestamp", "").split("T")[0]  # Just the date part
                    response.append(
                        f"  {timestamp} - {trade.get('action', '')} {trade.get('quantity', 0)} "
                        f"{trade.get('symbol', '')} @ ${trade.get('price', 0):.2f}"
                    )
            
            if recent_signals:
                response.append("")
                response.append("Recent Signals:")
                for signal in recent_signals[:5]:  # Show last 5 signals
                    timestamp = signal.get("timestamp", "").split("T")[0]  # Just the date part
                    confidence = signal.get("confidence", 0) * 100 if isinstance(signal.get("confidence"), float) else signal.get("confidence", 0)
                    response.append(
                        f"  {timestamp} - {signal.get('signal_type', '')} on {signal.get('symbol', '')} "
                        f"(Confidence: {confidence:.1f}%)"
                    )
            
            if not recent_trades and not recent_signals:
                response.append("No recent trading activity found.")
            
            return "\n".join(response)
        
        elif query_type == QueryType.STRATEGY_INFO:
            strategies_data = data.get("strategies", {})
            active_strategies = strategies_data.get("active_strategies", [])
            strategy_allocations = strategies_data.get("strategy_allocations", {})
            strategy_performance = strategies_data.get("strategy_performance", {})
            
            # Format the strategy info response
            if not active_strategies:
                return "No active trading strategies found."
            
            response = [
                f"Active Trading Strategies:",
                f"Number of Active Strategies: {len(active_strategies)}"
            ]
            
            # Show strategy allocations
            if strategy_allocations:
                response.append("")
                response.append("Strategy Allocations:")
                for strategy, allocation in strategy_allocations.items():
                    response.append(f"  {strategy}: {allocation:.2f}%")
            
            # Show strategy performance
            if strategy_performance:
                response.append("")
                response.append("Strategy Performance:")
                for strategy, perf in strategy_performance.items():
                    if isinstance(perf, dict):
                        # If performance is a dictionary of metrics
                        returns = perf.get("returns", 0)
                        sharpe = perf.get("sharpe_ratio", 0)
                        response.append(f"  {strategy}: Return {returns:.2f}%, Sharpe {sharpe:.2f}")
                    else:
                        # If performance is a single value
                        response.append(f"  {strategy}: {perf}")
            
            return "\n".join(response)
        
        elif query_type == QueryType.SYSTEM_STATUS:
            system = data.get("system", {})
            
            # Format the system status response
            market_status = "Open" if system.get("is_market_open", False) else "Closed"
            
            response = [
                f"System Status:",
                f"Market Status: {market_status}",
                f"Market Hours: {system.get('market_hours', 'N/A')}"
            ]
            
            # Show data providers
            data_providers = system.get("data_providers", [])
            if data_providers:
                response.append("")
                response.append("Connected Data Providers:")
                for provider in data_providers:
                    response.append(f"  {provider}")
            
            # Show connected brokers
            brokers = system.get("connected_brokers", [])
            if brokers:
                response.append("")
                response.append("Connected Brokers:")
                for broker in brokers:
                    response.append(f"  {broker}")
            
            # Show system health
            health = system.get("system_health", {})
            if health:
                response.append("")
                response.append("System Health:")
                for component, status in health.items():
                    # Format component from snake_case to Title Case
                    formatted_component = ' '.join(word.capitalize() for word in component.split('_'))
                    response.append(f"  {formatted_component}: {status}")
            
            return "\n".join(response)
        
        elif query_type == QueryType.LEARNING_STATUS:
            learning = data.get("learning", {})
            
            # Format the learning status response
            training_status = "In Progress" if learning.get("training_in_progress", False) else "Not Active"
            last_training = learning.get("last_training_time", "Never")
            
            response = [
                f"Learning Status:",
                f"Training Status: {training_status}",
                f"Last Training: {last_training}"
            ]
            
            # Show model status
            models_status = learning.get("models_status", {})
            if models_status:
                response.append("")
                response.append("Model Status:")
                for model, status in models_status.items():
                    response.append(f"  {model}: {status}")
            
            # Show recent learning metrics
            metrics = learning.get("recent_learning_metrics", {})
            if metrics:
                response.append("")
                response.append("Recent Learning Metrics:")
                for model, model_metrics in metrics.items():
                    response.append(f"  {model}:")
                    if isinstance(model_metrics, dict):
                        for metric, value in model_metrics.items():
                            # Format metric from snake_case to Title Case
                            formatted_metric = ' '.join(word.capitalize() for word in metric.split('_'))
                            response.append(f"    {formatted_metric}: {value}")
                    else:
                        response.append(f"    {model_metrics}")
            
            return "\n".join(response)
        
        # Unknown query type or error
        if "error" in data:
            return f"I'm sorry, I couldn't process your query. {data['error']}"
        
        return "I'm sorry, I don't understand your question about the trading portfolio."
    
    def update_portfolio_state(self, state_updates: Dict[str, Any]) -> None:
        """
        Update the portfolio state with new data.
        
        Args:
            state_updates: Dictionary with state updates to apply
        """
        # Handle portfolio data updates
        portfolio_data = state_updates.get("portfolio_data")
        if portfolio_data:
            self.portfolio_state.update_portfolio_data(
                cash=portfolio_data.get("cash"),
                positions=portfolio_data.get("positions"),
                total_value=portfolio_data.get("total_value"),
                asset_allocation=portfolio_data.get("asset_allocation")
            )
        
        # Handle performance metrics updates
        performance_metrics = state_updates.get("performance_metrics")
        if performance_metrics:
            self.portfolio_state.update_performance_metrics(performance_metrics)
        
        # Handle trade updates
        trade = state_updates.get("trade")
        if trade:
            self.portfolio_state.add_trade(trade)
        
        # Handle signal updates
        signal = state_updates.get("signal")
        if signal:
            self.portfolio_state.add_signal(signal)
        
        # Handle strategy data updates
        strategy_data = state_updates.get("strategy_data")
        if strategy_data:
            self.portfolio_state.update_strategy_data(
                active_strategies=strategy_data.get("active_strategies"),
                strategy_allocations=strategy_data.get("strategy_allocations"),
                strategy_performance=strategy_data.get("strategy_performance")
            )
        
        # Handle system status updates
        system_status = state_updates.get("system_status")
        if system_status:
            self.portfolio_state.update_system_status(
                is_market_open=system_status.get("is_market_open"),
                market_hours=system_status.get("market_hours"),
                data_providers=system_status.get("data_providers"),
                connected_brokers=system_status.get("connected_brokers"),
                system_health=system_status.get("system_health")
            )
        
        # Handle learning status updates
        learning_status = state_updates.get("learning_status")
        if learning_status:
            self.portfolio_state.update_learning_status(
                training_in_progress=learning_status.get("training_in_progress"),
                models_status=learning_status.get("models_status"),
                recent_learning_metrics=learning_status.get("recent_learning_metrics")
            )
        
        logger.info("Portfolio state updated")

# Example usage
if __name__ == "__main__":
    # Create an assistant context with a new portfolio state manager
    assistant = AssistantContext()
    
    # Process a sample query
    response = assistant.process_query("What's in my portfolio?")
    print(json.dumps(response, indent=2))
    
    # Update with some sample data
    assistant.update_portfolio_state({
        "portfolio_data": {
            "cash": 10000.0,
            "total_value": 50000.0,
            "positions": {
                "AAPL": {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_price": 150.0,
                    "current_price": 170.0,
                    "market_value": 17000.0,
                    "unrealized_pnl": 2000.0,
                    "unrealized_pnl_percent": 13.33
                }
            }
        }
    })
    
    # Process the same query again to see updated data
    response = assistant.process_query("What's in my portfolio?")
    print(json.dumps(response, indent=2)) 