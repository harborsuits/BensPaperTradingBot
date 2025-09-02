import os
import logging
import yaml
import json
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Strategy selector that uses market conditions and the core_strategies.yaml file
    to select the most appropriate trading strategies.
    """
    
    def __init__(self, config):
        """Initialize the strategy selector with configuration."""
        self.config = config
        self.strategies_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core_strategies.yaml")
        self.strategies_config = self._load_strategies_config()
        
        logger.info(f"Strategy Selector initialized with {len(self.strategies_config.get('strategies', {}))} strategies")
    
    def _load_strategies_config(self):
        """Load the strategies configuration from YAML file."""
        try:
            if os.path.exists(self.strategies_file):
                with open(self.strategies_file, "r") as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Strategies file not found: {self.strategies_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading strategies config: {str(e)}")
            return {}
    
    def get_available_strategies(self):
        """Get a list of all available strategies."""
        return list(self.strategies_config.get("strategies", {}).keys())
    
    def get_strategy_details(self, strategy_name):
        """Get details for a specific strategy."""
        return self.strategies_config.get("strategies", {}).get(strategy_name, {})
    
    def get_market_condition(self, market_indicators):
        """
        Determine the current market condition based on indicators.
        
        Args:
            market_indicators (dict): Dictionary of market indicators
        
        Returns:
            str: Market condition ('bullish', 'bearish', 'sideways', or 'high_volatility')
        """
        market_conditions = self.strategies_config.get("market_conditions", {})
        condition_scores = {}
        
        # Calculate score for each market condition based on matching indicators
        for condition, data in market_conditions.items():
            indicators = data.get("indicators", [])
            score = 0
            
            for indicator in indicators:
                # Parse indicator condition
                if indicator.startswith("spy_above"):
                    if market_indicators.get("spy_above_20dma", False) and market_indicators.get("spy_above_50dma", False):
                        score += 1
                elif indicator.startswith("spy_below"):
                    if market_indicators.get("spy_below_20dma", False) and market_indicators.get("spy_below_50dma", False):
                        score += 1
                elif indicator.startswith("spy_between"):
                    if market_indicators.get("spy_above_50dma", False) and not market_indicators.get("spy_above_20dma", False):
                        score += 1
                elif indicator.startswith("more_new_highs"):
                    if market_indicators.get("new_highs", 0) > market_indicators.get("new_lows", 0):
                        score += 1
                elif indicator.startswith("more_new_lows"):
                    if market_indicators.get("new_lows", 0) > market_indicators.get("new_highs", 0):
                        score += 1
                elif indicator.startswith("vix <"):
                    threshold = float(indicator.split("<")[1].strip())
                    if market_indicators.get("vix", 0) < threshold:
                        score += 1
                elif indicator.startswith("vix >"):
                    threshold = float(indicator.split(">")[1].strip())
                    if market_indicators.get("vix", 0) > threshold:
                        score += 1
                elif indicator.startswith("low_adr"):
                    if market_indicators.get("adr_percentage", 0) < 1.0:
                        score += 1
                elif indicator.startswith("large_daily_ranges"):
                    if market_indicators.get("adr_percentage", 0) > 1.5:
                        score += 1
                elif indicator.startswith("elevated_put_call_ratio"):
                    if market_indicators.get("put_call_ratio", 0) > 1.2:
                        score += 1
            
            # Calculate percentage of matching indicators
            if indicators:
                condition_scores[condition] = score / len(indicators)
            else:
                condition_scores[condition] = 0
        
        # Choose the condition with the highest score (at least 0.5)
        best_condition = max(condition_scores.items(), key=lambda x: x[1]) if condition_scores else (None, 0)
        
        if best_condition[1] >= 0.5:
            logger.info(f"Detected market condition: {best_condition[0]} (score: {best_condition[1]:.2f})")
            return best_condition[0]
        else:
            logger.info("No clear market condition detected, defaulting to 'sideways'")
            return "sideways"
    
    def get_recommended_strategies(self, market_condition, num_strategies=3):
        """
        Get recommended strategies for the current market condition.
        
        Args:
            market_condition (str): Current market condition
            num_strategies (int): Number of strategies to recommend
        
        Returns:
            list: List of recommended strategy names
        """
        market_conditions = self.strategies_config.get("market_conditions", {})
        condition_data = market_conditions.get(market_condition, {})
        
        # Get strategies to favor and avoid
        strategies_to_favor = condition_data.get("strategies_to_favor", [])
        strategies_to_avoid = condition_data.get("strategies_to_avoid", [])
        
        # If no specific strategies for this condition, use strategy combinations
        if not strategies_to_favor:
            # See if there's a strategy combination that matches the market condition
            for combo_name, combo_data in self.strategies_config.get("strategy_combinations", {}).items():
                if market_condition.lower() in combo_name.lower():
                    return combo_data.get("active_strategies", [])[:num_strategies]
        
        # Return strategies to favor, up to the requested number
        return strategies_to_favor[:num_strategies]
    
    def get_strategy_for_asset_type(self, asset_type, market_condition, direction=None):
        """
        Get the best strategy for a specific asset type and market condition.
        
        Args:
            asset_type (str): Asset type ('equity' or 'options')
            market_condition (str): Current market condition
            direction (str, optional): Trade direction ('bullish', 'bearish', 'neutral', 'bidirectional')
        
        Returns:
            str: Recommended strategy name
        """
        strategies = self.strategies_config.get("strategies", {})
        
        # Filter strategies by asset type
        filtered_strategies = {
            name: data for name, data in strategies.items() 
            if data.get("type") == asset_type
        }
        
        # Further filter by direction if specified
        if direction:
            filtered_strategies = {
                name: data for name, data in filtered_strategies.items() 
                if data.get("direction") == direction or data.get("direction") == "bidirectional"
            }
        
        # Get strategies recommended for this market condition
        recommended = self.get_recommended_strategies(market_condition)
        
        # Prioritize recommended strategies that match our filters
        for strategy_name in recommended:
            if strategy_name in filtered_strategies:
                return strategy_name
        
        # If no match in recommended strategies, pick the one with highest confidence
        confidence_sorted = sorted(
            filtered_strategies.items(), 
            key=lambda x: self._get_confidence_score(x[1], market_condition),
            reverse=True
        )
        
        if confidence_sorted:
            return confidence_sorted[0][0]
        
        # Fall back to default strategy
        return self.config.get("default_strategy", "rsi_ema")
    
    def _get_confidence_score(self, strategy_data, market_condition):
        """Calculate a confidence score for a strategy in the current market condition."""
        # Base confidence from strategy definition
        confidence_map = {"high": 3, "medium": 2, "low": 1}
        base_confidence = confidence_map.get(strategy_data.get("confidence", "medium"), 2)
        
        # Bonus if optimal market type matches current condition
        optimal_market = strategy_data.get("optimal_market_type", "")
        if market_condition.lower() in optimal_market.lower():
            base_confidence += 2
        elif "any" in optimal_market.lower():
            base_confidence += 1
        
        # Bonus for historical performance
        win_rate = strategy_data.get("performance", {}).get("historical_win_rate", 0.5)
        base_confidence += int(win_rate * 2)
        
        return base_confidence
    
    def get_risk_parameters(self, strategy_name, market_condition):
        """
        Get risk parameters for a specific strategy in the current market condition.
        
        Args:
            strategy_name (str): Strategy name
            market_condition (str): Current market condition
        
        Returns:
            dict: Risk parameters for the strategy
        """
        strategy_data = self.get_strategy_details(strategy_name)
        risk_management = self.strategies_config.get("risk_management", {})
        
        # Get base position sizing from strategy
        position_sizing = strategy_data.get("execution", {}).get("position_sizing", "2% account risk")
        
        # Adjust based on market condition
        adjustment = 1.0
        
        # If there's a strategy combination for this market condition, check if it has risk adjustment
        for combo_name, combo_data in self.strategies_config.get("strategy_combinations", {}).items():
            if market_condition.lower() in combo_name.lower():
                if strategy_name in combo_data.get("active_strategies", []):
                    risk_adj = combo_data.get("risk_adjustment", "standard")
                    if risk_adj != "standard":
                        try:
                            # Parse adjustments like "reduced by 25%"
                            if "reduced by" in risk_adj:
                                pct = float(risk_adj.split("reduced by")[1].strip().rstrip("%")) / 100
                                adjustment = 1.0 - pct
                            elif "increased by" in risk_adj:
                                pct = float(risk_adj.split("increased by")[1].strip().rstrip("%")) / 100
                                adjustment = 1.0 + pct
                        except Exception as e:
                            logger.error(f"Error parsing risk adjustment: {str(e)}")
        
        # Calculate position size
        try:
            if "%" in position_sizing:
                pct = float(position_sizing.split("%")[0].strip()) / 100
                adjusted_pct = pct * adjustment
                position_size = f"{adjusted_pct:.2%} account risk"
            else:
                position_size = position_sizing
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            position_size = position_sizing
        
        # Build risk parameters
        risk_params = {
            "position_sizing": position_size,
            "max_open_positions": risk_management.get("max_open_positions", 5),
            "daily_loss_limit": risk_management.get("daily_loss_limit", "3% of account"),
            "stop_loss": strategy_data.get("execution", {}).get("exit", {}).get("stop_loss", ""),
            "take_profit": strategy_data.get("execution", {}).get("exit", {}).get("take_profit", "")
        }
        
        return risk_params
    
    def should_execute_strategy(self, strategy_name, market_condition, market_indicators):
        """
        Determine if a strategy should be executed in the current market condition.
        
        Args:
            strategy_name (str): Strategy name
            market_condition (str): Current market condition
            market_indicators (dict): Dictionary of market indicators
        
        Returns:
            tuple: (should_execute, reason)
        """
        strategy_data = self.get_strategy_details(strategy_name)
        if not strategy_data:
            return False, f"Strategy {strategy_name} not found"
        
        # Check if strategy is in the avoid list for current market condition
        market_conditions = self.strategies_config.get("market_conditions", {})
        condition_data = market_conditions.get(market_condition, {})
        strategies_to_avoid = condition_data.get("strategies_to_avoid", [])
        
        if strategy_name in strategies_to_avoid:
            return False, f"Strategy {strategy_name} is not recommended for {market_condition} market conditions"
        
        # Check if current market type is optimal for this strategy
        optimal_market = strategy_data.get("optimal_market_type", "")
        if optimal_market and market_condition.lower() not in optimal_market.lower() and "any" not in optimal_market.lower():
            return False, f"Strategy {strategy_name} is optimized for {optimal_market} markets, not {market_condition}"
        
        # Check ideal conditions
        ideal_conditions = strategy_data.get("ideal_conditions", [])
        missing_conditions = []
        
        for condition in ideal_conditions:
            # Parse condition string (format: "condition_name: true/false")
            if ":" in condition:
                condition_name, expected_value = condition.split(":", 1)
                condition_name = condition_name.strip()
                expected_value = expected_value.strip().lower() == "true"
                
                # Check if condition is met
                if market_indicators.get(condition_name, not expected_value) != expected_value:
                    missing_conditions.append(condition_name)
        
        # If several ideal conditions are missing, don't execute
        if len(missing_conditions) > len(ideal_conditions) / 2:
            return False, f"Too many ideal conditions missing: {', '.join(missing_conditions)}"
        
        return True, "Strategy is appropriate for current market conditions" 