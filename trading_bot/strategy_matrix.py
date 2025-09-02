"""
Strategy Matrix

Implements a comprehensive strategy selection system based on market conditions.
Provides detailed strategy recommendations with implementation guidance.
"""

import json
import os
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import math

# Market Regime Classifications
class MarketRegime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    EARNINGS_DRIVEN = "earnings_driven"
    MACRO_DRIVEN = "macro_driven"
    SECTOR_ROTATION = "sector_rotation"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

class ConditionType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    CALENDAR = "calendar"
    RELATIVE_STRENGTH = "relative_strength"
    COMPLEX = "complex"

class StrategyMatrix:
    """
    Strategy Matrix class that implements the strategy selection logic
    based on current market conditions.
    """
    
    def __init__(self, matrix_file: str = "strategy_selection_matrix.json"):
        """
        Initialize the strategy matrix
        
        Args:
            matrix_file: Path to the strategy matrix JSON file
        """
        self.logger = logging.getLogger("StrategyMatrix")
        self.matrix_file = matrix_file
        self.matrix_data = None
        self.last_loaded = None
        
        # Load the strategy matrix data
        self.load_matrix()
    
    def load_matrix(self) -> bool:
        """
        Load the strategy matrix from file
        
        Returns:
            Whether the matrix was successfully loaded
        """
        try:
            if os.path.exists(self.matrix_file):
                with open(self.matrix_file, 'r') as f:
                    self.matrix_data = json.load(f)
                
                self.last_loaded = datetime.datetime.now()
                self.logger.info(f"Strategy matrix loaded successfully from {self.matrix_file}")
                return True
            else:
                self.logger.warning(f"Strategy matrix file not found: {self.matrix_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading strategy matrix: {e}")
            return False
    
    def save_matrix(self, matrix_data: Dict[str, Any]) -> bool:
        """
        Save the strategy matrix to file
        
        Args:
            matrix_data: Strategy matrix data to save
            
        Returns:
            Whether the matrix was successfully saved
        """
        try:
            # Update the version and timestamp
            matrix_data["version"] = matrix_data.get("version", "1.0.0")
            matrix_data["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d")
            
            with open(self.matrix_file, 'w') as f:
                json.dump(matrix_data, f, indent=2)
            
            self.matrix_data = matrix_data
            self.last_loaded = datetime.datetime.now()
            
            self.logger.info(f"Strategy matrix saved successfully to {self.matrix_file}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error saving strategy matrix: {e}")
            return False
    
    def evaluate_market_condition(self, condition: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """
        Evaluate if a specific market condition is true
        
        Args:
            condition: Condition object from strategy matrix
            market_data: Current market data
            
        Returns:
            Whether the condition is met
        """
        indicator = condition.get("indicator")
        operator = condition.get("operator")
        threshold = condition.get("threshold")
        
        # Handle missing data
        if not market_data or indicator not in market_data:
            self.logger.warning(f"Missing market data for {indicator}")
            return False
        
        value = market_data[indicator]
        
        # Handle special case for "forecast" comparison
        if threshold == "forecast" and f"{indicator}_forecast" in market_data:
            forecast_value = market_data[f"{indicator}_forecast"]
            
            if operator == ">":
                return value > forecast_value
            elif operator == "<":
                return value < forecast_value
            elif operator == "=":
                return value == forecast_value
            else:
                return False
        
        # Handle all-time high comparison
        if threshold == "all-time high" and f"{indicator}_ath" in market_data:
            ath_value = market_data[f"{indicator}_ath"]
            
            if operator == ">":
                return value > ath_value
            elif operator == "<":
                return value < ath_value
            elif operator == "=":
                return value == ath_value
            else:
                return False
        
        # Handle string thresholds
        if isinstance(threshold, str) and threshold not in ["forecast", "all-time high"]:
            # Try to handle special string cases
            if threshold == "active" and indicator == "Earnings Calendar":
                return market_data.get("earnings_season", False)
            elif threshold == "FOMC meeting" and indicator == "Economic Calendar":
                return market_data.get("fomc_meeting_week", False)
            else:
                # Can't compare with string threshold
                self.logger.warning(f"Unsupported string threshold: {threshold}")
                return False
        
        # Handle different operators for numeric comparisons
        if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
            if operator == ">":
                return value > threshold
            elif operator == "<":
                return value < threshold
            elif operator == "=":
                return value == threshold
            elif operator == ">=":
                return value >= threshold
            elif operator == "<=":
                return value <= threshold
            else:
                return False
        
        # Handle complex conditions
        if operator == "complex":
            return self.evaluate_complex_condition(condition, market_data)
        
        # Default case
        return False
    
    def evaluate_complex_condition(self, condition: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """
        Evaluate complex conditions that require multiple checks
        
        Args:
            condition: Condition object from strategy matrix
            market_data: Current market data
            
        Returns:
            Whether the complex condition is met
        """
        condition_id = condition.get("id")
        
        # Handle specific complex conditions based on condition ID
        if condition_id == "goldilocks_economy":
            return (market_data.get("jobs_report_beat", False) and 
                    market_data.get("wage_growth", 0) < market_data.get("wage_growth_expected", 0))
        
        elif condition_id == "tech_leadership":
            return (market_data.get("xlk_performance") is not None and 
                   market_data.get("spy_performance") is not None and
                   market_data.get("xlk_performance") > market_data.get("spy_performance"))
        
        elif condition_id == "defensive_rotation":
            return (market_data.get("xlp_performance") is not None and 
                   market_data.get("spy_performance") is not None and
                   market_data.get("xlp_performance") > market_data.get("spy_performance"))
        
        elif condition_id == "fomc_week":
            return market_data.get("fomc_meeting_week", False)
        
        elif condition_id == "new_ath_breakout":
            return (market_data.get("spy_price") is not None and 
                   market_data.get("spy_ath") is not None and
                   market_data.get("spy_price") > market_data.get("spy_ath"))
        
        else:
            self.logger.warning(f"Unknown complex condition: {condition_id}")
            return False
    
    def get_recommended_strategies(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommended strategies based on current market conditions
        
        Args:
            market_data: Current market data
            
        Returns:
            Recommended strategies and context
        """
        if not self.matrix_data:
            self.logger.error("Strategy matrix not loaded")
            return {
                "strategies": [],
                "applicable_conditions": [],
                "market_regime": "undetermined"
            }
        
        applicable_conditions = []
        
        # Find all conditions that are true
        for condition in self.matrix_data.get("strategy_matrix", []):
            if self.evaluate_market_condition(condition, market_data):
                applicable_conditions.append(condition)
        
        # No applicable conditions found
        if not applicable_conditions:
            return {
                "strategies": [],
                "applicable_conditions": [],
                "market_regime": "undetermined"
            }
        
        # Gather all recommended strategies from applicable conditions
        strategy_scores = {}
        strategy_details = {}
        regime_counts = {}
        
        # Process each applicable condition
        for condition in applicable_conditions:
            # Track market regimes
            regime = condition.get("market_regime")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Process preferred strategies
            for strategy in condition.get("preferred_strategies", []):
                strategy_name = strategy.get("name")
                
                # Initialize or update strategy score
                if strategy_name not in strategy_scores:
                    strategy_scores[strategy_name] = 0
                    strategy_details[strategy_name] = strategy
                
                # Add to score based on weight and condition
                strategy_scores[strategy_name] += strategy.get("weight", 0.5) * 10
            
            # Process strategies to avoid (negative scoring)
            for avoid_strategy in condition.get("avoid_strategies", []):
                strategy_name = avoid_strategy.get("name")
                if strategy_name in strategy_scores:
                    strategy_scores[strategy_name] -= 15  # Strong penalty for explicitly avoided
        
        # Determine dominant regime
        dominant_regime = None
        max_regime_count = 0
        for regime, count in regime_counts.items():
            if count > max_regime_count:
                max_regime_count = count
                dominant_regime = regime
        
        # Sort strategies by score
        ranked_strategies = []
        for name, score in strategy_scores.items():
            if score > 0:  # Only include positive scores
                ranked_strategies.append({
                    "name": name,
                    "score": score,
                    "details": strategy_details.get(name, {})
                })
        
        # Sort by score, highest first
        ranked_strategies.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "strategies": ranked_strategies,
            "applicable_conditions": [c.get("id") for c in applicable_conditions],
            "market_regime": dominant_regime
        }
    
    def generate_trade_signal(self, strategy: Dict[str, Any], market_data: Dict[str, Any], 
                             account: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trade signal based on selected strategy and market data
        
        Args:
            strategy: Selected strategy details
            market_data: Current market data
            account: Account information including risk settings
            
        Returns:
            Trade signal with parameters
        """
        strategy_details = strategy.get("details", {})
        implementation = strategy_details.get("implementation", {})
        trade_type = strategy_details.get("trade_type", "equity")
        
        # Base signal structure
        signal = {
            "strategy": strategy.get("name"),
            "trade_type": trade_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "market_condition": {
                "vix": market_data.get("vix"),
                "spy_price": market_data.get("spy_price"),
                "above_20dma": market_data.get("spy_price", 0) > market_data.get("spy_20dma", 0)
            }
        }
        
        # Apply risk adjustments if specified in strategy
        risk_adjustment = 1.0
        applicable_condition_ids = market_data.get("applicable_conditions", [])
        
        if applicable_condition_ids and self.matrix_data:
            for condition in self.matrix_data.get("strategy_matrix", []):
                if condition.get("id") in applicable_condition_ids:
                    # Parse risk adjustment string
                    risk_text = condition.get("risk_adjustment", "")
                    
                    if "Reduce" in risk_text:
                        import re
                        match = re.search(r"Reduce position size by (\d+)%", risk_text)
                        if match:
                            percentage = int(match.group(1))
                            risk_adjustment = 1 - (percentage / 100)
                    
                    elif "Increase" in risk_text:
                        import re
                        match = re.search(r"Increase position size by (\d+)%", risk_text)
                        if match:
                            percentage = int(match.group(1))
                            risk_adjustment = 1 + (percentage / 100)
                    
                    break  # Just use the first matching condition
        
        # Calculate position size based on account risk settings and strategy risk parameters
        account_value = account.get("balance", 10000)
        risk_per_trade = 0.01  # Default 1% risk
        
        # Extract risk from position sizing pattern
        if implementation and "position_sizing" in implementation:
            import re
            risk_match = re.search(r"Risk (\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?%", implementation["position_sizing"])
            if risk_match:
                if risk_match.group(2):  # Range provided (e.g., "1-2%")
                    min_risk = float(risk_match.group(1))
                    max_risk = float(risk_match.group(2))
                    risk_per_trade = (min_risk + max_risk) / 200  # Average and convert to decimal
                else:  # Single value
                    risk_per_trade = float(risk_match.group(1)) / 100
        
        adjusted_risk = risk_per_trade * risk_adjustment
        risk_amount = account_value * adjusted_risk
        
        # Add trade type specific parameters
        if trade_type == "equity":
            signal["direction"] = self.determine_direction(strategy, market_data)
            signal["risk_amount"] = risk_amount
            
            # For equity trades, estimate position size based on stop distance
            price = self.get_symbol_price(strategy, market_data)
            stop_distance = self.estimate_stop_distance(strategy, price, market_data)
            
            if price and stop_distance:
                signal["price"] = price
                signal["stop_loss"] = (price - stop_distance) if signal["direction"] == "buy" else (price + stop_distance)
                signal["quantity"] = math.floor(risk_amount / stop_distance)
            
        elif trade_type == "options":
            signal["direction"] = self.determine_direction(strategy, market_data)
            signal["risk_amount"] = risk_amount
            
            # Add options specific parameters
            if "option_structure" in implementation:
                signal["option_structure"] = implementation["option_structure"]
            
            if "expiration" in implementation:
                signal["expiration"] = self.parse_option_expiration(implementation["expiration"])
            
            if "strike_selection" in implementation:
                signal["strike_selection"] = implementation["strike_selection"]
        
        return signal
    
    def determine_direction(self, strategy: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """
        Determine trade direction based on strategy and market conditions
        
        Args:
            strategy: Selected strategy
            market_data: Current market data
            
        Returns:
            "buy" or "sell"
        """
        strategy_details = strategy.get("details", {})
        implementation = strategy_details.get("implementation", {})
        
        # If strategy has explicit direction, use it
        if "direction" in implementation:
            return implementation["direction"]
        
        # Otherwise infer from strategy type and market conditions
        strategy_name = strategy.get("name")
        is_above_20dma = market_data.get("spy_price", 0) > market_data.get("spy_20dma", 0)
        is_above_50dma = market_data.get("spy_price", 0) > market_data.get("spy_50dma", 0)
        vix_high = market_data.get("vix", 20) > 25
        
        # Directional bias based on strategy type
        if strategy_name in ["breakout_swing", "bull_call_spread", "bull_put_spread"]:
            return "buy"
        
        elif strategy_name == "pullback_to_moving_average":
            return "buy" if is_above_50dma else "sell"
        
        elif strategy_name in ["bear_call_spread", "short_equity"]:
            return "sell"
        
        elif strategy_name in ["iron_condor", "calendar_spread"]:
            # Non-directional strategies, default to market trend
            return "buy" if is_above_20dma else "sell"
        
        elif strategy_name == "volatility_squeeze":
            # Use recent price action to determine direction
            return "buy" if market_data.get("recent_momentum", 0) > 0 else "sell"
        
        else:
            # Default to market trend
            return "buy" if is_above_20dma and not vix_high else "sell"
    
    def estimate_stop_distance(self, strategy: Dict[str, Any], price: float, market_data: Dict[str, Any]) -> float:
        """
        Estimate appropriate stop distance based on strategy and volatility
        
        Args:
            strategy: Selected strategy
            price: Current price
            market_data: Current market data
            
        Returns:
            Estimated stop distance in price units
        """
        strategy_name = strategy.get("name")
        strategy_details = strategy.get("details", {})
        implementation = strategy_details.get("implementation", {})
        
        # If ATR is available, use it for volatility-adjusted stops
        atr_value = market_data.get("atr", price * 0.01)  # Default to 1% if ATR not available
        
        # Extract stop placement method from implementation if available
        if "stop_placement" in implementation:
            stop_info = implementation["stop_placement"]
            
            if "swing" in stop_info:
                # For swing-based stops, use recent swing high/low
                swing_size = market_data.get("recent_swing_size", price * 0.03)
                return swing_size * 1.1  # Add 10% buffer
            
            if "ATR" in stop_info:
                import re
                match = re.search(r"(\d+(?:\.\d+)?)\s*ATR", stop_info)
                if match:
                    atr_multiplier = float(match.group(1))
                    return atr_value * atr_multiplier
        
        # Strategy-specific defaults if not specified
        if strategy_name == "breakout_swing":
            return atr_value * 2
        
        elif strategy_name == "pullback_to_moving_average":
            return atr_value * 1.5
        
        elif strategy_name == "short_equity":
            return atr_value * 2.5
        
        else:
            # Default stop distance based on price and volatility
            volatility_factor = 1.5 if market_data.get("vix", 20) > 25 else 1.0
            return price * 0.02 * volatility_factor  # 2% of price, adjusted for volatility
    
    def get_symbol_price(self, strategy: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Get price for a symbol based on strategy and market data
        
        Args:
            strategy: Selected strategy
            market_data: Current market data
            
        Returns:
            Current price for symbol
        """
        # In a real implementation, this would select the specific symbol from the strategy
        # For now, just use SPY as a placeholder
        return market_data.get("spy_price", 400)
    
    def parse_option_expiration(self, exp_description: str) -> str:
        """
        Parse option expiration description into actual date
        
        Args:
            exp_description: Expiration description, e.g., "30-45 DTE"
            
        Returns:
            ISO date string for expiration
        """
        today = datetime.datetime.now().date()
        
        # Handle specific descriptions
        if "DTE" in exp_description:
            import re
            match = re.search(r"(\d+)(?:-(\d+))?\s*DTE", exp_description)
            if match:
                if match.group(2):  # Range provided (e.g., "30-45 DTE")
                    min_days = int(match.group(1))
                    max_days = int(match.group(2))
                    days = (min_days + max_days) // 2  # Average
                else:  # Single value
                    days = int(match.group(1))
                
                exp_date = today + datetime.timedelta(days=days)
                return exp_date.isoformat()
        
        if exp_description == "nearest_monthly":
            # Find next monthly expiration (typically 3rd Friday)
            year = today.year
            month = today.month
            
            # Calculate third Friday
            first_day = datetime.date(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7 + 1  # Friday is 4 in Python's weekday()
            third_friday = datetime.date(year, month, days_until_friday + 14)
            
            # If we're past third Friday, move to next month
            if today > third_friday:
                if month == 12:
                    month = 1
                    year += 1
                else:
                    month += 1
                
                first_day = datetime.date(year, month, 1)
                days_until_friday = (4 - first_day.weekday()) % 7 + 1
                third_friday = datetime.date(year, month, days_until_friday + 14)
            
            return third_friday.isoformat()
        
        # Default to 45 days out if nothing else matches
        default_exp = today + datetime.timedelta(days=45)
        return default_exp.isoformat()


# Usage example
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create an instance
    matrix = StrategyMatrix()
    
    # Create a sample market data
    market_data = {
        "vix": 28.5,
        "spy_price": 420.35,
        "spy_20dma": 415.20,
        "spy_50dma": 410.75,
        "spy_200dma": 395.80,
        "spy_ath": 415.0,
        "atr": 4.25,
        "recent_momentum": 1.2,
        "earnings_season": True
    }
    
    # Get recommended strategies
    recommendations = matrix.get_recommended_strategies(market_data)
    
    print(f"Market Regime: {recommendations['market_regime']}")
    print(f"Applicable Conditions: {recommendations['applicable_conditions']}")
    print(f"Top Strategies:")
    
    for strategy in recommendations["strategies"][:3]:  # Top 3 strategies
        print(f"  - {strategy['name']} (Score: {strategy['score']:.1f})")
        print(f"    Type: {strategy['details'].get('trade_type')}")
        print(f"    Time Frame: {strategy['details'].get('time_frame')}")
        print()
    
    # Generate a trade signal for the top strategy
    if recommendations["strategies"]:
        account = {"balance": 100000, "risk_mode": "balanced"}
        signal = matrix.generate_trade_signal(recommendations["strategies"][0], market_data, account)
        
        print("Trade Signal:")
        print(f"  Strategy: {signal['strategy']}")
        print(f"  Direction: {signal['direction']}")
        print(f"  Type: {signal['trade_type']}")
        if signal['trade_type'] == 'equity':
            print(f"  Price: ${signal.get('price', 0):.2f}")
            print(f"  Stop Loss: ${signal.get('stop_loss', 0):.2f}")
            print(f"  Quantity: {signal.get('quantity', 0)}")
        elif signal['trade_type'] == 'options':
            print(f"  Structure: {signal.get('option_structure', '')}")
            print(f"  Expiration: {signal.get('expiration', '')}")
            print(f"  Strike Selection: {signal.get('strike_selection', '')}")
        print(f"  Risk Amount: ${signal.get('risk_amount', 0):.2f}") 