import logging
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OptionSpreadsStrategy:
    """
    Strategy class for executing option spread trades based on option_spreads.json definitions.
    """
    
    def __init__(self):
        """Initialize the option spreads strategy."""
        self.spreads_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "option_spreads.json")
        self.spreads_config = self._load_spreads_config()
        
        logger.info(f"Option Spreads Strategy initialized with {len(self.spreads_config)} spread categories")
    
    def _load_spreads_config(self):
        """Load the option spreads configuration from JSON file."""
        try:
            if os.path.exists(self.spreads_file):
                with open(self.spreads_file, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Option spreads file not found: {self.spreads_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading option spreads config: {str(e)}")
            return {}
    
    def get_available_spreads(self, category=None):
        """
        Get a list of all available spread strategies.
        
        Args:
            category (str, optional): Filter by category (vertical_spreads, neutral_spreads, etc.)
            
        Returns:
            dict: Dictionary of available spread strategies by category
        """
        if not self.spreads_config:
            return {}
            
        if category and category in self.spreads_config:
            return {category: list(self.spreads_config[category].keys())}
        
        return {cat: list(strategies.keys()) for cat, strategies in self.spreads_config.items()}
    
    def get_spread_details(self, category, spread_name):
        """
        Get detailed configuration for a specific spread strategy.
        
        Args:
            category (str): Spread category (vertical_spreads, neutral_spreads, etc.)
            spread_name (str): Name of the spread strategy
            
        Returns:
            dict: Spread strategy configuration or empty dict if not found
        """
        return self.spreads_config.get(category, {}).get(spread_name, {})
    
    def check_ideal_conditions(self, category, spread_name, market_indicators):
        """
        Check if current market conditions are ideal for a particular spread strategy.
        
        Args:
            category (str): Spread category
            spread_name (str): Name of the spread strategy
            market_indicators (dict): Dictionary of market indicators
            
        Returns:
            tuple: (is_ideal, confidence_score, missing_conditions)
        """
        spread_config = self.get_spread_details(category, spread_name)
        if not spread_config:
            return False, 0, ["Strategy not found"]
        
        # Get ideal conditions
        ideal_conditions = spread_config.get("ideal_conditions", {})
        primary_conditions = ideal_conditions.get("primary", [])
        secondary_conditions = ideal_conditions.get("secondary", [])
        
        # Check conditions to avoid
        avoid_conditions = spread_config.get("avoid_when", [])
        
        # Check for red flags (conditions to avoid)
        for condition in avoid_conditions:
            condition_name = condition.split(" ")[0].lower()
            if market_indicators.get(condition_name, False):
                return False, 0, [f"Avoid condition present: {condition}"]
        
        # Check primary conditions
        primary_met = 0
        primary_total_weight = sum(cond.get("weight", 1) for cond in primary_conditions)
        missing_primary = []
        
        for condition in primary_conditions:
            condition_name = condition.get("condition", "").split(" ")[0].lower()
            weight = condition.get("weight", 1)
            
            if market_indicators.get(condition_name, False):
                primary_met += weight
            else:
                missing_primary.append(condition.get("condition", "unknown"))
        
        # Check secondary conditions
        secondary_met = 0
        secondary_total_weight = sum(cond.get("weight", 1) for cond in secondary_conditions)
        missing_secondary = []
        
        for condition in secondary_conditions:
            condition_name = condition.get("condition", "").split(" ")[0].lower()
            weight = condition.get("weight", 1)
            
            if market_indicators.get(condition_name, False):
                secondary_met += weight
            else:
                missing_secondary.append(condition.get("condition", "unknown"))
        
        # Calculate confidence score
        if primary_total_weight > 0:
            primary_score = primary_met / primary_total_weight
        else:
            primary_score = 1.0
            
        if secondary_total_weight > 0:
            secondary_score = secondary_met / secondary_total_weight
        else:
            secondary_score = 1.0
        
        # Weight primary conditions more heavily (70/30 split)
        confidence_score = (primary_score * 0.7) + (secondary_score * 0.3)
        
        # Strategy is ideal if confidence is above threshold
        is_ideal = confidence_score >= 0.6
        
        # Combine missing conditions
        missing_conditions = missing_primary + missing_secondary
        
        return is_ideal, confidence_score, missing_conditions
    
    def recommend_spread(self, market_indicators, underlying=None, directional_bias=None, vol_expectation=None):
        """
        Recommend the best option spread strategy based on current market conditions.
        
        Args:
            market_indicators (dict): Dictionary of market indicators
            underlying (str, optional): Symbol of the underlying asset
            directional_bias (str, optional): "bullish", "bearish", or "neutral"
            vol_expectation (str, optional): "increasing", "decreasing", or "stable"
            
        Returns:
            dict: Recommended spread strategy details
        """
        best_spread = None
        best_score = 0
        best_category = None
        
        # Filter by directional bias if provided
        categories_to_check = list(self.spreads_config.keys())
        
        if directional_bias:
            if directional_bias.lower() == "bullish":
                # Prioritize bullish strategies
                bullish_categories = ["vertical_spreads"]
                bullish_spreads = ["bull_call_spread", "bull_put_spread"]
                
                filtered_categories = {}
                for cat in bullish_categories:
                    if cat in self.spreads_config:
                        filtered_spreads = {}
                        for spread_name in bullish_spreads:
                            if spread_name in self.spreads_config.get(cat, {}):
                                filtered_spreads[spread_name] = self.spreads_config[cat][spread_name]
                        if filtered_spreads:
                            filtered_categories[cat] = filtered_spreads
                
                # Only check these if we have matches
                if filtered_categories:
                    categories_to_check = list(filtered_categories.keys())
            
            elif directional_bias.lower() == "bearish":
                # Prioritize bearish strategies
                bearish_categories = ["vertical_spreads"]
                bearish_spreads = ["bear_call_spread", "bear_put_spread"]
                
                filtered_categories = {}
                for cat in bearish_categories:
                    if cat in self.spreads_config:
                        filtered_spreads = {}
                        for spread_name in bearish_spreads:
                            if spread_name in self.spreads_config.get(cat, {}):
                                filtered_spreads[spread_name] = self.spreads_config[cat][spread_name]
                        if filtered_spreads:
                            filtered_categories[cat] = filtered_spreads
                
                # Only check these if we have matches
                if filtered_categories:
                    categories_to_check = list(filtered_categories.keys())
            
            elif directional_bias.lower() == "neutral":
                # Prioritize neutral strategies
                neutral_categories = ["neutral_spreads"]
                categories_to_check = [cat for cat in neutral_categories if cat in self.spreads_config]
        
        # Filter by volatility expectation if provided
        if vol_expectation:
            # Here you could further filter based on volatility expectation
            # For example, if increasing volatility, prioritize long straddles/strangles
            # If decreasing, prioritize iron condors, short straddles/strangles, etc.
            pass
        
        # Check each category and spread
        for category in categories_to_check:
            for spread_name in self.spreads_config.get(category, {}):
                is_ideal, confidence, missing = self.check_ideal_conditions(
                    category, spread_name, market_indicators
                )
                
                if confidence > best_score:
                    best_score = confidence
                    best_spread = spread_name
                    best_category = category
        
        if best_spread and best_category:
            result = {
                "category": best_category,
                "strategy": best_spread,
                "confidence": best_score,
                "details": self.get_spread_details(best_category, best_spread)
            }
            return result
        
        return None

def process_alert(alert_data, tradier_api, config):
    """
    Process a TradingView alert for option spread strategies.
    
    Expected alert_data format:
    {
        "strategy": "option_spreads_strategy",
        "ticker": "AAPL",
        "action": "open" or "close",
        "spread_type": "bull_call_spread", "iron_condor", etc.,
        "quantity": 1,
        "expiration": "YYYY-MM-DD",
        "strikes": [150, 160],  # For vertical spread
        "price": 2.50 (optional, for limit orders),
        "type": "market" or "limit" (optional)
    }
    
    or for more complex spreads like iron condor:
    {
        "spread_type": "iron_condor",
        "strikes": {
            "put_buy": 145,
            "put_sell": 150,
            "call_sell": 160,
            "call_buy": 165
        }
    }
    
    Returns:
        dict: Parameters for placing an order, or None if no trade should be executed
    """
    logger.info(f"Processing option spread alert: {json.dumps(alert_data)}")
    
    # Load option spreads data
    strategy = OptionSpreadsStrategy()
    
    # Extract key information
    ticker = alert_data.get("ticker")
    action = alert_data.get("action", "open")  # "open" or "close"
    spread_type = alert_data.get("spread_type")
    quantity = alert_data.get("quantity", config["trading"]["default_quantity"])
    expiration = alert_data.get("expiration")
    strikes = alert_data.get("strikes", [])
    price = alert_data.get("price")
    order_type = alert_data.get("type", config["trading"]["default_order_type"])
    
    # Extract category from spread_type
    if "put" in spread_type or "call" in spread_type:
        category = "vertical_spreads"
    elif spread_type in ["iron_condor", "calendar_spread", "double_diagonal"]:
        category = "neutral_spreads"
    elif spread_type in ["long_straddle", "long_strangle"]:
        category = "explosive_move_spreads"
    else:
        category = "advanced_strategies"
    
    # Validate required fields
    if not all([ticker, spread_type, expiration]):
        logger.error("Missing required fields in alert data")
        return None
    
    # Validate strikes based on spread type
    if isinstance(strikes, list) and len(strikes) < 2:
        logger.error(f"Not enough strikes provided for {spread_type}")
        return None
    
    # Apply risk management rules
    max_quantity = min(quantity, config["trading"]["max_position_size"])
    
    # Check if market is open
    market_status = tradier_api.get_market_status()
    if market_status.get("clock", {}).get("state") != "open":
        logger.warning("Market is not open, skipping trade")
        return None
    
    # Get spread details
    spread_details = strategy.get_spread_details(category, spread_type)
    if not spread_details:
        logger.error(f"Spread type {spread_type} not found in configuration")
        return None
    
    # Validate expiration date format (YYYY-MM-DD)
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        # Convert to Tradier format if needed
        formatted_exp = exp_date.strftime("%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid expiration date format: {expiration}")
        return None
    
    # Format is different based on spread type
    if spread_type in ["bull_call_spread", "bear_put_spread"]:
        # Debit spread
        # For bull call: Buy lower strike call, sell higher strike call
        # For bear put: Buy higher strike put, sell lower strike put
        if isinstance(strikes, list) and len(strikes) >= 2:
            strikes.sort()
            legs = []
            
            if spread_type == "bull_call_spread":
                # Buy lower strike call
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "call", strikes[0]),
                    "side": "buy_to_open" if action == "open" else "sell_to_close",
                    "quantity": max_quantity
                })
                
                # Sell higher strike call
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "call", strikes[1]),
                    "side": "sell_to_open" if action == "open" else "buy_to_close",
                    "quantity": max_quantity
                })
            
            elif spread_type == "bear_put_spread":
                # Buy higher strike put
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "put", strikes[1]),
                    "side": "buy_to_open" if action == "open" else "sell_to_close",
                    "quantity": max_quantity
                })
                
                # Sell lower strike put
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "put", strikes[0]),
                    "side": "sell_to_open" if action == "open" else "buy_to_close",
                    "quantity": max_quantity
                })
    
    elif spread_type in ["bull_put_spread", "bear_call_spread"]:
        # Credit spread
        # For bull put: Sell higher strike put, buy lower strike put
        # For bear call: Sell lower strike call, buy higher strike call
        if isinstance(strikes, list) and len(strikes) >= 2:
            strikes.sort()
            legs = []
            
            if spread_type == "bull_put_spread":
                # Sell higher strike put
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "put", strikes[1]),
                    "side": "sell_to_open" if action == "open" else "buy_to_close",
                    "quantity": max_quantity
                })
                
                # Buy lower strike put
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "put", strikes[0]),
                    "side": "buy_to_open" if action == "open" else "sell_to_close",
                    "quantity": max_quantity
                })
            
            elif spread_type == "bear_call_spread":
                # Sell lower strike call
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "call", strikes[0]),
                    "side": "sell_to_open" if action == "open" else "buy_to_close",
                    "quantity": max_quantity
                })
                
                # Buy higher strike call
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "call", strikes[1]),
                    "side": "buy_to_open" if action == "open" else "sell_to_close",
                    "quantity": max_quantity
                })
    
    elif spread_type == "iron_condor":
        # Iron Condor (combination of bull put and bear call)
        # Either using a list of 4 strikes or a dictionary with named strikes
        legs = []
        
        if isinstance(strikes, dict):
            # Dictionary format with named strikes
            put_buy = strikes.get("put_buy")
            put_sell = strikes.get("put_sell")
            call_sell = strikes.get("call_sell")
            call_buy = strikes.get("call_buy")
            
            if all([put_buy, put_sell, call_sell, call_buy]):
                # Bull put side: Sell put_sell, buy put_buy
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "put", put_sell),
                    "side": "sell_to_open" if action == "open" else "buy_to_close",
                    "quantity": max_quantity
                })
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "put", put_buy),
                    "side": "buy_to_open" if action == "open" else "sell_to_close",
                    "quantity": max_quantity
                })
                
                # Bear call side: Sell call_sell, buy call_buy
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "call", call_sell),
                    "side": "sell_to_open" if action == "open" else "buy_to_close",
                    "quantity": max_quantity
                })
                legs.append({
                    "option_symbol": format_option_symbol(ticker, formatted_exp, "call", call_buy),
                    "side": "buy_to_open" if action == "open" else "sell_to_close",
                    "quantity": max_quantity
                })
            else:
                logger.error("Missing required strikes for iron condor")
                return None
        
        elif isinstance(strikes, list) and len(strikes) >= 4:
            # List format with ascending strikes
            strikes.sort()
            
            # Bull put side: Sell strikes[1], buy strikes[0]
            legs.append({
                "option_symbol": format_option_symbol(ticker, formatted_exp, "put", strikes[1]),
                "side": "sell_to_open" if action == "open" else "buy_to_close",
                "quantity": max_quantity
            })
            legs.append({
                "option_symbol": format_option_symbol(ticker, formatted_exp, "put", strikes[0]),
                "side": "buy_to_open" if action == "open" else "sell_to_close",
                "quantity": max_quantity
            })
            
            # Bear call side: Sell strikes[2], buy strikes[3]
            legs.append({
                "option_symbol": format_option_symbol(ticker, formatted_exp, "call", strikes[2]),
                "side": "sell_to_open" if action == "open" else "buy_to_close",
                "quantity": max_quantity
            })
            legs.append({
                "option_symbol": format_option_symbol(ticker, formatted_exp, "call", strikes[3]),
                "side": "buy_to_open" if action == "open" else "sell_to_close",
                "quantity": max_quantity
            })
        else:
            logger.error("Invalid strikes format for iron condor")
            return None
    
    # Prepare trade parameters for complex order
    trade_params = {
        "asset_type": "option_spread",
        "class": spread_type,
        "symbol": ticker,
        "legs": legs,
        "order_type": order_type,
        "duration": "day"
    }
    
    # Add price for limit orders
    if order_type.lower() == "limit" and price:
        trade_params["price"] = price
    
    logger.info(f"Prepared option spread trade: {json.dumps(trade_params)}")
    return trade_params

def format_option_symbol(ticker, expiration, option_type, strike):
    """Format an option symbol in OCC format."""
    # Parse expiration date
    exp_date = datetime.strptime(expiration, "%Y-%m-%d")
    
    # Construct OCC option symbol
    # Format: Symbol + YY + MM + DD + C/P + Strike padded to 8 digits
    # Example: AAPL210917C00150000
    strike_padded = str(int(float(strike) * 1000)).zfill(8)
    option_char = "C" if option_type.lower() == "call" else "P"
    
    return f"{ticker.upper()}{exp_date.strftime('%y%m%d')}{option_char}{strike_padded}" 