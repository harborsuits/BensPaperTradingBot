import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def process_alert(alert_data, tradier_api, config):
    """
    Process a TradingView alert for RSI/EMA strategy.
    
    Expected alert_data format:
    {
        "strategy": "rsi_ema",
        "ticker": "AAPL",
        "action": "buy_to_open" or "sell_to_open" or "buy_to_close" or "sell_to_close",
        "option_type": "call" or "put",
        "strike": 150.0,
        "expiration": "YYYY-MM-DD",
        "quantity": 1,
        "price": 2.50 (optional, for limit orders),
        "type": "market" or "limit" (optional),
        "rsi": 30.5,
        "ema": 145.75
    }
    
    Returns:
        dict: Parameters for placing an order, or None if no trade should be executed
    """
    logger.info(f"Processing RSI/EMA alert: {json.dumps(alert_data)}")
    
    # Extract key information
    ticker = alert_data.get("ticker")
    action = alert_data.get("action")
    option_type = alert_data.get("option_type", "call")
    strike = alert_data.get("strike")
    expiration = alert_data.get("expiration")
    quantity = alert_data.get("quantity", config["trading"]["default_quantity"])
    price = alert_data.get("price")
    order_type = alert_data.get("type", config["trading"]["default_order_type"])
    
    # Validate required fields
    if not all([ticker, action, strike, expiration]):
        logger.error("Missing required fields in alert data")
        return None
    
    # Apply risk management rules
    max_quantity = min(quantity, config["trading"]["max_position_size"])
    
    # Check if market is open
    market_status = tradier_api.get_market_status()
    if market_status.get("clock", {}).get("state") != "open":
        logger.warning("Market is not open, skipping trade")
        return None
    
    # Validate expiration date format (YYYY-MM-DD)
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        # Convert to Tradier format if needed
        formatted_exp = exp_date.strftime("%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid expiration date format: {expiration}")
        return None
    
    # Construct OCC option symbol
    # Format: Symbol + YY + MM + DD + C/P + Strike padded to 8 digits
    # Example: AAPL210917C00150000
    strike_padded = str(int(float(strike) * 1000)).zfill(8)
    option_char = "C" if option_type.lower() == "call" else "P"
    option_symbol = f"{ticker.upper()}{exp_date.strftime('%y%m%d')}{option_char}{strike_padded}"
    
    # Get current option price for validation (for limit orders)
    if order_type.lower() == "limit" and price:
        option_chain = tradier_api.get_option_chain(ticker, expiration)
        if option_chain and "options" in option_chain:
            options = option_chain["options"].get("option", [])
            
            # Find our specific option
            for option in options:
                if option.get("symbol") == option_symbol:
                    current_price = option.get("last", 0)
                    
                    # Validate if limit price is reasonable
                    if action in ["buy_to_open", "buy_to_close"] and price < current_price * 0.8:
                        logger.warning(f"Limit price ({price}) is significantly below current price ({current_price})")
                    elif action in ["sell_to_open", "sell_to_close"] and price > current_price * 1.2:
                        logger.warning(f"Limit price ({price}) is significantly above current price ({current_price})")
    
    # Prepare trade parameters
    trade_params = {
        "asset_type": "option",
        "symbol": option_symbol,
        "side": action,
        "quantity": max_quantity,
        "order_type": order_type,
        "duration": "day"
    }
    
    # Add price for limit orders
    if order_type.lower() == "limit" and price:
        trade_params["price"] = price
    
    logger.info(f"Prepared option trade: {json.dumps(trade_params)}")
    return trade_params

def validate_rsi_ema_signal(ticker, rsi, ema, action, tradier_api):
    """
    Validate if the RSI/EMA signal is still valid by checking current market conditions.
    
    Args:
        ticker (str): Stock symbol
        rsi (float): RSI value from the alert
        ema (float): EMA value from the alert
        action (str): Order action
        tradier_api: Tradier API instance
    
    Returns:
        bool: True if the signal is still valid, False otherwise
    """
    # Get current quote
    quote = tradier_api.get_quote(ticker)
    
    if not quote or "quotes" not in quote or "quote" not in quote["quotes"]:
        logger.error(f"Failed to get quote for {ticker}")
        return False
    
    current_price = quote["quotes"]["quote"].get("last", 0)
    
    # Simple validation rules based on action and current price vs EMA
    if action == "buy_to_open" and current_price > ema * 1.02:
        logger.warning(f"Price has moved up too much since signal, current: {current_price}, EMA: {ema}")
        return False
    
    if action == "sell_to_open" and current_price < ema * 0.98:
        logger.warning(f"Price has moved down too much since signal, current: {current_price}, EMA: {ema}")
        return False
    
    return True 