import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def process_alert(alert_data, tradier_api, config):
    """
    Process a TradingView alert for the breakout swing strategy.
    
    Expected alert_data format:
    {
        "strategy": "breakout_swing",
        "ticker": "AAPL",
        "action": "buy" or "sell",
        "quantity": 10,
        "price": 150.0 (optional, for limit orders),
        "type": "market" or "limit" (optional),
        "stop_loss": 145.0 (optional),
        "take_profit": 160.0 (optional),
        "breakout_level": 148.5,
        "volume_ratio": 1.5
    }
    
    Returns:
        dict: Parameters for placing an order, or None if no trade should be executed
    """
    logger.info(f"Processing breakout swing alert: {json.dumps(alert_data)}")
    
    # Extract key information
    ticker = alert_data.get("ticker")
    action = alert_data.get("action")
    quantity = alert_data.get("quantity", config["trading"]["default_quantity"])
    price = alert_data.get("price")
    order_type = alert_data.get("type", config["trading"]["default_order_type"])
    stop_loss = alert_data.get("stop_loss")
    breakout_level = alert_data.get("breakout_level")
    volume_ratio = alert_data.get("volume_ratio", 1.0)
    
    # Validate required fields
    if not all([ticker, action]):
        logger.error("Missing required fields in alert data")
        return None
    
    # Apply risk management rules
    max_quantity = min(quantity, config["trading"]["max_position_size"])
    
    # Check if market is open
    market_status = tradier_api.get_market_status()
    if market_status.get("clock", {}).get("state") != "open":
        logger.warning("Market is not open, skipping trade")
        return None
    
    # Validate breakout with current price and volume
    quote = tradier_api.get_quote(ticker)
    if not quote or "quotes" not in quote or "quote" not in quote["quotes"]:
        logger.error(f"Failed to get quote for {ticker}")
        return None
    
    current_quote = quote["quotes"]["quote"]
    current_price = current_quote.get("last", 0)
    current_volume = current_quote.get("volume", 0)
    avg_volume = current_quote.get("average_volume", 0)
    
    # Check if breakout is still valid
    if action == "buy" and breakout_level:
        if current_price < float(breakout_level):
            logger.warning(f"Price dropped below breakout level, current: {current_price}, breakout: {breakout_level}")
            return None
    
    if action == "sell" and breakout_level:
        if current_price > float(breakout_level):
            logger.warning(f"Price moved above breakout level, current: {current_price}, breakout: {breakout_level}")
            return None
    
    # Check volume confirmation
    if volume_ratio and avg_volume > 0:
        if current_volume < avg_volume * float(volume_ratio):
            logger.warning(f"Volume too low for breakout confirmation, current: {current_volume}, avg: {avg_volume}, ratio: {volume_ratio}")
            return None
    
    # Calculate position size based on risk management
    if stop_loss and current_price:
        # Calculate position size based on risk per trade
        risk_pct = config["risk"]["max_loss_per_trade_percent"] / 100
        account_info = tradier_api.get_account_balance()
        
        if account_info and "balances" in account_info:
            account_value = account_info["balances"].get("total_equity", 0)
            risk_amount = account_value * risk_pct
            
            # Calculate stop loss distance
            if action == "buy":
                stop_distance = abs(current_price - float(stop_loss))
            else:
                stop_distance = abs(float(stop_loss) - current_price)
            
            if stop_distance > 0:
                # Calculate position size based on risk
                risk_based_quantity = int(risk_amount / stop_distance)
                max_quantity = min(max_quantity, risk_based_quantity)
                logger.info(f"Risk-adjusted position size: {max_quantity} shares")
    
    # Prepare trade parameters
    trade_params = {
        "asset_type": "equity",
        "symbol": ticker,
        "side": action,
        "quantity": max_quantity,
        "order_type": order_type,
        "duration": "day"
    }
    
    # Add price for limit orders
    if order_type.lower() == "limit" and price:
        trade_params["price"] = price
    
    logger.info(f"Prepared equity trade: {json.dumps(trade_params)}")
    return trade_params

def validate_breakout(ticker, breakout_level, direction, tradier_api):
    """
    Validate if a breakout is still valid by checking price action.
    
    Args:
        ticker (str): Stock symbol
        breakout_level (float): Price level for the breakout
        direction (str): 'up' or 'down'
        tradier_api: Tradier API instance
    
    Returns:
        bool: True if the breakout is still valid, False otherwise
    """
    quote = tradier_api.get_quote(ticker)
    
    if not quote or "quotes" not in quote or "quote" not in quote["quotes"]:
        logger.error(f"Failed to get quote for {ticker}")
        return False
    
    current_price = quote["quotes"]["quote"].get("last", 0)
    
    if direction == "up" and current_price < float(breakout_level):
        return False
    
    if direction == "down" and current_price > float(breakout_level):
        return False
    
    # Check for price consolidation after breakout
    historical_data = tradier_api.get_historical_data(
        ticker, 
        interval="daily",
        start_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    
    if historical_data and "history" in historical_data and "day" in historical_data["history"]:
        data = historical_data["history"]["day"]
        if len(data) >= 2:
            yesterday = data[-2]
            today = data[-1]
            
            # Check for follow-through after breakout
            if direction == "up":
                if today["close"] < yesterday["close"]:
                    logger.warning("Failed follow-through after upside breakout")
                    return False
            
            if direction == "down":
                if today["close"] > yesterday["close"]:
                    logger.warning("Failed follow-through after downside breakout")
                    return False
    
    return True 