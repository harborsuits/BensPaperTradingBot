#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingView Webhook Example

This script demonstrates how to set up and test the TradingView webhook integration
with the external signal strategy. It includes examples of how to format TradingView
alerts for different signal types.
"""

import json
import logging
import requests
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradingViewWebhookExample")

def simulate_tradingview_alert(webhook_url: str, data: Dict[str, Any]) -> None:
    """
    Simulate a TradingView alert by sending data to the webhook.
    
    Args:
        webhook_url: URL of the webhook endpoint
        data: Alert data to send
    """
    try:
        logger.info(f"Sending alert for {data.get('symbol')}: {data.get('action')}")
        logger.debug(f"Alert data: {json.dumps(data)}")
        
        response = requests.post(
            webhook_url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"Alert sent successfully: {response.json()}")
        else:
            logger.error(f"Failed to send alert: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"Error sending alert: {str(e)}")


def create_tradingview_entry_alert(
    symbol: str,
    direction: str,
    price: float,
    strategy_name: str = "TradingView Strategy",
    timeframe: str = "1h",
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a TradingView entry alert payload.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        direction: Trade direction ("buy" or "sell")
        price: Current price
        strategy_name: Name of the strategy generating the alert
        timeframe: Chart timeframe 
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        additional_data: Any additional data to include
        
    Returns:
        Dictionary with the alert payload
    """
    # Base payload
    payload = {
        "symbol": symbol,
        "action": direction.lower(),  # "buy" or "sell"
        "price": price,
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "strategy": strategy_name,
        "source": "tradingview"
    }
    
    # Add stop loss and take profit if provided
    if stop_loss is not None:
        payload["stop_loss"] = stop_loss
    
    if take_profit is not None:
        payload["take_profit"] = take_profit
    
    # Add any additional data
    if additional_data:
        payload.update(additional_data)
    
    return payload


def create_tradingview_exit_alert(
    symbol: str,
    price: float,
    strategy_name: str = "TradingView Strategy",
    timeframe: str = "1h",
    reason: str = "signal",
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a TradingView exit alert payload.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        price: Current price
        strategy_name: Name of the strategy generating the alert
        timeframe: Chart timeframe
        reason: Reason for exit
        additional_data: Any additional data to include
        
    Returns:
        Dictionary with the alert payload
    """
    # Base payload
    payload = {
        "symbol": symbol,
        "action": "exit",
        "price": price,
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "strategy": strategy_name,
        "source": "tradingview",
        "exit_reason": reason
    }
    
    # Add any additional data
    if additional_data:
        payload.update(additional_data)
    
    return payload


def create_tradingview_indicator_alert(
    symbol: str,
    indicators: Dict[str, Any],
    price: float,
    timeframe: str = "1h",
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a TradingView indicator update alert payload.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        indicators: Dictionary of indicator values
        price: Current price
        timeframe: Chart timeframe
        additional_data: Any additional data to include
        
    Returns:
        Dictionary with the alert payload
    """
    # Base payload
    payload = {
        "symbol": symbol,
        "price": price,
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "source": "tradingview",
        "indicators": indicators
    }
    
    # Add any additional data
    if additional_data:
        payload.update(additional_data)
    
    return payload


def main():
    """Run the TradingView webhook example."""
    # Webhook URL - adjust port and path as needed
    webhook_port = 5000
    webhook_path = "webhook"
    webhook_url = f"http://localhost:{webhook_port}/{webhook_path}"
    
    logger.info("TradingView Webhook Example")
    logger.info(f"Using webhook URL: {webhook_url}")
    
    # Wait for user confirmation
    input("Make sure your trading bot is running with the webhook handler enabled. Press Enter to continue...")
    
    # Example 1: Entry Signal - Buy EURUSD
    logger.info("\n--- Example 1: Buy EURUSD ---")
    buy_alert = create_tradingview_entry_alert(
        symbol="EURUSD",
        direction="buy",
        price=1.0550,
        strategy_name="MA Crossover",
        timeframe="1h",
        stop_loss=1.0500,
        take_profit=1.0650,
        additional_data={
            "indicators": {
                "ema_20": 1.0530,
                "ema_50": 1.0520,
                "rsi": 65.5
            },
            "confidence": 0.85
        }
    )
    
    simulate_tradingview_alert(webhook_url, buy_alert)
    time.sleep(2)  # Wait between alerts
    
    # Example 2: Entry Signal - Sell GBPUSD
    logger.info("\n--- Example 2: Sell GBPUSD ---")
    sell_alert = create_tradingview_entry_alert(
        symbol="GBPUSD",
        direction="sell",
        price=1.2850,
        strategy_name="Bollinger Band Reversal",
        timeframe="4h",
        stop_loss=1.2900,
        take_profit=1.2750,
        additional_data={
            "indicators": {
                "upper_band": 1.2900,
                "middle_band": 1.2825,
                "lower_band": 1.2750,
                "rsi": 28.3
            },
            "confidence": 0.78
        }
    )
    
    simulate_tradingview_alert(webhook_url, sell_alert)
    time.sleep(2)  # Wait between alerts
    
    # Example 3: Exit Signal - EURUSD
    logger.info("\n--- Example 3: Exit EURUSD ---")
    exit_alert = create_tradingview_exit_alert(
        symbol="EURUSD",
        price=1.0620,
        strategy_name="MA Crossover",
        timeframe="1h",
        reason="take_profit",
        additional_data={
            "profit_pips": 70,
            "trade_duration_hours": 5.5
        }
    )
    
    simulate_tradingview_alert(webhook_url, exit_alert)
    time.sleep(2)  # Wait between alerts
    
    # Example 4: Indicator Update - BTCUSD
    logger.info("\n--- Example 4: Indicator Update BTCUSD ---")
    indicator_alert = create_tradingview_indicator_alert(
        symbol="BTCUSD",
        price=38500.50,
        timeframe="1d",
        indicators={
            "rsi": 45.2,
            "ema_20": 37800.25,
            "ema_50": 36250.75,
            "macd": 125.5,
            "macd_signal": 100.2,
            "macd_histogram": 25.3,
            "atr": 850.25,
            "market_regime": "ranging"
        }
    )
    
    simulate_tradingview_alert(webhook_url, indicator_alert)
    
    logger.info("\nAll example alerts sent. Check your trading bot's logs and UI for the processed signals.")
    logger.info("You can use these examples as templates for setting up your actual TradingView alerts.")
    
    
if __name__ == "__main__":
    main()
