#!/usr/bin/env python3
"""
Simplified API for the Trading Dashboard
This is a standalone API that provides just the endpoints needed by the dashboard.
"""
import os
import sys
import json
import random
import logging
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any, Optional, Union

# Add the trading_bot directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_bot'))

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import broker infrastructure
try:
    import requests
    import yaml
    BROKERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required packages: {e}")
    BROKERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardAPI")

# Create FastAPI app
app = FastAPI(
    title="Trading Dashboard API",
    description="Simplified API for the Trading Dashboard",
    version="1.0.0"
)

# Add CORS middleware to allow requests from frontend with more detailed configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_origin_regex=".*",  # Allow all origins with regex pattern
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Global broker clients
tradier_config = None
alpaca_config = None
coinbase_config = None

def initialize_brokers():
    """Initialize broker connections using the trading config"""
    print("=== INITIALIZE_BROKERS CALLED ===")
    global tradier_config, alpaca_config, coinbase_config
    
    logger.info("Starting broker initialization...")
    
    if not BROKERS_AVAILABLE:
        logger.warning("Required packages not available - using mock data")
        return
    
    try:
        # Load configuration from trading_config.yaml
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'trading_bot', 'config', 'trading_config.yaml')
        print(f"Looking for config at: {config_path}")
        logger.info(f"Looking for config at: {config_path}")
        
        if not os.path.exists(config_path):
            config_path = os.path.join(base_dir, 'temp_keys', 'trading_config.yaml')
            print(f"Primary config not found, trying: {config_path}")
            logger.info(f"Primary config not found, trying: {config_path}")
        
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            logger.info(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                trading_config = yaml.safe_load(f)
            
            print(f"Config loaded successfully. Keys: {list(trading_config.keys())}")
            logger.info(f"Config loaded successfully. Keys: {list(trading_config.keys())}")
            
            # Store Tradier config
            if 'tradier' in trading_config:
                tradier_config = {
                    'api_key': trading_config['tradier'].get('api_key'),
                    'account_number': trading_config['tradier'].get('account_number'),
                    'base_url': 'https://sandbox.tradier.com/v1'  # Paper trading
                }
                print(f"Configured Tradier API with account: {tradier_config['account_number']}")
                logger.info(f"Configured Tradier API with account: {tradier_config['account_number']}")
            else:
                print("No 'tradier' section found in config")
                logger.warning("No 'tradier' section found in config")
            
            # Store Alpaca config
            if 'alpaca' in trading_config:
                alpaca_config = {
                    'api_key': trading_config['alpaca'].get('api_key'),
                    'api_secret': trading_config['alpaca'].get('api_secret'),
                    'base_url': 'https://paper-api.alpaca.markets/v2'  # Paper trading
                }
                print(f"Configured Alpaca API with key: {alpaca_config['api_key'][:10]}...")
                logger.info(f"Configured Alpaca API with key: {alpaca_config['api_key'][:10]}...")
            else:
                print("No 'alpaca' section found in config")
                logger.warning("No 'alpaca' section found in config")
            
            # Store Coinbase config
            if 'coinbase' in trading_config:
                coinbase_config = {
                    'api_key_name': trading_config['coinbase'].get('api_key_name'),
                    'private_key': trading_config['coinbase'].get('private_key'),
                    'base_url': 'https://api.coinbase.com'
                }
                print("Configured Coinbase API")
                logger.info("Configured Coinbase API")
            else:
                print("No 'coinbase' section found in config")
                logger.warning("No 'coinbase' section found in config")
        else:
            print(f"Config file not found at any expected location")
            logger.error(f"Config file not found at any expected location")
                
    except Exception as e:
        print(f"Failed to initialize brokers: {e}")
        logger.error(f"Failed to initialize brokers: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Traceback: {traceback.format_exc()}")

# Initialize brokers on startup
print("=== ABOUT TO INITIALIZE BROKERS ===")
try:
    initialize_brokers()
    print("=== BROKER INITIALIZATION COMPLETED ===")
except Exception as e:
    print(f"=== BROKER INITIALIZATION FAILED: {e} ===")
    import traceback
    traceback.print_exc()

# API Models
class PortfolioData(BaseModel):
    totalValue: float
    dailyChange: float
    dailyChangePercent: float
    monthlyReturn: float
    weeklyChange: float
    allocation: List[Dict[str, Any]]
    holdings: List[Dict[str, Any]]

def get_real_portfolio_data():
    """Get real portfolio data from brokers"""
    try:
        total_value = 0
        holdings = []
        allocation = []
        
        logger.info(f"Attempting to fetch real portfolio data...")
        logger.info(f"Tradier config available: {tradier_config is not None}")
        logger.info(f"Alpaca config available: {alpaca_config is not None}")
        logger.info(f"Coinbase config available: {coinbase_config is not None}")
        
        # Get Tradier portfolio
        if tradier_config:
            try:
                logger.info(f"Fetching Tradier data from {tradier_config['base_url']}")
                tradier_positions = requests.get(f"{tradier_config['base_url']}/accounts/{tradier_config['account_number']}/positions", headers={"Authorization": f"Bearer {tradier_config['api_key']}"}).json()
                tradier_balance = requests.get(f"{tradier_config['base_url']}/accounts/{tradier_config['account_number']}/balance", headers={"Authorization": f"Bearer {tradier_config['api_key']}"}).json()
                
                logger.info(f"Tradier balance response: {tradier_balance}")
                logger.info(f"Tradier positions response: {tradier_positions}")
                
                if tradier_balance:
                    tradier_value = float(tradier_balance.get('total_equity', 0))
                    total_value += tradier_value
                    
                    allocation.append({
                        "name": "Tradier (Stocks/Options)",
                        "value": tradier_value,
                        "percentage": 0  # Will calculate later
                    })
                
                if tradier_positions:
                    for position in tradier_positions:
                        holdings.append({
                            "symbol": position.get('symbol', 'UNKNOWN'),
                            "quantity": float(position.get('quantity', 0)),
                            "currentPrice": float(position.get('last', 0)),
                            "marketValue": float(position.get('quantity', 0)) * float(position.get('last', 0)),
                            "unrealizedPL": float(position.get('day_change', 0)),
                            "broker": "Tradier"
                        })
            except Exception as e:
                logger.error(f"Error fetching Tradier data: {e}")
        
        # Get Alpaca portfolio
        if alpaca_config:
            try:
                logger.info(f"Fetching Alpaca data from {alpaca_config['base_url']}")
                alpaca_account = requests.get(f"{alpaca_config['base_url']}/accounts/me", headers={"Authorization": f"Bearer {alpaca_config['api_key']}"}).json()
                alpaca_positions = requests.get(f"{alpaca_config['base_url']}/positions", headers={"Authorization": f"Bearer {alpaca_config['api_key']}"}).json()
                
                logger.info(f"Alpaca account response: {alpaca_account}")
                logger.info(f"Alpaca positions response: {alpaca_positions}")
                
                if alpaca_account:
                    alpaca_value = float(alpaca_account.get('portfolio_value', 0))
                    total_value += alpaca_value
                    
                    allocation.append({
                        "name": "Alpaca (Stocks)",
                        "value": alpaca_value,
                        "percentage": 0  # Will calculate later
                    })
                
                if alpaca_positions:
                    for position in alpaca_positions:
                        holdings.append({
                            "symbol": position.get('symbol', 'UNKNOWN'),
                            "quantity": float(position.get('qty', 0)),
                            "currentPrice": float(position.get('current_price', 0)),
                            "marketValue": float(position.get('market_value', 0)),
                            "unrealizedPL": float(position.get('unrealized_pl', 0)),
                            "broker": "Alpaca"
                        })
            except Exception as e:
                logger.error(f"Error fetching Alpaca data: {e}")
        
        # Get Coinbase portfolio
        if coinbase_config:
            try:
                logger.info(f"Fetching Coinbase data from {coinbase_config['base_url']}")
                coinbase_accounts = requests.get(f"{coinbase_config['base_url']}/accounts", headers={"Authorization": f"Bearer {coinbase_config['api_key_name']}"}).json()
                
                logger.info(f"Coinbase accounts response: {coinbase_accounts}")
                
                if coinbase_accounts:
                    coinbase_value = 0
                    for account in coinbase_accounts:
                        balance = float(account.get('available_balance', {}).get('value', 0))
                        coinbase_value += balance
                        
                        if balance > 0:
                            holdings.append({
                                "symbol": account.get('currency', 'UNKNOWN'),
                                "quantity": balance,
                                "currentPrice": 1.0,  # For crypto, this would need price lookup
                                "marketValue": balance,
                                "unrealizedPL": 0,
                                "broker": "Coinbase"
                            })
                    
                    total_value += coinbase_value
                    allocation.append({
                        "name": "Coinbase (Crypto)",
                        "value": coinbase_value,
                        "percentage": 0  # Will calculate later
                    })
            except Exception as e:
                logger.error(f"Error fetching Coinbase data: {e}")
        
        # Calculate allocation percentages
        if total_value > 0:
            for item in allocation:
                item["percentage"] = round((item["value"] / total_value) * 100, 1)
        
        logger.info(f"Total portfolio value from real brokers: ${total_value}")
        
        # If no real data, fall back to mock data
        if total_value == 0:
            logger.warning("No real broker data found, falling back to mock data")
            return get_mock_portfolio_data()
        
        return {
            "totalValue": round(total_value, 2),
            "dailyChange": round(random.uniform(-500, 500), 2),  # TODO: Calculate real daily change
            "dailyChangePercent": round(random.uniform(-2, 2), 2),
            "monthlyReturn": round(random.uniform(-5, 15), 2),
            "weeklyChange": round(random.uniform(-3, 3), 2),
            "allocation": allocation,
            "holdings": holdings
        }
        
    except Exception as e:
        logger.error(f"Error getting real portfolio data: {e}")
        return get_mock_portfolio_data()

def get_mock_portfolio_data():
    """Fallback mock data when real brokers aren't available"""
    # Generate mock portfolio data
    total_value = round(random.uniform(50000, 150000), 2)
    daily_change = round(random.uniform(-2000, 2000), 2)
    daily_change_percent = round((daily_change / total_value) * 100, 2)
    
    # Mock holdings
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    holdings = []
    
    for symbol in symbols[:5]:  # Show 5 holdings
        quantity = random.randint(10, 100)
        entry_price = round(random.uniform(100, 500), 2)
        current_price = round(entry_price * random.uniform(0.9, 1.1), 2)
        market_value = round(quantity * current_price, 2)
        unrealized_pl = round(quantity * (current_price - entry_price), 2)
        
        holdings.append({
            "symbol": symbol,
            "quantity": quantity,
            "currentPrice": current_price,
            "marketValue": market_value,
            "unrealizedPL": unrealized_pl,
            "broker": "Mock"
        })
    
    return {
        "totalValue": total_value,
        "dailyChange": daily_change,
        "dailyChangePercent": daily_change_percent,
        "monthlyReturn": round(random.uniform(-5, 15), 2),
        "weeklyChange": round(random.uniform(-3, 3), 2),
        "allocation": [
            {"name": "Stocks", "value": total_value * 0.6, "percentage": 60.0},
            {"name": "Options", "value": total_value * 0.2, "percentage": 20.0},
            {"name": "Cash", "value": total_value * 0.2, "percentage": 20.0}
        ],
        "holdings": holdings
    }

@app.get("/api/portfolio", response_model=PortfolioData)
async def get_portfolio(account: str = "live"):
    """Get portfolio data from real brokers or mock data"""
    try:
        if BROKERS_AVAILABLE and (tradier_config or alpaca_config or coinbase_config):
            data = get_real_portfolio_data()
            logger.info(f"Returning real portfolio data: ${data['totalValue']}")
        else:
            data = get_mock_portfolio_data()
            logger.info(f"Returning mock portfolio data: ${data['totalValue']}")
        
        return PortfolioData(**data)
    except Exception as e:
        logger.error(f"Error in get_portfolio: {str(e)}")
        # Return mock data as fallback
        data = get_mock_portfolio_data()
        return PortfolioData(**data)

@app.get("/api/strategies")
async def get_strategies():
    """Get trading strategies"""
    try:
        strategies = [
            {
                "id": "trend_following",
                "name": "Trend Following",
                "status": "active",
                "performance": {
                    "daily": round(random.uniform(-2, 2), 2),
                    "weekly": round(random.uniform(-5, 5), 2),
                    "monthly": round(random.uniform(-10, 10), 2),
                    "yearly": round(random.uniform(-20, 40), 2)
                },
                "allocation": round(random.uniform(2, 15), 1),
                "description": "Follows market trends using technical indicators",
                "riskLevel": "Medium",
                "activeTrades": random.randint(0, 5),
                "signalStrength": round(random.uniform(0, 1), 2),
                "lastUpdated": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat()
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "status": "active",
                "performance": {
                    "daily": round(random.uniform(-2, 2), 2),
                    "weekly": round(random.uniform(-5, 5), 2),
                    "monthly": round(random.uniform(-10, 10), 2),
                    "yearly": round(random.uniform(-20, 40), 2)
                },
                "allocation": round(random.uniform(2, 15), 1),
                "description": "Trades on price reversions to mean",
                "riskLevel": "Low",
                "activeTrades": random.randint(0, 5),
                "signalStrength": round(random.uniform(0, 1), 2),
                "lastUpdated": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat()
            }
        ]
        
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching strategies: {str(e)}")

@app.get("/api/trades")
async def get_trades(account: str = "live", limit: int = 10):
    """Get recent trades"""
    try:
        trades = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        for i in range(limit):
            symbol = random.choice(symbols)
            side = random.choice(["buy", "sell"])
            entry_price = round(random.uniform(100, 500), 2)
            quantity = random.randint(10, 100)
            current_price = round(entry_price * random.uniform(0.9, 1.1), 2)
            pnl = round(quantity * (current_price - entry_price) * (1 if side == "buy" else -1), 2)
            
            days_ago = random.randint(0, 30)
            trade_time = datetime.now() - timedelta(days=days_ago)
            
            trade = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": entry_price,
                "currentPrice": current_price,
                "pnl": pnl,
                "status": random.choice(["filled", "open", "cancelled"]),
                "timestamp": trade_time.isoformat(),
                "strategy": random.choice(["trend_following", "mean_reversion", "momentum"])
            }
            
            trades.append(trade)
        
        # Sort by timestamp (most recent first)
        trades.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {"trades": trades}
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {str(e)}")

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    try:
        # Generate performance data
        performance_data = []
        base_value = 100000
        
        for i in range(30):  # 30 days of data
            date = datetime.now() - timedelta(days=29-i)
            change = random.uniform(-0.02, 0.02)  # Daily volatility
            base_value *= (1 + change)
            
            performance_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(base_value, 2),
                "change": round(change * 100, 2)
            })
        
        return {"performance": performance_data}
    except Exception as e:
        logger.error(f"Error fetching performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching performance: {str(e)}")

@app.get("/api/alerts")
async def get_alerts(limit: int = 5):
    """Get recent alerts"""
    try:
        alerts = []
        alert_types = ["price_alert", "strategy_signal", "risk_warning", "news_alert"]
        
        for i in range(limit):
            alert_type = random.choice(alert_types)
            hours_ago = random.randint(1, 72)
            alert_time = datetime.now() - timedelta(hours=hours_ago)
            
            if alert_type == "price_alert":
                symbol = random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
                message = f"{symbol} price alert triggered"
                severity = "info"
            elif alert_type == "strategy_signal":
                strategy = random.choice(["Trend Following", "Mean Reversion"])
                message = f"New signal from {strategy} strategy"
                severity = "info"
            elif alert_type == "risk_warning":
                message = "Portfolio risk threshold exceeded"
                severity = "warning"
            else:  # news_alert
                message = "Market news alert"
                severity = "info"
            
            read = random.choice([True, False])
            
            alert = {
                "id": str(uuid.uuid4()),
                "type": alert_type,
                "message": message,
                "timestamp": alert_time.isoformat(),
                "read": read,
                "category": random.choice(["notification", "warning", "error"]) if random.random() < 0.7 else None,
                "severity": severity,
                "details": {}
            }
            
            alerts.append(alert)
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@app.get("/api/debug/brokers")
async def debug_brokers():
    """Debug endpoint to check broker configuration status"""
    return {
        "brokers_available": BROKERS_AVAILABLE,
        "tradier_config": tradier_config is not None,
        "alpaca_config": alpaca_config is not None,
        "coinbase_config": coinbase_config is not None,
        "tradier_details": {
            "api_key": tradier_config['api_key'][:10] + "..." if tradier_config and tradier_config.get('api_key') else None,
            "account_number": tradier_config.get('account_number') if tradier_config else None
        } if tradier_config else None,
        "alpaca_details": {
            "api_key": alpaca_config['api_key'][:10] + "..." if alpaca_config and alpaca_config.get('api_key') else None,
            "base_url": alpaca_config.get('base_url') if alpaca_config else None
        } if alpaca_config else None
    }

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765) 