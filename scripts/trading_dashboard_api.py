#!/usr/bin/env python3
"""
Simplified API for the Trading Dashboard
This is a standalone API that provides just the endpoints needed by the dashboard.
"""
import os
import json
import random
import logging
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any, Optional, Union

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# API Models
class PortfolioData(BaseModel):
    totalValue: float
    dailyChange: float
    dailyChangePercent: float
    monthlyReturn: float
    weeklyChange: float
    allocation: List[Dict[str, Any]]
    holdings: List[Dict[str, Any]]

class TradeData(BaseModel):
    id: str
    symbol: str
    entry: float
    entryPrice: float
    quantity: float
    currentPrice: float
    pnl: float
    pnlPercent: float
    status: str
    strategy: str
    side: str
    openedAt: str
    type: str

class StrategyData(BaseModel):
    name: str
    description: str
    status: str
    allocation: float
    daily: float
    weekly: float
    monthly: float
    yearly: float
    activeTrades: int
    signalStrength: float
    lastUpdated: str

class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    url: str
    source: str
    imageUrl: Optional[str] = None
    publishedAt: str
    sentiment: Optional[str] = None
    symbols: Optional[List[str]] = None
    impact: Optional[str] = None

class AlertItem(BaseModel):
    id: str
    message: str
    type: str
    timestamp: str
    source: str
    read: bool
    category: Optional[str] = None
    severity: str
    details: Optional[Dict[str, Any]] = None

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API info."""
    return {
        "message": "Trading Dashboard API",
        "version": "1.0.0",
        "endpoints": [
            "/api/portfolio",
            "/api/trades",
            "/api/strategies",
            "/api/news",
            "/api/alerts"
        ]
    }

# Portfolio Endpoints
@app.get("/api/portfolio", response_model=PortfolioData)
async def get_portfolio(account: str = "live"):
    """Get portfolio data for the dashboard."""
    try:
        # Create sample portfolio data
        total_value = round(random.uniform(50000, 150000), 2)
        daily_change = round(random.uniform(-2000, 2000), 2)
        daily_change_percent = round((daily_change / (total_value - daily_change) * 100), 2)
        
        # Create sample allocation data
        allocation = [
            {"category": "Stocks", "value": round(total_value * 0.6), "color": "#1976d2"},
            {"category": "Options", "value": round(total_value * 0.2), "color": "#388e3c"},
            {"category": "Cash", "value": round(total_value * 0.2), "color": "#f57c00"}
        ]
        
        # Create sample holdings data
        holdings = []
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
        for symbol in symbols:
            quantity = random.randint(10, 100)
            entry_price = round(random.uniform(100, 500), 2)
            current_price = round(entry_price * random.uniform(0.9, 1.1), 2)
            value = round(quantity * current_price, 2)
            unrealized_pnl = round(quantity * (current_price - entry_price), 2)
            unrealized_pnl_percent = round((current_price / entry_price - 1) * 100, 2)
            
            holdings.append({
                "symbol": symbol,
                "name": f"{symbol} Inc.",
                "quantity": quantity,
                "entryPrice": entry_price,
                "currentPrice": current_price,
                "value": value,
                "unrealizedPnl": unrealized_pnl,
                "unrealizedPnlPercent": unrealized_pnl_percent
            })
        
        portfolio_data = {
            "totalValue": total_value,
            "dailyChange": daily_change,
            "dailyChangePercent": daily_change_percent,
            "monthlyReturn": round(random.uniform(-5, 15), 2),
            "weeklyChange": round(random.uniform(-3, 3), 2),
            "allocation": allocation,
            "holdings": holdings
        }
        
        return portfolio_data
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@app.get("/api/performance")
async def get_performance_history(broker: str = "tradier"):
    """Get performance history for the dashboard."""
    try:
        # Create performance history data with some realistic trends
        base = 100.0
        daily_points = 30
        weekly_points = 12
        monthly_points = 12
        yearly_points = 5
        
        # Generate daily performance with some randomness but overall trend
        daily = []
        trend = random.choice([1, -1]) * 0.002  # Small daily trend
        for i in range(daily_points):
            change = trend + random.uniform(-0.01, 0.01)  # Daily volatility
            base = base * (1 + change)
            daily.append(round(base, 2))
        
        # Weekly performance (more significant changes)
        weekly = []
        base = 100.0
        trend = random.choice([1, -1]) * 0.01  # Larger weekly trend
        for i in range(weekly_points):
            change = trend + random.uniform(-0.03, 0.03)  # Weekly volatility
            base = base * (1 + change)
            weekly.append(round(base, 2))
        
        # Monthly performance
        monthly = []
        base = 100.0
        trend = random.choice([1, -1]) * 0.02  # Monthly trend
        for i in range(monthly_points):
            change = trend + random.uniform(-0.05, 0.05)  # Monthly volatility
            base = base * (1 + change)
            monthly.append(round(base, 2))
        
        # Yearly performance
        yearly = []
        base = 100.0
        trend = random.choice([1, -1]) * 0.05  # Yearly trend
        for i in range(yearly_points):
            change = trend + random.uniform(-0.1, 0.1)  # Yearly volatility
            base = base * (1 + change)
            yearly.append(round(base, 2))
        
        # Current return (last vs first)
        current_return = round((daily[-1] / daily[0] - 1) * 100, 2)
        
        return {
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly,
            "yearly": yearly,
            "currentReturn": current_return
        }
    except Exception as e:
        logger.error(f"Error fetching performance history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching performance history: {str(e)}")

# Trades Endpoints
@app.get("/api/trades", response_model=List[TradeData])
async def get_trades(account: str = "live", limit: int = 20):
    """Get trades data for the dashboard."""
    try:
        # Create sample trades data
        trades = []
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "TSLA", "AMD", "NVDA", "PYPL"]
        strategies = ["GapTrading", "TrendFollowing", "BreakoutStrategy", "MomentumStrategy", "RSIStrategy", "MACDStrategy"]
        
        for i in range(min(limit, 20)):
            symbol = random.choice(symbols)
            strategy = random.choice(strategies)
            side = random.choice(["long", "short"])
            status = random.choice(["open", "closed"] if i > 5 else ["open"])
            
            entry_price = round(random.uniform(100, 500), 2)
            quantity = random.randint(10, 100)
            current_price = round(entry_price * random.uniform(0.9, 1.1), 2)
            
            # Calculate P&L
            multiplier = 1 if side == "long" else -1
            pnl = round(multiplier * quantity * (current_price - entry_price), 2)
            pnl_percent = round(multiplier * (current_price / entry_price - 1) * 100, 2)
            
            # Create date within the last 30 days
            days_ago = random.randint(0, 30)
            opened_at = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            trade = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "entry": entry_price,
                "entryPrice": entry_price,
                "quantity": quantity,
                "currentPrice": current_price,
                "pnl": pnl,
                "pnlPercent": pnl_percent,
                "status": status,
                "strategy": strategy,
                "side": side,
                "openedAt": opened_at,
                "type": "stock"  # Could be stock, option, crypto, etc.
            }
            
            trades.append(trade)
        
        return trades
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {str(e)}")

# Strategies Endpoints
@app.get("/api/strategies", response_model=List[StrategyData])
async def get_strategies(status: Optional[str] = None):
    """Get trading strategies filtered by status."""
    try:
        # Define strategy templates based on memory of available strategies
        stock_strategies = [
            {"name": "StocksTrendFollowingStrategy", "description": "Follows medium-term market trends for stocks"},
            {"name": "GapTradingStrategy", "description": "Trades gaps in stock prices at market open"},
            {"name": "EarningsAnnouncementStrategy", "description": "Trades around earnings announcements"},
            {"name": "NewsSentimentStrategy", "description": "Trades based on news sentiment analysis"},
            {"name": "SectorRotationStrategy", "description": "Rotates between sectors based on market cycles"},
            {"name": "ShortSellingStrategy", "description": "Short sells overvalued or declining stocks"},
            {"name": "VolumeSurgeStrategy", "description": "Trades unusually high volume patterns"}
        ]
        
        options_strategies = [
            {"name": "BullCallSpreadStrategy", "description": "Bull call spread for moderately bullish outlook"},
            {"name": "BearPutSpreadStrategy", "description": "Bear put spread for moderately bearish outlook"},
            {"name": "IronCondorStrategy", "description": "Iron condor for range-bound markets"},
            {"name": "StraddleStrategy", "description": "Straddle for high volatility events"}
        ]
        
        forex_strategies = [
            {"name": "ForexTrendFollowingStrategy", "description": "Follows trends in currency pairs"},
            {"name": "ForexBreakoutStrategy", "description": "Trades breakouts in currency pairs"}
        ]
        
        # Combine all strategies
        all_strategies = []
        all_strategies.extend(stock_strategies)
        all_strategies.extend(options_strategies)
        all_strategies.extend(forex_strategies)
        
        # Create full strategy objects with performance metrics
        strategies = []
        for strategy_template in all_strategies:
            # Generate random performance data
            daily = round(random.uniform(-2, 2), 2)
            weekly = round(random.uniform(-5, 5), 2)
            monthly = round(random.uniform(-10, 10), 2)
            yearly = round(random.uniform(-20, 40), 2)
            
            # Determine status
            strategy_status = random.choice(["active", "inactive", "testing"]) if not status else status
            
            # Create strategy object
            strategy = {
                "name": strategy_template["name"],
                "description": strategy_template["description"],
                "status": strategy_status,
                "allocation": round(random.uniform(2, 15), 1),
                "daily": daily,
                "weekly": weekly,
                "monthly": monthly,
                "yearly": yearly,
                "activeTrades": random.randint(0, 5),
                "signalStrength": round(random.uniform(0, 1), 2),
                "lastUpdated": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat()
            }
            
            strategies.append(strategy)
        
        # Filter by status if provided
        if status:
            strategies = [s for s in strategies if s["status"] == status]
            
        return strategies
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching strategies: {str(e)}")

# News Endpoints
@app.get("/api/news", response_model=List[NewsItem])
async def get_news(symbol: Optional[str] = None, category: Optional[str] = None, limit: int = 20):
    """Get financial news, optionally filtered by symbol or category."""
    try:
        # News sources that match what was mentioned in memories
        sources = ["Alpha Vantage", "NewsData.io", "GNews", "MediaStack", "Currents", "NYTimes"]
        
        # Create sample news items
        news_items = []
        
        # Generate news with realistic titles and content
        for i in range(min(limit, 20)):
            # Generate a random date within the last 3 days
            hours_ago = random.randint(1, 72)
            published_at = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
            
            # Random symbols
            symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"] if not symbol else [symbol]
            
            # Create the news item
            news_item = {
                "id": str(uuid.uuid4()),
                "title": f"Market {random.choice(['update', 'analysis', 'news'])}: {random.choice(symbols)} {random.choice(['rises', 'falls', 'stabilizes'])}",
                "summary": f"Financial news summary about {random.choice(symbols)} with important market information and analysis.",
                "url": f"https://finance.example.com/article/{uuid.uuid4()}",
                "source": random.choice(sources),
                "imageUrl": f"https://via.placeholder.com/300x200?text={random.choice(symbols)}" if random.random() > 0.3 else None,
                "publishedAt": published_at,
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "symbols": symbols,
                "impact": random.choice(["high", "medium", "low"])
            }
            
            news_items.append(news_item)
        
        return news_items
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

# Alerts Endpoints
@app.get("/api/alerts", response_model=List[AlertItem])
async def get_alerts(limit: int = 20):
    """Get system alerts and notifications."""
    try:
        # Create sample alerts
        alerts = []
        
        # Alert types and categories
        alert_types = ["system", "trade", "strategy", "market", "account"]
        severities = ["critical", "high", "medium", "low", "info"]
        
        for i in range(min(limit, 20)):
            alert_type = random.choice(alert_types)
            severity = random.choice(severities)
            
            # Create timestamp (more recent for higher severity)
            if severity in ["critical", "high"]:
                minutes_ago = random.randint(1, 60)
            else:
                minutes_ago = random.randint(60, 1440)  # 1-24 hours
            
            timestamp = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
            
            # Create alert message based on type
            if alert_type == "system":
                message = f"System {random.choice(['health check', 'update', 'maintenance'])} {random.choice(['completed', 'failed', 'in progress'])}"
            elif alert_type == "trade":
                symbol = random.choice(["AAPL", "MSFT", "AMZN", "GOOGL", "META"])
                price = round(random.uniform(100, 500), 2)
                message = f"{symbol} trade {random.choice(['executed', 'failed', 'pending'])} at ${price}"
            elif alert_type == "strategy":
                strategy = random.choice(["TrendFollowing", "MACD", "RSI", "VolumeSurge"])
                message = f"{strategy} strategy {random.choice(['activated', 'deactivated', 'updated'])}"
            elif alert_type == "market":
                message = f"Market {random.choice(['volatility', 'trend change', 'news impact'])} detected"
            else:  # account
                message = f"Account {random.choice(['connection', 'balance', 'settings'])} {random.choice(['updated', 'changed', 'verified'])}"
            
            # Determine if alert has been read (older alerts more likely to be read)
            read = random.random() < (minutes_ago / 1440)
            
            alert = {
                "id": str(uuid.uuid4()),
                "message": message,
                "type": alert_type,
                "timestamp": timestamp,
                "source": random.choice(["System", "TradingEngine", "RiskManager", "MarketMonitor"]),
                "read": read,
                "category": random.choice(["notification", "warning", "error"]) if random.random() < 0.7 else None,
                "severity": severity,
                "details": {}
            }
            
            alerts.append(alert)
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return alerts
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
