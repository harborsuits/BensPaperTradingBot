#!/usr/bin/env python3
"""
Simple API server for the trading dashboard
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import random

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    print("ERROR: Please install FastAPI and uvicorn using:")
    print("pipx install fastapi uvicorn")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_api")

# Initialize FastAPI app
app = FastAPI(
    title="Trading Dashboard API",
    description="Simple API for the trading dashboard",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data generators
def generate_orders(limit=20):
    order_types = ["market", "limit", "stop", "stop_limit"]
    statuses = ["filled", "partially_filled", "open", "canceled", "rejected"]
    sides = ["buy", "sell"]
    
    orders = []
    for i in range(limit):
        filled_quantity = random.randint(0, 100) if statuses[i % len(statuses)] != "open" else 0
        order = {
            "id": f"ord-{i+1000}",
            "symbol": random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]),
            "side": random.choice(sides),
            "order_type": random.choice(order_types),
            "quantity": random.randint(1, 100),
            "filled_quantity": filled_quantity,
            "price": round(random.uniform(100, 500), 2),
            "status": statuses[i % len(statuses)],
            "created_at": (datetime.now() - timedelta(days=i % 10, hours=i % 24)).isoformat(),
            "updated_at": (datetime.now() - timedelta(hours=i % 12)).isoformat(),
        }
        orders.append(order)
    return orders

def generate_positions():
    positions = []
    for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]):
        is_profitable = random.choice([True, False])
        price_change = round(random.uniform(0.5, 5.5), 2)
        positions.append({
            "id": f"pos-{i+1000}",
            "symbol": symbol,
            "quantity": random.randint(10, 1000),
            "average_price": round(random.uniform(100, 500), 2),
            "current_price": round(random.uniform(100, 500), 2),
            "unrealized_pl": round(random.uniform(1000, 10000) * (1 if is_profitable else -1), 2),
            "unrealized_pl_percent": round(random.uniform(1, 15) * (1 if is_profitable else -1), 2),
            "market_value": round(random.uniform(10000, 100000), 2),
            "updated_at": datetime.now().isoformat(),
        })
    return positions

def generate_market_data():
    return {
        "market_regime": {
            "current_regime": random.choice(["bullish", "bearish", "neutral", "volatile"]),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "previous_regime": random.choice(["bullish", "bearish", "neutral", "volatile"]),
            "regime_change_date": (datetime.now() - timedelta(days=random.randint(5, 30))).strftime("%Y-%m-%d"),
            "supporting_indicators": [
                {"name": "RSI", "value": random.randint(30, 70), "contribution": round(random.uniform(0.1, 0.5), 2)},
                {"name": "MACD", "value": round(random.uniform(-2, 2), 2), "contribution": round(random.uniform(0.1, 0.5), 2)},
                {"name": "Moving Averages", "value": random.randint(0, 1), "contribution": round(random.uniform(0.1, 0.5), 2)},
            ],
            "trend_direction": random.choice(["bullish", "bearish", "neutral"]),
            "volatility_level": random.choice(["low", "medium", "high"]),
            "duration_days": random.randint(5, 90),
        },
        "sentiment": {
            "overall_sentiment": round(random.uniform(-0.8, 0.8), 2),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "news_sentiment": round(random.uniform(-0.8, 0.8), 2),
            "social_sentiment": round(random.uniform(-0.8, 0.8), 2),
            "analyst_sentiment": round(random.uniform(-0.8, 0.8), 2),
            "sources": [
                {"name": "News Articles", "count": random.randint(50, 500), "sentiment": round(random.uniform(-0.8, 0.8), 2)},
                {"name": "Twitter", "count": random.randint(1000, 10000), "sentiment": round(random.uniform(-0.8, 0.8), 2)},
                {"name": "Reddit", "count": random.randint(500, 5000), "sentiment": round(random.uniform(-0.8, 0.8), 2)},
                {"name": "StockTwits", "count": random.randint(200, 2000), "sentiment": round(random.uniform(-0.8, 0.8), 2)},
            ],
            "updated_at": datetime.now().isoformat(),
        },
        "market_indices": [
            {"name": "S&P 500", "value": round(random.uniform(4000, 5000), 2), "change": round(random.uniform(-2, 2), 2)},
            {"name": "Nasdaq", "value": round(random.uniform(12000, 16000), 2), "change": round(random.uniform(-2, 2), 2)},
            {"name": "Dow Jones", "value": round(random.uniform(30000, 36000), 2), "change": round(random.uniform(-2, 2), 2)},
            {"name": "Russell 2000", "value": round(random.uniform(1800, 2200), 2), "change": round(random.uniform(-2, 2), 2)},
        ]
    }

def generate_portfolio_summary():
    total_value = round(random.uniform(800000, 1200000), 2)
    daily_change_pct = round(random.uniform(-2, 2), 2)
    daily_change = round(total_value * daily_change_pct / 100, 2)
    
    return {
        "total_value": total_value,
        "daily_change": daily_change,
        "daily_change_percent": daily_change_pct,
        "monthly_return": round(random.uniform(-10, 10), 1),
        "ytd_return": round(random.uniform(-20, 20), 1),
        "total_return": round(random.uniform(-30, 30), 1),
        "account_details": {
            "cash_balance": round(random.uniform(100000, 300000), 2),
            "margin_used": round(random.uniform(0, 200000), 2),
            "buying_power": round(random.uniform(300000, 600000), 2),
        },
        "allocations": [
            {"category": "Stocks", "value": round(random.uniform(300000, 600000), 2), "percentage": round(random.uniform(30, 60), 1)},
            {"category": "Options", "value": round(random.uniform(50000, 150000), 2), "percentage": round(random.uniform(5, 15), 1)},
            {"category": "Crypto", "value": round(random.uniform(20000, 100000), 2), "percentage": round(random.uniform(2, 10), 1)},
            {"category": "Cash", "value": round(random.uniform(100000, 300000), 2), "percentage": round(random.uniform(10, 30), 1)},
        ],
        "updated_at": datetime.now().isoformat(),
    }

def generate_alerts(limit=10):
    severities = ["low", "medium", "high", "critical"]
    statuses = ["new", "acknowledged", "resolved", "dismissed"]
    categories = ["price", "volatility", "technical", "fundamental", "system"]
    
    alerts = []
    for i in range(limit):
        alerts.append({
            "id": f"alert-{i+1000}",
            "title": f"Alert #{i+1000}: {random.choice(['Price movement', 'Volatility spike', 'Technical breakout', 'Earnings announcement', 'System warning'])}",
            "message": f"This is a sample alert message with details about alert #{i+1000}.",
            "severity": severities[i % len(severities)],
            "status": statuses[i % len(statuses)],
            "category": categories[i % len(categories)],
            "created_at": (datetime.now() - timedelta(hours=i)).isoformat(),
            "updated_at": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
            "source": random.choice(["system", "user", "strategy", "market"]),
            "related_symbols": random.sample(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], k=random.randint(0, 3)),
        })
    return alerts

def generate_system_status():
    components = [
        {"name": "Data Feed", "status": random.choice(["operational", "degraded", "outage"])},
        {"name": "Order Execution", "status": random.choice(["operational", "degraded", "outage"])},
        {"name": "Strategy Engine", "status": random.choice(["operational", "degraded", "outage"])},
        {"name": "Portfolio Management", "status": random.choice(["operational", "degraded", "outage"])},
        {"name": "Risk Management", "status": random.choice(["operational", "degraded", "outage"])},
        {"name": "Market Data", "status": random.choice(["operational", "degraded", "outage"])},
    ]
    
    statuses = [comp["status"] for comp in components]
    overall_status = "operational"
    if "outage" in statuses:
        overall_status = "outage"
    elif "degraded" in statuses:
        overall_status = "degraded"
    
    return {
        "overall_status": overall_status,
        "components": components,
        "last_update": datetime.now().isoformat(),
        "uptime": random.randint(1, 30 * 24 * 60 * 60),  # Random uptime between 1 second and 30 days
        "resource_usage": {
            "cpu": round(random.uniform(10, 90), 1),
            "memory": round(random.uniform(20, 80), 1),
            "disk": round(random.uniform(30, 70), 1),
            "network": round(random.uniform(5, 60), 1),
        }
    }

# Routes
@app.get("/")
async def root():
    return {"message": "Trading Dashboard API is running"}

@app.get("/health")
async def health_check():
    """Check the health of the API."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/orders")
async def get_orders(
    limit: int = 20,
    status: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Get orders with optional filtering."""
    orders = generate_orders(limit)
    
    # Apply filters
    if status:
        orders = [order for order in orders if order["status"] == status]
    if symbol:
        orders = [order for order in orders if order["symbol"] == symbol]
    
    return orders

@app.get("/api/positions")
async def get_positions():
    """Get current positions."""
    return generate_positions()

@app.get("/api/context")
async def get_context():
    """Get market context data."""
    return generate_market_data()

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio summary."""
    return generate_portfolio_summary()

@app.get("/api/alerts")
async def get_alerts(
    limit: int = 10,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
):
    """Get alerts with optional filtering."""
    alerts = generate_alerts(limit)
    
    # Apply filters
    if severity:
        alerts = [alert for alert in alerts if alert["severity"] == severity]
    if status:
        alerts = [alert for alert in alerts if alert["status"] == status]
    if category:
        alerts = [alert for alert in alerts if alert["category"] == category]
    
    return alerts

@app.get("/api/system/status")
async def get_system_status():
    """Get system status."""
    return generate_system_status()

@app.post("/api/token")
async def login():
    """Get auth token."""
    return {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNjgzMDAwMDAwfQ.this_is_a_mock_token",
        "token_type": "bearer"
    }

if __name__ == "__main__":
    # Run the server when the script is executed directly
    print("Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
