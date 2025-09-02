#!/usr/bin/env python3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import random

app = FastAPI()

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }

@app.get("/api/portfolio")
def get_portfolio(account: str = "live"):
    """Get portfolio data matching the React component's expectations"""
    return {
        "totalValue": 852437.29,
        "dailyChange": 12483.57,
        "dailyChangePercent": 1.49,
        "monthlyReturn": 7.2,
        "allocation": [
            {"category": "Stocks", "value": 45, "color": "#4F8BFF"},
            {"category": "Options", "value": 15, "color": "#FF9800"},
            {"category": "Crypto", "value": 25, "color": "#4CAF50"},
            {"category": "Forex", "value": 10, "color": "#F44336"},
            {"category": "Cash", "value": 5, "color": "#9E9E9E"}
        ],
        "holdings": [
            {"symbol": "AAPL", "name": "Apple Inc.", "quantity": 200, "entryPrice": 155.25, "currentPrice": 173.75, "value": 34750, "unrealizedPnl": 3700, "unrealizedPnlPercent": 11.89},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "quantity": 100, "entryPrice": 287.70, "currentPrice": 312.79, "value": 31279, "unrealizedPnl": 2509, "unrealizedPnlPercent": 8.72},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "quantity": 150, "entryPrice": 108.42, "currentPrice": 124.67, "value": 18700.5, "unrealizedPnl": 2437.5, "unrealizedPnlPercent": 14.99},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "quantity": 120, "entryPrice": 96.30, "currentPrice": 109.82, "value": 13178.4, "unrealizedPnl": 1622.4, "unrealizedPnlPercent": 14.04},
            {"symbol": "TSLA", "name": "Tesla, Inc.", "quantity": 75, "entryPrice": 235.50, "currentPrice": 219.96, "value": 16497, "unrealizedPnl": -1165.5, "unrealizedPnlPercent": -6.59}
        ]
    }

@app.get("/api/strategies")
def get_strategies(status: str = None):
    """Get trading strategies data"""
    strategies = [
        {
            "id": "ST-241",
            "name": "Mean Reversion ETF",
            "type": "Stocks",
            "status": "Active",
            "performance": 15.7,
            "allocation": 12.5,
            "symbols": ["SPY", "QQQ", "IWM", "DIA"],
            "lastSignal": "2025-05-08 09:34:21",
            "signalType": "Buy"
        },
        {
            "id": "ST-156",
            "name": "Tech Momentum",
            "type": "Stocks",
            "status": "Active",
            "performance": 21.3,
            "allocation": 18.0,
            "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
            "lastSignal": "2025-05-08 10:12:34",
            "signalType": "Hold"
        },
        {
            "id": "ST-312",
            "name": "BTC Volatility Edge",
            "type": "Crypto",
            "status": "Active",
            "performance": 32.1,
            "allocation": 15.0,
            "symbols": ["BTC-USD", "ETH-USD"],
            "lastSignal": "2025-05-08 11:45:00",
            "signalType": "Buy"
        },
        {
            "id": "ST-127",
            "name": "Forex Range Trader",
            "type": "Forex",
            "status": "Active",
            "performance": 8.4,
            "allocation": 10.5,
            "symbols": ["EUR/USD", "GBP/USD", "USD/JPY"],
            "lastSignal": "2025-05-08 14:22:51",
            "signalType": "Sell"
        },
        {
            "id": "ST-198",
            "name": "Options Income",
            "type": "Options",
            "status": "Active",
            "performance": 12.1,
            "allocation": 15.0,
            "symbols": ["SPY", "QQQ", "AAPL", "TSLA"],
            "lastSignal": "2025-05-08 09:30:15",
            "signalType": "Sell"
        }
    ]
    
    # Filter by status if specified
    if status:
        strategies = [s for s in strategies if s["status"].lower() == status.lower()]
        
    return strategies

@app.get("/api/model/performance")
def get_model_performance():
    """Get model performance metrics"""
    return {
        "timestamp": "2025-05-08T15:00:00Z",
        "dataset": "holdout_test_set_v3",
        "metrics": {
            "accuracy": 0.87,
            "precision": 0.84, 
            "recall": 0.80,
            "f1_score": 0.82,
            "auc": 0.90
        }
    }

@app.get("/api/trades/recent")
def get_recent_trades(limit: int = 20):
    """Get recent trades data"""
    trades = [
        {
            "trade_id": "t1",
            "timestamp": "2025-05-08T15:30:12Z",
            "symbol": "AAPL", 
            "side": "buy",
            "quantity": 10,
            "price": 174.35,
            "pnl": 12.50
        },
        {
            "trade_id": "t2",
            "timestamp": "2025-05-08T15:45:02Z", 
            "symbol": "TSLA",
            "side": "sell",
            "quantity": 5,
            "price": 721.10,
            "pnl": -8.75
        },
        {
            "trade_id": "t3",
            "timestamp": "2025-05-08T16:10:45Z",
            "symbol": "MSFT",
            "side": "buy", 
            "quantity": 15,
            "price": 315.20,
            "pnl": 23.40
        }
    ]
    return trades[:limit]

@app.get("/api/market/data/{symbol}")
def get_market_data(symbol: str):
    """Get market data for a specific symbol"""
    price = round(random.uniform(100, 500), 2)
    change = round(random.uniform(-10, 10), 2)
    change_percent = round((change / (price - change) * 100) if price != change else 0, 2)
    
    return {
        "symbol": symbol,
        "name": f"{symbol} Inc.",
        "price": price,
        "change": change,
        "changePercent": change_percent,
        "volume": random.randint(100000, 10000000),
        "high": price + round(random.uniform(1, 5), 2),
        "low": price - round(random.uniform(1, 5), 2),
    }

@app.get("/api/context/features")
def get_context_features():
    """Get market context features"""
    return {
        "timestamp": datetime.now().isoformat(),
        "regime": "Bullish",
        "volatility": 1.2,
        "sentiment_score": 0.8,
        "momentum": 0.65,
        "correlation": 0.42,
        "features": {
            "rsi": 62.5,
            "macd": 1.87,
            "atr": 2.43,
            "volume_ratio": 1.24
        }
    }

if __name__ == "__main__":
    print("Starting minimal API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
