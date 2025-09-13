#!/usr/bin/env python3
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleTradingAPI")

# Initialize FastAPI app
app = FastAPI(
    title="Simple Trading Bot API",
    description="Simplified API for trading dashboard",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint returning API info."""
    return {"message": "Simple Trading Bot API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/portfolio")
async def get_portfolio(account: str = "live"):
    """Get portfolio data for the dashboard."""
    try:
        # Generate sample portfolio data
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
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio data: {str(e)}")

@app.get("/api/strategies")
async def get_strategies(status: str = None):
    """Get trading strategies filtered by status."""
    try:
        strategies = [
            {
                "id": "ST-241",
                "name": "Mean Reversion ETF",
                "type": "Stocks",
                "status": "Active",
                "performance": 15.7,
                "allocation": 12.5,
                "symbols": ["SPY", "QQQ", "IWM", "DIA"],
                "lastSignal": "2025-05-03 09:34:21",
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
                "lastSignal": "2025-05-04 10:12:34",
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
                "lastSignal": "2025-05-04 11:45:00",
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
                "lastSignal": "2025-05-02 14:22:51",
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
                "lastSignal": "2025-05-04 09:30:15",
                "signalType": "Sell"
            }
        ]
        
        # Filter by status if specified
        if status:
            strategies = [s for s in strategies if s["status"].lower() == status.lower()]
            
        return strategies
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching strategies: {str(e)}")

@app.get("/api/model/performance")
async def get_model_performance(background_tasks: BackgroundTasks, refresh: bool = Query(False)):
    """Return model performance metrics."""
    try:
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
    except Exception as e:
        logger.error(f"Error fetching model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching model performance: {str(e)}")

@app.get("/api/trades/recent")
async def get_recent_trades(limit: int = Query(20)):
    """Return recent trades."""
    try:
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
    except Exception as e:
        logger.error(f"Error fetching recent trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching recent trades: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
