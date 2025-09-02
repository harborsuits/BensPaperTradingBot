#!/usr/bin/env python3
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
import uvicorn

# Create FastAPI app
app = FastAPI(title="BenBot Pipeline API", version="1.0")

# Add CORS middleware to allow React dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline data with realistic trading strategies
@app.get("/api/strategies")
async def get_strategies():
    """Return active trading strategies in the pipeline."""
    return [
        {
            "name": "ML-Enhanced Mean Reversion",
            "description": "Machine learning model that identifies overbought/oversold conditions",
            "status": "active",
            "allocation": 25,
            "performance": {"daily": 0.3, "weekly": 1.2, "monthly": 3.5, "yearly": 14.8},
            "activeTrades": 3,
            "signalStrength": 7.5,
            "lastUpdated": datetime.now().isoformat(),
            "symbols": ["SPY", "QQQ", "IWM"]
        },
        {
            "name": "Adaptive Trend Following",
            "description": "Dynamic trend identification with regime detection",
            "status": "active",
            "allocation": 20,
            "performance": {"daily": -0.2, "weekly": 0.8, "monthly": 2.7, "yearly": 11.5},
            "activeTrades": 2,
            "signalStrength": 6.2,
            "lastUpdated": datetime.now().isoformat(),
            "symbols": ["AAPL", "MSFT", "GOOGL"]
        },
        {
            "name": "Volatility Breakout Pro",
            "description": "Identifies price breakouts during high volatility periods",
            "status": "active",
            "allocation": 18,
            "performance": {"daily": 0.5, "weekly": 1.5, "monthly": 4.2, "yearly": 16.7},
            "activeTrades": 1,
            "signalStrength": 8.3,
            "lastUpdated": datetime.now().isoformat(),
            "symbols": ["NVDA", "AMD", "TSLA"]
        },
    ]

@app.get("/api/backtest/results")
async def get_backtest_results():
    """Return backtest results for strategies being evaluated."""
    return [
        {
            "id": "BT-1001",
            "strategy_name": "Enhanced Counter-Trend (Gold/Silver)",
            "status": "completed",
            "start_date": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
            "end_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "metrics": {
                "total_return": 12.4,
                "sharpe": 1.68,
                "max_drawdown": -8.2,
                "win_rate": 62.5
            },
            "trades": 48,
            "symbols": ["GLD", "SLV"],
        },
        {
            "id": "BT-1002",
            "strategy_name": "Sector Rotation Premium",
            "status": "running",
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "progress": 74,
            "preliminary_metrics": {
                "total_return": 6.8,
                "sharpe": 1.32,
                "max_drawdown": -5.1
            },
            "trades": 16,
            "symbols": ["XLF", "XLK", "XLV", "XLE"],
        }
    ]

@app.get("/api/pipeline/queue")
async def get_strategy_queue():
    """Return strategies in the development pipeline queue."""
    return [
        {
            "id": "STQ-101",
            "strategy_name": "ML-Enhanced Mean Reversion",
            "status": "in_development",
            "priority": "high",
            "est_start_time": "03:30 PM",
            "complexity": "high",
            "symbols": ["SPY", "QQQ", "IWM"],
            "description": "Machine learning model that identifies overbought/oversold conditions"
        },
        {
            "id": "STQ-102",
            "strategy_name": "Fixed Income Tactical",
            "status": "pending_approval",
            "priority": "medium",
            "est_start_time": "05:15 PM",
            "complexity": "medium",
            "symbols": ["TLT", "IEF", "HYG"],
            "description": "Tactical rotation model for fixed income assets based on yield curve dynamics"
        },
        {
            "id": "STQ-103",
            "strategy_name": "Global Macro Rotation",
            "status": "queued",
            "priority": "low",
            "est_start_time": "Tomorrow",
            "complexity": "high",
            "symbols": ["SPY", "EFA", "EEM", "AGG"],
            "description": "Systematic global macro strategy with allocation shifts based on economic regime"
        },
    ]

@app.get("/api/portfolio")
async def get_portfolio():
    """Return current portfolio allocation and performance metrics."""
    return {
        "total_value": 124573.82,
        "day_change_value": 1532.45,
        "day_change_pct": 1.24,
        "total_return_value": 9872.34,
        "allocation": {
            "Equities": 72500.23,
            "Fixed Income": 32845.68,
            "Cash": 19227.91
        },
        "holdings": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 28,
                "entryPrice": 165.42,
                "currentPrice": 178.92,
                "value": 5009.76,
                "unrealizedPnl": 378.00,
                "unrealizedPnlPercent": 8.16
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corp.",
                "quantity": 15,
                "entryPrice": 312.78,
                "currentPrice": 325.64,
                "value": 4884.60,
                "unrealizedPnl": 192.90,
                "unrealizedPnlPercent": 4.11
            },
            {
                "symbol": "NVDA",
                "name": "NVIDIA Corp.",
                "quantity": 12,
                "entryPrice": 425.15,
                "currentPrice": 462.37,
                "value": 5548.44,
                "unrealizedPnl": 446.64,
                "unrealizedPnlPercent": 8.75
            }
        ]
    }

@app.get("/api/trades")
async def get_trades():
    """Return recent trades executed by the trading pipeline."""
    return [
        {
            "id": "T10045",
            "symbol": "AAPL",
            "price": 178.92,
            "quantity": 5,
            "status": "filled",
            "side": "buy",
            "execution_time": (datetime.now() - timedelta(hours=3)).isoformat(),
            "strategy": "ML-Enhanced Mean Reversion",
            "type": "market"
        },
        {
            "id": "T10044",
            "symbol": "SPY",
            "price": 501.35,
            "quantity": 8,
            "status": "filled",
            "side": "sell",
            "execution_time": (datetime.now() - timedelta(hours=5)).isoformat(),
            "strategy": "Adaptive Trend Following",
            "type": "limit"
        },
        {
            "id": "T10043",
            "symbol": "NVDA",
            "price": 462.37,
            "quantity": 3,
            "status": "filled",
            "side": "buy",
            "execution_time": (datetime.now() - timedelta(hours=8)).isoformat(),
            "strategy": "Volatility Breakout Pro",
            "type": "market"
        }
    ]

# Run the server when script is executed directly
if __name__ == "__main__":
    print("Starting BenBot Pipeline API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
