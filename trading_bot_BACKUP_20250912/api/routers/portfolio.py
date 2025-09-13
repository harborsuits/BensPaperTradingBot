"""
Portfolio API routes.
Provides endpoints for portfolio data and performance metrics.
"""
import logging
from typing import Dict, List, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import random
from datetime import datetime, timedelta

logger = logging.getLogger("TradingBotAPI.Portfolio")

router = APIRouter()
from trading_bot.core.journal.jsonl import read_latest as read_journal_latest

class PortfolioData(BaseModel):
    totalValue: float
    dailyChange: float
    dailyChangePercent: float
    monthlyReturn: float
    weeklyChange: float
    allocation: List[Dict[str, Any]]
    holdings: List[Dict[str, Any]]

class PerformanceHistory(BaseModel):
    daily: List[float]
    weekly: List[float]
    monthly: List[float]
    yearly: List[float] 
    currentReturn: float

from trading_bot.services.portfolio_sources import get_cash, get_positions


@router.get("/portfolio")
async def get_portfolio(account: str = "live", mode: Optional[Literal["paper","live"]] = Query(None)):
    """Get portfolio data for the dashboard.

    Compatibility behavior:
    - If `mode` query param is provided, return a canonical, minimal shape expected by the new UI:
      { mode, equity, cash, positions: [], updated_at }
    - Otherwise, return the legacy PortfolioData shape used by older UIs.
    """
    try:
        # Try journal snapshot first
        snap = read_journal_latest("portfolio") or {}

        # New UI canonical response
        if mode is not None:
            # Derive from snapshot when possible
            positions = snap.get("positions") or snap.get("holdings") or []
            canonical_positions: List[Dict[str, Any]] = []
            for p in positions:
                canonical_positions.append({
                    "symbol": p.get("symbol") or p.get("ticker") or "",
                    "qty": float(p.get("quantity") or p.get("qty") or 0),
                    "avg_price": float(p.get("entryPrice") or p.get("avg_price") or 0.0),
                    "mkt_price": float(p.get("currentPrice") or p.get("mkt_price") or 0.0),
                    "unrealized_pl": float(p.get("unrealizedPnl") or p.get("unrealized_pl") or 0.0),
                    "pl_pct": float(p.get("unrealizedPnlPercent") or p.get("pl_pct") or 0.0),
                })

            equity = (
                snap.get("total_equity")
                or snap.get("equity")
                or snap.get("totalValue")
                or 0.0
            )
            cash = (
                snap.get("cash_balance")
                or snap.get("cash")
                or 0.0
            )

            return {
                "mode": (mode or account).lower(),
                "equity": float(equity),
                "cash": float(cash),
                "positions": canonical_positions,
                "updated_at": datetime.utcnow().isoformat(),
            }
        
        # Legacy UI shape
        # Fallback: create sample portfolio data
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
        
        # If called by new UI with explicit mode, serve canonical via adapters
        m: Literal["paper","live"] = "paper" if (mode or account).lower() == "paper" else "live"
        cash_val = float(get_cash(m))
        pos = get_positions(m)
        eq = cash_val + sum((p.get("qty") or 0.0) * (p.get("mkt_price") or 0.0) for p in pos)
        return {
            "mode": m,
            "equity": float(eq),
            "cash": float(cash_val),
            "positions": pos,
            "updated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@router.get("/portfolio/{mode}/history")
async def get_portfolio_history(mode: Literal["paper","live"], days: int = 30) -> List[Dict[str, Any]]:
    """Minimal stub for UI chart. Returns empty list for now."""
    return []

@router.get("/performance", response_model=PerformanceHistory)
async def get_performance_history(broker: str = "tradier"):
    """Get performance history for the dashboard."""
    try:
        # In a real implementation, this would pull data from your broker API
        # For now, we'll create realistic data that matches the frontend expectations
        
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
