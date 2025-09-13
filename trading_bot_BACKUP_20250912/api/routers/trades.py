"""
Trades API routes.
Provides endpoints for trade data and execution.
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import random
from datetime import datetime, timedelta
import uuid

# Flag to indicate if real broker integration is available
REAL_BROKER_AVAILABLE = False

# Try to import broker adapters
try:
    # These imports would be implemented in a real system
    # from trading_bot.brokers.tradier import TradierBroker
    # from trading_bot.brokers.alpaca import AlpacaBroker
    # from trading_bot.core.trade_manager import TradeManager
    pass
except ImportError:
    pass

logger = logging.getLogger("TradingBotAPI.Trades")

router = APIRouter()
from trading_bot.core.journal.jsonl import read_recent as read_journal_recent

class TradeData(BaseModel):
    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: str
    account: str
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None
    order_id: Optional[str] = None
    fees: Optional[float] = None
    total_value: Optional[float] = None

class TradeMetadata(BaseModel):
    count: int
    buy_trades: int
    sell_trades: int
    total_volume: float
    total_value: float
    total_fees: float
    realized_pnl: float

class TradeResponse(BaseModel):
    data: List[TradeData]
    meta: TradeMetadata

def load_trade_data_from_file(account: str = "live") -> List[Dict[str, Any]]:
    """Load trade data from file to simulate persistent storage."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    file_path = os.path.join(cache_dir, f"trades_{account}.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trades from file: {str(e)}")
    
    # Generate initial data if file doesn't exist
    return generate_sample_trades(account, 200)  # Generate 200 sample trades

def save_trade_data_to_file(trades: List[Dict[str, Any]], account: str = "live"):
    """Save trade data to file to simulate persistent storage."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    file_path = os.path.join(cache_dir, f"trades_{account}.json")
    
    try:
        with open(file_path, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving trades to file: {str(e)}")

def generate_sample_trades(account: str, count: int = 50) -> List[Dict[str, Any]]:
    """Generate sample trades data for testing."""
    trades = []
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "TSLA", "AMD", "NVDA", "PYPL", 
               "SPY", "QQQ", "DIA", "IWM", "XLF", "XLE", "XLV", "XLK", "XLI", "XLU"]
    
    # Use strategy IDs that match our real strategies from memory
    strategies = [
        {"id": "gap-trading", "name": "Gap Trading Strategy"},
        {"id": "trend-following", "name": "Trend Following Strategy"},
        {"id": "breakout", "name": "Breakout Strategy"},
        {"id": "volume-surge", "name": "Volume Surge Strategy"},
        {"id": "rsi", "name": "RSI Strategy"},
        {"id": "macd", "name": "MACD Strategy"},
        {"id": "sector-rotation", "name": "Sector Rotation Strategy"},
        {"id": "earnings-announcement", "name": "Earnings Announcement Strategy"},
        {"id": "news-sentiment", "name": "News Sentiment Strategy"}
    ]
    
    # Generate trades over the last 90 days
    now = datetime.now()
    
    for i in range(count):
        symbol = random.choice(symbols)
        strategy = random.choice(strategies)
        side = random.choice(["buy", "sell"])
        
        price = round(random.uniform(50, 500), 2)
        quantity = random.randint(10, 100)
        
        # More recent trades for the first 20%
        if i < count * 0.2:
            days_ago = random.randint(0, 7)
        else:
            days_ago = random.randint(7, 90)
            
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        trade_time = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        timestamp = trade_time.isoformat()
        
        fees = round(random.uniform(0.5, 5.0), 2)
        total_value = price * quantity
        
        trade = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "account": account,
            "strategy_id": strategy["id"],
            "strategy_name": strategy["name"],
            "order_id": f"order-{uuid.uuid4().hex[:8]}",
            "fees": fees,
            "total_value": total_value
        }
        
        trades.append(trade)
    
    # Sort by timestamp, most recent first
    trades.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return trades

def get_trades_from_broker(account: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get trades from actual broker API.
    In a real implementation, this would use your broker APIs.
    """
    if not REAL_BROKER_AVAILABLE:
        return None
    
    try:
        # This would be implemented with real broker API calls
        # Example (not actual implementation):
        if account == "tradier":
            broker = TradierBroker()
            return broker.get_trades(limit=limit)
        elif account == "alpaca":
            broker = AlpacaBroker()
            return broker.get_trades(limit=limit)
        else:
            # Use trade manager for other account types
            trade_manager = TradeManager()
            return trade_manager.get_trades(account=account, limit=limit)
    except Exception as e:
        logger.error(f"Error getting trades from broker: {str(e)}")
        return None

@router.get("/trades", response_model=TradeResponse)
async def get_trades(
    account: str = "live", 
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy_id: Optional[str] = None
):
    """
    Get trades data for the dashboard.
    
    Parameters:
    - account: Account to fetch trades for (live, paper, tradier, alpaca)
    - limit: Maximum number of trades to return
    - start_date: Optional ISO format date to filter trades from
    - end_date: Optional ISO format date to filter trades until
    - strategy_id: Optional strategy ID to filter trades by
    """
    try:
        # Prefer journal if present, else broker, else sample file
        trades = read_journal_recent("trades", limit=1000)
        if not trades:
            trades = get_trades_from_broker(account, limit)
        if trades is None:
            trades = load_trade_data_from_file(account)
        
        # Apply filters
        filtered_trades = []
        
        for trade in trades:
            # Apply date filters if specified
            if start_date:
                trade_date = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00"))
                filter_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                if trade_date < filter_date:
                    continue
                    
            if end_date:
                trade_date = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00"))
                filter_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if trade_date > filter_date:
                    continue
            
            # Apply strategy filter if specified
            if strategy_id and trade.get("strategy_id") != strategy_id:
                continue
                
            filtered_trades.append(trade)
        
        # Apply limit after filtering
        result_trades = filtered_trades[:limit]
        
        # Calculate statistics for metadata
        buy_trades = sum(1 for t in filtered_trades if t["side"] == "buy")
        sell_trades = sum(1 for t in filtered_trades if t["side"] == "sell")
        total_volume = sum(t["quantity"] for t in filtered_trades)
        total_value = sum(t["price"] * t["quantity"] for t in filtered_trades)
        total_fees = sum(t.get("fees", 0) for t in filtered_trades)
        
        # Calculate realized P&L (simplified calculation)
        realized_pnl = sum(
            (t["price"] * t["quantity"]) if t["side"] == "sell" else (-t["price"] * t["quantity"])
            for t in filtered_trades
        )
        
        meta = {
            "count": len(filtered_trades),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "total_volume": total_volume,
            "total_value": total_value,
            "total_fees": total_fees,
            "realized_pnl": realized_pnl
        }
        
        return {
            "data": result_trades,
            "meta": meta
        }
        
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {str(e)}")
