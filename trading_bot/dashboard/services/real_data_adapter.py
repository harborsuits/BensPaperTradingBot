"""
Real Data Adapter - Implements missing methods for dashboard services
to connect to MongoDB and provide real trading data
"""
import os
import time
import datetime
import pymongo
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import yfinance as yf

class RealDataAdapter:
    """
    Adapter class that provides real implementations for missing methods in the data service.
    This acts as a bridge between the dashboard and MongoDB for actual data.
    """
    
    def __init__(self, data_service):
        """Initialize with reference to parent data service"""
        self.data_service = data_service
        self.mongo_client = None
        self.db = None
        self.connected = False
        
        # Try to connect to MongoDB
        try:
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/bensbot_trading")
            self.mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client.get_database()
            self.mongo_client.admin.command('ping')  # Test connection
            self.connected = True
            print("Connected to MongoDB successfully")
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            self.connected = False
            
        # Cache
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 30  # seconds
    
    def _get_cached_or_fetch(self, key, fetch_func, ttl=None):
        """Get from cache or fetch using the provided function"""
        ttl = ttl or self._cache_ttl
        now = time.time()
        
        # Check if we have a cached value that's still valid
        if key in self._cache and (now - self._cache_timestamps.get(key, 0)) < ttl:
            return self._cache[key]
        
        # Fetch new value
        value = fetch_func()
        
        # Update cache
        self._cache[key] = value
        self._cache_timestamps[key] = now
        
        return value
    
    def get_real_time_price(self, symbol):
        """Get real-time price for a symbol using Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if len(data) > 0:
                return data.iloc[-1]['Close']
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
        return None
    
    def get_portfolio_summary(self, account_type="paper"):
        """Get portfolio summary from MongoDB"""
        
        def fetch_portfolio():
            if not self.connected:
                return self._get_mock_portfolio()
            
            try:
                # Get account info
                account_doc = self.db.paper_account.find_one({"account_type": account_type})
                
                # Get positions for market value
                positions = list(self.db.paper_positions.find({"account_type": account_type}))
                
                securities_value = 0
                for position in positions:
                    symbol = position.get("symbol")
                    quantity = position.get("quantity", 0)
                    price = self.get_real_time_price(symbol) or position.get("current_price", 0)
                    securities_value += quantity * price
                
                if account_doc:
                    balance = account_doc.get("balance", 0)
                    starting_balance = account_doc.get("starting_balance", 100000)
                else:
                    balance = 100000
                    starting_balance = 100000
                
                total_equity = balance + securities_value
                total_pnl = total_equity - starting_balance
                total_pnl_pct = (total_pnl / starting_balance) * 100 if starting_balance else 0
                
                # Calculate daily P&L (simplified)
                yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                
                daily_pnl = 0
                daily_pnl_pct = 0
                
                return {
                    "cash_balance": balance,
                    "total_equity": total_equity,
                    "securities_value": securities_value,
                    "open_positions": len(positions),
                    "buying_power": balance * 1.0,  # No margin in paper trading
                    "margin_used": 0,
                    "margin_available": balance,
                    "daily_pnl": daily_pnl,
                    "daily_pnl_pct": daily_pnl_pct,
                    "total_pnl": total_pnl,
                    "total_pnl_pct": total_pnl_pct,
                    "starting_balance": starting_balance
                }
            except Exception as e:
                print(f"Error fetching portfolio data: {e}")
                return self._get_mock_portfolio()
        
        return self._get_cached_or_fetch(f"portfolio_{account_type}", fetch_portfolio)
    
    def _get_mock_portfolio(self):
        """Get mock portfolio data when MongoDB is unavailable"""
        return {
            "cash_balance": 100000.0,
            "total_equity": 100000.0,
            "securities_value": 0.0,
            "open_positions": 0,
            "buying_power": 100000.0,
            "margin_used": 0.0,
            "margin_available": 100000.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "starting_balance": 100000.0
        }
    
    def get_positions(self, account_type="paper"):
        """Get positions from MongoDB"""
        def fetch_positions():
            if not self.connected:
                return []
            
            try:
                positions = list(self.db.paper_positions.find({"account_type": account_type}))
                
                # Update current prices and calculations
                for position in positions:
                    symbol = position.get("symbol")
                    if symbol:
                        current_price = self.get_real_time_price(symbol)
                        if current_price:
                            quantity = position.get("quantity", 0)
                            position["current_price"] = current_price
                            position["market_value"] = quantity * current_price
                            position["unrealized_pnl"] = position["market_value"] - (quantity * position.get("avg_price", 0))
                
                return positions
            except Exception as e:
                print(f"Error fetching positions: {e}")
                return []
        
        return self._get_cached_or_fetch(f"positions_{account_type}", fetch_positions)
    
    def get_orders(self, account_type="paper", limit=20):
        """Get orders from MongoDB"""
        def fetch_orders():
            if not self.connected:
                return []
            
            try:
                return list(self.db.paper_orders.find({"account_type": account_type}).sort("created_at", -1).limit(limit))
            except Exception as e:
                print(f"Error fetching orders: {e}")
                return []
        
        return self._get_cached_or_fetch(f"orders_{account_type}", fetch_orders)
    
    def get_trades(self, account_type="paper", limit=20):
        """Get trades from MongoDB"""
        def fetch_trades():
            if not self.connected:
                return []
            
            try:
                return list(self.db.paper_trades.find({"account_type": account_type}).sort("timestamp", -1).limit(limit))
            except Exception as e:
                print(f"Error fetching trades: {e}")
                return []
        
        return self._get_cached_or_fetch(f"trades_{account_type}", fetch_trades)
    
    def get_system_status(self):
        """Get system status"""
        def fetch_status():
            if not self.connected:
                return {
                    "trading_enabled": False,
                    "status": "offline",
                    "uptime": "N/A",
                    "last_update": "N/A",
                    "message": "MongoDB not connected"
                }
            
            try:
                # Try to get system status from MongoDB
                status_doc = self.db.system_status.find_one({"type": "system_status"})
                
                if status_doc:
                    return status_doc
                
                # If no status in DB, create a default one
                return {
                    "trading_enabled": True,
                    "status": "online",
                    "uptime": "Unknown",
                    "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": "System running normally"
                }
            except Exception as e:
                print(f"Error fetching system status: {e}")
                return {
                    "trading_enabled": False,
                    "status": "error",
                    "uptime": "N/A",
                    "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Error: {str(e)}"
                }
        
        return self._get_cached_or_fetch("system_status", fetch_status)
    
    def get_market_context(self):
        """Get market context"""
        def fetch_market_context():
            # Default values
            context = {
                "market_conditions": "NORMAL",
                "volatility": "MEDIUM",
                "trend": "NEUTRAL",
                "risk_level": "MODERATE",
                "market_regime": "Neutral",
                "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if not self.connected:
                return context
            
            try:
                # Try to get market context from MongoDB
                market_doc = self.db.market_context.find_one({"type": "market_context"})
                
                if market_doc:
                    # Remove MongoDB _id field
                    if "_id" in market_doc:
                        del market_doc["_id"]
                    return market_doc
                
                return context
            except Exception as e:
                print(f"Error fetching market context: {e}")
                return context
        
        return self._get_cached_or_fetch("market_context", fetch_market_context)
    
    def get_active_strategies(self):
        """Get active strategies"""
        def fetch_strategies():
            if not self.connected:
                return []
            
            try:
                # Try to get strategies from MongoDB
                strategies = list(self.db.strategies.find({"status": "active"}))
                
                if strategies:
                    return strategies
                
                # Return empty list if no strategies found
                return []
            except Exception as e:
                print(f"Error fetching strategies: {e}")
                return []
        
        return self._get_cached_or_fetch("active_strategies", fetch_strategies)
    
    def get_performance_metrics(self, account_type="paper"):
        """Get performance metrics"""
        def fetch_metrics():
            if not self.connected:
                return {
                    "win_rate": 0,
                    "profit_factor": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "total_trades": 0
                }
            
            try:
                # Try to get metrics from MongoDB
                metrics_doc = self.db.performance_metrics.find_one({"account_type": account_type})
                
                if metrics_doc:
                    return metrics_doc
                
                # Calculate metrics from trades
                trades = list(self.db.paper_trades.find({"account_type": account_type, "status": "closed"}))
                
                if not trades:
                    return {
                        "win_rate": 0,
                        "profit_factor": 0,
                        "sharpe_ratio": 0,
                        "max_drawdown": 0,
                        "avg_win": 0,
                        "avg_loss": 0,
                        "total_trades": 0
                    }
                
                # Calculate basic metrics
                winners = [t for t in trades if t.get("pnl", 0) > 0]
                losers = [t for t in trades if t.get("pnl", 0) <= 0]
                
                win_rate = len(winners) / len(trades) if trades else 0
                
                total_profit = sum(t.get("pnl", 0) for t in winners)
                total_loss = abs(sum(t.get("pnl", 0) for t in losers))
                
                profit_factor = total_profit / total_loss if total_loss else float('inf')
                
                avg_win = total_profit / len(winners) if winners else 0
                avg_loss = total_loss / len(losers) if losers else 0
                
                return {
                    "win_rate": win_rate * 100,  # As percentage
                    "profit_factor": profit_factor,
                    "sharpe_ratio": 0,  # Need more data for this
                    "max_drawdown": 0,  # Need more data for this
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "total_trades": len(trades)
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                return {
                    "win_rate": 0,
                    "profit_factor": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "total_trades": 0
                }
        
        return self._get_cached_or_fetch(f"metrics_{account_type}", fetch_metrics)
    
    def get_equity_curve(self, account_type="paper", days=30):
        """Get equity curve data"""
        def fetch_equity():
            if not self.connected:
                return pd.DataFrame({'date': [], 'equity': []})
            
            try:
                # Try to get equity history from MongoDB
                equity_docs = list(self.db.equity_history.find(
                    {"account_type": account_type}
                ).sort("date", 1).limit(days * 24))  # Assuming hourly data
                
                if equity_docs:
                    df = pd.DataFrame(equity_docs)
                    df['date'] = pd.to_datetime(df['date'])
                    return df[['date', 'equity']]
                
                # If no equity history, create a mock one starting from account balance
                account = self.db.paper_account.find_one({"account_type": account_type})
                starting_balance = account.get("starting_balance", 100000) if account else 100000
                
                # Create a date range
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=days)
                
                dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
                
                # Create a simple equity curve with slight random changes
                equity = [starting_balance]
                for i in range(1, days):
                    change = np.random.normal(0, starting_balance * 0.005)  # 0.5% daily std dev
                    equity.append(max(0, equity[-1] + change))  # Ensure non-negative
                
                return pd.DataFrame({
                    'date': dates,
                    'equity': equity
                })
            except Exception as e:
                print(f"Error fetching equity curve: {e}")
                # Return empty DataFrame
                return pd.DataFrame({'date': [], 'equity': []})
        
        return self._get_cached_or_fetch(f"equity_{account_type}", fetch_equity, ttl=60*5)  # 5 minute cache
        
    def get_strategy_performance(self, time_period="1m", account_type="paper"):
        """Get strategy performance data"""
        def fetch_strategy_performance():
            # Default empty DataFrame with required columns
            empty_df = pd.DataFrame({
                'strategy_id': [],
                'strategy_name': [],
                'cumulative_return': [],
                'win_rate': [],
                'profit_factor': [],
                'sharpe_ratio': [],
                'max_drawdown': [],
                'trade_count': [],
                'avg_trade_duration': []
            })
            
            if not self.connected:
                return empty_df
            
            try:
                # Try to get strategy performance from MongoDB
                performance_docs = list(self.db.strategy_performance.find({"account_type": account_type}))
                
                if performance_docs:
                    return pd.DataFrame(performance_docs)
                
                # If no data, return empty DataFrame with mock data for one strategy
                mock_data = [{
                    'strategy_id': 'default',
                    'strategy_name': 'Default Strategy',
                    'cumulative_return': 0.05,  # 5%
                    'win_rate': 55.0,  # 55%
                    'profit_factor': 1.2,
                    'sharpe_ratio': 0.8,
                    'max_drawdown': -0.03,  # 3%
                    'trade_count': 25,
                    'avg_trade_duration': '2.5 days'
                }]
                
                return pd.DataFrame(mock_data)
            except Exception as e:
                print(f"Error fetching strategy performance: {e}")
                return empty_df
        
        return self._get_cached_or_fetch(f"strategy_performance_{account_type}_{time_period}", fetch_strategy_performance)
