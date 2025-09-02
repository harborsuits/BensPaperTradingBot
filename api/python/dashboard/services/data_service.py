"""
Data Service for Streamlit Dashboard

This service handles data retrieval from the trading bot's backend systems,
providing a clean interface for dashboard components to access data.
"""
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataService:
    """
    Service for retrieving and caching data from the trading bot backend.
    Provides methods for accessing strategy performance, trade logs, alerts, etc.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the data service.
        
        Args:
            api_base_url: Base URL for the trading bot API
        """
        self.api_base_url = api_base_url
        self.cache = {}
        self.cache_expiry = {}
        
        # Default cache duration in seconds
        self.default_cache_duration = 5
        
        # Connection tracking
        self.is_connected = False
        self.using_real_data = False
        self.connection_attempts = 0
        self.last_connection_time = None
        
        # Connect to backend
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check connection to the backend API."""
        try:
            self.connection_attempts += 1
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to trading bot API")
                self.is_connected = True
                self.using_real_data = True
                self.last_connection_time = datetime.now()
                return True
            else:
                logger.warning(f"API connection issue: Status code {response.status_code}")
                self.is_connected = False
                self.using_real_data = False
                return False
        except Exception as e:
            logger.error(f"Failed to connect to API: {str(e)}")
            # Mark as disconnected but continue with mock data
            self.is_connected = False
            self.using_real_data = False
            return False
    
    def _get_from_api(self, endpoint: str) -> Dict[str, Any]:
        """
        Get data from API with error handling.
        
        Args:
            endpoint: API endpoint to request
            
        Returns:
            Dict: Response data or empty dict on error
        """
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API request to {endpoint} failed: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching from {endpoint}: {str(e)}")
            return {}
    
    def _get_cached(self, key: str, fetch_func, expiry_seconds: int = None) -> Any:
        """
        Get data with caching.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data if not in cache
            expiry_seconds: Cache expiry in seconds
            
        Returns:
            Any: Cached or freshly fetched data
        """
        now = time.time()
        expiry = expiry_seconds or self.default_cache_duration
        
        # If cached and not expired, return from cache
        if key in self.cache and self.cache_expiry.get(key, 0) > now:
            return self.cache[key]
        
        # Fetch fresh data
        data = fetch_func()
        
        # Update cache
        self.cache[key] = data
        self.cache_expiry[key] = now + expiry
        
        return data
    
    # System status methods
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return self._get_cached(
            "system_status",
            lambda: self._get_from_api("/api/system/status") or {
                "status": "running",
                "trading_enabled": True,
                "uptime": "1d 4h 35m",
                "version": "1.0.0",
                "last_update": datetime.now().isoformat()
            }
        )
    
    # Performance data methods
    def get_strategy_performance(self, time_period: str = "All Time", account_type: str = None) -> pd.DataFrame:
        """
        Get performance metrics for all strategies.
        
        Args:
            time_period: Time period for data
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Strategy performance data
        """
        def fetch_data():
            endpoint = f"/api/strategies/performance?period={time_period}"
            if account_type:
                endpoint += f"&account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "strategies" in response:
                return pd.DataFrame(response["strategies"])
            
            # Demo data if API unavailable
            return pd.DataFrame([
                {
                    "id": "momentum_01",
                    "name": "Momentum Strategy",
                    "phase": "LIVE",
                    "status": "ACTIVE",
                    "pnl": 3245.75,
                    "pnl_pct": 12.45,
                    "sharpe_ratio": 1.82,
                    "win_rate": 0.68,
                    "max_drawdown_pct": -8.32,
                    "trade_count": 45,
                    "profit_factor": 2.1,
                    "avg_win": 320.5,
                    "avg_loss": -152.8,
                    "expectancy": 125.6
                },
                {
                    "id": "mean_reversion_01",
                    "name": "Mean Reversion",
                    "phase": "PAPER_TRADE",
                    "status": "ACTIVE",
                    "pnl": 1582.30,
                    "pnl_pct": 7.91,
                    "sharpe_ratio": 1.35,
                    "win_rate": 0.55,
                    "max_drawdown_pct": -5.77,
                    "trade_count": 63,
                    "profit_factor": 1.7,
                    "avg_win": 210.3,
                    "avg_loss": -125.4,
                    "expectancy": 65.2
                },
                {
                    "id": "pattern_breakout_01",
                    "name": "Pattern Breakout",
                    "phase": "PAPER_TRADE",
                    "status": "PAUSED",
                    "pnl": -450.20,
                    "pnl_pct": -2.25,
                    "sharpe_ratio": -0.38,
                    "win_rate": 0.32,
                    "max_drawdown_pct": -12.42,
                    "trade_count": 28,
                    "profit_factor": 0.78,
                    "avg_win": 315.7,
                    "avg_loss": -205.6,
                    "expectancy": -43.8
                }
            ])
        
        cache_key = f"strategy_performance_{time_period}_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data)
    
    def get_strategy_equity_curves(self, strategy_ids: List[str] = None, account_type: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get equity curves for strategies.
        
        Args:
            strategy_ids: List of strategy IDs or None for all
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            Dict: Strategy ID to equity curve DataFrame mapping
        """
        def fetch_data():
            endpoint = "/api/strategies/equity-curves"
            if strategy_ids:
                ids_param = ",".join(strategy_ids)
                endpoint += f"?ids={ids_param}"
            if account_type:
                endpoint += f"&account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "equity_curves" in response:
                result = {}
                for strategy_id, curve_data in response["equity_curves"].items():
                    result[strategy_id] = pd.DataFrame(curve_data)
                return result
            
            # Demo data if API unavailable
            result = {}
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
            
            # Generate sample equity curves
            for strategy_id in (strategy_ids or ["momentum_01", "mean_reversion_01", "pattern_breakout_01"]):
                if strategy_id == "momentum_01":
                    # Upward trending equity curve
                    base = 100000
                    daily_returns = np.random.normal(0.001, 0.01, len(dates))
                    daily_returns[0] = 0
                    cumulative_returns = np.cumprod(1 + daily_returns)
                    equity = base * cumulative_returns
                elif strategy_id == "mean_reversion_01":
                    # Choppy but positive equity curve
                    base = 100000
                    daily_returns = np.random.normal(0.0005, 0.008, len(dates))
                    daily_returns[0] = 0
                    cumulative_returns = np.cumprod(1 + daily_returns)
                    equity = base * cumulative_returns
                else:
                    # Negative equity curve
                    base = 100000
                    daily_returns = np.random.normal(-0.0003, 0.009, len(dates))
                    daily_returns[0] = 0
                    cumulative_returns = np.cumprod(1 + daily_returns)
                    equity = base * cumulative_returns
                    
                result[strategy_id] = pd.DataFrame({
                    'date': dates,
                    'equity': equity
                })
                
            return result
        
        cache_key = f"equity_curves_{','.join(strategy_ids) if strategy_ids else 'all'}_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data)
    
    # Trade log methods
    def get_trade_log(self, max_trades: int = 100, strategy_filter: str = None, account_type: str = None) -> pd.DataFrame:
        """
        Get recent trades.
        
        Args:
            max_trades: Maximum number of trades to return
            strategy_filter: Optional strategy ID to filter by
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Trade log data
        """
        def fetch_data():
            endpoint = f"/api/trades/recent?limit={max_trades}"
            if strategy_filter:
                endpoint += f"&strategy_id={strategy_filter}"
            if account_type:
                endpoint += f"&account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "trades" in response:
                return pd.DataFrame(response["trades"])
            
            # Demo data if API unavailable
            trades = []
            
            # Sample strategies and symbols
            strategies = ["Momentum Strategy", "Mean Reversion", "Pattern Breakout"]
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC/USD", "ETH/USD"]
            
            # Generate sample trades
            for i in range(max_trades):
                strategy = np.random.choice(strategies)
                
                # Skip if filter is set and doesn't match
                if strategy_filter and strategy != strategy_filter:
                    continue
                    
                symbol = np.random.choice(symbols)
                is_buy = np.random.choice([True, False])
                
                # Generate timestamps in descending order (newest first)
                timestamp = datetime.now() - timedelta(
                    minutes=np.random.randint(1, 60 * 24 * 3)
                )
                
                price = round(np.random.uniform(50, 1000) if symbol not in ["BTC/USD", "ETH/USD"] 
                              else np.random.uniform(10000, 50000) if symbol == "BTC/USD"
                              else np.random.uniform(1000, 5000), 2)
                
                quantity = round(np.random.uniform(1, 10) if symbol not in ["BTC/USD", "ETH/USD"]
                                else np.random.uniform(0.1, 1.0) if symbol == "BTC/USD"
                                else np.random.uniform(0.5, 5.0), 
                                4 if symbol in ["BTC/USD", "ETH/USD"] else 0)
                
                # For exit trades, add P&L
                pnl = None
                if np.random.choice([True, False]) and not is_buy:
                    pnl = round(np.random.uniform(-500, 1000), 2)
                
                trades.append({
                    "timestamp": timestamp.isoformat(),
                    "strategy": strategy,
                    "symbol": symbol,
                    "action": "BUY" if is_buy else "SELL",
                    "quantity": quantity,
                    "price": price,
                    "status": np.random.choice(["Filled", "Partially Filled", "Cancelled"], p=[0.8, 0.15, 0.05]),
                    "pnl": pnl
                })
            
            # Sort by timestamp (newest first)
            trades.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return pd.DataFrame(trades)
        
        cache_key = f"trade_log_{max_trades}_{strategy_filter or 'all'}_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data, expiry_seconds=3)  # Short cache for trades
    
    # Strategy monitor methods
    def get_active_strategies(self, account_type: str = None) -> pd.DataFrame:
        """
        Get active strategies data.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Active strategies data
        """
        def fetch_data():
            endpoint = "/api/strategies/list"
            if account_type:
                endpoint += f"?account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "strategies" in response:
                return pd.DataFrame(response["strategies"])
            
            # Demo data if API unavailable
            return pd.DataFrame([
                {
                    "id": "momentum_01",
                    "name": "Momentum Strategy",
                    "phase": "LIVE",
                    "status": "ACTIVE",
                    "asset_class": "Equities",
                    "daily_pnl": 215.50,
                    "daily_pnl_pct": 0.82,
                    "positions_count": 3,
                    "primary_position": "LONG AAPL 100 @ $175.25"
                },
                {
                    "id": "mean_reversion_01",
                    "name": "Mean Reversion",
                    "phase": "PAPER_TRADE",
                    "status": "ACTIVE",
                    "asset_class": "Equities",
                    "daily_pnl": 87.25,
                    "daily_pnl_pct": 0.35,
                    "positions_count": 2,
                    "primary_position": "SHORT MSFT 50 @ $310.75"
                },
                {
                    "id": "pattern_breakout_01",
                    "name": "Pattern Breakout",
                    "phase": "PAPER_TRADE",
                    "status": "PAUSED",
                    "asset_class": "Crypto",
                    "daily_pnl": -120.30,
                    "daily_pnl_pct": -0.48,
                    "positions_count": 0,
                    "primary_position": "None"
                },
                {
                    "id": "grid_trading_01",
                    "name": "Grid Trading",
                    "phase": "PAPER_TRADE",
                    "status": "ACTIVE",
                    "asset_class": "Forex",
                    "daily_pnl": 32.45,
                    "daily_pnl_pct": 0.13,
                    "positions_count": 5,
                    "primary_position": "LONG EUR/USD 0.1 @ 1.0825"
                }
            ])
        
        cache_key = f"active_strategies_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data)
    
    # Alerts methods
    def get_alerts(self, max_alerts: int = 20, account_type: str = None) -> pd.DataFrame:
        """
        Get recent system alerts.
        
        Args:
            max_alerts: Maximum number of alerts to return
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Alerts data
        """
        def fetch_data():
            endpoint = f"/api/system/alerts?limit={max_alerts}"
            if account_type:
                endpoint += f"&account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "alerts" in response:
                return pd.DataFrame(response["alerts"])
            
            # Demo data if API unavailable
            alerts = []
            
            # Alert types and templates
            alert_types = {
                "ERROR": [
                    "Broker {broker} disconnected. Attempting reconnection...",
                    "Order failed - insufficient funds in {broker}.",
                    "API rate limit exceeded for {broker}.",
                    "Failed to fetch market data for {symbol}."
                ],
                "WARNING": [
                    "Strategy {strategy} exceeded max drawdown and was paused.",
                    "High correlation detected between {strategy1} and {strategy2}.",
                    "Daily loss limit approached for {strategy}.",
                    "Market volatility higher than normal for {symbol}."
                ],
                "INFO": [
                    "Strategy {strategy} automatically paused due to market closure.",
                    "New strategy {strategy} registered in paper mode.",
                    "Broker {broker} reconnected successfully.",
                    "Position in {symbol} closed automatically due to stop loss."
                ],
                "SUCCESS": [
                    "Strategy {strategy} promoted from paper to live trading.",
                    "System update completed successfully.",
                    "Database backup completed successfully.",
                    "Broker {broker} authentication renewed."
                ]
            }
            
            brokers = ["Alpaca", "Interactive Brokers", "Binance"]
            strategies = ["Momentum Strategy", "Mean Reversion", "Pattern Breakout", "Grid Trading"]
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC/USD", "ETH/USD"]
            
            # Generate sample alerts
            for i in range(max_alerts):
                alert_type = np.random.choice(list(alert_types.keys()), p=[0.2, 0.3, 0.4, 0.1])
                template = np.random.choice(alert_types[alert_type])
                
                # Fill in template placeholders
                message = template.format(
                    broker=np.random.choice(brokers),
                    strategy=np.random.choice(strategies),
                    strategy1=strategies[0],
                    strategy2=strategies[1],
                    symbol=np.random.choice(symbols)
                )
                
                # Generate timestamps in descending order (newest first)
                timestamp = datetime.now() - timedelta(
                    minutes=np.random.randint(1, 60 * 24)
                )
                
                alerts.append({
                    "timestamp": timestamp.isoformat(),
                    "type": alert_type,
                    "message": message
                })
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return pd.DataFrame(alerts)
        
        cache_key = f"alerts_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data, expiry_seconds=5)
    
    # Portfolio methods
    def get_portfolio_summary(self, account_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get portfolio summary data, optionally filtered by account type.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            Dict: Portfolio summary
        """
        def fetch_data():
            endpoint = "/api/portfolio/summary"
            if account_type:
                endpoint += f"?account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response:
                return response
                
            # Demo data if API unavailable - different for Live vs Paper
            if account_type == "Paper":
                return {
                    "total_equity": 75000.00,
                    "daily_pnl": 850.25,
                    "daily_pnl_pct": 1.13,
                    "total_pnl": 15000.00,
                    "total_pnl_pct": 21.50,
                    "win_rate": 72.5,
                    "win_rate_change": 3.1,
                    "trades_today": 23,
                    "trades_change": 7
                }
            else:  # Live or None
                return {
                    "total_equity": 125000.00,
                    "daily_pnl": 1230.50,
                    "daily_pnl_pct": 0.98,
                    "total_pnl": 25000.00,
                    "total_pnl_pct": 25.00,
                    "win_rate": 68.5,
                    "win_rate_change": 2.3,
                    "trades_today": 15,
                    "trades_change": 3
                }
        
        # Use account type in cache key to maintain separate caches
        cache_key = f"portfolio_summary_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data)

    def get_broker_balances(self, account_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get broker account balances, optionally filtered by account type.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Broker account data
        """
        def fetch_data():
            endpoint = "/api/brokers/accounts"
            if account_type:
                endpoint += f"?account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "accounts" in response:
                return pd.DataFrame(response["accounts"])
            
            # Demo data if API unavailable
            accounts = [
                {"broker": "Binance", "account_id": "primary", "account_type": "Live", "equity": 45250.67, "cash": 12500.33, "margin_used": 32750.34, "margin_available": 12500.33},
                {"broker": "OANDA", "account_id": "fx-001", "account_type": "Live", "equity": 35250.00, "cash": 15250.00, "margin_used": 20000.00, "margin_available": 15250.00},
                {"broker": "Alpaca", "account_id": "stocks-demo", "account_type": "Paper", "equity": 25000.00, "cash": 10000.00, "margin_used": 15000.00, "margin_available": 10000.00},
                {"broker": "Interactive Brokers", "account_id": "demo-002", "account_type": "Paper", "equity": 50000.00, "cash": 25000.00, "margin_used": 25000.00, "margin_available": 25000.00}
            ]
            
            # Filter by account type if specified
            if account_type:
                accounts = [acc for acc in accounts if acc["account_type"] == account_type]
                
            return pd.DataFrame(accounts)
        
        # Use account type in cache key to maintain separate caches
        cache_key = f"broker_balances_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data)
    
    def get_positions(self, account_type: str = None) -> pd.DataFrame:
        """
        Get all open positions.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Positions data
        """
        def fetch_data():
            endpoint = "/api/portfolio/positions"
            if account_type:
                endpoint += f"?account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response and "positions" in response:
                return pd.DataFrame(response["positions"])
            
            # Demo data if API unavailable
            return pd.DataFrame([
                {
                    "broker": "Alpaca Securities",
                    "strategy": "Momentum Strategy",
                    "symbol": "AAPL",
                    "quantity": 100,
                    "side": "LONG",
                    "entry_price": 175.25,
                    "current_price": 178.50,
                    "market_value": 17850.00,
                    "unrealized_pnl": 325.00,
                    "unrealized_pnl_pct": 1.85
                },
                {
                    "broker": "Alpaca Securities",
                    "strategy": "Momentum Strategy",
                    "symbol": "MSFT",
                    "quantity": 50,
                    "side": "LONG",
                    "entry_price": 305.50,
                    "current_price": 310.75,
                    "market_value": 15537.50,
                    "unrealized_pnl": 262.50,
                    "unrealized_pnl_pct": 1.72
                },
                {
                    "broker": "Alpaca Securities",
                    "strategy": "Mean Reversion",
                    "symbol": "TSLA",
                    "quantity": 20,
                    "side": "SHORT",
                    "entry_price": 225.75,
                    "current_price": 220.50,
                    "market_value": 4410.00,
                    "unrealized_pnl": 105.00,
                    "unrealized_pnl_pct": 2.33
                },
                {
                    "broker": "Binance",
                    "strategy": "Grid Trading",
                    "symbol": "BTC/USD",
                    "quantity": 0.5,
                    "side": "LONG",
                    "entry_price": 42500.00,
                    "current_price": 43200.00,
                    "market_value": 21600.00,
                    "unrealized_pnl": 350.00,
                    "unrealized_pnl_pct": 1.64
                }
            ])
        
        cache_key = f"positions_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data)
    
    # Market context methods
    def get_market_context(self, account_type: str = None) -> Dict[str, Any]:
        """
        Get market context data.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            Dict: Market context data
        """
        def fetch_data():
            endpoint = "/api/market/context"
            if account_type:
                endpoint += f"?account_type={account_type.lower()}"
                
            response = self._get_from_api(endpoint)
            if response:
                return response
            
            # Demo data if API unavailable
            return {
                "market_regime": np.random.choice(["Bullish", "Bearish", "Neutral"], p=[0.4, 0.2, 0.4]),
                "vix": round(np.random.uniform(15, 25), 2),
                "fear_greed_index": round(np.random.uniform(30, 70)),
                "major_indices": [
                    {"name": "S&P 500", "price": round(np.random.uniform(4000, 4200), 2), "change_pct": round(np.random.uniform(-1, 1), 2)},
                    {"name": "NASDAQ", "price": round(np.random.uniform(13000, 14000), 2), "change_pct": round(np.random.uniform(-1, 1), 2)},
                    {"name": "Dow Jones", "price": round(np.random.uniform(33000, 34000), 2), "change_pct": round(np.random.uniform(-0.8, 0.8), 2)}
                ],
                "crypto": [
                    {"name": "Bitcoin", "price": round(np.random.uniform(40000, 45000), 2), "change_pct": round(np.random.uniform(-2, 2), 2)},
                    {"name": "Ethereum", "price": round(np.random.uniform(2800, 3200), 2), "change_pct": round(np.random.uniform(-2, 2), 2)}
                ],
                "forex": [
                    {"name": "EUR/USD", "price": round(np.random.uniform(1.07, 1.09), 4), "change_pct": round(np.random.uniform(-0.5, 0.5), 2)},
                    {"name": "GBP/USD", "price": round(np.random.uniform(1.25, 1.28), 4), "change_pct": round(np.random.uniform(-0.5, 0.5), 2)}
                ]
            }
        
        cache_key = f"market_context_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data, expiry_seconds=60)  # Longer cache for market data
    
        # Broker intelligence methods
    def get_broker_intelligence_data(self, account_type: str = None) -> Dict[str, Any]:
        """
        Get broker intelligence data including health status, recommendations,
        and circuit breakers.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            Dict: Broker intelligence data
        """
        def fetch_data():
            # In a real implementation, this would fetch data from the API
            response = self._get_from_api(f"/api/broker/intelligence?account_type={account_type if account_type else 'all'}")
            if response and "health_status" in response:
                return response
            
            # Demo data if API unavailable
            # Status options: NORMAL, CAUTION, CRITICAL
            status = np.random.choice(["NORMAL", "CAUTION", "CRITICAL"], p=[0.7, 0.2, 0.1])
            
            # Sample broker data
            brokers = [
                {"id": "tradier", "type": "equities"},
                {"id": "alpaca", "type": "equities"},
                {"id": "interactive_brokers", "type": "multi-asset"},
                {"id": "oanda", "type": "forex"}
            ]
            
            # Generate broker health data
            broker_health = {}
            broker_scores = {}
            circuit_breakers = {}
            
            for broker in brokers:
                broker_id = broker["id"]
                base_score = np.random.uniform(60, 95)
                
                # Circuit breaker (10% chance of being active)
                circuit_breaker_active = np.random.random() < 0.1
                
                if circuit_breaker_active:
                    now = time.time()
                    reset_after = np.random.randint(60, 600)  # 1-10 minutes
                    reason = np.random.choice([
                        "High error rate: 35.2%",
                        "Low availability: 87.5%",
                        "Connection timeout"
                    ])
                    
                    circuit_breakers[broker_id] = {
                        "active": True,
                        "reason": reason,
                        "tripped_at": now - np.random.randint(0, 300),  # 0-5 minutes ago
                        "reset_after": reset_after,
                        "reset_time": now + reset_after
                    }
                
                # Create broker performance scores
                factor_scores = {
                    "latency": np.random.uniform(50, 95),
                    "reliability": np.random.uniform(70, 99),
                    "execution_quality": np.random.uniform(60, 90),
                    "cost": np.random.uniform(50, 95),
                    "circuit_breaker": 0 if circuit_breaker_active else 100
                }
                
                # Final score calculation
                if circuit_breaker_active:
                    base_score *= 0.3  # Significant penalty for circuit breaker
                
                # Store broker health data
                broker_health[broker_id] = {
                    "performance_score": base_score,
                    "factor_scores": factor_scores,
                    "circuit_breaker_active": circuit_breaker_active,
                    "metrics": {
                        "latency": {"mean_ms": np.random.uniform(50, 500)},
                        "reliability": {"availability": np.random.uniform(90, 99.9), "errors": np.random.randint(0, 10)},
                        "execution_quality": {"avg_slippage_pct": np.random.uniform(0, 0.5)},
                        "costs": {"avg_commission": np.random.uniform(0.5, 5.0)}
                    }
                }
                
                broker_scores[broker_id] = base_score
            
            # Generate recommendations for common asset/operation pairs
            recommendations = {}
            
            for asset_class in ["equities", "forex", "futures", "options", "crypto"]:
                for operation in ["order", "quote", "data"]:
                    # Sort brokers by score
                    sorted_brokers = sorted(
                        [(k, v) for k, v in broker_scores.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    primary_broker = sorted_brokers[0][0] if sorted_brokers else None
                    backup_brokers = [b[0] for b in sorted_brokers[1:]] if len(sorted_brokers) > 1 else []
                    
                    # Determine blacklisted brokers (those with active circuit breakers)
                    blacklisted = [broker_id for broker_id in broker_scores.keys() 
                                 if broker_id in circuit_breakers and circuit_breakers[broker_id]["active"]]
                    
                    # 15% chance of recommending failover
                    failover_recommended = np.random.random() < 0.15
                    
                    recommendations[f"{asset_class}_{operation}"] = {
                        "asset_class": asset_class,
                        "operation_type": operation,
                        "primary_broker_id": primary_broker,
                        "backup_broker_ids": backup_brokers,
                        "blacklisted_broker_ids": blacklisted,
                        "priority_scores": broker_scores,
                        "is_failover_recommended": failover_recommended,
                        "advisory_notes": [
                            f"Primary recommendation based on overall performance score: {broker_scores.get(primary_broker, 0):.1f}" if primary_broker else "No suitable broker found"
                        ] + ([f"Recommend failover from current broker to {primary_broker}"] if failover_recommended else [])
                    }
            
            return {
                "health_status": status,
                "health_report": {
                    "overall_status": status,
                    "timestamp": time.time(),
                    "brokers": broker_health
                },
                "circuit_breakers": circuit_breakers,
                "recommendations": recommendations
            }
        
        cache_key = f"broker_intelligence_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data, expiry_seconds=5)  # Short cache - intelligence data changes frequently
    
    def get_registered_brokers(self, account_type: str = None) -> List[str]:
        """
        Get list of registered broker IDs.
        
        Args:
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            List[str]: List of broker IDs
        """
        def fetch_data():
            # In a real implementation, this would fetch data from the API
            response = self._get_from_api(f"/api/broker/registered?account_type={account_type if account_type else 'all'}")
            if response and "brokers" in response:
                return response["brokers"]
            
            # Demo data if API unavailable
            return ["tradier", "alpaca", "interactive_brokers", "oanda"]
        
        cache_key = f"registered_brokers_{account_type if account_type else 'all'}"
        return self._get_cached(cache_key, fetch_data, expiry_seconds=30)
    
    # Webhook methods
    def get_webhook_signals(self, max_signals: int = 50, account_type: str = None) -> pd.DataFrame:
        """
        Get recent webhook signals.
        
        Args:
            max_signals: Maximum number of signals to return
            account_type: Optional filter for 'Live' or 'Paper' accounts
            
        Returns:
            DataFrame: Webhook signals data
        """
        def fetch_data():
            response = self._get_from_api(f"/api/webhooks/recent?limit={max_signals}")
            if response and "signals" in response:
                return pd.DataFrame(response["signals"])
            
            # Demo data if API unavailable
            signals = []
            
            # Sample signal templates
            signal_templates = [
                "Supertrend flipped to {direction}",
                "MACD {direction} crossover",
                "RSI {condition}",
                "Price broke {direction} key level",
                "Bollinger band {condition}"
            ]
            
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC/USD", "ETH/USD"]
            
            # Direction and condition options
            directions = ["buy", "sell"]
            conditions = ["oversold", "overbought", "neutral", "bullish", "bearish"]
            
            # Generate sample signals
            for i in range(max_signals):
                template = np.random.choice(signal_templates)
                
                # Fill in template placeholders
                message = template.format(
                    direction=np.random.choice(directions),
                    condition=np.random.choice(conditions)
                )
                
                # Generate timestamps in descending order (newest first)
                timestamp = datetime.now() - timedelta(
                    minutes=np.random.randint(1, 60 * 24)
                )
                
                signals.append({
                    "timestamp": timestamp.isoformat(),
                    "source": "TradingView Alert",
                    "symbol": np.random.choice(symbols),
                    "message": message,
                    "status": np.random.choice(["Processed", "Ignored"]),
                    "action_taken": "BUY" if "buy" in message.lower() else "SELL" if "sell" in message.lower() else "None"
                })
            
            # Sort by timestamp (newest first)
            signals.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return pd.DataFrame(signals)
        
        return self._get_cached("webhook_signals", fetch_data, expiry_seconds=10)
    
    # Risk Management Data Methods
    def get_mock_margin_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get mock margin status data for all brokers when risk manager is unavailable
        
        Returns:
            Dict: Broker key to margin status mapping
        """
        # Create mock margin status for 3 brokers
        brokers = ["tradier", "alpaca", "interactive_brokers"]
        
        margin_status = {}
        for broker in brokers:
            # Generate slightly different values for each broker
            if broker == "tradier":
                ratio = 0.65  # 65% usage
                cash = 15000.0
                margin_used = 35000.0
                maintenance_req = 50000.0
            elif broker == "alpaca":
                ratio = 0.40  # 40% usage
                cash = 25000.0
                margin_used = 30000.0
                maintenance_req = 75000.0
            else:  # interactive_brokers
                ratio = 0.85  # 85% usage - close to margin call
                cash = 10000.0
                margin_used = 85000.0
                maintenance_req = 100000.0
            
            buying_power = maintenance_req * 2  # Simplified calculation
            
            margin_status[broker] = {
                "account_id": f"demo_{broker}_account",
                "cash": cash,
                "margin_used": margin_used,
                "buying_power": buying_power,
                "maintenance_requirement": maintenance_req
            }
        
        return margin_status
    
    def get_mock_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get mock circuit breaker status data when circuit breaker is unavailable
        
        Returns:
            Dict: Circuit breaker status information
        """
        # Determine if any breakers should be active with 20% chance
        is_triggered = np.random.random() < 0.2
        
        # If triggered, randomly select which breakers are active
        active_breakers = []
        if is_triggered:
            breaker_types = ["intraday", "overall", "volatility"]
            for breaker_type in breaker_types:
                if np.random.random() < 0.3:  # 30% chance for each type to trigger
                    active_breakers.append(breaker_type)
            
            # Ensure at least one is active if triggered
            if not active_breakers:
                active_breakers.append(np.random.choice(["intraday", "overall", "volatility"]))
        
        # Generate realistic drawdown values
        current_equity = 95000.0
        peak_equity = 100000.0
        daily_peak_equity = 98000.0
        
        # Calculate drawdowns
        overall_drawdown = (peak_equity - current_equity) / peak_equity
        intraday_drawdown = (daily_peak_equity - current_equity) / daily_peak_equity
        
        # Set thresholds based on configuration
        intraday_threshold = 0.05  # 5%
        overall_threshold = 0.10   # 10%
        volatility_threshold = 0.025  # 2.5%
        
        # Current volatility value
        current_volatility = 0.022  # 2.2%
        
        return {
            "is_triggered": is_triggered,
            "active_breakers": active_breakers,
            "current_equity": current_equity,
            "peak_equity": peak_equity,
            "daily_peak_equity": daily_peak_equity,
            "overall_drawdown": overall_drawdown,
            "intraday_drawdown": intraday_drawdown,
            "overall_threshold": overall_threshold,
            "intraday_threshold": intraday_threshold,
            "volatility_threshold": volatility_threshold,
            "current_volatility": current_volatility
        }
    
    def get_mock_trading_pause_status(self) -> Dict[str, Any]:
        """
        Get mock trading pause status when orchestrator is unavailable
        
        Returns:
            Dict: Trading pause status information
        """
        # 25% chance to be paused in demo mode
        is_paused = np.random.random() < 0.25
        
        if is_paused:
            # Sample pause reasons
            reasons = [
                "intraday_drawdown_breaker",
                "overall_drawdown_breaker",
                "volatility_breaker",
                "margin_call_tradier",
                "manual_pause"
            ]
            
            # Random pause duration (between 5 and 60 minutes ago)
            pause_minutes = np.random.randint(5, 60)
            pause_time = datetime.now() - timedelta(minutes=pause_minutes)
            
            return {
                "paused": True,
                "reason": np.random.choice(reasons),
                "pause_time": pause_time
            }
        else:
            return {
                "paused": False,
                "reason": None,
                "pause_time": None
            }
    
    def get_forced_exit_history(self, max_exits: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of forced exits for risk management
        
        Args:
            max_exits: Maximum number of exits to return
            
        Returns:
            List of forced exit events
        """
        if self.is_connected:
            response = self._get_from_api(f"/api/risk/forced_exits?limit={max_exits}")
            if response and "exits" in response:
                return response["exits"]
        
        # Generate mock data if API unavailable
        mock_exits = []
        
        # Symbols for mock data
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "FB", "V", "JPM", "NFLX"]
        
        # Reasons for forced exits
        reasons = [
            "margin_call_tradier",
            "margin_call_interactive_brokers",
            "intraday_drawdown_breaker",
            "overall_drawdown_breaker",
            "volatility_circuit_breaker"
        ]
        
        # Generate exits over the past 24 hours
        now = datetime.now()
        
        # Decide how many exits to generate (0 to max_exits)
        num_exits = np.random.randint(0, max_exits + 1)
        
        for i in range(num_exits):
            # Generate a time within the past 24 hours, sorted from newest to oldest
            hours_ago = i * 24 / max(num_exits, 1)  # Spread evenly over 24 hours
            timestamp = now - timedelta(hours=hours_ago, minutes=np.random.randint(0, 60))
            
            # Generate a symbol
            symbol = np.random.choice(symbols)
            
            # Generate a quantity (between 10 and 1000, divisible by 10)
            qty = np.random.randint(1, 100) * 10
            
            # Generate a price
            price = np.random.uniform(50, 500)
            
            # Generate a P&L (between -5000 and 5000)
            pnl = np.random.uniform(-5000, 5000)
            
            mock_exits.append({
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "qty": qty,
                "reason": np.random.choice(reasons),
                "price": price,
                "pnl": pnl
            })
        
        # Sort exits by timestamp (newest first)
        mock_exits.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return mock_exits
