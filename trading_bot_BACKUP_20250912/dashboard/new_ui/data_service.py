#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Data Service

Handles data fetching from the BensBot backend components and provides
a clean interface for the UI components.
"""

import pandas as pd
import numpy as np
import datetime
import time
import logging
import threading
import random
import json
from pathlib import Path
import sys

# Ensure proper path for imports
root_dir = str(Path(__file__).parent.parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import BensBot components (try/except to handle potential import errors)
try:
    from trading_bot.portfolio.portfolio_optimizer import PortfolioOptimizer
    from trading_bot.core.portfolio_state import PortfolioStateManager
    from trading_bot.core.trade_journal import TradeJournal
    BENSBOT_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logging.warning(f"Could not import some BensBot components: {e}")
    logging.warning("Using mock data instead of real-time data")
    BENSBOT_IMPORTS_SUCCESSFUL = False

# Configure logging
logger = logging.getLogger("BensBot-DataService")

class DataService:
    """
    Service to fetch and cache data from BensBot backend components.
    Acts as a bridge between the UI and the trading system.
    """
    
    def __init__(self):
        """Initialize the data service"""
        self.last_refresh = None
        self.portfolio_mgr = None
        self.trade_journal = None
        self.connected = False
        self.connection_error = None
        
        # Try to initialize actual BensBot components
        try:
            if BENSBOT_IMPORTS_SUCCESSFUL:
                self.portfolio_mgr = self._init_portfolio_manager()
                self.trade_journal = self._init_trade_journal()
                self.connected = True
        except Exception as e:
            logger.error(f"Error initializing BensBot components: {e}")
            self.connection_error = str(e)
            self.connected = False
        
        # Set up internal data cache
        self.portfolio_data = {}
        self.positions = []
        self.trades = []
        self.strategies = []
        self.alerts = []
        self.strategy_candidates = []
        
        # Initial data load
        self.refresh_data()
        logger.info("DataService initialized")
    
    def _init_portfolio_manager(self):
        """Initialize the portfolio state manager"""
        # In a real implementation, this would connect to BensBot's actual
        # portfolio manager component
        try:
            # This is a placeholder - would be replaced with actual code
            # Example: return PortfolioStateManager(config_path="/path/to/config.json")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize portfolio manager: {e}")
            return None
    
    def _init_trade_journal(self):
        """Initialize the trade journal"""
        # In a real implementation, this would connect to BensBot's actual
        # trade journal component
        try:
            # This is a placeholder - would be replaced with actual code
            # Example: return TradeJournal(db_path="/path/to/trades.db")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize trade journal: {e}")
            return None
    
    def refresh_data(self):
        """Refresh all data from the backend"""
        self.last_refresh = datetime.datetime.now()
        
        # If we have actual BensBot components, use them
        if self.connected and self.portfolio_mgr and self.trade_journal:
            try:
                self._fetch_real_data()
            except Exception as e:
                logger.error(f"Error fetching real data: {e}")
                self._fetch_mock_data()
        else:
            # Otherwise use mock data
            self._fetch_mock_data()
        
        logger.debug("Data refreshed")
    
    def _fetch_real_data(self):
        """Fetch real data from BensBot components"""
        # In a real implementation, this would fetch data from actual components
        # For now, we'll use mock data instead
        self._fetch_mock_data()
    
    def _fetch_mock_data(self):
        """Generate mock data for demonstration"""
        # Generate mock portfolio data
        self.portfolio_data = {
            'cash': random.uniform(8000, 12000),
            'portfolio_value': random.uniform(25000, 30000),
            'daily_change': random.uniform(-2.5, 3.5),
            'total_pnl': random.uniform(-1000, 3000),
            'daily_pnl': random.uniform(-500, 800),
            'historical_values': [random.uniform(24000, 30000) for _ in range(30)]
        }
        
        # Generate mock positions
        self.positions = []
        symbols = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "FB", "NVDA", "AMD"]
        for symbol in random.sample(symbols, min(5, len(symbols))):
            self.positions.append({
                'symbol': symbol,
                'quantity': random.randint(10, 100),
                'avg_price': random.uniform(50, 500),
                'current_price': random.uniform(50, 500),
                'unrealized_pnl': random.uniform(-1000, 2000),
                'unrealized_pnl_pct': random.uniform(-10, 20),
                'strategy': random.choice(["Momentum", "Mean Reversion", "Breakout", "Trend-Following"])
            })
        
        # Generate mock trades (more recent ones with higher timestamp)
        self.trades = []
        for i in range(20):
            timestamp = datetime.datetime.now() - datetime.timedelta(minutes=i*5)
            self.trades.append({
                'id': f"trade_{i}",
                'timestamp': timestamp,
                'symbol': random.choice(symbols),
                'action': random.choice(["BUY", "SELL"]),
                'quantity': random.randint(5, 50),
                'price': random.uniform(50, 500),
                'strategy': random.choice(["Momentum", "Mean Reversion", "Breakout", "Trend-Following"]),
                'pnl': random.uniform(-500, 800) if random.random() > 0.5 else None
            })
        
        # Sort trades by timestamp (most recent first)
        self.trades = sorted(self.trades, key=lambda x: x['timestamp'], reverse=True)
        
        # Generate mock strategies
        strategy_types = ["Momentum", "Mean Reversion", "Breakout", "Trend-Following", "MACD", "Fibonacci"]
        self.strategies = []
        for i in range(3):
            strategy_type = random.choice(strategy_types)
            self.strategies.append({
                'id': f"strategy_{i}",
                'name': f"{strategy_type} {i+1}",
                'type': strategy_type,
                'status': random.choice(["Running", "Paused"]),
                'returns': random.uniform(-15, 25),
                'sharpe': random.uniform(0.5, 3.0),
                'win_rate': random.uniform(40, 70),
                'trades_count': random.randint(10, 100),
                'avg_trade_duration': random.randint(30, 300),
                'parameters': {
                    'timeframe': random.choice(["1m", "5m", "15m", "1h", "4h", "1d"]),
                    'risk_per_trade': f"{random.uniform(0.5, 2.0):.2f}%",
                    'stop_loss': f"{random.uniform(1.0, 5.0):.2f}%",
                    'take_profit': f"{random.uniform(2.0, 10.0):.2f}%"
                }
            })
        
        # Generate mock alerts
        alert_types = ["INFO", "WARNING", "ERROR"]
        alert_messages = [
            "Strategy 'Momentum' entered a new position in AAPL",
            "Broker connection lost, retrying...",
            "Order for GOOG failed due to insufficient funds",
            "Strategy 'Mean Reversion' exited position in TSLA with profit of $320",
            "New optimal portfolio weights calculated",
            "Risk limit reached for AMZN position",
            "Trading paused due to high market volatility",
            "Strategy 'Breakout' generated a new signal for NVDA",
            "Market closed, stopping day trading strategies",
            "Alternative data integration complete: sentiment analysis updated"
        ]
        
        self.alerts = []
        for i in range(15):
            timestamp = datetime.datetime.now() - datetime.timedelta(minutes=i*15)
            alert_type = random.choice(alert_types)
            self.alerts.append({
                'id': f"alert_{i}",
                'timestamp': timestamp,
                'type': alert_type,
                'message': random.choice(alert_messages),
                'source': random.choice(["System", "BrokerAPI", "Strategy", "RiskManager"])
            })
        
        # Sort alerts by timestamp (most recent first)
        self.alerts = sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)
        
        # Generate mock strategy candidates
        self.strategy_candidates = []
        for i in range(5):
            strategy_type = random.choice(strategy_types)
            backtest_score = random.uniform(60, 95)
            sharpe = random.uniform(0.8, 3.5)
            
            # Higher backtest scores tend to have better metrics
            factor = backtest_score / 100
            
            self.strategy_candidates.append({
                'id': f"candidate_{i}",
                'name': f"{strategy_type} Candidate {i+1}",
                'type': strategy_type,
                'backtest_score': backtest_score,
                'sharpe_ratio': sharpe,
                'cumulative_return': random.uniform(5, 40) * factor,
                'win_rate': random.uniform(45, 75) * factor,
                'max_drawdown': random.uniform(5, 20) * (1 - factor),
                'status': random.choice(["Ready", "In Paper Trading", "Promoted", "Failed"]),
                'parameters': {
                    'timeframe': random.choice(["1m", "5m", "15m", "1h", "4h", "1d"]),
                    'indicators': random.sample(["RSI", "MACD", "EMA", "Bollinger", "ADX", "Stochastic"], 3),
                    'asset_class': random.choice(["Stocks", "Forex", "Crypto", "Options"]),
                    'risk_per_trade': f"{random.uniform(0.5, 2.0):.2f}%"
                }
            })
    
    def get_portfolio_data(self):
        """Get current portfolio data"""
        return self.portfolio_data
    
    def get_positions(self):
        """Get current positions"""
        return self.positions
    
    def get_trades(self, limit=None, symbol=None, strategy=None):
        """
        Get trade history with optional filtering
        
        Args:
            limit: Maximum number of trades to return
            symbol: Filter by symbol
            strategy: Filter by strategy
        """
        filtered_trades = self.trades
        
        if symbol:
            filtered_trades = [t for t in filtered_trades if t['symbol'] == symbol]
        
        if strategy:
            filtered_trades = [t for t in filtered_trades if t['strategy'] == strategy]
        
        if limit:
            filtered_trades = filtered_trades[:limit]
        
        return filtered_trades
    
    def get_strategies(self):
        """Get active strategies"""
        return self.strategies
    
    def get_alerts(self, limit=None, alert_type=None):
        """
        Get system alerts with optional filtering
        
        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by alert type
        """
        filtered_alerts = self.alerts
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a['type'] == alert_type]
        
        if limit:
            filtered_alerts = filtered_alerts[:limit]
        
        return filtered_alerts
    
    def get_strategy_candidates(self):
        """Get strategy candidates from the lab"""
        return self.strategy_candidates
    
    def get_connection_status(self):
        """Get the current connection status"""
        if self.connected:
            return "Connected", None
        else:
            return "Disconnected", self.connection_error
    
    def get_daily_change(self):
        """Get daily portfolio change percentage"""
        return self.portfolio_data.get('daily_change', 0)
    
    def get_daily_pnl(self):
        """Get daily P&L in dollars"""
        return self.portfolio_data.get('daily_pnl', 0)
    
    def get_daily_pnl_percent(self):
        """Get daily P&L as percentage"""
        if 'portfolio_value' in self.portfolio_data and self.portfolio_data['portfolio_value'] > 0:
            return (self.portfolio_data.get('daily_pnl', 0) / 
                    (self.portfolio_data['portfolio_value'] - self.portfolio_data.get('daily_pnl', 0))) * 100
        return 0

    # Methods for control actions (would be integrated with BensBot components)
    def start_trading(self):
        """Start the trading system"""
        logger.info("Starting trading system")
        # In a real implementation, this would call BensBot's trading control
        # Example: AutomatedTrader.start_trading()
        return True, None
    
    def pause_trading(self):
        """Pause the trading system"""
        logger.info("Pausing trading system")
        # In a real implementation, this would call BensBot's trading control
        # Example: AutomatedTrader.pause_trading()
        return True, None
    
    def stop_trading(self):
        """Stop the trading system"""
        logger.info("Stopping trading system")
        # In a real implementation, this would call BensBot's trading control
        # Example: AutomatedTrader.stop_trading()
        return True, None
    
    def close_all_positions(self):
        """Close all positions"""
        logger.info("Closing all positions")
        # In a real implementation, this would call BensBot's trading control
        # Example: portfolio_mgr.close_all_positions()
        return True, None
    
    def restart_trading_loop(self):
        """Restart the trading loop"""
        logger.info("Restarting trading loop")
        # In a real implementation, this would call BensBot's trading control
        # Example: stop_trading() followed by start_trading()
        return True, None
    
    def upload_strategy(self, strategy_file):
        """Upload a new strategy file"""
        logger.info(f"Uploading strategy file: {strategy_file.name}")
        # In a real implementation, this would save the file and load it
        # Example: save file to strategies directory and import it
        return True, None
    
    def promote_strategy(self, strategy_id, to_paper=True):
        """
        Promote a strategy to paper or live trading
        
        Args:
            strategy_id: ID of the strategy to promote
            to_paper: If True, promote to paper trading, otherwise to live
        """
        target = "paper trading" if to_paper else "live trading"
        logger.info(f"Promoting strategy {strategy_id} to {target}")
        # In a real implementation, this would call BensBot's strategy manager
        # Example: strategy_mgr.promote_strategy(strategy_id, is_paper=to_paper)
        return True, None
    
    def backtest_strategy(self, strategy_id):
        """Run a backtest for a strategy"""
        logger.info(f"Running backtest for strategy {strategy_id}")
        # In a real implementation, this would call BensBot's backtester
        # Example: backtester.run_backtest(strategy_id)
        return True, None
