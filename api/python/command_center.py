"""
Command Center for Trading Bot

A comprehensive command-line interface for managing all aspects of the trading system,
including trade execution, market regime analysis, strategy management, and monitoring.
"""

import argparse
import logging
import json
import time
import datetime
import os
import sys
import threading
import queue
import signal
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import requests

# Import trading components
from trading_bot.strategy_loader import StrategyLoader
from trading_bot.trade_executor import TradeExecutor, TradeResult, TradeType, OrderSide, OrderType
from trading_bot.trade_executor_journal_integration import apply_journal_to_executor
from trading_bot.market_regime_integration import MarketAwareTrader, create_market_aware_trader
from trading_bot.strategy_matrix import StrategyMatrix

# Command types for better organization
class CommandType(str, Enum):
    TRADE = "trade"
    MONITOR = "monitor"
    STRATEGY = "strategy"
    RISK = "risk"
    MARKET = "market"
    DATA = "data"
    SYSTEM = "system"

# Risk modes
class RiskMode(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"
    DEFENSIVE = "defensive"

# Market data sources
class MarketDataSource(str, Enum):
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    MOCK = "mock"
    IEX = "iex"
    POLYGON = "polygon"

# Configuration schema
@dataclass
class CommandCenterConfig:
    """Configuration settings for Command Center"""
    # Account settings
    initial_balance: float = 5000.0
    default_risk_mode: str = "balanced"
    
    # Data sources
    market_data_source: str = "mock"
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # System settings
    log_level: str = "INFO"
    log_file: str = "command_center.log"
    data_directory: str = "data"
    
    # Trading settings
    default_universe: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
    trading_hours: Dict[str, str] = field(default_factory=lambda: {"start": "09:30", "end": "16:00"})
    max_positions: int = 5
    max_trades_per_day: int = 10
    
    # Integration settings
    auto_trade_frequency: int = 3600  # Seconds between auto-trading runs
    
    # Notification settings
    enable_notifications: bool = False
    notification_endpoints: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandCenterConfig':
        """Create configuration from dictionary"""
        # Filter the dictionary to only include fields that exist in the dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "initial_balance": self.initial_balance,
            "default_risk_mode": self.default_risk_mode,
            "market_data_source": self.market_data_source,
            "api_keys": self.api_keys,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "data_directory": self.data_directory,
            "default_universe": self.default_universe,
            "trading_hours": self.trading_hours,
            "max_positions": self.max_positions,
            "max_trades_per_day": self.max_trades_per_day,
            "auto_trade_frequency": self.auto_trade_frequency,
            "enable_notifications": self.enable_notifications,
            "notification_endpoints": self.notification_endpoints
        } 

class MarketDataManager:
    """Manager for retrieving and caching market data"""
    
    def __init__(self, config: CommandCenterConfig):
        """
        Initialize the market data manager
        
        Args:
            config: Command center configuration
        """
        self.config = config
        self.logger = logging.getLogger("MarketDataManager")
        
        # Cache for market data
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.market_stats_cache: Dict[str, Any] = {}
        
        # Data source configuration
        self.data_source = config.market_data_source
        self.api_keys = config.api_keys
        
        # Cache directory
        self.cache_dir = os.path.join(config.data_directory, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Historical data storage
        self.historical_dir = os.path.join(config.data_directory, "historical")
        os.makedirs(self.historical_dir, exist_ok=True)
    
    def get_price(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get current price data for a symbol
        
        Args:
            symbol: Symbol to retrieve price for
            force_refresh: Whether to force a refresh from source
            
        Returns:
            Price data dictionary
        """
        current_time = time.time()
        
        # Check cache first unless force refresh
        if not force_refresh and symbol in self.price_cache:
            cached_data = self.price_cache[symbol]
            # Use cache if less than 60 seconds old
            if current_time - cached_data.get("timestamp", 0) < 60:
                return cached_data
        
        # Different data source implementations
        if self.data_source == MarketDataSource.MOCK.value:
            # Generate mock data based on symbol hash for consistency
            price_data = self._get_mock_price(symbol)
        
        elif self.data_source == MarketDataSource.YAHOO.value:
            # Would implement real Yahoo Finance API call
            price_data = self._get_yahoo_price(symbol)
            
        elif self.data_source == MarketDataSource.ALPHA_VANTAGE.value:
            # Would implement Alpha Vantage API call
            api_key = self.api_keys.get("alpha_vantage")
            if not api_key:
                self.logger.warning("Alpha Vantage API key not found, using mock data")
                price_data = self._get_mock_price(symbol)
            else:
                price_data = self._get_alpha_vantage_price(symbol, api_key)
        
        else:
            # Fallback to mock data
            self.logger.warning(f"Unsupported data source: {self.data_source}, using mock data")
            price_data = self._get_mock_price(symbol)
        
        # Add timestamp to cache
        price_data["timestamp"] = current_time
        
        # Update cache
        self.price_cache[symbol] = price_data
        
        return price_data
    
    def _get_mock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock price data for a symbol
        
        Args:
            symbol: Symbol to generate data for
            
        Returns:
            Mock price data dictionary
        """
        # Use hash of symbol to generate consistent mock data
        symbol_hash = abs(hash(symbol)) % 1000
        base_price = 50 + (symbol_hash % 200)
        
        # Add some time-based variation
        time_factor = (int(time.time() / 300) % 10) / 10.0  # Changes every 5 minutes
        price_mod = base_price * 0.01 * time_factor
        
        return {
            "symbol": symbol,
            "price": base_price + price_mod,
            "bid": base_price + price_mod - 0.05,
            "ask": base_price + price_mod + 0.05,
            "volume": 10000 + (symbol_hash % 90000),
            "high": base_price + price_mod + 0.5,
            "low": base_price + price_mod - 0.5,
            "open": base_price,
            "source": "mock"
        }
    
    def _get_yahoo_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get price data from Yahoo Finance
        
        Args:
            symbol: Symbol to retrieve price for
            
        Returns:
            Price data dictionary
        """
        # This would be implemented using the yahoo_fin package or similar
        # For now, return mock data
        data = self._get_mock_price(symbol)
        data["source"] = "yahoo"
        return data
    
    def _get_alpha_vantage_price(self, symbol: str, api_key: str) -> Dict[str, Any]:
        """
        Get price data from Alpha Vantage
        
        Args:
            symbol: Symbol to retrieve price for
            api_key: Alpha Vantage API key
            
        Returns:
            Price data dictionary
        """
        # This would be implemented using the Alpha Vantage API
        # For now, return mock data
        data = self._get_mock_price(symbol)
        data["source"] = "alpha_vantage"
        return data
    
    def get_market_stats(self) -> Dict[str, Any]:
        """
        Get overall market statistics
        
        Returns:
            Market statistics dictionary
        """
        current_time = time.time()
        
        # Check cache
        if self.market_stats_cache and current_time - self.market_stats_cache.get("timestamp", 0) < 300:
            return self.market_stats_cache
        
        # Generate or fetch market stats
        if self.data_source == MarketDataSource.MOCK.value:
            stats = self._get_mock_market_stats()
        else:
            # Would implement real data fetching for other sources
            stats = self._get_mock_market_stats()
        
        # Add timestamp to cache
        stats["timestamp"] = current_time
        
        # Update cache
        self.market_stats_cache = stats
        
        return stats
    
    def _get_mock_market_stats(self) -> Dict[str, Any]:
        """
        Generate mock market statistics
        
        Returns:
            Mock market statistics dictionary
        """
        # Time-based variation for some realism
        time_hash = int(time.time() / 900)  # Changes every 15 minutes
        time_factor = (time_hash % 10) / 10.0
        
        # VIX tends to be inverse to market direction
        spy_trend = 0.5 + time_factor
        vix_value = 30 - (spy_trend * 15)
        
        return {
            "vix": vix_value,
            "spy_price": 450 * spy_trend,
            "spy_20dma": 445,
            "spy_50dma": 440,
            "spy_200dma": 430,
            "spy_above_20dma": spy_trend > 0.5,
            "new_highs": int(200 * spy_trend),
            "new_lows": int(200 * (1 - spy_trend)),
            "more_new_highs_than_lows": spy_trend > 0.5,
            "advance_decline_ratio": 1.0 + (spy_trend - 0.5) * 2,
            "put_call_ratio": 1.0 - (spy_trend - 0.5),
            "is_earnings_season": (datetime.datetime.now().month % 3) == 0,
            "is_fomc_week": False,
            "is_opex_week": (datetime.datetime.now().day > 15 and datetime.datetime.now().day < 22),
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "source": "mock"
        }
    
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1y",
                          interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Symbol to retrieve data for
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with historical price data
        """
        # Check if we have cached data
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{period}_{interval}.csv")
        
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            # Use cache if less than 24 hours old
            if file_age < 86400:
                try:
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
                except Exception as e:
                    self.logger.warning(f"Error reading cache file: {e}")
        
        # Generate or fetch historical data
        if self.data_source == MarketDataSource.MOCK.value:
            df = self._get_mock_historical_data(symbol, period, interval)
        else:
            # Would implement real data fetching for other sources
            df = self._get_mock_historical_data(symbol, period, interval)
        
        # Save to cache
        try:
            df.to_csv(cache_file)
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")
        
        return df
    
    def _get_mock_historical_data(self, 
                                symbol: str, 
                                period: str,
                                interval: str) -> pd.DataFrame:
        """
        Generate mock historical data
        
        Args:
            symbol: Symbol to generate data for
            period: Time period
            interval: Data interval
            
        Returns:
            DataFrame with mock historical data
        """
        # Determine number of data points based on period and interval
        period_days = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 3650
        }
        
        interval_minutes = {
            "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, 
            "60m": 60, "90m": 90, "1h": 60, "1d": 1440, 
            "5d": 7200, "1wk": 10080, "1mo": 43200, "3mo": 129600
        }
        
        days = period_days.get(period, 365)
        minutes = interval_minutes.get(interval, 1440)
        
        # Calculate number of data points
        num_points = min(int((days * 1440) / minutes), 10000)  # Cap at 10k points
        
        # Use hash of symbol to generate consistent base price
        symbol_hash = abs(hash(symbol)) % 1000
        base_price = 50 + (symbol_hash % 200)
        
        # Generate date range
        end_date = datetime.datetime.now()
        date_range = pd.date_range(end=end_date, periods=num_points, freq=f"{minutes}min")
        
        # Initialize with base price and add random walk
        prices = [base_price]
        for i in range(1, num_points):
            # Random walk with drift based on symbol hash
            drift = (symbol_hash % 10) / 1000  # Small positive drift for most symbols
            random_walk = (np.random.random() - 0.5) * 0.02  # +/- 1% random movement
            new_price = prices[-1] * (1 + drift + random_walk)
            prices.append(new_price)
        
        # Create dataframe
        df = pd.DataFrame({
            "Open": prices,
            "High": [p * (1 + np.random.random() * 0.01) for p in prices],
            "Low": [p * (1 - np.random.random() * 0.01) for p in prices],
            "Close": [p * (1 + (np.random.random() - 0.5) * 0.005) for p in prices],
            "Volume": [int(1000000 * np.random.random()) for _ in range(num_points)]
        }, index=date_range)
        
        return df
    
    def get_earnings_calendar(self, days_ahead: int = 10) -> List[Dict[str, Any]]:
        """
        Get upcoming earnings announcements
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of earnings announcement dictionaries
        """
        # Generate mock earnings announcements
        earnings = []
        current_date = datetime.datetime.now().date()
        
        for i in range(days_ahead):
            announce_date = current_date + datetime.timedelta(days=i)
            
            # Generate more earnings on Tuesday-Thursday
            weekday = announce_date.weekday()
            num_announcements = 5 if 1 <= weekday <= 3 else 2
            
            for j in range(num_announcements):
                symbol_idx = (i * 10 + j) % len(self.config.default_universe)
                symbol = self.config.default_universe[symbol_idx]
                
                earnings.append({
                    "symbol": symbol,
                    "date": announce_date.strftime("%Y-%m-%d"),
                    "time": "After Market" if j % 2 == 0 else "Before Market",
                    "expected_eps": round(np.random.random() * 2, 2),
                    "expected_revenue": round(np.random.random() * 10000, 2)
                })
        
        return earnings
    
    def get_economic_calendar(self, days_ahead: int = 10) -> List[Dict[str, Any]]:
        """
        Get upcoming economic events
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of economic event dictionaries
        """
        # Common economic events
        event_types = [
            "FOMC Meeting", "GDP Release", "CPI Data", "Unemployment",
            "Retail Sales", "PMI", "Housing Starts", "Trade Balance"
        ]
        
        events = []
        current_date = datetime.datetime.now().date()
        
        for i in range(days_ahead):
            announce_date = current_date + datetime.timedelta(days=i)
            
            # Randomly decide if there's an event this day
            if np.random.random() < 0.3:
                event_idx = (i * 3) % len(event_types)
                
                events.append({
                    "event": event_types[event_idx],
                    "date": announce_date.strftime("%Y-%m-%d"),
                    "time": f"{8 + (i % 8):02d}:30",
                    "importance": "High" if event_types[event_idx] in ["FOMC Meeting", "GDP Release", "CPI Data"] else "Medium"
                })
        
        return events 

class NotificationManager:
    """Manager for sending notifications and alerts"""
    
    def __init__(self, config: CommandCenterConfig):
        """
        Initialize the notification manager
        
        Args:
            config: Command center configuration
        """
        self.config = config
        self.logger = logging.getLogger("NotificationManager")
        
        self.enable_notifications = config.enable_notifications
        self.endpoints = config.notification_endpoints
    
    def send_notification(self, 
                        title: str, 
                        message: str, 
                        level: str = "info",
                        data: Dict[str, Any] = None):
        """
        Send a notification to configured endpoints
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level ('info', 'warning', 'error', 'success')
            data: Optional additional data
        """
        if not self.enable_notifications:
            return
        
        # Prepare notification data
        notification = {
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data or {}
        }
        
        # Send to webhook if configured
        webhook_url = self.endpoints.get("webhook")
        if webhook_url:
            try:
                response = requests.post(webhook_url, json=notification, timeout=5)
                if response.status_code not in (200, 201, 202, 204):
                    self.logger.warning(f"Webhook notification failed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error sending webhook notification: {e}")
        
        # Send to other notification services if configured
        # (e.g., email, SMS, Slack, Telegram, etc.)
    
    def send_trade_notification(self, trade_result: TradeResult):
        """
        Send a notification about a trade execution
        
        Args:
            trade_result: TradeResult object
        """
        if trade_result.success:
            level = "success"
            title = f"Trade Executed: {trade_result.symbol}"
        else:
            level = "error"
            title = f"Trade Failed: {trade_result.symbol}"
        
        message = f"{trade_result.quantity} {trade_result.symbol} @ {trade_result.order_status}"
        if not trade_result.success and trade_result.error_message:
            message += f" - Error: {trade_result.error_message}"
        
        self.send_notification(
            title=title,
            message=message,
            level=level,
            data=trade_result.to_dict()
        )
    
    def send_alert(self, alert_type: str, severity: str, message: str, data: Dict[str, Any] = None):
        """
        Send an alert notification
        
        Args:
            alert_type: Type of alert
            severity: Alert severity ('high', 'medium', 'low')
            message: Alert message
            data: Optional additional data
        """
        # Map severity to notification level
        level_map = {"high": "error", "medium": "warning", "low": "info"}
        level = level_map.get(severity, "info")
        
        title = f"Alert [{severity.upper()}]: {alert_type}"
        
        self.send_notification(
            title=title,
            message=message,
            level=level,
            data=data
        )
    
    def send_performance_update(self, performance: Dict[str, Any]):
        """
        Send a performance update notification
        
        Args:
            performance: Performance metrics dictionary
        """
        # Determine if positive or negative performance
        net_pnl = performance.get("net_pnl", 0.0)
        level = "success" if net_pnl >= 0 else "warning"
        
        title = "Daily Performance Update"
        message = f"Net P&L: ${net_pnl:.2f}, Win Rate: {performance.get('win_rate', 0):.1f}%"
        
        self.send_notification(
            title=title,
            message=message,
            level=level,
            data=performance
        ) 

class CommandCenter:
    """
    Advanced command center for managing trading operations,
    integrating strategy management, execution, monitoring, and reporting.
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the command center
        
        Args:
            config_file: Path to configuration file
        """
        # Set up logging first
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.logger.info("Initializing Command Center...")
        
        # Core components
        self.loader = StrategyLoader()
        self.loader.load_all()
        
        self.balance = self.config.initial_balance
        self.risk_mode = self.config.default_risk_mode
        
        # Market data system
        self.data_manager = MarketDataManager(self.config)
        
        # Trading components
        self.executor = TradeExecutor(
            loader=self.loader, 
            account_balance=self.balance, 
            risk_mode=self.risk_mode
        )
        
        # Enhance with journal integration
        self.journaled_executor = apply_journal_to_executor(self.executor)
        
        # Market regime aware trader
        self.market_trader = create_market_aware_trader(
            loader=self.loader,
            executor=self.executor,
            config={
                "universe": self.config.default_universe,
                "max_positions": self.config.max_positions,
                "max_trades_per_day": self.config.max_trades_per_day
            }
        )
        
        # Supporting systems
        self.notifier = NotificationManager(self.config)
        
        # Command queue for asynchronous operation
        self.command_queue = queue.Queue()
        self.is_running = False
        self.command_thread = None
        
        # Record of executed commands
        self.command_history: List[Dict[str, Any]] = []
        
        # Strategy matrix for market condition based strategy selection
        self.strategy_matrix = StrategyMatrix()
        
        self.logger.info("Command Center initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logger for this class
        self.logger = logging.getLogger("CommandCenter")
        self.logger.setLevel(logging.INFO)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)
        
        # Root logger configuration
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self, config_file: str) -> CommandCenterConfig:
        """
        Load configuration from file
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            CommandCenterConfig object
        """
        # Default configuration
        default_config = CommandCenterConfig()
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self.logger.info(f"Configuration loaded from {config_file}")
                return CommandCenterConfig.from_dict(config_data)
            else:
                self.logger.warning(f"Configuration file {config_file} not found, using defaults")
                
                # Save default configuration for future use
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    yaml.dump(default_config.to_dict(), f, default_flow_style=False)
                
                return default_config
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def start(self):
        """Start the command center operations"""
        self.logger.info("Starting Command Center...")
        
        # Start command processing thread
        self.is_running = True
        self.command_thread = threading.Thread(target=self._process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        # Start executor thread for async trade execution
        self.executor.start_execution_thread()
        
        self.logger.info("Command Center started successfully")
    
    def stop(self):
        """Stop the command center operations"""
        self.logger.info("Stopping Command Center...")
        
        # Stop command processing
        self.is_running = False
        if self.command_thread:
            self.command_thread.join(timeout=5.0)
        
        # Stop executor thread
        self.executor.stop_execution_thread()
        
        self.logger.info("Command Center stopped successfully")
    
    def _process_commands(self):
        """Background thread for processing commands"""
        while self.is_running:
            try:
                # Get command with timeout
                command, args, callback = self.command_queue.get(timeout=1.0)
                
                # Execute command
                self.logger.info(f"Processing command: {command}")
                result = self._execute_command(command, args)
                
                # Call callback with result if provided
                if callback:
                    callback(result)
                
                # Record command in history
                self.command_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "command": command,
                    "args": args,
                    "success": result.get("success", False) if isinstance(result, dict) else True
                })
                
                # Limit history length
                if len(self.command_history) > 100:
                    self.command_history = self.command_history[-100:]
                
                # Mark task as done
                self.command_queue.task_done()
                
            except queue.Empty:
                # Timeout on empty queue, just continue
                continue
                
            except Exception as e:
                self.logger.error(f"Error processing command: {e}")
    
    def _execute_command(self, command: str, args: Dict[str, Any]) -> Any:
        """
        Execute a command
        
        Args:
            command: Command to execute
            args: Command arguments
            
        Returns:
            Command execution result
        """
        # Map command to handler method
        command_handlers = {
            # Trade commands
            "manual_trade": self.manual_trade,
            "exit_trade": self.exit_trade,
            "update_stop_loss": self._handle_update_stop_loss,
            
            # Risk commands
            "toggle_risk_mode": self.toggle_risk_mode,
            "set_balance": self._handle_set_balance,
            
            # Integration commands
            "run_market_trader": self.run_market_trader,
            "analyze_market_regime": self._handle_analyze_market_regime,
            
            # Market commands
            "get_market_stats": self._handle_get_market_stats,
            "get_price": self._handle_get_price,
            
            # System commands
            "reload_strategies": self._handle_reload_strategies,
            
            # Strategy Matrix commands
            "get_strategy_recommendations": self._handle_get_strategy_recommendations,
            "generate_trade_signal": self._handle_generate_trade_signal,
            "execute_strategy_recommendation": self.execute_strategy_recommendation
        }
        
        # Get handler for command
        handler = command_handlers.get(command)
        if not handler:
            self.logger.warning(f"Unknown command: {command}")
            return {"success": False, "error": f"Unknown command: {command}"}
        
        # Execute handler with arguments
        try:
            return handler(**args)
        except Exception as e:
            self.logger.error(f"Error executing command '{command}': {e}")
            return {"success": False, "error": str(e)}
    
    def queue_command(self, 
                    command: str, 
                    args: Dict[str, Any] = None, 
                    callback: Callable[[Any], None] = None) -> bool:
        """
        Queue a command for asynchronous execution
        
        Args:
            command: Command to execute
            args: Command arguments
            callback: Optional callback function to call with result
            
        Returns:
            Whether command was successfully queued
        """
        if not self.is_running:
            self.logger.warning("Cannot queue command: Command Center is not running")
            return False
        
        self.command_queue.put((command, args or {}, callback))
        return True
    
    # --- Trade Management Commands ---
    
    def toggle_risk_mode(self, new_mode: str) -> Dict[str, Any]:
        """
        Switch to a different risk mode
        
        Args:
            new_mode: New risk mode to use
            
        Returns:
            Result dictionary
        """
        self.logger.info(f"Switching risk mode to {new_mode.upper()}...")
        
        if new_mode not in [mode.value for mode in RiskMode]:
            return {
                "success": False,
                "error": f"Invalid risk mode: {new_mode}. Valid modes: {', '.join([mode.value for mode in RiskMode])}"
            }
        
        # Update risk mode
        self.risk_mode = new_mode
        self.executor.risk_mode = new_mode
        
        # Get and set risk profile
        risk_profile = self.loader.get_risk_profile(new_mode)
        
        # Send notification
        self.notifier.send_notification(
            title="Risk Mode Changed",
            message=f"Risk mode switched to {new_mode.upper()}",
            level="info"
        )
        
        return {
            "success": True,
            "mode": new_mode,
            "profile": risk_profile
        }
    
    def _handle_set_balance(self, balance: float) -> Dict[str, Any]:
        """
        Update account balance
        
        Args:
            balance: New account balance
            
        Returns:
            Result dictionary
        """
        if balance <= 0:
            return {
                "success": False,
                "error": "Balance must be positive"
            }
        
        old_balance = self.balance
        self.balance = balance
        self.executor.account_balance = balance
        
        self.logger.info(f"Account balance updated: ${old_balance:.2f} -> ${balance:.2f}")
        
        return {
            "success": True,
            "old_balance": old_balance,
            "new_balance": balance
        }
    
    def manual_trade(self, 
                   symbol: str, 
                   strategy: str, 
                   direction: str = "buy",
                   trade_type: str = "equity", 
                   quantity: int = None,
                   contracts: int = 1,
                   price: float = None) -> Dict[str, Any]:
        """
        Manually execute a trade
        
        Args:
            symbol: Symbol to trade
            strategy: Strategy to use
            direction: Trade direction ('buy' or 'sell')
            trade_type: Type of trade ('equity' or 'options')
            quantity: Quantity for equity trades
            contracts: Number of contracts for options trades
            price: Optional override price
            
        Returns:
            Trade execution result
        """
        self.logger.info(f"Manual trade requested: {symbol} {direction} using {strategy}")
        
        # Get current price if not provided
        if price is None:
            price_data = self.data_manager.get_price(symbol)
            price = price_data.get("price", 100.0)
        
        # Prepare trade signal
        trade_signal = {
            "type": trade_type,
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy,
            "price": price
        }
        
        # Add trade type specific parameters
        if trade_type.lower() == "options":
            trade_signal["contracts"] = contracts
        elif quantity is not None:
            trade_signal["quantity"] = quantity
        
        # Execute trade
        result = self.executor.route_trade(trade_signal)
        
        # Send notification
        self.notifier.send_trade_notification(result)
        
        # Log result
        if result.success:
            self.logger.info(f"Manual trade executed: {result}")
        else:
            self.logger.warning(f"Manual trade failed: {result.error_message}")
        
        return result.to_dict()
    
    def exit_trade(self, trade_id: str, reason: str = "manual") -> Dict[str, Any]:
        """
        Exit an open trade
        
        Args:
            trade_id: ID of the trade to exit
            reason: Reason for exiting
            
        Returns:
            Trade exit result
        """
        self.logger.info(f"Exiting trade {trade_id}...")
        
        result = self.executor.exit_trade(trade_id, reason)
        
        # Notify if successful
        if result.success:
            self.notifier.send_notification(
                title="Trade Exited",
                message=f"Trade {trade_id} exited: {reason}",
                level="info",
                data=result.to_dict()
            )
        else:
            self.logger.warning(f"Failed to exit trade {trade_id}: {result.error_message}")
        
        return result.to_dict()
    
    def _handle_update_stop_loss(self, 
                               trade_id: str, 
                               stop_loss: float, 
                               profit_target: float = None) -> Dict[str, Any]:
        """
        Update stop loss and optional profit target for a trade
        
        Args:
            trade_id: ID of the trade to update
            stop_loss: New stop loss price
            profit_target: Optional new profit target price
            
        Returns:
            Update result
        """
        self.logger.info(f"Updating exit parameters for trade {trade_id}...")
        
        # Update in executor
        success = self.executor.update_exit_parameters(
            trade_id=trade_id,
            stop_loss=stop_loss,
            profit_target=profit_target
        )
        
        if success:
            updates = f"stop loss = ${stop_loss:.2f}"
            if profit_target:
                updates += f", profit target = ${profit_target:.2f}"
                
            self.logger.info(f"Trade {trade_id} updated: {updates}")
            
            return {
                "success": True,
                "trade_id": trade_id,
                "stop_loss": stop_loss,
                "profit_target": profit_target
            }
        else:
            self.logger.warning(f"Failed to update trade {trade_id}")
            
            return {
                "success": False,
                "error": f"Failed to update trade {trade_id}"
            }
    
    # --- Market Regime and Integration Commands ---
    
    def _handle_analyze_market_regime(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the current market regime
        
        Args:
            market_data: Optional market data (if None, will fetch current data)
            
        Returns:
            Market regime analysis result
        """
        # Get market data if not provided
        if market_data is None:
            market_data = self.data_manager.get_market_stats()
        
        # Analyze regime
        regime = self.market_trader.analyze_market_regime(market_data)
        
        regime_details = self.market_trader.get_current_regime()
        
        self.logger.info(f"Market regime analysis: {regime}")
        
        return {
            "success": True,
            "regime": regime,
            "details": regime_details
        }
    
    def run_market_trader(self, 
                       symbol: str = None,
                       symbols: List[str] = None) -> Dict[str, Any]:
        """
        Run the market-aware trader on specified symbols
        
        Args:
            symbol: Single symbol to run on
            symbols: List of symbols to run on
            
        Returns:
            Execution result
        """
        # Determine symbols to use
        target_symbols = []
        if symbol:
            target_symbols = [symbol]
        elif symbols:
            target_symbols = symbols
        else:
            target_symbols = self.config.default_universe[:3]  # Use first 3 from default universe
        
        # Get market context
        market_data = self.data_manager.get_market_stats()
        
        # Get price data for symbols
        price_data = {}
        for sym in target_symbols:
            price_info = self.data_manager.get_price(sym)
            price_data[sym] = price_info.get("price", 100.0)
        
        self.logger.info(f"Running market-aware trader on symbols: {', '.join(target_symbols)}")
        
        # Run full trading cycle
        result = self.market_trader.run_full_cycle(market_data, price_data)
        
        # Process results
        executed_trades = result.get("trades_executed", 0)
        
        # Return summary
        if executed_trades > 0:
            self.logger.info(f"Market trader executed {executed_trades} trades successfully")
            
            # Send notification
            self.notifier.send_notification(
                title="Market Trader Executed Trades",
                message=f"Executed {executed_trades} trades in {result.get('regime', 'unknown')} regime",
                level="success",
                data={"regime": result.get("regime", "unknown")}
            )
        else:
            self.logger.info("Market trader did not execute any trades")
        
        return {
            "success": True,
            "result": result
        }
    
    # --- Market Data Commands ---
    
    def _handle_get_market_stats(self) -> Dict[str, Any]:
        """
        Get current market statistics
        
        Returns:
            Market statistics
        """
        stats = self.data_manager.get_market_stats()
        return {
            "success": True,
            "stats": stats
        }
    
    def _handle_get_price(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get price data for a symbol
        
        Args:
            symbol: Symbol to get price for
            force_refresh: Whether to force refresh from source
            
        Returns:
            Price data
        """
        price_data = self.data_manager.get_price(symbol, force_refresh)
        return {
            "success": True,
            "price_data": price_data
        }
    
    # --- System Commands ---
    
    def _handle_reload_strategies(self) -> Dict[str, Any]:
        """
        Reload strategies from disk
        
        Returns:
            Reload result dictionary
        """
        try:
            self.loader.load_all()
            
            # Count loaded strategies
            strategy_count = 0
            try:
                strategy_dict = self.loader.strategies.get("strategies", {})
                strategy_count = len(strategy_dict)
            except:
                pass
            
            self.logger.info(f"Strategies reloaded successfully: {strategy_count} strategies loaded")
            
            return {
                "success": True,
                "strategy_count": strategy_count
            }
        except Exception as e:
            self.logger.error(f"Error reloading strategies: {e}")
            
            return {
                "success": False,
                "error": f"Error reloading strategies: {e}"
            }
    
    # --- Strategy Matrix Commands ---
    
    def _handle_get_strategy_recommendations(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get strategy recommendations based on current market conditions
        
        Args:
            market_data: Optional market data (if None, will fetch current data)
            
        Returns:
            Strategy recommendations
        """
        # Get market data if not provided
        if market_data is None:
            market_data = self.data_manager.get_market_stats()
        
        # Get recommendations from strategy matrix
        recommendations = self.strategy_matrix.get_recommended_strategies(market_data)
        
        self.logger.info(f"Strategy recommendations generated: {len(recommendations['strategies'])} strategies for {recommendations['market_regime']} regime")
        
        return {
            "success": True,
            "recommendations": recommendations,
            "market_data": market_data
        }
    
    def _handle_generate_trade_signal(self, 
                                    strategy_name: str = None, 
                                    market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a trade signal for a specific strategy
        
        Args:
            strategy_name: Name of the strategy to use (if None, will use top recommended)
            market_data: Optional market data (if None, will fetch current data)
            
        Returns:
            Trade signal
        """
        # Get market data if not provided
        if market_data is None:
            market_data = self.data_manager.get_market_stats()
        
        # Get recommendations
        recommendations = self.strategy_matrix.get_recommended_strategies(market_data)
        
        # Find strategy to use
        strategy = None
        if strategy_name:
            # Find specified strategy
            for s in recommendations["strategies"]:
                if s["name"] == strategy_name:
                    strategy = s
                    break
            
            if not strategy:
                self.logger.warning(f"Strategy '{strategy_name}' not found in recommendations")
                return {
                    "success": False,
                    "error": f"Strategy '{strategy_name}' not found in recommendations"
                }
        else:
            # Use top recommended strategy
            if recommendations["strategies"]:
                strategy = recommendations["strategies"][0]
            else:
                self.logger.warning("No recommended strategies available")
                return {
                    "success": False,
                    "error": "No recommended strategies available"
                }
        
        # Generate trade signal
        account = {
            "balance": self.balance,
            "risk_mode": self.risk_mode
        }
        
        # Add applicable conditions to market data for risk adjustments
        market_data["applicable_conditions"] = recommendations["applicable_conditions"]
        
        signal = self.strategy_matrix.generate_trade_signal(strategy, market_data, account)
        
        self.logger.info(f"Trade signal generated for strategy '{strategy['name']}': {signal['direction']} {signal['trade_type']}")
        
        return {
            "success": True,
            "signal": signal,
            "strategy": strategy,
            "market_regime": recommendations["market_regime"]
        }
    
    def execute_strategy_recommendation(self, 
                                     strategy_name: str = None, 
                                     market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a trade based on strategy recommendation
        
        Args:
            strategy_name: Name of the strategy to use (if None, will use top recommended)
            market_data: Optional market data (if None, will fetch current data)
            
        Returns:
            Trade execution result
        """
        self.logger.info(f"Executing strategy recommendation: {strategy_name if strategy_name else 'top recommended'}")
        
        # Generate trade signal
        signal_result = self._handle_generate_trade_signal(strategy_name, market_data)
        
        if not signal_result["success"]:
            return signal_result
        
        signal = signal_result["signal"]
        
        # Convert signal to trade parameters
        if signal["trade_type"] == "equity":
            # Execute equity trade
            result = self.manual_trade(
                symbol=signal.get("symbol", "SPY"),  # Default to SPY if not specified
                strategy=signal["strategy"],
                direction=signal["direction"],
                trade_type="equity",
                quantity=signal.get("quantity"),
                price=signal.get("price")
            )
            
        elif signal["trade_type"] == "options":
            # Execute options trade
            # In a real implementation, we would extract specific options parameters
            # For now, just pass the basics
            result = self.manual_trade(
                symbol=signal.get("symbol", "SPY"),  # Default to SPY if not specified
                strategy=signal["strategy"],
                direction=signal["direction"],
                trade_type="options",
                contracts=1,  # Default to 1 contract
                price=signal.get("price")
            )
            
        else:
            self.logger.error(f"Unsupported trade type: {signal['trade_type']}")
            return {
                "success": False,
                "error": f"Unsupported trade type: {signal['trade_type']}"
            }
        
        # Return combined result
        return {
            "success": result.get("success", False),
            "trade_result": result,
            "signal": signal,
            "strategy": signal_result["strategy"],
            "market_regime": signal_result["market_regime"]
        }
    
    # --- Utility Methods ---
    
    def generate_status_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system status report
        
        Returns:
            Status report dictionary
        """
        # Get account information
        account_info = {
            "balance": self.balance,
            "risk_mode": self.risk_mode
        }
        
        # Get positions
        open_trades = self.executor.get_open_trades()
        
        # Get market stats
        market_stats = self.data_manager.get_market_stats()
        
        # Get market regime if available
        market_regime = None
        if hasattr(self.market_trader, "current_regime") and self.market_trader.current_regime:
            market_regime = self.market_trader.current_regime
        else:
            # Try to analyze regime
            try:
                market_regime = self.market_trader.analyze_market_regime(market_stats)
            except:
                pass
        
        # Get command statistics
        command_stats = {
            "total_executed": len(self.command_history),
            "recent_commands": self.command_history[-5:] if self.command_history else []
        }
        
        # Compile report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "account": account_info,
            "positions": open_trades,
            "market": {
                "vix": market_stats.get("vix", 0),
                "spy_above_20dma": market_stats.get("spy_above_20dma", False),
                "regime": market_regime
            },
            "commands": command_stats,
            "system_status": {
                "command_queue_size": self.command_queue.qsize(),
                "is_running": self.is_running
            }
        }
        
        return report
    
    def format_status_report(self, report: Dict[str, Any] = None) -> str:
        """
        Format status report as text for display
        
        Args:
            report: Optional report dictionary (if None, will generate)
            
        Returns:
            Formatted status report text
        """
        if report is None:
            report = self.generate_status_report()
        
        # Format account section
        account_section = f"""
=== ACCOUNT STATUS ===
Balance: ${report['account']['balance']:.2f}
Risk Mode: {report['account']['risk_mode'].upper()}
        """
        
        # Format positions section
        positions = report.get("positions", [])
        if positions:
            positions_section = f"""
=== POSITIONS ({len(positions)}) ===
"""
            for pos in positions:
                positions_section += f"- {pos.get('symbol', '')}: {pos.get('quantity', 0)} @ ${pos.get('entry_price', 0.0):.2f} ({pos.get('direction', '')})\n"
        else:
            positions_section = """
=== POSITIONS (0) ===
No open positions
            """
        
        # Format market section
        market = report.get("market", {})
        market_section = f"""
=== MARKET CONDITIONS ===
VIX: {market.get('vix', 0):.2f}
SPY > 20 DMA: {'Yes' if market.get('spy_above_20dma', False) else 'No'}
Market Regime: {market.get('regime', 'Unknown')}
        """
        
        # Format system status
        system = report.get("system_status", {})
        system_section = f"""
=== SYSTEM STATUS ===
Running: {system.get('is_running', False)}
Command Queue: {system.get('command_queue_size', 0)} pending
        """
        
        # Combine all sections
        return f"""
========================================
COMMAND CENTER STATUS REPORT
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================
{account_section}
{positions_section}
{market_section}
{system_section}
========================================
        """ 

# Main CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot Command Center")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    
    # General commands
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    # Trading commands
    trade_group = parser.add_argument_group("Trading Commands")
    trade_group.add_argument("--trade", nargs="+", metavar=("SYMBOL", "STRATEGY"), help="Execute trade (symbol strategy [direction] [quantity])")
    trade_group.add_argument("--exit", type=str, metavar="TRADE_ID", help="Exit a trade by ID")
    
    # Risk commands
    risk_group = parser.add_argument_group("Risk Management")
    risk_group.add_argument("--risk-mode", type=str, choices=[m.value for m in RiskMode], help="Set risk mode")
    risk_group.add_argument("--balance", type=float, help="Set account balance")
    
    # Market regime commands
    regime_group = parser.add_argument_group("Market Regime")
    regime_group.add_argument("--analyze-regime", action="store_true", help="Analyze current market regime")
    regime_group.add_argument("--run-trader", nargs="*", metavar="SYMBOL", help="Run market-aware trader on symbols")
    
    # Market data commands
    data_group = parser.add_argument_group("Market Data")
    data_group.add_argument("--market-stats", action="store_true", help="Show current market statistics")
    data_group.add_argument("--price", type=str, metavar="SYMBOL", help="Get price for a symbol")
    
    # System commands
    system_group = parser.add_argument_group("System")
    system_group.add_argument("--reload", action="store_true", help="Reload strategies")
    
    # Strategy matrix commands
    strategy_matrix_group = parser.add_argument_group("Strategy Matrix")
    strategy_matrix_group.add_argument("--recommend-strategies", action="store_true", help="Get strategy recommendations based on current market conditions")
    strategy_matrix_group.add_argument("--generate-signal", nargs="?", const=None, metavar="STRATEGY", help="Generate a trade signal based on strategy recommendation")
    strategy_matrix_group.add_argument("--execute-recommendation", nargs="?", const=None, metavar="STRATEGY", help="Execute a trade based on strategy recommendation")
    
    args = parser.parse_args()
    
    # Create and start command center
    cc = CommandCenter(config_file=args.config)
    cc.start()
    
    try:
        # Process command line arguments
        if args.risk_mode:
            result = cc.toggle_risk_mode(args.risk_mode)
            print(f"Risk mode changed to {args.risk_mode}: {'Success' if result['success'] else 'Failed - ' + result.get('error', '')}")
        
        if args.balance:
            result = cc._handle_set_balance(args.balance)
            if result["success"]:
                print(f"Balance updated to ${args.balance:.2f}")
            else:
                print(f"Failed to update balance: {result.get('error', '')}")
        
        if args.trade:
            # Parse trade arguments
            symbol = args.trade[0]
            strategy = args.trade[1]
            direction = args.trade[2] if len(args.trade) > 2 else "buy"
            quantity = int(args.trade[3]) if len(args.trade) > 3 else None
            
            result = cc.manual_trade(
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                quantity=quantity
            )
            
            if result["success"]:
                print(f"Trade executed: {symbol} {direction} - Trade ID: {result.get('trade_id', 'Unknown')}")
            else:
                print(f"Trade failed: {result.get('error_message', 'Unknown error')}")
        
        if args.exit:
            result = cc.exit_trade(args.exit)
            if result["success"]:
                print(f"Trade {args.exit} exited successfully")
            else:
                print(f"Failed to exit trade: {result.get('error_message', 'Unknown error')}")
        
        if args.analyze_regime:
            result = cc._handle_analyze_market_regime()
            if result["success"]:
                print(f"Current market regime: {result['regime']}")
                print(f"Details: {json.dumps(result['details'], indent=2)}")
            else:
                print("Failed to analyze market regime")
        
        if args.run_trader is not None:  # Check if the argument was provided at all
            symbols = args.run_trader if args.run_trader else None
            result = cc.run_market_trader(symbols=symbols)
            print(f"Market trader execution: {'Success' if result['success'] else 'Failed'}")
            if result["success"]:
                print(f"Trades executed: {result['result'].get('trades_executed', 0)}")
                print(f"Regime: {result['result'].get('regime', 'Unknown')}")
        
        if args.market_stats:
            result = cc._handle_get_market_stats()
            if result["success"]:
                print("Current Market Statistics:")
                for key, value in result["stats"].items():
                    if key != "timestamp":
                        print(f"  {key}: {value}")
            else:
                print("Failed to get market statistics")
        
        if args.price:
            result = cc._handle_get_price(args.price)
            if result["success"]:
                price_data = result["price_data"]
                print(f"Price data for {args.price}:")
                print(f"  Current price: ${price_data.get('price', 0):.2f}")
                print(f"  Bid: ${price_data.get('bid', 0):.2f}")
                print(f"  Ask: ${price_data.get('ask', 0):.2f}")
                print(f"  Volume: {price_data.get('volume', 0)}")
            else:
                print(f"Failed to get price for {args.price}")
        
        if args.reload:
            result = cc._handle_reload_strategies()
            if result["success"]:
                print(f"Strategies reloaded: {result.get('strategy_count', 0)} strategies loaded")
            else:
                print(f"Failed to reload strategies: {result.get('error', '')}")
        
        if args.recommend_strategies:
            result = cc._handle_get_strategy_recommendations()
            if result["success"]:
                recommendations = result["recommendations"]
                print(f"Market Regime: {recommendations['market_regime']}")
                print(f"Applicable Conditions: {recommendations['applicable_conditions']}")
                print(f"Top Strategies:")
                
                for strategy in recommendations["strategies"][:3]:  # Top 3 strategies
                    print(f"  - {strategy['name']} (Score: {strategy['score']:.1f})")
                    print(f"    Type: {strategy['details'].get('trade_type')}")
                    print(f"    Time Frame: {strategy['details'].get('time_frame')}")
                    print()
            else:
                print(f"Failed to get strategy recommendations: {result.get('error', 'Unknown error')}")
        
        if args.generate_signal is not None:  # None if not provided, empty string if provided with no value
            strategy_name = args.generate_signal if args.generate_signal else None
            result = cc._handle_generate_trade_signal(strategy_name)
            
            if result["success"]:
                signal = result["signal"]
                strategy = result["strategy"]
                
                print(f"Trade Signal for {strategy['name']} in {result['market_regime']} regime:")
                print(f"  Direction: {signal['direction']}")
                print(f"  Type: {signal['trade_type']}")
                
                if signal['trade_type'] == 'equity':
                    print(f"  Symbol: {signal.get('symbol', 'SPY')}")
                    print(f"  Price: ${signal.get('price', 0):.2f}")
                    print(f"  Stop Loss: ${signal.get('stop_loss', 0):.2f}")
                    print(f"  Quantity: {signal.get('quantity', 0)}")
                elif signal['trade_type'] == 'options':
                    print(f"  Symbol: {signal.get('symbol', 'SPY')}")
                    print(f"  Structure: {signal.get('option_structure', '')}")
                    print(f"  Expiration: {signal.get('expiration', '')}")
                    print(f"  Strike Selection: {signal.get('strike_selection', '')}")
                    
                print(f"  Risk Amount: ${signal.get('risk_amount', 0):.2f}")
            else:
                print(f"Failed to generate trade signal: {result.get('error', 'Unknown error')}")
        
        if args.execute_recommendation is not None:
            strategy_name = args.execute_recommendation if args.execute_recommendation else None
            result = cc.execute_strategy_recommendation(strategy_name)
            
            if result["success"]:
                print(f"Trade executed successfully based on {result['strategy']['name']} strategy")
                print(f"  Trade ID: {result['trade_result'].get('trade_id', 'Unknown')}")
                print(f"  Direction: {result['signal']['direction']}")
                print(f"  Type: {result['signal']['trade_type']}")
                print(f"  Market Regime: {result['market_regime']}")
            else:
                print(f"Failed to execute trade: {result.get('error', 'Unknown error')}")
        
        # If no other commands specified, show status
        if not any([args.risk_mode, args.balance, args.trade, args.exit, args.analyze_regime, 
                  args.run_trader is not None, args.market_stats, args.price, args.reload,
                  args.recommend_strategies, args.generate_signal is not None, 
                  args.execute_recommendation is not None]) or args.status:
            report = cc.generate_status_report()
            print(cc.format_status_report(report))
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always stop the command center
        cc.stop() 