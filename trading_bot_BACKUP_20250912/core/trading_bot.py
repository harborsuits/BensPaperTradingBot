"""
Advanced Trading Bot with Psychological Risk Management

This module provides a sophisticated trading bot that integrates psychological risk management,
position sizing, and trade execution. It includes built-in safeguards against emotional trading
and provides comprehensive performance analytics.
"""

import os
import logging
import json
import datetime
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import traceback
import importlib
import threading

# Import components
from trading_bot.risk.psychological_risk import PsychologicalRiskManager
from trading_bot.position_sizing import PositionSizer, SizingMethod
from trading_bot.brokers.trade_executor import TradeExecutor
from trading_bot.strategy_loader import StrategyLoader
from trading_bot.brokers.tradier_client import TradierClient, TradierAPIError
from trading_bot.strategies.micro_strategies import MicroStrategy, MicroMomentumStrategy, MicroBreakoutStrategy
from trading_bot.market_context.context_analyzer import MarketContextAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class TradingMode(str, Enum):
    """Trading mode determines how the bot executes trades"""
    LIVE = "live"            # Real money trading
    PAPER = "paper"          # Paper trading
    BACKTEST = "backtest"    # Backtesting
    SIMULATION = "simulation" # Simulation mode

class TradeResult(Enum):
    """Enum for trade result status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"

class TradeAction(Enum):
    """Enum for trade actions"""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"

class TradeType(Enum):
    """Enum for trade types"""
    ENTRY = "entry"
    EXIT = "exit"
    ADJUSTMENT = "adjustment"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize the trading strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.logger = logging.getLogger(f"strategy.{name}")
        
    def should_enter_trade(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Determine if we should enter a trade based on market data.
        
        Args:
            symbol: The symbol to check
            market_data: Market data for analysis
            
        Returns:
            True if we should enter a trade, False otherwise
        """
        raise NotImplementedError("Subclasses must implement should_enter_trade")
        
    def should_exit_trade(self, symbol: str, position: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """
        Determine if we should exit a trade based on position and market data.
        
        Args:
            symbol: The symbol to check
            position: Current position data
            market_data: Market data for analysis
            
        Returns:
            True if we should exit the trade, False otherwise
        """
        raise NotImplementedError("Subclasses must implement should_exit_trade")
        
    def get_entry_price(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the entry price for a trade.
        
        Args:
            symbol: The symbol to check
            market_data: Market data for analysis
            
        Returns:
            The entry price, or None for market orders
        """
        return None
        
    def get_exit_price(self, symbol: str, position: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the exit price for a trade.
        
        Args:
            symbol: The symbol to check
            position: Current position data
            market_data: Market data for analysis
            
        Returns:
            The exit price, or None for market orders
        """
        return None
        
    def get_stop_loss(self, symbol: str, entry_price: float, market_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the stop loss price for a trade.
        
        Args:
            symbol: The symbol to check
            entry_price: Entry price
            market_data: Market data for analysis
            
        Returns:
            The stop loss price, or None if no stop loss
        """
        return None
        
    def get_take_profit(self, symbol: str, entry_price: float, market_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the take profit price for a trade.
        
        Args:
            symbol: The symbol to check
            entry_price: Entry price
            market_data: Market data for analysis
            
        Returns:
            The take profit price, or None if no take profit
        """
        return None
        
    def get_position_size(self, symbol: str, account_value: float, risk_per_trade: float, 
                         market_data: Dict[str, Any]) -> int:
        """
        Get the position size for a trade.
        
        Args:
            symbol: The symbol to trade
            account_value: Total account value
            risk_per_trade: Risk per trade as a percentage of account value
            market_data: Market data for analysis
            
        Returns:
            The position size in number of shares/contracts
        """
        raise NotImplementedError("Subclasses must implement get_position_size")
        
    def get_trade_action(self, symbol: str, market_data: Dict[str, Any]) -> TradeAction:
        """
        Get the trade action for a new trade.
        
        Args:
            symbol: The symbol to trade
            market_data: Market data for analysis
            
        Returns:
            The trade action (buy, sell, short, cover)
        """
        return TradeAction.BUY

class MovingAverageCrossoverStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, name: str = "MA_Crossover", params: Dict[str, Any] = None):
        """
        Initialize the moving average crossover strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters including:
                - short_period: Short moving average period
                - long_period: Long moving average period
                - risk_pct: Risk percentage per trade
        """
        default_params = {
            "short_period": 20,
            "long_period": 50,
            "risk_pct": 1.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }
        
        # Override defaults with provided params
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
        
    def calculate_moving_averages(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate moving averages from market data.
        
        Args:
            market_data: Dictionary containing historical prices
            
        Returns:
            Dictionary with short_ma and long_ma values
        """
        if "history" not in market_data or "day" not in market_data["history"]:
            self.logger.error("Market data does not contain historical prices")
            return {"short_ma": 0, "long_ma": 0}
            
        history = market_data["history"]["day"]
        
        # Get close prices
        closes = [day["close"] for day in history]
        
        # Calculate moving averages
        short_period = self.params["short_period"]
        long_period = self.params["long_period"]
        
        if len(closes) < long_period:
            self.logger.warning(f"Not enough data for MA calculation. Need {long_period} days, got {len(closes)}.")
            return {"short_ma": 0, "long_ma": 0}
            
        short_ma = sum(closes[-short_period:]) / short_period
        long_ma = sum(closes[-long_period:]) / long_period
        
        return {"short_ma": short_ma, "long_ma": long_ma}
        
    def should_enter_trade(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Check if short MA has crossed above long MA.
        
        Args:
            symbol: The symbol to check
            market_data: Market data containing historical prices
            
        Returns:
            True if we should enter a trade, False otherwise
        """
        # Calculate current MAs
        current = self.calculate_moving_averages(market_data)
        
        # Check if we have enough data
        if current["short_ma"] == 0 or current["long_ma"] == 0:
            return False
            
        # Get yesterday's data by removing last day
        yesterday_data = market_data.copy()
        yesterday_data["history"]["day"] = market_data["history"]["day"][:-1]
        
        # Calculate yesterday's MAs
        yesterday = self.calculate_moving_averages(yesterday_data)
        
        # Check for crossover (short MA crosses above long MA)
        return (current["short_ma"] > current["long_ma"] and 
                yesterday["short_ma"] <= yesterday["long_ma"])
                
    def should_exit_trade(self, symbol: str, position: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """
        Check if short MA has crossed below long MA.
        
        Args:
            symbol: The symbol to check
            position: Current position data
            market_data: Market data containing historical prices
            
        Returns:
            True if we should exit the trade, False otherwise
        """
        # Calculate current MAs
        current = self.calculate_moving_averages(market_data)
        
        # Check if we have enough data
        if current["short_ma"] == 0 or current["long_ma"] == 0:
            return False
            
        # Get yesterday's data by removing last day
        yesterday_data = market_data.copy()
        yesterday_data["history"]["day"] = market_data["history"]["day"][:-1]
        
        # Calculate yesterday's MAs
        yesterday = self.calculate_moving_averages(yesterday_data)
        
        # Check for crossover (short MA crosses below long MA)
        return (current["short_ma"] < current["long_ma"] and 
                yesterday["short_ma"] >= yesterday["long_ma"])
                
    def get_stop_loss(self, symbol: str, entry_price: float, market_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculate stop loss based on risk percentage.
        
        Args:
            symbol: The symbol
            entry_price: Entry price
            market_data: Market data
            
        Returns:
            Stop loss price
        """
        stop_pct = self.params["stop_loss_pct"] / 100.0
        return round(entry_price * (1 - stop_pct), 2)
        
    def get_take_profit(self, symbol: str, entry_price: float, market_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculate take profit based on reward percentage.
        
        Args:
            symbol: The symbol
            entry_price: Entry price
            market_data: Market data
            
        Returns:
            Take profit price
        """
        profit_pct = self.params["take_profit_pct"] / 100.0
        return round(entry_price * (1 + profit_pct), 2)
        
    def get_position_size(self, symbol: str, account_value: float, risk_per_trade: float, 
                         market_data: Dict[str, Any]) -> int:
        """
        Calculate position size based on risk.
        
        Args:
            symbol: The symbol to trade
            account_value: Total account value
            risk_per_trade: Risk per trade as a percentage of account value
            market_data: Market data for analysis
            
        Returns:
            The position size in number of shares/contracts
        """
        # Get latest price
        if "quotes" not in market_data or "quote" not in market_data["quotes"]:
            self.logger.error("Market data does not contain quote information")
            return 0
            
        quote = market_data["quotes"]["quote"]
        price = quote["last"]
        
        # Calculate stop loss
        stop_loss = self.get_stop_loss(symbol, price, market_data)
        
        if stop_loss is None:
            self.logger.error("Could not calculate stop loss")
            return 0
            
        # Calculate dollar risk
        dollar_risk = account_value * (risk_per_trade / 100.0)
        
        # Calculate risk per share
        risk_per_share = price - stop_loss
        
        if risk_per_share <= 0:
            self.logger.error("Invalid risk per share calculation")
            return 0
            
        # Calculate position size
        position_size = int(dollar_risk / risk_per_share)
        
        # Limit position size to reasonable amount
        max_position_value = account_value * 0.2  # Max 20% of account in one position
        max_shares = int(max_position_value / price)
        
        return min(position_size, max_shares)

class Trade:
    """Class to represent a trade"""
    
    def __init__(self, 
                symbol: str, 
                action: TradeAction, 
                quantity: int,
                entry_price: Optional[float] = None,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                strategy_name: str = "manual",
                trade_type: TradeType = TradeType.ENTRY):
        """
        Initialize a trade.
        
        Args:
            symbol: Symbol to trade
            action: Trade action (buy, sell, short, cover)
            quantity: Number of shares/contracts
            entry_price: Entry price for limit orders
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy_name: Name of the strategy that generated this trade
            trade_type: Type of trade (entry, exit, etc.)
        """
        self.symbol = symbol
        self.action = action if isinstance(action, TradeAction) else TradeAction(action)
        self.quantity = quantity
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_name = strategy_name
        self.trade_type = trade_type if isinstance(trade_type, TradeType) else TradeType(trade_type)
        self.timestamp = datetime.now()
        self.order_id = None
        self.result = TradeResult.PENDING
        self.filled_quantity = 0
        self.filled_price = None
        self.commission = 0.0
        self.notes = ""

class TradingBot:
    """
    Main trading bot class that integrates all components
    
    Handles:
    - Account management
    - Strategy management
    - Market context integration
    - Trade execution
    - Position management
    - Logging and reporting
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to config file (JSON or YAML)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Configure logging
        log_level = self.config.get("log_level", "INFO")
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get("log_file", "trading_bot.log")),
                logging.StreamHandler()
            ]
        )
        
        # Initialize Tradier client
        self.client = TradierClient(
            api_key=self.config.get("tradier_api_key"),
            account_id=self.config.get("tradier_account_id"),
            sandbox=self.config.get("use_sandbox", True)
        )
        
        # Initialize trade executor
        self.executor = TradeExecutor(
            tradier_client=self.client,
            max_position_pct=self.config.get("max_position_pct", 0.05),
            max_risk_pct=self.config.get("max_risk_pct", 0.01),
            order_type=self.config.get("default_order_type", "market"),
            order_duration=self.config.get("default_order_duration", "day")
        )
        
        # Initialize market context analyzer
        self.context_analyzer = MarketContextAnalyzer({
            "MARKETAUX_API_KEY": self.config.get("marketaux_api_key"),
            "OPENAI_API_KEY": self.config.get("openai_api_key"),
            "CACHE_EXPIRY_MINUTES": self.config.get("context_cache_minutes", 30)
        })
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Active symbols to monitor
        self.watchlist = self.config.get("watchlist", [])
        
        # Bot state
        self.is_running = False
        self.last_check_time = None
        self.market_context = None
        self.market_context_time = None
        
        logger.info(f"Trading bot initialized with {len(self.strategies)} strategies "
                   f"and {len(self.watchlist)} symbols in watchlist")
        
        # Market data cache
        self.market_data_cache = {}
        self.market_data_expiry = {}
        
        # Trade history
        self.trade_history = []
        
        # Active orders
        self.active_orders = {}
        
        # Thread for updating market data
        self.market_data_thread = None
        self.keep_running = False
        
        # Track account data
        self.account_info = {}
        self.positions = {}
        
        # Trading enabled flag
        self.trading_enabled = self.config.get("trading_enabled", False)
        
        self.logger.info(f"Initialized trading bot with trading {'enabled' if self.trading_enabled else 'disabled'}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary with configuration
        """
        default_config = {
            "tradier_api_key": os.environ.get("TRADIER_API_KEY", ""),
            "tradier_account_id": os.environ.get("TRADIER_ACCOUNT_ID", ""),
            "marketaux_api_key": os.environ.get("MARKETAUX_API_KEY", "7PgROm6BE4m6ejBW8unmZnnYS6kIygu5lwzpfd9K"),
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "use_sandbox": os.environ.get("USE_SANDBOX", "true").lower() == "true",
            "log_level": os.environ.get("LOG_LEVEL", "INFO"),
            "log_file": "trading_bot.log",
            "check_interval_seconds": 60,
            "max_position_pct": 0.05,
            "max_risk_pct": 0.01,
            "default_order_type": "market",
            "default_order_duration": "day",
            "context_refresh_minutes": 60,
            "watchlist": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
            "enabled_strategies": ["micro_momentum", "micro_breakout"],
            "trade_during_market_hours_only": True,
            "trading_enabled": False
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    # Assume YAML
                    import yaml
                    config = yaml.safe_load(f)
            
            # Merge with defaults
            merged_config = default_config.copy()
            merged_config.update(config)
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            return default_config
    
    def _initialize_strategies(self) -> Dict[str, MicroStrategy]:
        """
        Initialize trading strategies
        
        Returns:
            Dictionary of strategy name -> strategy object
        """
        strategies = {}
        
        # Get enabled strategies
        enabled_strategy_names = self.config.get("enabled_strategies", ["micro_momentum"])
        
        # Add micro momentum strategy
        if "micro_momentum" in enabled_strategy_names:
            strategies["micro_momentum"] = MicroMomentumStrategy(
                client=self.client,
                max_risk_per_trade=self.config.get("max_risk_pct", 0.01),
                max_risk_total=self.config.get("max_risk_total", 0.05),
                min_rrr=self.config.get("min_risk_reward", 2.0),
                commission_per_trade=self.config.get("commission_per_trade", 1.0)
            )
        
        # Add micro breakout strategy
        if "micro_breakout" in enabled_strategy_names:
            strategies["micro_breakout"] = MicroBreakoutStrategy(
                client=self.client,
                max_risk_per_trade=self.config.get("max_risk_pct", 0.01),
                max_risk_total=self.config.get("max_risk_total", 0.05),
                min_rrr=self.config.get("min_risk_reward", 2.0),
                commission_per_trade=self.config.get("commission_per_trade", 1.0)
            )
        
        return strategies
    
    def update_market_context(self, force_refresh: bool = False) -> Dict:
        """
        Update market context
        
        Args:
            force_refresh: Force refresh of context
            
        Returns:
            Market context data
        """
        try:
            # Check if we need to refresh
            now = datetime.now()
            context_age_minutes = 0
            if self.market_context_time:
                context_age_minutes = (now - self.market_context_time).total_seconds() / 60
            
            # Refresh if needed
            if (force_refresh or 
                not self.market_context or 
                context_age_minutes > self.config.get("context_refresh_minutes", 60)):
                
                logger.info("Updating market context...")
                
                # Get new context
                self.market_context = self.context_analyzer.get_market_context(force_refresh=True)
                self.market_context_time = now
                
                logger.info(f"Market context updated: {self.market_context.get('bias', 'unknown')} "
                           f"(confidence: {self.market_context.get('confidence', 0):.2f})")
            else:
                logger.debug(f"Using cached market context ({context_age_minutes:.1f} min old)")
            
            return self.market_context
            
        except Exception as e:
            logger.error(f"Error updating market context: {str(e)}")
            if not self.market_context:
                # Create default neutral context if none exists
                self.market_context = {
                    "bias": "neutral",
                    "confidence": 0.5,
                    "timestamp": datetime.now().isoformat()
                }
            
            return self.market_context
    
    def check_for_signals(self) -> List[Dict]:
        """
        Check all strategies for trading signals
        
        Returns:
            List of signal dictionaries
        """
        signals = []
        market_context = self.update_market_context()
        market_bias = market_context.get("bias", "neutral")
        
        # Check if market is open if configured to trade only during market hours
        if self.config.get("trade_during_market_hours_only", True):
            is_market_open = self.client.is_market_open()
            if not is_market_open:
                logger.info("Market is closed, skipping signal check")
                return []
        
        # Cycle through all symbols in watchlist
        for symbol in self.watchlist:
            try:
                # Get historical data (reused for all strategies)
                hist_data = None
                
                # Check each strategy for signals
                for strategy_name, strategy in self.strategies.items():
                    try:
                        # Get historical data if needed
                        if hist_data is None:
                            # Get data from Tradier
                            history_response = self.client.get_historical_data(
                                symbol=symbol,
                                interval="daily",
                                days_back=30
                            )
                            
                            # Convert to DataFrame
                            if "day" in history_response and history_response["day"]:
                                data = history_response["day"]
                                df = pd.DataFrame(data)
                                
                                # Convert date string to datetime
                                df['date'] = pd.to_datetime(df['date'])
                                df.set_index('date', inplace=True)
                                hist_data = df
                            else:
                                logger.warning(f"No historical data for {symbol}")
                                continue
                        
                        # Check for signal
                        signal = strategy.check_signal(symbol, hist_data)
                        
                        # If we have a signal, adjust it for market context
                        if signal:
                            adjusted_signal = strategy.adjust_for_market_bias(signal, market_bias)
                            signals.append(adjusted_signal)
                            
                            logger.info(f"Signal found for {symbol} using {strategy_name} strategy")
                    
                    except Exception as strategy_error:
                        logger.error(f"Error checking {strategy_name} strategy for {symbol}: {str(strategy_error)}")
                        continue
            
            except Exception as symbol_error:
                logger.error(f"Error processing symbol {symbol}: {str(symbol_error)}")
                continue
        
        return signals
    
    def execute_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Execute trading signals
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for signal in signals:
            try:
                # Extract signal details
                symbol = signal.get("symbol")
                side = signal.get("side")
                entry_price = signal.get("entry_price")
                stop_price = signal.get("stop_price")
                target_price = signal.get("target_price")
                shares = signal.get("shares")
                risk_pct = signal.get("risk_pct")
                strategy_name = signal.get("strategy")
                metadata = signal.get("metadata", {})
                
                # Execute the trade
                trade = self.executor.execute_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    shares=shares,
                    risk_pct=risk_pct,
                    strategy_name=strategy_name,
                    metadata=metadata
                )
                
                executed_trades.append(trade)
                
                logger.info(f"Executed {side} trade for {shares} shares of {symbol} "
                           f"at ${entry_price:.2f} (ID: {trade.get('id')})")
                
            except Exception as e:
                logger.error(f"Error executing signal for {signal.get('symbol')}: {str(e)}")
                continue
        
        return executed_trades
    
    def manage_positions(self) -> None:
        """
        Manage existing positions (check stops, targets, etc.)
        """
        # Get open trades
        open_trades = self.executor.get_open_trades()
        
        if not open_trades:
            return
            
        logger.info(f"Managing {len(open_trades)} open positions")
        
        for trade in open_trades:
            try:
                trade_id = trade.get("id")
                symbol = trade.get("symbol")
                
                # Get current price
                quote = self.client.get_quote(symbol)
                current_price = float(quote.get("last", 0))
                
                if current_price <= 0:
                    logger.warning(f"Could not get valid price for {symbol}, skipping position check")
                    continue
                
                # Exit logic:
                exit_triggered = False
                exit_reason = None
                
                # 1. Stop loss hit
                if trade.get("side") == "buy" and current_price <= trade.get("stop_price", 0):
                    exit_triggered = True
                    exit_reason = "stop_loss"
                elif trade.get("side") == "sell" and current_price >= trade.get("stop_price", float('inf')):
                    exit_triggered = True
                    exit_reason = "stop_loss"
                
                # 2. Target hit
                if trade.get("side") == "buy" and current_price >= trade.get("target_price", float('inf')):
                    exit_triggered = True
                    exit_reason = "target_hit"
                elif trade.get("side") == "sell" and current_price <= trade.get("target_price", 0):
                    exit_triggered = True
                    exit_reason = "target_hit"
                
                # If exit triggered, close the position
                if exit_triggered:
                    logger.info(f"Exit triggered for {symbol}: {exit_reason} (Current: ${current_price:.2f})")
                    
                    # Exit the trade
                    self.executor.exit_trade(
                        trade_id=trade_id,
                        price=current_price,
                        exit_reason=exit_reason
                    )
                
            except Exception as e:
                logger.error(f"Error managing position for {trade.get('symbol')}: {str(e)}")
                continue
    
    def run_once(self) -> None:
        """Run one iteration of the bot"""
        try:
            # Update market context
            market_context = self.update_market_context()
            
            # Check for new signals
            signals = self.check_for_signals()
            
            # Execute any new signals
            if signals:
                executed_trades = self.execute_signals(signals)
                logger.info(f"Executed {len(executed_trades)} trades")
            
            # Manage existing positions
            self.manage_positions()
            
            # Update last check time
            self.last_check_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in trading bot run: {str(e)}")
            logger.error(traceback.format_exc())
    
    def run(self) -> None:
        """Run the trading bot in a loop"""
        self.is_running = True
        logger.info("Starting trading bot...")
        
        check_interval = self.config.get("check_interval_seconds", 60)
        
        try:
            while self.is_running:
                # Run once
                self.run_once()
                
                # Sleep until next check
                logger.info(f"Sleeping for {check_interval} seconds")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.is_running = False
            
        except Exception as e:
            logger.error(f"Unexpected error in trading bot: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_running = False
            
        logger.info("Trading bot stopped")
    
    def stop(self) -> None:
        """Stop the trading bot"""
        self.is_running = False
        logger.info("Trading bot stopping...")
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary
        
        Returns:
            Dictionary with account summary
        """
        try:
            # Get account summary from executor
            account_summary = self.client.get_account_summary()
            
            # Add bot state
            account_summary["bot_state"] = {
                "is_running": self.is_running,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "market_context": {
                    "bias": self.market_context.get("bias", "unknown") if self.market_context else "unknown",
                    "confidence": self.market_context.get("confidence", 0) if self.market_context else 0,
                    "updated_at": self.market_context_time.isoformat() if self.market_context_time else None
                },
                "active_strategies": list(self.strategies.keys()),
                "watchlist_count": len(self.watchlist)
            }
            
            return account_summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return {
                "error": str(e),
                "is_running": self.is_running
            }
    
    def get_bot_status(self) -> Dict:
        """
        Get trading bot status
        
        Returns:
            Dictionary with bot status
        """
        try:
            # Get open trades
            open_trades = self.executor.get_open_trades()
            
            # Get account summary
            account_summary = self.get_account_summary()
            
            # Get trading metrics
            trading_metrics = self.executor.calculate_metrics()
            
            return {
                "account": account_summary,
                "open_positions": open_trades,
                "metrics": trading_metrics,
                "bot_status": {
                    "is_running": self.is_running,
                    "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
                    "market_context": {
                        "bias": self.market_context.get("bias", "unknown") if self.market_context else "unknown",
                        "confidence": self.market_context.get("confidence", 0) if self.market_context else 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting bot status: {str(e)}")
            return {
                "error": str(e),
                "is_running": self.is_running
            }
    
    def update_market_data(self, symbols: List[str] = None, force: bool = False):
        """
        Update market data for specified symbols or watchlist.
        
        Args:
            symbols: List of symbols to update (None for watchlist)
            force: Force update even if cache is fresh
        """
        if symbols is None:
            symbols = self.config.get("watchlist", [])
            
        if not symbols:
            self.logger.warning("No symbols to update market data for")
            return
            
        now = datetime.now()
        symbols_to_update = []
        
        # Check which symbols need updates
        for symbol in symbols:
            if (force or symbol not in self.market_data_cache or 
                symbol not in self.market_data_expiry or 
                now >= self.market_data_expiry[symbol]):
                symbols_to_update.append(symbol)
                
        if not symbols_to_update:
            return
            
        self.logger.debug(f"Updating market data for {len(symbols_to_update)} symbols")
        
        try:
            # Get quotes
            quotes = self.client.get_quotes(symbols_to_update)
            
            # Get historical data for each symbol
            for symbol in symbols_to_update:
                # Get historical data
                history = self.client.get_historical_quotes(
                    symbol, 
                    interval="daily",
                    start_date=(datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
                )
                
                # Store in cache
                self.market_data_cache[symbol] = {
                    "quotes": {
                        "quote": next(
                            (q for q in quotes.get("quotes", {}).get("quote", []) 
                             if q.get("symbol") == symbol), 
                            {}
                        )
                    },
                    "history": history.get("history", {})
                }
                
                # Set expiry time (15 minutes)
                refresh_minutes = self.config.get("context_refresh_minutes", 15)
                self.market_data_expiry[symbol] = now + timedelta(minutes=refresh_minutes)
                
        except Exception as e:
            self.logger.error(f"Failed to update market data: {str(e)}")
            
    def update_account_info(self):
        """Update account information and positions"""
        try:
            # Get account balances
            balances = self.client.get_account_balances()
            self.account_info = balances.get("balances", {})
            
            # Get positions
            positions_data = self.client.get_positions()
            
            # Process positions
            positions = positions_data.get("positions", {}).get("position", [])
            
            # If only one position is returned, it's not in a list
            if positions and not isinstance(positions, list):
                positions = [positions]
                
            # Update positions dictionary
            self.positions = {}
            for position in positions:
                symbol = position.get("symbol")
                if symbol:
                    self.positions[symbol] = position
                    
            self.logger.debug(f"Updated account info. Balance: ${self.account_info.get('total_equity', 0)}, " +
                            f"Positions: {len(self.positions)}")
                            
        except Exception as e:
            self.logger.error(f"Failed to update account info: {str(e)}")
            
    def check_for_signals(self):
        """Check for entry and exit signals across all strategies and symbols"""
        if not self.trading_enabled:
            self.logger.info("Trading is disabled, skipping signal check")
            return
            
        watchlist = self.config.get("watchlist", [])
        
        # First update market data
        self.update_market_data(watchlist)
        
        # Update account info
        self.update_account_info()
        
        # Check for exit signals first (for existing positions)
        for symbol, position in self.positions.items():
            if symbol in self.market_data_cache:
                market_data = self.market_data_cache[symbol]
                
                for strategy_name, strategy in self.strategies.items():
                    if strategy.should_exit_trade(symbol, position, market_data):
                        self.logger.info(f"Exit signal for {symbol} from {strategy_name}")
                        
                        # Create exit trade
                        self._handle_exit_signal(symbol, position, strategy, market_data)
        
        # Check for entry signals
        for symbol in watchlist:
            # Skip if we already have a position
            if symbol in self.positions:
                continue
                
            if symbol in self.market_data_cache:
                market_data = self.market_data_cache[symbol]
                
                for strategy_name, strategy in self.strategies.items():
                    if strategy.should_enter_trade(symbol, market_data):
                        self.logger.info(f"Entry signal for {symbol} from {strategy_name}")
                        
                        # Create entry trade
                        self._handle_entry_signal(symbol, strategy, market_data)
                        
                        # Only take one entry signal per symbol
                        break
                        
    def _handle_entry_signal(self, symbol: str, strategy: TradingStrategy, market_data: Dict[str, Any]):
        """
        Handle an entry signal from a strategy.
        
        Args:
            symbol: Symbol to trade
            strategy: Strategy that generated the signal
            market_data: Market data for analysis
        """
        account_value = self.account_info.get("total_equity", 0)
        risk_per_trade = self.config.get("max_risk_pct", 1.0)
        
        # Get position size
        quantity = strategy.get_position_size(symbol, account_value, risk_per_trade, market_data)
        
        if quantity <= 0:
            self.logger.warning(f"Position size calculation resulted in {quantity} shares, skipping trade")
            return
            
        # Get entry price (None for market orders)
        entry_price = strategy.get_entry_price(symbol, market_data)
        
        # Get trade action
        action = strategy.get_trade_action(symbol, market_data)
        
        # Get stop loss and take profit
        if entry_price is None:
            # Use current price for stop loss calculation on market orders
            current_price = market_data["quotes"]["quote"]["last"]
            stop_loss = strategy.get_stop_loss(symbol, current_price, market_data)
            take_profit = strategy.get_take_profit(symbol, current_price, market_data)
        else:
            stop_loss = strategy.get_stop_loss(symbol, entry_price, market_data)
            take_profit = strategy.get_take_profit(symbol, entry_price, market_data)
        
        # Create trade object
        trade = Trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy.name,
            trade_type=TradeType.ENTRY
        )
        
        # Execute the trade
        self._execute_trade(trade)
        
    def _handle_exit_signal(self, symbol: str, position: Dict[str, Any], strategy: TradingStrategy, 
                           market_data: Dict[str, Any]):
        """
        Handle an exit signal from a strategy.
        
        Args:
            symbol: Symbol to trade
            position: Current position data
            strategy: Strategy that generated the signal
            market_data: Market data for analysis
        """
        # Get position details
        quantity = int(position.get("quantity", 0))
        side = position.get("side", "")
        
        if quantity <= 0:
            self.logger.warning(f"Position has {quantity} shares, skipping exit")
            return
            
        # Determine exit action based on position side
        if side.lower() == "long":
            action = TradeAction.SELL
        elif side.lower() == "short":
            action = TradeAction.COVER
        else:
            self.logger.error(f"Unknown position side: {side}")
            return
            
        # Get exit price (None for market orders)
        exit_price = strategy.get_exit_price(symbol, position, market_data)
        
        # Create trade object
        trade = Trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_price=exit_price,  # For limit orders
            strategy_name=strategy.name,
            trade_type=TradeType.EXIT
        )
        
        # Execute the trade
        self._execute_trade(trade)
        
    def _execute_trade(self, trade: Trade):
        """
        Execute a trade using the Tradier API.
        
        Args:
            trade: Trade object with details
        """
        if not self.trading_enabled:
            self.logger.info(f"Trading disabled, would have executed: {trade.action.value} {trade.quantity} {trade.symbol}")
            return
            
        # Map trade action to order side
        action_to_side = {
            TradeAction.BUY: "buy",
            TradeAction.SELL: "sell",
            TradeAction.SHORT: "sell_short",
            TradeAction.COVER: "buy_to_cover"
        }
        
        order_side = action_to_side[trade.action]
        
        # Determine order type
        if trade.entry_price is not None:
            order_type = OrderType.LIMIT
            price = trade.entry_price
        else:
            order_type = OrderType.MARKET
            price = None
            
        # Get order duration
        order_duration = self.config.get("default_order_duration", "day")
        
        try:
            # Place the order
            response = self.client.place_equity_order(
                symbol=trade.symbol,
                side=order_side,
                quantity=trade.quantity,
                order_type=order_type,
                duration=order_duration,
                price=price
            )
            
            # Get order ID and update trade object
            if response and "order" in response and "id" in response["order"]:
                order_id = response["order"]["id"]
                trade.order_id = order_id
                
                # Record trade in history
                self.trade_history.append(trade)
                
                # Add to active orders
                self.active_orders[order_id] = trade
                
                self.logger.info(f"Placed order: {order_side} {trade.quantity} {trade.symbol}, " +
                               f"Order ID: {order_id}")
                               
                # If we have stop loss, place it as a separate order if the main order fills
                if trade.stop_loss and trade.action in [TradeAction.BUY, TradeAction.COVER]:
                    # We'll check for fills later and place stop orders
                    pass
                    
            else:
                self.logger.error(f"Failed to place order: {response}")
                trade.result = TradeResult.FAILED
                trade.notes = "Failed to place order"
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            trade.result = TradeResult.FAILED
            trade.notes = f"Error: {str(e)}"
            self.trade_history.append(trade)
            
    def check_order_status(self):
        """Check status of active orders and update trades"""
        order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            try:
                # Get order status
                response = self.client.get_order(order_id)
                
                if "order" in response:
                    order = response["order"]
                    status = order.get("status")
                    
                    trade = self.active_orders[order_id]
                    
                    if status == "filled":
                        # Order is filled
                        trade.result = TradeResult.SUCCESS
                        trade.filled_quantity = int(order.get("quantity", 0))
                        trade.filled_price = float(order.get("avg_fill_price", 0))
                        
                        # Remove from active orders
                        del self.active_orders[order_id]
                        
                        self.logger.info(f"Order {order_id} filled: {trade.filled_quantity} shares " +
                                      f"of {trade.symbol} at ${trade.filled_price}")
                                      
                        # If this was an entry with stop loss, place stop loss order
                        if (trade.stop_loss and trade.trade_type == TradeType.ENTRY and 
                            trade.action in [TradeAction.BUY, TradeAction.COVER]):
                            self._place_stop_loss(trade)
                            
                    elif status == "rejected" or status == "expired" or status == "canceled":
                        # Order failed
                        trade.result = TradeResult.FAILED
                        trade.notes = f"Order {status}: {order.get('reason', '')}"
                        
                        # Remove from active orders
                        del self.active_orders[order_id]
                        
                        self.logger.warning(f"Order {order_id} {status}: {trade.notes}")
                        
                    elif status == "partially_filled":
                        # Partially filled
                        trade.result = TradeResult.PARTIAL
                        trade.filled_quantity = int(order.get("filled_quantity", 0))
                        trade.filled_price = float(order.get("avg_fill_price", 0))
                        
                        self.logger.info(f"Order {order_id} partially filled: {trade.filled_quantity}/{trade.quantity} " +
                                      f"shares of {trade.symbol} at ${trade.filled_price}")
                
            except Exception as e:
                self.logger.error(f"Error checking order {order_id}: {str(e)}")
                
    def _place_stop_loss(self, entry_trade: Trade):
        """
        Place a stop loss order for an entry trade.
        
        Args:
            entry_trade: The entry trade that was filled
        """
        if not self.trading_enabled:
            self.logger.info(f"Trading disabled, would have placed stop loss for {entry_trade.symbol} at ${entry_trade.stop_loss}")
            return
            
        if not entry_trade.stop_loss:
            return
            
        # Create stop loss trade
        action = TradeAction.SELL if entry_trade.action == TradeAction.BUY else TradeAction.COVER
        
        stop_trade = Trade(
            symbol=entry_trade.symbol,
            action=action,
            quantity=entry_trade.filled_quantity,
            entry_price=None,  # Will be set by stop price
            strategy_name=entry_trade.strategy_name,
            trade_type=TradeType.STOP_LOSS
        )
        
        # Place stop order
        try:
            response = self.client.place_equity_order(
                symbol=stop_trade.symbol,
                side=action.value,
                quantity=stop_trade.quantity,
                order_type=OrderType.STOP,
                duration=self.config.get("default_order_duration", "day"),
                stop=entry_trade.stop_loss
            )
            
            # Get order ID and update trade object
            if response and "order" in response and "id" in response["order"]:
                order_id = response["order"]["id"]
                stop_trade.order_id = order_id
                
                # Record trade in history
                self.trade_history.append(stop_trade)
                
                # Add to active orders
                self.active_orders[order_id] = stop_trade
                
                self.logger.info(f"Placed stop loss order for {stop_trade.symbol} at ${entry_trade.stop_loss}, " +
                               f"Order ID: {order_id}")
                               
            else:
                self.logger.error(f"Failed to place stop loss order: {response}")
                
        except Exception as e:
            self.logger.error(f"Error placing stop loss order: {str(e)}")
            
    def market_data_worker(self):
        """Background worker to update market data"""
        while self.keep_running:
            try:
                # Update market data for watchlist
                self.update_market_data()
                
                # Update account info
                self.update_account_info()
                
                # Check orders
                self.check_order_status()
                
            except Exception as e:
                self.logger.error(f"Error in market data worker: {str(e)}")
                
            # Sleep until next check
            time.sleep(self.config.get("check_interval_seconds", 60))
            
    def start(self):
        """Start the trading bot"""
        self.keep_running = True
        
        # Start market data thread
        self.market_data_thread = threading.Thread(target=self.market_data_worker)
        self.market_data_thread.daemon = True
        self.market_data_thread.start()
        
        self.logger.info("Trading bot started")
        
    def stop(self):
        """Stop the trading bot"""
        self.keep_running = False
        
        if self.market_data_thread:
            self.market_data_thread.join(timeout=5.0)
            
        self.logger.info("Trading bot stopped")
        
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the account status.
        
        Returns:
            Dictionary with account summary
        """
        return {
            "account_id": self.client.account_id,
            "balance": self.account_info.get("total_equity", 0),
            "buying_power": self.account_info.get("buying_power", 0),
            "cash": self.account_info.get("cash", 0),
            "positions": len(self.positions),
            "position_value": self.account_info.get("long_market_value", 0),
            "active_orders": len(self.active_orders),
            "trading_enabled": self.trading_enabled,
            "strategies": list(self.strategies.keys()),
            "watchlist": self.config.get("watchlist", [])
        }
        
    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of current positions.
        
        Returns:
            List of position summaries
        """
        result = []
        
        for symbol, position in self.positions.items():
            # Get latest quote
            self.update_market_data([symbol])
            market_data = self.market_data_cache.get(symbol, {})
            quote = market_data.get("quotes", {}).get("quote", {})
            
            current_price = quote.get("last", 0)
            entry_price = float(position.get("cost_basis", 0))
            quantity = int(position.get("quantity", 0))
            
            # Calculate P&L
            pnl = (current_price - entry_price) * quantity
            pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
            
            result.append({
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "value": current_price * quantity,
                "side": position.get("side", "")
            })
            
        return result
        
    def enable_trading(self, enabled: bool = True):
        """
        Enable or disable trading.
        
        Args:
            enabled: Whether trading should be enabled
        """
        self.trading_enabled = enabled
        self.logger.info(f"Trading {'enabled' if enabled else 'disabled'}")
        

# Example usage
if __name__ == "__main__":
    # Create bot
    bot = TradingBot()
    
    # Start bot
    bot.start()
    
    try:
        # Print account summary
        summary = bot.get_account_summary()
        print(f"Account: {summary['account_id']}")
        print(f"Balance: ${summary['balance']}")
        print(f"Positions: {summary['positions']}")
        print(f"Trading enabled: {summary['trading_enabled']}")
        print(f"Strategies: {', '.join(summary['strategies'])}")
        print(f"Watchlist: {', '.join(summary['watchlist'])}")
        
        # Check for signals once
        bot.check_for_signals()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping trading bot...")
        
    finally:
        # Stop bot
        bot.stop() 