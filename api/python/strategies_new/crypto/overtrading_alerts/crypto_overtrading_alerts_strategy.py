#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Overtrading Alerts Strategy

This meta-strategy monitors trading frequency, drawdowns, and risk metrics
to prevent excessive trading and protect capital. It works alongside other
active strategies to provide risk guardrails and trading hygiene.

Key features:
1. Trading frequency monitoring (per symbol, timeframe, and strategy)
2. Drawdown tracking (per trade, daily, and overall)
3. Risk concentration alerts (exposure to correlated assets)
4. Performance metrics tracking (win rate, profit factor)
5. Psychological alerts (revenge trading, FOMO detection)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import math
from enum import Enum
from collections import defaultdict, deque

# Import base strategy
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.factory.strategy_factory import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

# Define alert severity levels
class AlertSeverity(Enum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@register_strategy(
    name="CryptoOvertradingAlertsStrategy",
    category="crypto",
    description="A meta-strategy that monitors trading activity to prevent overtrading and excessive risk",
    parameters={
        # Monitoring scope
        "monitoring_symbols": {
            "type": "list",
            "default": ["ALL"],  # "ALL" or list of specific symbols
            "description": "Symbols to monitor for overtrading (ALL for all traded symbols)"
        },
        "monitoring_strategies": {
            "type": "list",
            "default": ["ALL"],  # "ALL" or list of strategy names
            "description": "Strategies to monitor (ALL for all active strategies)"
        },
        
        # Trading frequency thresholds
        "max_trades_per_day": {
            "type": "int",
            "default": 10,
            "description": "Maximum number of trades per day across all monitored symbols"
        },
        "max_trades_per_symbol_per_day": {
            "type": "int",
            "default": 5,
            "description": "Maximum number of trades per symbol per day"
        },
        "min_time_between_trades": {
            "type": "int",
            "default": 15,
            "description": "Minimum time between trades in minutes for same symbol"
        },
        "weekend_reduction_factor": {
            "type": "float",
            "default": 0.5,
            "description": "Factor to reduce max trades threshold during weekends (crypto markets)"
        },
        
        # Drawdown thresholds
        "max_daily_drawdown_pct": {
            "type": "float",
            "default": 3.0,
            "description": "Maximum allowed daily drawdown in percent before alerts/actions"
        },
        "max_account_drawdown_pct": {
            "type": "float",
            "default": 10.0,
            "description": "Maximum allowed account drawdown in percent before alerts/actions"
        },
        "max_consecutive_losses": {
            "type": "int",
            "default": 3,
            "description": "Maximum allowed consecutive losing trades before alerts"
        },
        
        # Risk thresholds
        "max_position_size_pct": {
            "type": "float",
            "default": 15.0,
            "description": "Maximum position size as percentage of account"
        },
        "max_correlated_exposure_pct": {
            "type": "float",
            "default": 25.0,
            "description": "Maximum exposure to correlated assets (e.g., same sector)"
        },
        "max_open_positions": {
            "type": "int",
            "default": 8,
            "description": "Maximum number of open positions at any time"
        },
        
        # Performance thresholds
        "min_win_rate_threshold": {
            "type": "float",
            "default": 0.4,
            "description": "Minimum win rate before triggering alerts"
        },
        "min_profit_factor": {
            "type": "float",
            "default": 1.2,
            "description": "Minimum profit factor (gross profits / gross losses) before alerts"
        },
        
        # Recovery settings
        "trading_cooldown_minutes": {
            "type": "int",
            "default": 120,
            "description": "Cooldown period in minutes after hitting critical thresholds"
        },
        "reduced_size_after_drawdown": {
            "type": "float",
            "default": 0.5,
            "description": "Position size multiplier after hitting drawdown thresholds"
        },
        "gradual_recovery_days": {
            "type": "int",
            "default": 3,
            "description": "Number of days to gradually recover to full position size after drawdown"
        },
        
        # Alert notification settings
        "notification_level": {
            "type": "str",
            "default": "medium",
            "enum": ["info", "low", "medium", "high", "critical"],
            "description": "Minimum severity level of alerts to notify"
        },
        "auto_reduce_positions": {
            "type": "bool",
            "default": True,
            "description": "Whether to automatically reduce positions when critical thresholds are hit"
        },
        "auto_pause_trading": {
            "type": "bool",
            "default": True,
            "description": "Whether to automatically pause trading when critical thresholds are hit"
        },
        
        # Psychological pattern detection
        "detect_revenge_trading": {
            "type": "bool",
            "default": True,
            "description": "Whether to detect and alert on revenge trading patterns"
        },
        "detect_fomo": {
            "type": "bool",
            "default": True,
            "description": "Whether to detect and alert on FOMO (Fear Of Missing Out) patterns"
        },
        
        # Market condition adaptation
        "adapt_to_volatility": {
            "type": "bool",
            "default": True,
            "description": "Whether to adapt thresholds based on market volatility"
        },
        "high_volatility_reduction": {
            "type": "float",
            "default": 0.7,
            "description": "Factor to reduce max trades during high volatility"
        }
    }
)
class CryptoOvertradingAlertsStrategy(CryptoBaseStrategy):
    """
    A meta-strategy that monitors trading activity to prevent overtrading and excessive risk.
    Works alongside other active strategies to provide risk guardrails and trading hygiene.
    """
    
    def __init__(self, session: CryptoSession, parameters: Dict[str, Any] = None):
        """
        Initialize the Crypto Overtrading Alerts Strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize trade tracking
        self.trades_today = 0  # Total trades today
        self.trades_by_symbol = defaultdict(int)  # Trades per symbol today
        self.trades_by_strategy = defaultdict(int)  # Trades per strategy today
        self.last_trade_time_by_symbol = {}  # Last trade time per symbol
        self.recent_trades = deque(maxlen=100)  # Store recent trades for analysis
        
        # Initialize drawdown tracking
        self.consecutive_losses = 0
        self.daily_high_equity = self.session.get_account_balance()
        self.daily_low_equity = self.daily_high_equity
        self.account_high_watermark = self.daily_high_equity
        self.daily_drawdown_pct = 0.0
        self.account_drawdown_pct = 0.0
        
        # Initialize performance tracking
        self.trades_win_count = 0
        self.trades_loss_count = 0
        self.gross_profits = 0.0
        self.gross_losses = 0.0
        
        # Initialize cooldown/recovery state
        self.trading_paused = False
        self.trading_paused_until = None
        self.position_size_multiplier = 1.0
        self.recovery_start_time = None
        
        # Initialize alert state
        self.active_alerts = {}  # alert_id -> alert_details
        self.alert_history = []  # List of past alerts (limited to last 100)
        self.next_alert_id = 1
        
        # Market state tracking
        self.is_high_volatility = False
        self.volatility_by_symbol = {}
        
        # Initialize correlation matrix for assets
        self.correlation_matrix = None
        self.sector_exposure = defaultdict(float)
        
        # Date tracking for daily reset
        self.current_date = datetime.now().date()
        
        # Register event handlers
        self._register_events()
        
        logger.info(f"Initialized Crypto Overtrading Alerts Strategy monitoring {len(self.parameters['monitoring_symbols'])} symbols")
    
    def _register_events(self) -> None:
        """
        Register for relevant events for the strategy.
        """
        # Register for trade related events
        self.event_bus.subscribe("TRADE_EXECUTED", self._on_trade_executed)
        self.event_bus.subscribe("TRADE_CLOSED", self._on_trade_closed)
        self.event_bus.subscribe("POSITION_UPDATE", self._on_position_update)
        self.event_bus.subscribe("ORDER_UPDATE", self._on_order_update)
        
        # Register for account events
        self.event_bus.subscribe("ACCOUNT_UPDATE", self._on_account_update)
        
        # Register for strategy signal events
        self.event_bus.subscribe("SIGNAL", self._on_signal_event)
        
        # Register for market data events
        self.event_bus.subscribe("MARKET_DATA_UPDATE", self._on_market_data_update)
        
        # Register for volatility update events
        self.event_bus.subscribe("VOLATILITY_UPDATE", self._on_volatility_update)
        
        # Register for timeframe events for periodic checks
        self.event_bus.subscribe("TIMEFRAME_1m", self._on_minute_update)
        self.event_bus.subscribe("TIMEFRAME_1h", self._on_hourly_update)
        self.event_bus.subscribe("TIMEFRAME_1d", self._on_daily_update)
    
    def _on_trade_executed(self, event: Event) -> None:
        """
        Handle trade executed events to track trading frequency.
        
        Args:
            event: Trade executed event
        """
        if not event.data:
            return
            
        # Extract trade data
        trade = event.data
        symbol = trade.get("symbol")
        strategy = trade.get("metadata", {}).get("strategy")
        
        # Check if this symbol and strategy are monitored
        if not self._is_monitored_symbol(symbol) or not self._is_monitored_strategy(strategy):
            return
        
        # Update trade counters
        self.trades_today += 1
        self.trades_by_symbol[symbol] += 1
        self.trades_by_strategy[strategy] += 1
        self.last_trade_time_by_symbol[symbol] = datetime.now()
        
        # Store trade info for analysis
        trade_info = {
            "symbol": symbol,
            "strategy": strategy,
            "direction": trade.get("direction"),
            "quantity": trade.get("quantity"),
            "price": trade.get("price"),
            "timestamp": datetime.now(),
            "id": trade.get("id")
        }
        self.recent_trades.append(trade_info)
        
        # Check for overtrading
        self._check_overtrading(symbol, strategy)
        
        # Check for position size limits
        self._check_position_size_limits(symbol)
        
        # Check for psychological patterns
        self._check_psychological_patterns(symbol, strategy)
    
    def _on_trade_closed(self, event: Event) -> None:
        """
        Handle trade closed events to track performance metrics.
        
        Args:
            event: Trade closed event
        """
        if not event.data:
            return
            
        # Extract trade data
        trade = event.data
        symbol = trade.get("symbol")
        strategy = trade.get("metadata", {}).get("strategy")
        profit_loss = trade.get("realized_pnl", 0)
        
        # Check if this symbol and strategy are monitored
        if not self._is_monitored_symbol(symbol) or not self._is_monitored_strategy(strategy):
            return
        
        # Update performance tracking
        if profit_loss > 0:
            self.trades_win_count += 1
            self.gross_profits += profit_loss
            self.consecutive_losses = 0
        else:
            self.trades_loss_count += 1
            self.gross_losses += abs(profit_loss)
            self.consecutive_losses += 1
            
            # Check for consecutive losses
            if self.consecutive_losses >= self.parameters["max_consecutive_losses"]:
                self._create_alert(
                    alert_type="CONSECUTIVE_LOSSES",
                    symbol=symbol,
                    severity=AlertSeverity.HIGH,
                    message=f"{self.consecutive_losses} consecutive losses detected",
                    metadata={
                        "consecutive_losses": self.consecutive_losses,
                        "last_symbol": symbol,
                        "last_strategy": strategy
                    }
                )
                
                # Adjust position size after consecutive losses
                if self.parameters["auto_reduce_positions"]:
                    self._reduce_position_size()
        
        # Calculate and check performance metrics
        self._check_performance_metrics()
    
    def _on_position_update(self, event: Event) -> None:
        """
        Handle position update events to track risk exposure.
        
        Args:
            event: Position update event
        """
        if not event.data:
            return
            
        # Extract position data
        position = event.data
        
        # Check open positions count
        open_positions = self.session.get_open_positions()
        if len(open_positions) > self.parameters["max_open_positions"]:
            self._create_alert(
                alert_type="TOO_MANY_POSITIONS",
                symbol=None,
                severity=AlertSeverity.HIGH,
                message=f"Too many open positions: {len(open_positions)} (max: {self.parameters['max_open_positions']})",
                metadata={
                    "open_positions": len(open_positions),
                    "max_allowed": self.parameters["max_open_positions"]
                }
            )
        
        # Check for correlated exposure
        self._update_sector_exposure()
        self._check_correlation_exposure()
    
    def _on_order_update(self, event: Event) -> None:
        """
        Handle order update events to track pending risk.
        
        Args:
            event: Order update event
        """
        if not event.data:
            return
        
        # We'll implement order tracking if needed
        pass
    
    def _on_account_update(self, event: Event) -> None:
        """
        Handle account update events to track equity and drawdown.
        
        Args:
            event: Account update event
        """
        if not event.data:
            return
            
        # Extract account data
        account = event.data
        current_equity = account.get("equity")
        
        if not current_equity:
            return
        
        # Update high watermarks and drawdowns
        self._update_drawdown_metrics(current_equity)
    
    def _on_signal_event(self, event: Event) -> None:
        """
        Handle signal events to check if we should block signals during cooldown.
        
        Args:
            event: Signal event
        """
        if not event.data:
            return
            
        # Extract signal data
        signal = event.data
        symbol = signal.get("symbol")
        strategy = signal.get("strategy")
        
        # Check if trading is paused
        if self.trading_paused:
            now = datetime.now()
            if self.trading_paused_until and now < self.trading_paused_until:
                # Still in cooldown period, block the signal
                logger.warning(f"Blocking signal for {symbol} from {strategy} due to trading cooldown")
                
                # Publish a blocked signal event
                blocked_signal_event = {
                    "original_signal": signal,
                    "reason": "TRADING_COOLDOWN",
                    "cooldown_until": self.trading_paused_until,
                    "timestamp": now
                }
                self.event_bus.publish("SIGNAL_BLOCKED", blocked_signal_event)
                
                return
            else:
                # Cooldown period ended
                self.trading_paused = False
                self.trading_paused_until = None
                
                # Publish a cooldown ended event
                cooldown_ended_event = {
                    "timestamp": now,
                    "message": "Trading cooldown period ended",
                    "recovery_mode": self.position_size_multiplier < 1.0
                }
                self.event_bus.publish("TRADING_COOLDOWN_ENDED", cooldown_ended_event)
        
        # Check for signal frequency (potential algorithmic issues)
        self._check_signal_frequency(symbol, strategy)
    
    def _on_market_data_update(self, event: Event) -> None:
        """
        Handle market data updates.
        
        Args:
            event: Market data update event
        """
        if not event.data:
            return
        
        # We use this primarily to keep market data current for analysis
        symbol = event.data.get("symbol")
        if not self._is_monitored_symbol(symbol):
            return
    
    def _on_volatility_update(self, event: Event) -> None:
        """
        Handle volatility update events to adjust thresholds.
        
        Args:
            event: Volatility update event
        """
        if not event.data or not self.parameters["adapt_to_volatility"]:
            return
            
        # Extract volatility data
        volatility_data = event.data
        symbol = volatility_data.get("symbol")
        regime = volatility_data.get("regime")
        
        if not symbol or not regime:
            return
            
        # Store volatility info
        self.volatility_by_symbol[symbol] = {
            "regime": regime,
            "timestamp": datetime.now(),
            "metrics": volatility_data.get("metrics", {})
        }
        
        # Adjust thresholds for high volatility
        if regime in ["HIGH", "VERY_HIGH", "EXTREME"]:
            self.is_high_volatility = True
        else:
            # Check if all/most symbols are in lower volatility regimes
            high_vol_symbols = sum(1 for v in self.volatility_by_symbol.values() 
                                 if v.get("regime") in ["HIGH", "VERY_HIGH", "EXTREME"])
            self.is_high_volatility = high_vol_symbols > len(self.volatility_by_symbol) * 0.3
    
    def _on_minute_update(self, event: Event) -> None:
        """
        Handle minute timeframe events for frequent checks.
        
        Args:
            event: Minute timeframe event
        """
        # Check recovery state and cooldown expiration
        self._check_recovery_state()
        
        # Process and clean up expired alerts
        self._process_expired_alerts()
    
    def _on_hourly_update(self, event: Event) -> None:
        """
        Handle hourly timeframe events for periodic checks.
        
        Args:
            event: Hourly timeframe event
        """
        # Update correlation matrix for risk analysis
        self._update_correlation_matrix()
        
        # Check for intraday drawdown
        current_equity = self.session.get_account_balance()
        self._update_drawdown_metrics(current_equity)
    
    def _on_daily_update(self, event: Event) -> None:
        """
        Handle daily timeframe events for daily resets and reports.
        
        Args:
            event: Daily timeframe event
        """
        # Reset daily counters and metrics
        self._reset_daily_metrics()
        
        # Generate daily trading activity report
        self._generate_daily_report()
    
    def _is_monitored_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is being monitored by this strategy.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if monitored, False otherwise
        """
        if not symbol:
            return False
            
        monitoring_symbols = self.parameters["monitoring_symbols"]
        return "ALL" in monitoring_symbols or symbol in monitoring_symbols
    
    def _is_monitored_strategy(self, strategy: str) -> bool:
        """
        Check if a strategy is being monitored by this meta-strategy.
        
        Args:
            strategy: Strategy name to check
            
        Returns:
            True if monitored, False otherwise
        """
        if not strategy:
            return False
            
        monitoring_strategies = self.parameters["monitoring_strategies"]
        return "ALL" in monitoring_strategies or strategy in monitoring_strategies
    
    def _check_overtrading(self, symbol: str, strategy: str) -> None:
        """
        Check for overtrading based on frequency thresholds.
        
        Args:
            symbol: Symbol that was traded
            strategy: Strategy that generated the trade
        """
        # Get thresholds, adjusted for volatility and day of week if needed
        max_trades_per_day = self.parameters["max_trades_per_day"]
        max_trades_per_symbol = self.parameters["max_trades_per_symbol_per_day"]
        
        # Adjust for weekend if needed (crypto markets)
        if datetime.now().weekday() >= 5:  # 5=Saturday, 6=Sunday
            weekend_factor = self.parameters["weekend_reduction_factor"]
            max_trades_per_day = int(max_trades_per_day * weekend_factor)
            max_trades_per_symbol = int(max_trades_per_symbol * weekend_factor)
        
        # Adjust for volatility if needed
        if self.is_high_volatility and self.parameters["adapt_to_volatility"]:
            volatility_factor = self.parameters["high_volatility_reduction"]
            max_trades_per_day = int(max_trades_per_day * volatility_factor)
            max_trades_per_symbol = int(max_trades_per_symbol * volatility_factor)
        
        # Check overall trade count
        if self.trades_today >= max_trades_per_day:
            self._create_alert(
                alert_type="OVERTRADING_DAILY_LIMIT",
                symbol=symbol,
                severity=AlertSeverity.HIGH,
                message=f"Daily trade limit reached: {self.trades_today} trades (max: {max_trades_per_day})",
                metadata={
                    "trades_today": self.trades_today,
                    "max_trades": max_trades_per_day,
                    "latest_symbol": symbol,
                    "latest_strategy": strategy
                }
            )
            
            # Pause trading if configured
            if self.parameters["auto_pause_trading"]:
                self._pause_trading()
        
        # Check symbol-specific trade count
        if self.trades_by_symbol[symbol] >= max_trades_per_symbol:
            self._create_alert(
                alert_type="OVERTRADING_SYMBOL_LIMIT",
                symbol=symbol,
                severity=AlertSeverity.MEDIUM,
                message=f"Symbol trade limit reached for {symbol}: {self.trades_by_symbol[symbol]} trades",
                metadata={
                    "symbol": symbol,
                    "trades_for_symbol": self.trades_by_symbol[symbol],
                    "max_trades_per_symbol": max_trades_per_symbol
                }
            )
        
        # Check time between trades for same symbol
        min_time_between_trades = self.parameters["min_time_between_trades"]
        last_trade_time = self.last_trade_time_by_symbol.get(symbol)
        
        if last_trade_time:
            time_since_last_trade = (datetime.now() - last_trade_time).total_seconds() / 60
            if time_since_last_trade < min_time_between_trades:
                self._create_alert(
                    alert_type="RAPID_TRADING",
                    symbol=symbol,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Trading too frequently on {symbol}: {time_since_last_trade:.1f} minutes since last trade",
                    metadata={
                        "symbol": symbol,
                        "minutes_since_last_trade": time_since_last_trade,
                        "min_required": min_time_between_trades
                    }
                )
    
    def _update_drawdown_metrics(self, current_equity: float) -> None:
        """
        Update drawdown metrics based on current equity.
        
        Args:
            current_equity: Current account equity
        """
        # Update daily high/low water marks
        today = datetime.now().date()
        
        # Reset for new day if needed
        if today != self.current_date:
            self.current_date = today
            self.daily_high_equity = current_equity
            self.daily_low_equity = current_equity
        
        # Update daily metrics
        if current_equity > self.daily_high_equity:
            self.daily_high_equity = current_equity
        elif current_equity < self.daily_low_equity:
            self.daily_low_equity = current_equity
        
        # Update account high water mark
        if current_equity > self.account_high_watermark:
            self.account_high_watermark = current_equity
        
        # Calculate drawdowns
        if self.daily_high_equity > 0:
            self.daily_drawdown_pct = (self.daily_high_equity - current_equity) / self.daily_high_equity * 100
        
        if self.account_high_watermark > 0:
            self.account_drawdown_pct = (self.account_high_watermark - current_equity) / self.account_high_watermark * 100
        
        # Check drawdown thresholds
        max_daily_dd = self.parameters["max_daily_drawdown_pct"]
        max_account_dd = self.parameters["max_account_drawdown_pct"]
        
        # Check daily drawdown threshold
        if self.daily_drawdown_pct >= max_daily_dd:
            self._create_alert(
                alert_type="DAILY_DRAWDOWN_EXCEEDED",
                symbol=None,
                severity=AlertSeverity.HIGH,
                message=f"Daily drawdown threshold exceeded: {self.daily_drawdown_pct:.2f}% (max: {max_daily_dd}%)",
                metadata={
                    "daily_drawdown": self.daily_drawdown_pct,
                    "max_allowed": max_daily_dd,
                    "current_equity": current_equity,
                    "daily_high": self.daily_high_equity
                }
            )
            
            # Take action if configured
            if self.parameters["auto_reduce_positions"]:
                self._reduce_risk_exposure()
                
            if self.parameters["auto_pause_trading"]:
                self._pause_trading()
        
        # Check account drawdown threshold
        if self.account_drawdown_pct >= max_account_dd:
            self._create_alert(
                alert_type="ACCOUNT_DRAWDOWN_EXCEEDED",
                symbol=None,
                severity=AlertSeverity.CRITICAL,
                message=f"Account drawdown threshold exceeded: {self.account_drawdown_pct:.2f}% (max: {max_account_dd}%)",
                metadata={
                    "account_drawdown": self.account_drawdown_pct,
                    "max_allowed": max_account_dd,
                    "current_equity": current_equity,
                    "account_high": self.account_high_watermark
                }
            )
            
            # Take action if configured - this is critical, so always take action
            self._reduce_risk_exposure(critical=True)
            self._pause_trading(extended=True)
    
    def _check_position_size_limits(self, symbol: str) -> None:
        """
        Check if position size exceeds configured limits.
        
        Args:
            symbol: Symbol to check position size for
        """
        position = self.session.get_position(symbol)
        if not position or not position.is_open:
            return
            
        # Calculate position size as percentage of account
        account_balance = self.session.get_account_balance()
        if account_balance <= 0:
            return
            
        position_value = position.quantity * position.avg_price
        position_pct = position_value / account_balance * 100
        
        # Check against threshold
        max_position_pct = self.parameters["max_position_size_pct"]
        
        if position_pct > max_position_pct:
            self._create_alert(
                alert_type="POSITION_SIZE_EXCEEDED",
                symbol=symbol,
                severity=AlertSeverity.HIGH,
                message=f"Position size limit exceeded for {symbol}: {position_pct:.2f}% (max: {max_position_pct}%)",
                metadata={
                    "symbol": symbol,
                    "position_pct": position_pct,
                    "max_allowed": max_position_pct,
                    "position_value": position_value,
                    "account_balance": account_balance
                }
            )
            
            # Reduce position if configured
            if self.parameters["auto_reduce_positions"]:
                # Calculate how much to reduce
                target_value = account_balance * (max_position_pct / 100)
                excess_value = position_value - target_value
                excess_quantity = excess_value / position.avg_price
                
                # Reduce position
                if excess_quantity > 0:
                    logger.warning(f"Reducing {symbol} position by {excess_quantity} units due to size limit")
                    self.session.reduce_position(symbol=symbol, quantity=excess_quantity)
    
    def _create_alert(self, alert_type: str, symbol: str, severity: AlertSeverity, message: str, metadata: Dict[str, Any] = None) -> None:
        """
        Create a new alert and handle it based on severity.
        
        Args:
            alert_type: Type of alert
            symbol: Symbol the alert is related to (can be None for account-level alerts)
            severity: Alert severity level
            message: Alert message
            metadata: Additional alert metadata
        """
        # Create alert object
        alert_id = str(self.next_alert_id)
        self.next_alert_id += 1
        
        alert = {
            "id": alert_id,
            "type": alert_type,
            "symbol": symbol,
            "severity": severity.name,
            "severity_level": severity.value,
            "message": message,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
            "acknowledged": False
        }
        
        # Store in active alerts
        self.active_alerts[alert_id] = alert
        
        # Log the alert
        log_message = f"ALERT [{severity.name}]: {message}"
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            logger.error(log_message)
        elif severity == AlertSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Publish alert event
        self.event_bus.publish("TRADING_ALERT", alert)
        
        # Check if we need to take automatic action based on severity
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            # High severity alerts may trigger risk reduction
            if self.parameters["auto_reduce_positions"] and alert_type in [
                "CONSECUTIVE_LOSSES", "DAILY_DRAWDOWN_EXCEEDED", "ACCOUNT_DRAWDOWN_EXCEEDED",
                "OVERTRADING_DAILY_LIMIT", "RAPID_TRADING_PATTERN", "REVENGE_TRADING_DETECTED"
            ]:
                self._reduce_risk_exposure(critical=(severity == AlertSeverity.CRITICAL))
            
            # Critical alerts may pause trading
            if self.parameters["auto_pause_trading"] and severity == AlertSeverity.CRITICAL:
                self._pause_trading(extended=True)
    
    def _pause_trading(self, extended: bool = False) -> None:
        """
        Pause trading for a cooldown period.
        
        Args:
            extended: Whether to use an extended cooldown period for serious issues
        """
        if self.trading_paused:
            # Already paused
            return
            
        # Set cooldown period
        cooldown_minutes = self.parameters["trading_cooldown_minutes"]
        if extended:
            cooldown_minutes *= 3  # Extend cooldown for serious issues
            
        self.trading_paused = True
        self.trading_paused_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        logger.warning(f"Trading paused for {cooldown_minutes} minutes until {self.trading_paused_until}")
        
        # Publish trading paused event
        pause_event = {
            "timestamp": datetime.now(),
            "reason": "RISK_THRESHOLD_EXCEEDED",
            "cooldown_until": self.trading_paused_until,
            "extended": extended
        }
        self.event_bus.publish("TRADING_PAUSED", pause_event)
    
    def _reduce_position_size(self) -> None:
        """
        Reduce position size multiplier after risk events.
        """
        # Set reduced position size multiplier
        self.position_size_multiplier = self.parameters["reduced_size_after_drawdown"]
        self.recovery_start_time = datetime.now()
        
        logger.warning(f"Reducing position size multiplier to {self.position_size_multiplier} due to risk event")
        
        # Publish position size reduction event
        reduction_event = {
            "timestamp": datetime.now(),
            "new_multiplier": self.position_size_multiplier,
            "reason": "RISK_THRESHOLD_EXCEEDED",
            "recovery_days": self.parameters["gradual_recovery_days"]
        }
        self.event_bus.publish("POSITION_SIZE_REDUCED", reduction_event)
    
    def _reduce_risk_exposure(self, critical: bool = False) -> None:
        """
        Reduce overall risk exposure by scaling down positions.
        
        Args:
            critical: Whether this is a critical risk event warranting stronger action
        """
        # Get all open positions
        open_positions = self.session.get_open_positions()
        if not open_positions:
            return
            
        # Determine reduction percentage
        reduction_pct = 0.5  # Default 50% reduction
        if critical:
            reduction_pct = 0.75  # Critical events reduce by 75%
            
        logger.warning(f"Reducing risk exposure by {reduction_pct * 100:.0f}% across all positions")
        
        # Reduce each position
        for position in open_positions:
            if position.is_open and position.quantity > 0:
                reduce_qty = position.quantity * reduction_pct
                if reduce_qty > 0:
                    logger.warning(f"Reducing {position.symbol} position by {reduce_qty} units")
                    self.session.reduce_position(symbol=position.symbol, quantity=reduce_qty)
        
        # Also reduce position size for future trades
        self._reduce_position_size()
        
        # Publish risk reduction event
        reduction_event = {
            "timestamp": datetime.now(),
            "reduction_percentage": reduction_pct * 100,
            "reason": "RISK_THRESHOLD_EXCEEDED",
            "critical": critical,
            "positions_affected": len(open_positions)
        }
        self.event_bus.publish("RISK_EXPOSURE_REDUCED", reduction_event)
    
    def _check_recovery_state(self) -> None:
        """
        Check and update recovery state for position sizing.
        """
        # Check if we're in recovery mode
        if self.position_size_multiplier >= 1.0 or not self.recovery_start_time:
            return
            
        # Calculate days in recovery
        days_in_recovery = (datetime.now() - self.recovery_start_time).total_seconds() / (24 * 60 * 60)
        recovery_days = self.parameters["gradual_recovery_days"]
        
        if days_in_recovery >= recovery_days:
            # Recovery period complete
            self.position_size_multiplier = 1.0
            self.recovery_start_time = None
            
            logger.info("Position size recovery complete, returned to normal sizing")
            
            # Publish recovery complete event
            recovery_event = {
                "timestamp": datetime.now(),
                "message": "Position size recovery complete",
                "days_in_recovery": days_in_recovery
            }
            self.event_bus.publish("POSITION_SIZE_RECOVERY_COMPLETE", recovery_event)
        else:
            # Gradually increase position size during recovery
            recovery_progress = days_in_recovery / recovery_days
            reduced_size = self.parameters["reduced_size_after_drawdown"]
            new_multiplier = reduced_size + (1.0 - reduced_size) * recovery_progress
            
            # Update multiplier if changed significantly
            if abs(new_multiplier - self.position_size_multiplier) > 0.05:
                old_multiplier = self.position_size_multiplier
                self.position_size_multiplier = new_multiplier
                
                logger.info(f"Gradually increasing position size multiplier: {old_multiplier:.2f} -> {new_multiplier:.2f}")
    
    def _process_expired_alerts(self) -> None:
        """
        Process expired alerts and move them to history.
        """
        # Find alerts older than 24 hours
        now = datetime.now()
        expired_alert_ids = []
        
        for alert_id, alert in self.active_alerts.items():
            alert_time = alert.get("timestamp")
            if alert_time and (now - alert_time).total_seconds() > 24 * 60 * 60:
                # Alert is over 24 hours old, move to history
                expired_alert_ids.append(alert_id)
                self.alert_history.append(alert)
        
        # Remove expired alerts from active alerts
        for alert_id in expired_alert_ids:
            del self.active_alerts[alert_id]
        
        # Limit alert history size
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def _reset_daily_metrics(self) -> None:
        """
        Reset daily trading metrics.
        """
        # Reset trade counters
        self.trades_today = 0
        self.trades_by_symbol = defaultdict(int)
        self.trades_by_strategy = defaultdict(int)
        
        # Reset daily equity metrics
        current_equity = self.session.get_account_balance()
        self.daily_high_equity = current_equity
        self.daily_low_equity = current_equity
        
        logger.info("Daily trading metrics reset")
    
    def _generate_daily_report(self) -> None:
        """
        Generate and publish a daily trading activity report.
        """
        # Get performance metrics
        win_rate = 0.0
        if (self.trades_win_count + self.trades_loss_count) > 0:
            win_rate = self.trades_win_count / (self.trades_win_count + self.trades_loss_count)
            
        profit_factor = 1.0
        if self.gross_losses > 0:
            profit_factor = self.gross_profits / self.gross_losses
            
        # Build report
        yesterday = (datetime.now() - timedelta(days=1)).date()
        report = {
            "date": yesterday,
            "timestamp": datetime.now(),
            "total_trades": self.trades_win_count + self.trades_loss_count,
            "winning_trades": self.trades_win_count,
            "losing_trades": self.trades_loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": self.gross_profits,
            "gross_loss": self.gross_losses,
            "net_pnl": self.gross_profits - self.gross_losses,
            "alerts": {
                "high": sum(1 for a in self.alert_history if a.get("severity") in ["HIGH", "CRITICAL"]),
                "medium": sum(1 for a in self.alert_history if a.get("severity") == "MEDIUM"),
                "low": sum(1 for a in self.alert_history if a.get("severity") in ["LOW", "INFO"])
            },
            "most_traded_symbols": dict(sorted(self.trades_by_symbol.items(), key=lambda x: x[1], reverse=True)[:5]),
            "most_active_strategies": dict(sorted(self.trades_by_strategy.items(), key=lambda x: x[1], reverse=True)[:3])
        }
        
        # Log summary
        logger.info(f"Daily Report: {report['total_trades']} trades, Win Rate: {win_rate:.1%}, P/F: {profit_factor:.2f}")
        
        # Publish report event
        self.event_bus.publish("DAILY_TRADING_REPORT", report)
    
    def _update_correlation_matrix(self) -> None:
        """
        Update the correlation matrix for currently traded symbols.
        """
        # Get open positions
        open_positions = self.session.get_open_positions()
        if not open_positions:
            return
            
        # Get symbols from open positions
        symbols = [position.symbol for position in open_positions if position.is_open]
        if len(symbols) < 2:  # Need at least 2 symbols for correlation
            return
            
        # Get price data for correlation calculation
        price_data = {}
        for symbol in symbols:
            data = self.session.get_historical_data(symbol, "1h", 100)  # 100 hours of data
            if data is not None and not data.empty:
                price_data[symbol] = data["close"]
        
        if len(price_data) < 2:
            return
            
        # Create DataFrame and calculate correlation matrix
        try:
            df = pd.DataFrame(price_data)
            self.correlation_matrix = df.pct_change().corr()
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
    
    def _update_sector_exposure(self) -> None:
        """
        Update sector exposure based on open positions.
        """
        # Reset sector exposure
        self.sector_exposure = defaultdict(float)
        
        # Get open positions
        open_positions = self.session.get_open_positions()
        if not open_positions:
            return
            
        # Get account balance for percentage calculations
        account_balance = self.session.get_account_balance()
        if account_balance <= 0:
            return
            
        # Calculate exposure per sector
        for position in open_positions:
            if not position.is_open or position.quantity <= 0:
                continue
                
            # Get symbol metadata - in a real implementation, this would come from a proper source
            # For this example, we'll use a simple approach
            symbol = position.symbol
            sector = self._get_symbol_sector(symbol)
            
            # Calculate position value and add to sector exposure
            position_value = position.quantity * position.avg_price
            position_pct = position_value / account_balance * 100
            
            self.sector_exposure[sector] += position_pct
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Get the sector for a symbol (simplified implementation).
        
        Args:
            symbol: Symbol to get sector for
            
        Returns:
            Sector name
        """
        # In a real implementation, this would come from a proper reference data source
        # For this example, we'll use a simplified approach based on symbol patterns
        
        # Stablecoins
        if symbol.startswith("USD") or "USD" in symbol:
            return "stablecoin"
            
        # Major cryptocurrencies
        if symbol in ["BTC", "ETH", "XRP", "LTC", "BCH"]:
            return "major"
            
        # DeFi tokens (simplified)
        if symbol in ["AAVE", "UNI", "COMP", "MKR", "SUSHI", "YFI", "CAKE", "CRV"]:
            return "defi"
            
        # Exchange tokens
        if symbol in ["BNB", "FTT", "CRO", "HT", "OKB", "LEO"]:
            return "exchange"
            
        # Layer-1 blockchains
        if symbol in ["SOL", "ADA", "DOT", "AVAX", "ALGO", "ATOM", "NEAR", "FTM"]:
            return "layer1"
            
        # Layer-2 scaling
        if symbol in ["MATIC", "OP", "ARB", "LRC", "IMX", "ZKS"]:
            return "layer2"
            
        # Gaming/Metaverse
        if symbol in ["AXS", "MANA", "SAND", "ENJ", "GALA", "ILV", "ALICE"]:
            return "gaming"
            
        # Default to "other" if no match
        return "other"
    
    def _check_correlation_exposure(self) -> None:
        """
        Check for excessive exposure to correlated assets.
        """
        # Check sector exposure first (simpler)
        max_sector_exposure = self.parameters["max_correlated_exposure_pct"]
        
        for sector, exposure in self.sector_exposure.items():
            if exposure > max_sector_exposure:
                self._create_alert(
                    alert_type="SECTOR_EXPOSURE_EXCEEDED",
                    symbol=None,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Excessive exposure to {sector} sector: {exposure:.2f}% (max: {max_sector_exposure}%)",
                    metadata={
                        "sector": sector,
                        "exposure": exposure,
                        "max_allowed": max_sector_exposure
                    }
                )
        
        # Check correlation matrix if available
        if self.correlation_matrix is not None and not self.correlation_matrix.empty:
            # Find highly correlated pairs
            high_corr_pairs = []
            
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    symbol1 = self.correlation_matrix.columns[i]
                    symbol2 = self.correlation_matrix.columns[j]
                    corr = self.correlation_matrix.iloc[i, j]
                    
                    if corr > 0.8:  # High positive correlation threshold
                        high_corr_pairs.append((symbol1, symbol2, corr))
            
            # Alert if we have highly correlated pairs
            if high_corr_pairs:
                # Get the highest correlated pair
                highest_pair = max(high_corr_pairs, key=lambda x: x[2])
                symbol1, symbol2, corr = highest_pair
                
                self._create_alert(
                    alert_type="HIGH_CORRELATION_EXPOSURE",
                    symbol=None,
                    severity=AlertSeverity.MEDIUM,
                    message=f"High correlation ({corr:.2f}) between {symbol1} and {symbol2}",
                    metadata={
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "correlation": corr,
                        "all_high_corr_pairs": high_corr_pairs
                    }
                )
    
    def _check_performance_metrics(self) -> None:
        """
        Check performance metrics against thresholds.
        """
        # Calculate win rate and profit factor
        total_trades = self.trades_win_count + self.trades_loss_count
        
        if total_trades < 10:  # Need enough trades for meaningful metrics
            return
            
        win_rate = self.trades_win_count / total_trades if total_trades > 0 else 0
        profit_factor = self.gross_profits / self.gross_losses if self.gross_losses > 0 else 1.0
        
        # Check against thresholds
        min_win_rate = self.parameters["min_win_rate_threshold"]
        min_profit_factor = self.parameters["min_profit_factor"]
        
        if win_rate < min_win_rate:
            self._create_alert(
                alert_type="LOW_WIN_RATE",
                symbol=None,
                severity=AlertSeverity.MEDIUM,
                message=f"Low win rate: {win_rate:.1%} (min: {min_win_rate:.1%})",
                metadata={
                    "win_rate": win_rate,
                    "min_threshold": min_win_rate,
                    "winning_trades": self.trades_win_count,
                    "losing_trades": self.trades_loss_count
                }
            )
        
        if profit_factor < min_profit_factor:
            self._create_alert(
                alert_type="LOW_PROFIT_FACTOR",
                symbol=None,
                severity=AlertSeverity.MEDIUM,
                message=f"Low profit factor: {profit_factor:.2f} (min: {min_profit_factor:.2f})",
                metadata={
                    "profit_factor": profit_factor,
                    "min_threshold": min_profit_factor,
                    "gross_profits": self.gross_profits,
                    "gross_losses": self.gross_losses
                }
            )
    
    def _check_signal_frequency(self, symbol: str, strategy: str) -> None:
        """
        Check frequency of signals from a strategy (to detect potential algorithmic issues).
        
        Args:
            symbol: Symbol that generated the signal
            strategy: Strategy that generated the signal
        """
        # This is a simplified implementation
        # In a real system, you would track signal timestamps per strategy/symbol
        pass
    
    def _check_psychological_patterns(self, symbol: str, strategy: str) -> None:
        """
        Check for psychological trading patterns like revenge trading or FOMO.
        
        Args:
            symbol: Symbol that was traded
            strategy: Strategy that generated the trade
        """
        # Only check if features are enabled
        if not self.parameters["detect_revenge_trading"] and not self.parameters["detect_fomo"]:
            return
            
        # We need trade history to detect patterns
        if len(self.recent_trades) < 5:
            return
            
        # Check for revenge trading pattern (simplified implementation)
        # Revenge trading is typically characterized by:
        # 1. A losing trade
        # 2. Followed quickly by a larger trade on the same symbol
        # 3. Often with the same direction (trying to "make back" the loss)
        if self.parameters["detect_revenge_trading"] and self.consecutive_losses > 0:
            # Get last 5 trades
            recent_trades = list(self.recent_trades)[-5:]
            symbol_trades = [t for t in recent_trades if t["symbol"] == symbol]
            
            if len(symbol_trades) >= 2:
                # Check for increasing position sizes after losses
                for i in range(1, len(symbol_trades)):
                    curr_trade = symbol_trades[i]
                    prev_trade = symbol_trades[i-1]
                    
                    # If previous trade was closed recently with a loss, and this trade is bigger
                    time_diff = (curr_trade["timestamp"] - prev_trade["timestamp"]).total_seconds() / 60
                    
                    if time_diff < 30 and curr_trade["quantity"] > prev_trade["quantity"] * 1.5:
                        self._create_alert(
                            alert_type="REVENGE_TRADING_DETECTED",
                            symbol=symbol,
                            severity=AlertSeverity.HIGH,
                            message=f"Possible revenge trading detected on {symbol}: {time_diff:.1f}min after previous trade, size increased by {curr_trade['quantity']/prev_trade['quantity']:.1f}x",
                            metadata={
                                "symbol": symbol,
                                "strategy": strategy,
                                "minutes_since_previous": time_diff,
                                "size_increase_factor": curr_trade["quantity"]/prev_trade["quantity"]
                            }
                        )
                        break
        
        # Check for FOMO (Fear Of Missing Out) pattern (simplified implementation)
        # FOMO is typically characterized by:
        # 1. Entering after a large price movement
        # 2. Trading with increasing frequency during volatile markets
        # 3. Entering at progressively worse prices
        if self.parameters["detect_fomo"]:
            # This would require price action data which is beyond this implementation
            pass
    
    # Abstract method implementations required by CryptoBaseStrategy
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        This strategy doesn't calculate traditional indicators, as it's a meta-strategy.
        We use it to track risk and trading patterns, not generate signals directly.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Empty dictionary as we don't calculate technical indicators
        """
        # This is a meta-strategy that doesn't calculate technical indicators
        # or generate trading signals directly
        return {}
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        This strategy doesn't generate trading signals directly, as it's a meta-strategy.
        
        Args:
            data: Price data DataFrame
            indicators: Calculated indicators (empty in this case)
            
        Returns:
            Empty dictionary as we don't generate trading signals
        """
        # This is a meta-strategy that doesn't generate trading signals directly
        return {}
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size, applying any risk-based adjustments from this strategy.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Price data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Position size multiplier (not actual position size)
        """
        # The responsibility of this meta-strategy is to adjust position sizing
        # based on risk conditions, not to calculate actual position sizes
        return self.position_size_multiplier
    
    def regime_compatibility(self, regime_data: Dict[str, Any]) -> float:
        """
        This meta-strategy is compatible with all regimes as it provides risk guardrails.
        
        Args:
            regime_data: Market regime data
            
        Returns:
            Compatibility score (always high for this meta-strategy)
        """
        # This meta-strategy is always relevant as it provides risk guardrails
        # Its operation might adjust based on regime, but it's always applicable
        return 0.9  # High compatibility in all regimes
