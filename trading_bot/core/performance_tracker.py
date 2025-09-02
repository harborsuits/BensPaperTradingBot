#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Tracker

Tracks performance metrics for both paper and live strategies in a unified way,
enabling apples-to-apples comparisons for strategy evaluation and promotion.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import defaultdict

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and calculates performance metrics for strategies.
    
    This class:
    1. Subscribes to order and fill events from both paper and live brokers
    2. Stores trade data in a unified format
    3. Calculates key performance metrics (P&L, Sharpe, drawdown, etc.)
    4. Provides data for dashboard displays and evaluation
    """
    
    def __init__(
        self,
        data_dir: str = "data/performance",
        lookback_days: int = 90,
        save_interval_minutes: int = 30
    ):
        """
        Initialize performance tracker.
        
        Args:
            data_dir: Directory to store performance data
            lookback_days: Maximum days of data to keep in memory
            save_interval_minutes: How often to save data to disk
        """
        self._data_dir = data_dir
        self._lookback_days = lookback_days
        self._save_interval = save_interval_minutes * 60  # Convert to seconds
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Storage for strategy instances
        self._strategies = {}
        
        # Trade data storage
        self._trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._positions: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._daily_returns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._equity_curves: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Cached metrics
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._last_calculated: Dict[str, datetime] = {}
        
        # Event bus for order/fill events
        self._event_bus = ServiceRegistry.get_instance().get_service(EventBus)
        if self._event_bus:
            self._setup_event_listeners()
        
        # Last save timestamp
        self._last_save = datetime.now()
        
        logger.info("Performance tracker initialized")
        
        # Load existing data if available
        self._load_performance_data()
    
    def _setup_event_listeners(self) -> None:
        """Set up event listeners for order and fill events."""
        
        def on_order_placed(event: Event) -> None:
            """Handle order placed events."""
            data = event.data
            strategy_id = data.get("strategy_id")
            if not strategy_id:
                return
                
            # Track open orders
            # (Implementation depends on your event structure)
            pass
        
        def on_order_filled(event: Event) -> None:
            """Handle order filled events."""
            data = event.data
            strategy_id = data.get("strategy_id")
            if not strategy_id:
                # Try to get from tags
                tags = data.get("tags", [])
                for tag in tags:
                    if tag in self._strategies:
                        strategy_id = tag
                        break
            
            if not strategy_id:
                return
            
            # Add trade to history
            self.record_trade(
                strategy_id=strategy_id,
                symbol=data.get("symbol"),
                side=data.get("side"),
                quantity=data.get("quantity"),
                price=data.get("price"),
                timestamp=data.get("timestamp"),
                order_id=data.get("order_id"),
                tags=data.get("tags", [])
            )
        
        def on_position_update(event: Event) -> None:
            """Handle position update events."""
            data = event.data
            strategy_id = data.get("strategy_id")
            if not strategy_id:
                return
                
            # Update position data
            symbol = data.get("symbol")
            if not symbol:
                return
                
            self.update_position(
                strategy_id=strategy_id,
                symbol=symbol,
                quantity=data.get("quantity", 0),
                average_price=data.get("average_price", 0),
                current_price=data.get("current_price", 0),
                unrealized_pl=data.get("unrealized_pl", 0)
            )
        
        # Register event handlers
        self._event_bus.subscribe(EventType.ORDER_PLACED, on_order_placed)
        self._event_bus.subscribe(EventType.ORDER_FILLED, on_order_filled)
        self._event_bus.subscribe(EventType.POSITION_UPDATE, on_position_update)
    
    def register_strategy(self, strategy_id: str, strategy_instance=None) -> None:
        """
        Register a strategy for performance tracking.
        
        Args:
            strategy_id: Strategy ID
            strategy_instance: Optional strategy object reference
        """
        self._strategies[strategy_id] = strategy_instance
        
        # Initialize data structures if not already present
        if strategy_id not in self._trades:
            self._trades[strategy_id] = []
        
        if strategy_id not in self._positions:
            self._positions[strategy_id] = {}
        
        if strategy_id not in self._daily_returns:
            self._daily_returns[strategy_id] = {}
        
        if strategy_id not in self._equity_curves:
            self._equity_curves[strategy_id] = {}
        
        logger.info(f"Registered strategy '{strategy_id}' for performance tracking")
    
    def get_strategy(self, strategy_id: str):
        """
        Get strategy instance.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Strategy instance or None if not found
        """
        return self._strategies.get(strategy_id)
    
    def record_trade(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: Optional[Union[str, datetime]] = None,
        order_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            strategy_id: Strategy ID
            symbol: Symbol traded
            side: Buy or sell
            quantity: Number of shares/contracts
            price: Execution price
            timestamp: Trade timestamp
            order_id: Order ID
            tags: Optional tags (e.g., [PAPER])
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Recording trade for unregistered strategy '{strategy_id}'")
            self.register_strategy(strategy_id)
        
        # Convert timestamp to datetime if string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        elif timestamp is None:
            timestamp = datetime.now()
        
        # Create trade record
        trade = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side.lower(),
            "quantity": float(quantity),
            "price": float(price),
            "timestamp": timestamp.isoformat(),
            "order_id": order_id,
            "tags": tags or [],
            "value": float(quantity) * float(price),
            "is_paper": "PAPER" in (tags or [])
        }
        
        # Add to trade history
        self._trades[strategy_id].append(trade)
        
        # Update position
        self._update_position_from_trade(strategy_id, trade)
        
        # Update daily returns
        trade_date = timestamp.date().isoformat()
        self._update_daily_returns(strategy_id, trade_date)
        
        # Update equity curve
        self._update_equity_curve(strategy_id, trade_date)
        
        # Invalidate metrics cache
        if strategy_id in self._metrics_cache:
            del self._metrics_cache[strategy_id]
        
        # Auto-save after interval
        now = datetime.now()
        if (now - self._last_save).total_seconds() >= self._save_interval:
            self.save_performance_data()
            self._last_save = now
        
        logger.debug(f"Recorded {side} trade for '{strategy_id}': {quantity} {symbol} @ {price}")
    
    def update_position(
        self,
        strategy_id: str,
        symbol: str,
        quantity: float,
        average_price: float,
        current_price: float,
        unrealized_pl: Optional[float] = None
    ) -> None:
        """
        Update a position with current market data.
        
        Args:
            strategy_id: Strategy ID
            symbol: Symbol
            quantity: Current position size
            average_price: Average entry price
            current_price: Current market price
            unrealized_pl: Optional unrealized P&L
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Updating position for unregistered strategy '{strategy_id}'")
            self.register_strategy(strategy_id)
        
        # Calculate unrealized P&L if not provided
        if unrealized_pl is None and quantity != 0:
            unrealized_pl = quantity * (current_price - average_price)
        
        # Update position
        self._positions[strategy_id][symbol] = {
            "symbol": symbol,
            "quantity": float(quantity),
            "average_price": float(average_price),
            "current_price": float(current_price),
            "unrealized_pl": float(unrealized_pl) if unrealized_pl is not None else 0.0,
            "value": float(quantity) * float(current_price),
            "updated_at": datetime.now().isoformat()
        }
        
        # Update equity curve with current day
        today = datetime.now().date().isoformat()
        self._update_equity_curve(strategy_id, today)
        
        # Invalidate metrics cache
        if strategy_id in self._metrics_cache:
            del self._metrics_cache[strategy_id]
    
    def _update_position_from_trade(self, strategy_id: str, trade: Dict[str, Any]) -> None:
        """
        Update position based on a new trade.
        
        Args:
            strategy_id: Strategy ID
            trade: Trade record
        """
        symbol = trade["symbol"]
        side = trade["side"]
        quantity = trade["quantity"]
        price = trade["price"]
        
        # Get current position
        position = self._positions[strategy_id].get(symbol, {
            "symbol": symbol,
            "quantity": 0.0,
            "average_price": 0.0,
            "current_price": price,
            "unrealized_pl": 0.0,
            "value": 0.0,
            "updated_at": datetime.now().isoformat()
        })
        
        current_qty = position["quantity"]
        current_avg_price = position["average_price"]
        
        # Update position
        if side == "buy":
            # Adding to position
            if current_qty >= 0:
                # Increasing long position
                new_qty = current_qty + quantity
                new_avg_price = ((current_qty * current_avg_price) + (quantity * price)) / new_qty
                position["quantity"] = new_qty
                position["average_price"] = new_avg_price
            else:
                # Covering short position
                new_qty = current_qty + quantity
                if new_qty >= 0:
                    # Fully covered and possibly long now
                    realized_pl = -current_qty * (price - current_avg_price)
                    position["quantity"] = new_qty
                    position["average_price"] = price if new_qty > 0 else 0.0
                else:
                    # Partially covered short
                    realized_pl = quantity * (current_avg_price - price)
                    position["quantity"] = new_qty
                    # Average price stays the same
        else:  # sell
            # Reducing position
            if current_qty > 0:
                # Reducing long position
                new_qty = current_qty - quantity
                if new_qty >= 0:
                    # Still long or flat
                    realized_pl = quantity * (price - current_avg_price)
                    position["quantity"] = new_qty
                    # Average price stays the same if still long
                    if new_qty == 0:
                        position["average_price"] = 0.0
                else:
                    # Flipped to short
                    realized_pl = current_qty * (price - current_avg_price)
                    position["quantity"] = new_qty
                    position["average_price"] = price
            else:
                # Increasing short position
                new_qty = current_qty - quantity
                new_avg_price = ((abs(current_qty) * current_avg_price) + (quantity * price)) / abs(new_qty)
                position["quantity"] = new_qty
                position["average_price"] = new_avg_price
        
        # Update current price and value
        position["current_price"] = price
        position["value"] = position["quantity"] * price
        position["updated_at"] = datetime.now().isoformat()
        
        # Calculate unrealized P&L
        if position["quantity"] != 0:
            position["unrealized_pl"] = position["quantity"] * (position["current_price"] - position["average_price"])
        else:
            position["unrealized_pl"] = 0.0
        
        # Store updated position
        self._positions[strategy_id][symbol] = position
    
    def _update_daily_returns(self, strategy_id: str, date_str: str) -> None:
        """
        Update daily returns based on trades and positions.
        
        Args:
            strategy_id: Strategy ID
            date_str: Date string (YYYY-MM-DD)
        """
        # Calculate realized P&L for the day from trades
        day_trades = [
            t for t in self._trades[strategy_id]
            if t["timestamp"].startswith(date_str)
        ]
        
        # Simple implementation - more sophisticated calculation would track
        # inventory and match buys/sells for actual realized P&L
        
        # For now, just sum the day's trade values (sell positive, buy negative)
        daily_pnl = 0.0
        for trade in day_trades:
            if trade["side"] == "buy":
                daily_pnl -= trade["value"]
            else:
                daily_pnl += trade["value"]
        
        # Add unrealized P&L from open positions
        unrealized_pnl = sum(position["unrealized_pl"] for position in self._positions[strategy_id].values())
        
        # Total daily P&L
        total_pnl = daily_pnl + unrealized_pnl
        
        # Store daily return
        self._daily_returns[strategy_id][date_str] = total_pnl
    
    def _update_equity_curve(self, strategy_id: str, date_str: str) -> None:
        """
        Update equity curve for a strategy.
        
        Args:
            strategy_id: Strategy ID
            date_str: Date string (YYYY-MM-DD)
        """
        # Get previous equity if available
        dates = sorted(self._equity_curves[strategy_id].keys())
        prev_equity = 100000.0  # Default starting equity
        if dates:
            prev_equity = self._equity_curves[strategy_id][dates[-1]]
        
        # Get daily return
        daily_return = self._daily_returns[strategy_id].get(date_str, 0.0)
        
        # Calculate new equity
        new_equity = prev_equity + daily_return
        
        # Store in equity curve
        self._equity_curves[strategy_id][date_str] = new_equity
    
    def get_trades(self, strategy_id: str, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trades for a strategy.
        
        Args:
            strategy_id: Strategy ID
            days: Optional number of days to look back
            
        Returns:
            List[Dict]: Trade records
        """
        if strategy_id not in self._trades:
            return []
        
        trades = self._trades[strategy_id]
        
        if days is not None:
            # Filter by lookback period
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            trades = [t for t in trades if t["timestamp"] >= cutoff]
        
        return trades
    
    def get_positions(self, strategy_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dict[str, Dict]: Positions by symbol
        """
        if strategy_id not in self._positions:
            return {}
        
        return self._positions[strategy_id]
    
    def get_equity_curve(
        self,
        strategy_id: str,
        days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get equity curve for a strategy.
        
        Args:
            strategy_id: Strategy ID
            days: Optional number of days to look back
            
        Returns:
            Dict[str, float]: Equity curve by date
        """
        if strategy_id not in self._equity_curves:
            return {}
        
        equity_curve = self._equity_curves[strategy_id]
        
        if days is not None:
            # Filter by lookback period
            cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
            equity_curve = {d: v for d, v in equity_curve.items() if d >= cutoff}
        
        return equity_curve
    
    def get_daily_returns(
        self,
        strategy_id: str,
        days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get daily returns for a strategy.
        
        Args:
            strategy_id: Strategy ID
            days: Optional number of days to look back
            
        Returns:
            Dict[str, float]: Daily returns by date
        """
        if strategy_id not in self._daily_returns:
            return {}
        
        daily_returns = self._daily_returns[strategy_id]
        
        if days is not None:
            # Filter by lookback period
            cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
            daily_returns = {d: v for d, v in daily_returns.items() if d >= cutoff}
        
        return daily_returns
    
    def get_strategy_metrics(
        self,
        strategy_id: str,
        days: Optional[int] = None,
        force_recalculate: bool = False
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            days: Optional number of days to look back
            force_recalculate: Force metric recalculation
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Requesting metrics for unregistered strategy '{strategy_id}'")
            return {}
        
        # Check if we have cached metrics
        cache_key = f"{strategy_id}_{days or 'all'}"
        if not force_recalculate and cache_key in self._metrics_cache:
            # Check if cache is still valid (less than 1 hour old)
            last_calc = self._last_calculated.get(cache_key)
            if last_calc and (datetime.now() - last_calc).total_seconds() < 3600:
                return self._metrics_cache[cache_key]
        
        # Get equity curve and trades
        equity_curve = self.get_equity_curve(strategy_id, days)
        trades = self.get_trades(strategy_id, days)
        
        if not equity_curve or not trades:
            return {
                "strategy_id": strategy_id,
                "days_active": 0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0
            }
        
        # Calculate metrics
        try:
            metrics = self._calculate_metrics(strategy_id, equity_curve, trades, days)
            
            # Cache results
            self._metrics_cache[cache_key] = metrics
            self._last_calculated[cache_key] = datetime.now()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for strategy '{strategy_id}': {str(e)}")
            return {
                "strategy_id": strategy_id,
                "error": str(e)
            }
    
    def _calculate_metrics(
        self,
        strategy_id: str,
        equity_curve: Dict[str, float],
        trades: List[Dict[str, Any]],
        days: Optional[int]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from equity curve and trades.
        
        Args:
            strategy_id: Strategy ID
            equity_curve: Equity curve data
            trades: Trade records
            days: Lookback period in days
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Sort equity curve by date
        dates = sorted(equity_curve.keys())
        values = [equity_curve[d] for d in dates]
        
        if not dates or not values:
            return {
                "strategy_id": strategy_id,
                "days_active": 0,
                "error": "No equity curve data"
            }
        
        # Basic metrics
        start_equity = values[0]
        end_equity = values[-1]
        total_return = end_equity - start_equity
        total_return_pct = (total_return / start_equity) * 100.0
        
        # Calculate trading days active
        days_active = len(dates)
        
        # Win rate and profit factor
        winning_trades = [t for t in trades if (t["side"] == "buy" and t["price"] < t.get("exit_price", t["price"])) or 
                                             (t["side"] == "sell" and t["price"] > t.get("exit_price", t["price"]))]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # Gross profit and loss
        gross_profit = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0)
        gross_loss = sum(abs(t.get("profit", 0)) for t in trades if t.get("profit", 0) < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)
        
        # Convert to pandas Series for more complex calculations
        equity_series = pd.Series(values, index=pd.to_datetime(dates))
        
        # Calculate daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        sharpe_ratio = 0.0
        if not daily_returns.empty:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        max_return = cumulative_returns.cummax()
        drawdown = (cumulative_returns / max_return) - 1
        max_drawdown_pct = drawdown.min() * 100 if not drawdown.empty else 0.0
        
        # Annualized return
        if len(dates) > 1:
            days_elapsed = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days
            if days_elapsed > 0:
                annualized_return_pct = ((1 + total_return_pct/100.0) ** (365.0/days_elapsed) - 1) * 100.0
            else:
                annualized_return_pct = 0.0
        else:
            annualized_return_pct = 0.0
        
        return {
            "strategy_id": strategy_id,
            "days_active": days_active,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_all_strategy_metrics(self, days: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all strategies.
        
        Args:
            days: Optional number of days to look back
            
        Returns:
            Dict[str, Dict[str, Any]]: Metrics by strategy ID
        """
        return {
            strategy_id: self.get_strategy_metrics(strategy_id, days)
            for strategy_id in self._strategies
        }
    
    def save_performance_data(self) -> bool:
        """
        Save performance data to disk.
        
        Returns:
            bool: Success status
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self._data_dir, exist_ok=True)
            
            # Save trades
            trades_file = os.path.join(self._data_dir, "trades.json")
            with open(trades_file, 'w') as f:
                # Convert trades to JSON-compatible format
                json_trades = {}
                for strategy_id, trades in self._trades.items():
                    json_trades[strategy_id] = []
                    for trade in trades:
                        json_trade = trade.copy()
                        json_trades[strategy_id].append(json_trade)
                
                json.dump(json_trades, f, indent=2)
            
            # Save positions
            positions_file = os.path.join(self._data_dir, "positions.json")
            with open(positions_file, 'w') as f:
                json.dump(self._positions, f, indent=2)
            
            # Save equity curves
            equity_file = os.path.join(self._data_dir, "equity_curves.json")
            with open(equity_file, 'w') as f:
                json.dump(self._equity_curves, f, indent=2)
            
            # Save daily returns
            returns_file = os.path.join(self._data_dir, "daily_returns.json")
            with open(returns_file, 'w') as f:
                json.dump(self._daily_returns, f, indent=2)
            
            logger.info(f"Saved performance data to {self._data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
            return False
    
    def _load_performance_data(self) -> None:
        """Load performance data from disk."""
        try:
            # Load trades
            trades_file = os.path.join(self._data_dir, "trades.json")
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self._trades = json.load(f)
            
            # Load positions
            positions_file = os.path.join(self._data_dir, "positions.json")
            if os.path.exists(positions_file):
                with open(positions_file, 'r') as f:
                    self._positions = json.load(f)
            
            # Load equity curves
            equity_file = os.path.join(self._data_dir, "equity_curves.json")
            if os.path.exists(equity_file):
                with open(equity_file, 'r') as f:
                    self._equity_curves = json.load(f)
            
            # Load daily returns
            returns_file = os.path.join(self._data_dir, "daily_returns.json")
            if os.path.exists(returns_file):
                with open(returns_file, 'r') as f:
                    self._daily_returns = json.load(f)
            
            logger.info(f"Loaded performance data from {self._data_dir}")
            
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
    
    def cleanup_old_data(self) -> None:
        """Remove data older than lookback period."""
        cutoff = (datetime.now() - timedelta(days=self._lookback_days)).isoformat()
        
        # Clean up trades
        for strategy_id in self._trades:
            self._trades[strategy_id] = [
                t for t in self._trades[strategy_id]
                if t["timestamp"] >= cutoff
            ]
        
        # Clean up equity curves and daily returns
        cutoff_date = cutoff.split('T')[0]  # Just the date part
        for strategy_id in self._equity_curves:
            self._equity_curves[strategy_id] = {
                d: v for d, v in self._equity_curves[strategy_id].items()
                if d >= cutoff_date
            }
        
        for strategy_id in self._daily_returns:
            self._daily_returns[strategy_id] = {
                d: v for d, v in self._daily_returns[strategy_id].items()
                if d >= cutoff_date
            }
        
        # Invalidate metrics cache
        self._metrics_cache = {}
        
        logger.info(f"Cleaned up data older than {self._lookback_days} days")
