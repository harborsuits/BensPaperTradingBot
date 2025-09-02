#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto High-Frequency Trading (HFT) Strategy

This strategy focuses on ultra-short-term trading with very high frequency,
capitalizing on small price movements and market inefficiencies. It relies on
low latency execution, orderbook analysis, and statistical arbitrage techniques.

Key characteristics:
- Extremely short holding periods (seconds to minutes)
- Very high trade frequency
- Tiny profit targets per trade
- Focus on execution speed and efficiency
- Heavy reliance on orderbook data and market microstructure
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.crypto.hft.crypto_hft_strategy_utils import (
    analyze_orderbook,
    calculate_z_score,
    detect_price_spikes,
    calculate_tick_size,
    calculate_execution_efficiency,
    check_for_latency_issues,
    detect_market_regime
)

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoHFTStrategy",
    market_type="crypto",
    description="High-frequency trading strategy for crypto markets using orderbook analysis and statistical methods",
    timeframes=["M1", "M5"],  # HFT often operates on even lower timeframes, using tick or orderbook data
    parameters={
        # Orderbook analysis parameters
        "orderbook_depth": {"type": "int", "default": 10, "min": 5, "max": 50},
        "min_bid_ask_spread_pct": {"type": "float", "default": 0.0005, "min": 0.0001, "max": 0.01},
        "imbalance_threshold": {"type": "float", "default": 1.5, "min": 1.1, "max": 5.0},
        "liquidity_threshold": {"type": "float", "default": 10000.0, "min": 1000.0, "max": 1000000.0},
        
        # Statistical parameters
        "price_mean_period": {"type": "int", "default": 100, "min": 20, "max": 1000},
        "price_std_dev_period": {"type": "int", "default": 100, "min": 20, "max": 1000},
        "z_score_entry": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0},
        "z_score_exit": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0},
        
        # Execution parameters
        "max_trades_per_minute": {"type": "int", "default": 10, "min": 1, "max": 60},
        "min_trade_interval_ms": {"type": "int", "default": 1000, "min": 100, "max": 10000},
        "profit_target_ticks": {"type": "int", "default": 2, "min": 1, "max": 10},
        "max_hold_time_seconds": {"type": "int", "default": 60, "min": 5, "max": 300},
        
        # Risk management
        "max_position_value_pct": {"type": "float", "default": 0.02, "min": 0.001, "max": 0.1},
        "max_open_positions": {"type": "int", "default": 5, "min": 1, "max": 20},
        "max_daily_loss_pct": {"type": "float", "default": 0.5, "min": 0.1, "max": 2.0},
        "position_sizing_method": {"type": "str", "default": "fixed", "enum": ["fixed", "volatility_adjusted", "portfolio_percentage"]},
        
        # Advanced
        "latency_threshold_ms": {"type": "int", "default": 50, "min": 10, "max": 500},
        "use_maker_only_orders": {"type": "bool", "default": True},
        "use_iceberg_orders": {"type": "bool", "default": False},
        "enable_mean_reversion": {"type": "bool", "default": True},
        "enable_momentum": {"type": "bool", "default": False},
        "enable_pairs_trading": {"type": "bool", "default": False}
    }
)
class CryptoHFTStrategy(CryptoBaseStrategy):
    """
    A high-frequency trading strategy for cryptocurrency markets.
    
    This strategy:
    1. Uses orderbook data to identify short-term trading opportunities
    2. Employs statistical arbitrage and mean reversion techniques
    3. Makes many small trades with tight spreads and quick exits
    4. Focuses on execution speed and transaction cost minimization
    5. Uses advanced risk controls to prevent large drawdowns
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto HFT strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # HFT-specific state variables
        self.orderbook_depth = self.parameters["orderbook_depth"]
        self.min_trade_interval = self.parameters["min_trade_interval_ms"] / 1000.0  # Convert to seconds
        self.max_hold_time = self.parameters["max_hold_time_seconds"]
        
        # Performance tracking
        self.trade_count = 0
        self.trades_this_minute = 0
        self.last_trade_time = None
        self.last_minute_reset = datetime.utcnow()
        self.execution_latencies = []
        self.daily_pnl = 0.0
        self.initial_daily_balance = 10000.0  # Mock value, would come from exchange API
        
        # Statistical state
        self.price_history = []
        self.mean_price = None
        self.std_dev_price = None
        self.z_scores = []
        
        # Orderbook state
        self.last_orderbook_update = None
        self.orderbook_imbalance_history = []
        self.spread_history = []
        
        # Execution state
        self.pending_orders = []
        self.active_hft_positions = []
        self.cancelled_orders = 0
        self.order_execution_times = []
        self.position_hold_times = []
        
        # Set up position tracking with unique identifiers for HFT positions
        self.next_position_id = 1
        
        # Risk management
        self.start_of_day = datetime.utcnow().date()
        self.max_daily_loss = self.initial_daily_balance * (self.parameters["max_daily_loss_pct"] / 100.0)
        self.risk_status = "normal"  # Can be normal, warning, or high
        
        # Initialize execution mode based on parameters
        self.using_maker_only = self.parameters["use_maker_only_orders"]
        self.using_iceberg = self.parameters["use_iceberg_orders"]
        
        # Strategy modes
        self.mean_reversion_enabled = self.parameters["enable_mean_reversion"]
        self.momentum_enabled = self.parameters["enable_momentum"]
        self.pairs_trading_enabled = self.parameters["enable_pairs_trading"]
        
        # For pairs trading (if enabled)
        self.correlated_pairs = []
        
        logger.info(f"Initialized crypto HFT strategy for {self.session.symbol} with {self.orderbook_depth} levels of depth")
        logger.info(f"Mode: Mean Reversion={self.mean_reversion_enabled}, Momentum={self.momentum_enabled}, Pairs={self.pairs_trading_enabled}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for HFT strategy.
        
        HFT relies more on orderbook data and market microstructure than traditional
        technical indicators, but we still calculate some statistical metrics.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty:
            return indicators
        
        # Current market data
        indicators["current_price"] = data["close"].iloc[-1]
        indicators["current_volume"] = data["volume"].iloc[-1]
        
        # Track price history (for statistical calculations)
        current_price = indicators["current_price"]
        self.price_history.append(current_price)
        
        # Keep price history to manageable size
        max_history = max(1000, self.parameters["price_mean_period"] * 2)
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        
        # Statistical calculations for mean reversion strategies
        if len(self.price_history) >= self.parameters["price_mean_period"]:
            # Use the specified period for calculations
            period = self.parameters["price_mean_period"]
            period_prices = self.price_history[-period:]
            
            # Calculate mean and standard deviation
            self.mean_price = sum(period_prices) / len(period_prices)
            squared_diffs = [(p - self.mean_price) ** 2 for p in period_prices]
            self.std_dev_price = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
            
            # Calculate z-score (how many standard deviations from mean)
            if self.std_dev_price > 0:
                z_score = (current_price - self.mean_price) / self.std_dev_price
                self.z_scores.append(z_score)
                indicators["z_score"] = z_score
                
                # Only keep recent z-scores
                if len(self.z_scores) > 100:
                    self.z_scores = self.z_scores[-100:]
        
        # Orderbook indicators (if orderbook data available)
        if hasattr(self, "orderbook") and self.orderbook:
            # Use the utility function to analyze orderbook
            ob_indicators = analyze_orderbook(self.orderbook, depth=self.orderbook_depth)
            
            # Add orderbook metrics to indicators
            indicators.update(ob_indicators)
            
            # Track bid-ask spread and imbalance history
            self.spread_history.append(ob_indicators["bid_ask_spread_pct"])
            self.orderbook_imbalance_history.append(ob_indicators["book_imbalance"])
            
            # Keep histories to manageable size
            if len(self.spread_history) > 1000:
                self.spread_history = self.spread_history[-1000:]
            if len(self.orderbook_imbalance_history) > 1000:
                self.orderbook_imbalance_history = self.orderbook_imbalance_history[-1000:]
            
            # Calculate tick size for this price level
            indicators["tick_size"] = calculate_tick_size(indicators["current_price"])
            
            # Calculate profit target in absolute terms (based on ticks)
            indicators["profit_target"] = indicators["tick_size"] * self.parameters["profit_target_ticks"]
        
        # Market regime detection (if we have enough price data)
        if len(data) >= 20:
            # Calculate returns for regime detection
            returns = data["close"].pct_change().dropna().values.tolist()
            volumes = data["volume"].values.tolist()
            
            indicators["market_regime"] = detect_market_regime(returns, volumes)
            
            # Check for abnormal price behavior
            indicators["price_spike_detected"] = detect_price_spikes(
                data["close"].values.tolist(), threshold=3.0
            )
        
        # Execution analytics
        if self.order_execution_times:
            efficiency_metrics = calculate_execution_efficiency(self.order_execution_times)
            indicators.update(efficiency_metrics)
            
            # Check for latency issues
            indicators["latency_issue_detected"] = check_for_latency_issues(
                self.order_execution_times[-20:] if len(self.order_execution_times) > 20 else self.order_execution_times,
                self.parameters["latency_threshold_ms"]
            )
        
        # Risk management indicators
        current_date = datetime.utcnow().date()
        if current_date > self.start_of_day:
            # Reset daily counters
            self.start_of_day = current_date
            self.daily_pnl = 0.0
            self.risk_status = "normal"
        
        # Set risk status based on daily PnL
        loss_threshold_warning = self.max_daily_loss * 0.5
        loss_threshold_high = self.max_daily_loss * 0.8
        
        if self.daily_pnl < -loss_threshold_high:
            self.risk_status = "high"
        elif self.daily_pnl < -loss_threshold_warning:
            self.risk_status = "warning"
        else:
            self.risk_status = "normal"
            
        indicators["risk_status"] = self.risk_status
        indicators["daily_pnl"] = self.daily_pnl
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for HFT strategy.
        
        HFT strategies primarily focus on orderbook dynamics, statistical arbitrage,
        and micro-trend detection for very short-term trades.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators including orderbook metrics
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "long_entry": False,
            "short_entry": False,
            "long_exit": False,
            "short_exit": False,
            "signal_strength": 0.0,
            "strategy_type": None,
            "trade_id": None,
            "entry_price": None,
            "profit_target": None,
            "stop_loss": None,
        }
        
        if not indicators or data.empty:
            return signals
        
        # Rate limiting: Check if we're trading too frequently
        current_time = datetime.utcnow()
        
        # Reset the per-minute trade counter if a minute has passed
        if (current_time - self.last_minute_reset).total_seconds() >= 60:
            self.trades_this_minute = 0
            self.last_minute_reset = current_time
        
        # Enforce trade frequency limits
        if self.trades_this_minute >= self.parameters["max_trades_per_minute"]:
            logger.debug(f"Trade frequency limit reached: {self.trades_this_minute} trades this minute")
            return signals
            
        # Enforce minimum time between trades
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
            return signals
        
        # Enforce risk limits
        if self.risk_status == "high":
            logger.warning(f"Risk status HIGH - daily loss approaching limit. No new trades.")
            # Only generate exit signals in high risk status
            if self.positions:
                signals["long_exit"] = True
                signals["short_exit"] = True
            return signals
        
        # Current price and key metrics
        current_price = indicators.get("current_price", 0)
        if current_price <= 0:
            return signals
        
        # Position count limits
        if len(self.positions) >= self.parameters["max_open_positions"]:
            logger.debug(f"Maximum open positions reached: {len(self.positions)}/{self.parameters['max_open_positions']}")
            return signals
            
        # Check for price spikes/flash crashes - avoid trading during extreme volatility
        if indicators.get("price_spike_detected", False):
            logger.warning(f"Price spike detected - avoiding new trades")
            return signals
        
        # HFT Signal Generation Logic
        # ---------------------------
        
        # 1. Mean Reversion Signals
        if self.mean_reversion_enabled and "z_score" in indicators:
            z_score = indicators["z_score"]
            z_score_entry = self.parameters["z_score_entry"]
            
            # Mean reversion: Sell when price is too high, buy when price is too low
            if z_score > z_score_entry and not self._has_short_position():
                signals["short_entry"] = True
                signals["strategy_type"] = "mean_reversion_short"
                signals["signal_strength"] = min(abs(z_score) / z_score_entry, 1.0)
                logger.info(f"Mean reversion SHORT signal: z-score {z_score:.2f} > {z_score_entry}")
                
            elif z_score < -z_score_entry and not self._has_long_position():
                signals["long_entry"] = True
                signals["strategy_type"] = "mean_reversion_long"
                signals["signal_strength"] = min(abs(z_score) / z_score_entry, 1.0)
                logger.info(f"Mean reversion LONG signal: z-score {z_score:.2f} < -{z_score_entry}")
        
        # 2. Orderbook Imbalance Signals
        if "book_imbalance" in indicators and "bid_ask_spread_pct" in indicators:
            imbalance = indicators["book_imbalance"]
            spread_pct = indicators["bid_ask_spread_pct"]
            imbalance_threshold = self.parameters["imbalance_threshold"]
            min_spread = self.parameters["min_bid_ask_spread_pct"]
            
            # Only consider imbalance opportunities with acceptable spreads
            if spread_pct > min_spread:
                # Strong buying pressure (positive imbalance)
                if imbalance > imbalance_threshold and not self._has_long_position():
                    # Only override existing signals if stronger
                    if not signals["long_entry"] or signals["signal_strength"] < (imbalance / imbalance_threshold):
                        signals["long_entry"] = True
                        signals["short_entry"] = False  # Prioritize this signal
                        signals["strategy_type"] = "orderbook_imbalance_long"
                        signals["signal_strength"] = min(imbalance / imbalance_threshold, 1.0)
                        logger.info(f"Orderbook imbalance LONG signal: {imbalance:.2f} > {imbalance_threshold}")
                
                # Strong selling pressure (negative imbalance)
                elif imbalance < -imbalance_threshold and not self._has_short_position():
                    # Only override existing signals if stronger
                    if not signals["short_entry"] or signals["signal_strength"] < (abs(imbalance) / imbalance_threshold):
                        signals["short_entry"] = True
                        signals["long_entry"] = False  # Prioritize this signal
                        signals["strategy_type"] = "orderbook_imbalance_short"
                        signals["signal_strength"] = min(abs(imbalance) / imbalance_threshold, 1.0)
                        logger.info(f"Orderbook imbalance SHORT signal: {imbalance:.2f} < -{imbalance_threshold}")
        
        # 3. Momentum Signals (if enabled)
        if self.momentum_enabled and len(data) >= 20:
            # Simple momentum calculation using recent price changes
            short_term_returns = data["close"].pct_change(3).iloc[-1] * 100  # 3-period returns
            
            # Strong positive momentum
            if short_term_returns > 0.2 and not self._has_long_position():  # 0.2% threshold
                if not signals["long_entry"] or signals["signal_strength"] < (short_term_returns / 0.2):
                    signals["long_entry"] = True
                    signals["strategy_type"] = "momentum_long"
                    signals["signal_strength"] = min(short_term_returns / 0.2, 1.0)
                    logger.info(f"Momentum LONG signal: returns {short_term_returns:.2f}%")
            
            # Strong negative momentum
            elif short_term_returns < -0.2 and not self._has_short_position():  # -0.2% threshold
                if not signals["short_entry"] or signals["signal_strength"] < (abs(short_term_returns) / 0.2):
                    signals["short_entry"] = True
                    signals["strategy_type"] = "momentum_short"
                    signals["signal_strength"] = min(abs(short_term_returns) / 0.2, 1.0)
                    logger.info(f"Momentum SHORT signal: returns {short_term_returns:.2f}%")
        
        # If a trade signal is generated, set additional parameters
        if signals["long_entry"] or signals["short_entry"]:
            # Generate unique trade ID
            signals["trade_id"] = f"HFT_{self.session.symbol}_{int(time.time() * 1000)}_{self.next_position_id}"
            self.next_position_id += 1
            
            # Set entry price from current price or orderbook
            if "best_ask" in indicators and signals["long_entry"]:
                signals["entry_price"] = indicators["best_ask"]
            elif "best_bid" in indicators and signals["short_entry"]:
                signals["entry_price"] = indicators["best_bid"]
            else:
                signals["entry_price"] = current_price
            
            # Set profit target based on ticks or percentage
            if "tick_size" in indicators and "profit_target" in indicators:
                tick_size = indicators["tick_size"]
                profit_target_ticks = self.parameters["profit_target_ticks"]
                
                if signals["long_entry"]:
                    signals["profit_target"] = signals["entry_price"] + (tick_size * profit_target_ticks)
                    # Use tight stop loss for HFT (usually 1-2 ticks)
                    signals["stop_loss"] = signals["entry_price"] - (tick_size * 2)
                else:  # short
                    signals["profit_target"] = signals["entry_price"] - (tick_size * profit_target_ticks)
                    signals["stop_loss"] = signals["entry_price"] + (tick_size * 2)
        
        # Exit signals for existing positions
        for position in self.positions:
            # Check holding time limits
            hold_time = (current_time - position.open_time).total_seconds() if hasattr(position, "open_time") else 0
            
            # Force exit if position held too long (exceeds max hold time)
            if hold_time > self.max_hold_time:
                if position.direction == "long":
                    signals["long_exit"] = True
                    logger.info(f"Long position held for {hold_time:.1f}s - exceeds max hold time of {self.max_hold_time}s")
                else:
                    signals["short_exit"] = True
                    logger.info(f"Short position held for {hold_time:.1f}s - exceeds max hold time of {self.max_hold_time}s")
            
            # Exit mean reversion trades when price returns to mean
            if "z_score" in indicators and position.metadata.get("strategy_type", "").startswith("mean_reversion"):
                z_score = indicators["z_score"]
                z_score_exit = self.parameters["z_score_exit"]
                
                if position.direction == "long" and z_score >= -z_score_exit:
                    signals["long_exit"] = True
                    logger.info(f"Mean reversion long exit: z-score {z_score:.2f} >= -{z_score_exit}")
                    
                elif position.direction == "short" and z_score <= z_score_exit:
                    signals["short_exit"] = True
                    logger.info(f"Mean reversion short exit: z-score {z_score:.2f} <= {z_score_exit}")
        
        return signals
    
    def _has_long_position(self) -> bool:
        """Check if we have an open long position."""
        return any(p.direction == "long" for p in self.positions)
    
    def _has_short_position(self) -> bool:
        """Check if we have an open short position."""
        return any(p.direction == "short" for p in self.positions)
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size for HFT trades.
        
        HFT strategies typically use small position sizes but execute many trades.
        Position sizing accounts for volatility, spread, and risk limits.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        if data.empty or not indicators:
            return 0.0
        
        # Account balance (in base currency)
        account_balance = 10000.0  # Mock value, would come from exchange API
        
        # Maximum position value as percentage of account
        max_position_value_pct = self.parameters["max_position_value_pct"]
        max_position_value = account_balance * max_position_value_pct
        
        # Current price
        current_price = indicators.get("current_price", 0)
        if current_price <= 0:
            return 0.0
        
        # Get position sizing method
        sizing_method = self.parameters["position_sizing_method"]
        position_size_base = 0.0  # Position size in base currency (e.g., USD)
        
        # Calculate base position size using the selected method
        if sizing_method == "fixed":
            # Simple fixed percentage of account
            position_size_base = max_position_value
            
        elif sizing_method == "volatility_adjusted":
            # Adjust position size based on market volatility
            if "atr" in indicators and indicators["atr"] > 0:
                atr = indicators["atr"]
                # Inverse relationship: higher volatility = smaller position
                volatility_factor = current_price / atr
                position_size_base = max_position_value * min(volatility_factor / 20, 1.0)
            else:
                # Default to half the max if ATR not available
                position_size_base = max_position_value * 0.5
        
        elif sizing_method == "portfolio_percentage":
            # Proportional to remaining risk budget for the day
            daily_loss_pct = self.parameters["max_daily_loss_pct"]
            max_daily_loss = account_balance * (daily_loss_pct / 100.0)
            
            # Calculate remaining risk budget
            remaining_risk_pct = 1.0
            if max_daily_loss > 0:
                remaining_risk_pct = max(0, (max_daily_loss + self.daily_pnl) / max_daily_loss)
            
            # Scale position size by remaining risk
            position_size_base = max_position_value * remaining_risk_pct
        
        # Factor in bid-ask spread - reduce size for wider spreads
        if "bid_ask_spread_pct" in indicators:
            spread_pct = indicators["bid_ask_spread_pct"]
            min_spread = self.parameters["min_bid_ask_spread_pct"]
            
            if spread_pct > min_spread:
                # Reduce position size as spread increases
                spread_factor = min_spread / max(min_spread, spread_pct)
                position_size_base *= spread_factor
        
        # Adjust for signal strength if available
        if "signal_strength" in self.signals and self.signals["signal_strength"] > 0:
            position_size_base *= self.signals["signal_strength"]
        
        # Convert to crypto units
        position_size_crypto = position_size_base / current_price
        
        # Ensure minimum trade size
        min_trade_size = self.session.min_trade_size
        position_size_crypto = max(position_size_crypto, min_trade_size)
        
        # Apply precision appropriate for the asset
        decimals = 8 if self.session.symbol.startswith("BTC") else 6
        position_size_crypto = round(position_size_crypto, decimals)
        
        logger.info(f"HFT position size: {position_size_crypto} {self.session.symbol.split('-')[0]} "
                  f"({sizing_method}, signal strength: {self.signals.get('signal_strength', 0):.2f})")
        
        return position_size_crypto
    
    def _on_orderbook_updated(self, event: Event) -> None:
        """
        Override the base method to be much more responsive to orderbook changes.
        
        For HFT, orderbook changes can trigger immediate trade decisions without
        waiting for price candle completion.
        """
        # Call the base implementation first
        super()._on_orderbook_updated(event)
        
        # Skip if not our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
            
        # Record orderbook update time for latency calculations
        current_time = datetime.utcnow()
        previous_update = self.last_orderbook_update
        self.last_orderbook_update = current_time
        
        # Track orderbook update frequency
        if previous_update:
            update_interval_ms = (current_time - previous_update).total_seconds() * 1000
            if update_interval_ms > 0:
                # Only process if we're getting timely updates
                if update_interval_ms > self.parameters["latency_threshold_ms"] * 2:
                    logger.warning(f"Slow orderbook updates: {update_interval_ms:.0f}ms > "
                                 f"{self.parameters['latency_threshold_ms'] * 2}ms threshold")
        
        # For HFT, we need to react immediately to orderbook changes
        # if we have a position or potential trading opportunity
        if self.is_active and self.orderbook:
            # Only proceed if we have all necessary data
            if not self.market_data.empty:
                # Re-calculate indicators with latest orderbook
                fresh_indicators = self.calculate_indicators(self.market_data)
                
                # Generate signals based on fresh indicators
                fresh_signals = self.generate_signals(self.market_data, fresh_indicators)
                
                # Execute trades if needed
                if fresh_signals.get("long_entry") or fresh_signals.get("short_entry") or \
                   fresh_signals.get("long_exit") or fresh_signals.get("short_exit"):
                    self.signals = fresh_signals
                    self._check_for_trade_opportunities()
                    
                # Process any limit orders that might be executable
                self._check_limit_orders(fresh_indicators)
    
    def _check_limit_orders(self, indicators: Dict[str, Any]) -> None:
        """
        Check if any pending limit orders should be executed or canceled.
        
        In real trading, this would be handled by the exchange, but we simulate it here
        for backtesting purposes.
        """
        if not self.pending_orders or not indicators:
            return
            
        current_time = datetime.utcnow()
        current_ask = indicators.get("best_ask", 0)
        current_bid = indicators.get("best_bid", 0)
        
        if current_ask <= 0 or current_bid <= 0:
            return
            
        # Check each pending order
        executed_orders = []
        canceled_orders = []
        
        for order in self.pending_orders:
            # Skip orders not ready for processing
            if "order_id" not in order or "created_at" not in order:
                continue
                
            # Check for execution timeout
            order_age = (current_time - order["created_at"]).total_seconds()
            if order_age > self.parameters["execution_timeout_seconds"]:
                canceled_orders.append(order["order_id"])
                logger.info(f"Canceling limit order {order['order_id']} due to timeout after {order_age:.1f}s")
                continue
                
            # Check if order can be executed
            if order["side"] == "buy" and "limit_price" in order and current_ask <= order["limit_price"]:
                executed_orders.append(order["order_id"])
                logger.info(f"Executing buy limit order at {order['limit_price']} (current ask: {current_ask})")
                
            elif order["side"] == "sell" and "limit_price" in order and current_bid >= order["limit_price"]:
                executed_orders.append(order["order_id"])
                logger.info(f"Executing sell limit order at {order['limit_price']} (current bid: {current_bid})")
        
        # Remove processed orders
        self.pending_orders = [order for order in self.pending_orders 
                           if order.get("order_id") not in executed_orders + canceled_orders]
        
        # Update counters
        self.cancelled_orders += len(canceled_orders)
    
    def _on_position_opened(self, event: Event) -> None:
        """
        Handle position opened events with HFT-specific tracking.
        """
        super()._on_position_opened(event)
        
        # Update HFT-specific counters
        position_id = event.data.get('position_id')
        if position_id:
            self.trade_count += 1
            self.trades_this_minute += 1
            self.last_trade_time = datetime.utcnow()
            
            # Track execution latency
            if "signal_time" in event.data:
                signal_time = event.data["signal_time"]
                execution_latency_ms = (self.last_trade_time - signal_time).total_seconds() * 1000
                self.execution_latencies.append(execution_latency_ms)
                self.order_execution_times.append(execution_latency_ms)
                
                # Keep histories manageable
                if len(self.execution_latencies) > 100:
                    self.execution_latencies = self.execution_latencies[-100:]
                if len(self.order_execution_times) > 100:
                    self.order_execution_times = self.order_execution_times[-100:]
                    
                logger.info(f"HFT execution latency: {execution_latency_ms:.2f}ms")
    
    def _on_position_closed(self, event: Event) -> None:
        """
        Handle position closed events with HFT-specific tracking.
        """
        super()._on_position_closed(event)
        
        # Track position performance
        position_id = event.data.get('position_id')
        pnl = event.data.get('pnl', 0.0)
        hold_time = event.data.get('hold_time', 0.0)
        
        # Update daily PnL tracking
        self.daily_pnl += pnl
        
        # Record holding time
        if hold_time > 0:
            self.position_hold_times.append(hold_time)
            if len(self.position_hold_times) > 100:
                self.position_hold_times = self.position_hold_times[-100:]
            
            avg_hold_time = sum(self.position_hold_times) / len(self.position_hold_times)
            logger.info(f"Closed position {position_id} with PnL {pnl:.2f}, held for {hold_time:.1f}s "
                      f"(avg: {avg_hold_time:.1f}s)")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        HFT strategies work best in high-liquidity, normal volatility conditions.
        They struggle in extremely volatile or illiquid markets.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "ranging": 0.85,        # Very good in stable, ranging markets
            "volatile": 0.40,       # Poor in highly volatile markets (execution risk)
            "trending": 0.65,       # Moderate in trending markets
            "calm": 0.70,           # Good in calm markets if liquidity is sufficient
            "breakout": 0.30,       # Poor during breakouts (high slippage risk)
            "high_volume": 0.95,    # Excellent during high volume periods
            "low_volume": 0.20,      # Very poor during low volume (execution risk)
            "high_liquidity": 0.95, # Excellent in high liquidity markets
            "low_liquidity": 0.15,  # Very poor in low liquidity markets
            "building_momentum": 0.60, # Moderate during momentum build-up phases
        }
        
        return compatibility_map.get(market_regime, 0.60)  # Default compatibility
