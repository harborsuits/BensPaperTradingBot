#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straddle Trading Strategy Module

This module implements a straddle trading strategy for capturing volatility around 
high-volatility events by purchasing both a call and put at the same strike.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)
from trading_bot.utils.straddle_utils import (
    price_straddle, calculate_iv_rank, calculate_iv_percentile, 
    find_catalyst_events, find_atm_strike, generate_straddle_trade_plan,
    check_recent_gaps
)
from trading_bot.data.market_data import MarketData
from trading_bot.config.straddle_trading_config import STRADDLE_CONFIG

# Setup logging
logger = logging.getLogger(__name__)

class StraddleTradingStrategy(StrategyOptimizable):
    """
    Straddle Trading Strategy designed to profit from large price moves in either direction.
    
    This strategy involves purchasing both a call and put option at the same strike price,
    typically around catalyst events where significant volatility is expected.
    """
    
    def __init__(
        self,
        market_data: MarketData,
        config: Dict = None,
        **kwargs
    ):
        """
        Initialize Straddle Trading strategy.
        
        Args:
            market_data: Market data source
            config: Configuration dictionary (overrides default STRADDLE_CONFIG)
            **kwargs: Additional parameters
        """
        self.name = "Straddle Trading"
        self.description = "Buys call and put options to profit from significant price movements around events"
        self.tags = ["options", "straddle", "volatility", "event-driven"]
        
        # Use default config if none provided
        self.config = config if config else STRADDLE_CONFIG
        
        # Initialize the base class
        super().__init__(market_data, **kwargs)
        
        # Strategy-specific initialization
        self.active_positions = {}
        self.pending_signals = {}
        self.watchlist = self.config.get("watchlist", [])
        self.event_calendar = {}
        self.update_event_calendar()
        
        # Set up optimization parameters
        self.optimization_params = {
            "days_before_event": {
                "type": "int", 
                "min": 1, 
                "max": 10, 
                "default": self.config.get("days_before_event", 5)
            },
            "days_after_event": {
                "type": "int", 
                "min": 1, 
                "max": 15, 
                "default": self.config.get("days_after_event", 7)
            },
            "iv_rank_threshold": {
                "type": "float", 
                "min": 0.0, 
                "max": 100.0, 
                "default": self.config.get("iv_rank_threshold", 50.0)
            },
            "profit_target_pct": {
                "type": "float", 
                "min": 10.0, 
                "max": 100.0, 
                "default": self.config.get("profit_target_pct", 50.0)
            },
            "max_loss_pct": {
                "type": "float", 
                "min": 20.0, 
                "max": 100.0, 
                "default": self.config.get("max_loss_pct", 80.0)
            },
            "position_sizing_pct": {
                "type": "float", 
                "min": 0.5, 
                "max": 5.0, 
                "default": self.config.get("position_sizing_pct", 2.0)
            }
        }
        
        logger.info(f"Initialized Straddle Trading strategy: {self.name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            # Section 3: Option & Strike Selection
            "strike_atm_offset": [-1, 0, 1],
            "min_days_to_expiration": [15, 20, 25, 30],
            "max_days_to_expiration": [25, 30, 35, 40],
            
            # Section 4: Greeks & Risk Filters
            "min_iv_rank": [50, 60, 70, 80],
            "max_iv_percentile": [80, 85, 90, 95],
            "max_theta_burn_daily": [0.005, 0.01, 0.015, 0.02],
            
            # Section 7: Exit Rules
            "profit_take_pct": [50, 60, 70, 80],
            "pre_event_exit_days": [1, 2, 3],
            "post_event_exit_days": [1, 2],
            
            # Section 6: Position Sizing & Risk Controls
            "risk_pct_per_trade": [0.5, 1.0, 1.5, 2.0],
            "max_concurrent_positions": [1, 2, 3, 4],
        }
    
    def _calculate_iv_rank(self, symbol: str, current_iv: float) -> Tuple[float, float]:
        """
        Calculate IV rank and percentile for a symbol.
        
        Args:
            symbol: Symbol to analyze
            current_iv: Current implied volatility
            
        Returns:
            Tuple of (iv_rank, iv_percentile)
        """
        # Check if we have IV history for this symbol
        if symbol not in self.iv_history or not self.iv_history[symbol]:
            # Default to medium values if no history
            return 50.0, 50.0
        
        iv_values = self.iv_history[symbol]
        iv_min = min(iv_values)
        iv_max = max(iv_values)
        
        # Calculate IV rank (where is current IV in its historical range)
        if iv_max == iv_min:
            iv_rank = 50.0  # Default to middle if no range
        else:
            iv_rank = 100.0 * (current_iv - iv_min) / (iv_max - iv_min)
        
        # Calculate IV percentile (what percentage of historical IVs are below current)
        iv_percentile = 100.0 * sum(1 for iv in iv_values if iv < current_iv) / len(iv_values)
        
        return iv_rank, iv_percentile
    
    def _update_iv_history(self, symbol: str, iv_value: float, max_history: int = 252):
        """
        Update IV history for a symbol.
        
        Args:
            symbol: Symbol to update
            iv_value: New IV value to add
            max_history: Maximum length of history to maintain
        """
        if symbol not in self.iv_history:
            self.iv_history[symbol] = []
        
        self.iv_history[symbol].append(iv_value)
        
        # Limit history length
        if len(self.iv_history[symbol]) > max_history:
            self.iv_history[symbol] = self.iv_history[symbol][-max_history:]
    
    def _find_atm_strike(self, symbol: str, current_price: float, option_chain: Any) -> float:
        """
        Find at-the-money strike based on current price.
        
        Args:
            symbol: Symbol to find strike for
            current_price: Current price of underlying
            option_chain: Option chain data (placeholder)
            
        Returns:
            ATM strike price
        """
        # TODO: Implement proper ATM strike selection from option chain
        
        # Placeholder implementation: find nearest strike based on typical strike intervals
        if current_price < 10:
            strike_interval = 0.5
        elif current_price < 50:
            strike_interval = 1.0
        elif current_price < 100:
            strike_interval = 2.5
        elif current_price < 200:
            strike_interval = 5.0
        else:
            strike_interval = 10.0
        
        # Round to nearest interval
        base_strike = round(current_price / strike_interval) * strike_interval
        
        # Apply offset if configured
        offset = self.parameters.get("strike_atm_offset", 0)
        return base_strike + (offset * strike_interval)
    
    def _select_expiration(self, symbol: str, current_date: datetime, event_date: datetime, option_chain: Any) -> datetime:
        """
        Select appropriate expiration based on DTE criteria and event date.
        
        Args:
            symbol: Symbol to find expiration for
            current_date: Current date
            event_date: Date of the catalyst event
            option_chain: Option chain data (placeholder)
            
        Returns:
            Selected expiration date
        """
        # TODO: Implement proper expiration selection using actual option chain data
        
        # Get DTE parameters
        min_dte = self.parameters.get("min_days_to_expiration", 20)
        max_dte = self.parameters.get("max_days_to_expiration", 30)
        
        # Calculate days to event
        days_to_event = (event_date - current_date).days
        
        # If event is beyond the max DTE window, select based on max DTE
        if days_to_event > max_dte:
            target_dte = max_dte
        # If event is within our DTE window, select expiration after the event
        elif min_dte <= days_to_event <= max_dte:
            target_dte = days_to_event + 5  # Add buffer days after event
        # If event is too soon, use min DTE
        else:
            target_dte = min_dte
        
        # Mock expiration date based on target DTE
        return current_date + timedelta(days=target_dte)
    
    def _calculate_greeks(self, symbol: str, current_price: float, strike: float, 
                        days_to_expiration: int, volatility: float, 
                        option_type: str) -> Dict[str, float]:
        """
        Calculate option greeks.
        
        Args:
            symbol: Symbol for the option
            current_price: Current price of underlying
            strike: Strike price
            days_to_expiration: Days to expiration
            volatility: Implied volatility
            option_type: Option type ('call' or 'put')
            
        Returns:
            Dictionary with calculated greeks
        """
        # TODO: Implement proper greeks calculation using Black-Scholes or other model
        
        # Placeholder implementation with simplified calculations
        T = days_to_expiration / 365.0  # Convert days to years
        
        # Very simplified delta approximation
        if option_type == 'call':
            delta = 0.5 + 0.5 * (current_price - strike) / (volatility * current_price * np.sqrt(T))
            delta = max(0.0, min(1.0, delta))  # Constrain between 0 and 1
        else:  # put
            delta = -0.5 - 0.5 * (current_price - strike) / (volatility * current_price * np.sqrt(T))
            delta = max(-1.0, min(0.0, delta))  # Constrain between -1 and 0
        
        # Simplified vega approximation (sensitivity to volatility changes)
        vega = 0.01 * current_price * np.sqrt(T)
        
        # Simplified theta approximation (time decay, negative value)
        theta = -0.01 * current_price * volatility / np.sqrt(T)
        
        # Simplified gamma approximation (delta's sensitivity to price changes)
        gamma = 0.01 / (current_price * volatility * np.sqrt(T))
        
        return {
            "delta": delta,
            "vega": vega,
            "theta": theta,
            "gamma": gamma
        }
    
    def _calculate_straddle_price(self, symbol: str, current_price: float, strike: float,
                                days_to_expiration: int, volatility: float) -> Dict[str, Any]:
        """
        Calculate straddle price and characteristics.
        
        Args:
            symbol: Symbol for the straddle
            current_price: Current price of underlying
            strike: Strike price
            days_to_expiration: Days to expiration
            volatility: Implied volatility
            
        Returns:
            Dictionary with straddle pricing details
        """
        # TODO: Implement proper option pricing for straddle
        
        # Placeholder pricing implementation
        T = days_to_expiration / 365.0  # Convert days to years
        
        # Basic Black-Scholes inspired formula for ATM options
        atm_price_factor = 0.4 * volatility * np.sqrt(T)
        call_price = current_price * atm_price_factor
        put_price = current_price * atm_price_factor
        
        # Calculate combined premium
        straddle_price = call_price + put_price
        
        # Calculate call greeks
        call_greeks = self._calculate_greeks(
            symbol, current_price, strike, days_to_expiration, volatility, 'call'
        )
        
        # Calculate put greeks
        put_greeks = self._calculate_greeks(
            symbol, current_price, strike, days_to_expiration, volatility, 'put'
        )
        
        # Calculate combined greeks
        combined_greeks = {
            "delta": call_greeks["delta"] + put_greeks["delta"],
            "vega": call_greeks["vega"] + put_greeks["vega"],
            "theta": call_greeks["theta"] + put_greeks["theta"],
            "gamma": call_greeks["gamma"] + put_greeks["gamma"]
        }
        
        # Calculate breakeven moves
        breakeven_pct = straddle_price / current_price
        
        return {
            "call_price": call_price,
            "put_price": put_price,
            "straddle_price": straddle_price,
            "call_greeks": call_greeks,
            "put_greeks": put_greeks,
            "combined_greeks": combined_greeks,
            "breakeven_pct": breakeven_pct
        }
    
    def _has_recent_gaps(self, data: pd.DataFrame, days: int, threshold: float) -> bool:
        """
        Check if there have been extreme price gaps in recent days.
        
        Args:
            data: DataFrame with price data
            days: Number of days to check
            threshold: Gap threshold as fraction of price
            
        Returns:
            Boolean indicating if there were recent gaps
        """
        if len(data) < days + 1:
            return False
        
        recent_data = data.iloc[-days-1:]
        
        for i in range(1, len(recent_data)):
            prev_close = recent_data.iloc[i-1]['close']
            current_open = recent_data.iloc[i]['open']
            gap_pct = abs(current_open - prev_close) / prev_close
            
            if gap_pct > threshold:
                return True
        
        return False
    
    def _check_straddle_eligibility(self, symbol: str, data: pd.DataFrame, 
                                  current_iv: float, option_chain: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a symbol is eligible for a straddle trade.
        
        Args:
            symbol: Symbol to check
            data: DataFrame with price data
            current_iv: Current implied volatility
            option_chain: Option chain data (placeholder)
            
        Returns:
            Tuple of (is_eligible, reason_details)
        """
        reasons = {}
        
        # Check if there's a catalyst event scheduled
        if symbol not in self.event_calendar:
            reasons["catalyst"] = "No catalyst event scheduled"
            return False, reasons
        
        # Get current price
        if data.empty:
            reasons["data"] = "Insufficient price data"
            return False, reasons
            
        current_price = data.iloc[-1]['close']
        
        # Check for recent gaps
        days = self.parameters.get("gap_check_days", 3)
        threshold = self.parameters.get("max_gap_percentage", 0.04)
        if self._has_recent_gaps(data, days, threshold):
            reasons["gaps"] = "Recent extreme price gaps detected"
            return False, reasons
        
        # Calculate IV rank and percentile
        iv_rank, iv_percentile = self._calculate_iv_rank(symbol, current_iv)
        
        # Check IV rank
        min_iv_rank = self.parameters.get("min_iv_rank", 60)
        if iv_rank < min_iv_rank:
            reasons["iv_rank"] = f"IV rank too low: {iv_rank:.1f}% < {min_iv_rank}%"
            return False, reasons
        
        # Check IV percentile
        max_iv_percentile = self.parameters.get("max_iv_percentile", 90)
        if iv_percentile > max_iv_percentile:
            reasons["iv_percentile"] = f"IV percentile too high: {iv_percentile:.1f}% > {max_iv_percentile}%"
            return False, reasons
        
        # Find ATM strike
        strike = self._find_atm_strike(symbol, current_price, option_chain)
        
        # Check option liquidity
        # TODO: Implement actual option liquidity check with option chain data
        
        # Calculate straddle price and characteristics
        event_date = self.event_calendar[symbol][0]["date"]
        current_date = pd.Timestamp.now()
        days_to_expiration = (self._select_expiration(symbol, current_date, event_date, option_chain) - current_date).days
        
        straddle_details = self._calculate_straddle_price(
            symbol, current_price, strike, days_to_expiration, current_iv
        )
        
        # Check breakeven move target
        min_target = self.parameters.get("breakeven_move_target_min", 0.05)
        max_target = self.parameters.get("breakeven_move_target_max", 0.08)
        breakeven_pct = straddle_details["breakeven_pct"]
        
        if breakeven_pct < min_target:
            reasons["breakeven_too_small"] = f"Breakeven % too small: {breakeven_pct:.1%} < {min_target:.1%}"
            return False, reasons
            
        if breakeven_pct > max_target:
            reasons["breakeven_too_large"] = f"Breakeven % too large: {breakeven_pct:.1%} > {max_target:.1%}"
            return False, reasons
        
        # Check theta burn rate
        max_theta_daily = self.parameters.get("max_theta_burn_daily", 0.01)
        theta_pct = abs(straddle_details["combined_greeks"]["theta"]) / straddle_details["straddle_price"]
        
        if theta_pct > max_theta_daily:
            reasons["theta_too_high"] = f"Daily theta burn too high: {theta_pct:.2%} > {max_theta_daily:.2%}"
            return False, reasons
        
        # Check delta neutrality
        delta_threshold = self.parameters.get("delta_neutrality_threshold", 0.05)
        delta = abs(straddle_details["combined_greeks"]["delta"])
        
        if delta > delta_threshold:
            reasons["delta_imbalance"] = f"Delta not neutral enough: {delta:.3f} > {delta_threshold:.3f}"
            return False, reasons
        
        # Check vega
        min_vega = self.parameters.get("min_vega", 0.15)
        vega_normalized = straddle_details["combined_greeks"]["vega"] / current_price
        
        if vega_normalized < min_vega:
            reasons["vega_too_low"] = f"Vega too low: {vega_normalized:.3f} < {min_vega:.3f}"
            return False, reasons
        
        # All checks passed, return eligibility and details
        eligible_details = {
            "strike": strike,
            "expiration": self._select_expiration(symbol, current_date, event_date, option_chain),
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "straddle_price": straddle_details["straddle_price"],
            "breakeven_pct": breakeven_pct,
            "combined_greeks": straddle_details["combined_greeks"],
            "call_price": straddle_details["call_price"],
            "put_price": straddle_details["put_price"]
        }
        
        return True, eligible_details
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate indicators for straddle trading strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate implied volatility (placeholder - in a real system, you'd use option chain data)
                # For now, estimate IV from historical volatility
                close_returns = np.log(df['close'] / df['close'].shift(1))
                hist_volatility = close_returns.rolling(window=20).std() * np.sqrt(252)
                
                # Artificial IV calculation for demo - normally you'd fetch this from option data
                iv_estimate = 1.2 * hist_volatility  # IV typically exceeds realized vol
                
                # Update IV history with the latest value
                self._update_iv_history(symbol, iv_estimate.iloc[-1])
                
                # Calculate IV rank and percentile
                iv_rank, iv_percentile = self._calculate_iv_rank(symbol, iv_estimate.iloc[-1])
                
                # Calculate daily price changes to detect gaps
                daily_change_pct = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                abs_daily_change = abs(daily_change_pct)
                
                # Store indicators
                indicators[symbol] = {
                    "hist_volatility": pd.DataFrame({"hist_volatility": hist_volatility}),
                    "implied_volatility": pd.DataFrame({"implied_volatility": iv_estimate}),
                    "iv_rank": pd.DataFrame({"iv_rank": pd.Series([iv_rank] * len(df), index=df.index)}),
                    "iv_percentile": pd.DataFrame({"iv_percentile": pd.Series([iv_percentile] * len(df), index=df.index)}),
                    "daily_gap_pct": pd.DataFrame({"daily_gap_pct": daily_change_pct}),
                    "abs_daily_gap": pd.DataFrame({"abs_daily_gap": abs_daily_change})
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, current_date: datetime = None) -> List[Dict]:
        """
        Generate trading signals based on the straddle strategy rules.
        
        Args:
            current_date: Current date for signal generation
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        current_date = current_date or datetime.now()
        
        # Update event calendar if needed
        if not self.event_calendar or self._is_calendar_outdated():
            self.update_event_calendar()
        
        # Check for entry opportunities based on upcoming events
        for symbol, events in self.event_calendar.items():
            # Skip if we already have an active position for this symbol
            if symbol in self.active_positions:
                continue
                
            for event in events:
                event_date = event.get("date")
                event_type = event.get("type")
                
                # Skip if the event is in the past
                if event_date < current_date:
                    continue
                
                # Check if it's time to enter based on days before event
                days_until_event = (event_date - current_date).days
                days_before_event = self.config.get("days_before_event", 5)
                
                if days_until_event == days_before_event:
                    # Check IV conditions
                    current_price = self.market_data.get_latest_price(symbol)
                    historical_iv = self.market_data.get_historical_iv(symbol, days=180)
                    
                    if historical_iv is not None and len(historical_iv) > 0:
                        current_iv = historical_iv[-1] if len(historical_iv) > 0 else None
                        iv_rank = calculate_iv_rank(current_iv, historical_iv)
                        iv_percentile = calculate_iv_percentile(current_iv, historical_iv)
                        
                        # Only trade if IV rank is below our threshold (expecting IV expansion)
                        iv_rank_threshold = self.config.get("iv_rank_threshold", 50.0)
                        if iv_rank < iv_rank_threshold:
                            # Find the appropriate ATM strike
                            atm_strike = find_atm_strike(
                                symbol=symbol,
                                current_price=current_price,
                                offset_pct=self.config.get("strike_offset_pct", 0)
                            )
                            
                            # Calculate days for option expiration (after event)
                            days_after_event = self.config.get("days_after_event", 7)
                            expiration_date = event_date + timedelta(days=days_after_event)
                            
                            # Check for recent gaps that might indicate elevated volatility
                            recent_gaps = check_recent_gaps(
                                symbol=symbol,
                                lookback_days=self.config.get("gap_lookback_days", 20),
                                threshold_pct=self.config.get("gap_threshold_pct", 3.0)
                            )
                            
                            # Generate trade plan with all necessary details
                            trade_plan = generate_straddle_trade_plan(
                                symbol=symbol,
                                current_price=current_price,
                                atm_strike=atm_strike,
                                event_date=event_date,
                                event_type=event_type,
                                expiration_date=expiration_date,
                                current_iv=current_iv,
                                iv_rank=iv_rank,
                                iv_percentile=iv_percentile,
                                position_sizing_pct=self.config.get("position_sizing_pct", 2.0),
                                account_size=self.config.get("account_size", 100000),
                                recent_gaps=recent_gaps
                            )
                            
                            # Create the signal
                            signal = {
                                "symbol": symbol,
                                "type": "straddle_entry",
                                "direction": "neutral",
                                "timestamp": current_date,
                                "event_date": event_date,
                                "event_type": event_type,
                                "expiration_date": expiration_date,
                                "strike": atm_strike,
                                "iv_rank": iv_rank,
                                "iv_percentile": iv_percentile,
                                "trade_plan": trade_plan
                            }
                            
                            signals.append(signal)
                            self.pending_signals[symbol] = signal
                            logger.info(f"Generated straddle entry signal for {symbol} around {event_type} on {event_date}")
        
        # Check for exit opportunities on active positions
        for symbol, position in list(self.active_positions.items()):
            entry_price = position.get("entry_price", 0)
            entry_date = position.get("entry_date")
            event_date = position.get("event_date")
            strike = position.get("strike")
            profit_target = position.get("profit_target")
            stop_loss = position.get("stop_loss")
            expiration_date = position.get("expiration_date")
            
            # Current price of the straddle
            current_straddle_price = price_straddle(
                symbol=symbol,
                strike=strike,
                expiration_date=expiration_date,
                current_date=current_date
            ).get("price", 0)
            
            # Calculate current P&L
            pnl_pct = ((current_straddle_price / entry_price) - 1) * 100
            
            # Check exit conditions
            exit_signal = None
            
            # 1. Profit target reached
            if pnl_pct >= profit_target:
                exit_signal = {
                    "symbol": symbol,
                    "type": "straddle_exit",
                    "reason": "profit_target",
                    "timestamp": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_straddle_price,
                    "pnl_pct": pnl_pct
                }
            
            # 2. Stop loss reached
            elif pnl_pct <= -stop_loss:
                exit_signal = {
                    "symbol": symbol,
                    "type": "straddle_exit",
                    "reason": "stop_loss",
                    "timestamp": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_straddle_price,
                    "pnl_pct": pnl_pct
                }
            
            # 3. One day before event (IV expansion may have occurred)
            elif event_date and (event_date - current_date).days == 1:
                # Only exit if we've made a profit
                if pnl_pct > 0:
                    exit_signal = {
                        "symbol": symbol,
                        "type": "straddle_exit",
                        "reason": "pre_event_profit",
                        "timestamp": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_straddle_price,
                        "pnl_pct": pnl_pct
                    }
            
            # 4. After event management (1 day after event)
            elif event_date and (current_date - event_date).days == 1:
                # Evaluate post-event exit strategy
                exit_signal = {
                    "symbol": symbol,
                    "type": "straddle_exit",
                    "reason": "post_event",
                    "timestamp": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_straddle_price,
                    "pnl_pct": pnl_pct
                }
            
            # 5. Approaching expiration (2 days before)
            elif expiration_date and (expiration_date - current_date).days <= 2:
                exit_signal = {
                    "symbol": symbol,
                    "type": "straddle_exit",
                    "reason": "expiration_approach",
                    "timestamp": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_straddle_price,
                    "pnl_pct": pnl_pct
                }
            
            if exit_signal:
                signals.append(exit_signal)
                # Remove from active positions on exit signal
                del self.active_positions[symbol]
                logger.info(f"Generated straddle exit signal for {symbol}: {exit_signal['reason']} with PNL {pnl_pct:.2f}%")
        
        return signals
    
    def update_event_calendar(self) -> None:
        """
        Updates the event calendar with upcoming catalyst events for the watchlist.
        """
        self.event_calendar = {}
        for symbol in self.watchlist:
            events = find_catalyst_events(
                symbol=symbol,
                days_forward=self.config.get("event_calendar_days_forward", 30),
                include_earnings=self.config.get("include_earnings", True),
                include_economic=self.config.get("include_economic", True),
                include_corporate=self.config.get("include_corporate", True)
            )
            if events:
                self.event_calendar[symbol] = events
                logger.info(f"Found {len(events)} events for {symbol}: {events}")
    
    def process_fills(self, fills: List[Dict]) -> None:
        """
        Process execution fills to update active positions.
        
        Args:
            fills: List of fill dictionaries
        """
        for fill in fills:
            symbol = fill.get("symbol")
            fill_type = fill.get("type")
            
            if fill_type == "straddle_entry":
                # Move from pending to active positions
                if symbol in self.pending_signals:
                    signal = self.pending_signals[symbol]
                    
                    self.active_positions[symbol] = {
                        "entry_date": fill.get("timestamp"),
                        "entry_price": fill.get("price"),
                        "strike": signal.get("strike"),
                        "expiration_date": signal.get("expiration_date"),
                        "event_date": signal.get("event_date"),
                        "event_type": signal.get("event_type"),
                        "profit_target": self.config.get("profit_target_pct", 50.0),
                        "stop_loss": self.config.get("max_loss_pct", 80.0)
                    }
                    
                    # Remove from pending signals
                    del self.pending_signals[symbol]
                    logger.info(f"Straddle position opened for {symbol} at {fill.get('price')}")
            
            elif fill_type == "straddle_exit":
                # Position already removed in generate_signals when exit signal created
                logger.info(f"Straddle position closed for {symbol} at {fill.get('price')}")
    
    def _is_calendar_outdated(self) -> bool:
        """
        Check if the event calendar needs to be updated.
        
        Returns:
            Boolean indicating if calendar needs update
        """
        if not self.event_calendar:
            return True
            
        # Check if any events are in the past
        current_date = datetime.now()
        for symbol, events in self.event_calendar.items():
            has_future_events = any(event.get("date") > current_date for event in events)
            if not has_future_events:
                return True
                
        # Check if calendar is older than refresh interval
        refresh_days = self.config.get("calendar_refresh_days", 7)
        if hasattr(self, '_last_calendar_update'):
            days_since_update = (current_date - self._last_calendar_update).days
            if days_since_update >= refresh_days:
                return True
        else:
            # First time checking - set the attribute and update
            self._last_calendar_update = current_date
            return True
            
        return False
    
    def optimize(self, historical_data: Dict, metric: str = "sharpe_ratio") -> Dict:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            historical_data: Dictionary containing historical market data
            metric: Performance metric to optimize for
            
        Returns:
            Dictionary with optimized parameters
        """
        # Implementation of optimization algorithm
        logger.info(f"Optimizing straddle strategy for {metric}")
        
        # Placeholder for actual optimization logic
        # In a real implementation, this would test different parameter combinations
        # and return the best performing set
        
        # Return default parameters as placeholder
        return {param: details["default"] for param, details in self.optimization_params.items()}
    
    def get_parameters(self) -> Dict:
        """
        Get the current strategy parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {param: details["default"] for param, details in self.optimization_params.items()}
    
    def set_parameters(self, parameters: Dict) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        for param, value in parameters.items():
            if param in self.optimization_params:
                param_details = self.optimization_params[param]
                
                # Apply limits
                if param_details["type"] == "int":
                    value = int(max(param_details["min"], min(param_details["max"], value)))
                elif param_details["type"] == "float":
                    value = float(max(param_details["min"], min(param_details["max"], value)))
                
                # Update config
                self.config[param] = value
        
        logger.info(f"Updated strategy parameters: {parameters}")

# TODO: Implement function to fetch actual option chain data
# TODO: Implement function to calculate proper option pricing and greeks
# TODO: Implement function to track actual catalyst events
# TODO: Implement function for ML-based entry and exit optimization
# TODO: Implement function for dynamic hedging and position adjustment 