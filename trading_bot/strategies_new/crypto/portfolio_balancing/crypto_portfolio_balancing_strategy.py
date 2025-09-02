#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Portfolio Balancing Strategy Module

This module implements a portfolio balancing strategy for cryptocurrencies, which focuses
on maintaining a diversified portfolio of crypto assets with regular rebalancing to
target allocations. The strategy can use various allocation methodologies including equal-weight,
market-cap-weighted, volatility-weighted, and custom allocations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import json

# Import base strategy
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoPortfolioBalancingStrategy",
    market_type="crypto",
    description="A portfolio balancing strategy for cryptocurrencies that maintains target allocations across multiple assets with regular rebalancing",
    timeframes=["1d", "1w", "1M"],
    parameters={
        # Core parameters
        "allocation_method": {
            "type": "str",
            "default": "equal_weight",
            "enum": ["equal_weight", "market_cap_weight", "volatility_weight", "custom"],
            "description": "Method to determine asset allocations"
        },
        "custom_allocations": {
            "type": "dict",
            "default": {},
            "description": "Dictionary of {asset_symbol: allocation_weight} for custom allocations"
        },
        "rebalance_threshold_pct": {
            "type": "float",
            "default": 0.05,
            "description": "Percentage deviation from target allocation that triggers rebalancing"
        },
        "rebalance_frequency_days": {
            "type": "int",
            "default": 30,
            "description": "Maximum days between rebalances regardless of threshold"
        },
        "min_rebalance_interval_days": {
            "type": "int",
            "default": 7,
            "description": "Minimum days between rebalances to avoid excessive trading"
        },
        "assets": {
            "type": "list",
            "default": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"],
            "description": "List of crypto asset symbols to include in portfolio"
        },
        "max_assets": {
            "type": "int",
            "default": 10,
            "description": "Maximum number of assets to include in the portfolio"
        },
        "min_allocation_pct": {
            "type": "float",
            "default": 0.05,
            "description": "Minimum allocation percentage for any asset"
        },
        "max_allocation_pct": {
            "type": "float",
            "default": 0.40,
            "description": "Maximum allocation percentage for any asset"
        },
        "enable_volatility_targeting": {
            "type": "bool",
            "default": True,
            "description": "Whether to adjust allocations based on asset volatility"
        },
        "volatility_lookback_days": {
            "type": "int",
            "default": 90,
            "description": "Number of days to look back for volatility calculation"
        },
        "volatility_weighting_factor": {
            "type": "float",
            "default": 1.0,
            "description": "Factor for weighting volatility in allocation (0-2, higher means more impact)"
        },
        "enable_momentum_tilting": {
            "type": "bool",
            "default": True,
            "description": "Whether to tilt allocations toward assets with positive momentum"
        },
        "momentum_lookback_days": {
            "type": "int",
            "default": 30,
            "description": "Number of days to look back for momentum calculation"
        },
        "momentum_weighting_factor": {
            "type": "float",
            "default": 0.5,
            "description": "Factor for weighting momentum in allocation (0-1, higher means more impact)"
        },
        "min_trade_value": {
            "type": "float",
            "default": 50.0,
            "description": "Minimum USD value for a rebalancing trade"
        },
        "target_portfolio_value_usd": {
            "type": "float",
            "default": 10000.0,
            "description": "Target USD value of the portfolio for allocation calculations"
        },
        "allow_new_assets": {
            "type": "bool",
            "default": True,
            "description": "Whether to allow purchasing assets not currently in portfolio"
        },
        "allow_asset_removal": {
            "type": "bool",
            "default": True,
            "description": "Whether to allow complete removal of assets from portfolio"
        },
        # Risk management parameters
        "max_rebalance_value_pct": {
            "type": "float",
            "default": 0.20,
            "description": "Maximum portfolio percentage to rebalance in a single period"
        },
        "enable_correlation_balancing": {
            "type": "bool",
            "default": True,
            "description": "Whether to consider correlations between assets when balancing"
        },
        "correlation_lookback_days": {
            "type": "int",
            "default": 90,
            "description": "Number of days to look back for correlation calculation"
        },
        "enable_drawdown_protection": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable portfolio protection during market-wide drawdowns"
        },
        "market_drawdown_threshold": {
            "type": "float",
            "default": 0.15,
            "description": "Market drawdown percentage that triggers protection measures"
        },
        "drawdown_cash_allocation": {
            "type": "float",
            "default": 0.30,
            "description": "Percentage to allocate to cash/stablecoins during drawdowns"
        },
        "stablecoin_symbols": {
            "type": "list",
            "default": ["USDT-USD", "USDC-USD", "DAI-USD"],
            "description": "List of stablecoin symbols to use for cash portion"
        },
        # Performance tracking parameters
        "track_vs_benchmark": {
            "type": "bool",
            "default": True,
            "description": "Whether to track performance against benchmark"
        },
        "benchmark_symbol": {
            "type": "str",
            "default": "BTC-USD",
            "description": "Symbol to use as benchmark for performance comparison"
        },
        "enable_reallocation_on_regime_change": {
            "type": "bool",
            "default": True,
            "description": "Whether to trigger reallocation when market regime changes significantly"
        },
    },
    asset_classes=["crypto"],
    timeframes=["1d", "1w", "1M"]
)
class CryptoPortfolioBalancingStrategy(CryptoBaseStrategy):
    """
    A cryptocurrency portfolio balancing strategy that maintains target allocations
    across multiple assets with regular rebalancing.
    
    The strategy can use various allocation methodologies including equal-weight,
    market-cap-weighted, volatility-weighted, and custom allocations, with options
    for volatility targeting, momentum tilting, and correlation balancing.
    """
    
    def __init__(self, session: CryptoSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Portfolio Balancing strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize strategy-specific state variables
        self.asset_data = {}  # Dict of asset data {symbol: {"price": float, "market_cap": float, etc.}}
        self.target_allocations = {}  # Dict of target allocations {symbol: allocation_pct}
        self.current_allocations = {}  # Dict of current allocations {symbol: allocation_pct}
        self.last_rebalance_date = None
        self.rebalance_history = []  # List of rebalance events
        self.correlation_matrix = None
        self.volatility_data = {}  # Dict of asset volatilities {symbol: volatility}
        self.momentum_data = {}  # Dict of asset momentum {symbol: momentum_score}
        self.portfolio_value = self.parameters["target_portfolio_value_usd"]
        self.asset_positions = {}  # Dict of asset positions {symbol: {"quantity": float, "value_usd": float}}
        
        # Market state tracking
        self.market_regime = "normal"  # Can be "normal", "trending", "volatile", "drawdown"
        self.market_drawdown = 0.0
        self.benchmark_price_history = []
        self.benchmark_ath = 0.0
        
        # Performance metrics
        self.portfolio_returns = []  # List of {date, value, return_pct}
        self.benchmark_returns = []  # List of {date, value, return_pct}
        
        # Initialize asset tracking
        self._initialize_asset_tracking()
        
        # Register for additional events
        self.register_event_handler("market_data_updated", self._on_market_data_updated)
        self.register_event_handler("market_regime_changed", self._on_market_regime_changed)
        
        logger.info(f"CryptoPortfolioBalancingStrategy initialized with "
                   f"{len(self.parameters['assets'])} assets and "
                   f"{self.parameters['allocation_method']} allocation method")
    
    def _initialize_asset_tracking(self) -> None:
        """Initialize asset tracking for all assets in the portfolio."""
        for symbol in self.parameters["assets"]:
            self.asset_data[symbol] = {
                "price": 0.0,
                "market_cap": 0.0,
                "volume": 0.0,
                "last_updated": None,
                "available_data": False
            }
            
            # Request market data for this asset
            self.event_bus.publish(Event("request_market_data", {
                "symbol": symbol,
                "timeframe": self.session.timeframe,
                "lookback_periods": max(
                    self.parameters["volatility_lookback_days"],
                    self.parameters["correlation_lookback_days"],
                    self.parameters["momentum_lookback_days"]
                )
            }))
        
        # Initialize with placeholder equal allocations
        equal_allocation = 1.0 / len(self.parameters["assets"])
        for symbol in self.parameters["assets"]:
            self.target_allocations[symbol] = equal_allocation
            self.current_allocations[symbol] = 0.0  # Will be updated with actual data
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updates for portfolio assets.
        
        Args:
            event: Market data updated event
        """
        data = event.data
        symbol = data.get("symbol")
        
        # Only process data for assets we're tracking
        if symbol not in self.asset_data:
            return
        
        # Update asset data
        price = data.get("price")
        market_cap = data.get("market_cap")
        volume = data.get("volume")
        
        if price is not None:
            self.asset_data[symbol]["price"] = price
            self.asset_data[symbol]["last_updated"] = datetime.now()
            self.asset_data[symbol]["available_data"] = True
            
            if market_cap is not None:
                self.asset_data[symbol]["market_cap"] = market_cap
            
            if volume is not None:
                self.asset_data[symbol]["volume"] = volume
            
            logger.debug(f"Updated market data for {symbol}: price=${price:.2f}")
        
        # If this is our benchmark, update benchmark tracking
        if symbol == self.parameters["benchmark_symbol"] and self.parameters["track_vs_benchmark"]:
            self._update_benchmark_tracking(price)
        
        # Check if we have data for all assets
        all_assets_updated = all(self.asset_data[s]["available_data"] for s in self.asset_data)
        
        if all_assets_updated:
            # Calculate volatility, momentum, and correlations if we have enough data
            if len(data.get("ohlcv_data", [])) >= self.parameters["volatility_lookback_days"]:
                self._calculate_asset_metrics(symbol, data.get("ohlcv_data"))
            
            # Update current allocations based on latest prices
            self._update_current_allocations()
            
            # Check if rebalancing is needed
            self._check_rebalance_needed()
    
    def _on_market_regime_changed(self, event: Event) -> None:
        """
        Handle market regime change events.
        
        Args:
            event: Market regime changed event
        """
        if not self.parameters["enable_reallocation_on_regime_change"]:
            return
        
        new_regime = event.data.get("regime")
        if new_regime and new_regime != self.market_regime:
            old_regime = self.market_regime
            self.market_regime = new_regime
            
            logger.info(f"Market regime changed from {old_regime} to {new_regime}, "
                      f"triggering portfolio reallocation")
            
            # Force reallocation on significant regime changes
            significant_change = (
                (old_regime in ["normal", "trending"] and new_regime in ["volatile", "drawdown"]) or
                (old_regime in ["volatile", "drawdown"] and new_regime in ["normal", "trending"])
            )
            
            if significant_change:
                self._recalculate_target_allocations(force=True)
    
    def _update_benchmark_tracking(self, price: float) -> None:
        """
        Update benchmark price tracking and calculate drawdown.
        
        Args:
            price: Current benchmark price
        """
        if price <= 0:
            return
        
        # Update ATH if needed
        if price > self.benchmark_ath:
            self.benchmark_ath = price
        
        # Calculate current drawdown
        if self.benchmark_ath > 0:
            self.market_drawdown = (self.benchmark_ath - price) / self.benchmark_ath
        
        # Add to price history
        self.benchmark_price_history.append({
            "date": datetime.now(),
            "price": price,
            "drawdown": self.market_drawdown
        })
        
        # Limit history size
        max_history = 1000
        if len(self.benchmark_price_history) > max_history:
            self.benchmark_price_history = self.benchmark_price_history[-max_history:]
        
        # Check if we're in a significant drawdown
        if (self.parameters["enable_drawdown_protection"] and 
            self.market_drawdown >= self.parameters["market_drawdown_threshold"]):
            if self.market_regime != "drawdown":
                self.market_regime = "drawdown"
                logger.info(f"Entered market drawdown regime (drawdown: {self.market_drawdown:.2%}), "
                          f"adjusting allocations for protection")
                self._recalculate_target_allocations(force=True)
        elif self.market_regime == "drawdown" and self.market_drawdown < self.parameters["market_drawdown_threshold"] * 0.7:
            # Exit drawdown mode when recovered to 70% of the threshold
            self.market_regime = "normal"
            logger.info(f"Exited market drawdown regime (drawdown: {self.market_drawdown:.2%}), "
                      f"restoring normal allocations")
            self._recalculate_target_allocations(force=True)
    
    def _calculate_asset_metrics(self, symbol: str, ohlcv_data: List[Dict[str, Any]]) -> None:
        """
        Calculate volatility, momentum, and update correlation matrix for an asset.
        
        Args:
            symbol: Asset symbol
            ohlcv_data: OHLCV data for the asset
        """
        # Extract close prices from OHLCV data
        if not ohlcv_data:
            return
        
        # Convert to DataFrame if it's not already
        if not isinstance(ohlcv_data, pd.DataFrame):
            df = pd.DataFrame(ohlcv_data)
        else:
            df = ohlcv_data
        
        if "close" not in df.columns:
            return
        
        # Calculate volatility
        volatility_window = min(len(df), self.parameters["volatility_lookback_days"])
        if volatility_window > 1:
            returns = df["close"].pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.tail(volatility_window).std() * np.sqrt(365)  # Annualized
                self.volatility_data[symbol] = volatility
        
        # Calculate momentum
        momentum_window = min(len(df), self.parameters["momentum_lookback_days"])
        if momentum_window > 1:
            start_price = df["close"].iloc[-momentum_window]
            end_price = df["close"].iloc[-1]
            if start_price > 0:
                momentum = (end_price / start_price) - 1  # Simple return over period
                self.momentum_data[symbol] = momentum
        
        # TODO: Update correlation matrix when we have data for all assets
    
    def _update_current_allocations(self) -> None:
        """Update current portfolio allocations based on latest prices."""
        total_value = 0.0
        
        # Calculate current value of each position
        for symbol, position in self.asset_positions.items():
            if symbol in self.asset_data and self.asset_data[symbol]["price"] > 0:
                price = self.asset_data[symbol]["price"]
                quantity = position.get("quantity", 0.0)
                value = price * quantity
                self.asset_positions[symbol]["value_usd"] = value
                total_value += value
        
        # Add value of assets not in portfolio but in our tracking list
        for symbol in self.asset_data:
            if symbol not in self.asset_positions:
                self.asset_positions[symbol] = {"quantity": 0.0, "value_usd": 0.0}
        
        # Update portfolio value
        if total_value > 0:
            self.portfolio_value = total_value
        
        # Calculate current allocations
        if self.portfolio_value > 0:
            for symbol, position in self.asset_positions.items():
                value = position.get("value_usd", 0.0)
                allocation = value / self.portfolio_value
                self.current_allocations[symbol] = allocation
        
        # Log current portfolio state
        if self.portfolio_value > 0:
            allocations_str = ", ".join([f"{s}: {a:.1%}" for s, a in self.current_allocations.items() if a > 0.01])
            logger.info(f"Current portfolio: ${self.portfolio_value:.2f} - {allocations_str}")
    
    def _check_rebalance_needed(self) -> bool:
        """
        Check if portfolio rebalancing is needed based on drift and time elapsed.
        
        Returns:
            True if rebalancing is needed, False otherwise
        """
        # Skip if we don't have all the necessary data
        if not self.target_allocations or not self.current_allocations:
            return False
        
        # Calculate time since last rebalance
        days_since_rebalance = 0
        if self.last_rebalance_date:
            days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
        else:
            # First run, set last rebalance date but don't trigger yet
            self.last_rebalance_date = datetime.now()
            return False
        
        # Check minimum rebalance interval
        if days_since_rebalance < self.parameters["min_rebalance_interval_days"]:
            return False
        
        # Check maximum rebalance interval
        force_rebalance = days_since_rebalance >= self.parameters["rebalance_frequency_days"]
        
        # Check allocation drift
        max_drift = 0.0
        for symbol, target in self.target_allocations.items():
            current = self.current_allocations.get(symbol, 0.0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)
        
        drift_rebalance = max_drift >= self.parameters["rebalance_threshold_pct"]
        
        # Determine if rebalancing is needed
        rebalance_needed = force_rebalance or drift_rebalance
        
        if rebalance_needed:
            if force_rebalance:
                logger.info(f"Triggering rebalance due to time elapsed ({days_since_rebalance} days)")
            else:
                logger.info(f"Triggering rebalance due to allocation drift ({max_drift:.2%})")
            
            # Recalculate target allocations before rebalancing
            self._recalculate_target_allocations()
            
            return True
        
        return False
