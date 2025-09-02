#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Index Investing Strategy

This strategy implements an index investing approach for cryptocurrencies,
constructing and maintaining portfolios (baskets) of cryptocurrencies 
based on different methodologies like market cap weighting, equal weighting,
or sector-specific allocations.

The strategy automatically rebalances positions according to the chosen
methodology and periodicity while managing risk through diversification.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import math
import json
from collections import defaultdict

# Import base strategy
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.factory.strategy_factory import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoIndexInvestingStrategy",
    category="crypto",
    description="A strategy that constructs and manages baskets of cryptocurrencies based on different allocation methodologies",
    parameters={
        # Core strategy parameters
        "index_type": {
            "type": "str",
            "default": "market_cap_weighted",
            "enum": ["market_cap_weighted", "equal_weighted", "sector_weighted", "volatility_weighted", "custom_weighted"],
            "description": "Type of index weighting methodology to use"
        },
        "rebalance_frequency": {
            "type": "str",
            "default": "weekly",
            "enum": ["daily", "weekly", "biweekly", "monthly", "quarterly"],
            "description": "How often to rebalance the crypto portfolio"
        },
        "num_assets": {
            "type": "int",
            "default": 10,
            "description": "Number of assets to include in the index"
        },
        "min_market_cap_usd": {
            "type": "float",
            "default": 1000000000,  # $1B
            "description": "Minimum market cap for assets to be considered (in USD)"
        },
        "min_daily_volume_usd": {
            "type": "float",
            "default": 50000000,  # $50M
            "description": "Minimum daily trading volume for assets to be considered (in USD)"
        },
        "require_historical_data": {
            "type": "bool",
            "default": True,
            "description": "Whether to require minimum historical data for assets"
        },
        "min_price_history_days": {
            "type": "int",
            "default": 90,
            "description": "Minimum days of price history required"
        },
        "primary_timeframe": {
            "type": "str",
            "default": "1d",
            "description": "Primary timeframe for analysis"
        },
        "rebalance_threshold_pct": {
            "type": "float",
            "default": 0.05,  # 5%
            "description": "Threshold percentage difference to trigger rebalancing"
        },
        
        # Market cap weighted parameters
        "mcap_min_weightage": {
            "type": "float",
            "default": 0.02,  # 2%
            "description": "Minimum weight for an asset in market cap weighted index"
        },
        "mcap_max_weightage": {
            "type": "float",
            "default": 0.25,  # 25%
            "description": "Maximum weight for an asset in market cap weighted index"
        },
        "cap_dominant_assets": {
            "type": "bool",
            "default": True,
            "description": "Whether to cap weights of dominant assets (like BTC and ETH)"
        },
        
        # Equal weighted parameters
        "equal_weight_adjustment_factor": {
            "type": "float",
            "default": 1.0,
            "description": "Adjustment factor for equal weighting (1.0 means perfect equal weighting)"
        },
        
        # Sector weighted parameters
        "sector_allocations": {
            "type": "dict",
            "default": {
                "smart_contract_platforms": 0.30,
                "defi": 0.20,
                "layer2": 0.15,
                "exchange_tokens": 0.10,
                "privacy": 0.05,
                "storage_and_computing": 0.05,
                "gaming_and_metaverse": 0.10,
                "other": 0.05
            },
            "description": "Allocation percentages by sector (must sum to 1.0)"
        },
        
        # Volatility weighted parameters
        "volatility_lookback_days": {
            "type": "int",
            "default": 30,
            "description": "Lookback period for volatility calculation"
        },
        "inverse_volatility_weighting": {
            "type": "bool",
            "default": True,
            "description": "Whether to weight by inverse volatility (lower vol = higher weight)"
        },
        
        # Custom index parameters
        "custom_assets": {
            "type": "list",
            "default": ["BTC", "ETH", "BNB", "SOL", "ADA", "DOT", "AVAX", "LINK", "MATIC", "UNI"],
            "description": "Custom list of assets to include in the index"
        },
        "custom_weights": {
            "type": "list",
            "default": [0.25, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "description": "Custom weights for assets (must sum to 1.0 and match length of custom_assets)"
        },
        
        # Risk management parameters
        "max_asset_allocation": {
            "type": "float",
            "default": 0.30,  # 30%
            "description": "Maximum allocation to any single asset"
        },
        "volatility_cap": {
            "type": "float",
            "default": 0.03,  # 3% daily vol
            "description": "Target portfolio volatility cap"
        },
        "enable_volatility_targeting": {
            "type": "bool",
            "default": False,
            "description": "Whether to target a specific portfolio volatility"
        },
        "enable_stop_loss": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable portfolio-level stop loss"
        },
        "portfolio_stop_loss_pct": {
            "type": "float",
            "default": 0.15,  # 15%
            "description": "Portfolio-level stop loss percentage"
        },
        "enable_trailing_stop": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable portfolio-level trailing stop"
        },
        "trailing_stop_pct": {
            "type": "float",
            "default": 0.10,  # 10%
            "description": "Trailing stop percentage from highest portfolio value"
        },
        
        # Execution parameters
        "execution_model": {
            "type": "str",
            "default": "gradual",
            "enum": ["immediate", "gradual", "twap"],
            "description": "How to execute portfolio changes"
        },
        "execution_duration_hours": {
            "type": "int",
            "default": 24,
            "description": "Duration over which to spread execution (for gradual/twap models)"
        },
        "slippage_model": {
            "type": "str",
            "default": "market_impact",
            "enum": ["fixed", "market_impact", "volume_based"],
            "description": "Model for estimating slippage"
        },
        "slippage_factor": {
            "type": "float",
            "default": 0.001,  # 0.1%
            "description": "Slippage factor for the chosen model"
        }
    }
)
class CryptoIndexInvestingStrategy(CryptoBaseStrategy):
    """
    A strategy that implements index investing for cryptocurrencies, constructing
    and maintaining baskets of assets according to different methodologies.
    """
    
    def __init__(self, session: CryptoSession, parameters: Dict[str, Any] = None):
        """
        Initialize the Crypto Index Investing Strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize internal state
        self.index_composition = {}  # Current index composition {symbol: weight}
        self.target_allocations = {}  # Target allocations {symbol: weight}
        self.current_allocations = {}  # Current actual allocations {symbol: weight}
        self.sector_mapping = {}  # Maps assets to sectors {symbol: sector}
        self.last_rebalance_time = None
        self.portfolio_value_history = {}  # {timestamp: value}
        self.portfolio_high_watermark = 0
        self.asset_market_data = {}  # {symbol: {market_cap, volume, etc.}}
        self.sector_performance = {}  # {sector: performance}
        
        # Load or initialize sector mappings
        self._initialize_sector_mappings()
        
        # Register for events
        self._register_events()
        
        logger.info(f"Initialized Crypto Index Investing Strategy with {self.parameters['index_type']} methodology")
    
    def _register_events(self) -> None:
        """
        Register for relevant events for the strategy.
        """
        # Register for market data events
        self.event_bus.subscribe("MARKET_DATA_UPDATE", self._on_market_data_update)
        self.event_bus.subscribe("MARKET_METADATA_UPDATE", self._on_market_metadata_update)
        
        # Register for timeframe events
        for timeframe in ["1d", "1h"]:
            self.event_bus.subscribe(f"TIMEFRAME_{timeframe}", self._on_timeframe_event)
        
        # Register for trade execution events
        self.event_bus.subscribe("TRADE_EXECUTED", self._on_trade_executed)
        self.event_bus.subscribe("POSITION_UPDATE", self._on_position_update)
        
        # Register for account/portfolio events
        self.event_bus.subscribe("ACCOUNT_UPDATE", self._on_account_update)
        self.event_bus.subscribe("PORTFOLIO_VALUE_UPDATE", self._on_portfolio_value_update)
    
    def _initialize_sector_mappings(self) -> None:
        """
        Initialize sector mappings for cryptocurrencies.
        This maps each supported cryptocurrency to its primary sector/category.
        """
        # Default sector mappings for common cryptocurrencies
        self.sector_mapping = {
            # Smart Contract Platforms
            "ETH": "smart_contract_platforms",
            "SOL": "smart_contract_platforms",
            "ADA": "smart_contract_platforms",
            "DOT": "smart_contract_platforms",
            "AVAX": "smart_contract_platforms",
            "NEAR": "smart_contract_platforms",
            "ATOM": "smart_contract_platforms",
            "ALGO": "smart_contract_platforms",
            "FTM": "smart_contract_platforms",
            "ONE": "smart_contract_platforms",
            
            # Layer 2 Solutions
            "MATIC": "layer2",
            "LRC": "layer2",
            "OMG": "layer2",
            "ARBITRUM": "layer2",
            "OP": "layer2",
            "IMX": "layer2",
            
            # DeFi
            "UNI": "defi",
            "AAVE": "defi",
            "MKR": "defi",
            "COMP": "defi",
            "SNX": "defi",
            "CAKE": "defi",
            "SUSHI": "defi",
            "YFI": "defi",
            "CRV": "defi",
            "BAL": "defi",
            "1INCH": "defi",
            
            # Exchange Tokens
            "BNB": "exchange_tokens",
            "FTT": "exchange_tokens",
            "CRO": "exchange_tokens",
            "KCS": "exchange_tokens",
            "OKB": "exchange_tokens",
            "HT": "exchange_tokens",
            "GT": "exchange_tokens",
            
            # Privacy
            "XMR": "privacy",
            "ZEC": "privacy",
            "DASH": "privacy",
            "SCRT": "privacy",
            
            # Storage and Computing
            "FIL": "storage_and_computing",
            "STORJ": "storage_and_computing",
            "AR": "storage_and_computing",
            "SC": "storage_and_computing",
            "RLC": "storage_and_computing",
            "GLM": "storage_and_computing",
            
            # Gaming and Metaverse
            "AXS": "gaming_and_metaverse",
            "SAND": "gaming_and_metaverse",
            "MANA": "gaming_and_metaverse",
            "ENJ": "gaming_and_metaverse",
            "ILV": "gaming_and_metaverse",
            "GALA": "gaming_and_metaverse",
            "THETA": "gaming_and_metaverse",
            
            # Payments and Store of Value
            "BTC": "payments_and_store_of_value",
            "LTC": "payments_and_store_of_value",
            "XRP": "payments_and_store_of_value",
            "XLM": "payments_and_store_of_value",
            "DOGE": "payments_and_store_of_value",
            "BCH": "payments_and_store_of_value",
            
            # Other
            "LINK": "oracle",
            "GRT": "data_and_analytics",
            "OCEAN": "data_and_analytics",
            "BAT": "advertising",
            "HNT": "iot",
            "FET": "ai"
        }
        
        # Try to load custom mappings from a file if it exists
        try:
            with open("data/crypto_sector_mappings.json", "r") as f:
                custom_mappings = json.load(f)
                self.sector_mapping.update(custom_mappings)
                logger.info(f"Loaded custom sector mappings for {len(custom_mappings)} assets")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load custom sector mappings: {e}")
    
    def _on_market_data_update(self, event: Event) -> None:
        """
        Handle market data updates.
        
        Args:
            event: Market data update event
        """
        # Extract relevant data
        symbol = event.data.get("symbol")
        if not symbol:
            return
            
        # Store market data for the asset
        if symbol not in self.asset_market_data:
            self.asset_market_data[symbol] = {}
            
        self.asset_market_data[symbol]["price"] = event.data.get("close")
        self.asset_market_data[symbol]["last_update"] = datetime.now()
        
        # Check if we need to update our current allocations based on price changes
        self._update_current_allocations()
        
        # Check if we need to rebalance based on allocation drift
        if self._should_rebalance_on_drift():
            logger.info(f"Initiating rebalance due to allocation drift exceeding threshold")
            self._rebalance_portfolio()
    
    def _on_market_metadata_update(self, event: Event) -> None:
        """
        Handle market metadata updates (market cap, volume, etc.).
        
        Args:
            event: Market metadata update event
        """
        # Extract relevant data
        symbol = event.data.get("symbol")
        if not symbol:
            return
            
        # Store metadata for the asset
        if symbol not in self.asset_market_data:
            self.asset_market_data[symbol] = {}
            
        metadata = event.data.get("metadata", {})
        
        # Update market cap and volume data
        if "market_cap" in metadata:
            self.asset_market_data[symbol]["market_cap"] = metadata["market_cap"]
            
        if "volume_24h" in metadata:
            self.asset_market_data[symbol]["volume_24h"] = metadata["volume_24h"]
            
        # If we're using market cap weighting, we may need to recalculate
        if self.parameters["index_type"] == "market_cap_weighted" and self._should_recalculate_weights():
            self._calculate_target_allocations()
    
    def _on_timeframe_event(self, event: Event) -> None:
        """
        Handle timeframe events for periodic operations.
        
        Args:
            event: Timeframe event
        """
        timeframe = event.data.get("timeframe")
        if not timeframe:
            return
            
        # Check if we need to perform a periodic rebalance
        if timeframe == "1d" and self._should_rebalance_periodic():
            logger.info(f"Performing scheduled periodic rebalance")
            self._rebalance_portfolio()
            
        # Recalculate daily performance statistics
        if timeframe == "1d":
            self._calculate_performance_metrics()
            
        # Update volatility metrics for volatility-weighted allocations
        if timeframe == "1d" and self.parameters["index_type"] == "volatility_weighted":
            self._update_volatility_metrics()
            self._calculate_target_allocations()
    
    def _on_trade_executed(self, event: Event) -> None:
        """
        Handle trade execution events.
        
        Args:
            event: Trade execution event
        """
        # Update current allocations after trades
        symbol = event.data.get("symbol")
        if symbol and symbol in self.target_allocations:
            self._update_current_allocations()
    
    def _on_position_update(self, event: Event) -> None:
        """
        Handle position update events.
        
        Args:
            event: Position update event
        """
        # Update current allocations when positions change
        self._update_current_allocations()
    
    def _on_account_update(self, event: Event) -> None:
        """
        Handle account update events.
        
        Args:
            event: Account update event
        """
        # Recalculate allocations based on updated account value
        self._update_current_allocations()
    
    def _on_portfolio_value_update(self, event: Event) -> None:
        """
        Handle portfolio value update events.
        
        Args:
            event: Portfolio value update event
        """
        # Extract portfolio value
        timestamp = event.data.get("timestamp", datetime.now())
        portfolio_value = event.data.get("value", 0)
        
        if portfolio_value <= 0:
            return
            
        # Record portfolio value history
        self.portfolio_value_history[timestamp] = portfolio_value
        
        # Update high watermark if needed
        if portfolio_value > self.portfolio_high_watermark:
            self.portfolio_high_watermark = portfolio_value
            
        # Check trailing stop if enabled
        if self.parameters["enable_trailing_stop"]:
            self._check_trailing_stop(portfolio_value)
            
        # Clean up old history entries
        self._clean_portfolio_history()
    
    def _update_current_allocations(self) -> None:
        """
        Update the current allocation of each asset in the portfolio.
        """
        total_portfolio_value = self._get_total_portfolio_value()
        if total_portfolio_value <= 0:
            logger.warning("Cannot update allocations: total portfolio value is zero or negative")
            return
            
        # Reset current allocations
        self.current_allocations = {}
        
        # Calculate current allocation for each asset in the index
        for symbol in self.index_composition:
            position_value = self._get_position_value(symbol)
            
            # Calculate allocation as percentage of total portfolio
            self.current_allocations[symbol] = position_value / total_portfolio_value
            
        # Calculate allocation differences
        allocation_diffs = {}
        for symbol in self.target_allocations:
            target_alloc = self.target_allocations.get(symbol, 0)
            current_alloc = self.current_allocations.get(symbol, 0)
            allocation_diffs[symbol] = current_alloc - target_alloc
            
        # Log significant deviations
        threshold = self.parameters["rebalance_threshold_pct"]
        for symbol, diff in allocation_diffs.items():
            if abs(diff) > threshold:
                logger.info(f"Allocation for {symbol} deviates by {diff:.2%} from target")
    
    def _get_total_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value
        """
        # Get account equity
        account_value = self.session.get_account_value()
        
        # Add up all positions related to the index
        portfolio_value = 0
        for symbol in self.index_composition:
            portfolio_value += self._get_position_value(symbol)
            
        # If we have no index positions yet, return the account value
        if portfolio_value <= 0 and account_value > 0:
            return account_value
            
        return portfolio_value
    
    def _get_position_value(self, symbol: str) -> float:
        """
        Get the current value of a position.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Position value in account currency
        """
        # Get position size
        position = self.session.get_position(symbol)
        if not position:
            return 0
            
        # Get current price
        price = 0
        if symbol in self.asset_market_data and "price" in self.asset_market_data[symbol]:
            price = self.asset_market_data[symbol]["price"]
        else:
            price = self.session.get_last_price(symbol)
            
        if price <= 0:
            return 0
            
        # Calculate position value
        return position.quantity * price
    
    def _should_rebalance_periodic(self) -> bool:
        """
        Check if we should perform a periodic rebalance based on schedule.
        
        Returns:
            True if a periodic rebalance is due, False otherwise
        """
        now = datetime.now()
        if not self.last_rebalance_time:
            return True
            
        # Calculate time since last rebalance
        time_diff = now - self.last_rebalance_time
        
        # Check based on rebalance frequency
        frequency = self.parameters["rebalance_frequency"]
        
        if frequency == "daily" and time_diff.days >= 1:
            return True
        elif frequency == "weekly" and time_diff.days >= 7:
            return True
        elif frequency == "biweekly" and time_diff.days >= 14:
            return True
        elif frequency == "monthly" and time_diff.days >= 30:
            return True
        elif frequency == "quarterly" and time_diff.days >= 90:
            return True
            
        return False
    
    def _should_rebalance_on_drift(self) -> bool:
        """
        Check if we should rebalance due to allocation drift exceeding threshold.
        
        Returns:
            True if rebalance is needed, False otherwise
        """
        # If we just rebalanced, don't check again too soon
        now = datetime.now()
        if self.last_rebalance_time and (now - self.last_rebalance_time).total_seconds() < 3600:  # 1 hour
            return False
            
        # Check if any allocation exceeds the drift threshold
        threshold = self.parameters["rebalance_threshold_pct"]
        
        for symbol in self.target_allocations:
            target_alloc = self.target_allocations.get(symbol, 0)
            current_alloc = self.current_allocations.get(symbol, 0)
            
            if abs(current_alloc - target_alloc) > threshold:
                logger.info(f"Rebalance needed: {symbol} allocation drift {abs(current_alloc - target_alloc):.2%} > threshold {threshold:.2%}")
                return True
                
        return False
    
    def _should_recalculate_weights(self) -> bool:
        """
        Check if we should recalculate target weights.
        
        Returns:
            True if recalculation is needed, False otherwise
        """
        # Initial calculation
        if not self.target_allocations:
            return True
            
        # Check if we're due for a rebalance which would also recalculate
        if self._should_rebalance_periodic():
            return True
            
        # Always recalculate after significant market events (not implemented here)
        return False
    
    def _clean_portfolio_history(self, max_days: int = 90) -> None:
        """
        Clean up old portfolio history entries to manage memory usage.
        
        Args:
            max_days: Maximum number of days to keep history for
        """
        cutoff_time = datetime.now() - timedelta(days=max_days)
        to_remove = []
        
        for timestamp in self.portfolio_value_history:
            if timestamp < cutoff_time:
                to_remove.append(timestamp)
                
        for timestamp in to_remove:
            del self.portfolio_value_history[timestamp]
    
    def _check_trailing_stop(self, current_value: float) -> None:
        """
        Check if the trailing stop has been triggered and take action if needed.
        
        Args:
            current_value: Current portfolio value
        """
        if self.portfolio_high_watermark <= 0 or current_value <= 0:
            return
            
        # Calculate drawdown percentage
        drawdown_pct = 1.0 - (current_value / self.portfolio_high_watermark)
        trailing_stop_pct = self.parameters["trailing_stop_pct"] 
        
        # Check if drawdown exceeds the trailing stop percentage
        if drawdown_pct > trailing_stop_pct:
            logger.warning(f"Trailing stop triggered: drawdown {drawdown_pct:.2%} exceeds stop {trailing_stop_pct:.2%}")
            self._handle_stop_loss_triggered("trailing_stop", drawdown_pct)
    
    def _handle_stop_loss_triggered(self, stop_type: str, drawdown: float) -> None:
        """
        Handle a triggered stop loss (either regular or trailing).
        
        Args:
            stop_type: Type of stop loss triggered ("portfolio_stop" or "trailing_stop")
            drawdown: Current drawdown percentage
        """
        # Publish event for the stop loss
        event_data = {
            "stop_type": stop_type,
            "drawdown": drawdown,
            "portfolio_value": self._get_total_portfolio_value(),
            "high_watermark": self.portfolio_high_watermark,
            "timestamp": datetime.now(),
            "strategy": self.__class__.__name__
        }
        
        self.event_bus.publish("STOP_LOSS_TRIGGERED", event_data)
        
        # Action to take depends on the parameter settings
        if stop_type == "portfolio_stop" and self.parameters["enable_stop_loss"]:
            logger.info(f"Executing portfolio-wide stop loss actions")
            self._reduce_risk_exposure(drawdown)
        elif stop_type == "trailing_stop" and self.parameters["enable_trailing_stop"]:
            logger.info(f"Executing trailing stop actions")
            self._reduce_risk_exposure(drawdown)
    
    def _reduce_risk_exposure(self, drawdown: float) -> None:
        """
        Reduce risk exposure after a stop loss is triggered.
        The higher the drawdown, the more aggressive the risk reduction.
        
        Args:
            drawdown: Current drawdown percentage
        """
        # Determine how much to reduce exposure based on drawdown severity
        if drawdown < 0.1:  # Less than 10%
            # Small reduction
            reduction_factor = 0.2  # Reduce by 20%
        elif drawdown < 0.15:  # 10-15%
            # Medium reduction
            reduction_factor = 0.5  # Reduce by 50%
        else:  # More than 15%
            # Large reduction or full exit
            reduction_factor = 0.8  # Reduce by 80%
            
        logger.info(f"Reducing portfolio risk exposure by {reduction_factor:.0%}")
        
        # Close positions proportionally
        for symbol in list(self.index_composition.keys()):
            position = self.session.get_position(symbol)
            if not position or position.quantity <= 0:
                continue
                
            # Calculate quantity to close
            quantity_to_close = position.quantity * reduction_factor
            
            # Create reduce-only order
            self._create_reduce_position_order(symbol, quantity_to_close)
            
        # Record this event
        self.last_risk_reduction_time = datetime.now()
    
    def _create_reduce_position_order(self, symbol: str, quantity: float) -> None:
        """
        Create an order to reduce a position.
        
        Args:
            symbol: Symbol to reduce position for
            quantity: Quantity to reduce by
        """
        if quantity <= 0:
            return
            
        # Determine position direction
        position = self.session.get_position(symbol)
        if not position:
            return
            
        direction = "long" if position.quantity > 0 else "short"
        order_direction = "sell" if direction == "long" else "buy"
        
        # Create market order to reduce position
        order_params = {
            "symbol": symbol,
            "order_type": "market",
            "direction": order_direction,
            "quantity": abs(quantity),
            "reduce_only": True,
            "time_in_force": "GTC",
            "source": self.__class__.__name__
        }
        
        try:
            self.session.create_order(**order_params)
            logger.info(f"Created {order_direction} order for {quantity} {symbol} to reduce position")
        except Exception as e:
            logger.error(f"Failed to create reduce position order for {symbol}: {e}")
    
    def _calculate_target_allocations(self) -> None:
        """
        Calculate target allocations based on the chosen index methodology.
        """
        # Clear existing allocations
        self.target_allocations = {}
        
        # Select the calculation method based on index type
        index_type = self.parameters["index_type"]
        
        if index_type == "market_cap_weighted":
            self._calculate_market_cap_allocations()
        elif index_type == "equal_weighted":
            self._calculate_equal_weight_allocations()
        elif index_type == "sector_weighted":
            self._calculate_sector_weight_allocations()
        elif index_type == "volatility_weighted":
            self._calculate_volatility_weighted_allocations()
        elif index_type == "custom_weighted":
            self._calculate_custom_weight_allocations()
        else:
            logger.error(f"Unknown index type: {index_type}")
            # Default to equal weight if unknown type
            self._calculate_equal_weight_allocations()
        
        # Update the index composition
        self.index_composition = self.target_allocations.copy()
        
        # Log the new allocations
        allocation_str = ", ".join([f"{s}: {w:.2%}" for s, w in self.target_allocations.items()])
        logger.info(f"Calculated new target allocations: {allocation_str}")
        
        # Publish the index composition event
        self._publish_index_composition_event()
    
    def _calculate_market_cap_allocations(self) -> None:
        """
        Calculate target allocations based on market capitalization weighting.
        """
        # Get parameters
        num_assets = self.parameters["num_assets"]
        min_market_cap = self.parameters["min_market_cap_usd"]
        min_weightage = self.parameters["mcap_min_weightage"]
        max_weightage = self.parameters["mcap_max_weightage"]
        cap_dominant = self.parameters["cap_dominant_assets"]
        
        # Filter assets that meet minimum market cap
        eligible_assets = []
        
        for symbol, data in self.asset_market_data.items():
            if "market_cap" in data and data["market_cap"] >= min_market_cap:
                eligible_assets.append((symbol, data["market_cap"]))
        
        # Sort by market cap (descending)
        eligible_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N assets
        selected_assets = eligible_assets[:num_assets]
        
        if not selected_assets:
            logger.warning("No eligible assets found for market cap weighting")
            return
        
        # Calculate total market cap of selected assets
        total_market_cap = sum(asset[1] for asset in selected_assets)
        
        if total_market_cap <= 0:
            logger.warning("Total market cap is zero or negative")
            return
        
        # Calculate initial weights based on market cap
        preliminary_weights = {}
        for symbol, market_cap in selected_assets:
            weight = market_cap / total_market_cap
            preliminary_weights[symbol] = weight
        
        # Apply capping to dominant assets if requested
        if cap_dominant and len(preliminary_weights) >= 2:
            # Identify assets that exceed the maximum weight
            excess_assets = [s for s, w in preliminary_weights.items() if w > max_weightage]
            excess_weight = sum(preliminary_weights[s] - max_weightage for s in excess_assets)
            
            if excess_weight > 0 and len(excess_assets) > 0:
                # Cap the weights of dominant assets
                for symbol in excess_assets:
                    preliminary_weights[symbol] = max_weightage
                
                # Redistribute excess weight to other assets proportionally
                non_excess_assets = [s for s in preliminary_weights if s not in excess_assets]
                
                if non_excess_assets:
                    total_non_excess_weight = sum(preliminary_weights[s] for s in non_excess_assets)
                    
                    if total_non_excess_weight > 0:
                        for symbol in non_excess_assets:
                            proportion = preliminary_weights[symbol] / total_non_excess_weight
                            preliminary_weights[symbol] += excess_weight * proportion
        
        # Ensure no asset has less than the minimum weight
        for symbol in list(preliminary_weights.keys()):
            if preliminary_weights[symbol] < min_weightage:
                # Remove assets below minimum weight
                del preliminary_weights[symbol]
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(preliminary_weights.values())
        if total_weight > 0:
            self.target_allocations = {s: w / total_weight for s, w in preliminary_weights.items()}
    
    def _calculate_equal_weight_allocations(self) -> None:
        """
        Calculate target allocations based on equal weighting.
        """
        # Get parameters
        num_assets = self.parameters["num_assets"]
        min_market_cap = self.parameters["min_market_cap_usd"]
        adjustment_factor = self.parameters["equal_weight_adjustment_factor"]
        
        # Filter assets that meet minimum market cap and trading volume
        eligible_assets = []
        
        for symbol, data in self.asset_market_data.items():
            if "market_cap" in data and data["market_cap"] >= min_market_cap:
                if "volume_24h" in data and data["volume_24h"] >= self.parameters["min_daily_volume_usd"]:
                    eligible_assets.append(symbol)
        
        # If we have custom assets and no eligible ones, use the custom list
        if not eligible_assets and self.parameters["custom_assets"]:
            eligible_assets = self.parameters["custom_assets"][:num_assets]
        
        # Select top N assets
        selected_assets = eligible_assets[:num_assets]
        
        if not selected_assets:
            logger.warning("No eligible assets found for equal weighting")
            return
        
        # Calculate equal weights
        equal_weight = 1.0 / len(selected_assets)
        
        # Apply adjustment factor if requested (can be used to slightly overweight/underweight certain assets)
        if adjustment_factor != 1.0 and len(selected_assets) > 1:
            # For now, just use equal weights (adjustment factor is for custom implementations)
            pass
        
        # Assign equal weights to all selected assets
        self.target_allocations = {symbol: equal_weight for symbol in selected_assets}
    
    def _calculate_sector_weight_allocations(self) -> None:
        """
        Calculate target allocations based on sector weighting.
        """
        # Get parameters
        sector_allocations = self.parameters["sector_allocations"]
        min_market_cap = self.parameters["min_market_cap_usd"]
        
        # Group assets by sector
        assets_by_sector = defaultdict(list)
        
        for symbol, data in self.asset_market_data.items():
            if "market_cap" in data and data["market_cap"] >= min_market_cap:
                # Get sector for this asset
                sector = self.sector_mapping.get(symbol, "other")
                assets_by_sector[sector].append((symbol, data["market_cap"]))
        
        # Sort assets within each sector by market cap
        for sector in assets_by_sector:
            assets_by_sector[sector].sort(key=lambda x: x[1], reverse=True)
        
        # Initialize weights
        preliminary_weights = {}
        
        # Allocate weights based on sector allocations
        for sector, allocation in sector_allocations.items():
            if sector not in assets_by_sector or not assets_by_sector[sector]:
                continue
                
            sector_assets = assets_by_sector[sector]
            total_sector_mcap = sum(asset[1] for asset in sector_assets)
            
            if total_sector_mcap <= 0:
                continue
                
            # Distribute the sector allocation among assets within the sector based on market cap
            for symbol, market_cap in sector_assets:
                weight = allocation * (market_cap / total_sector_mcap)
                preliminary_weights[symbol] = weight
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(preliminary_weights.values())
        if total_weight > 0:
            self.target_allocations = {s: w / total_weight for s, w in preliminary_weights.items()}
    
    def _update_volatility_metrics(self) -> None:
        """
        Update volatility metrics for all tracked assets.
        """
        lookback = self.parameters["volatility_lookback_days"]
        
        # For each asset, calculate historical volatility
        for symbol in list(self.asset_market_data.keys()):
            # Get historical data
            data = self.session.get_historical_data(symbol, "1d", lookback + 10)  # Extra periods for calculation
            
            if data is None or data.empty or len(data) < lookback:
                logger.warning(f"Insufficient data for volatility calculation for {symbol}")
                continue
                
            # Calculate daily returns
            returns = data["close"].pct_change().dropna()
            
            if len(returns) < lookback:
                continue
                
            # Calculate annualized volatility (standard deviation of returns * sqrt(252))
            volatility = returns.iloc[-lookback:].std() * np.sqrt(252)
            
            # Store volatility in asset data
            if symbol not in self.asset_market_data:
                self.asset_market_data[symbol] = {}
                
            self.asset_market_data[symbol]["volatility"] = volatility
    
    def _calculate_volatility_weighted_allocations(self) -> None:
        """
        Calculate target allocations based on volatility weighting.
        Inverse volatility weighting gives higher weights to assets with lower volatility.
        """
        # Get parameters
        num_assets = self.parameters["num_assets"]
        min_market_cap = self.parameters["min_market_cap_usd"]
        inverse_weighting = self.parameters["inverse_volatility_weighting"]
        
        # Ensure volatility metrics are up to date
        self._update_volatility_metrics()
        
        # Filter assets that meet minimum criteria and have volatility data
        eligible_assets = []
        
        for symbol, data in self.asset_market_data.items():
            if "market_cap" in data and data["market_cap"] >= min_market_cap:
                if "volatility" in data and data["volatility"] > 0:
                    eligible_assets.append((symbol, data["volatility"]))
        
        # Sort by volatility (ascending or descending based on inverse_weighting)
        eligible_assets.sort(key=lambda x: x[1], reverse=not inverse_weighting)
        
        # Select top N assets
        selected_assets = eligible_assets[:num_assets]
        
        if not selected_assets:
            logger.warning("No eligible assets found for volatility weighting")
            return
        
        # Calculate weights based on volatility
        preliminary_weights = {}
        
        if inverse_weighting:
            # For inverse volatility weighting: weight is proportional to 1/volatility
            inverse_volatilities = [(s, 1.0 / v) for s, v in selected_assets]
            total_inverse = sum(iv[1] for iv in inverse_volatilities)
            
            if total_inverse > 0:
                for symbol, inverse_vol in inverse_volatilities:
                    preliminary_weights[symbol] = inverse_vol / total_inverse
        else:
            # For volatility targeting: weight is proportional to target_vol / asset_vol
            target_vol = self.parameters["volatility_cap"]
            vol_ratios = [(s, target_vol / v) for s, v in selected_assets]
            total_ratio = sum(vr[1] for vr in vol_ratios)
            
            if total_ratio > 0:
                for symbol, vol_ratio in vol_ratios:
                    preliminary_weights[symbol] = vol_ratio / total_ratio
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(preliminary_weights.values())
        if total_weight > 0:
            self.target_allocations = {s: w / total_weight for s, w in preliminary_weights.items()}
    
    def _calculate_custom_weight_allocations(self) -> None:
        """
        Calculate target allocations based on custom weightings.
        """
        # Get parameters
        custom_assets = self.parameters["custom_assets"]
        custom_weights = self.parameters["custom_weights"]
        
        # Ensure we have both assets and weights
        if not custom_assets or not custom_weights:
            logger.error("Missing custom assets or weights for custom index")
            # Fall back to equal weighting
            self._calculate_equal_weight_allocations()
            return
            
        # Match assets with weights
        min_length = min(len(custom_assets), len(custom_weights))
        preliminary_weights = {custom_assets[i]: custom_weights[i] for i in range(min_length)}
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(preliminary_weights.values())
        if total_weight > 0:
            self.target_allocations = {s: w / total_weight for s, w in preliminary_weights.items()}
        else:
            logger.warning("Total weight for custom allocations is zero or negative")
    
    def _publish_index_composition_event(self) -> None:
        """
        Publish an event with the current index composition.
        """
        event_data = {
            "index_type": self.parameters["index_type"],
            "composition": self.index_composition,
            "timestamp": datetime.now(),
            "strategy": self.__class__.__name__
        }
        
        self.event_bus.publish("INDEX_COMPOSITION_UPDATE", event_data)
    
    def _rebalance_portfolio(self) -> None:
        """
        Rebalance the portfolio to match target allocations.
        """
        logger.info("Starting portfolio rebalance process")
        
        # Make sure we have target allocations
        if not self.target_allocations:
            self._calculate_target_allocations()
            
        if not self.target_allocations:
            logger.error("Cannot rebalance: failed to calculate target allocations")
            return
            
        # Get current portfolio value
        portfolio_value = self._get_total_portfolio_value()
        if portfolio_value <= 0:
            logger.error("Cannot rebalance: portfolio value is zero or negative")
            return
            
        # Update current allocations
        self._update_current_allocations()
        
        # Calculate required adjustments for each asset
        adjustments = {}
        for symbol in set(list(self.target_allocations.keys()) + list(self.current_allocations.keys())):
            target_pct = self.target_allocations.get(symbol, 0.0)
            current_pct = self.current_allocations.get(symbol, 0.0)
            
            # Calculate value adjustment
            value_adjustment = (target_pct - current_pct) * portfolio_value
            
            if abs(value_adjustment) > 10:  # Minimum adjustment value
                current_price = 0
                if symbol in self.asset_market_data and "price" in self.asset_market_data[symbol]:
                    current_price = self.asset_market_data[symbol]["price"]
                else:
                    current_price = self.session.get_last_price(symbol)
                    
                if current_price <= 0:
                    logger.warning(f"Cannot calculate quantity adjustment for {symbol}: price is zero or negative")
                    continue
                    
                # Calculate quantity adjustment
                quantity_adjustment = value_adjustment / current_price
                
                # Only include significant adjustments
                if abs(quantity_adjustment) > 0.0001:  # Minimum quantity threshold
                    adjustments[symbol] = quantity_adjustment
                
        # Execute rebalance trades using the specified execution model
        self._execute_rebalance_trades(adjustments, portfolio_value)
        
        # Update rebalance time
        self.last_rebalance_time = datetime.now()
        
        # Publish rebalance event
        event_data = {
            "rebalance_type": "periodic" if self._should_rebalance_periodic() else "drift",
            "adjustments": {s: round(q, 6) for s, q in adjustments.items()},
            "timestamp": datetime.now(),
            "strategy": self.__class__.__name__
        }
        
        self.event_bus.publish("PORTFOLIO_REBALANCE", event_data)
        
        logger.info(f"Completed portfolio rebalance with {len(adjustments)} adjustments")
    
    def _execute_rebalance_trades(self, adjustments: Dict[str, float], portfolio_value: float) -> None:
        """
        Execute trades to rebalance the portfolio based on the specified execution model.
        
        Args:
            adjustments: Dictionary of quantity adjustments by symbol
            portfolio_value: Current portfolio value
        """
        execution_model = self.parameters["execution_model"]
        
        if execution_model == "immediate":
            # Execute all trades immediately
            for symbol, quantity in adjustments.items():
                self._create_rebalance_order(symbol, quantity)
        elif execution_model == "gradual" or execution_model == "twap":
            # Schedule gradual execution (handled by a separate process)
            execution_duration = self.parameters["execution_duration_hours"]
            
            # Create a rebalance plan
            rebalance_plan = {
                "adjustments": adjustments,
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(hours=execution_duration),
                "portfolio_value": portfolio_value,
                "execution_model": execution_model
            }
            
            # Publish the plan for execution by the order execution service
            self.event_bus.publish("EXECUTION_PLAN", {
                "plan": rebalance_plan,
                "strategy": self.__class__.__name__
            })
            
            logger.info(f"Created {execution_model} execution plan for rebalance over {execution_duration} hours")
            
            # For immediate feedback, execute a small portion of each adjustment
            first_portion = min(0.2, 1.0 / (execution_duration * 2))  # Initial increment
            
            for symbol, quantity in adjustments.items():
                initial_quantity = quantity * first_portion
                if abs(initial_quantity) > 0.0001:  # Minimum quantity threshold
                    self._create_rebalance_order(symbol, initial_quantity)
    
    def _create_rebalance_order(self, symbol: str, quantity: float) -> None:
        """
        Create an order to adjust a position for rebalancing.
        
        Args:
            symbol: Symbol to adjust position for
            quantity: Quantity adjustment (positive for buy, negative for sell)
        """
        if abs(quantity) <= 0.0001:  # Minimum quantity threshold
            return
            
        # Determine order direction
        direction = "buy" if quantity > 0 else "sell"
        
        # Create market order for the adjustment
        order_params = {
            "symbol": symbol,
            "order_type": "market",
            "direction": direction,
            "quantity": abs(quantity),
            "time_in_force": "GTC",
            "source": self.__class__.__name__
        }
        
        try:
            self.session.create_order(**order_params)
            logger.info(f"Created {direction} order for {abs(quantity)} {symbol} for rebalancing")
        except Exception as e:
            logger.error(f"Failed to create rebalance order for {symbol}: {e}")
    
    def _calculate_performance_metrics(self) -> None:
        """
        Calculate performance metrics for the index portfolio.
        """
        # Get recent portfolio values
        if len(self.portfolio_value_history) < 2:
            return
            
        portfolio_values = sorted([(ts, val) for ts, val in self.portfolio_value_history.items()])
        
        # Calculate daily performance
        current_value = portfolio_values[-1][1]
        prev_day_idx = -2
        
        # Try to find a value from ~1 day ago
        for i in range(len(portfolio_values) - 2, -1, -1):
            if (portfolio_values[-1][0] - portfolio_values[i][0]).days >= 1:
                prev_day_idx = i
                break
                
        if prev_day_idx >= 0 and portfolio_values[prev_day_idx][1] > 0:
            daily_return = (current_value / portfolio_values[prev_day_idx][1]) - 1
            logger.info(f"Daily portfolio return: {daily_return:.2%}")
            
            # Publish performance update
            self.event_bus.publish("PORTFOLIO_PERFORMANCE", {
                "daily_return": daily_return,
                "current_value": current_value,
                "timestamp": datetime.now(),
                "strategy": self.__class__.__name__
            })
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for the strategy. For index investing, these are portfolio-level metrics.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        # Initialize indicators dictionary
        indicators = {}
        
        # Basic price info (can be used for benchmarking)
        if not data.empty:
            indicators["current_price"] = data["close"].iloc[-1]
            
            # Calculate simple benchmark performance
            if len(data) > 1:
                daily_return = (data["close"].iloc[-1] / data["close"].iloc[-2]) - 1
                indicators["benchmark_daily_return"] = daily_return
                
            # Calculate longer-term performance
            if len(data) > 30:
                monthly_return = (data["close"].iloc[-1] / data["close"].iloc[-30]) - 1
                indicators["benchmark_monthly_return"] = monthly_return
        
        # Portfolio diversification metrics
        if self.current_allocations:
            # Number of assets
            indicators["num_assets"] = len(self.current_allocations)
            
            # Concentration metrics (Herfindahl-Hirschman Index)
            hhi = sum(weight ** 2 for weight in self.current_allocations.values())
            indicators["concentration_hhi"] = hhi
            
            # Maximum allocation
            indicators["max_allocation"] = max(self.current_allocations.values()) if self.current_allocations else 0
            
            # Asset class/sector distribution
            sector_allocations = {}
            for symbol, weight in self.current_allocations.items():
                sector = self.sector_mapping.get(symbol, "other")
                sector_allocations[sector] = sector_allocations.get(sector, 0) + weight
                
            indicators["sector_allocations"] = sector_allocations
        
        # Portfolio risk metrics
        # These would typically be calculated from historical data analysis
        # Simplified calculation here for illustration
        portfolio_volatility = 0.0
        if self.portfolio_value_history and len(self.portfolio_value_history) > 10:
            # Convert to pandas series
            values = pd.Series([v for _, v in sorted(self.portfolio_value_history.items())])
            returns = values.pct_change().dropna()
            
            if len(returns) > 0:
                # Annualized volatility
                portfolio_volatility = returns.std() * np.sqrt(252)  # Annualized
                indicators["portfolio_volatility"] = portfolio_volatility
        
        # Rebalance indicators
        if self.target_allocations and self.current_allocations:
            # Calculate tracking error (how far from target allocations)
            tracking_error = 0.0
            for symbol in set(list(self.target_allocations.keys()) + list(self.current_allocations.keys())):
                target = self.target_allocations.get(symbol, 0.0)
                current = self.current_allocations.get(symbol, 0.0)
                tracking_error += abs(target - current)
                
            indicators["tracking_error"] = tracking_error
            indicators["needs_rebalance"] = tracking_error > self.parameters["rebalance_threshold_pct"]
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on the index strategy and current state.
        
        Args:
            data: Market data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Dictionary of signals
        """
        signals = {
            "signal": "neutral",
            "direction": "neutral",
            "confidence": 0.0,
            "triggers": []
        }
        
        # Index strategies are primarily based on periodic rebalancing rather than
        # tactical entry/exit signals. However, we can generate signals for portfolio adjustments.
        
        # Check if rebalance is needed
        if indicators.get("needs_rebalance", False):
            signals["signal"] = "rebalance"
            signals["triggers"].append("tracking_error")
            signals["confidence"] = min(1.0, indicators.get("tracking_error", 0) * 10)  # Scale by tracking error
        
        # Check risk thresholds
        if indicators.get("portfolio_volatility", 0) > self.parameters["volatility_cap"] and self.parameters["enable_volatility_targeting"]:
            signals["signal"] = "reduce_risk"
            signals["triggers"].append("volatility_cap_exceeded")
            signals["confidence"] = 0.8
        
        # Handle concentration risk
        if indicators.get("max_allocation", 0) > self.parameters["max_asset_allocation"]:
            signals["signal"] = "reduce_concentration"
            signals["triggers"].append("concentration_exceeded")
            signals["confidence"] = 0.7
        
        # Initial portfolio setup
        if not self.current_allocations and self.target_allocations:
            signals["signal"] = "initialize_portfolio"
            signals["triggers"].append("new_portfolio")
            signals["confidence"] = 1.0
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on the index methodology and current portfolio state.
        For index strategies, position sizes are determined by target allocations rather than
        traditional sizing calculations.
        
        Args:
            direction: Order direction ('long' or 'short')
            data: Market data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Position size as a decimal (0.0-1.0) representing account % to risk
        """
        # For index strategies, this method is less applicable since positions are based on allocations
        # However, we can use it to determine the total portfolio exposure
        
        # Default to a conservative position size
        default_size = 0.5  # 50% of available capital
        
        # Adjust based on volatility if targeting is enabled
        if self.parameters["enable_volatility_targeting"] and "portfolio_volatility" in indicators:
            portfolio_volatility = indicators["portfolio_volatility"]
            target_volatility = self.parameters["volatility_cap"]
            
            if portfolio_volatility > 0:
                # Scale position size to target volatility
                vol_ratio = target_volatility / portfolio_volatility
                size = default_size * vol_ratio
                return max(0.1, min(1.0, size))  # Ensure between 10-100%
        
        # Otherwise, return a default size
        return default_size
    
    def regime_compatibility(self, market_regime: Dict[str, Any]) -> float:
        """
        Calculate compatibility score for the strategy with the current market regime.
        
        Args:
            market_regime: Dictionary describing the current market regime
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # Index investing strategies are generally robust across regimes,
        # but can be adjusted based on certain conditions
        
        # Default moderate compatibility
        compatibility = 0.6
        
        # Get regime properties
        regime_type = market_regime.get("regime_type", "unknown")
        volatility = market_regime.get("volatility", "normal")
        correlation = market_regime.get("correlation", "normal")
        liquidity = market_regime.get("liquidity", "normal")
        
        # Index type specific adjustments
        index_type = self.parameters["index_type"]
        
        # Adjust based on regime type
        if regime_type == "bull_market" or regime_type == "risk_on":
            # Bull markets favor diversified index strategies
            compatibility += 0.2
        elif regime_type == "bear_market" or regime_type == "risk_off":
            # Bear markets are slightly less favorable
            compatibility -= 0.1
            
            # But volatility-weighted indices may perform better
            if index_type == "volatility_weighted":
                compatibility += 0.15
        elif regime_type == "sideways" or regime_type == "consolidation":
            # Sideways markets are neutral for index strategies
            pass
        
        # Adjust based on volatility
        if volatility == "high":
            # High volatility favors volatility-weighted approaches
            if index_type == "volatility_weighted":
                compatibility += 0.15
            else:
                compatibility -= 0.05
        elif volatility == "low":
            # Low volatility is good for most index approaches
            compatibility += 0.05
        
        # Adjust based on correlation
        if correlation == "high":
            # High correlation reduces diversification benefits
            compatibility -= 0.1
            
            # Sector-weighted may help in high correlation environments
            if index_type == "sector_weighted":
                compatibility += 0.05
        elif correlation == "low":
            # Low correlation enhances diversification benefits
            compatibility += 0.1
        
        # Adjust based on liquidity
        if liquidity == "low":
            # Low liquidity can be problematic for frequent rebalancing
            compatibility -= 0.2
            
            # Equal-weighted may suffer more in low liquidity
            if index_type == "equal_weighted":
                compatibility -= 0.05
        
        # Ensure the result is between 0 and 1
        return max(0.1, min(1.0, compatibility))
