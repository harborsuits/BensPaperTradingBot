#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto On-Chain & Order-Book Analysis Strategy

This strategy combines on-chain blockchain data analytics with order book analysis
to generate trading signals. It leverages blockchain-specific metrics such as 
network activity, wallet movements, mining difficulty, and exchange flows alongside
order book depth and imbalance for a comprehensive approach to crypto trading.

Key characteristics:
- Utilizes blockchain data for medium-term trend analysis
- Combines with order book data for short-term entry/exit timing
- Includes whale wallet monitoring for large holder movements
- Analyzes exchange inflows/outflows for liquidity shifts
- Identifies supply-demand imbalances from multiple sources
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from collections import deque

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoOnChainStrategy",
    market_type="crypto",
    description="Strategy combining on-chain blockchain data with order book analysis for crypto markets",
    timeframes=["H1", "H4", "D1"],  # On-chain analysis typically works best on longer timeframes
    parameters={
        # On-Chain Data Parameters
        "enable_onchain_data": {"type": "bool", "default": True},
        "onchain_update_interval": {"type": "int", "default": 3600, "min": 300, "max": 86400},
        "whale_threshold_btc": {"type": "float", "default": 100.0, "min": 10.0, "max": 1000.0},
        "whale_threshold_eth": {"type": "float", "default": 1000.0, "min": 100.0, "max": 10000.0},
        "exchange_flow_threshold": {"type": "float", "default": 0.02, "min": 0.005, "max": 0.1},
        "network_congestion_threshold": {"type": "float", "default": 0.7, "min": 0.3, "max": 0.9},
        
        # Order Book Parameters
        "enable_orderbook_analysis": {"type": "bool", "default": True},
        "ob_depth_levels": {"type": "int", "default": 10, "min": 5, "max": 50},
        "ob_imbalance_threshold": {"type": "float", "default": 1.5, "min": 1.1, "max": 5.0},
        "ob_wall_detection_multiple": {"type": "float", "default": 3.0, "min": 1.5, "max": 10.0},
        "ob_liquidity_threshold": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.5},
        
        # Signal Generation
        "onchain_signal_weight": {"type": "float", "default": 0.6, "min": 0.1, "max": 0.9},
        "orderbook_signal_weight": {"type": "float", "default": 0.4, "min": 0.1, "max": 0.9},
        "combined_signal_threshold": {"type": "float", "default": 0.6, "min": 0.3, "max": 0.9},
        "min_onchain_confidence": {"type": "float", "default": 0.5, "min": 0.3, "max": 0.9},
        "min_orderbook_confidence": {"type": "float", "default": 0.5, "min": 0.3, "max": 0.9},
        
        # Transaction Parameters
        "use_laddered_entries": {"type": "bool", "default": True},
        "ladder_levels": {"type": "int", "default": 3, "min": 2, "max": 5},
        "ladder_price_step": {"type": "float", "default": 0.005, "min": 0.001, "max": 0.02},
        
        # Risk Management
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 5},
        "use_adaptive_stops": {"type": "bool", "default": True},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "stop_atr_multiple": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        
        # API Configuration
        "enable_api_calls": {"type": "bool", "default": False},
        "api_rate_limit": {"type": "int", "default": 5, "min": 1, "max": 20},
        "api_call_timeout": {"type": "int", "default": 10, "min": 3, "max": 30},
        "use_api_cache": {"type": "bool", "default": True},
        "api_cache_ttl": {"type": "int", "default": 900, "min": 60, "max": 3600},
    }
)
class CryptoOnChainStrategy(CryptoBaseStrategy):
    """
    A strategy combining on-chain blockchain data with order book analysis.
    
    This strategy:
    1. Collects and analyzes on-chain data (transaction volumes, wallet analytics, mining stats)
    2. Combines with order book metrics (depth, walls, imbalance)
    3. Generates trading signals considering both long-term (on-chain) and short-term (order book) factors
    4. Uses adaptive entry and exit strategies based on current blockchain conditions
    5. Implements risk management tailored to the volatility of crypto assets
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto on-chain and order book analysis strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific state
        self.onchain_data = {}
        self.onchain_history = []
        self.onchain_signal = "neutral"
        self.onchain_confidence = 0.0
        self.last_onchain_update = None
        
        self.orderbook_analysis = {}
        self.orderbook_history = deque(maxlen=100)
        self.orderbook_signal = "neutral"
        self.orderbook_confidence = 0.0
        
        self.combined_signal = "neutral"
        self.combined_signal_strength = 0.0
        
        # API caching
        self.api_cache = {}
        self.api_call_timestamps = deque(maxlen=100)
        
        # Entry ladders and exit targets
        self.entry_ladders = {}  # position_id -> list of entry prices
        self.exit_targets = {}   # position_id -> list of exit targets
        
        # Initialize supported assets
        self.supported_assets = ["BTC", "ETH", "SOL", "BNB", "XRP"]
        self.current_asset = self.session.symbol.split("-")[0]
        
        # Check if current asset is supported
        if self.current_asset not in self.supported_assets:
            logger.warning(f"Asset {self.current_asset} has limited on-chain data support. Some features may not work.")
        
        logger.info(f"Initialized crypto on-chain strategy for {self.current_asset} with "
                   f"on-chain weight: {self.parameters['onchain_signal_weight']}, "
                   f"order book weight: {self.parameters['orderbook_signal_weight']}")
                   
        # Initialize on-chain data if enabled
        if self.parameters["enable_onchain_data"]:
            self._initialize_onchain_data()
    
    def _initialize_onchain_data(self) -> None:
        """Initialize on-chain data structures."""
        # In a real implementation, this would fetch initial blockchain data
        # For this mock version, we'll just set up placeholder data
        self.onchain_data = {
            "network_activity": {
                "transactions_per_day": 0,
                "active_addresses": 0,
                "avg_transaction_value": 0,
                "avg_transaction_fee": 0,
                "mempool_size": 0,
                "network_hash_rate": 0,
                "network_difficulty": 0,
            },
            "exchange_flows": {
                "inflow_24h": 0,
                "outflow_24h": 0,
                "net_flow_24h": 0,
                "exchange_balance": 0,
                "exchange_balance_change_pct": 0,
            },
            "wallet_analytics": {
                "whale_addresses_count": 0,
                "whale_holdings_pct": 0,
                "whale_transactions_24h": 0,
                "distribution_trend": "neutral",
                "accumulation_wallets_pct": 0,
                "distribution_wallets_pct": 0,
            },
            "miner_activity": {
                "miner_outflow": 0,
                "miner_netflow": 0,
                "mining_difficulty_change": 0,
                "hash_ribbons_signal": "neutral",
            },
            "staking_metrics": {
                "total_staked": 0,
                "staking_yield": 0,
                "staking_ratio": 0,
            },
            "defi_metrics": {
                "total_value_locked": 0,
                "tvl_change_24h": 0,
                "defi_dominance": 0,
            },
            "last_updated": datetime.utcnow(),
        }
        
        # Initialize history with current values
        self.onchain_history.append(self.onchain_data.copy())
    
    def _update_onchain_data(self) -> None:
        """
        Update on-chain metrics from blockchain APIs.
        
        In a production environment, this would make API calls to services 
        like Glassnode, Chainalysis, CoinMetrics, or direct blockchain nodes.
        """
        # Check if update is needed based on interval
        current_time = datetime.utcnow()
        if self.last_onchain_update:
            seconds_since_update = (current_time - self.last_onchain_update).total_seconds()
            if seconds_since_update < self.parameters["onchain_update_interval"]:
                logger.debug(f"Skipping on-chain update, next update in {self.parameters['onchain_update_interval'] - seconds_since_update:.0f}s")
                return
        
        # Skip API calls if disabled in parameters
        if not self.parameters["enable_api_calls"]:
            self._simulate_onchain_data()
            self.last_onchain_update = current_time
            logger.info(f"Updated on-chain data (simulation mode)")
            return
        
        # In a real implementation, make API calls for real data
        try:
            # Rate limit check
            self._check_api_rate_limit()
            
            # Cache check
            cache_key = f"onchain_{self.current_asset}_{current_time.strftime('%Y%m%d')}"
            if self.parameters["use_api_cache"] and cache_key in self.api_cache:
                cache_entry = self.api_cache[cache_key]
                cache_age = (current_time - cache_entry["timestamp"]).total_seconds()
                
                if cache_age < self.parameters["api_cache_ttl"]:
                    self.onchain_data = cache_entry["data"]
                    logger.debug(f"Using cached on-chain data ({cache_age:.0f}s old)")
                    return
            
            # This would be replaced with actual API calls in production
            # For example:
            # network_data = requests.get("https://api.glassnode.com/v1/metrics/network_activity", 
            #                            params={"api_key": "YOUR_API_KEY", "asset": self.current_asset})
            
            self._simulate_onchain_data()
            
            # Update cache
            self.api_cache[cache_key] = {
                "data": self.onchain_data.copy(),
                "timestamp": current_time
            }
            
            # Prune old cache entries
            self._prune_api_cache()
            
            # Update history
            self.onchain_history.append(self.onchain_data.copy())
            if len(self.onchain_history) > 24:  # Keep 24 hours of history
                self.onchain_history = self.onchain_history[-24:]
                
            self.last_onchain_update = current_time
            logger.info(f"Updated on-chain data via API")
            
        except Exception as e:
            logger.error(f"Error updating on-chain data: {str(e)}")
            
    def _simulate_onchain_data(self) -> None:
        """Simulate on-chain data for testing when APIs are not available."""
        # For simulation, we'll create realistic but random values
        # In production, these would come from actual blockchain data
        
        # Get base values from current price data for realism
        current_price = 0
        if not self.market_data.empty:
            current_price = self.market_data["close"].iloc[-1]
        
        # Simulate network activity
        self.onchain_data["network_activity"]["transactions_per_day"] = np.random.randint(200000, 500000)
        self.onchain_data["network_activity"]["active_addresses"] = np.random.randint(800000, 1200000)
        self.onchain_data["network_activity"]["avg_transaction_value"] = current_price * np.random.uniform(0.1, 0.3)
        self.onchain_data["network_activity"]["avg_transaction_fee"] = current_price * np.random.uniform(0.0001, 0.001)
        self.onchain_data["network_activity"]["mempool_size"] = np.random.randint(5000, 20000)
        
        if self.current_asset == "BTC":
            self.onchain_data["network_activity"]["network_hash_rate"] = np.random.uniform(100, 120) * 1e18
            self.onchain_data["network_activity"]["network_difficulty"] = np.random.uniform(35, 40) * 1e12
        elif self.current_asset == "ETH":
            self.onchain_data["network_activity"]["network_hash_rate"] = np.random.uniform(800, 900) * 1e12
            self.onchain_data["network_activity"]["network_difficulty"] = np.random.uniform(10, 12) * 1e15
            
        # Simulate exchange flows
        daily_volume = current_price * np.random.uniform(50000, 200000)
        self.onchain_data["exchange_flows"]["inflow_24h"] = daily_volume * np.random.uniform(0.4, 0.6)
        self.onchain_data["exchange_flows"]["outflow_24h"] = daily_volume * np.random.uniform(0.4, 0.6)
        self.onchain_data["exchange_flows"]["net_flow_24h"] = self.onchain_data["exchange_flows"]["inflow_24h"] - self.onchain_data["exchange_flows"]["outflow_24h"]
        self.onchain_data["exchange_flows"]["exchange_balance"] = current_price * np.random.uniform(1000000, 3000000)
        self.onchain_data["exchange_flows"]["exchange_balance_change_pct"] = np.random.uniform(-0.02, 0.02)
        
        # Simulate wallet analytics
        self.onchain_data["wallet_analytics"]["whale_addresses_count"] = np.random.randint(80, 120)
        self.onchain_data["wallet_analytics"]["whale_holdings_pct"] = np.random.uniform(0.4, 0.6)
        self.onchain_data["wallet_analytics"]["whale_transactions_24h"] = np.random.randint(50, 200)
        
        # Randomize distribution trend
        rand_trend = np.random.random()
        if rand_trend < 0.33:
            self.onchain_data["wallet_analytics"]["distribution_trend"] = "accumulation"
            self.onchain_data["wallet_analytics"]["accumulation_wallets_pct"] = np.random.uniform(0.55, 0.7)
            self.onchain_data["wallet_analytics"]["distribution_wallets_pct"] = np.random.uniform(0.3, 0.45)
        elif rand_trend < 0.66:
            self.onchain_data["wallet_analytics"]["distribution_trend"] = "neutral"
            self.onchain_data["wallet_analytics"]["accumulation_wallets_pct"] = np.random.uniform(0.45, 0.55)
            self.onchain_data["wallet_analytics"]["distribution_wallets_pct"] = np.random.uniform(0.45, 0.55)
        else:
            self.onchain_data["wallet_analytics"]["distribution_trend"] = "distribution"
            self.onchain_data["wallet_analytics"]["accumulation_wallets_pct"] = np.random.uniform(0.3, 0.45)
            self.onchain_data["wallet_analytics"]["distribution_wallets_pct"] = np.random.uniform(0.55, 0.7)
            
        # Simulate miner activity
        self.onchain_data["miner_activity"]["miner_outflow"] = current_price * np.random.uniform(1000, 5000)
        self.onchain_data["miner_activity"]["miner_netflow"] = self.onchain_data["miner_activity"]["miner_outflow"] * np.random.uniform(-0.2, 0.2)
        self.onchain_data["miner_activity"]["mining_difficulty_change"] = np.random.uniform(-0.05, 0.08)
        
        # Randomize hash ribbons signal
        hash_signal = np.random.random()
        if hash_signal < 0.2:
            self.onchain_data["miner_activity"]["hash_ribbons_signal"] = "capitulation"
        elif hash_signal < 0.8:
            self.onchain_data["miner_activity"]["hash_ribbons_signal"] = "neutral"
        else:
            self.onchain_data["miner_activity"]["hash_ribbons_signal"] = "bullish"
            
        # Staking metrics (mainly for PoS chains)
        if self.current_asset in ["ETH", "SOL", "BNB"]:
            self.onchain_data["staking_metrics"]["total_staked"] = np.random.uniform(0.5, 0.7) * 1e8
            self.onchain_data["staking_metrics"]["staking_yield"] = np.random.uniform(0.03, 0.08)
            self.onchain_data["staking_metrics"]["staking_ratio"] = np.random.uniform(0.5, 0.7)
            
        # DeFi metrics
        self.onchain_data["defi_metrics"]["total_value_locked"] = current_price * np.random.uniform(5e9, 8e9)
        self.onchain_data["defi_metrics"]["tvl_change_24h"] = np.random.uniform(-0.05, 0.08)
        self.onchain_data["defi_metrics"]["defi_dominance"] = np.random.uniform(0.1, 0.25)
        
        # Update timestamp
        self.onchain_data["last_updated"] = datetime.utcnow()
    
    def _check_api_rate_limit(self) -> None:
        """
        Check if API calls are within rate limit and delay if necessary.
        """
        current_time = datetime.utcnow()
        self.api_call_timestamps.append(current_time)
        
        # Check if we've made too many calls recently
        if len(self.api_call_timestamps) >= self.parameters["api_rate_limit"]:
            # Calculate time since oldest call
            oldest_call = self.api_call_timestamps[0]
            time_diff = (current_time - oldest_call).total_seconds()
            
            # If we've made too many calls in less than a second, delay
            if time_diff < 1.0:
                sleep_time = 1.1 - time_diff  # Add a little buffer
                logger.debug(f"Rate limiting API calls, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
    def _prune_api_cache(self) -> None:
        """
        Remove expired entries from the API cache.
        """
        current_time = datetime.utcnow()
        keys_to_remove = []
        
        for key, entry in self.api_cache.items():
            cache_age = (current_time - entry["timestamp"]).total_seconds()
            if cache_age > self.parameters["api_cache_ttl"] * 2:  # Double TTL for cleanup
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.api_cache[key]
            
        if keys_to_remove:
            logger.debug(f"Pruned {len(keys_to_remove)} expired entries from API cache")
            
    def _analyze_orderbook(self) -> Dict[str, Any]:
        """
        Analyze the current order book for patterns, imbalances, and liquidity.
        
        Returns:
            Dictionary of order book analysis results
        """
        if not self.orderbook:
            return {}
            
        # Extract order book data
        bids = self.orderbook.get('bids', [])
        asks = self.orderbook.get('asks', [])
        
        if not bids or not asks:
            return {}
            
        # Limit depth according to parameters
        depth = self.parameters["ob_depth_levels"]
        bids = bids[:depth] if len(bids) > depth else bids
        asks = asks[:depth] if len(asks) > depth else asks
        
        # Calculate total volume and average price
        bid_volume = sum([float(bid[1]) for bid in bids])
        ask_volume = sum([float(ask[1]) for ask in asks])
        
        bid_value = sum([float(bid[0]) * float(bid[1]) for bid in bids])
        ask_value = sum([float(ask[0]) * float(ask[1]) for ask in asks])
        
        bid_avg_price = bid_value / bid_volume if bid_volume > 0 else 0
        ask_avg_price = ask_value / ask_volume if ask_volume > 0 else 0
        
        # Calculate price levels
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        midpoint = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        spread_pct = (spread / midpoint) * 100 if midpoint else 0
        
        # Calculate order book imbalance
        volume_imbalance = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        value_imbalance = bid_value / ask_value if ask_value > 0 else float('inf')
        
        # Detect large orders (walls)
        wall_threshold = self.parameters["ob_wall_detection_multiple"] * (bid_volume / len(bids)) if bids else 0
        bid_walls = []
        for i, bid in enumerate(bids):
            if float(bid[1]) > wall_threshold:
                bid_walls.append((float(bid[0]), float(bid[1])))
                
        wall_threshold = self.parameters["ob_wall_detection_multiple"] * (ask_volume / len(asks)) if asks else 0
        ask_walls = []
        for i, ask in enumerate(asks):
            if float(ask[1]) > wall_threshold:
                ask_walls.append((float(ask[0]), float(ask[1])))
                
        # Measure liquidity as volume within 1% of midpoint
        near_bid_volume = 0
        for bid in bids:
            if float(bid[0]) > midpoint * 0.99:
                near_bid_volume += float(bid[1])
                
        near_ask_volume = 0
        for ask in asks:
            if float(ask[0]) < midpoint * 1.01:
                near_ask_volume += float(ask[1])
                
        near_midpoint_volume = near_bid_volume + near_ask_volume
        liquidity_score = near_midpoint_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Return analysis dictionary
        return {
            "timestamp": datetime.utcnow(),
            "midpoint": midpoint,
            "spread": spread,
            "spread_pct": spread_pct,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "volume_imbalance": volume_imbalance,
            "value_imbalance": value_imbalance,
            "bid_avg_price": bid_avg_price,
            "ask_avg_price": ask_avg_price,
            "bid_walls": bid_walls,
            "ask_walls": ask_walls,
            "liquidity_score": liquidity_score,
            "near_midpoint_volume": near_midpoint_volume,
        }
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators using both on-chain data and traditional market data.
        
        Args:
            data: Market price data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty:
            return indicators
        
        # Update on-chain data if enabled
        if self.parameters["enable_onchain_data"]:
            self._update_onchain_data()
            
        # Calculate traditional technical indicators for comparison
        # SMA for trend direction
        indicators["sma50"] = data["close"].rolling(window=50).mean()
        indicators["sma200"] = data["close"].rolling(window=200).mean()
        indicators["trend_direction"] = "bullish" if indicators["sma50"].iloc[-1] > indicators["sma200"].iloc[-1] else "bearish"
        
        # RSI for overbought/oversold conditions
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR for volatility measurement
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = tr.rolling(window=self.parameters["atr_period"]).mean()
        
        # Analyze order book if enabled
        if self.parameters["enable_orderbook_analysis"] and self.orderbook:
            ob_analysis = self._analyze_orderbook()
            if ob_analysis:
                # Add order book metrics to indicators
                for key, value in ob_analysis.items():
                    indicators[f"ob_{key}"] = value
                    
                # Store in history for trend analysis
                self.orderbook_history.append(ob_analysis)
                
                # Analyze order book trends
                if len(self.orderbook_history) >= 3:
                    self._analyze_orderbook_trends(indicators)
        
        # Process on-chain signals if data is available
        if self.onchain_data and "last_updated" in self.onchain_data:
            self._process_onchain_signals(indicators)
            
        # Process order book signals
        if "ob_volume_imbalance" in indicators:
            self._process_orderbook_signals(indicators)
            
        # Combine signals
        self._combine_signals(indicators)
            
        return indicators
        
    def _analyze_orderbook_trends(self, indicators: Dict[str, Any]) -> None:
        """
        Analyze trends in order book metrics over time.
        
        Args:
            indicators: Indicators dictionary to update with trend metrics
        """
        # Get last few order book snapshots for trend analysis
        recent_obs = list(self.orderbook_history)[-3:]
        
        # Calculate bid/ask volume trends
        bid_volumes = [ob.get("bid_volume", 0) for ob in recent_obs]
        ask_volumes = [ob.get("ask_volume", 0) for ob in recent_obs]
        
        indicators["ob_bid_volume_trend"] = "increasing" if bid_volumes[-1] > bid_volumes[0] else "decreasing"
        indicators["ob_ask_volume_trend"] = "increasing" if ask_volumes[-1] > ask_volumes[0] else "decreasing"
        
        # Calculate imbalance trends
        imbalances = [ob.get("volume_imbalance", 1.0) for ob in recent_obs]
        indicators["ob_imbalance_trend"] = "increasing" if imbalances[-1] > imbalances[0] else "decreasing"
        
        # Calculate liquidity trends
        liquidity = [ob.get("liquidity_score", 0) for ob in recent_obs]
        indicators["ob_liquidity_trend"] = "improving" if liquidity[-1] > liquidity[0] else "decreasing"
        
        # Detect order book walls appearing or disappearing
        current_ob = recent_obs[-1]
        previous_ob = recent_obs[0]
        
        indicators["ob_new_bid_walls"] = len(current_ob.get("bid_walls", [])) > len(previous_ob.get("bid_walls", []))
        indicators["ob_new_ask_walls"] = len(current_ob.get("ask_walls", [])) > len(previous_ob.get("ask_walls", []))
        
    def _process_onchain_signals(self, indicators: Dict[str, Any]) -> None:
        """
        Process on-chain data to generate trading signals.
        
        Args:
            indicators: Indicators dictionary to update with on-chain signals
        """
        # Extract key on-chain metrics
        network_data = self.onchain_data.get("network_activity", {})
        exchange_flows = self.onchain_data.get("exchange_flows", {})
        wallet_data = self.onchain_data.get("wallet_analytics", {})
        miner_data = self.onchain_data.get("miner_activity", {})
        
        # Calculate network congestion (higher values indicate congestion)
        if "mempool_size" in network_data and "transactions_per_day" in network_data:
            mempool_ratio = network_data["mempool_size"] / (network_data["transactions_per_day"] / 24)
            indicators["network_congestion"] = min(1.0, mempool_ratio / 0.1)  # Normalize to 0-1
        
        # Exchange flow analysis (negative is outflow from exchanges, potentially bullish)
        net_flow_ratio = 0
        if "net_flow_24h" in exchange_flows and "exchange_balance" in exchange_flows and exchange_flows["exchange_balance"] > 0:
            net_flow_ratio = exchange_flows["net_flow_24h"] / exchange_flows["exchange_balance"]
            indicators["exchange_flow_ratio"] = net_flow_ratio
        
        # Wallet distribution analysis
        distribution_trend = wallet_data.get("distribution_trend", "neutral")
        indicators["wallet_distribution"] = distribution_trend
        
        # Miner activity analysis
        hash_signal = miner_data.get("hash_ribbons_signal", "neutral")
        indicators["hash_ribbons"] = hash_signal
        
        # Generate on-chain signal
        onchain_bull_points = 0
        onchain_bear_points = 0
        signal_confidence = 0.5  # Start at neutral
        
        # Network congestion: High congestion can lead to high fees and decreased usage
        if "network_congestion" in indicators:
            congestion = indicators["network_congestion"]
            if congestion > self.parameters["network_congestion_threshold"]:
                onchain_bear_points += 1
                signal_confidence += 0.05
            elif congestion < self.parameters["network_congestion_threshold"] / 2:
                onchain_bull_points += 1
                signal_confidence += 0.05
        
        # Exchange flows: Outflows from exchanges often precede price increases
        if "exchange_flow_ratio" in indicators:
            flow_ratio = indicators["exchange_flow_ratio"]
            flow_threshold = self.parameters["exchange_flow_threshold"]
            
            if flow_ratio < -flow_threshold:  # Net outflow (bullish)
                onchain_bull_points += 2  # Stronger signal
                signal_confidence += 0.15
            elif flow_ratio > flow_threshold:  # Net inflow (bearish)
                onchain_bear_points += 2
                signal_confidence += 0.15
        
        # Wallet distribution: Accumulation patterns from whale wallets
        if indicators["wallet_distribution"] == "accumulation":
            onchain_bull_points += 1
            signal_confidence += 0.1
        elif indicators["wallet_distribution"] == "distribution":
            onchain_bear_points += 1
            signal_confidence += 0.1
        
        # Miner activity: Hash ribbons can signal miner capitulation or confidence
        if indicators["hash_ribbons"] == "bullish":
            onchain_bull_points += 1
            signal_confidence += 0.1
        elif indicators["hash_ribbons"] == "capitulation":
            onchain_bear_points += 1
            signal_confidence += 0.1
        
        # Determine final on-chain signal
        onchain_signal = "neutral"
        if onchain_bull_points > onchain_bear_points + 1:  # Require stronger bullish confirmation
            onchain_signal = "bullish"
        elif onchain_bear_points > onchain_bull_points:
            onchain_signal = "bearish"
            
        # Store signal and confidence
        self.onchain_signal = onchain_signal
        self.onchain_confidence = min(1.0, signal_confidence)
        
        # Add to indicators
        indicators["onchain_signal"] = onchain_signal
        indicators["onchain_confidence"] = self.onchain_confidence
        
        logger.info(f"On-chain signal: {onchain_signal} (confidence: {self.onchain_confidence:.2f})")
        
    def _process_orderbook_signals(self, indicators: Dict[str, Any]) -> None:
        """
        Process order book data to generate short-term trading signals.
        
        Args:
            indicators: Indicators dictionary to update with order book signals
        """
        # Extract key order book metrics
        volume_imbalance = indicators.get("ob_volume_imbalance", 1.0)
        bid_walls = indicators.get("ob_bid_walls", [])
        ask_walls = indicators.get("ob_ask_walls", [])
        liquidity_score = indicators.get("ob_liquidity_score", 0.5)
        imbalance_trend = indicators.get("ob_imbalance_trend", "neutral")
        
        # Set thresholds from parameters
        imbalance_threshold = self.parameters["ob_imbalance_threshold"]
        liquidity_threshold = self.parameters["ob_liquidity_threshold"]
        
        # Initialize signal points
        ob_bull_points = 0
        ob_bear_points = 0
        signal_confidence = 0.5  # Start at neutral
        
        # Volume imbalance analysis
        if volume_imbalance > imbalance_threshold:  # More bids than asks (bullish)
            ob_bull_points += 1
            signal_confidence += 0.1
        elif volume_imbalance < 1.0 / imbalance_threshold:  # More asks than bids (bearish)
            ob_bear_points += 1
            signal_confidence += 0.1
            
        # Order walls analysis
        if len(bid_walls) > len(ask_walls):  # Strong support levels (bullish)
            ob_bull_points += 1
            signal_confidence += 0.05
        elif len(ask_walls) > len(bid_walls):  # Strong resistance levels (bearish)
            ob_bear_points += 1
            signal_confidence += 0.05
            
        # Liquidity analysis
        if liquidity_score > liquidity_threshold:  # High liquidity near midpoint
            # This is market-neutral but slightly bullish as it suggests stability
            ob_bull_points += 0.5
            signal_confidence += 0.05
        elif liquidity_score < liquidity_threshold / 2:  # Low liquidity (potential volatility)
            # This is slightly bearish as it suggests potential for slippage
            ob_bear_points += 0.5
            signal_confidence += 0.05
            
        # Trend analysis
        if imbalance_trend == "increasing" and volume_imbalance > 1.0:  # Increasing bid pressure
            ob_bull_points += 1
            signal_confidence += 0.1
        elif imbalance_trend == "increasing" and volume_imbalance < 1.0:  # Increasing ask pressure
            ob_bear_points += 1
            signal_confidence += 0.1
            
        # New wall detection
        if indicators.get("ob_new_bid_walls", False):  # New support levels forming
            ob_bull_points += 1
            signal_confidence += 0.05
        if indicators.get("ob_new_ask_walls", False):  # New resistance levels forming
            ob_bear_points += 1
            signal_confidence += 0.05
        
        # Determine final order book signal
        ob_signal = "neutral"
        if ob_bull_points > ob_bear_points:
            ob_signal = "bullish"
        elif ob_bear_points > ob_bull_points:
            ob_signal = "bearish"
            
        # Store signal and confidence
        self.orderbook_signal = ob_signal
        self.orderbook_confidence = min(1.0, signal_confidence)
        
        # Add to indicators
        indicators["orderbook_signal"] = ob_signal
        indicators["orderbook_confidence"] = self.orderbook_confidence
        
        logger.info(f"Order book signal: {ob_signal} (confidence: {self.orderbook_confidence:.2f})")
    
    def _combine_signals(self, indicators: Dict[str, Any]) -> None:
        """
        Combine on-chain and order book signals to generate the final trading signal.
        
        The combination weights the signals based on configured parameters and signal confidence.
        
        Args:
            indicators: Indicators dictionary to update with combined signal
        """
        # Check if we have both signals
        if not hasattr(self, "onchain_signal") or not hasattr(self, "orderbook_signal"):
            return
        
        # Get signal weights from parameters
        onchain_weight = self.parameters["onchain_signal_weight"]
        orderbook_weight = self.parameters["orderbook_signal_weight"]
        
        # Adjust weights based on signal confidence
        if self.onchain_confidence < self.parameters["min_onchain_confidence"]:
            # Reduce on-chain weight if confidence is low
            adj_onchain_weight = onchain_weight * (self.onchain_confidence / self.parameters["min_onchain_confidence"])
            adj_orderbook_weight = orderbook_weight + (onchain_weight - adj_onchain_weight)
        elif self.orderbook_confidence < self.parameters["min_orderbook_confidence"]:
            # Reduce order book weight if confidence is low
            adj_orderbook_weight = orderbook_weight * (self.orderbook_confidence / self.parameters["min_orderbook_confidence"])
            adj_onchain_weight = onchain_weight + (orderbook_weight - adj_orderbook_weight)
        else:
            # Use configured weights
            adj_onchain_weight = onchain_weight
            adj_orderbook_weight = orderbook_weight
        
        # Normalize weights
        total_weight = adj_onchain_weight + adj_orderbook_weight
        if total_weight > 0:
            adj_onchain_weight /= total_weight
            adj_orderbook_weight /= total_weight
        
        # Calculate bull and bear points using weighted scoring
        bull_score = 0
        bear_score = 0
        
        # Add on-chain contribution
        if self.onchain_signal == "bullish":
            bull_score += adj_onchain_weight
        elif self.onchain_signal == "bearish":
            bear_score += adj_onchain_weight
        
        # Add order book contribution
        if self.orderbook_signal == "bullish":
            bull_score += adj_orderbook_weight
        elif self.orderbook_signal == "bearish":
            bear_score += adj_orderbook_weight
        
        # Calculate signal strength as the difference between bull and bear scores
        signal_strength = abs(bull_score - bear_score)
        
        # Determine combined signal
        signal = "neutral"
        if bull_score > bear_score and signal_strength >= self.parameters["combined_signal_threshold"]:
            signal = "bullish"
        elif bear_score > bull_score and signal_strength >= self.parameters["combined_signal_threshold"]:
            signal = "bearish"
        
        # Store combined signal
        self.combined_signal = signal
        self.combined_signal_strength = signal_strength
        
        # Add to indicators
        indicators["combined_signal"] = signal
        indicators["combined_signal_strength"] = signal_strength
        indicators["adj_onchain_weight"] = adj_onchain_weight
        indicators["adj_orderbook_weight"] = adj_orderbook_weight
        
        logger.info(f"Combined signal: {signal} (strength: {signal_strength:.2f}, "
                  f"weights: onchain={adj_onchain_weight:.2f}, orderbook={adj_orderbook_weight:.2f})")
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on combined on-chain and order book analysis.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "long_entry": False,
            "short_entry": False,
            "long_exit": False,
            "short_exit": False,
            "signal_strength": 0.0,
            "stop_loss": None,
            "take_profit": None,
        }
        
        if data.empty or not indicators or "combined_signal" not in indicators:
            return signals
        
        # Extract key indicators
        combined_signal = indicators["combined_signal"]
        signal_strength = indicators["combined_signal_strength"]
        current_price = data["close"].iloc[-1]
        atr = indicators.get("atr", None)
        
        # Check if signal strength meets threshold
        if signal_strength < self.parameters["combined_signal_threshold"]:
            logger.info(f"Signal strength {signal_strength:.2f} below threshold "
                      f"{self.parameters['combined_signal_threshold']}, no trade signal generated")
            return signals
        
        # Generate entry signals
        if combined_signal == "bullish" and self.position_count < self.parameters["max_open_positions"]:
            signals["long_entry"] = True
            signals["signal_strength"] = signal_strength
            
            # Log the components that contributed to this signal
            components = []
            if self.onchain_signal == "bullish":
                components.append(f"on-chain ({self.onchain_confidence:.2f})")
            if self.orderbook_signal == "bullish":
                components.append(f"order book ({self.orderbook_confidence:.2f})")
                
            logger.info(f"LONG entry signal generated with strength {signal_strength:.2f}, "
                      f"based on: {', '.join(components)}")
            
            # Generate stop loss and take profit levels
            if atr is not None and len(atr) > 0:
                atr_value = atr.iloc[-1]
                signals["stop_loss"] = current_price - (atr_value * self.parameters["stop_atr_multiple"])
                signals["take_profit"] = current_price + (atr_value * self.parameters["stop_atr_multiple"] * 2)
                
                # If enabled, set up ladder entries
                if self.parameters["use_laddered_entries"]:
                    self._setup_ladder_entries("long", current_price, atr_value, signals)
                    
        elif combined_signal == "bearish" and self.position_count < self.parameters["max_open_positions"]:
            signals["short_entry"] = True
            signals["signal_strength"] = signal_strength
            
            # Log the components that contributed to this signal
            components = []
            if self.onchain_signal == "bearish":
                components.append(f"on-chain ({self.onchain_confidence:.2f})")
            if self.orderbook_signal == "bearish":
                components.append(f"order book ({self.orderbook_confidence:.2f})")
                
            logger.info(f"SHORT entry signal generated with strength {signal_strength:.2f}, "
                      f"based on: {', '.join(components)}")
            
            # Generate stop loss and take profit levels
            if atr is not None and len(atr) > 0:
                atr_value = atr.iloc[-1]
                signals["stop_loss"] = current_price + (atr_value * self.parameters["stop_atr_multiple"])
                signals["take_profit"] = current_price - (atr_value * self.parameters["stop_atr_multiple"] * 2)
                
                # If enabled, set up ladder entries
                if self.parameters["use_laddered_entries"]:
                    self._setup_ladder_entries("short", current_price, atr_value, signals)
        
        # Generate exit signals
        for position in self.positions:
            # Exit long positions on bearish signal
            if position.direction == "long" and combined_signal == "bearish":
                signals["long_exit"] = True
                logger.info(f"Exit LONG signal: Combined signal turned bearish with strength {signal_strength:.2f}")
                
            # Exit short positions on bullish signal
            elif position.direction == "short" and combined_signal == "bullish":
                signals["short_exit"] = True
                logger.info(f"Exit SHORT signal: Combined signal turned bullish with strength {signal_strength:.2f}")
                
            # Check on-chain specific exit conditions
            self._check_onchain_exit_conditions(position, indicators, signals)
        
        return signals
    
    def _setup_ladder_entries(self, direction: str, current_price: float, atr_value: float, signals: Dict[str, Any]) -> None:
        """
        Set up ladder (scaled) entries for a position based on current price and volatility.
        
        Args:
            direction: Trading direction ('long' or 'short')
            current_price: Current market price
            atr_value: Current ATR value for volatility-based spacing
            signals: Signals dictionary to update with ladder information
        """
        ladder_levels = self.parameters["ladder_levels"]
        ladder_price_step = self.parameters["ladder_price_step"]
        
        # Calculate price steps based on ATR and parameter
        price_step = current_price * ladder_price_step
        
        # Calculate ladder entry prices
        ladder_prices = []
        if direction == "long":
            # For long positions, laddered entries are below current price
            for i in range(ladder_levels):
                ladder_prices.append(current_price * (1 - (i + 1) * ladder_price_step))
        else:  # short
            # For short positions, laddered entries are above current price
            for i in range(ladder_levels):
                ladder_prices.append(current_price * (1 + (i + 1) * ladder_price_step))
        
        # Add to signals
        signals["ladder_entries"] = ladder_prices
        signals["ladder_quantities"] = [100 / ladder_levels] * ladder_levels  # Equal size for each level
        
        logger.info(f"Set up {ladder_levels} ladder entries for {direction} position at prices: "
                  f"{[f'{p:.2f}' for p in ladder_prices]}")
    
    def _check_onchain_exit_conditions(self, position, indicators: Dict[str, Any], signals: Dict[str, Any]) -> None:
        """
        Check for exit conditions based on specific on-chain metrics.
        
        Args:
            position: Current position to check
            indicators: Current indicators dictionary
            signals: Signals dictionary to update with exit signals
        """
        # Check for specific on-chain warning signs
        
        # 1. Exchange flow spikes can indicate imminent volatility
        if "exchange_flow_ratio" in indicators:
            flow_ratio = indicators["exchange_flow_ratio"]
            flow_threshold = self.parameters["exchange_flow_threshold"] * 2  # Higher threshold for exits
            
            if position.direction == "long" and flow_ratio > flow_threshold:
                # Large inflows to exchanges often precede selling pressure
                signals["long_exit"] = True
                logger.info(f"Exit LONG signal: Large exchange inflows detected (ratio: {flow_ratio:.3f})")
                
            elif position.direction == "short" and flow_ratio < -flow_threshold:
                # Large outflows from exchanges often indicate accumulation
                signals["short_exit"] = True
                logger.info(f"Exit SHORT signal: Large exchange outflows detected (ratio: {flow_ratio:.3f})")
        
        # 2. Whale wallet movements
        if "wallet_distribution" in indicators:
            distribution = indicators["wallet_distribution"]
            
            if position.direction == "long" and distribution == "distribution":
                # Whales distributing is bearish
                signals["long_exit"] = True
                logger.info(f"Exit LONG signal: Whale distribution pattern detected")
                
            elif position.direction == "short" and distribution == "accumulation":
                # Whales accumulating is bullish
                signals["short_exit"] = True
                logger.info(f"Exit SHORT signal: Whale accumulation pattern detected")
        
        # 3. Network congestion spikes can predict volatility
        if "network_congestion" in indicators:
            congestion = indicators["network_congestion"]
            if congestion > self.parameters["network_congestion_threshold"] * 1.5:
                # Extreme congestion can lead to price volatility
                if position.direction == "long":
                    signals["long_exit"] = True
                    logger.info(f"Exit LONG signal: Extreme network congestion (value: {congestion:.2f})")
                elif position.direction == "short":
                    signals["short_exit"] = True
                    logger.info(f"Exit SHORT signal: Extreme network congestion (value: {congestion:.2f})")
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters, on-chain metrics, and order book liquidity.
        
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
        risk_percentage = self.parameters["risk_per_trade"]
        risk_amount = account_balance * risk_percentage
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Default position size
        default_position_size = (account_balance * 0.1) / current_price
        
        # Get stop loss distance
        stop_loss = self.signals.get("stop_loss", None)
        if stop_loss is None:
            # Calculate based on ATR if available
            if "atr" in indicators and len(indicators["atr"]) > 0:
                atr_value = indicators["atr"].iloc[-1]
                stop_distance = atr_value * self.parameters["stop_atr_multiple"]
            else:
                # Fallback to fixed percentage
                stop_distance = current_price * 0.02  # 2% default stop
        else:
            stop_distance = abs(current_price - stop_loss)
        
        # Calculate base position size based on risk and stop distance
        if stop_distance > 0:
            base_position_size = risk_amount / stop_distance
        else:
            base_position_size = default_position_size
        
        # Convert to crypto units
        position_size_crypto = base_position_size / current_price
        
        # Adjust based on signal strength
        signal_strength = self.signals.get("signal_strength", 0.5)
        position_size_crypto *= max(0.3, signal_strength)  # Min 30% of base size
        
        # Adjust based on orderbook liquidity if available
        if "ob_liquidity_score" in indicators:
            liquidity_score = indicators["ob_liquidity_score"]
            if liquidity_score < self.parameters["ob_liquidity_threshold"]:
                # Reduce size in illiquid markets to avoid slippage
                liquidity_factor = max(0.5, liquidity_score / self.parameters["ob_liquidity_threshold"])
                position_size_crypto *= liquidity_factor
                logger.info(f"Reduced position size due to low liquidity: factor={liquidity_factor:.2f}")
        
        # Adjust based on network congestion if available
        if "network_congestion" in indicators:
            congestion = indicators["network_congestion"]
            if congestion > self.parameters["network_congestion_threshold"]:
                # Reduce size when network is congested (higher fees, slower confirms)
                congestion_factor = max(0.5, 1 - (congestion - self.parameters["network_congestion_threshold"]) * 2)
                position_size_crypto *= congestion_factor
                logger.info(f"Reduced position size due to network congestion: factor={congestion_factor:.2f}")
        
        # Apply precision appropriate for the asset
        decimals = 8 if self.session.symbol.startswith("BTC") else 6
        position_size_crypto = round(position_size_crypto, decimals)
        
        # Ensure minimum trade size
        min_trade_size = self.session.min_trade_size
        position_size_crypto = max(position_size_crypto, min_trade_size)
        
        logger.info(f"On-chain position size: {position_size_crypto} {self.session.symbol.split('-')[0]} "
                  f"(signal strength: {signal_strength:.2f})")
        
        return position_size_crypto
    
    def _on_orderbook_updated(self, event: Event) -> None:
        """
        Handle orderbook updated events.
        
        For on-chain strategies, we use orderbook data for short-term signals
        to complement the medium/long-term on-chain signals.
        """
        super()._on_orderbook_updated(event)
        
        # Skip if not enabled or not our symbol
        if not self.parameters["enable_orderbook_analysis"] or event.data.get('symbol') != self.session.symbol:
            return
        
        # Only re-process when we have a sufficient update
        if self.is_active and self.orderbook and not self.market_data.empty:
            # Calculate new indicators with latest orderbook
            self.indicators = self.calculate_indicators(self.market_data)
            
            # Generate new signals
            self.signals = self.generate_signals(self.market_data, self.indicators)
            
            # Check for trade opportunities if signals changed
            if self.signals.get("long_entry") or self.signals.get("short_entry") or \
               self.signals.get("long_exit") or self.signals.get("short_exit"):
                self._check_for_trade_opportunities()
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        This is when we update on-chain data and generate primary signals.
        """
        super()._on_timeframe_completed(event)
        
        # Only process if this is our timeframe
        if event.data.get('timeframe') != self.session.timeframe:
            return
        
        # Calculate new indicators with updated market data
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Generate trading signals
        self.signals = self.generate_signals(self.market_data, self.indicators)
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        On-chain analysis tends to work best in trending and accumulation/distribution
        phases, and less well in choppy or highly technical markets.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.80,          # Very good in trending markets (captures macro flows)
            "ranging": 0.50,           # Moderate in ranging markets (orderbook component helps)
            "volatile": 0.70,          # Good in volatile markets (can detect whale movements)
            "calm": 0.60,              # Moderate in calm markets (good for accumulation detection)
            "breakout": 0.75,          # Good during breakouts (often preceded by on-chain signals)
            "high_volume": 0.85,       # Very good during high volume periods
            "low_volume": 0.65,        # Good in low volume (can detect insider movements)
            "high_liquidity": 0.70,    # Good in high liquidity markets
            "low_liquidity": 0.65,     # Good in low liquidity markets
            "accumulation": 0.90,      # Excellent during accumulation phases
            "distribution": 0.90,      # Excellent during distribution phases
        }
        
        return compatibility_map.get(market_regime, 0.70)  # Default compatibility
