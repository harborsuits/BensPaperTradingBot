#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Scalping Strategy

This strategy is designed for very short-term crypto trading, aiming to profit from
small price movements with quick entries and exits. It uses high-frequency technical 
indicators and orderbook analysis to identify micro-trends and liquidity zones.

Key characteristics:
- Very short holding periods (minutes)
- High trade frequency
- Small profit targets (0.5-2%)
- Tight stop losses
- Focus on high-liquidity pairs
- Uses both technical indicators and order book analysis
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoScalpingStrategy",
    market_type="crypto",
    description="High-frequency scalping strategy for crypto markets using technical indicators and orderbook analysis",
    timeframes=["M1", "M5", "M15"],
    parameters={
        # Technical indicators
        "rsi_period": {"type": "int", "default": 7, "min": 3, "max": 14},
        "rsi_overbought": {"type": "float", "default": 70.0, "min": 60.0, "max": 80.0},
        "rsi_oversold": {"type": "float", "default": 30.0, "min": 20.0, "max": 40.0},
        "bb_period": {"type": "int", "default": 20, "min": 10, "max": 30},
        "bb_std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
        "ema_short": {"type": "int", "default": 8, "min": 3, "max": 12},
        "ema_medium": {"type": "int", "default": 13, "min": 8, "max": 21},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        
        # Orderbook parameters
        "use_orderbook": {"type": "bool", "default": True},
        "min_bid_ask_ratio": {"type": "float", "default": 1.2, "min": 1.0, "max": 2.0},
        "price_levels_depth": {"type": "int", "default": 5, "min": 3, "max": 10},
        
        # Trade execution
        "profit_target_pct": {"type": "float", "default": 0.008, "min": 0.003, "max": 0.03},
        "stop_loss_atr_multiplier": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
        "max_trades_per_hour": {"type": "int", "default": 6, "min": 1, "max": 20},
        "min_trade_interval_seconds": {"type": "int", "default": 120, "min": 30, "max": 600},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.005, "min": 0.001, "max": 0.01},
        "max_open_positions": {"type": "int", "default": 3, "min": 1, "max": 5},
    }
)
class CryptoScalpingStrategy(CryptoBaseStrategy):
    """
    A high-frequency scalping strategy for cryptocurrency markets.
    
    This strategy:
    1. Uses multiple short-term technical indicators (RSI, Bollinger Bands, EMA)
    2. Analyzes orderbook data to identify liquidity and potential price movements
    3. Takes quick profits with tight stop losses
    4. Implements trade frequency limitations to avoid overtrading
    5. Has risk management tailored for high-frequency trading
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the crypto scalping strategy.
        
        Args:
            session: Crypto trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Merge default parameters with any provided parameters
        default_params = {
            "rsi_period": 7,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "ema_short": 8,
            "ema_medium": 13,
            "atr_period": 14,
            "use_orderbook": True,
            "min_bid_ask_ratio": 1.2,
            "price_levels_depth": 5,
            "profit_target_pct": 0.008,  # 0.8%
            "stop_loss_atr_multiplier": 1.0,
            "max_trades_per_hour": 6,
            "min_trade_interval_seconds": 120,
            "risk_per_trade": 0.005,
            "max_open_positions": 3
        }
        
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy
        super().__init__(session, data_pipeline, default_params)
        
        # Strategy-specific initialization
        self.last_signal = None
        self.last_trade_time = datetime.utcnow() - timedelta(hours=1)  # Initialize to allow immediate trading
        self.hourly_trade_count = 0
        self.hour_reset_time = datetime.utcnow() + timedelta(hours=1)
        
        logger.info(f"Initialized {self.name} with parameters: {self.parameters}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        if len(data) < max(self.parameters["bb_period"], self.parameters["rsi_period"]) + 10:
            return {}
            
        indicators = {}
        
        # RSI - Relative Strength Index
        close_diff = data["close"].diff()
        gain = close_diff.where(close_diff > 0, 0).rolling(window=self.parameters["rsi_period"]).mean()
        loss = -close_diff.where(close_diff < 0, 0).rolling(window=self.parameters["rsi_period"]).mean()
        
        if loss.iloc[-1] != 0:
            rs = gain.iloc[-1] / loss.iloc[-1]
            indicators["rsi"] = 100 - (100 / (1 + rs))
        else:
            indicators["rsi"] = 100.0
        
        # Bollinger Bands
        middle_band = data["close"].rolling(window=self.parameters["bb_period"]).mean()
        std_dev = data["close"].rolling(window=self.parameters["bb_period"]).std()
        
        indicators["bb_middle"] = middle_band.iloc[-1]
        indicators["bb_upper"] = middle_band.iloc[-1] + (self.parameters["bb_std_dev"] * std_dev.iloc[-1])
        indicators["bb_lower"] = middle_band.iloc[-1] - (self.parameters["bb_std_dev"] * std_dev.iloc[-1])
        
        # Price position relative to Bollinger Bands
        current_price = data["close"].iloc[-1]
        indicators["price_vs_bb"] = (current_price - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
        
        # EMAs
        indicators["ema_short"] = data["close"].ewm(span=self.parameters["ema_short"], adjust=False).mean().iloc[-1]
        indicators["ema_medium"] = data["close"].ewm(span=self.parameters["ema_medium"], adjust=False).mean().iloc[-1]
        
        # ATR for volatility measurement and stop loss calculation
        indicators["atr"] = self._calculate_atr(data, self.parameters["atr_period"])[-1]
        
        # Calculate recent volume profile
        volume_ma_5 = data["volume"].rolling(window=5).mean().iloc[-1]
        volume_ma_20 = data["volume"].rolling(window=20).mean().iloc[-1]
        indicators["volume_ratio"] = volume_ma_5 / volume_ma_20 if volume_ma_20 > 0 else 1.0
        
        # Orderbook indicators (if available)
        if self.parameters["use_orderbook"] and self.orderbook:
            indicators.update(self._calculate_orderbook_indicators())
        
        # Micro-trend detection
        indicators["micro_trend"] = self._detect_micro_trend(data, indicators)
        
        return indicators
    
    def _calculate_orderbook_indicators(self) -> Dict[str, Any]:
        """Calculate indicators based on orderbook data."""
        orderbook_indicators = {}
        
        # Check if we have valid orderbook data
        if not self.orderbook or "bids" not in self.orderbook or "asks" not in self.orderbook:
            return orderbook_indicators
        
        # Process only specified depth of the orderbook
        depth = self.parameters["price_levels_depth"]
        bids = self.orderbook["bids"][:depth] if len(self.orderbook["bids"]) > 0 else []
        asks = self.orderbook["asks"][:depth] if len(self.orderbook["asks"]) > 0 else []
        
        if not bids or not asks:
            return orderbook_indicators
        
        # Calculate total volume at bid and ask
        bid_volume = sum(qty for price, qty in bids)
        ask_volume = sum(qty for price, qty in asks)
        
        # Calculate orderbook imbalance
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            orderbook_indicators["bid_ask_ratio"] = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            orderbook_indicators["bid_volume_pct"] = (bid_volume / total_volume) * 100
            orderbook_indicators["ask_volume_pct"] = (ask_volume / total_volume) * 100
            
            # Imbalance index (-1.0 to 1.0, positive means more buying pressure)
            orderbook_indicators["imbalance"] = (bid_volume - ask_volume) / total_volume
        
        # Calculate spread
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            orderbook_indicators["spread"] = (best_ask - best_bid) / best_bid
            orderbook_indicators["mid_price"] = (best_bid + best_ask) / 2
        
        # Calculate price impact to buy/sell significant amount
        price_impact_buy = 0
        price_impact_sell = 0
        
        if asks:
            # Calculate price impact of buying 1% of visible ask liquidity
            target_qty = total_volume * 0.01
            qty_sum = 0
            for price, qty in asks:
                qty_sum += qty
                if qty_sum >= target_qty:
                    price_impact_buy = (price - best_ask) / best_ask
                    break
        
        if bids:
            # Calculate price impact of selling 1% of visible bid liquidity
            target_qty = total_volume * 0.01
            qty_sum = 0
            for price, qty in bids:
                qty_sum += qty
                if qty_sum >= target_qty:
                    price_impact_sell = (best_bid - price) / best_bid
                    break
        
        orderbook_indicators["price_impact_buy"] = price_impact_buy
        orderbook_indicators["price_impact_sell"] = price_impact_sell
        
        # Detect liquidity walls
        if len(bids) > 1 and len(asks) > 1:
            # Check for unusually large orders
            avg_bid_size = sum(qty for _, qty in bids) / len(bids)
            avg_ask_size = sum(qty for _, qty in asks) / len(asks)
            
            # Find largest orders relative to average
            max_bid_ratio = max((qty / avg_bid_size) for _, qty in bids)
            max_ask_ratio = max((qty / avg_ask_size) for _, qty in asks)
            
            orderbook_indicators["liquidity_wall_bid"] = max_bid_ratio > 3.0
            orderbook_indicators["liquidity_wall_ask"] = max_ask_ratio > 3.0
        
        return orderbook_indicators
    
    def _detect_micro_trend(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """
        Detect micro-trends based on technical indicators and recent price action.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        # Short-term price action (last 3-5 candles)
        last_n = 5
        if len(data) < last_n:
            return "neutral"
        
        recent_data = data.iloc[-last_n:]
        price_direction = 1 if recent_data["close"].iloc[-1] > recent_data["close"].iloc[0] else -1
        
        # Count bullish vs bearish candles
        bullish_candles = sum(1 for i in range(len(recent_data)) if recent_data["close"].iloc[i] > recent_data["open"].iloc[i])
        bearish_candles = len(recent_data) - bullish_candles
        
        # Technical signals
        rsi_signal = 0
        if "rsi" in indicators:
            if indicators["rsi"] < self.parameters["rsi_oversold"]:
                rsi_signal = 1  # Oversold, potentially bullish
            elif indicators["rsi"] > self.parameters["rsi_overbought"]:
                rsi_signal = -1  # Overbought, potentially bearish
        
        bb_signal = 0
        if all(k in indicators for k in ["price_vs_bb", "bb_lower", "bb_upper"]):
            if indicators["price_vs_bb"] < 0.2:  # Price near lower band
                bb_signal = 1
            elif indicators["price_vs_bb"] > 0.8:  # Price near upper band
                bb_signal = -1
        
        ema_signal = 0
        if all(k in indicators for k in ["ema_short", "ema_medium"]):
            if indicators["ema_short"] > indicators["ema_medium"]:
                ema_signal = 1
            else:
                ema_signal = -1
        
        # Orderbook signals
        ob_signal = 0
        if "imbalance" in indicators:
            if indicators["imbalance"] > 0.2:  # Strong buying pressure
                ob_signal = 1
            elif indicators["imbalance"] < -0.2:  # Strong selling pressure
                ob_signal = -1
        
        # Combine signals with weights
        signal_sum = (
            (price_direction * 0.3) + 
            ((bullish_candles - bearish_candles) / len(recent_data) * 0.15) +
            (rsi_signal * 0.15) + 
            (bb_signal * 0.15) + 
            (ema_signal * 0.15) + 
            (ob_signal * 0.1)
        )
        
        # Determine trend direction
        if signal_sum > 0.15:
            return "bullish"
        elif signal_sum < -0.15:
            return "bearish"
        else:
            return "neutral"
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on calculated indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        if not indicators or len(data) < 2:
            return {"entry": {}, "exit": {}, "position_adjust": {}}
            
        signals = {
            "entry": {"long": 0.0, "short": 0.0},
            "exit": {"long": 0.0, "short": 0.0},
            "position_adjust": {"long": 0.0, "short": 0.0}
        }
        
        # Check trade frequency limits
        current_time = datetime.utcnow()
        
        # Reset hourly counter if needed
        if current_time > self.hour_reset_time:
            self.hourly_trade_count = 0
            self.hour_reset_time = current_time + timedelta(hours=1)
        
        # Check if we've reached the maximum trades per hour
        if self.hourly_trade_count >= self.parameters["max_trades_per_hour"]:
            logger.debug(f"Maximum trades per hour reached: {self.hourly_trade_count}")
            return signals
        
        # Check if minimum time between trades has elapsed
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds()
        if time_since_last_trade < self.parameters["min_trade_interval_seconds"]:
            logger.debug(f"Minimum trade interval not reached. Time since last trade: {time_since_last_trade}s")
            return signals
        
        # Check if we've reached the maximum open positions
        if len(self.positions) >= self.parameters["max_open_positions"]:
            logger.debug(f"Maximum open positions reached: {len(self.positions)}")
            return signals
        
        # Get the micro-trend
        micro_trend = indicators.get("micro_trend", "neutral")
        
        # Entry signals based on indicators
        if micro_trend == "bullish":
            # RSI conditions for long entry
            rsi_condition = indicators.get("rsi", 50) < 40 and indicators.get("rsi", 50) > 25
            
            # Bollinger Band conditions
            bb_condition = (data["close"].iloc[-1] < indicators.get("bb_middle", 0) and 
                          data["close"].iloc[-1] > indicators.get("bb_lower", 0))
            
            # EMA conditions
            ema_condition = indicators.get("ema_short", 0) > indicators.get("ema_medium", 0)
            
            # Orderbook conditions for long entry
            ob_condition = True
            if self.parameters["use_orderbook"] and "bid_ask_ratio" in indicators:
                ob_condition = indicators["bid_ask_ratio"] > self.parameters["min_bid_ask_ratio"]
            
            # Recent price action
            price_condition = data["close"].iloc[-1] > data["close"].iloc[-2]
            
            # Calculate long entry signal strength
            signal_count = sum([rsi_condition, bb_condition, ema_condition, ob_condition, price_condition])
            long_strength = signal_count / 5.0
            
            signals["entry"]["long"] = long_strength
            
        elif micro_trend == "bearish":
            # RSI conditions for short entry
            rsi_condition = indicators.get("rsi", 50) > 60 and indicators.get("rsi", 50) < 75
            
            # Bollinger Band conditions
            bb_condition = (data["close"].iloc[-1] > indicators.get("bb_middle", 0) and 
                          data["close"].iloc[-1] < indicators.get("bb_upper", 0))
            
            # EMA conditions
            ema_condition = indicators.get("ema_short", 0) < indicators.get("ema_medium", 0)
            
            # Orderbook conditions for short entry
            ob_condition = True
            if self.parameters["use_orderbook"] and "bid_ask_ratio" in indicators:
                ob_condition = indicators["bid_ask_ratio"] < (1 / self.parameters["min_bid_ask_ratio"])
            
            # Recent price action
            price_condition = data["close"].iloc[-1] < data["close"].iloc[-2]
            
            # Calculate short entry signal strength
            signal_count = sum([rsi_condition, bb_condition, ema_condition, ob_condition, price_condition])
            short_strength = signal_count / 5.0
            
            signals["entry"]["short"] = short_strength
        
        # Exit signals for existing positions
        for position in self.positions:
            if position.direction == "long" and micro_trend == "bearish":
                signals["exit"]["long"] = 0.8
            elif position.direction == "short" and micro_trend == "bullish":
                signals["exit"]["short"] = 0.8
        
        # Save the signal for reference
        self.last_signal = signals
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters and ATR.
        
        For scalping, we typically use smaller position sizes than trend strategies,
        but with tighter stop losses to maintain the same risk-per-trade.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        if not indicators or "atr" not in indicators or len(data) < 1:
            return 0.0
            
        # Get the current ATR value
        atr = indicators["atr"]
        
        # Account equity would come from the broker/portfolio in a real system
        # Using a placeholder value here
        account_equity = 10000.0  # Example value
        
        # Calculate the risk amount in account currency
        risk_amount = account_equity * self.parameters["risk_per_trade"]
        
        # For scalping, we use tighter stops
        stop_loss_distance = atr * self.parameters["stop_loss_atr_multiplier"]
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Calculate position size
        if stop_loss_distance > 0 and current_price > 0:
            # Position size in crypto units
            position_size = risk_amount / stop_loss_distance
            
            # Convert to BTC or ETH equivalent for position sizing
            position_size_in_crypto = position_size / current_price
            
            # Ensure position meets minimum trade size
            if position_size_in_crypto < self.session.min_trade_size:
                logger.debug(f"Calculated position size {position_size_in_crypto} below minimum {self.session.min_trade_size}")
                return 0.0
            
            # Round to appropriate precision for the asset
            # Typically 8 decimals for BTC, 6 for ETH, etc.
            decimals = 8 if self.session.symbol.startswith("BTC") else 6
            position_size_in_crypto = round(position_size_in_crypto, decimals)
            
            # Track this trade for frequency limitations
            self.last_trade_time = datetime.utcnow()
            self.hourly_trade_count += 1
            
            return position_size_in_crypto
        
        return 0.0
    
    def _on_orderbook_updated(self, event: Event) -> None:
        """
        Override the base method to be more responsive to orderbook changes.
        
        For scalping, orderbook changes can trigger immediate trade decisions.
        """
        if event.data.get('symbol') != self.session.symbol:
            return
            
        self.orderbook = event.data.get('orderbook', {})
        
        # For scalping, we may want to react to significant orderbook changes
        # even without new price data
        if self.is_active and self.parameters["use_orderbook"]:
            # Only proceed if we have all necessary data
            if not self.market_data.empty and self.orderbook:
                # Calculate orderbook indicators
                ob_indicators = self._calculate_orderbook_indicators()
                
                # Check for significant imbalances that might warrant immediate action
                if "imbalance" in ob_indicators:
                    imbalance = ob_indicators["imbalance"]
                    
                    # Extreme imbalance might trigger immediate position adjustment
                    if abs(imbalance) > 0.5:  # Extreme imbalance (>50% in one direction)
                        logger.info(f"Detected extreme orderbook imbalance: {imbalance:.2f}")
                        
                        # Check existing positions
                        for position in self.positions:
                            # If imbalance is against our position direction, consider exiting
                            if (position.direction == "long" and imbalance < -0.5) or \
                               (position.direction == "short" and imbalance > 0.5):
                                logger.info(f"Orderbook imbalance suggests closing {position.direction} position")
                                self._close_position(position.id)
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Scalping strategies typically work well in ranging and moderately volatile markets,
        but can struggle in strongly trending or extremely volatile conditions.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "ranging": 0.90,        # Excellent in ranging markets
            "volatile": 0.70,       # Good in moderately volatile markets
            "trending": 0.40,       # Poor in strongly trending markets
            "calm": 0.60,           # Moderate in calm markets
            "breakout": 0.30,       # Poor during breakouts
            "high_volume": 0.85,    # Great during high volume periods
            "low_volume": 0.30,     # Poor during low volume periods
            "high_liquidity": 0.95, # Excellent in high liquidity markets
            "low_liquidity": 0.20,  # Very poor in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.50)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.values
