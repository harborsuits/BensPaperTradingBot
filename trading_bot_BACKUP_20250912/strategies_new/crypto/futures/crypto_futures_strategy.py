#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Futures Trading Strategy

This strategy is specialized for cryptocurrency perpetual futures contracts.
It accounts for funding rates, liquidation risk, and optimal leverage utilization
while implementing trend-based entries with aggressive risk management.

Key characteristics:
- Uses funding rate analytics
- Employs careful leverage management
- Implements liquidation risk controls
- Targets medium-term trends
- Includes hedging capabilities
"""

import logging
from typing import Dict, Any, List
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
    name="CryptoFuturesStrategy",
    market_type="crypto",
    description="Strategy for crypto perpetual futures with funding rate optimization and leverage management",
    timeframes=["H1", "H4", "D1"],
    parameters={
        # Technical indicators
        "ema_short": {"type": "int", "default": 21, "min": 10, "max": 50},
        "ema_medium": {"type": "int", "default": 55, "min": 30, "max": 100},
        "ema_long": {"type": "int", "default": 200, "min": 100, "max": 300},
        "rsi_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        
        # Funding rate parameters
        "funding_rate_threshold": {"type": "float", "default": 0.0001, "min": 0.00001, "max": 0.001},
        "counter_funding_rate_enabled": {"type": "bool", "default": True},
        
        # Leverage parameters
        "base_leverage": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
        "max_leverage": {"type": "float", "default": 5.0, "min": 1.0, "max": 10.0},
        "dynamic_leverage": {"type": "bool", "default": True},
        
        # Risk parameters
        "max_position_value_pct": {"type": "float", "default": 0.2, "min": 0.05, "max": 0.5},
        "liquidation_buffer_pct": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.5},
        "stop_loss_atr_multi": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
        "take_profit_atr_multi": {"type": "float", "default": 4.0, "min": 2.0, "max": 8.0},
        
        # Strategy specific
        "trend_confirmation_bars": {"type": "int", "default": 3, "min": 1, "max": 5},
        "use_hedging": {"type": "bool", "default": False},
    }
)
class CryptoFuturesStrategy(CryptoBaseStrategy):
    """
    A strategy designed specifically for crypto perpetual futures contracts.
    
    This strategy:
    1. Monitors and utilizes funding rates for optimal entry timing
    2. Implements dynamic leverage based on volatility and trend strength
    3. Maintains strict risk controls to prevent liquidation
    4. Targets medium to long-term trends in crypto futures markets
    5. Can employ hedging strategies across correlated assets
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto futures trading strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Ensure we're using a perpetual futures session
        if not session.is_perpetual:
            logger.warning("CryptoFuturesStrategy is designed for perpetual futures. Setting is_perpetual=True")
            session.is_perpetual = True
        
        # Futures-specific state
        self.funding_rate_history = []
        self.last_funding_time = None
        self.current_leverage = self.parameters["base_leverage"]
        self.liquidation_prices = {}  # For each position
        self.hedged_positions = {}    # For hedging pairs
        
        # Risk monitoring
        self.margin_utilization = 0.0
        self.liquidation_risk_level = "low"
        
        logger.info(f"Initialized crypto futures strategy for {self.session.symbol} with base leverage {self.current_leverage}x")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for futures trading strategy."""
        if data.empty or len(data) < self.parameters["ema_long"]:
            return {}
        
        indicators = {}
        
        # Moving averages for trend identification
        indicators["ema_short"] = data["close"].ewm(span=self.parameters["ema_short"], adjust=False).mean()
        indicators["ema_medium"] = data["close"].ewm(span=self.parameters["ema_medium"], adjust=False).mean()
        indicators["ema_long"] = data["close"].ewm(span=self.parameters["ema_long"], adjust=False).mean()
        
        # RSI for momentum
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.parameters["rsi_period"]).mean()
        avg_loss = loss.rolling(window=self.parameters["rsi_period"]).mean()
        rs = avg_gain / avg_loss
        indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR for volatility measurement and position sizing
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = true_range.rolling(window=self.parameters["atr_period"]).mean()
        
        # Trend strength and direction
        ema_short_slope = indicators["ema_short"].diff(self.parameters["trend_confirmation_bars"])
        ema_medium_slope = indicators["ema_medium"].diff(self.parameters["trend_confirmation_bars"])
        indicators["trend_strength"] = (
            ema_short_slope / indicators["ema_short"] + 
            ema_medium_slope / indicators["ema_medium"]
        ) * 100  # Normalize to percentage
        
        indicators["trend_direction"] = np.sign(indicators["trend_strength"])
        
        # Funding rate analysis
        if self.funding_rate_history:
            # Calculate average funding rate
            recent_rates = [rate for _, rate in self.funding_rate_history[-10:]]
            indicators["avg_funding_rate"] = sum(recent_rates) / len(recent_rates) if recent_rates else 0
            
            # Funding rate direction (positive means longs pay shorts)
            indicators["funding_rate_direction"] = "positive" if indicators["avg_funding_rate"] > 0 else "negative"
            
            # Funding rate opportunity
            threshold = self.parameters["funding_rate_threshold"]
            if abs(indicators["avg_funding_rate"]) > threshold:
                # If counter-funding trading is enabled, go against funding direction
                if self.parameters["counter_funding_rate_enabled"]:
                    indicators["funding_opportunity"] = "short" if indicators["avg_funding_rate"] > 0 else "long"
                else:
                    # Otherwise, use funding as confirmation with trend
                    indicators["funding_opportunity"] = None
            else:
                indicators["funding_opportunity"] = None
        
        # Calculate optimal leverage based on volatility
        if self.parameters["dynamic_leverage"]:
            # Lower leverage during high volatility, higher during low volatility
            volatility_factor = indicators["atr"].iloc[-1] / indicators["atr"].mean() if len(indicators["atr"]) > 0 else 1
            base_leverage = self.parameters["base_leverage"]
            max_leverage = self.parameters["max_leverage"]
            
            # Inverse relationship between volatility and leverage
            indicators["optimal_leverage"] = base_leverage / max(0.5, volatility_factor)
            
            # Cap at maximum allowed leverage
            indicators["optimal_leverage"] = min(indicators["optimal_leverage"], max_leverage)
        else:
            indicators["optimal_leverage"] = self.parameters["base_leverage"]
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals for futures strategy."""
        signals = {
            "long_entry": False,
            "short_entry": False,
            "long_exit": False,
            "short_exit": False,
            "adjust_leverage": False,
            "new_leverage": None,
            "funding_signal": None,
        }
        
        if not indicators or data.empty:
            return signals
        
        # Current price and indicator values
        current_price = data["close"].iloc[-1]
        
        # Trend detection
        trend_direction = indicators.get("trend_direction", [0])
        if isinstance(trend_direction, pd.Series):
            trend_direction = trend_direction.iloc[-1]
            
        trend_strength = indicators.get("trend_strength", [0])
        if isinstance(trend_strength, pd.Series):
            trend_strength = trend_strength.iloc[-1]
        
        # EMA setup
        ema_short = indicators["ema_short"].iloc[-1]
        ema_medium = indicators["ema_medium"].iloc[-1]
        ema_long = indicators["ema_long"].iloc[-1]
        
        # RSI for confirmation
        rsi = indicators["rsi"].iloc[-1]
        
        # Funding rate opportunity
        funding_opportunity = indicators.get("funding_opportunity")
        signals["funding_signal"] = funding_opportunity
        
        # Strong trend setup (short-term MA crossed medium-term MA in direction of long-term trend)
        strong_bullish = (ema_short > ema_medium > ema_long) and (trend_direction > 0) and (abs(trend_strength) > 1)
        strong_bearish = (ema_short < ema_medium < ema_long) and (trend_direction < 0) and (abs(trend_strength) > 1)
        
        # Leverage adjustment signal
        optimal_leverage = indicators.get("optimal_leverage")
        if optimal_leverage is not None and abs(optimal_leverage - self.current_leverage) > 0.5:
            signals["adjust_leverage"] = True
            signals["new_leverage"] = optimal_leverage
            logger.info(f"Adjusting leverage from {self.current_leverage}x to {optimal_leverage:.1f}x based on market conditions")
        
        # Entry signals
        if strong_bullish:
            # Only use funding rate if it aligns or we're ignoring it
            if funding_opportunity is None or funding_opportunity == "long":
                signals["long_entry"] = True
                logger.info(f"Futures LONG signal for {self.session.symbol} - Strong bullish trend")
        
        if strong_bearish:
            # Only use funding rate if it aligns or we're ignoring it
            if funding_opportunity is None or funding_opportunity == "short":
                signals["short_entry"] = True
                logger.info(f"Futures SHORT signal for {self.session.symbol} - Strong bearish trend")
        
        # Exit signals - more conservative for futures due to leverage
        for position in self.positions:
            # Calculate current unrealized PnL
            if position.direction == "long":
                unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
                
                # Exit conditions for long positions
                if (ema_short < ema_medium) or (unrealized_pnl_pct < -0.1):
                    signals["long_exit"] = True
                    logger.info(f"Exit LONG signal - Trend reversal or stop loss")
                
            elif position.direction == "short":
                unrealized_pnl_pct = (position.entry_price - current_price) / position.entry_price
                
                # Exit conditions for short positions
                if (ema_short > ema_medium) or (unrealized_pnl_pct < -0.1):
                    signals["short_exit"] = True
                    logger.info(f"Exit SHORT signal - Trend reversal or stop loss")
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """Calculate position size for futures trading with leverage considerations."""
        if data.empty or not indicators:
            return 0.0
        
        # Account balance (in base currency)
        account_balance = 10000.0  # Mock value, would come from exchange API
        
        # Maximum position value as percentage of account
        max_position_value_pct = self.parameters["max_position_value_pct"]
        max_position_value = account_balance * max_position_value_pct
        
        # Current leverage setting
        leverage = self.current_leverage
        
        # At 5x leverage, a 20% position value is actually 4% of account as collateral
        collateral_needed = max_position_value / leverage
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Calculate base position size without considering liquidation risk
        position_size_base = max_position_value / current_price
        
        # Now adjust for liquidation risk using ATR
        if "atr" in indicators:
            atr = indicators["atr"].iloc[-1]
            
            # Calculate potential price movement
            price_risk = atr * self.parameters["stop_loss_atr_multi"]
            
            # Liquidation price distance as a percentage of price
            liquidation_distance_pct = 1 / leverage  # Simplified: at 5x leverage, a 20% move liquidates
            
            # Add buffer to avoid liquidation
            buffer = self.parameters["liquidation_buffer_pct"]
            safe_distance_pct = liquidation_distance_pct * (1 + buffer)
            
            # If potential price movement is too close to liquidation, reduce position size
            if price_risk / current_price > safe_distance_pct:
                risk_ratio = (price_risk / current_price) / safe_distance_pct
                position_size_base = position_size_base / risk_ratio
                logger.info(f"Reduced position size due to liquidation risk - ratio: {risk_ratio:.2f}")
        
        # Ensure minimum trade size
        min_trade_size = self.session.min_trade_size
        position_size_crypto = max(position_size_base, min_trade_size)
        
        # Round to appropriate precision
        decimals = 8 if self.session.symbol.startswith("BTC") else 6
        position_size_crypto = round(position_size_crypto, decimals)
        
        # Store potential liquidation price for risk management
        if direction == "long":
            liquidation_price = current_price * (1 - (1 / leverage))
        else:  # short
            liquidation_price = current_price * (1 + (1 / leverage))
        
        # Track this for position monitoring
        self.liquidation_prices[f"{direction}_{datetime.utcnow()}"] = liquidation_price
        
        logger.info(f"Calculated futures position size: {position_size_crypto} {self.session.symbol.split('-')[0]} at {leverage}x leverage")
        logger.info(f"Estimated liquidation price: {liquidation_price:.2f}")
        
        return position_size_crypto
    
    def _on_funding_rate_updated(self, event: Event) -> None:
        """Handle funding rate updated events for perpetual futures."""
        super()._on_funding_rate_updated(event)
        
        # Extract funding data
        symbol = event.data.get('symbol')
        funding_rate = event.data.get('funding_rate', 0.0)
        funding_time = event.data.get('timestamp', datetime.utcnow())
        
        if symbol == self.session.symbol:
            # Update funding rate in session
            self.session.update_funding_rate(funding_rate)
            
            # Store funding rate history
            self.funding_rate_history.append((funding_time, funding_rate))
            self.last_funding_time = funding_time
            
            # Cap history length
            if len(self.funding_rate_history) > 100:
                self.funding_rate_history = self.funding_rate_history[-100:]
            
            # Log funding rate update
            logger.info(f"Funding rate updated for {symbol}: {funding_rate:.6f}")
            
            # Check if we need to adjust positions based on funding rate
            if abs(funding_rate) > self.parameters["funding_rate_threshold"]:
                # If we're paying funding, consider closing or hedging
                if (funding_rate > 0 and self._has_long_position()) or \
                   (funding_rate < 0 and self._has_short_position()):
                    logger.info(f"Paying funding rate {funding_rate:.6f} - evaluating position adjustments")
                    self._adjust_for_funding_rate(funding_rate, funding_time)
    
    def _has_long_position(self) -> bool:
        """Check if we have an open long position."""
        return any(p.direction == "long" for p in self.positions)
    
    def _has_short_position(self) -> bool:
        """Check if we have an open short position."""
        return any(p.direction == "short" for p in self.positions)
    
    def regime_compatibility(self, market_regime: str) -> float:
        """Calculate how compatible this strategy is with the current market regime."""
        compatibility_map = {
            "trending": 0.90,        # Excellent in trending markets
            "ranging": 0.60,         # Moderate in ranging markets
            "volatile": 0.75,        # Good in volatile markets (with adjusted leverage)
            "calm": 0.65,            # Moderate in calm markets
            "breakout": 0.80,        # Very good during breakouts
            "high_volume": 0.85,     # Very good during high volume
            "low_volume": 0.50,      # Moderate during low volume
            "high_liquidity": 0.90,  # Excellent in high liquidity markets
            "low_liquidity": 0.40,   # Poor in low liquidity markets (higher slippage risk)
        }
        
        return compatibility_map.get(market_regime, 0.65)  # Default compatibility
