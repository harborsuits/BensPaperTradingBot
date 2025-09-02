"""
Forex Swing Trading Strategy

This strategy identifies and trades medium-term price swings in forex markets,
targeting movements that typically last from several days to a couple of weeks.
It combines trend identification, momentum confirmation, and strategic entry timing
to capture significant price moves while managing risk effectively.

Features:
- Multi-timeframe analysis for trend identification
- Support and resistance zone detection
- Multiple technical confirmation signals
- Pullback and breakout entry methods
- Risk-adjusted position sizing
- Trailing stop management
- Fundamental and sentiment filters
"""

import logging
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set

from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.models.signal import Signal


@register_strategy(
    asset_class="forex",
    strategy_type="swing",
    name="ForexSwingTrading",
    description="Medium-term swing trading for forex using multi-timeframe trend and momentum analysis",
    parameters={
        "default": {
            # Timeframe parameters
            "primary_timeframe": "4h",
            "trend_timeframe": "1d",
            "entry_timeframe": "1h",
            "minimum_bars": 100,  # Minimum bars needed for analysis
            
            # Trend detection parameters
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_very_slow": 200,
            "adx_period": 14,
            "adx_threshold": 25,  # Minimum ADX for trend confirmation
            
            # Swing detection parameters
            "swing_lookback_periods": 10,
            "swing_threshold_pips": 30,  # Minimum swing size to consider
            "swing_duration_min": 5,  # Minimum bars for a valid swing
            "swing_duration_max": 30,  # Maximum bars for a valid swing
            
            # Momentum parameters
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            
            # Support/Resistance parameters
            "sr_lookback_periods": 100,
            "sr_min_touches": 2,
            "sr_zone_pips": 15,  # Zone width around SR levels
            
            # Entry parameters
            "entry_methods": ["pullback", "breakout"],  # Valid methods: pullback, breakout, momentum
            "pullback_threshold": 0.382,  # Fib retracement level for pullback entries
            "breakout_confirmation_bars": 2,  # Bars needed to confirm breakout
            "entry_filter_methods": ["momentum", "volatility"],  # Additional entry filters
            
            # Exit parameters
            "profit_target_atr_multiple": 3.0,  # Take profit as multiple of ATR
            "initial_stop_atr_multiple": 1.5,  # Initial stop loss as multiple of ATR
            "trailing_stop_activation_atr": 2.0,  # Activate trailing after this ATR multiple
            "trailing_stop_atr_multiple": 2.0,  # Trailing stop as multiple of ATR
            "time_stop_bars": 20,  # Maximum bars to hold if not moving favorably
            "use_chandelier_exit": True,  # Use chandelier exits (ATR-based trailing stop)
            "chandelier_atr_multiple": 3.0,
            "chandelier_lookback": 10,
            
            # Risk management
            "risk_per_trade": 0.01,  # Risk 1% per trade
            "max_risk_multiple_correlated": 1.5,  # Maximum risk if correlated positions exist
            "max_open_trades": 5,  # Maximum concurrent swing trades
            "min_reward_risk_ratio": 1.8,  # Minimum reward-to-risk ratio
            "max_daily_drawdown": 0.03,  # Maximum acceptable daily drawdown
            "max_position_size": 2.0,  # Maximum position size in lots
            "correlation_threshold": 0.7,  # Correlation level to consider pairs related
            
            # Filter parameters
            "atr_period": 14,
            "min_atr_pips": 30,  # Minimum volatility for trading
            "max_atr_pips": 200,  # Maximum volatility for trading
            "volatility_rank_threshold": 0.3,  # Minimum volatility rank for trading
            "volume_filter_enabled": True,
            "volume_period": 20,
            "min_volume_percentile": 40,  # Minimum volume percentile for entry
            
            # Sentiment parameters
            "use_sentiment_filter": False,  # Placeholder for sentiment data integration
            "sentiment_threshold": 0.6,
            
            # Event risk parameters
            "use_event_risk_filter": True,
            "high_impact_event_window_hours": 12,  # Avoid entries this many hours before high impact news
            
            # General parameters
            "timezone": pytz.UTC,
            "preferred_pairs": [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF"
            ],
            "pair_selection_count": 5  # Maximum number of pairs to trade concurrently
        },
        
        # Aggressive configuration for strong trends
        "aggressive": {
            "entry_methods": ["breakout", "momentum"],
            "profit_target_atr_multiple": 4.0,
            "initial_stop_atr_multiple": 1.2,
            "trailing_stop_activation_atr": 1.5,
            "pullback_threshold": 0.236,  # Shallower pullbacks
            "risk_per_trade": 0.015,
            "min_reward_risk_ratio": 2.0
        },
        
        # Conservative configuration for choppy markets
        "conservative": {
            "entry_methods": ["pullback"],
            "adx_threshold": 30,  # Require stronger trends
            "profit_target_atr_multiple": 2.5,
            "initial_stop_atr_multiple": 1.8,
            "risk_per_trade": 0.008,
            "min_reward_risk_ratio": 2.2,
            "pullback_threshold": 0.5,  # Deeper pullbacks
            "swing_threshold_pips": 40  # Require larger swings
        }
    }
)
class ForexSwingTradingStrategy(ForexBaseStrategy):
    """
    A strategy that identifies and trades medium-term price swings in forex markets.
    
    Swing trading seeks to capture a portion of an identifiable price move or "swing"
    by using technical analysis to identify potential turning points and trends.
    This implementation uses multi-timeframe analysis, trend identification, and
    strategic entry timing to capture significant price moves.
    """
    
    def __init__(self, session=None):
        """
        Initialize the swing trading strategy.
        
        Args:
            session: Trading session object with configuration
        """
        super().__init__(session)
        self.name = "ForexSwingTrading"
        self.description = "Medium-term swing trading for forex"
        self.logger = logging.getLogger(__name__)
        
        # Active trades and signal tracking
        self.active_trades = {}  # symbol -> trade data
        self.pending_signals = {}  # symbol -> signal data
        self.signals_generated = {}  # Keep track of recent signals to avoid duplication
        
        # Support/Resistance levels
        self.support_resistance_levels = {}  # symbol -> list of SR levels
        
        # Last analysis times to avoid repeated calculations
        self.last_analysis_time = {}  # symbol -> last analysis time
        
        # Market regime tracking (trending, ranging, etc.)
        self.market_regimes = {}  # symbol -> regime data
        
        # Correlation data
        self.correlation_matrix = None
        self.last_correlation_update = None
        
        # Performance statistics
        self.stats = {
            "trades_taken": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "average_holding_days": 0,
            "average_profit_pips": 0,
            "largest_win_pips": 0,
            "largest_loss_pips": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pips_gained": 0,
            "total_pips_lost": 0
        }
        
        # Event risk calendar (to be populated from economic calendar)
        self.economic_events = {}  # date -> list of high impact events
        
    def initialize(self) -> None:
        """Initialize strategy and load any required data."""
        super().initialize()
        
        # Initialize correlation matrix
        self._update_correlation_matrix()
        
        # Initialize market regimes
        self.market_regimes = {}
        
        # Initialize active trades
        self.active_trades = {}
        self.pending_signals = {}
        
        self.logger.info(f"Initialized {self.name} strategy")
        
        # For real implementation, we would load upcoming economic events here
        # self._load_economic_calendar()
        
    def _update_correlation_matrix(self) -> None:
        """Update the correlation matrix for currency pairs."""
        # In a real implementation, this would calculate correlations from historical price data
        # For now, we'll use a simplified approximation based on currency relationships
        pairs = self.parameters["preferred_pairs"]
        
        # Initialize an empty correlation matrix
        correlation_matrix = pd.DataFrame(np.eye(len(pairs)), index=pairs, columns=pairs)
        
        # Fill with approximate correlations (in a real system, this would use actual price data)
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i == j:
                    continue  # Skip diagonal (self-correlation)
                
                # Extract currencies
                base1, quote1 = pair1[:3], pair1[3:6]
                base2, quote2 = pair2[:3], pair2[3:6]
                
                # Count shared currencies
                shared = 0
                if base1 == base2 or base1 == quote2:
                    shared += 1
                if quote1 == base2 or quote1 == quote2:
                    shared += 1
                
                # Simple correlation approximation
                if shared == 2:  # Same currencies in reverse (e.g., EURUSD vs USDEUR)
                    corr = -0.95
                elif shared == 1:  # One currency in common
                    # If the shared currency is in the same position, correlation tends to be positive
                    if (base1 == base2) or (quote1 == quote2):
                        corr = 0.7
                    else:
                        corr = -0.3
                else:  # No currencies in common
                    corr = 0.1  # Slight baseline correlation due to market factors
                
                correlation_matrix.loc[pair1, pair2] = corr
        
        self.correlation_matrix = correlation_matrix
        self.last_correlation_update = datetime.now(self.parameters["timezone"])
        
        self.logger.info(f"Updated correlation matrix for {len(pairs)} currency pairs")
        
    def _identify_market_regime(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Identify the current market regime (trending, ranging, etc.) for a currency pair.
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            
        Returns:
            Dictionary with regime identification
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return {"regime": "unknown", "strength": 0, "direction": "neutral"}
        
        # Extract parameters
        adx_period = self.parameters["adx_period"]
        adx_threshold = self.parameters["adx_threshold"]
        
        # Calculate ADX for trend strength
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().multiply(-1)
        
        # When +DM is larger and positive, keep +DM, otherwise set to 0
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        
        # When -DM is larger and positive, keep -DM, otherwise set to 0
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(window=adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(window=adx_period).mean()
        
        current_adx = adx.iloc[-1] if not adx.empty else 0
        current_plus_di = plus_di.iloc[-1] if not plus_di.empty else 0
        current_minus_di = minus_di.iloc[-1] if not minus_di.empty else 0
        
        # Calculate EMAs for trend direction
        ema_fast = close.ewm(span=self.parameters["ema_fast"], adjust=False).mean()
        ema_slow = close.ewm(span=self.parameters["ema_slow"], adjust=False).mean()
        ema_very_slow = close.ewm(span=self.parameters["ema_very_slow"], adjust=False).mean()
        
        current_fast = ema_fast.iloc[-1] if not ema_fast.empty else 0
        current_slow = ema_slow.iloc[-1] if not ema_slow.empty else 0
        current_very_slow = ema_very_slow.iloc[-1] if not ema_very_slow.empty else 0
        
        # Determine trend direction based on EMAs
        if current_fast > current_slow and current_slow > current_very_slow:
            trend_direction = "bullish"
        elif current_fast < current_slow and current_slow < current_very_slow:
            trend_direction = "bearish"
        else:
            # Mixed signals
            if current_fast > current_slow:
                trend_direction = "weakly_bullish"
            elif current_fast < current_slow:
                trend_direction = "weakly_bearish"
            else:
                trend_direction = "neutral"
        
        # Determine market regime
        if current_adx >= adx_threshold:
            # Trending market
            if current_plus_di > current_minus_di:
                regime = "trending_up"
            else:
                regime = "trending_down"
            
            # Trend strength based on ADX (0-100 scale)
            trend_strength = current_adx / 100
        else:
            # Ranging market
            regime = "ranging"
            trend_strength = current_adx / adx_threshold  # Normalized to 0-1 range
        
        # Detect choppy market
        price_changes = close.pct_change().abs()
        recent_volatility = price_changes.rolling(window=20).std().iloc[-1] if len(price_changes) >= 20 else 0
        if regime == "ranging" and recent_volatility > 0.003:  # High volatility range
            regime = "choppy"
        
        # Check if price is in a tight consolidation
        atr = self._calculate_atr(data)
        recent_atr = atr / close.iloc[-1]  # ATR as percentage of price
        if recent_atr < 0.001 and regime == "ranging":
            regime = "consolidation"
        
        # Store the regime information for later use
        self.market_regimes[symbol] = {
            "regime": regime,
            "direction": trend_direction,
            "strength": trend_strength,
            "adx": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "last_updated": datetime.now(self.parameters["timezone"])
        }
        
        return self.market_regimes[symbol]
    
    def _identify_swings(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Identify recent price swings (pivots and moves).
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            
        Returns:
            List of dictionaries with swing information
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return []
        
        # Extract parameters
        lookback = self.parameters["swing_lookback_periods"]
        min_swing_pips = self.parameters["swing_threshold_pips"]
        min_duration = self.parameters["swing_duration_min"]
        max_duration = self.parameters["swing_duration_max"]
        
        # Determine pip value
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        
        # Get price data
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # Find potential swing points using zigzag-like algorithm
        # This is a simplified implementation - a real one might use more sophisticated techniques
        swing_points = []
        
        # Initialize with the first point
        last_swing = {"index": 0, "price": close.iloc[0], "type": "low", "confirmed": True}
        swing_points.append(last_swing)
        
        # Process remaining points
        for i in range(1, len(close)):
            current_price = close.iloc[i]
            
            # Determine swing direction
            if last_swing["type"] == "low":
                # Looking for a high
                higher_than_last = current_price > last_swing["price"]
                
                # Check if this could be a new swing high
                if higher_than_last:
                    # If we haven't added a potential high yet, add this
                    if len(swing_points) == 1 or swing_points[-1]["type"] == "low":
                        swing_points.append({
                            "index": i,
                            "price": current_price,
                            "type": "high",
                            "confirmed": False
                        })
                    # Otherwise, check if this is higher than the current potential high
                    elif current_price > swing_points[-1]["price"]:
                        swing_points[-1] = {
                            "index": i,
                            "price": current_price,
                            "type": "high",
                            "confirmed": False
                        }
            else:
                # Looking for a low
                lower_than_last = current_price < last_swing["price"]
                
                # Check if this could be a new swing low
                if lower_than_last:
                    # If we haven't added a potential low yet, add this
                    if len(swing_points) == 1 or swing_points[-1]["type"] == "high":
                        swing_points.append({
                            "index": i,
                            "price": current_price,
                            "type": "low",
                            "confirmed": False
                        })
                    # Otherwise, check if this is lower than the current potential low
                    elif current_price < swing_points[-1]["price"]:
                        swing_points[-1] = {
                            "index": i,
                            "price": current_price,
                            "type": "low",
                            "confirmed": False
                        }
            
            # Check for confirmation of the previous point
            if len(swing_points) >= 2 and not swing_points[-1]["confirmed"]:
                prev_swing = swing_points[-2]
                curr_swing = swing_points[-1]
                
                # Confirmation happens when we're starting to move away from the potential swing point
                if curr_swing["type"] == "high" and current_price < (curr_swing["price"] - pip_value * min_swing_pips / 2):
                    curr_swing["confirmed"] = True
                    last_swing = curr_swing
                elif curr_swing["type"] == "low" and current_price > (curr_swing["price"] + pip_value * min_swing_pips / 2):
                    curr_swing["confirmed"] = True
                    last_swing = curr_swing
        
        # Filter swings to ensure they meet criteria (minimum size, etc.)
        valid_swings = []
        for i in range(1, len(swing_points)):
            if not swing_points[i]["confirmed"]:
                continue
                
            current = swing_points[i]
            previous = swing_points[i-1]
            
            # Calculate swing size in pips
            swing_size_pips = abs(current["price"] - previous["price"]) / pip_value
            
            # Calculate swing duration in bars
            duration = current["index"] - previous["index"]
            
            # Check if swing meets criteria
            if (swing_size_pips >= min_swing_pips and 
                duration >= min_duration and 
                duration <= max_duration):
                
                # Create swing record
                swing = {
                    "start_index": previous["index"],
                    "end_index": current["index"],
                    "start_price": previous["price"],
                    "end_price": current["price"],
                    "type": "up" if current["price"] > previous["price"] else "down",
                    "size_pips": swing_size_pips,
                    "duration": duration,
                    "start_date": data.index[previous["index"]] if len(data.index) > previous["index"] else None,
                    "end_date": data.index[current["index"]] if len(data.index) > current["index"] else None
                }
                
                valid_swings.append(swing)
        
        # Return the most recent swings (limited to lookback parameter)
        return valid_swings[-lookback:] if len(valid_swings) > lookback else valid_swings
    
    def _identify_support_resistance(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Identify key support and resistance levels.
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            
        Returns:
            List of dictionaries with support/resistance information
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return []
        
        # Extract parameters
        lookback = self.parameters["sr_lookback_periods"]
        min_touches = self.parameters["sr_min_touches"]
        zone_pips = self.parameters["sr_zone_pips"]
        
        # Determine pip value
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        zone_size = zone_pips * pip_value
        
        # Get recent price swings
        swings = self._identify_swings(data, symbol)
        
        # Extract swing high/low prices
        swing_highs = [s["end_price"] for s in swings if s["type"] == "up"]
        swing_lows = [s["end_price"] for s in swings if s["type"] == "down"]
        
        # Add recent price extremes
        recent_data = data.iloc[-lookback:]
        highest_high = recent_data["high"].max()
        lowest_low = recent_data["low"].min()
        
        if highest_high not in swing_highs:
            swing_highs.append(highest_high)
        if lowest_low not in swing_lows:
            swing_lows.append(lowest_low)
        
        # Identify zones of support/resistance by clustering similar price levels
        sr_zones = []
        
        # Process swing highs (resistance)
        for price in sorted(swing_highs):
            # Check if this price belongs to an existing zone
            found_zone = False
            for zone in sr_zones:
                if zone["type"] == "resistance" and abs(price - zone["price"]) <= zone_size:
                    # Update zone with new touch
                    zone["touches"] += 1
                    zone["prices"].append(price)
                    zone["price"] = sum(zone["prices"]) / len(zone["prices"])  # Average price
                    found_zone = True
                    break
            
            # If not found in any zone, create a new one
            if not found_zone:
                sr_zones.append({
                    "type": "resistance",
                    "price": price,
                    "prices": [price],
                    "touches": 1,
                    "zone_size": zone_size,
                    "strength": 1.0  # Will be calculated later
                })
        
        # Process swing lows (support)
        for price in sorted(swing_lows):
            # Check if this price belongs to an existing zone
            found_zone = False
            for zone in sr_zones:
                if zone["type"] == "support" and abs(price - zone["price"]) <= zone_size:
                    # Update zone with new touch
                    zone["touches"] += 1
                    zone["prices"].append(price)
                    zone["price"] = sum(zone["prices"]) / len(zone["prices"])  # Average price
                    found_zone = True
                    break
            
            # If not found in any zone, create a new one
            if not found_zone:
                sr_zones.append({
                    "type": "support",
                    "price": price,
                    "prices": [price],
                    "touches": 1,
                    "zone_size": zone_size,
                    "strength": 1.0  # Will be calculated later
                })
        
        # Filter out zones with insufficient touches
        sr_zones = [zone for zone in sr_zones if zone["touches"] >= min_touches]
        
        # Calculate zone strength based on number of touches and recency
        if sr_zones:
            max_touches = max(zone["touches"] for zone in sr_zones)
            
            for zone in sr_zones:
                # Strength based on number of touches (normalized to 0-1)
                touch_strength = zone["touches"] / max_touches if max_touches > 0 else 0
                
                # Recency factor - more recent zones are stronger
                # This would be more sophisticated in a real implementation
                recency_strength = 1.0
                
                # Final strength combination
                zone["strength"] = (touch_strength * 0.7) + (recency_strength * 0.3)
        
        # Store the zones for future reference
        self.support_resistance_levels[symbol] = sr_zones
        
        return sr_zones
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """
        Calculate Average True Range for volatility assessment.
        
        Args:
            data: OHLCV data
            
        Returns:
            ATR value
        """
        atr_period = self.parameters["atr_period"]
        
        if len(data) < atr_period + 1:
            return 0.0
            
        # Calculate true range
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=atr_period).mean().iloc[-1]
        
        return atr
    
    def _calculate_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate momentum indicators.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with momentum indicator values
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return {"rsi": 50, "macd": 0, "macd_signal": 0, "macd_histogram": 0, "momentum_signal": "neutral"}
        
        close = data["close"]
        
        # Calculate RSI
        rsi_period = self.parameters["rsi_period"]
        rsi = self._calculate_rsi(close, rsi_period)
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate MACD
        macd_fast = self.parameters["macd_fast"]
        macd_slow = self.parameters["macd_slow"]
        macd_signal_period = self.parameters["macd_signal"]
        
        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=macd_signal_period, adjust=False).mean()
        macd_histogram = macd_line - macd_signal_line
        
        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0
        current_macd_signal = macd_signal_line.iloc[-1] if not macd_signal_line.empty else 0
        current_macd_histogram = macd_histogram.iloc[-1] if not macd_histogram.empty else 0
        
        # Combined momentum signal
        momentum_signal = "neutral"
        
        # RSI signals
        if current_rsi > self.parameters["rsi_overbought"]:
            rsi_signal = "bearish"
        elif current_rsi < self.parameters["rsi_oversold"]:
            rsi_signal = "bullish"
        else:
            # In the middle zone, use RSI direction
            rsi_change = rsi.diff(3).iloc[-1] if len(rsi) > 3 else 0
            if rsi_change > 3:  # Increasing RSI
                rsi_signal = "bullish"
            elif rsi_change < -3:  # Decreasing RSI
                rsi_signal = "bearish"
            else:
                rsi_signal = "neutral"
        
        # MACD signals
        if current_macd > current_macd_signal and current_macd_histogram > 0:
            macd_signal = "bullish"
        elif current_macd < current_macd_signal and current_macd_histogram < 0:
            macd_signal = "bearish"
        else:
            # Check for divergence or convergence
            if current_macd_histogram > 0 and current_macd_histogram > macd_histogram.iloc[-2]:
                macd_signal = "bullish_strengthening"
            elif current_macd_histogram < 0 and current_macd_histogram < macd_histogram.iloc[-2]:
                macd_signal = "bearish_strengthening"
            elif current_macd_histogram > 0 and current_macd_histogram < macd_histogram.iloc[-2]:
                macd_signal = "bullish_weakening"
            elif current_macd_histogram < 0 and current_macd_histogram > macd_histogram.iloc[-2]:
                macd_signal = "bearish_weakening"
            else:
                macd_signal = "neutral"
        
        # Combined signal logic
        if rsi_signal == "bullish" and (macd_signal.startswith("bullish")):
            momentum_signal = "strongly_bullish"
        elif rsi_signal == "bearish" and (macd_signal.startswith("bearish")):
            momentum_signal = "strongly_bearish"
        elif rsi_signal == "bullish" or (macd_signal.startswith("bullish")):
            momentum_signal = "weakly_bullish"
        elif rsi_signal == "bearish" or (macd_signal.startswith("bearish")):
            momentum_signal = "weakly_bearish"
        
        return {
            "rsi": current_rsi,
            "macd": current_macd,
            "macd_signal": current_macd_signal,
            "macd_histogram": current_macd_histogram,
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "momentum_signal": momentum_signal
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        if len(prices) <= period:
            return pd.Series([50] * len(prices), index=prices.index)  # Default to neutral
            
        # Get price changes
        delta = prices.diff()
        delta = delta[1:]  # Remove first NA
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate RS and RSI
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Fill NAs with 50 (neutral)
        rsi = rsi.fillna(50)
        
        return rsi
    
    def _check_entry_conditions(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Check if entry conditions are met.
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            
        Returns:
            Dictionary with entry signal information, or None if no signal
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return None
        
        # Skip if we recently generated a signal for this symbol
        current_time = datetime.now(self.parameters["timezone"])
        if symbol in self.signals_generated:
            last_signal_time = self.signals_generated[symbol]["time"]
            if (current_time - last_signal_time).total_seconds() < 86400:  # 24 hours
                return None
        
        # Skip if we already have an active trade for this symbol
        if symbol in self.active_trades:
            return None
        
        # Get current price and analysis data
        current_price = data["close"].iloc[-1]
        regime = self._identify_market_regime(data, symbol)
        sr_levels = self._identify_support_resistance(data, symbol)
        momentum = self._calculate_momentum(data)
        swings = self._identify_swings(data, symbol)
        atr = self._calculate_atr(data)
        
        # Determine pip value
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        
        # Check if volatility is within acceptable range
        atr_pips = atr / pip_value
        if atr_pips < self.parameters["min_atr_pips"] or atr_pips > self.parameters["max_atr_pips"]:
            return None
        
        # Check for high impact economic events
        if self.parameters["use_event_risk_filter"] and self._check_event_risk(symbol, current_time):
            return None
        
        # Entry variables
        entry_signal = None
        entry_type = None
        stop_loss = None
        take_profit = None
        entry_price = current_price
        signal_strength = 0.0
        entry_notes = []
        
        # Determine potential entry methods based on market regime
        valid_entry_methods = self.parameters["entry_methods"].copy()
        
        # Adjust entry methods based on market regime
        if regime["regime"] == "trending_up" or regime["regime"] == "trending_down":
            # Favor breakout and momentum entries in trending markets
            if "breakout" in valid_entry_methods:
                valid_entry_methods.insert(0, valid_entry_methods.pop(valid_entry_methods.index("breakout")))
            if "momentum" in valid_entry_methods:
                valid_entry_methods.insert(0, valid_entry_methods.pop(valid_entry_methods.index("momentum")))
        elif regime["regime"] == "ranging" or regime["regime"] == "consolidation":
            # Favor pullback entries in ranging markets
            if "pullback" in valid_entry_methods:
                valid_entry_methods.insert(0, valid_entry_methods.pop(valid_entry_methods.index("pullback")))
        
        # Try entry methods in order of priority
        for method in valid_entry_methods:
            if entry_signal is not None:
                break
                
            if method == "pullback":
                # Check for pullback entry (retracement to support/resistance in overall trend)
                pullback_signal = self._check_pullback_entry(data, symbol, regime, sr_levels, momentum, swings)
                if pullback_signal is not None:
                    entry_signal = pullback_signal["signal"]
                    entry_type = "pullback"
                    entry_price = pullback_signal["entry_price"]
                    stop_loss = pullback_signal["stop_loss"]
                    take_profit = pullback_signal["take_profit"]
                    signal_strength = pullback_signal["strength"]
                    entry_notes.append(pullback_signal["note"])
                    
            elif method == "breakout":
                # Check for breakout entry (price breaking through key levels with momentum)
                breakout_signal = self._check_breakout_entry(data, symbol, regime, sr_levels, momentum, swings)
                if breakout_signal is not None:
                    entry_signal = breakout_signal["signal"]
                    entry_type = "breakout"
                    entry_price = breakout_signal["entry_price"]
                    stop_loss = breakout_signal["stop_loss"]
                    take_profit = breakout_signal["take_profit"]
                    signal_strength = breakout_signal["strength"]
                    entry_notes.append(breakout_signal["note"])
                    
            elif method == "momentum":
                # Check for momentum entry (strong momentum in trending market)
                momentum_signal = self._check_momentum_entry(data, symbol, regime, momentum)
                if momentum_signal is not None:
                    entry_signal = momentum_signal["signal"]
                    entry_type = "momentum"
                    entry_price = momentum_signal["entry_price"]
                    stop_loss = momentum_signal["stop_loss"]
                    take_profit = momentum_signal["take_profit"]
                    signal_strength = momentum_signal["strength"]
                    entry_notes.append(momentum_signal["note"])
        
        # If no entry signal was found, return None
        if entry_signal is None:
            return None
            
        # Apply additional entry filters
        filter_methods = self.parameters["entry_filter_methods"]
        
        # Volatility filter
        if "volatility" in filter_methods:
            if atr_pips < self.parameters["min_atr_pips"]:
                entry_notes.append(f"Rejected due to low volatility: {atr_pips:.1f} pips ATR")
                return None
        
        # Momentum filter
        if "momentum" in filter_methods and entry_type != "momentum":
            # Ensure momentum aligns with entry direction
            if (entry_signal == "buy" and not momentum["momentum_signal"].endswith("bullish")) or \
               (entry_signal == "sell" and not momentum["momentum_signal"].endswith("bearish")):
                entry_notes.append(f"Rejected due to conflicting momentum: {momentum['momentum_signal']}")
                return None
        
        # Volume filter
        if "volume" in filter_methods and self.parameters["volume_filter_enabled"] and "volume" in data:
            volume_period = self.parameters["volume_period"]
            min_percentile = self.parameters["min_volume_percentile"]
            
            recent_volume = data["volume"].iloc[-volume_period:]
            current_volume = recent_volume.iloc[-1]
            volume_percentile = sum(current_volume > recent_volume) / len(recent_volume) * 100
            
            if volume_percentile < min_percentile:
                entry_notes.append(f"Rejected due to low volume: {volume_percentile:.1f}% percentile")
                return None
        
        # Correlation filter - check for correlated positions
        if self.correlation_matrix is not None and len(self.active_trades) > 0:
            # Check correlation with active trades
            for active_symbol in self.active_trades.keys():
                if active_symbol in self.correlation_matrix.index and symbol in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[active_symbol, symbol])
                    
                    if correlation > self.parameters["correlation_threshold"]:
                        active_direction = self.active_trades[active_symbol]["direction"]
                        
                        # If correlation is positive and directions match, or correlation is negative and directions differ
                        # then the positions are correlated (taking same market risk)
                        is_correlated = (correlation > 0 and active_direction == entry_signal) or \
                                       (correlation < 0 and active_direction != entry_signal)
                        
                        if is_correlated:
                            entry_notes.append(f"Reduced position size due to correlation with {active_symbol}: {correlation:.2f}")
                            # We'll still allow the trade but reduce position size later
                            break
        
        # Calculate reward-to-risk ratio
        if stop_loss is not None and take_profit is not None:
            if entry_signal == "buy":
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # sell
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                
            reward_risk_ratio = reward / risk if risk > 0 else 0
            
            # Check minimum reward-to-risk ratio
            if reward_risk_ratio < self.parameters["min_reward_risk_ratio"]:
                entry_notes.append(f"Rejected due to insufficient reward-to-risk ratio: {reward_risk_ratio:.2f}")
                return None
        else:
            reward_risk_ratio = 0
        
        # Record the signal generation time
        self.signals_generated[symbol] = {
            "signal": entry_signal,
            "time": current_time,
            "type": entry_type
        }
        
        # Return entry signal information
        return {
            "symbol": symbol,
            "signal": entry_signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_type": entry_type,
            "strength": signal_strength,
            "atr": atr,
            "atr_pips": atr_pips,
            "notes": entry_notes,
            "reward_risk_ratio": reward_risk_ratio,
            "regime": regime["regime"],
            "momentum": momentum["momentum_signal"]
        }
    
    def _check_pullback_entry(self, data: pd.DataFrame, symbol: str, 
                             regime: Dict[str, Any], sr_levels: List[Dict[str, Any]], 
                             momentum: Dict[str, Any], swings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Check for pullback entry opportunities.
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            regime: Market regime information
            sr_levels: Support and resistance levels
            momentum: Momentum indicator values
            swings: Recent price swings
            
        Returns:
            Dictionary with entry signal information, or None if no signal
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return None
            
        # Get current price and recent price action
        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]
        
        # Get trend direction from regime
        trend_direction = regime["direction"]
        
        # Pullback threshold (Fibonacci retracement level)
        pullback_threshold = self.parameters["pullback_threshold"]
        
        # Skip if there's no clear trend or not enough swings
        if trend_direction == "neutral" or len(swings) < 2:
            return None
            
        # ATR for stop loss and take profit calculation
        atr = self._calculate_atr(data)
        
        # For pullbacks, we need a larger trend with a recent counter-move
        # Look at the most recent swing
        if len(swings) < 1:
            return None
            
        last_swing = swings[-1]
        last_swing_type = last_swing["type"]
        
        # In an uptrend, we want a recent down swing (pullback)
        # In a downtrend, we want a recent up swing (pullback)
        valid_pullback = ((trend_direction.startswith("bullish") and last_swing_type == "down") or 
                          (trend_direction.startswith("bearish") and last_swing_type == "up"))
                          
        if not valid_pullback:
            return None
            
        # Find the most recent swing high and low to calculate retracement
        swing_high = None
        swing_low = None
        
        # Extract the most recent swing high and low
        for swing in reversed(swings):
            if swing["type"] == "up" and swing_high is None:
                swing_high = swing["end_price"]
            elif swing["type"] == "down" and swing_low is None:
                swing_low = swing["end_price"]
                
            if swing_high is not None and swing_low is not None:
                break
                
        # If we don't have both high and low, we can't calculate retracement
        if swing_high is None or swing_low is None:
            return None
            
        # Calculate the retracement level
        swing_range = abs(swing_high - swing_low)
        
        # Ensure swing_range is meaningful
        if swing_range < atr * 0.5:
            return None
            
        # Calculate the target retracement level
        if trend_direction.startswith("bullish"):
            # In an uptrend, retracement will be from high back down toward low
            retracement_level = swing_high - (swing_range * pullback_threshold)
            
            # Check if price has retraced enough but not too much
            if current_price > retracement_level or current_price < swing_low:
                return None
                
            # Identify any support levels near the retracement
            nearby_support = None
            for level in sr_levels:
                if level["type"] == "support" and abs(level["price"] - current_price) < atr:
                    if nearby_support is None or level["strength"] > nearby_support["strength"]:
                        nearby_support = level
            
            # Entry criteria: price near retracement level or support, showing signs of reversal
            bullish_momentum = momentum["momentum_signal"].endswith("bullish")
            
            if not bullish_momentum:
                # Check short-term reversal patterns
                if len(close) >= 3:
                    # Check for bullish reversal candle patterns
                    prev_close = close.iloc[-2]
                    prev_open = data["open"].iloc[-2]
                    prev_low = low.iloc[-2]
                    
                    current_open = data["open"].iloc[-1]
                    current_low = low.iloc[-1]
                    
                    # Bullish engulfing or hammer pattern
                    bullish_pattern = ((current_close > current_open and  # Bullish candle
                                      prev_close < prev_open and  # Previous bearish candle
                                      current_open < prev_close and  # Opens below previous close
                                      current_close > prev_open) or  # Closes above previous open
                                     (current_low < current_open and  # Has a lower wick
                                      (current_close - current_low) > 2 * (current_open - current_close))  # Hammer ratio
                                    )
                    
                    if not bullish_pattern:
                        return None
                else:
                    return None
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            
            # Stop below the recent low or nearby support, with buffer
            if nearby_support is not None:
                stop_loss = min(swing_low - (atr * 0.3), nearby_support["price"] - (atr * 0.5))
            else:
                stop_loss = swing_low - (atr * 0.3)
                
            # Take profit at the previous high or the next resistance level
            take_profit = swing_high + (atr * self.parameters["profit_target_atr_multiple"])
            
            # Check nearby resistance levels for potential take profit adjustment
            for level in sr_levels:
                if level["type"] == "resistance" and level["price"] > current_price:
                    # Consider this as a potential take profit level
                    take_profit = min(take_profit, level["price"])
                    break
            
            return {
                "signal": "buy",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strength": 0.7 if nearby_support else 0.5,
                "note": f"Bullish pullback entry at {pullback_threshold:.1%} retracement"
            }
            
        else:  # Bearish trend
            # In a downtrend, retracement will be from low back up toward high
            retracement_level = swing_low + (swing_range * pullback_threshold)
            
            # Check if price has retraced enough but not too much
            if current_price < retracement_level or current_price > swing_high:
                return None
                
            # Identify any resistance levels near the retracement
            nearby_resistance = None
            for level in sr_levels:
                if level["type"] == "resistance" and abs(level["price"] - current_price) < atr:
                    if nearby_resistance is None or level["strength"] > nearby_resistance["strength"]:
                        nearby_resistance = level
            
            # Entry criteria: price near retracement level or resistance, showing signs of reversal
            bearish_momentum = momentum["momentum_signal"].endswith("bearish")
            
            if not bearish_momentum:
                # Check short-term reversal patterns
                if len(close) >= 3:
                    # Check for bearish reversal candle patterns
                    prev_close = close.iloc[-2]
                    prev_open = data["open"].iloc[-2]
                    prev_high = high.iloc[-2]
                    
                    current_open = data["open"].iloc[-1]
                    current_high = high.iloc[-1]
                    
                    # Bearish engulfing or shooting star pattern
                    bearish_pattern = ((current_close < current_open and  # Bearish candle
                                      prev_close > prev_open and  # Previous bullish candle
                                      current_open > prev_close and  # Opens above previous close
                                      current_close < prev_open) or  # Closes below previous open
                                     (current_high > current_open and  # Has an upper wick
                                      (current_high - current_close) > 2 * (current_close - current_open))  # Shooting star ratio
                                    )
                    
                    if not bearish_pattern:
                        return None
                else:
                    return None
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            
            # Stop above the recent high or nearby resistance, with buffer
            if nearby_resistance is not None:
                stop_loss = max(swing_high + (atr * 0.3), nearby_resistance["price"] + (atr * 0.5))
            else:
                stop_loss = swing_high + (atr * 0.3)
                
            # Take profit at the previous low or the next support level
            take_profit = swing_low - (atr * self.parameters["profit_target_atr_multiple"])
            
            # Check nearby support levels for potential take profit adjustment
            for level in sr_levels:
                if level["type"] == "support" and level["price"] < current_price:
                    # Consider this as a potential take profit level
                    take_profit = max(take_profit, level["price"])
                    break
            
            return {
                "signal": "sell",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strength": 0.7 if nearby_resistance else 0.5,
                "note": f"Bearish pullback entry at {pullback_threshold:.1%} retracement"
            }
        
        return None
    
    def _check_breakout_entry(self, data: pd.DataFrame, symbol: str, 
                             regime: Dict[str, Any], sr_levels: List[Dict[str, Any]], 
                             momentum: Dict[str, Any], swings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Check for breakout entry opportunities.
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            regime: Market regime information
            sr_levels: Support and resistance levels
            momentum: Momentum indicator values
            swings: Recent price swings
            
        Returns:
            Dictionary with entry signal information, or None if no signal
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return None
            
        # Get current price and recent price action
        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]
        
        # Confirmation bars required for breakout
        confirmation_bars = self.parameters["breakout_confirmation_bars"]
        
        # Skip if not enough bars for confirmation
        if len(close) < confirmation_bars + 5:  # Need extra bars for context
            return None
            
        # ATR for stop loss and take profit calculation
        atr = self._calculate_atr(data)
        
        # Identify potential breakout levels
        potential_breakouts = []
        
        # Check for resistance breakouts (bullish)
        for level in sr_levels:
            if level["type"] == "resistance":
                # Check if price has recently broken above the level
                prior_bars_below = all(close.iloc[-i-confirmation_bars-1] < level["price"] for i in range(3))
                post_bars_above = all(close.iloc[-i-1] > level["price"] for i in range(confirmation_bars))
                
                if prior_bars_below and post_bars_above:
                    # Confirmed breakout of resistance
                    potential_breakouts.append({
                        "level": level,
                        "direction": "buy",
                        "strength": level["strength"],
                        "distance": abs(current_price - level["price"]) / atr
                    })
        
        # Check for support breakouts (bearish)
        for level in sr_levels:
            if level["type"] == "support":
                # Check if price has recently broken below the level
                prior_bars_above = all(close.iloc[-i-confirmation_bars-1] > level["price"] for i in range(3))
                post_bars_below = all(close.iloc[-i-1] < level["price"] for i in range(confirmation_bars))
                
                if prior_bars_above and post_bars_below:
                    # Confirmed breakout of support
                    potential_breakouts.append({
                        "level": level,
                        "direction": "sell",
                        "strength": level["strength"],
                        "distance": abs(current_price - level["price"]) / atr
                    })
        
        # No potential breakouts found
        if not potential_breakouts:
            return None
            
        # Sort breakouts by strength and recency (distance)
        breakouts_sorted = sorted(potential_breakouts, 
                                 key=lambda x: (x["strength"], -x["distance"]), 
                                 reverse=True)
        
        # Take the strongest breakout
        best_breakout = breakouts_sorted[0]
        
        # Check if the breakout aligns with the overall trend and momentum
        if best_breakout["direction"] == "buy":
            trend_aligned = regime["direction"].startswith("bullish")
            momentum_aligned = momentum["momentum_signal"].endswith("bullish")
        else:  # sell
            trend_aligned = regime["direction"].startswith("bearish")
            momentum_aligned = momentum["momentum_signal"].endswith("bearish")
        
        # Increase strength if aligned with trend and momentum
        strength_modifier = 1.0
        if trend_aligned:
            strength_modifier += 0.2
        if momentum_aligned:
            strength_modifier += 0.2
            
        final_strength = min(0.9, best_breakout["strength"] * strength_modifier)
        
        # Check if the breakout is still valid (not too far from level)
        if best_breakout["distance"] > 2.0:  # More than 2 ATRs away
            # Breakout too far gone, not ideal for entry
            return None
            
        # Calculate entry, stop loss, and take profit
        entry_price = current_price
        
        if best_breakout["direction"] == "buy":
            # Buy breakout
            # Stop loss just below the breakout level
            stop_loss = best_breakout["level"]["price"] - (atr * 0.5)
            
            # Take profit based on ATR multiple
            take_profit = entry_price + (atr * self.parameters["profit_target_atr_multiple"])
            
            # Check for potential resistance levels that might limit upside
            for level in sr_levels:
                if level["type"] == "resistance" and level["price"] > entry_price:
                    take_profit = min(take_profit, level["price"] - (atr * 0.3))
                    break
                    
            return {
                "signal": "buy",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strength": final_strength,
                "note": f"Bullish breakout of resistance at {best_breakout['level']['price']:.5f}"
            }
            
        else:  # sell breakout
            # Stop loss just above the breakout level
            stop_loss = best_breakout["level"]["price"] + (atr * 0.5)
            
            # Take profit based on ATR multiple
            take_profit = entry_price - (atr * self.parameters["profit_target_atr_multiple"])
            
            # Check for potential support levels that might limit downside
            for level in sr_levels:
                if level["type"] == "support" and level["price"] < entry_price:
                    take_profit = max(take_profit, level["price"] + (atr * 0.3))
                    break
                    
            return {
                "signal": "sell",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strength": final_strength,
                "note": f"Bearish breakout of support at {best_breakout['level']['price']:.5f}"
            }
    
    def _check_momentum_entry(self, data: pd.DataFrame, symbol: str, 
                            regime: Dict[str, Any], momentum: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check for momentum entry opportunities.
        
        Args:
            data: OHLCV data
            symbol: Currency pair symbol
            regime: Market regime information
            momentum: Momentum indicator values
            
        Returns:
            Dictionary with entry signal information, or None if no signal
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return None
            
        # Get current price
        close = data["close"]
        current_price = close.iloc[-1]
        
        # ATR for stop loss and take profit calculation
        atr = self._calculate_atr(data)
        
        # Momentum entries require a strong trend and momentum
        is_trending = regime["regime"] == "trending_up" or regime["regime"] == "trending_down"
        strong_momentum = momentum["momentum_signal"].startswith("strongly_")
        
        if not is_trending or not strong_momentum:
            return None
            
        # For momentum entries, need alignment between trend, momentum, and price action
        trend_direction = regime["direction"]
        momentum_signal = momentum["momentum_signal"]
        
        # RSI and MACD direction check
        rsi = momentum["rsi"]
        rsi_rising = momentum["rsi_signal"] == "bullish"
        macd_hist = momentum["macd_histogram"]
        macd_rising = macd_hist > 0 and macd_hist > momentum["macd_signal"]
        
        # Check for alignment
        bullish_aligned = (trend_direction.startswith("bullish") and 
                         momentum_signal.endswith("bullish") and
                         (rsi_rising or macd_rising))
                         
        bearish_aligned = (trend_direction.startswith("bearish") and 
                         momentum_signal.endswith("bearish") and
                         (not rsi_rising or not macd_rising))
        
        if not (bullish_aligned or bearish_aligned):
            return None
            
        # Look for price pullbacks in the direction of the main trend
        # For a bullish trend, we want a small pullback (1-3 bars) followed by continuation
        # For a bearish trend, we want a small rally (1-3 bars) followed by continuation
        
        # Calculate recent short-term price action (last 5 bars)
        if len(close) < 5:
            return None
            
        recent_changes = [close.iloc[-i] - close.iloc[-i-1] for i in range(1, 5)]
        
        if bullish_aligned:
            # For bullish momentum entry, look for a recent pullback followed by a bullish bar
            # Pattern: down, down, [optional down], up with strong close
            pullback_pattern = (recent_changes[0] > 0 and  # Current bar is up
                              sum(1 for x in recent_changes[1:] if x < 0) >= 1)  # At least one recent down bar
            
            if not pullback_pattern:
                return None
                
            # Calculate entry, stop, and target
            entry_price = current_price
            stop_loss = entry_price - (atr * self.parameters["initial_stop_atr_multiple"])
            take_profit = entry_price + (atr * self.parameters["profit_target_atr_multiple"])
            
            return {
                "signal": "buy",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strength": 0.8 if strong_momentum else 0.6,
                "note": f"Bullish momentum entry with {momentum_signal} momentum"
            }
            
        elif bearish_aligned:
            # For bearish momentum entry, look for a recent rally followed by a bearish bar
            # Pattern: up, up, [optional up], down with strong close
            rally_pattern = (recent_changes[0] < 0 and  # Current bar is down
                           sum(1 for x in recent_changes[1:] if x > 0) >= 1)  # At least one recent up bar
            
            if not rally_pattern:
                return None
                
            # Calculate entry, stop, and target
            entry_price = current_price
            stop_loss = entry_price + (atr * self.parameters["initial_stop_atr_multiple"])
            take_profit = entry_price - (atr * self.parameters["profit_target_atr_multiple"])
            
            return {
                "signal": "sell",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strength": 0.8 if strong_momentum else 0.6,
                "note": f"Bearish momentum entry with {momentum_signal} momentum"
            }
        
        return None
    
    def _check_event_risk(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if there are any high impact economic events that may affect this pair.
        
        Args:
            symbol: Currency pair symbol
            current_time: Current time
            
        Returns:
            True if there is event risk, False otherwise
        """
        # In a real implementation, this would check an economic calendar
        # For now, we'll use a simplified placeholder implementation
        
        # Extract currencies from the symbol
        if len(symbol) >= 6:
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            # Check if we have any high impact events for these currencies
            window_hours = self.parameters["high_impact_event_window_hours"]
            event_window_end = current_time + timedelta(hours=window_hours)
            
            # Check each day in the window
            current_day = current_time.date()
            window_end_day = event_window_end.date()
            
            day = current_day
            while day <= window_end_day:
                if day in self.economic_events:
                    # Check if any event affects our currencies
                    for event in self.economic_events[day]:
                        if event.get("currency") in [base_currency, quote_currency] and event.get("impact") == "high":
                            return True
                day += timedelta(days=1)
        
        return False
    
    def _generate_signal(self, symbol: str, data: pd.DataFrame, current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Generate trading signals based on swing trading conditions.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV data
            current_time: Current time
            
        Returns:
            Signal dictionary or None if no signal
        """
        if data.empty or len(data) < self.parameters["minimum_bars"]:
            return None
            
        # Check for high-impact news events
        if self._check_event_risk(symbol, current_time):
            self.logger.info(f"Skipping signal generation for {symbol} due to upcoming high-impact events")
            return None
            
        # Analyze market regime and identify trend
        regime = self._identify_market_regime(data)
        
        # Skip if market is in consolidation and we're not looking for range trades
        if regime["regime"] == "consolidation" and not self.parameters["allow_consolidation_entries"]:
            return None
            
        # Calculate momentum indicators
        momentum = self._calculate_momentum_indicators(data)
        
        # Skip if volatility is too low
        if not self._check_volatility(data):
            return None
            
        # Skip if correlation regime suggests avoiding this pair
        if not self._check_correlation(symbol, current_time):
            return None
            
        # Identify support and resistance levels
        sr_levels = self._identify_support_resistance(data)
        
        # Identify price swings
        swings = self._identify_swings(data)
        
        # Check for different entry types based on strategy parameters
        entry_signal = None
        
        # Order of priority: Pullbacks, Breakouts, Momentum
        # This can be adjusted based on strategy configuration
        
        # Check for pullback entries
        if self.parameters["enable_pullback_entries"]:
            entry_signal = self._check_pullback_entry(data, symbol, regime, sr_levels, momentum, swings)
            
        # If no pullback entry, check for breakout entries
        if entry_signal is None and self.parameters["enable_breakout_entries"]:
            entry_signal = self._check_breakout_entry(data, symbol, regime, sr_levels, momentum, swings)
            
        # If no breakout entry, check for momentum entries
        if entry_signal is None and self.parameters["enable_momentum_entries"]:
            entry_signal = self._check_momentum_entry(data, symbol, regime, momentum)
            
        # No valid entry found
        if entry_signal is None:
            return None
            
        # Add additional signal metadata
        entry_signal["symbol"] = symbol
        entry_signal["timestamp"] = current_time
        entry_signal["strategy_type"] = "swing"
        entry_signal["timeframe"] = self.parameters["timeframe"]
        
        # Calculate reward-to-risk ratio
        risk = abs(entry_signal["entry_price"] - entry_signal["stop_loss"])
        reward = abs(entry_signal["entry_price"] - entry_signal["take_profit"])
        entry_signal["reward_risk_ratio"] = reward / risk if risk > 0 else 0
        
        # Don't take trades with poor reward-to-risk
        if entry_signal["reward_risk_ratio"] < self.parameters["minimum_reward_risk_ratio"]:
            self.logger.info(f"Rejecting {symbol} signal due to inadequate reward-risk: {entry_signal['reward_risk_ratio']:.2f}")
            return None
            
        return entry_signal
    
    def _calculate_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Signal dictionary with entry and stop loss
            account_info: Account balance and other info
            
        Returns:
            Position size in units of the base currency
        """
        # Default to risk percentage of account
        risk_per_trade_pct = self.parameters["risk_per_trade_pct"] / 100.0
        account_balance = account_info.get("balance", 0)
        currency = account_info.get("currency", "USD")
        
        # Amount willing to risk in account currency
        risk_amount = account_balance * risk_per_trade_pct
        
        # Price difference between entry and stop (risk per unit)
        price_risk = abs(signal["entry_price"] - signal["stop_loss"])
        
        # Adjust for low volatility and/or strong signals
        volatility_factor = 1.0
        
        # Increase position size for stronger signals
        if signal["strength"] > 0.7:
            volatility_factor *= 1.2
        elif signal["strength"] < 0.5:
            volatility_factor *= 0.8
            
        # Apply correlation adjustment if available
        if "correlation_adjustment" in signal:
            volatility_factor *= signal["correlation_adjustment"]
            
        # Adjust risk amount based on factors
        adjusted_risk_amount = risk_amount * volatility_factor
        
        # Calculate position size (units of base currency)
        if price_risk > 0:
            position_size = adjusted_risk_amount / price_risk
        else:
            # Fallback to a small default position if we can't calculate properly
            position_size = (account_balance * 0.01) / signal["entry_price"]
            self.logger.warning(f"Using fallback position sizing for {signal['symbol']} due to zero price risk")
            
        # Apply maximum position size limit as a percentage of account
        max_position_value = account_balance * (self.parameters["max_position_size_pct"] / 100.0)
        position_value = position_size * signal["entry_price"]
        
        if position_value > max_position_value:
            position_size = max_position_value / signal["entry_price"]
            
        return position_size
    
    def _adjust_stops_and_targets(self, 
                                 symbol: str, 
                                 position: Dict[str, Any], 
                                 current_data: pd.DataFrame,
                                 current_time: datetime) -> Dict[str, Any]:
        """
        Adjust stop losses and take profits based on price action and time in trade.
        
        Args:
            symbol: Currency pair
            position: Current position details
            current_data: Current market data
            current_time: Current time
            
        Returns:
            Updated position with new stops and targets
        """
        if current_data.empty:
            return position
            
        current_price = current_data["close"].iloc[-1]
        entry_price = position["entry_price"]
        current_stop = position["stop_loss"]
        direction = 1 if position["position_type"] == "buy" else -1
        atr = self._calculate_atr(current_data)
        
        # Calculate trade duration
        trade_start_time = position.get("entry_time", current_time - timedelta(days=1))
        trade_duration = current_time - trade_start_time
        hours_in_trade = trade_duration.total_seconds() / 3600
        
        # Calculate unrealized profit/loss as R-multiple (reward-to-risk ratio)
        initial_risk = abs(entry_price - position["initial_stop_loss"])
        current_profit = direction * (current_price - entry_price)
        r_multiple = current_profit / initial_risk if initial_risk > 0 else 0
        
        # Store current R-multiple in position
        position["current_r_multiple"] = r_multiple
        
        # Trailing stop logic
        if self.parameters["use_trailing_stop"]:
            # Different trailing stop strategies based on profit
            if r_multiple >= 2.0:
                # Move stop to breakeven + buffer
                new_stop = entry_price + (direction * atr * 0.3)
                
                # Ensure we're moving the stop in the right direction
                if (direction > 0 and new_stop > current_stop) or (direction < 0 and new_stop < current_stop):
                    position["stop_loss"] = new_stop
                    position["stop_type"] = "trailing_2R"
                    
            elif r_multiple >= 1.0:
                # Move stop to breakeven
                new_stop = entry_price
                
                # Ensure we're moving the stop in the right direction
                if (direction > 0 and new_stop > current_stop) or (direction < 0 and new_stop < current_stop):
                    position["stop_loss"] = new_stop
                    position["stop_type"] = "breakeven"
        
        # Time-based exits
        if self.parameters["use_time_based_exit"] and hours_in_trade > self.parameters["max_trade_duration_hours"]:
            position["exit_pending"] = True
            position["exit_reason"] = "time_exit"
            
        # Partial profit taking
        if self.parameters["use_partial_exits"] and not position.get("partial_exit_taken", False):
            # Take partial profits at a specific R-multiple
            if r_multiple >= self.parameters["partial_exit_r_multiple"]:
                position["partial_exit_pending"] = True
                position["partial_exit_percentage"] = self.parameters["partial_exit_percentage"]
                
        return position
    
    def _check_exit_conditions(self, 
                            symbol: str, 
                            position: Dict[str, Any], 
                            current_data: pd.DataFrame,
                            current_time: datetime) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for the current position.
        
        Args:
            symbol: Currency pair
            position: Current position details
            current_data: Current market data
            current_time: Current time
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if current_data.empty:
            return False, ""
            
        current_price = current_data["close"].iloc[-1]
        direction = 1 if position["position_type"] == "buy" else -1
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # Check stop loss
        if (direction > 0 and current_price <= stop_loss) or (direction < 0 and current_price >= stop_loss):
            return True, "stop_loss"
            
        # Check take profit
        if (direction > 0 and current_price >= take_profit) or (direction < 0 and current_price <= take_profit):
            return True, "take_profit"
            
        # Check time-based exit or other pending exits
        if position.get("exit_pending", False):
            return True, position.get("exit_reason", "manual_exit")
            
        # Check for pattern-based reversal signals
        if self.parameters["use_pattern_based_exits"]:
            # Calculate momentum indicators
            momentum = self._calculate_momentum_indicators(current_data)
            
            # Exit longs on bearish momentum shifts or shorts on bullish momentum shifts
            if ((direction > 0 and momentum["momentum_signal"].endswith("bearish")) or
                (direction < 0 and momentum["momentum_signal"].endswith("bullish"))):
                
                # Only exit if the momentum is strong enough
                if momentum["momentum_signal"].startswith("strongly_"):
                    return True, "momentum_reversal"
        
        # No exit condition met        
        return False, ""
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: datetime):
        """
        Process new market data and generate signals.
        
        Args:
            data: Dictionary of DataFrames with symbol as key
            timestamp: Current timestamp
        """
        self.current_time = timestamp
        
        # Update internal state - store new data
        for symbol, df in data.items():
            if symbol in self.symbols:
                self.market_data[symbol] = df.copy()
        
        # Process active positions first
        positions_to_update = {}
        for position_id, position in self.active_positions.items():
            symbol = position["symbol"]
            
            # Skip if we don't have data for this symbol
            if symbol not in self.market_data:
                continue
                
            # Get current data for this symbol
            current_data = self.market_data[symbol]
            
            # Adjust stops and targets based on current market conditions
            updated_position = self._adjust_stops_and_targets(symbol, position, current_data, timestamp)
            
            # Check if exit conditions are met
            should_exit, exit_reason = self._check_exit_conditions(symbol, updated_position, current_data, timestamp)
            
            if should_exit:
                self.logger.info(f"Exit signal for {symbol} position {position_id}: {exit_reason}")
                
                # Create exit event
                exit_event = {
                    "type": "exit",
                    "position_id": position_id,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "price": current_data["close"].iloc[-1],
                    "reason": exit_reason,
                    "position": updated_position
                }
                
                # Emit exit event
                self.events.append(exit_event)
                
                # Remove from active positions
                # We don't actually remove it here, we'll do it in update() to avoid modifying while iterating
                positions_to_update[position_id] = {"status": "pending_exit", "position": updated_position}
            elif updated_position.get("partial_exit_pending", False):
                # Create partial exit event
                exit_event = {
                    "type": "partial_exit",
                    "position_id": position_id,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "price": current_data["close"].iloc[-1],
                    "percentage": updated_position["partial_exit_percentage"],
                    "reason": "partial_profit_taking",
                    "position": updated_position
                }
                
                # Emit partial exit event
                self.events.append(exit_event)
                
                # Mark partial exit as taken
                updated_position["partial_exit_pending"] = False
                updated_position["partial_exit_taken"] = True
                positions_to_update[position_id] = {"status": "active", "position": updated_position}
            else:
                # Just update the position with new stops/targets
                positions_to_update[position_id] = {"status": "active", "position": updated_position}
        
        # Apply position updates
        for position_id, update_info in positions_to_update.items():
            if update_info["status"] == "pending_exit":
                # Will be removed in update()
                self.positions_to_close.append(position_id)
            else:
                self.active_positions[position_id] = update_info["position"]
        
        # Check for new entry signals
        account_info = {"balance": self.account_balance, "currency": self.account_currency}
        
        for symbol, df in self.market_data.items():
            # Skip if we're already at max positions
            if len(self.active_positions) >= self.parameters["max_concurrent_positions"]:
                break
                
            # Skip if we already have a position in this symbol
            if any(pos["symbol"] == symbol for pos in self.active_positions.values()):
                continue
                
            # Generate signal for this symbol
            signal = self._generate_signal(symbol, df, timestamp)
            
            if signal is not None:
                # Calculate position size
                position_size = self._calculate_position_size(signal, account_info)
                
                # Create a new position
                position_id = f"swing_{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}"
                
                position = {
                    "id": position_id,
                    "symbol": symbol,
                    "position_type": signal["signal"],  # "buy" or "sell"
                    "entry_price": signal["entry_price"],
                    "entry_time": timestamp,
                    "stop_loss": signal["stop_loss"],
                    "initial_stop_loss": signal["stop_loss"],  # Keep track of initial stop for R-multiple calculations
                    "take_profit": signal["take_profit"],
                    "size": position_size,
                    "strategy": self.strategy_name,
                    "note": signal["note"],
                    "reward_risk_ratio": signal["reward_risk_ratio"]
                }
                
                # Create entry event
                entry_event = {
                    "type": "entry",
                    "position_id": position_id,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "price": signal["entry_price"],
                    "direction": signal["signal"],
                    "size": position_size,
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "reason": signal["note"],
                    "position": position
                }
                
                # Emit entry event
                self.events.append(entry_event)
                
                # Add to active positions
                self.active_positions[position_id] = position
                
                self.logger.info(f"New {signal['signal']} signal for {symbol}: {signal['note']}")
    
    def update(self):
        """
        Process any pending events and update strategy state.
        """
        # Process any positions that need to be closed
        for position_id in self.positions_to_close:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                
                # Calculate profit/loss for the position
                entry_price = position["entry_price"]
                symbol = position["symbol"]
                
                # Use the most recent close price as exit price
                exit_price = None
                if symbol in self.market_data:
                    df = self.market_data[symbol]
                    if not df.empty:
                        exit_price = df["close"].iloc[-1]
                
                if exit_price is not None:
                    direction = 1 if position["position_type"] == "buy" else -1
                    profit_pips = direction * (exit_price - entry_price) * 10000  # Convert to pips
                    
                    # Update performance metrics
                    self.trades_count += 1
                    if profit_pips > 0:
                        self.winning_trades += 1
                    self.total_profit_pips += profit_pips
                    
                    # Calculate R-multiple
                    initial_risk = abs(entry_price - position["initial_stop_loss"])
                    r_multiple = direction * (exit_price - entry_price) / initial_risk if initial_risk > 0 else 0
                    self.r_multiples.append(r_multiple)
                    
                    self.logger.info(f"Closed position {position_id} with {profit_pips:.1f} pips profit (R = {r_multiple:.2f})")
                
                # Remove from active positions
                del self.active_positions[position_id]
        
        # Clear list of positions to close
        self.positions_to_close = []
        
        # Calculate and log current performance metrics
        if self.trades_count > 0 and len(self.r_multiples) > 0:
            win_rate = self.winning_trades / self.trades_count * 100
            avg_r = sum(self.r_multiples) / len(self.r_multiples)
            
            if self.current_time.hour % 6 == 0 and self.current_time.minute == 0:
                self.logger.info(f"Performance metrics - Win rate: {win_rate:.1f}%, "
                               f"Avg R: {avg_r:.2f}, "
                               f"Total profit: {self.total_profit_pips:.1f} pips, "
                               f"Trades: {self.trades_count}")
    
    def shutdown(self):
        """
        Clean up resources and save state when shutting down.
        """
        # Log final performance metrics
        if self.trades_count > 0 and len(self.r_multiples) > 0:
            win_rate = self.winning_trades / self.trades_count * 100
            avg_r = sum(self.r_multiples) / len(self.r_multiples)
            
            self.logger.info(f"Final performance metrics - Win rate: {win_rate:.1f}%, "
                           f"Avg R: {avg_r:.2f}, "
                           f"Total profit: {self.total_profit_pips:.1f} pips, "
                           f"Trades: {self.trades_count}")
            
        # Close any open positions if requested
        if self.parameters["close_positions_on_shutdown"] and self.active_positions:
            self.logger.info(f"Closing {len(self.active_positions)} positions on shutdown")
            
            for position_id, position in self.active_positions.items():
                self.positions_to_close.append(position_id)
                
            # Process the closing
            self.update()
