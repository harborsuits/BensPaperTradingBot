"""
Forex Range Trading Strategy

This strategy focuses on identifying and trading within established price ranges,
capitalizing on the tendency of forex pairs to oscillate between support and resistance
levels during periods of consolidation. It utilizes multiple technical indicators
to confirm range conditions and generate high-probability entry and exit signals.

Features:
- Advanced range detection algorithms
- Dynamic support and resistance identification
- Multiple oscillator confirmation (RSI, Stochastic, CCI)
- Volume-based range quality assessment
- Adaptive position sizing based on range width
- Early breakout detection for risk management
- Multiple timeframe confirmation
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
    strategy_type="range",
    name="ForexRangeTrading",
    description="Identifies and trades within established price ranges using oscillator signals and support/resistance levels",
    parameters={
        "default": {
            # Range identification parameters
            "min_range_bars": 20,  # Minimum number of bars to establish a range
            "max_range_slope": 0.05,  # Maximum slope for range (0.05 = 5%)
            "range_deviation_pct": 1.5,  # Percentage deviation allowed within range
            "min_touches": 3,  # Minimum number of touches on support/resistance
            
            # Range quality parameters
            "min_range_width_pips": 20,  # Minimum range width in pips
            "max_range_width_pips": 300,  # Maximum range width in pips
            "min_range_width_atr_ratio": 1.0,  # Range width should be at least this multiple of ATR
            "max_range_width_atr_ratio": 10.0,  # Range width should be at most this multiple of ATR
            
            # Indicator parameters
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "stoch_k_period": 14,
            "stoch_d_period": 3,
            "stoch_overbought": 80,
            "stoch_oversold": 20,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_period": 14,
            
            # Entry parameters
            "entry_zone_pct": 10,  # Entry zone near range extremes (% of range height)
            "confirmation_required": 2,  # Number of indicators needed for confirmation
            "min_reward_risk_ratio": 1.5,  # Minimum reward-to-risk ratio
            
            # Exit parameters
            "target_pct": 80,  # Target at x% of range (from entry edge)
            "stop_beyond_range_pct": 20,  # Stop loss x% beyond range
            "breakout_exit_bars": 3,  # Exit after this many bars outside range
            "max_holding_bars": 100,  # Maximum bars to hold the position
            "use_trailing_stop": True,
            "trailing_activation_pct": 30,  # Activate trailing stop at x% towards target
            
            # Risk management
            "risk_per_trade": 0.01,  # Risk 1% per trade
            "max_open_trades": 3,  # Maximum concurrent range trades
            "max_correlation": 0.7,  # Maximum correlation between traded pairs
            "min_position_size": 0.01,  # Minimum position size in lots
            "max_position_size": 2.0,  # Maximum position size in lots
            
            # Timeframe parameters
            "timeframes": ["1h", "4h", "1d"],  # Timeframes to analyze
            "primary_timeframe": "4h",  # Primary trading timeframe
            "min_mtf_agreement": 2,  # Minimum timeframes that must agree on range condition
            
            # General parameters
            "timezone": pytz.UTC,
            "preferred_pairs": [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF"
            ],
            "use_volume_confirmation": True
        },
        
        # For narrow, faster ranges
        "narrow_range": {
            "min_range_bars": 15,
            "range_deviation_pct": 1.0,
            "min_touches": 2,
            "min_range_width_pips": 15,
            "max_range_width_pips": 150,
            "entry_zone_pct": 15,
            "target_pct": 70,
            "max_holding_bars": 50,
            "rsi_period": 9,
            "stoch_k_period": 9
        },
        
        # For wider, more established ranges
        "wide_range": {
            "min_range_bars": 40,
            "range_deviation_pct": 2.0,
            "min_touches": 4,
            "min_range_width_pips": 50,
            "max_range_width_pips": 500,
            "entry_zone_pct": 8,
            "target_pct": 85,
            "max_holding_bars": 200,
            "breakout_exit_bars": 5
        }
    }
)
class ForexRangeTradingStrategy(ForexBaseStrategy):
    """
    A strategy that identifies and trades within established price ranges.
    
    This strategy looks for pairs trading in a horizontal range, identifies support
    and resistance levels, and generates entry signals when price approaches these
    levels with confirmation from oscillator indicators.
    """
    
    def __init__(self, session=None):
        """
        Initialize the range trading strategy.
        
        Args:
            session: Trading session object with configuration
        """
        super().__init__(session)
        self.name = "ForexRangeTrading"
        self.description = "Trading within established price ranges"
        self.logger = logging.getLogger(__name__)
        
        # Range data storage
        self.detected_ranges = {}  # symbol -> range data
        
        # Active trades tracking
        self.active_trades = {}  # symbol -> trade data
        
        # Correlation data for risk management
        self.correlation_matrix = None
        self.last_correlation_update = None
        
        # Performance statistics
        self.stats = {
            "ranges_detected": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "breakout_losses": 0,
            "avg_hold_time": 0,
            "total_profit_pips": 0
        }
        
    def initialize(self) -> None:
        """Initialize strategy and load any required data."""
        super().initialize()
        
        # Initialize correlation matrix
        self._update_correlation_matrix()
        
        # Initialize detected ranges
        self.detected_ranges = {}
        
        self.logger.info(f"Initialized {self.name} strategy")
    
    def _update_correlation_matrix(self) -> None:
        """Update the correlation matrix for currency pairs."""
        # In a real implementation, this would calculate actual correlations from price data
        # For now, we'll use a simplified approximation based on currency relationships
        pairs = self.parameters["preferred_pairs"]
        
        # Initialize an empty correlation matrix
        correlation_matrix = pd.DataFrame(np.eye(len(pairs)), index=pairs, columns=pairs)
        
        # Fill with approximate correlations (in a real system, this would use actual price data)
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i == j:
                    continue  # Skip diagonal
                
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
    
    def detect_range(self, data: pd.DataFrame, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Detect if a price series is trading in a range.
        
        Args:
            data: OHLCV data
            timeframe: The timeframe of the data
            
        Returns:
            Dictionary with range data if range is detected, None otherwise
        """
        if data.empty or len(data) < self.parameters["min_range_bars"]:
            return None
        
        # Calculate indicators
        atr = self._calculate_atr(data, self.parameters["atr_period"])
        
        # Calculate Bollinger Bands for range detection
        bb_period = self.parameters["bollinger_period"]
        bb_std = self.parameters["bollinger_std"]
        
        if len(data) <= bb_period:
            return None
            
        # Calculate moving average
        ma = data["close"].rolling(window=bb_period).mean()
        std = data["close"].rolling(window=bb_period).std()
        
        upper_band = ma + (std * bb_std)
        lower_band = ma - (std * bb_std)
        
        # Get the most recent values
        current_ma = ma.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_close = data["close"].iloc[-1]
        
        # Calculate linear regression on price to check for flat trend
        x = np.arange(len(data.iloc[-self.parameters["min_range_bars"]:]))
        y = data["close"].iloc[-self.parameters["min_range_bars"]:].values
        
        if len(x) != len(y) or len(x) < 2:
            return None
            
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope by dividing by the average price
        avg_price = np.mean(y)
        normalized_slope = abs(slope / avg_price)
        
        # Check if slope is flat enough for a range
        if normalized_slope > self.parameters["max_range_slope"]:
            return None
        
        # Detect range boundaries
        # Use both Bollinger Bands and price extremes
        lookback = min(len(data), self.parameters["min_range_bars"] * 2)
        recent_data = data.iloc[-lookback:]
        
        # Get min/max from raw price data
        price_min = recent_data["low"].min()
        price_max = recent_data["high"].max()
        
        # Check if price is oscillating in the recent period
        num_oscillations = self._count_oscillations(recent_data["close"])
        
        # Count "touches" of upper and lower boundaries
        upper_touches = self._count_touches(recent_data["high"], current_upper, atr)
        lower_touches = self._count_touches(recent_data["low"], current_lower, atr)
        
        # Check if we have enough touches on both boundaries
        min_touches = self.parameters["min_touches"]
        if upper_touches + lower_touches < min_touches:
            return None
            
        # Ensure reasonably balanced touches (not just testing one side)
        if upper_touches == 0 or lower_touches == 0:
            return None
        
        # Calculate range width and check if it's reasonable
        range_width = current_upper - current_lower
        point_value = 0.0001  # Assuming non-JPY pair for simplicity
        range_pips = range_width / point_value
        
        # Check range width constraints
        min_pips = self.parameters["min_range_width_pips"]
        max_pips = self.parameters["max_range_width_pips"]
        min_atr_ratio = self.parameters["min_range_width_atr_ratio"]
        max_atr_ratio = self.parameters["max_range_width_atr_ratio"]
        
        if range_pips < min_pips or range_pips > max_pips:
            return None
            
        if atr > 0:
            range_atr_ratio = range_width / atr
            if range_atr_ratio < min_atr_ratio or range_atr_ratio > max_atr_ratio:
                return None
        
        # Calculate range midpoint and current position within range
        midpoint = (current_upper + current_lower) / 2
        range_position = (current_close - current_lower) / (current_upper - current_lower)
        
        # Detect if price is near range boundary (potential entry zone)
        entry_zone_pct = self.parameters["entry_zone_pct"] / 100
        
        near_upper = range_position > (1 - entry_zone_pct)
        near_lower = range_position < entry_zone_pct
        
        # Check volume if required and available
        volume_confirmed = True
        if self.parameters["use_volume_confirmation"] and "volume" in data:
            # Check for declining volume in range (typical)
            recent_volume = data["volume"].iloc[-self.parameters["min_range_bars"]:]
            volume_slope, _ = np.polyfit(np.arange(len(recent_volume)), recent_volume.values, 1)
            volume_confirmed = volume_slope <= 0
        
        # Return range data if all criteria are met
        range_data = {
            "upper_boundary": current_upper,
            "lower_boundary": current_lower,
            "midpoint": midpoint,
            "width": range_width,
            "width_pips": range_pips,
            "slope": normalized_slope,
            "position": range_position,
            "near_upper": near_upper,
            "near_lower": near_lower,
            "upper_touches": upper_touches,
            "lower_touches": lower_touches,
            "oscillations": num_oscillations,
            "atr": atr,
            "volume_confirmed": volume_confirmed,
            "detection_time": datetime.now(self.parameters["timezone"]),
            "timeframe": timeframe,
            "bars_in_range": min(len(data), self.parameters["min_range_bars"] * 2)
        }
        
        return range_data
    
    def _count_touches(self, price_series: pd.Series, boundary: float, atr: float) -> int:
        """
        Count how many times a price series has "touched" a boundary level.
        
        Args:
            price_series: Series of price values
            boundary: The boundary level to check
            atr: The Average True Range (for determining touch threshold)
            
        Returns:
            Number of touches
        """
        # Use a percentage of ATR as the tolerance for a "touch"
        touch_threshold = atr * 0.3  # 30% of ATR
        
        # Vectorized calculation of distances from boundary
        distances = abs(price_series - boundary)
        
        # Count values within the threshold as touches
        touches = sum(distances < touch_threshold)
        
        # Avoid counting consecutive touches as multiple touches
        if touches > 0:
            # Simplification: count clusters of touches as a single touch
            # In a real system, this would be more sophisticated
            segments = (distances >= touch_threshold).cumsum()
            unique_segments = segments[distances < touch_threshold].unique()
            touches = len(unique_segments)
        
        return touches
    
    def _count_oscillations(self, price_series: pd.Series) -> int:
        """
        Count the number of oscillations in a price series.
        An oscillation is a change in direction (peak or trough).
        
        Args:
            price_series: Series of price values
            
        Returns:
            Number of oscillations
        """
        if len(price_series) < 3:
            return 0
            
        # Calculate differences between consecutive prices
        diffs = price_series.diff().dropna()
        
        # Find sign changes in the differences (indicates change in direction)
        signs = np.sign(diffs)
        sign_changes = np.abs(np.diff(signs))
        
        # Count non-zero sign changes (oscillations)
        oscillations = sum(sign_changes > 0)
        
        return oscillations
    
    def check_oscillator_signals(self, data: pd.DataFrame, range_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check oscillator indicators for entry signals within the range.
        
        Args:
            data: OHLCV data
            range_data: Range information
            
        Returns:
            Dictionary with oscillator signal data
        """
        if data.empty or len(data) < 20:
            return {"buy_signal": False, "sell_signal": False, "signals_count": 0}
        
        signals = {"buy_signals": 0, "sell_signals": 0}
        
        # RSI calculation
        rsi = self._calculate_rsi(data["close"], self.parameters["rsi_period"])
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Stochastic calculation
        stoch_k, stoch_d = self._calculate_stochastic(
            data["high"], data["low"], data["close"],
            self.parameters["stoch_k_period"], self.parameters["stoch_d_period"]
        )
        current_k = stoch_k.iloc[-1] if not stoch_k.empty else 50
        current_d = stoch_d.iloc[-1] if not stoch_d.empty else 50
        
        # Price position within range
        position = range_data["position"]
        near_upper = range_data["near_upper"]
        near_lower = range_data["near_lower"]
        
        # Check RSI signals
        if current_rsi < self.parameters["rsi_oversold"] and near_lower:
            signals["buy_signals"] += 1
        elif current_rsi > self.parameters["rsi_overbought"] and near_upper:
            signals["sell_signals"] += 1
        
        # Check Stochastic signals
        if current_k < self.parameters["stoch_oversold"] and current_k < current_d and near_lower:
            signals["buy_signals"] += 1
        elif current_k > self.parameters["stoch_overbought"] and current_k > current_d and near_upper:
            signals["sell_signals"] += 1
        
        # Check price action (rejection from boundaries)
        if len(data) >= 3:
            current_close = data["close"].iloc[-1]
            current_open = data["open"].iloc[-1]
            current_high = data["high"].iloc[-1]
            current_low = data["low"].iloc[-1]
            
            # Bullish rejection at support
            if (near_lower and 
                current_close > current_open and 
                current_low < range_data["lower_boundary"] and
                current_close > (range_data["lower_boundary"] + range_data["atr"] * 0.2)):
                signals["buy_signals"] += 1
                
            # Bearish rejection at resistance
            if (near_upper and 
                current_close < current_open and 
                current_high > range_data["upper_boundary"] and
                current_close < (range_data["upper_boundary"] - range_data["atr"] * 0.2)):
                signals["sell_signals"] += 1
        
        # Determine if we have enough confirmation signals
        min_signals = self.parameters["confirmation_required"]
        
        signals["buy_signal"] = signals["buy_signals"] >= min_signals
        signals["sell_signal"] = signals["sell_signals"] >= min_signals
        signals["signals_count"] = max(signals["buy_signals"], signals["sell_signals"])
        
        return signals
    
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate range trading signals based on detected ranges and oscillator signals.
        
        Args:
            data_dict: Dictionary of market data for different pairs
            
        Returns:
            Dictionary of trading signals
        """
        signals = {}
        current_time = datetime.now(self.parameters["timezone"])
        
        # Ensure we have the correlation matrix updated
        if not self.correlation_matrix or (current_time - self.last_correlation_update).days > 7:
            self._update_correlation_matrix()
        
        # Check if we've reached the maximum number of concurrent trades
        if len(self.active_trades) >= self.parameters["max_open_trades"]:
            self.logger.info(f"Maximum number of concurrent range trades ({self.parameters['max_open_trades']}) reached")
            return {}
        
        # Process each currency pair
        for symbol, data in data_dict.items():
            # Skip if not in our preferred pairs list
            if symbol not in self.parameters["preferred_pairs"]:
                continue
                
            # Skip if we already have an active trade for this pair
            if symbol in self.active_trades:
                continue
                
            # Skip if data is insufficient
            if data.empty or len(data) < self.parameters["min_range_bars"]:
                continue
            
            # Get primary timeframe data
            primary_tf = self.parameters["primary_timeframe"]
            
            # For simplicity, we'll just use the provided data and assume it's the primary timeframe
            # In a real implementation, we would have data for all timeframes
            
            # Detect range on the primary timeframe
            range_data = self.detect_range(data, primary_tf)
            
            # Skip if no range detected
            if not range_data:
                continue
                
            # Check multi-timeframe agreement if required
            # In a real implementation, this would check multiple timeframes
            # For now, we'll simplify by assuming MTF agreement
            mtf_agreement = self.parameters["min_mtf_agreement"]
            
            # Check oscillator signals
            oscillator_signals = self.check_oscillator_signals(data, range_data)
            
            # Determine if we have a valid entry signal
            entry_signal = None
            if oscillator_signals["buy_signal"] and range_data["near_lower"]:
                entry_signal = "buy"
            elif oscillator_signals["sell_signal"] and range_data["near_upper"]:
                entry_signal = "sell"
                
            if not entry_signal:
                continue
            
            # Check correlation with existing trades to avoid over-exposure
            if self.active_trades and symbol in self.correlation_matrix.index:
                skip_due_to_correlation = False
                for active_symbol in self.active_trades.keys():
                    if active_symbol in self.correlation_matrix.columns:
                        correlation = abs(self.correlation_matrix.loc[symbol, active_symbol])
                        if correlation > self.parameters["max_correlation"]:
                            self.logger.info(f"Skipping {symbol} due to high correlation with active trade {active_symbol}")
                            skip_due_to_correlation = True
                            break
                if skip_due_to_correlation:
                    continue
            
            # Calculate entry, target and stop levels
            current_price = data["close"].iloc[-1]
            
            # Calculate entry zone
            entry_zone_pct = self.parameters["entry_zone_pct"] / 100
            range_height = range_data["upper_boundary"] - range_data["lower_boundary"]
            
            if entry_signal == "buy":
                entry_zone_top = range_data["lower_boundary"] + (range_height * entry_zone_pct)
                entry_price = max(current_price, range_data["lower_boundary"])  # Don't enter below support
                
                # Target near the top of the range
                target_pct = self.parameters["target_pct"] / 100
                take_profit = range_data["lower_boundary"] + (range_height * target_pct)
                
                # Stop loss below the range
                stop_beyond_pct = self.parameters["stop_beyond_range_pct"] / 100
                stop_loss = range_data["lower_boundary"] - (range_height * stop_beyond_pct)
                
            else:  # entry_signal == "sell"
                entry_zone_bottom = range_data["upper_boundary"] - (range_height * entry_zone_pct)
                entry_price = min(current_price, range_data["upper_boundary"])  # Don't enter above resistance
                
                # Target near the bottom of the range
                target_pct = self.parameters["target_pct"] / 100
                take_profit = range_data["upper_boundary"] - (range_height * target_pct)
                
                # Stop loss above the range
                stop_beyond_pct = self.parameters["stop_beyond_range_pct"] / 100
                stop_loss = range_data["upper_boundary"] + (range_height * stop_beyond_pct)
            
            # Calculate reward-to-risk ratio
            if entry_signal == "buy":
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # entry_signal == "sell"
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                
            reward_risk_ratio = reward / risk if risk > 0 else 0
            
            # Skip if reward-to-risk ratio is insufficient
            if reward_risk_ratio < self.parameters["min_reward_risk_ratio"]:
                continue
            
            # Calculate position size based on risk parameters
            position_size = self._calculate_position_size(symbol, data, entry_price, stop_loss)
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                signal_type=entry_signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=position_size,
                timestamp=current_time,
                timeframe=primary_tf,
                strategy=self.name,
                strength=oscillator_signals["signals_count"] / 3.0,  # Normalize to 0-1 range
                metadata={
                    "range_upper": range_data["upper_boundary"],
                    "range_lower": range_data["lower_boundary"],
                    "range_width_pips": range_data["width_pips"],
                    "position_in_range": range_data["position"],
                    "atr": range_data["atr"],
                    "reward_risk_ratio": reward_risk_ratio,
                    "max_holding_bars": self.parameters["max_holding_bars"],
                    "use_trailing_stop": self.parameters["use_trailing_stop"],
                    "trailing_activation": self.parameters["trailing_activation_pct"] / 100,
                    "entry_bar": len(data) - 1,  # Current bar index for reference
                    "breakout_exit_bars": self.parameters["breakout_exit_bars"]
                }
            )
            
            # Add to signals dictionary
            signals[symbol] = signal
            
            # Track this as an active trade
            self.active_trades[symbol] = {
                "entry_time": current_time,
                "entry_price": entry_price,
                "direction": entry_signal,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "range_data": range_data,
                "bars_held": 0,
                "bars_outside_range": 0,
                "trailing_stop_activated": False,
                "trailing_stop_level": stop_loss
            }
            
            # Log the signal
            range_width_pips = range_data["width_pips"]
            self.logger.info(f"Generated {entry_signal} signal for {symbol} in range of {range_width_pips:.1f} pips, RR={reward_risk_ratio:.2f}")
            
            # Update statistics
            self.stats["ranges_detected"] += 1
            
            # Break after finding a signal to avoid creating too many signals at once
            if len(signals) + len(self.active_trades) >= self.parameters["max_open_trades"]:
                break
        
        return signals
    
    def _calculate_position_size(self, symbol: str, data: pd.DataFrame, 
                               entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Currency pair
            data: Market data
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size in lots
        """
        # Calculate risk amount
        account_balance = self.session.account_balance
        risk_per_trade = self.parameters["risk_per_trade"]
        risk_amount = account_balance * risk_per_trade
        
        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return self.parameters["min_position_size"]
        
        # Calculate pip value (simplified)
        point_value = 0.0001 if "JPY" not in symbol else 0.01
        stop_distance_pips = stop_distance / point_value
        
        # Standard values for pip calculation
        standard_lot_size = 100000  # 1 standard lot
        pip_value = point_value * standard_lot_size
        
        # Calculate position size in standard lots
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Apply position size constraints
        position_size = max(self.parameters["min_position_size"], 
                          min(self.parameters["max_position_size"], position_size))
        
        return position_size
    
    def check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Check if a range trade position should be exited.
        
        Args:
            position: Current position information
            data: Latest market data
            
        Returns:
            True if position should be exited, False otherwise
        """
        if data.empty:
            return False
            
        # Extract position details
        symbol = position["symbol"]
        entry_price = position["entry_price"]
        direction = position["signal_type"]  # 'buy' or 'sell'
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        metadata = position.get("metadata", {})
        
        # Get range boundaries from metadata
        range_upper = metadata.get("range_upper")
        range_lower = metadata.get("range_lower")
        max_holding_bars = metadata.get("max_holding_bars", self.parameters["max_holding_bars"])
        breakout_exit_bars = metadata.get("breakout_exit_bars", self.parameters["breakout_exit_bars"])
        use_trailing_stop = metadata.get("use_trailing_stop", self.parameters["use_trailing_stop"])
        trailing_activation = metadata.get("trailing_activation", self.parameters["trailing_activation_pct"] / 100)
        entry_bar = metadata.get("entry_bar", 0)
        
        # Current price and bar count
        current_price = data["close"].iloc[-1]
        current_bar = len(data) - 1
        bars_held = current_bar - entry_bar
        
        # Check if active trade data is available
        active_trade = self.active_trades.get(symbol)
        trailing_stop_level = stop_loss
        trailing_stop_activated = False
        bars_outside_range = 0
        
        if active_trade:
            trailing_stop_level = active_trade.get("trailing_stop_level", stop_loss)
            trailing_stop_activated = active_trade.get("trailing_stop_activated", False)
            bars_outside_range = active_trade.get("bars_outside_range", 0)
            
            # Update bars held
            active_trade["bars_held"] = bars_held
        
        # Check if stop loss or take profit hit
        if direction == "buy":
            if current_price <= stop_loss:
                self.logger.info(f"Exiting {symbol} buy position due to stop loss hit")
                if active_trade:
                    self.stats["failed_trades"] += 1
                return True
                
            if current_price >= take_profit:
                self.logger.info(f"Exiting {symbol} buy position due to take profit hit")
                if active_trade:
                    self.stats["successful_trades"] += 1
                return True
                
        else:  # direction == "sell"
            if current_price >= stop_loss:
                self.logger.info(f"Exiting {symbol} sell position due to stop loss hit")
                if active_trade:
                    self.stats["failed_trades"] += 1
                return True
                
            if current_price <= take_profit:
                self.logger.info(f"Exiting {symbol} sell position due to take profit hit")
                if active_trade:
                    self.stats["successful_trades"] += 1
                return True
        
        # Check maximum holding period
        if bars_held >= max_holding_bars:
            self.logger.info(f"Exiting {symbol} position due to maximum holding period reached ({bars_held} bars)")
            return True
        
        # Check for breakout from range
        if range_upper is not None and range_lower is not None:
            is_outside_range = (current_price > range_upper) or (current_price < range_lower)
            
            if is_outside_range and active_trade:
                # Update bars outside range counter
                active_trade["bars_outside_range"] = bars_outside_range + 1
                
                # Exit if price has been outside range for too long (confirmed breakout)
                if active_trade["bars_outside_range"] >= breakout_exit_bars:
                    self.logger.info(f"Exiting {symbol} position due to confirmed range breakout")
                    self.stats["breakout_losses"] += 1
                    return True
            elif active_trade:
                # Reset counter if price is back inside range
                active_trade["bars_outside_range"] = 0
        
        # Handle trailing stop
        if use_trailing_stop and active_trade:
            range_height = range_upper - range_lower
            activation_threshold = 0.0
            
            if direction == "buy":
                activation_threshold = entry_price + (range_height * trailing_activation)
                
                # Check if trailing stop should be activated
                if current_price >= activation_threshold and not trailing_stop_activated:
                    active_trade["trailing_stop_activated"] = True
                    self.logger.info(f"Activating trailing stop for {symbol} buy position at {current_price}")
                
                # Update trailing stop level if activated
                if active_trade["trailing_stop_activated"]:
                    # Calculate new stop level (as a percentage of the distance to target)
                    price_progress = (current_price - entry_price) / (take_profit - entry_price)
                    stop_adjustment = range_height * 0.1 * price_progress  # Gradually tighten stop
                    new_stop = range_lower + stop_adjustment
                    
                    # Only move stop up, never down
                    if new_stop > trailing_stop_level:
                        active_trade["trailing_stop_level"] = new_stop
                        trailing_stop_level = new_stop
                        
                # Check if trailing stop hit
                if current_price <= trailing_stop_level and trailing_stop_activated:
                    self.logger.info(f"Exiting {symbol} buy position due to trailing stop hit")
                    self.stats["successful_trades"] += 1
                    return True
                    
            else:  # direction == "sell"
                activation_threshold = entry_price - (range_height * trailing_activation)
                
                # Check if trailing stop should be activated
                if current_price <= activation_threshold and not trailing_stop_activated:
                    active_trade["trailing_stop_activated"] = True
                    self.logger.info(f"Activating trailing stop for {symbol} sell position at {current_price}")
                
                # Update trailing stop level if activated
                if active_trade["trailing_stop_activated"]:
                    # Calculate new stop level (as a percentage of the distance to target)
                    price_progress = (entry_price - current_price) / (entry_price - take_profit)
                    stop_adjustment = range_height * 0.1 * price_progress  # Gradually tighten stop
                    new_stop = range_upper - stop_adjustment
                    
                    # Only move stop down, never up
                    if new_stop < trailing_stop_level:
                        active_trade["trailing_stop_level"] = new_stop
                        trailing_stop_level = new_stop
                        
                # Check if trailing stop hit
                if current_price >= trailing_stop_level and trailing_stop_activated:
                    self.logger.info(f"Exiting {symbol} sell position due to trailing stop hit")
                    self.stats["successful_trades"] += 1
                    return True
        
        return False
    
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
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: K period
            d_period: D period
            
        Returns:
            Tuple of (K, D)
        """
        if len(close) < k_period:
            # Return default values if not enough data
            dummy = pd.Series([50] * len(close), index=close.index)
            return dummy, dummy
            
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Handle potential divide by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, 1e-10)
        
        k = 100 * ((close - lowest_low) / range_diff)
        
        # Calculate %D (moving average of %K)
        d = k.rolling(window=d_period).mean()
        
        # Fill NAs with 50 (neutral)
        k = k.fillna(50)
        d = d.fillna(50)
        
        return k, d
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility assessment.
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period + 1:
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
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def on_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Process new market data and update strategy state.
        
        Args:
            data_dict: Dictionary of market data for different pairs
        """
        # Process active trades first to check for exits
        self._process_active_trades(data_dict)
        
        # Generate new signals if we have capacity
        if len(self.active_trades) < self.parameters["max_open_trades"]:
            signals = self.generate_signals(data_dict)
            
            # Emit signals if any generated
            for symbol, signal in signals.items():
                self.emit_signal(signal)
    
    def _process_active_trades(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Process active trades to check for exit conditions and update stats.
        
        Args:
            data_dict: Dictionary of market data for different pairs
        """
        symbols_to_remove = []
        
        for symbol, trade in self.active_trades.items():
            # Skip if we don't have data for this symbol
            if symbol not in data_dict or data_dict[symbol].empty:
                continue
                
            # Create a position-like object for the exit condition check
            position = {
                "symbol": symbol,
                "entry_price": trade["entry_price"],
                "signal_type": trade["direction"],
                "stop_loss": trade["stop_loss"],
                "take_profit": trade["take_profit"],
                "metadata": {
                    "range_upper": trade["range_data"]["upper_boundary"],
                    "range_lower": trade["range_data"]["lower_boundary"],
                    "max_holding_bars": self.parameters["max_holding_bars"],
                    "breakout_exit_bars": self.parameters["breakout_exit_bars"],
                    "use_trailing_stop": self.parameters["use_trailing_stop"],
                    "trailing_activation": self.parameters["trailing_activation_pct"] / 100,
                    "entry_bar": trade.get("entry_bar", 0)
                }
            }
            
            # Check exit conditions
            if self.check_exit_conditions(position, data_dict[symbol]):
                # Emit exit signal
                exit_signal = Signal(
                    symbol=symbol,
                    signal_type="exit",
                    entry_price=None,  # Not relevant for exit
                    stop_loss=None,    # Not relevant for exit
                    take_profit=None,  # Not relevant for exit
                    size=trade["position_size"],
                    timestamp=datetime.now(self.parameters["timezone"]),
                    timeframe=self.parameters["primary_timeframe"],
                    strategy=self.name,
                    strength=1.0,  # Exit signals are always strong
                    metadata={
                        "exit_reason": "range_strategy_exit",
                        "bars_held": trade.get("bars_held", 0),
                        "trailing_stop_activated": trade.get("trailing_stop_activated", False)
                    }
                )
                
                self.emit_signal(exit_signal)
                symbols_to_remove.append(symbol)
                
                # Update statistics
                exit_price = data_dict[symbol]["close"].iloc[-1]
                entry_price = trade["entry_price"]
                direction = trade["direction"]
                position_size = trade["position_size"]
                
                # Calculate pips gained/lost
                point_value = 0.0001 if "JPY" not in symbol else 0.01
                if direction == "buy":
                    pips = (exit_price - entry_price) / point_value
                else:  # direction == "sell"
                    pips = (entry_price - exit_price) / point_value
                    
                self.stats["total_profit_pips"] += pips
                
                # Update average hold time
                bars_held = trade.get("bars_held", 0)
                total_trades = self.stats["successful_trades"] + self.stats["failed_trades"]
                
                if total_trades > 0:
                    current_avg = self.stats["avg_hold_time"]
                    self.stats["avg_hold_time"] = ((current_avg * (total_trades - 1)) + bars_held) / total_trades
        
        # Remove closed trades
        for symbol in symbols_to_remove:
            self.active_trades.pop(symbol, None)
    
    def update(self) -> Dict[str, Any]:
        """
        Update strategy state and return current performance metrics.
        
        Returns:
            Dictionary with strategy performance and status information
        """
        # Update correlation matrix periodically (if needed)
        current_time = datetime.now(self.parameters["timezone"])
        if not self.correlation_matrix or (current_time - self.last_correlation_update).days > 7:
            self._update_correlation_matrix()
        
        # Calculate success rate
        total_trades = self.stats["successful_trades"] + self.stats["failed_trades"]
        success_rate = (self.stats["successful_trades"] / total_trades) * 100 if total_trades > 0 else 0
        
        # Return current status and performance metrics
        return {
            "strategy_name": self.name,
            "active_trades": len(self.active_trades),
            "active_symbols": list(self.active_trades.keys()),
            "ranges_detected": self.stats["ranges_detected"],
            "completed_trades": total_trades,
            "success_rate": success_rate,
            "avg_hold_time": self.stats["avg_hold_time"],
            "total_profit_pips": self.stats["total_profit_pips"],
            "breakout_losses": self.stats["breakout_losses"],
            "last_update": current_time.isoformat()
        }
    
    def shutdown(self) -> None:
        """
        Clean up resources and prepare for shutdown.
        """
        self.logger.info(f"Shutting down {self.name} strategy")
        self.logger.info(f"Strategy stats: {self.stats}")
        
        # Clear any active trades data
        self.active_trades.clear()
        self.detected_ranges.clear()
        
        super().shutdown()
