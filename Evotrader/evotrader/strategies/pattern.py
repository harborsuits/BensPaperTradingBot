"""Pattern-based strategy implementations."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from enum import Enum

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicators import sma


class PatternType(Enum):
    """Enum representing pattern types."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"


class HeadAndShouldersStrategy(Strategy):
    """
    Head and Shoulders pattern detection strategy.
    
    Detects regular and inverse head and shoulders patterns and generates
    trading signals upon pattern completion and breakout.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="lookback_period",
                default_value=30,
                min_value=20,
                max_value=100,
                step=5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="peak_distance",
                default_value=3,
                min_value=2,
                max_value=10,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="height_threshold",
                default_value=1.5,  # percentage
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="volume_confirmation",
                default_value=True,
                is_mutable=True
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=10.0,
                min_value=1.0,
                max_value=30.0,
                step=1.0,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="breakout_confirmation",
                default_value=1,
                min_value=1,
                max_value=3,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            )
        ]
        
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with parameters."""
        super().__init__(strategy_id, parameters)
        self.logger = logging.getLogger(f"evotrader.strategy.{self.strategy_id}")
        
        # Position tracking
        self.current_positions = {}
        
        # Price history
        self.price_history = {}  # symbol -> list of prices
        self.high_history = {}   # symbol -> list of highs
        self.low_history = {}    # symbol -> list of lows
        self.volume_history = {}  # symbol -> list of volumes
        
        # Pattern state tracking
        self.detected_patterns = {}  # symbol -> {pattern_type, pattern_data}
        self.necklines = {}  # symbol -> neckline price
        self.breakout_counters = {}  # symbol -> consecutive breakout days
        
    def _find_peaks_and_troughs(self, data: List[float], distance: int) -> Tuple[List[int], List[int]]:
        """
        Find peaks and troughs in price data.
        
        Args:
            data: Price data series
            distance: Minimum distance between peaks/troughs
            
        Returns:
            Tuple of (peak_indices, trough_indices)
        """
        peaks = []
        troughs = []
        
        if len(data) < (2 * distance + 1):
            return peaks, troughs
            
        for i in range(distance, len(data) - distance):
            # Check if this point is a peak
            is_peak = True
            for j in range(i - distance, i):
                if data[j] >= data[i]:
                    is_peak = False
                    break
            
            if is_peak:
                for j in range(i + 1, i + distance + 1):
                    if data[j] >= data[i]:
                        is_peak = False
                        break
                        
            if is_peak:
                peaks.append(i)
                continue
                
            # Check if this point is a trough
            is_trough = True
            for j in range(i - distance, i):
                if data[j] <= data[i]:
                    is_trough = False
                    break
            
            if is_trough:
                for j in range(i + 1, i + distance + 1):
                    if data[j] <= data[i]:
                        is_trough = False
                        break
                        
            if is_trough:
                troughs.append(i)
                
        return peaks, troughs
        
    def _detect_head_and_shoulders(self, prices: List[float], highs: List[float], lows: List[float], 
                                 volumes: List[float], peak_distance: int, height_threshold_pct: float) -> Dict[str, Any]:
        """
        Detect regular and inverse head and shoulders patterns.
        
        Args:
            prices: List of closing prices
            highs: List of high prices
            lows: List of low prices
            volumes: List of volume data
            peak_distance: Minimum distance between pattern peaks
            height_threshold_pct: Minimum height as percentage of price
            
        Returns:
            Dictionary with pattern information if detected, otherwise empty
        """
        # Find peaks and troughs
        peak_indices, trough_indices = self._find_peaks_and_troughs(prices, peak_distance)
        
        if len(peak_indices) < 3 or len(trough_indices) < 2:
            return {}
            
        # Try to detect regular H&S (bearish) pattern
        # We need 3 peaks with the middle peak (head) higher than the others
        for i in range(len(peak_indices) - 2):
            # Get potential shoulder-head-shoulder peaks
            left_shoulder_idx = peak_indices[i]
            head_idx = peak_indices[i + 1]
            right_shoulder_idx = peak_indices[i + 2]
            
            # Check if the middle peak is higher
            if (prices[head_idx] > prices[left_shoulder_idx] and 
                prices[head_idx] > prices[right_shoulder_idx]):
                
                # Check if shoulders are at approximately the same height (within 5%)
                shoulder_height_diff = abs(prices[left_shoulder_idx] - prices[right_shoulder_idx])
                avg_shoulder_height = (prices[left_shoulder_idx] + prices[right_shoulder_idx]) / 2
                
                if shoulder_height_diff < (avg_shoulder_height * 0.05):
                    # Check if the pattern is significant (head is higher than shoulders by threshold)
                    head_height = prices[head_idx]
                    min_shoulder_height = min(prices[left_shoulder_idx], prices[right_shoulder_idx])
                    height_diff_pct = (head_height - min_shoulder_height) / min_shoulder_height * 100
                    
                    if height_diff_pct >= height_threshold_pct:
                        # Find neckline (connecting the troughs between shoulders and head)
                        # Find troughs between peaks
                        left_trough_idx = None
                        right_trough_idx = None
                        
                        for trough in trough_indices:
                            if left_shoulder_idx < trough < head_idx:
                                left_trough_idx = trough
                                break
                                
                        for trough in reversed(trough_indices):
                            if head_idx < trough < right_shoulder_idx:
                                right_trough_idx = trough
                                break
                                
                        if left_trough_idx is not None and right_trough_idx is not None:
                            # Calculate neckline as line between two troughs
                            left_trough_price = prices[left_trough_idx]
                            right_trough_price = prices[right_trough_idx]
                            
                            # Neckline at the right shoulder position (for breakout determination)
                            x1, y1 = left_trough_idx, left_trough_price
                            x2, y2 = right_trough_idx, right_trough_price
                            
                            # Extend neckline to current bar
                            if x2 != x1:  # Avoid division by zero
                                slope = (y2 - y1) / (x2 - x1)
                                neckline_at_end = y2 + slope * (len(prices) - 1 - x2)
                            else:
                                neckline_at_end = y2
                                
                            return {
                                "pattern_type": PatternType.HEAD_AND_SHOULDERS,
                                "left_shoulder_idx": left_shoulder_idx,
                                "head_idx": head_idx,
                                "right_shoulder_idx": right_shoulder_idx,
                                "left_trough_idx": left_trough_idx,
                                "right_trough_idx": right_trough_idx,
                                "neckline_current": neckline_at_end,
                                "completion_idx": right_shoulder_idx
                            }
        
        # Try to detect inverse H&S (bullish) pattern
        # We need 3 troughs with the middle trough (head) lower than the others
        for i in range(len(trough_indices) - 2):
            # Get potential shoulder-head-shoulder troughs
            left_shoulder_idx = trough_indices[i]
            head_idx = trough_indices[i + 1]
            right_shoulder_idx = trough_indices[i + 2]
            
            # Check if the middle trough is lower
            if (prices[head_idx] < prices[left_shoulder_idx] and 
                prices[head_idx] < prices[right_shoulder_idx]):
                
                # Check if shoulders are at approximately the same height (within 5%)
                shoulder_height_diff = abs(prices[left_shoulder_idx] - prices[right_shoulder_idx])
                avg_shoulder_height = (prices[left_shoulder_idx] + prices[right_shoulder_idx]) / 2
                
                if shoulder_height_diff < (avg_shoulder_height * 0.05):
                    # Check if the pattern is significant (head is lower than shoulders by threshold)
                    head_height = prices[head_idx]
                    max_shoulder_height = max(prices[left_shoulder_idx], prices[right_shoulder_idx])
                    height_diff_pct = (max_shoulder_height - head_height) / max_shoulder_height * 100
                    
                    if height_diff_pct >= height_threshold_pct:
                        # Find neckline (connecting the peaks between shoulders and head)
                        # Find peaks between troughs
                        left_peak_idx = None
                        right_peak_idx = None
                        
                        for peak in peak_indices:
                            if left_shoulder_idx < peak < head_idx:
                                left_peak_idx = peak
                                break
                                
                        for peak in reversed(peak_indices):
                            if head_idx < peak < right_shoulder_idx:
                                right_peak_idx = peak
                                break
                                
                        if left_peak_idx is not None and right_peak_idx is not None:
                            # Calculate neckline as line between two peaks
                            left_peak_price = prices[left_peak_idx]
                            right_peak_price = prices[right_peak_idx]
                            
                            # Neckline at the right shoulder position (for breakout determination)
                            x1, y1 = left_peak_idx, left_peak_price
                            x2, y2 = right_peak_idx, right_peak_price
                            
                            # Extend neckline to current bar
                            if x2 != x1:  # Avoid division by zero
                                slope = (y2 - y1) / (x2 - x1)
                                neckline_at_end = y2 + slope * (len(prices) - 1 - x2)
                            else:
                                neckline_at_end = y2
                                
                            return {
                                "pattern_type": PatternType.INVERSE_HEAD_AND_SHOULDERS,
                                "left_shoulder_idx": left_shoulder_idx,
                                "head_idx": head_idx,
                                "right_shoulder_idx": right_shoulder_idx,
                                "left_peak_idx": left_peak_idx,
                                "right_peak_idx": right_peak_idx,
                                "neckline_current": neckline_at_end,
                                "completion_idx": right_shoulder_idx
                            }
        
        return {}
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on head and shoulders patterns.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        lookback_period = self.parameters["lookback_period"]
        peak_distance = self.parameters["peak_distance"]
        height_threshold = self.parameters["height_threshold"]
        volume_confirmation = self.parameters["volume_confirmation"]
        risk_percent = self.parameters["risk_percent"]
        breakout_confirmation = self.parameters["breakout_confirmation"]
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Extract current price data
            current_price = data.get("price", 0)
            current_high = data.get("high", current_price)
            current_low = data.get("low", current_price)
            current_volume = data.get("volume", 0)
            
            # Skip if invalid data
            if current_price <= 0:
                continue
                
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.high_history[symbol] = []
                self.low_history[symbol] = []
                self.volume_history[symbol] = []
                self.breakout_counters[symbol] = 0
                
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            self.volume_history[symbol].append(current_volume)
            
            # Limit history size
            max_history = max(lookback_period * 2, 100)
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            if len(self.price_history[symbol]) < lookback_period:
                continue
                
            # Detect patterns on recent data
            prices = self.price_history[symbol][-lookback_period:]
            highs = self.high_history[symbol][-lookback_period:]
            lows = self.low_history[symbol][-lookback_period:]
            volumes = self.volume_history[symbol][-lookback_period:]
            
            pattern_info = self._detect_head_and_shoulders(
                prices, highs, lows, volumes, peak_distance, height_threshold
            )
            
            # Check if we have a pattern
            if pattern_info:
                pattern_type = pattern_info["pattern_type"]
                neckline_price = pattern_info["neckline_current"]
                
                # Update pattern info
                self.detected_patterns[symbol] = pattern_info
                self.necklines[symbol] = neckline_price
                
                # Determine if this is a fresh pattern completion
                is_new_pattern = True
                if symbol in self.detected_patterns:
                    old_pattern = self.detected_patterns.get(symbol, {})
                    if old_pattern.get("completion_idx") == pattern_info["completion_idx"]:
                        is_new_pattern = False
                
                # Check for breakout
                is_breakout = False
                breakout_direction = None
                
                if pattern_type == PatternType.HEAD_AND_SHOULDERS:
                    # Bearish pattern - breakout is when price breaks below neckline
                    if current_price < neckline_price:
                        is_breakout = True
                        breakout_direction = "bearish"
                        self.breakout_counters[symbol] += 1
                    else:
                        self.breakout_counters[symbol] = 0
                
                elif pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS:
                    # Bullish pattern - breakout is when price breaks above neckline
                    if current_price > neckline_price:
                        is_breakout = True
                        breakout_direction = "bullish"
                        self.breakout_counters[symbol] += 1
                    else:
                        self.breakout_counters[symbol] = 0
                
                # Check for confirmed breakout
                is_confirmed = self.breakout_counters[symbol] >= breakout_confirmation
                
                # Check volume confirmation if required
                volume_confirmed = True
                if volume_confirmation and len(volumes) > 5:
                    avg_volume = sum(volumes[-6:-1]) / 5
                    volume_confirmed = current_volume > avg_volume * 1.2
                
                # Generate signal if we have a confirmed breakout
                current_position = symbol in self.current_positions
                
                if is_breakout and is_confirmed and volume_confirmed:
                    if breakout_direction == "bullish" and not current_position:
                        # Calculate target based on pattern height
                        pattern_height = abs(prices[pattern_info["head_idx"]] - neckline_price)
                        price_target = neckline_price + pattern_height
                        stop_loss = min(lows[-5:]) * 0.99  # Just below recent lows
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.8,
                            reason=f"Inverse Head & Shoulders breakout at ${current_price:.2f}",
                            params={
                                "risk_percent": risk_percent,
                                "entry_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit": price_target,
                                "pattern_type": "inverse_head_and_shoulders",
                                "neckline": neckline_price
                            }
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated BUY signal for {symbol} on inverse H&S breakout at {current_price:.2f}")
                        
                    elif breakout_direction == "bearish" and current_position:
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.8,
                            reason=f"Head & Shoulders breakdown at ${current_price:.2f}",
                            params={
                                "risk_percent": risk_percent,
                                "pattern_type": "head_and_shoulders",
                                "neckline": neckline_price
                            }
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated SELL signal for {symbol} on H&S breakdown at {current_price:.2f}")
            
            # Exit signals - check for stop loss and take profit
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                
                if "stop_loss" in position and current_price <= position["stop_loss"]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.9,
                        reason=f"Stop loss triggered at ${current_price:.2f}",
                        params={"risk_percent": risk_percent}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on stop loss at {current_price:.2f}")
                
                elif "take_profit" in position and current_price >= position["take_profit"]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        reason=f"Take profit target reached at ${current_price:.2f}",
                        params={"risk_percent": risk_percent}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on take profit at {current_price:.2f}")
        
        return signals
        
    def on_order_filled(self, order_data: Dict[str, Any]) -> None:
        """Update strategy state when an order is filled."""
        order = order_data.get("order")
        if not order:
            return
            
        symbol = order.symbol
        side = order.side
        
        # Track positions
        if str(side) == "buy":
            # Save important parameters
            params = {}
            signal_params = order_data.get("signal_params", {})
            
            if "stop_loss" in signal_params:
                params["stop_loss"] = signal_params["stop_loss"]
            if "take_profit" in signal_params:
                params["take_profit"] = signal_params["take_profit"]
            if "pattern_type" in signal_params:
                params["pattern_type"] = signal_params["pattern_type"]
                
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity,
                **params
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
