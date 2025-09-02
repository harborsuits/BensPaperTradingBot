"""Double top/bottom pattern strategies."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from enum import Enum

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from .pattern import PatternType


class DoubleTopBottomStrategy(Strategy):
    """
    Double Top and Double Bottom pattern detection strategy.
    
    Detects double top (bearish) and double bottom (bullish) patterns
    and generates trading signals upon pattern completion and breakout.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="lookback_period",
                default_value=30,
                min_value=15,
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
                name="peak_similarity",
                default_value=3.0,  # percentage
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="min_pattern_height",
                default_value=2.0,  # percentage
                min_value=0.5,
                max_value=10.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="confirmation_bars",
                default_value=1,
                min_value=1,
                max_value=3,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=10.0,
                min_value=1.0,
                max_value=30.0,
                step=1.0,
                is_mutable=True,
                mutation_factor=0.2
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
        
    def _detect_double_top_bottom(self, prices: List[float], highs: List[float], lows: List[float], 
                                peak_distance: int, peak_similarity_pct: float, min_height_pct: float) -> Dict[str, Any]:
        """
        Detect double top and double bottom patterns.
        
        Args:
            prices: List of closing prices
            highs: List of high prices
            lows: List of low prices
            peak_distance: Minimum distance between pattern peaks
            peak_similarity_pct: Maximum percentage difference between peaks
            min_height_pct: Minimum pattern height as percentage of price
            
        Returns:
            Dictionary with pattern information if detected, otherwise empty
        """
        # Find peaks and troughs
        peak_indices, trough_indices = self._find_peaks_and_troughs(prices, peak_distance)
        
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return {}
            
        # Try to detect Double Top (bearish) pattern
        # We need 2 peaks at similar levels with a trough between them
        for i in range(len(peak_indices) - 1):
            first_peak_idx = peak_indices[i]
            second_peak_idx = peak_indices[i + 1]
            
            # Check if peaks are similar in height (within similarity threshold)
            first_peak_price = prices[first_peak_idx]
            second_peak_price = prices[second_peak_idx]
            
            price_diff = abs(first_peak_price - second_peak_price)
            avg_price = (first_peak_price + second_peak_price) / 2
            price_diff_pct = (price_diff / avg_price) * 100
            
            # Ensure peaks are similar and at least one bar separates them
            if price_diff_pct <= peak_similarity_pct and (second_peak_idx - first_peak_idx) > 1:
                # Check if there's a trough between the peaks
                middle_trough_idx = None
                for trough in trough_indices:
                    if first_peak_idx < trough < second_peak_idx:
                        middle_trough_idx = trough
                        break
                
                if middle_trough_idx is not None:
                    # Confirm sufficient pattern height
                    trough_price = prices[middle_trough_idx]
                    peak_height = min(first_peak_price, second_peak_price) - trough_price
                    height_pct = (peak_height / trough_price) * 100
                    
                    if height_pct >= min_height_pct:
                        # Pattern height is significant
                        # Neckline is at the central trough level
                        neckline_price = trough_price
                        
                        return {
                            "pattern_type": PatternType.DOUBLE_TOP,
                            "first_peak_idx": first_peak_idx,
                            "second_peak_idx": second_peak_idx,
                            "middle_trough_idx": middle_trough_idx,
                            "neckline_price": neckline_price,
                            "completion_idx": second_peak_idx
                        }
        
        # Try to detect Double Bottom (bullish) pattern
        # We need 2 troughs at similar levels with a peak between them
        for i in range(len(trough_indices) - 1):
            first_trough_idx = trough_indices[i]
            second_trough_idx = trough_indices[i + 1]
            
            # Check if troughs are similar in height (within similarity threshold)
            first_trough_price = prices[first_trough_idx]
            second_trough_price = prices[second_trough_idx]
            
            price_diff = abs(first_trough_price - second_trough_price)
            avg_price = (first_trough_price + second_trough_price) / 2
            price_diff_pct = (price_diff / avg_price) * 100
            
            # Ensure troughs are similar and at least one bar separates them
            if price_diff_pct <= peak_similarity_pct and (second_trough_idx - first_trough_idx) > 1:
                # Check if there's a peak between the troughs
                middle_peak_idx = None
                for peak in peak_indices:
                    if first_trough_idx < peak < second_trough_idx:
                        middle_peak_idx = peak
                        break
                
                if middle_peak_idx is not None:
                    # Confirm sufficient pattern height
                    peak_price = prices[middle_peak_idx]
                    trough_height = peak_price - max(first_trough_price, second_trough_price)
                    height_pct = (trough_height / peak_price) * 100
                    
                    if height_pct >= min_height_pct:
                        # Pattern height is significant
                        # Neckline is at the central peak level
                        neckline_price = peak_price
                        
                        return {
                            "pattern_type": PatternType.DOUBLE_BOTTOM,
                            "first_trough_idx": first_trough_idx,
                            "second_trough_idx": second_trough_idx,
                            "middle_peak_idx": middle_peak_idx,
                            "neckline_price": neckline_price,
                            "completion_idx": second_trough_idx
                        }
                        
        return {}
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on double top/bottom patterns.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        lookback_period = self.parameters["lookback_period"]
        peak_distance = self.parameters["peak_distance"]
        peak_similarity = self.parameters["peak_similarity"]
        min_pattern_height = self.parameters["min_pattern_height"]
        confirmation_bars = self.parameters["confirmation_bars"]
        risk_percent = self.parameters["risk_percent"]
        
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
            
            pattern_info = self._detect_double_top_bottom(
                prices, highs, lows, peak_distance, peak_similarity, min_pattern_height
            )
            
            # Check if we have a pattern
            if pattern_info:
                pattern_type = pattern_info["pattern_type"]
                neckline_price = pattern_info["neckline_price"]
                
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
                
                if pattern_type == PatternType.DOUBLE_TOP:
                    # Bearish pattern - breakout is when price breaks below neckline
                    if current_price < neckline_price:
                        is_breakout = True
                        breakout_direction = "bearish"
                        self.breakout_counters[symbol] += 1
                    else:
                        self.breakout_counters[symbol] = 0
                
                elif pattern_type == PatternType.DOUBLE_BOTTOM:
                    # Bullish pattern - breakout is when price breaks above neckline
                    if current_price > neckline_price:
                        is_breakout = True
                        breakout_direction = "bullish"
                        self.breakout_counters[symbol] += 1
                    else:
                        self.breakout_counters[symbol] = 0
                
                # Check for confirmed breakout
                is_confirmed = self.breakout_counters[symbol] >= confirmation_bars
                
                # Generate signal if we have a confirmed breakout
                current_position = symbol in self.current_positions
                
                if is_breakout and is_confirmed:
                    if breakout_direction == "bullish" and not current_position:
                        # Calculate double bottom target (pattern height projected upward)
                        pattern_height = neckline_price - min(
                            prices[pattern_info["first_trough_idx"]],
                            prices[pattern_info["second_trough_idx"]]
                        )
                        price_target = neckline_price + pattern_height
                        stop_loss = min(lows[-5:]) * 0.99  # Just below recent lows
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.8,
                            reason=f"Double Bottom breakout at ${current_price:.2f}",
                            params={
                                "risk_percent": risk_percent,
                                "entry_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit": price_target,
                                "pattern_type": "double_bottom",
                                "neckline": neckline_price
                            }
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated BUY signal for {symbol} on Double Bottom breakout at {current_price:.2f}")
                        
                    elif breakout_direction == "bearish" and current_position:
                        # For bearish pattern, generate a sell signal if already in position
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.8,
                            reason=f"Double Top breakdown at ${current_price:.2f}",
                            params={
                                "risk_percent": risk_percent,
                                "pattern_type": "double_top",
                                "neckline": neckline_price
                            }
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated SELL signal for {symbol} on Double Top breakdown at {current_price:.2f}")
            
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
