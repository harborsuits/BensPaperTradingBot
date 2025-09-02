"""Time-frame based strategy implementations."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import math
import numpy as np

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicators import sma, ema, bollinger_bands, rsi, atr, stochastic


class SwingTradingStrategy(Strategy):
    """
    Swing Trading Strategy.
    
    Aims to capture "swings" in the market using a multi-indicator approach
    with a longer timeframe perspective.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="trend_ma_period",
                default_value=50,
                min_value=20,
                max_value=200,
                step=5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="signal_ma_period",
                default_value=20,
                min_value=10,
                max_value=50,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="rsi_period",
                default_value=14,
                min_value=7,
                max_value=21,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="rsi_upper",
                default_value=70,
                min_value=60,
                max_value=80,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="rsi_lower",
                default_value=30,
                min_value=20,
                max_value=40,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="bb_period",
                default_value=20,
                min_value=10,
                max_value=50,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="bb_std_dev",
                default_value=2.0,
                min_value=1.0,
                max_value=3.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="atr_period",
                default_value=14,
                min_value=7,
                max_value=28,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="atr_multiplier",
                default_value=3.0,
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="min_swing_bars",
                default_value=3,
                min_value=2,
                max_value=10,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="profit_atr_multiple",
                default_value=2.0,
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=2.0,
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="trend_filter",
                default_value=True,
                is_mutable=True
            ),
            StrategyParameter(
                name="reversion_mode",
                default_value=False,
                is_mutable=True
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
        
        # Swing state tracking
        self.swing_highs = {}  # symbol -> list of swing high points (index, price)
        self.swing_lows = {}   # symbol -> list of swing low points (index, price)
        self.current_trend = {}  # symbol -> 'up', 'down', or 'sideways'
        self.signal_cooldown = {}  # symbol -> bars since last signal
        
    def _identify_swing_points(self, prices: List[float], highs: List[float], lows: List[float], min_bars: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Identify swing high and swing low points.
        
        Args:
            prices: List of closing prices
            highs: List of high prices
            lows: List of low prices
            min_bars: Minimum number of bars for swing point confirmation
            
        Returns:
            Tuple of (swing_highs, swing_lows) where each is a list of (index, price) tuples
        """
        swing_highs = []
        swing_lows = []
        
        # Need at least 2*min_bars+1 candles for this calculation
        if len(prices) < (2 * min_bars + 1):
            return swing_highs, swing_lows
            
        # Find swing highs - a local high with min_bars lower highs on each side
        for i in range(min_bars, len(highs) - min_bars):
            is_swing_high = True
            
            # Check min_bars candles to the left
            for j in range(i - min_bars, i):
                if highs[j] > highs[i]:
                    is_swing_high = False
                    break
                    
            # Check min_bars candles to the right
            if is_swing_high:
                for j in range(i + 1, i + min_bars + 1):
                    if highs[j] > highs[i]:
                        is_swing_high = False
                        break
                        
            if is_swing_high:
                swing_highs.append((i, highs[i]))
                
        # Find swing lows - a local low with min_bars higher lows on each side
        for i in range(min_bars, len(lows) - min_bars):
            is_swing_low = True
            
            # Check min_bars candles to the left
            for j in range(i - min_bars, i):
                if lows[j] < lows[i]:
                    is_swing_low = False
                    break
                    
            # Check min_bars candles to the right
            if is_swing_low:
                for j in range(i + 1, i + min_bars + 1):
                    if lows[j] < lows[i]:
                        is_swing_low = False
                        break
                        
            if is_swing_low:
                swing_lows.append((i, lows[i]))
                
        return swing_highs, swing_lows
        
    def _determine_trend(self, prices: List[float], trend_ma_period: int) -> str:
        """
        Determine the current market trend.
        
        Args:
            prices: List of closing prices
            trend_ma_period: Period for the trend moving average
            
        Returns:
            String indicating trend direction: 'up', 'down', or 'sideways'
        """
        if len(prices) < trend_ma_period:
            return "sideways"  # Default if not enough data
            
        # Calculate trend moving average
        trend_ma = sma(prices, trend_ma_period)
        
        # Get current price and MA
        current_price = prices[-1]
        current_ma = trend_ma[-1]
        
        # Determine trend based on price relation to MA and MA slope
        ma_slope = 0
        if len(trend_ma) > 5:
            ma_slope = (trend_ma[-1] - trend_ma[-5]) / trend_ma[-5] * 100
            
        # Strong uptrend: price above MA and MA slope is positive
        if current_price > current_ma and ma_slope > 0.1:
            return "up"
        # Strong downtrend: price below MA and MA slope is negative
        elif current_price < current_ma and ma_slope < -0.1:
            return "down"
        # Sideways trend: price near MA or MA slope is flat
        else:
            return "sideways"
            
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on swing trading strategy.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        trend_ma_period = self.parameters["trend_ma_period"]
        signal_ma_period = self.parameters["signal_ma_period"]
        rsi_period = self.parameters["rsi_period"]
        rsi_upper = self.parameters["rsi_upper"]
        rsi_lower = self.parameters["rsi_lower"]
        bb_period = self.parameters["bb_period"]
        bb_std_dev = self.parameters["bb_std_dev"]
        atr_period = self.parameters["atr_period"]
        atr_multiplier = self.parameters["atr_multiplier"]
        min_swing_bars = self.parameters["min_swing_bars"]
        profit_multiple = self.parameters["profit_atr_multiple"]
        risk_percent = self.parameters["risk_percent"]
        use_trend_filter = self.parameters["trend_filter"]
        reversion_mode = self.parameters["reversion_mode"]
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Extract current price data
            current_price = data.get("price", 0)
            current_high = data.get("high", current_price)
            current_low = data.get("low", current_price)
            
            # Skip if invalid data
            if current_price <= 0:
                continue
                
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.high_history[symbol] = []
                self.low_history[symbol] = []
                self.signal_cooldown[symbol] = 0
                self.current_trend[symbol] = "sideways"
                self.swing_highs[symbol] = []
                self.swing_lows[symbol] = []
                
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            
            # Update signal cooldown
            self.signal_cooldown[symbol] += 1
            
            # Limit history size
            max_history = max(trend_ma_period * 2, 200)
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            lookback_needed = max(trend_ma_period, signal_ma_period, rsi_period, bb_period, atr_period)
            if len(self.price_history[symbol]) < lookback_needed + min_swing_bars * 2:
                continue
                
            # Get price history arrays
            prices = self.price_history[symbol]
            highs = self.high_history[symbol]
            lows = self.low_history[symbol]
            
            # Identify swing points
            recent_swing_highs, recent_swing_lows = self._identify_swing_points(
                prices, highs, lows, min_swing_bars
            )
            
            self.swing_highs[symbol] = recent_swing_highs
            self.swing_lows[symbol] = recent_swing_lows
            
            # Determine market trend
            self.current_trend[symbol] = self._determine_trend(prices, trend_ma_period)
            
            # Calculate technical indicators
            # RSI
            rsi_values = rsi(prices, rsi_period)
            current_rsi = rsi_values[-1] if rsi_values and rsi_values[-1] is not None else 50
            
            # Bollinger Bands
            mid_band, upper_band, lower_band = bollinger_bands(prices, bb_period, bb_std_dev)
            
            # ATR for stop loss and position sizing
            atr_values = atr(highs, lows, prices, atr_period)
            current_atr = atr_values[-1] if atr_values and atr_values[-1] is not None else 0
            
            # Signal MA (faster MA for entry timing)
            signal_ma_values = ema(prices, signal_ma_period)  # Using EMA for faster response
            current_signal_ma = signal_ma_values[-1] if signal_ma_values else prices[-1]
            
            # Trend direction
            trend = self.current_trend[symbol]
            
            # Current position info
            in_position = symbol in self.current_positions
            
            # Generate signals based on strategy type
            # Normal mode: trend following, Reversion mode: counter-trend
            
            # Need cooldown of at least 3 bars between signals
            signal_ready = self.signal_cooldown[symbol] >= 3
            
            if not in_position and signal_ready:
                # Entry signals
                if not reversion_mode:
                    # Trend-following mode
                    if (trend == "up" or not use_trend_filter) and current_price > current_signal_ma:
                        # Look for a recent swing low for entry
                        if recent_swing_lows and len(recent_swing_lows) > 0:
                            # Use most recent swing low for stop loss placement
                            _, swing_low_price = recent_swing_lows[-1]
                            
                            # Only enter if price is in lower part of its range (value entry)
                            if ((lower_band and current_price < (mid_band[-1] + lower_band[-1]) / 2) or
                                (current_rsi < 60)):  # Avoid overbought conditions
                                
                                # Calculate stop loss using swing low with buffer
                                stop_loss = min(swing_low_price * 0.99, current_price - current_atr * atr_multiplier)
                                take_profit = current_price + (current_price - stop_loss) * profit_multiple
                                
                                # Risk check - don't take trades with too tight stops
                                if (current_price - stop_loss) > (current_price * 0.005):  # Minimum 0.5% risk
                                    signal = Signal(
                                        symbol=symbol,
                                        signal_type=SignalType.BUY,
                                        confidence=0.75,
                                        reason=f"Swing Trading entry at ${current_price:.2f}, trend: {trend}",
                                        params={
                                            "risk_percent": risk_percent,
                                            "entry_price": current_price,
                                            "stop_loss": stop_loss,
                                            "take_profit": take_profit,
                                            "atr": current_atr,
                                            "rsi": current_rsi
                                        }
                                    )
                                    signals.append(signal)
                                    self.logger.debug(f"Generated BUY signal for {symbol} on swing strategy at {current_price:.2f}")
                                    
                                    # Reset cooldown
                                    self.signal_cooldown[symbol] = 0
                                    
                else:
                    # Mean reversion mode (counter-trend)
                    if current_rsi < rsi_lower and (lower_band and current_price < lower_band[-1]):
                        # Oversold condition - look for reversal entry
                        stop_loss = current_price - current_atr * atr_multiplier
                        take_profit = current_price + current_atr * profit_multiple
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.7,
                            reason=f"Swing Trading reversion entry at ${current_price:.2f}, RSI: {current_rsi:.1f}",
                            params={
                                "risk_percent": risk_percent,
                                "entry_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "atr": current_atr,
                                "rsi": current_rsi
                            }
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated BUY signal for {symbol} on reversion strategy at {current_price:.2f}")
                        
                        # Reset cooldown
                        self.signal_cooldown[symbol] = 0
            
            # Exit signals - check for stop loss, take profit, and trend changes
            elif in_position and symbol in self.current_positions:
                position = self.current_positions[symbol]
                
                # Check stop loss
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
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
                
                # Check take profit
                elif "take_profit" in position and current_price >= position["take_profit"]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.8,
                        reason=f"Take profit triggered at ${current_price:.2f}",
                        params={"risk_percent": risk_percent}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on take profit at {current_price:.2f}")
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
                
                # Trend reversal exit
                elif trend == "down" and not reversion_mode and use_trend_filter:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        reason=f"Trend reversal exit at ${current_price:.2f}",
                        params={"risk_percent": risk_percent}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on trend reversal at {current_price:.2f}")
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
                
                # Exit on overbought conditions if in reversion mode
                elif reversion_mode and current_rsi > rsi_upper:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        reason=f"Overbought exit at ${current_price:.2f}, RSI: {current_rsi:.1f}",
                        params={"risk_percent": risk_percent}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on RSI overbought at {current_price:.2f}")
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
        
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
            if "atr" in signal_params:
                params["atr"] = signal_params["atr"]
                
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity,
                **params
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
