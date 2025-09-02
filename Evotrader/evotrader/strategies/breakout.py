"""Breakout strategy implementations."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicators import bollinger_bands, atr


class ChannelBreakoutStrategy(Strategy):
    """
    Channel Breakout trading strategy.
    
    This strategy identifies price channels using highs and lows over a certain period,
    then generates signals when price breaks out of these channels.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="channel_period",
                default_value=20,
                min_value=5,
                max_value=50,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=10.0,
                min_value=1.0,
                max_value=50.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stop_loss_atr_multiple",
                default_value=2.0,
                min_value=0.5,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="take_profit_atr_multiple",
                default_value=3.0,
                min_value=1.0,
                max_value=10.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="confirmation_periods",
                default_value=1,
                min_value=1,
                max_value=5,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="volatility_filter",
                default_value=True,
                is_mutable=True
            )
        ]
    
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with parameters."""
        super().__init__(strategy_id, parameters)
        self.logger = logging.getLogger(f"evotrader.strategy.{self.strategy_id}")
        
        # Position tracking
        self.current_positions = {}
        
        # Channel tracking
        self.channel_highs = {}  # symbol -> highest high in channel_period
        self.channel_lows = {}   # symbol -> lowest low in channel_period
        self.price_history = {}  # symbol -> list of prices
        self.high_history = {}   # symbol -> list of highs
        self.low_history = {}    # symbol -> list of lows
        self.breakout_signals = {}  # symbol -> consecutive breakout signals
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on channel breakouts.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        channel_period = self.parameters["channel_period"]
        risk_percent = self.parameters["risk_percent"]
        stop_loss_atr = self.parameters["stop_loss_atr_multiple"]
        take_profit_atr = self.parameters["take_profit_atr_multiple"]
        confirmation_periods = self.parameters["confirmation_periods"]
        use_volatility_filter = self.parameters["volatility_filter"]
        
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
                self.breakout_signals[symbol] = {"upper": 0, "lower": 0}
                
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            
            # Limit history size
            max_history = max(channel_period * 2, 50)  # Keep enough history for calculations
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            if len(self.price_history[symbol]) < channel_period:
                continue
                
            # Calculate channel high and low
            highs = self.high_history[symbol][-channel_period:]
            lows = self.low_history[symbol][-channel_period:]
            channel_high = max(highs[:-1])  # Exclude current candle
            channel_low = min(lows[:-1])    # Exclude current candle
            
            # Store channel values
            self.channel_highs[symbol] = channel_high
            self.channel_lows[symbol] = channel_low
            
            # Calculate ATR for volatility filter and stop loss
            if use_volatility_filter and len(self.high_history[symbol]) > 14:
                # Use ATR as volatility filter
                atr_values = atr(
                    self.high_history[symbol], 
                    self.low_history[symbol], 
                    self.price_history[symbol],
                    14
                )
                current_atr = atr_values[-1] if atr_values[-1] is not None else 0
                channel_width = channel_high - channel_low
                
                # Skip if channel is too narrow relative to volatility
                if current_atr > 0 and channel_width < current_atr:
                    continue
            else:
                current_atr = (channel_high - channel_low) / 10  # Rough estimate if no ATR
            
            # Current position info
            in_position = symbol in self.current_positions
            
            # Check for breakouts
            is_upper_breakout = current_high > channel_high
            is_lower_breakout = current_low < channel_low
            
            # Track consecutive breakout signals
            if is_upper_breakout:
                self.breakout_signals[symbol]["upper"] += 1
                self.breakout_signals[symbol]["lower"] = 0
            elif is_lower_breakout:
                self.breakout_signals[symbol]["lower"] += 1
                self.breakout_signals[symbol]["upper"] = 0
            else:
                # No breakout, reset counters
                self.breakout_signals[symbol]["upper"] = 0
                self.breakout_signals[symbol]["lower"] = 0
            
            # Generate signals based on breakouts
            confirmed_upper = self.breakout_signals[symbol]["upper"] >= confirmation_periods
            confirmed_lower = self.breakout_signals[symbol]["lower"] >= confirmation_periods
            
            # Upper breakout - buy signal
            if confirmed_upper and not in_position:
                # Calculate stop loss and take profit levels
                stop_loss = current_price - (current_atr * stop_loss_atr)
                take_profit = current_price + (current_atr * take_profit_atr)
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.8,
                    reason=f"Upper channel breakout (${channel_high:.2f})",
                    params={
                        "risk_percent": risk_percent,
                        "entry_price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "channel_high": channel_high,
                        "channel_low": channel_low
                    }
                )
                signals.append(signal)
                self.logger.debug(f"Generated BUY signal for {symbol} on upper breakout at {current_price:.2f}")
            
            # Lower breakout - sell signal if in position (exit long)
            elif confirmed_lower and in_position:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.8,
                    reason=f"Lower channel breakout (${channel_low:.2f})",
                    params={
                        "risk_percent": risk_percent,
                        "channel_high": channel_high,
                        "channel_low": channel_low
                    }
                )
                signals.append(signal)
                self.logger.debug(f"Generated SELL signal for {symbol} on lower breakout at {current_price:.2f}")
                
            # Additional signals:
            # Check for stop loss and take profit levels if in position
            elif in_position and symbol in self.current_positions:
                position = self.current_positions[symbol]
                
                if "stop_loss" in position and current_price <= position["stop_loss"]:
                    # Stop loss hit
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
                    # Take profit hit
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.9,
                        reason=f"Take profit triggered at ${current_price:.2f}",
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
            # Save stop loss and take profit from the signal params
            params = {}
            signal_params = order_data.get("signal_params", {})
            
            if "stop_loss" in signal_params:
                params["stop_loss"] = signal_params["stop_loss"]
            if "take_profit" in signal_params:
                params["take_profit"] = signal_params["take_profit"]
                
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity,
                **params
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]


class SupportResistanceStrategy(Strategy):
    """
    Support and Resistance breakout strategy.
    
    This strategy identifies significant support and resistance levels
    and trades breakouts from these levels.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="lookback_period",
                default_value=50,
                min_value=20,
                max_value=200,
                step=5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="level_strength",
                default_value=3,
                min_value=2,
                max_value=10,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="breakout_threshold",
                default_value=0.5,  # percentage of price
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=10.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="max_active_levels",
                default_value=3,
                min_value=1,
                max_value=10,
                step=1,
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
        
        # Support and resistance levels
        self.support_levels = {}  # symbol -> list of support levels
        self.resistance_levels = {}  # symbol -> list of resistance levels
        
        # Level validation (avoid trading the same level repeatedly)
        self.traded_levels = {}  # symbol -> list of recently traded levels
        
    def _find_support_resistance(self, highs: List[float], lows: List[float], strength: int) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            highs: List of price highs
            lows: List of price lows
            strength: Number of points required to confirm a level
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if len(highs) < strength * 2 + 1 or len(lows) < strength * 2 + 1:
            return [], []
            
        resistance_levels = []
        support_levels = []
        
        # Find resistance levels (local highs with 'strength' points on either side)
        for i in range(strength, len(highs) - strength):
            is_resistance = True
            pivot = highs[i]
            
            # Check left side
            for j in range(i - strength, i):
                if highs[j] > pivot:
                    is_resistance = False
                    break
                    
            # Check right side
            if is_resistance:
                for j in range(i + 1, i + strength + 1):
                    if highs[j] > pivot:
                        is_resistance = False
                        break
            
            if is_resistance:
                # Check if similar to existing levels
                is_unique = True
                for level in resistance_levels:
                    # Consider levels within 0.5% to be duplicates
                    if abs(pivot - level) / level < 0.005:
                        is_unique = False
                        break
                        
                if is_unique:
                    resistance_levels.append(pivot)
        
        # Find support levels (local lows with 'strength' points on either side)
        for i in range(strength, len(lows) - strength):
            is_support = True
            pivot = lows[i]
            
            # Check left side
            for j in range(i - strength, i):
                if lows[j] < pivot:
                    is_support = False
                    break
                    
            # Check right side
            if is_support:
                for j in range(i + 1, i + strength + 1):
                    if lows[j] < pivot:
                        is_support = False
                        break
            
            if is_support:
                # Check if similar to existing levels
                is_unique = True
                for level in support_levels:
                    # Consider levels within 0.5% to be duplicates
                    if abs(pivot - level) / level < 0.005:
                        is_unique = False
                        break
                        
                if is_unique:
                    support_levels.append(pivot)
        
        # Sort levels
        resistance_levels.sort()
        support_levels.sort()
        
        return support_levels, resistance_levels
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on support/resistance breakouts.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        lookback = self.parameters["lookback_period"]
        level_strength = self.parameters["level_strength"]
        breakout_threshold_pct = self.parameters["breakout_threshold"] / 100
        risk_percent = self.parameters["risk_percent"]
        max_active_levels = self.parameters["max_active_levels"]
        
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
                self.traded_levels[symbol] = []
                
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            
            # Limit history size
            max_history = max(lookback * 2, 200)  # Keep enough history for calculations
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            if len(self.price_history[symbol]) < lookback:
                continue
                
            # Find support and resistance levels
            highs = self.high_history[symbol][-lookback:-1]  # Exclude current candle
            lows = self.low_history[symbol][-lookback:-1]    # Exclude current candle
            
            support_levels, resistance_levels = self._find_support_resistance(
                highs, lows, level_strength
            )
            
            # Update stored levels
            self.support_levels[symbol] = support_levels
            self.resistance_levels[symbol] = resistance_levels
            
            # Check for tradeable levels
            active_resistance = []
            active_support = []
            
            # Find resistance levels just above current price
            for level in resistance_levels:
                if level > current_price and level < current_price * 1.1:
                    # Check if we've traded this level recently
                    if level not in self.traded_levels[symbol]:
                        active_resistance.append(level)
            
            # Find support levels just below current price
            for level in support_levels:
                if level < current_price and level > current_price * 0.9:
                    # Check if we've traded this level recently
                    if level not in self.traded_levels[symbol]:
                        active_support.append(level)
            
            # Limit number of active levels
            active_resistance = sorted(active_resistance)[:max_active_levels]
            active_support = sorted(active_support, reverse=True)[:max_active_levels]
            
            # Current position info
            in_position = symbol in self.current_positions
            
            # Check for breakouts
            for level in active_resistance:
                breakout_threshold = level * breakout_threshold_pct
                
                # Resistance breakout (price breaks above resistance)
                if current_price > level + breakout_threshold and not in_position:
                    # Calculate potential stop loss (below the breakout level)
                    stop_loss = level * 0.99
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.75,
                        reason=f"Resistance breakout at ${level:.2f}",
                        params={
                            "risk_percent": risk_percent,
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "resistance_level": level
                        }
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated BUY signal for {symbol} on resistance breakout at {current_price:.2f}")
                    
                    # Add to traded levels
                    self.traded_levels[symbol].append(level)
                    if len(self.traded_levels[symbol]) > 10:
                        self.traded_levels[symbol] = self.traded_levels[symbol][-10:]
                    
                    # Only generate one signal per bar
                    break
            
            # Check for support breakdown (only if in position)
            if in_position:
                for level in active_support:
                    breakout_threshold = level * breakout_threshold_pct
                    
                    # Support breakdown (price breaks below support)
                    if current_price < level - breakout_threshold:
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.75,
                            reason=f"Support breakdown at ${level:.2f}",
                            params={
                                "risk_percent": risk_percent,
                                "support_level": level
                            }
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated SELL signal for {symbol} on support breakdown at {current_price:.2f}")
                        
                        # Add to traded levels
                        self.traded_levels[symbol].append(level)
                        if len(self.traded_levels[symbol]) > 10:
                            self.traded_levels[symbol] = self.traded_levels[symbol][-10:]
                        
                        # Only generate one signal per bar
                        break
        
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
            if "resistance_level" in signal_params:
                params["resistance_level"] = signal_params["resistance_level"]
                
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity,
                **params
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
