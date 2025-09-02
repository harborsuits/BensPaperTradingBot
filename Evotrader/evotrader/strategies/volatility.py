"""Volatility-based strategy implementations."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import math

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicators import atr, bollinger_bands


class ATRPositionSizing(Strategy):
    """
    ATR Position Sizing strategy.
    
    Uses Average True Range for dynamic position sizing and stop loss placement.
    Combines trend signals with volatility-adjusted position sizing.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="atr_period",
                default_value=14,
                min_value=5,
                max_value=30,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="trend_period",
                default_value=50,
                min_value=20,
                max_value=200,
                step=5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stop_loss_atr_multiple",
                default_value=3.0,
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="position_risk_percent",
                default_value=1.0,  # Risk 1% of account per trade
                min_value=0.1,
                max_value=3.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="profit_target_atr_multiple",
                default_value=2.0,
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="trend_filter",
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
        
        # Price history
        self.price_history = {}  # symbol -> list of prices
        self.high_history = {}   # symbol -> list of highs
        self.low_history = {}    # symbol -> list of lows
        
        # Signal tracking
        self.signal_cooldown = {}  # symbol -> bars since last signal
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals with ATR-based position sizing.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        atr_period = self.parameters["atr_period"]
        trend_period = self.parameters["trend_period"]
        stop_loss_multiple = self.parameters["stop_loss_atr_multiple"]
        position_risk_pct = self.parameters["position_risk_percent"]
        profit_target_multiple = self.parameters["profit_target_atr_multiple"]
        use_trend_filter = self.parameters["trend_filter"]
        
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
                
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            
            # Update signal cooldown
            if symbol in self.signal_cooldown:
                self.signal_cooldown[symbol] += 1
            
            # Limit history size
            max_history = max(trend_period * 2, 200)  # Keep enough history for calculations
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            if len(self.price_history[symbol]) < max(atr_period, trend_period):
                continue
                
            # Calculate ATR
            atr_values = atr(
                self.high_history[symbol],
                self.low_history[symbol],
                self.price_history[symbol],
                atr_period
            )
            
            # Skip if ATR is not available
            if not atr_values or atr_values[-1] is None:
                continue
                
            current_atr = atr_values[-1]
            
            # Check trend if filter enabled
            in_uptrend = True  # Default if no trend filter
            
            if use_trend_filter:
                # Simple trend definition: price above moving average
                sma_key = f"sma_{trend_period}"
                if sma_key in data:
                    trend_ma = data[sma_key]
                    in_uptrend = current_price > trend_ma
                else:
                    # Calculate our own MA if market data doesn't provide it
                    trend_prices = self.price_history[symbol][-trend_period:]
                    trend_ma = sum(trend_prices) / len(trend_prices)
                    in_uptrend = current_price > trend_ma
            
            # Current position info
            in_position = symbol in self.current_positions
            
            # Define signal conditions
            # Only generate signals if we're not in a cooldown period (3 bars minimum)
            cooldown_ready = self.signal_cooldown.get(symbol, 0) >= 3
            
            if not in_position and in_uptrend and cooldown_ready:
                # Calculate stop loss and position size based on ATR
                stop_loss = current_price - (current_atr * stop_loss_multiple)
                take_profit = current_price + (current_atr * profit_target_multiple)
                
                # Calculate dollar risk per share/unit
                dollar_risk_per_unit = current_price - stop_loss
                
                # Calculate position size based on account risk
                # Note: We'll need to get actual account balance from bot in a real implementation
                account_balance = 1.0  # Placeholder, will be replaced by actual balance
                max_dollar_risk = account_balance * (position_risk_pct / 100.0)
                
                # Ensure we don't divide by zero
                if dollar_risk_per_unit > 0:
                    # Units to trade
                    position_size = max_dollar_risk / dollar_risk_per_unit
                else:
                    position_size = 0
                
                # Only generate signal if position size is valid
                if position_size > 0:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.8,
                        reason=f"ATR position sizing entry, stop: ${stop_loss:.2f}",
                        params={
                            "risk_percent": position_risk_pct,
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": current_atr,
                            "position_size": position_size
                        }
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated BUY signal for {symbol} at {current_price:.2f}, ATR: {current_atr:.2f}")
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
            
            # Exit signals for positions
            elif in_position and symbol in self.current_positions:
                position = self.current_positions[symbol]
                
                # Check stop loss
                if "stop_loss" in position and current_price <= position["stop_loss"]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.9,
                        reason=f"ATR stop loss triggered at ${current_price:.2f}",
                        params={"risk_percent": position_risk_pct}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} at stop loss {position['stop_loss']:.2f}")
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
                
                # Check take profit
                elif "take_profit" in position and current_price >= position["take_profit"]:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.8,
                        reason=f"ATR take profit triggered at ${current_price:.2f}",
                        params={"risk_percent": position_risk_pct}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} at take profit {position['take_profit']:.2f}")
                    
                    # Reset cooldown
                    self.signal_cooldown[symbol] = 0
                
                # Exit if trend reverses (optional)
                elif use_trend_filter and not in_uptrend:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        reason=f"Trend reversal exit at ${current_price:.2f}",
                        params={"risk_percent": position_risk_pct}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on trend reversal at {current_price:.2f}")
                    
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


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy.
    
    Trades breakouts from periods of low volatility, using Bollinger
    Band squeeze as a volatility contraction indicator.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
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
                name="squeeze_threshold",
                default_value=2.0,  # % of price
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="breakout_threshold",
                default_value=1.0,  # % of band width
                min_value=0.1,
                max_value=2.0,
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
                default_value=15.0,
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
        
        # Volatility state tracking
        self.in_squeeze = {}  # symbol -> bool
        self.band_width_history = {}  # symbol -> list of band widths
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on volatility breakouts.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        bb_period = self.parameters["bb_period"]
        bb_std_dev = self.parameters["bb_std_dev"]
        squeeze_threshold_pct = self.parameters["squeeze_threshold"] / 100
        breakout_threshold = self.parameters["breakout_threshold"]
        check_volume = self.parameters["volume_confirmation"]
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
                self.band_width_history[symbol] = []
                self.in_squeeze[symbol] = False
                
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            self.volume_history[symbol].append(current_volume)
            
            # Limit history size
            max_history = max(bb_period * 3, 100)  # Keep enough history for calculations
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
                if len(self.band_width_history[symbol]) > max_history:
                    self.band_width_history[symbol] = self.band_width_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            if len(self.price_history[symbol]) < bb_period:
                continue
                
            # Calculate Bollinger Bands
            middle_band, upper_band, lower_band = bollinger_bands(
                self.price_history[symbol], bb_period, bb_std_dev
            )
            
            # Skip if bands are not available
            if not middle_band or middle_band[-1] is None:
                continue
                
            # Calculate current band width as percentage of price
            band_width = (upper_band[-1] - lower_band[-1]) / middle_band[-1]
            self.band_width_history[symbol].append(band_width)
            
            # Calculate average band width (for squeeze detection)
            if len(self.band_width_history[symbol]) < bb_period:
                continue
                
            avg_band_width = sum(self.band_width_history[symbol][-bb_period:]) / bb_period
            
            # Detect squeeze (low volatility)
            was_in_squeeze = self.in_squeeze[symbol]
            current_squeeze = band_width < squeeze_threshold_pct
            self.in_squeeze[symbol] = current_squeeze
            
            # Detect breakout from squeeze
            breakout_from_squeeze = was_in_squeeze and not current_squeeze
            
            # Check for price breakout
            price_breakout = False
            breakout_direction = None
            
            if breakout_from_squeeze:
                # Bullish breakout: price breaks above the upper band
                if current_price > upper_band[-1] * (1 + breakout_threshold / 100):
                    price_breakout = True
                    breakout_direction = "bullish"
                # Bearish breakout: price breaks below the lower band
                elif current_price < lower_band[-1] * (1 - breakout_threshold / 100):
                    price_breakout = True
                    breakout_direction = "bearish"
            
            # Check volume confirmation if required
            volume_confirmed = True
            if check_volume and len(self.volume_history[symbol]) > bb_period:
                avg_volume = sum(self.volume_history[symbol][-(bb_period+1):-1]) / bb_period
                volume_confirmed = current_volume > avg_volume * 1.5
            
            # Current position info
            in_position = symbol in self.current_positions
            
            # Generate signals based on breakouts
            if price_breakout and volume_confirmed:
                if breakout_direction == "bullish" and not in_position:
                    # Bullish breakout - buy signal
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.8,
                        reason=f"Volatility breakout (bullish) at ${current_price:.2f}",
                        params={
                            "risk_percent": risk_percent,
                            "entry_price": current_price,
                            "stop_loss": lower_band[-1],
                            "band_width": band_width,
                            "middle_band": middle_band[-1]
                        }
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated BUY signal for {symbol} on volatility breakout at {current_price:.2f}")
                
                elif breakout_direction == "bearish" and in_position:
                    # Bearish breakout - sell signal
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.8,
                        reason=f"Volatility breakout (bearish) at ${current_price:.2f}",
                        params={
                            "risk_percent": risk_percent,
                            "band_width": band_width
                        }
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on volatility breakout at {current_price:.2f}")
            
            # Exit signals - check for stop loss
            elif in_position and symbol in self.current_positions:
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
                
                # Exit on return to middle band (profit taking)
                elif "middle_band" in position and current_price >= position["middle_band"] * 1.03:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        reason=f"Profit target reached at ${current_price:.2f}",
                        params={"risk_percent": risk_percent}
                    )
                    signals.append(signal)
                    self.logger.debug(f"Generated SELL signal for {symbol} on profit target at {current_price:.2f}")
        
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
            if "middle_band" in signal_params:
                params["middle_band"] = signal_params["middle_band"]
                
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity,
                **params
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
