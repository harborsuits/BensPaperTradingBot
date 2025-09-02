"""Day Trading strategy implementation."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicators import ema, rsi, bollinger_bands, stochastic, atr


class DayTradingStrategy(Strategy):
    """
    Day Trading Strategy.
    
    Focuses on short-term price movements using momentum indicators,
    volatility breakouts, and pattern recognition with quick entries/exits.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="fast_ema",
                default_value=9,
                min_value=5,
                max_value=20,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="slow_ema",
                default_value=21,
                min_value=15,
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
                name="rsi_overbought",
                default_value=70,
                min_value=65,
                max_value=85,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="rsi_oversold",
                default_value=30,
                min_value=15,
                max_value=35,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stoch_k",
                default_value=14,
                min_value=5,
                max_value=21,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stoch_d",
                default_value=3,
                min_value=2,
                max_value=7,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stoch_overbought",
                default_value=80,
                min_value=70,
                max_value=90,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stoch_oversold",
                default_value=20,
                min_value=10,
                max_value=30,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="bb_period",
                default_value=20,
                min_value=10,
                max_value=30,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="bb_std_dev",
                default_value=2.0,
                min_value=1.5,
                max_value=3.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="atr_period",
                default_value=14,
                min_value=7,
                max_value=21,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="profit_factor",
                default_value=1.5,
                min_value=1.0,
                max_value=3.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="max_trades_per_day",
                default_value=3,
                min_value=1,
                max_value=10,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=1.0,
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="strategy_type",
                default_value="momentum",  # 'momentum', 'reversal', 'breakout'
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
        self.volume_history = {}  # symbol -> list of volumes
        
        # Trading session tracking
        self.trades_today = {}  # symbol -> count of trades today
        self.day_counter = 0  # Simple counter to track "days"
        self.last_day = 0  # Track day changes
        
        # Signal tracking
        self.signal_cooldown = {}  # symbol -> bars since last signal
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on day trading strategy.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check if we've moved to a new day (rudimentary day tracking)
        current_day = market_data.get("day", 0)
        if current_day != self.last_day:
            # Reset trades counter on new day
            self.trades_today = {}
            self.last_day = current_day
        
        # Extract parameters
        fast_ema_period = self.parameters["fast_ema"]
        slow_ema_period = self.parameters["slow_ema"]
        rsi_period = self.parameters["rsi_period"]
        rsi_overbought = self.parameters["rsi_overbought"]
        rsi_oversold = self.parameters["rsi_oversold"]
        stoch_k_period = self.parameters["stoch_k"]
        stoch_d_period = self.parameters["stoch_d"]
        stoch_overbought = self.parameters["stoch_overbought"]
        stoch_oversold = self.parameters["stoch_oversold"]
        bb_period = self.parameters["bb_period"]
        bb_std_dev = self.parameters["bb_std_dev"]
        atr_period = self.parameters["atr_period"]
        profit_factor = self.parameters["profit_factor"]
        max_trades = self.parameters["max_trades_per_day"]
        risk_percent = self.parameters["risk_percent"]
        strategy_type = self.parameters["strategy_type"]
        
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
                
            # Initialize tracking for this symbol if needed
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.high_history[symbol] = []
                self.low_history[symbol] = []
                self.volume_history[symbol] = []
                self.trades_today[symbol] = 0
                self.signal_cooldown[symbol] = 0
                
            # Update price history
            self.price_history[symbol].append(current_price)
            self.high_history[symbol].append(current_high)
            self.low_history[symbol].append(current_low)
            self.volume_history[symbol].append(current_volume)
            
            # Update signal cooldown
            self.signal_cooldown[symbol] += 1
            
            # Limit history size to keep memory usage reasonable
            max_history = 200  # We don't need as much history for day trading
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.high_history[symbol] = self.high_history[symbol][-max_history:]
                self.low_history[symbol] = self.low_history[symbol][-max_history:]
                self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
                
            # Skip if we don't have enough history
            lookback_needed = max(slow_ema_period, rsi_period, stoch_k_period, bb_period, atr_period)
            if len(self.price_history[symbol]) < lookback_needed:
                continue
                
            # Calculate indicators
            prices = self.price_history[symbol]
            highs = self.high_history[symbol]
            lows = self.low_history[symbol]
            
            # EMAs for trend detection
            fast_ema_values = ema(prices, fast_ema_period)
            slow_ema_values = ema(prices, slow_ema_period)
            
            current_fast_ema = fast_ema_values[-1] if fast_ema_values else prices[-1]
            current_slow_ema = slow_ema_values[-1] if slow_ema_values else prices[-1]
            
            # EMA crossover detection
            ema_cross_up = False
            ema_cross_down = False
            if len(fast_ema_values) > 1 and len(slow_ema_values) > 1:
                ema_cross_up = (fast_ema_values[-2] <= slow_ema_values[-2] and 
                               fast_ema_values[-1] > slow_ema_values[-1])
                ema_cross_down = (fast_ema_values[-2] >= slow_ema_values[-2] and 
                                 fast_ema_values[-1] < slow_ema_values[-1])
            
            # RSI for overbought/oversold detection
            rsi_values = rsi(prices, rsi_period)
            current_rsi = rsi_values[-1] if rsi_values and rsi_values[-1] is not None else 50
            
            # Stochastic oscillator
            k_values, d_values = stochastic(highs, lows, prices, stoch_k_period, stoch_d_period)
            current_k = k_values[-1] if k_values and k_values[-1] is not None else 50
            current_d = d_values[-1] if d_values and d_values[-1] is not None else 50
            
            # Stochastic crossover detection
            stoch_cross_up = False
            stoch_cross_down = False
            if len(k_values) > 1 and len(d_values) > 1:
                stoch_cross_up = (k_values[-2] <= d_values[-2] and 
                                 k_values[-1] > d_values[-1])
                stoch_cross_down = (k_values[-2] >= d_values[-2] and 
                                   k_values[-1] < d_values[-1])
            
            # Bollinger Bands for volatility and price extremes
            mid_band, upper_band, lower_band = bollinger_bands(prices, bb_period, bb_std_dev)
            
            # ATR for stop loss and position sizing
            atr_values = atr(highs, lows, prices, atr_period)
            current_atr = atr_values[-1] if atr_values and atr_values[-1] is not None else 0
            
            # Strategy rules based on strategy_type
            in_position = symbol in self.current_positions
            trades_made_today = self.trades_today.get(symbol, 0)
            can_trade_today = trades_made_today < max_trades
            
            # Only generate signals if cooldown period has passed (min 2 bars between trades)
            signal_ready = self.signal_cooldown[symbol] >= 2
            
            # Entry signals
            if not in_position and can_trade_today and signal_ready:
                if strategy_type == "momentum":
                    # Momentum strategy: Buy on strong upward momentum
                    if (ema_cross_up or (current_fast_ema > current_slow_ema)) and current_rsi > 50 and current_rsi < rsi_overbought:
                        # Additional confirmation with stochastic crossover
                        if stoch_cross_up or (current_k > current_d and current_k < stoch_overbought):
                            stop_loss = current_price - current_atr * 2
                            take_profit = current_price + current_atr * profit_factor
                            
                            signal = Signal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=0.75,
                                reason=f"Day trading momentum entry at ${current_price:.2f}",
                                params={
                                    "risk_percent": risk_percent,
                                    "entry_price": current_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "atr": current_atr,
                                    "strategy": "momentum"
                                }
                            )
                            signals.append(signal)
                            self.logger.debug(f"Generated BUY signal for {symbol} on momentum at {current_price:.2f}")
                            
                            # Reset cooldown and increment trade counter
                            self.signal_cooldown[symbol] = 0
                            self.trades_today[symbol] = trades_made_today + 1
                            
                elif strategy_type == "reversal":
                    # Reversal strategy: Buy on oversold conditions expecting a bounce
                    if current_rsi < rsi_oversold and current_price < lower_band[-1]:
                        if current_k < stoch_oversold and current_k > current_d:  # Starting to turn up
                            stop_loss = current_price - current_atr * 1.5
                            take_profit = current_price + current_atr * profit_factor
                            
                            signal = Signal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=0.7,
                                reason=f"Day trading reversal entry at ${current_price:.2f}",
                                params={
                                    "risk_percent": risk_percent,
                                    "entry_price": current_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "atr": current_atr,
                                    "strategy": "reversal"
                                }
                            )
                            signals.append(signal)
                            self.logger.debug(f"Generated BUY signal for {symbol} on reversal at {current_price:.2f}")
                            
                            # Reset cooldown and increment trade counter
                            self.signal_cooldown[symbol] = 0
                            self.trades_today[symbol] = trades_made_today + 1
                            
                elif strategy_type == "breakout":
                    # Breakout strategy: Buy on upper band breakout with increasing volume
                    if current_price > upper_band[-1]:
                        # Check for volume confirmation (current volume > avg of last 5 bars)
                        vol_increase = False
                        if len(self.volume_history[symbol]) >= 5:
                            avg_vol = sum(self.volume_history[symbol][-6:-1]) / 5
                            vol_increase = current_volume > avg_vol * 1.2
                            
                        if vol_increase or len(self.volume_history[symbol]) < 5:
                            stop_loss = current_price - current_atr * 2
                            take_profit = current_price + current_atr * profit_factor
                            
                            signal = Signal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=0.8,
                                reason=f"Day trading breakout entry at ${current_price:.2f}",
                                params={
                                    "risk_percent": risk_percent,
                                    "entry_price": current_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "atr": current_atr,
                                    "strategy": "breakout"
                                }
                            )
                            signals.append(signal)
                            self.logger.debug(f"Generated BUY signal for {symbol} on breakout at {current_price:.2f}")
                            
                            # Reset cooldown and increment trade counter
                            self.signal_cooldown[symbol] = 0
                            self.trades_today[symbol] = trades_made_today + 1
            
            # Exit signals
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
                
                # Strategy-specific exit signals
                elif "strategy" in position:
                    strategy = position["strategy"]
                    
                    if strategy == "momentum" and (ema_cross_down or current_rsi > rsi_overbought):
                        # Exit momentum trade on trend reversal
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.7,
                            reason=f"Momentum strategy exit at ${current_price:.2f}",
                            params={"risk_percent": risk_percent}
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated SELL signal for {symbol} on momentum exit at {current_price:.2f}")
                        
                        # Reset cooldown
                        self.signal_cooldown[symbol] = 0
                        
                    elif strategy == "reversal" and current_rsi > 60:
                        # Exit reversal trade once no longer oversold
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.7,
                            reason=f"Reversal strategy exit at ${current_price:.2f}",
                            params={"risk_percent": risk_percent}
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated SELL signal for {symbol} on reversal exit at {current_price:.2f}")
                        
                        # Reset cooldown
                        self.signal_cooldown[symbol] = 0
                        
                    elif strategy == "breakout" and current_price < mid_band[-1]:
                        # Exit breakout trade when price returns to middle band
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.7,
                            reason=f"Breakout strategy exit at ${current_price:.2f}",
                            params={"risk_percent": risk_percent}
                        )
                        signals.append(signal)
                        self.logger.debug(f"Generated SELL signal for {symbol} on breakout exit at {current_price:.2f}")
                        
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
            if "strategy" in signal_params:
                params["strategy"] = signal_params["strategy"]
                
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity,
                **params
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
