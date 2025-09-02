"""Trend following strategy implementations."""

from typing import Dict, List, Any, Optional, Tuple
import logging

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    This strategy generates buy signals when a fast moving average crosses
    above a slow moving average, and sell signals when it crosses below.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="fast_period",
                default_value=5,
                min_value=2,
                max_value=20,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="slow_period",
                default_value=20,
                min_value=5,
                max_value=50,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
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
                name="take_profit_multiplier",
                default_value=2.0,
                min_value=1.1,
                max_value=5.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="stop_loss_multiplier",
                default_value=1.0,
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="debug_mode",
                default_value=False,
                is_mutable=False
            )
        ]
    
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with parameters."""
        super().__init__(strategy_id, parameters)
        self.logger = logging.getLogger(f"evotrader.strategy.{self.strategy_id}")
        
        # Position tracking - maintain a memory of current positions
        self.current_positions = {}
        self.prev_sma_fast = {}
        self.prev_sma_slow = {}
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        risk_percent = self.parameters["risk_percent"]
        debug_mode = self.parameters.get("debug_mode", False)
        
        # Debug current positions
        if debug_mode:
            self.logger.debug(f"[STRATEGY] Current positions: {list(self.current_positions.keys())}")
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Debug market data structure
            if symbol == "BTC/USD" and self.parameters.get("debug_mode", False):
                self.logger.debug(f"Market data for {symbol}: {list(data.keys())}")
                if 'history' in data:
                    self.logger.debug(f"History keys: {list(data['history'].keys())}")
                    self.logger.debug(f"History lengths: prices={len(data['history'].get('prices', []))}, highs={len(data['history'].get('highs', []))}")
        
            # Skip if we don't have the necessary data
            sma_fast_key = f"sma_{fast_period}"
            sma_slow_key = f"sma_{slow_period}"
            
            # Use closest available SMAs if exact ones not available
            if sma_fast_key not in data:
                # Use the closest available SMA
                available_smas = [k for k in data.keys() if k.startswith("sma_")]
                if available_smas:
                    sma_fast_key = min(available_smas, key=lambda x: abs(int(x.split("_")[1]) - fast_period))
                else:
                    continue  # No SMAs available
            
            if sma_slow_key not in data:
                available_smas = [k for k in data.keys() if k.startswith("sma_")]
                if available_smas:
                    sma_slow_key = min(available_smas, key=lambda x: abs(int(x.split("_")[1]) - slow_period))
                else:
                    continue  # No SMAs available
                    
            # Extract current SMAs
            sma_fast = data.get(sma_fast_key)
            sma_slow = data.get(sma_slow_key)
            
            # Skip if SMAs are not available
            if sma_fast is None or sma_slow is None:
                # If indicators not provided by data feed, calculate them ourselves if history is available
                if 'history' in data and 'prices' in data['history'] and len(data['history']['prices']) > slow_period:
                    history = data['history']['prices']
                    # Calculate simple moving averages
                    fast_sum = sum(history[-fast_period:])
                    slow_sum = sum(history[-slow_period:])
                    sma_fast = fast_sum / fast_period
                    sma_slow = slow_sum / slow_period
                    
                    if symbol == "BTC/USD" and self.parameters.get("debug_mode", False):
                        self.logger.debug(f"Calculated SMAs for {symbol}: fast={sma_fast:.2f}, slow={sma_slow:.2f} from {len(history)} price points")
                else:
                    if symbol == "BTC/USD" and self.parameters.get("debug_mode", False):
                        self.logger.debug(f"Skipping {symbol} - insufficient history or missing SMAs")
                    continue
            
            # Check if we have previous SMAs for this symbol
            prev_fast = self.prev_sma_fast.get(symbol)
            prev_slow = self.prev_sma_slow.get(symbol)
            
            # Store current SMAs for next iteration
            self.prev_sma_fast[symbol] = sma_fast
            self.prev_sma_slow[symbol] = sma_slow
            
            # Skip if we don't have previous SMAs (first iteration)
            if prev_fast is None or prev_slow is None:
                continue
                
            # Check for crossover
            bullish_cross = prev_fast <= prev_slow and sma_fast > sma_slow
            bearish_cross = prev_fast >= prev_slow and sma_fast < sma_slow
            
            # Current position info
            in_position = symbol in self.current_positions
            current_price = data.get("price", 0)
            
            # Debug logging for crossover detection
            if debug_mode and (bullish_cross or bearish_cross):
                self.logger.debug(f"[STRATEGY] Detected crossover for {symbol} - Bullish: {bullish_cross}, Bearish: {bearish_cross}")
                self.logger.debug(f"[STRATEGY] Previous SMAs - Fast: {prev_fast:.2f}, Slow: {prev_slow:.2f}")
                self.logger.debug(f"[STRATEGY] Current SMAs - Fast: {sma_fast:.2f}, Slow: {sma_slow:.2f}")
                self.logger.debug(f"[STRATEGY] In position: {in_position}, Price: {current_price}")
            
            # Generate signals based on crossovers
            if bullish_cross:
                # Always generate a BUY signal on bullish crossover for debugging
                # But mark it based on position state
                confidence = 0.9 if not in_position else 0.4  # Lower confidence if already in position
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    reason=f"{fast_period}-period SMA crossed above {slow_period}-period SMA",
                    params={
                        "risk_percent": risk_percent,
                        "entry_price": current_price
                    }
                )
                signals.append(signal)
                self.logger.info(f"[STRATEGY] Generated BUY signal for {symbol} at {current_price}, in_position={in_position}")
                
            elif bearish_cross:
                # Always generate a SELL signal on bearish crossover for debugging
                # But mark it based on position state
                confidence = 0.9 if in_position else 0.4  # Lower confidence if not in position
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    reason=f"{fast_period}-period SMA crossed below {slow_period}-period SMA",
                    params={"risk_percent": risk_percent}
                )
                signals.append(signal)
                self.logger.info(f"[STRATEGY] Generated SELL signal for {symbol} at {current_price}, in_position={in_position}")
                
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
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.
    
    Buys when RSI is oversold (below threshold) and sells when
    RSI is overbought (above threshold).
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="rsi_period",
                default_value=14,
                min_value=2,
                max_value=30,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="oversold_threshold",
                default_value=30.0,
                min_value=10.0,
                max_value=40.0,
                step=1.0,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="overbought_threshold",
                default_value=70.0,
                min_value=60.0,
                max_value=90.0,
                step=1.0,
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
                name="cooldown_bars",
                default_value=3,
                min_value=0,
                max_value=10,
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
        
        # Signal tracking (to avoid repeated signals)
        self.last_signal_day = {}
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on RSI values.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        rsi_period = self.parameters["rsi_period"]
        oversold = self.parameters["oversold_threshold"]
        overbought = self.parameters["overbought_threshold"]
        risk_percent = self.parameters["risk_percent"]
        cooldown = self.parameters["cooldown_bars"]
        
        # Current day
        current_day = next(iter(market_data.values())).get("day", 0) if market_data else 0
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Skip if we don't have the necessary data
            rsi_key = f"rsi_{rsi_period}"
            
            # Use closest available RSI if exact one not available
            if rsi_key not in data:
                available_rsis = [k for k in data.keys() if k.startswith("rsi_")]
                if available_rsis:
                    rsi_key = min(available_rsis, key=lambda x: abs(int(x.split("_")[1]) - rsi_period))
                else:
                    continue  # No RSIs available
                    
            # Extract current RSI
            rsi = data.get(rsi_key)
            
            # Skip if RSI is not available
            if rsi is None:
                continue
                
            # Check cooldown period
            last_signal = self.last_signal_day.get(symbol, -999)
            if current_day - last_signal <= cooldown:
                continue
                
            # Current position info
            in_position = symbol in self.current_positions
            current_price = data.get("price", 0)
            
            # Generate signals based on RSI
            if rsi <= oversold and not in_position:
                # Buy signal - RSI oversold
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.7,
                    reason=f"RSI({rsi_period}) is oversold at {rsi:.1f}",
                    params={
                        "risk_percent": risk_percent,
                        "entry_price": current_price,
                        "rsi": rsi
                    }
                )
                signals.append(signal)
                self.last_signal_day[symbol] = current_day
                self.logger.debug(f"Generated BUY signal for {symbol} at {current_price}, RSI: {rsi:.1f}")
                
            elif rsi >= overbought and in_position:
                # Sell signal - RSI overbought
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.7,
                    reason=f"RSI({rsi_period}) is overbought at {rsi:.1f}",
                    params={
                        "risk_percent": risk_percent,
                        "rsi": rsi
                    }
                )
                signals.append(signal)
                self.last_signal_day[symbol] = current_day
                self.logger.debug(f"Generated SELL signal for {symbol} at {current_price}, RSI: {rsi:.1f}")
                
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
            self.current_positions[symbol] = {
                "entry_price": order.executed_price,
                "quantity": order.quantity
            }
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
