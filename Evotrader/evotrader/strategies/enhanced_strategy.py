"""
Enhanced strategy implementations that leverage the object-oriented indicator system.
These strategies extend the existing strategy framework while adding the benefits
of our class-based indicator approach.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter
from ..utils.indicator_system import (
    Indicator, IndicatorFactory, RSI, MACD, BollingerBands, SMA, EMA
)


class EnhancedStrategy(Strategy):
    """
    Base class for enhanced strategies that use the object-oriented indicator system.
    This class provides common functionality while maintaining compatibility with
    the existing strategy framework.
    """
    
    def __init__(self, strategy_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with parameters."""
        super().__init__(strategy_id, parameters)
        self.logger = logging.getLogger(f"evotrader.strategy.{self.strategy_id}")
        
        # Position tracking - maintain a memory of current positions
        self.current_positions = {}
        
        # Store indicators by symbol
        self.indicators: Dict[str, Dict[str, Indicator]] = {}
        self.symbols: Set[str] = set()
        
        # Use for tracking cooldown periods
        self.last_signal_day = {}
    
    def setup_indicators(self, symbol: str) -> None:
        """
        Set up indicators for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        # Override in subclasses to set up specific indicators
        pass
    
    def ensure_indicators_exist(self, symbol: str) -> None:
        """
        Ensure indicators exist for a symbol, creating them if necessary.
        
        Args:
            symbol: Trading symbol
        """
        if symbol not in self.indicators:
            self.indicators[symbol] = {}
            self.setup_indicators(symbol)
            self.symbols.add(symbol)
    
    def update_indicators(self, symbol: str, candle: Dict[str, Any]) -> None:
        """
        Update all indicators for a symbol with new data.
        
        Args:
            symbol: Trading symbol
            candle: Price candle with OHLCV data
        """
        self.ensure_indicators_exist(symbol)
        
        for indicator in self.indicators[symbol].values():
            indicator.update(candle)
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on enhanced indicators.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Current day
        current_day = next(iter(market_data.values())).get("day", 0) if market_data else 0
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Convert market data to a candle format for our indicators
            candle = self.create_candle_from_data(data)
            
            # Update indicators for this symbol
            self.update_indicators(symbol, candle)
            
            # Generate symbol-specific signals (implemented by subclasses)
            symbol_signals = self.generate_symbol_signals(symbol, data, current_day)
            signals.extend(symbol_signals)
        
        return signals
    
    def generate_symbol_signals(self, symbol: str, data: Dict[str, Any], current_day: int) -> List[Signal]:
        """
        Generate signals for a specific symbol.
        
        Args:
            symbol: Trading symbol
            data: Market data for the symbol
            current_day: Current simulation day
            
        Returns:
            List of trading signals
        """
        # Override in subclasses
        return []
    
    def create_candle_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a candle dictionary from market data.
        
        Args:
            data: Market data
            
        Returns:
            Candle dictionary with OHLCV data
        """
        # Get price history
        price_history = data.get('history', {})
        prices = price_history.get('prices', [])
        highs = price_history.get('highs', [])
        lows = price_history.get('lows', [])
        
        candle = {
            'timestamp': data.get('timestamp', 0),
            'close': prices[-1] if prices and len(prices) > 0 else None,
            'high': highs[-1] if highs and len(highs) > 0 else None,
            'low': lows[-1] if lows and len(lows) > 0 else None,
            'volume': data.get('volume', 0)
        }
        
        # Include open price if available
        if 'open' in data:
            candle['open'] = data['open']
        elif prices and len(prices) > 1:
            # Use previous close as open
            candle['open'] = prices[-2]
        else:
            candle['open'] = candle['close']
            
        return candle
    
    def on_order_filled(self, order_data: Dict[str, Any]) -> None:
        """Update strategy state when an order is filled."""
        order = order_data.get("order")
        if not order:
            return
            
        symbol = order.symbol
        side = order.side
        fill_price = order_data.get("fill_price", 0)
        pnl = order_data.get("pnl", 0)
        
        # Track positions using OrderSide enum instead of strings
        from ..core.trading_types import OrderSide
        
        self.logger.debug(f"[STRATEGY] Order filled for {symbol}: {side} at {fill_price}")
        
        if side == OrderSide.BUY:
            self.current_positions[symbol] = {
                "entry_price": fill_price,
                "quantity": order.quantity
            }
            self.logger.debug(f"[STRATEGY] Added position for {symbol} at {fill_price}")
        elif side == OrderSide.SELL and symbol in self.current_positions:
            entry_price = self.current_positions[symbol].get("entry_price", 0)
            profit_loss = (fill_price - entry_price) * order.quantity
            self.logger.info(f"[STRATEGY] Closed position for {symbol}. P&L: ${profit_loss:.2f}")
            del self.current_positions[symbol]


class EnhancedRSIStrategy(EnhancedStrategy):
    """
    Enhanced RSI strategy using object-oriented indicators.
    Buys when RSI is oversold and sells when RSI is overbought.
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
                mutation_factor=0.3
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
                default_value=5.0,
                min_value=1.0,
                max_value=20.0,
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
            ),
            StrategyParameter(
                name="debug_mode",
                default_value=False,
                is_mutable=False
            )
        ]
    
    def setup_indicators(self, symbol: str) -> None:
        """Setup RSI indicator for a symbol."""
        # Extract parameters
        rsi_period = self.parameters["rsi_period"]
        
        # Create RSI indicator
        self.indicators[symbol]['rsi'] = IndicatorFactory.create(
            'rsi', symbol, {'period': rsi_period}
        )
    
    def generate_symbol_signals(self, symbol: str, data: Dict[str, Any], current_day: int) -> List[Signal]:
        """Generate RSI-based signals for a symbol."""
        signals = []
        
        # Extract parameters
        rsi_period = self.parameters["rsi_period"]
        oversold = self.parameters["oversold_threshold"]
        overbought = self.parameters["overbought_threshold"]
        risk_percent = self.parameters["risk_percent"]
        cooldown = self.parameters["cooldown_bars"]
        
        # Get the RSI indicator
        if symbol not in self.indicators or 'rsi' not in self.indicators[symbol]:
            return signals
            
        rsi_indicator = self.indicators[symbol]['rsi']
        
        # Get current RSI value
        rsi = rsi_indicator.get_last_value()
        
        # Skip if RSI is not available or not ready
        if rsi is None or not rsi_indicator.is_ready:
            # Fall back to the data provider's RSI if available
            # This maintains backward compatibility
            rsi_key = f"rsi_{rsi_period}"
            if rsi_key in data:
                rsi = data[rsi_key]
            else:
                return signals
        
        # Check cooldown period
        last_signal = self.last_signal_day.get(symbol, -999)
        if current_day - last_signal <= cooldown:
            return signals
        
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


class EnhancedMACDStrategy(EnhancedStrategy):
    """
    Enhanced MACD strategy using object-oriented indicators.
    Buys on bullish MACD crossovers and sells on bearish crossovers.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="fast_period",
                default_value=12,
                min_value=3,
                max_value=24,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="slow_period",
                default_value=26,
                min_value=5,
                max_value=52,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="signal_period",
                default_value=9,
                min_value=3,
                max_value=18,
                step=1,
                is_mutable=True,
                mutation_factor=0.3
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=5.0,
                min_value=1.0,
                max_value=20.0,
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
            ),
            StrategyParameter(
                name="histogram_threshold",
                default_value=0.0,
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="debug_mode",
                default_value=False,
                is_mutable=False
            )
        ]
    
    def setup_indicators(self, symbol: str) -> None:
        """Setup MACD indicator for a symbol."""
        # Extract parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        signal_period = self.parameters["signal_period"]
        
        # Create MACD indicator
        self.indicators[symbol]['macd'] = IndicatorFactory.create(
            'macd', symbol, {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }
        )
    
    def generate_symbol_signals(self, symbol: str, data: Dict[str, Any], current_day: int) -> List[Signal]:
        """Generate MACD-based signals for a symbol."""
        signals = []
        
        # Extract parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        signal_period = self.parameters["signal_period"]
        risk_percent = self.parameters["risk_percent"]
        cooldown = self.parameters["cooldown_bars"]
        histogram_threshold = self.parameters["histogram_threshold"]
        
        # Get the MACD indicator
        if symbol not in self.indicators or 'macd' not in self.indicators[symbol]:
            return signals
            
        macd_indicator = self.indicators[symbol]['macd']
        
        # Skip if MACD is not ready
        if not macd_indicator.is_ready:
            return signals
        
        # Get MACD values (MACD line, signal line, histogram)
        last_values = macd_indicator.get_last_complete_value()
        if not last_values:
            return signals
            
        macd_line, signal_line, histogram = last_values
        
        # Fall back to data provider's MACD if indicator not ready
        if macd_line is None or signal_line is None:
            if 'macd_line' in data and 'macd_signal' in data:
                macd_line = data['macd_line']
                signal_line = data['macd_signal']
                histogram = data.get('macd_histogram', 0)
            else:
                return signals
        
        # Check cooldown period
        last_signal_day = self.last_signal_day.get(symbol, -999)
        if current_day - last_signal_day <= cooldown:
            return signals
        
        # Current position info
        in_position = symbol in self.current_positions
        current_price = data.get("price", 0)
        
        # Check for crossover conditions
        bullish_crossover = False
        bearish_crossover = False
        
        # First check if we have data for crossover detection
        macd_values = macd_indicator.get_macd_line()
        signal_values = macd_indicator.get_signal_line()
        
        if len(macd_values) > 1 and len(signal_values) > 1:
            # Current and previous values
            curr_macd = macd_values[-1]
            curr_signal = signal_values[-1]
            prev_macd = macd_values[-2]
            prev_signal = signal_values[-2]
            
            # Detect crossovers
            if all(v is not None for v in [curr_macd, curr_signal, prev_macd, prev_signal]):
                bullish_crossover = curr_macd > curr_signal and prev_macd <= prev_signal
                bearish_crossover = curr_macd < curr_signal and prev_macd >= prev_signal
        
        # Fall back to data provider's crossover signals if available
        # This maintains backward compatibility
        if not bullish_crossover and data.get('macd_bullish', False):
            bullish_crossover = True
            
        if not bearish_crossover and data.get('macd_bearish', False):
            bearish_crossover = True
        
        # Generate signals based on MACD crossovers
        if bullish_crossover and not in_position and abs(histogram) > histogram_threshold:
            # Buy signal - Bullish MACD crossover
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.7,
                reason=f"MACD({fast_period},{slow_period},{signal_period}) bullish crossover",
                params={
                    "risk_percent": risk_percent,
                    "entry_price": current_price,
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "histogram": histogram
                }
            )
            signals.append(signal)
            self.last_signal_day[symbol] = current_day
            self.logger.debug(f"Generated BUY signal for {symbol} at {current_price}, MACD crossover")
            
        elif bearish_crossover and in_position and abs(histogram) > histogram_threshold:
            # Sell signal - Bearish MACD crossover
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=0.7,
                reason=f"MACD({fast_period},{slow_period},{signal_period}) bearish crossover",
                params={
                    "risk_percent": risk_percent,
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "histogram": histogram
                }
            )
            signals.append(signal)
            self.last_signal_day[symbol] = current_day
            self.logger.debug(f"Generated SELL signal for {symbol} at {current_price}, MACD crossover")
        
        return signals
