"""Mean reversion strategy implementations."""

from typing import Dict, List, Any, Optional, Tuple
import logging
import math

from ..core.strategy import Strategy, Signal, SignalType, StrategyParameter


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands mean reversion strategy.
    
    Buys when price touches the lower band and sells when it touches the upper band.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="lookback_period",
                default_value=20,
                min_value=5,
                max_value=50,
                step=1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="band_width",
                default_value=2.0,
                min_value=1.0,
                max_value=4.0,
                step=0.1,
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
                name="exit_band_width",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="max_open_trades",
                default_value=3,
                min_value=1,
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
        
        # Price history for calculating bands (in case not provided by market data)
        self.price_history = {}
        
    def _calculate_bollinger_bands(self, prices: List[float], period: int, width: float) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands for a list of prices.
        
        Args:
            prices: List of historical prices
            period: Lookback period 
            width: Number of standard deviations for the bands
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        if len(prices) < period:
            return None, None, None
            
        # Use only the last 'period' prices
        prices = prices[-period:]
        
        # Calculate middle band (SMA)
        middle_band = sum(prices) / period
        
        # Calculate standard deviation
        squared_diffs = [(price - middle_band) ** 2 for price in prices]
        variance = sum(squared_diffs) / period
        std_dev = math.sqrt(variance)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (width * std_dev)
        lower_band = middle_band - (width * std_dev)
        
        return middle_band, upper_band, lower_band
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        lookback = self.parameters["lookback_period"]
        band_width = self.parameters["band_width"]
        risk_percent = self.parameters["risk_percent"]
        exit_band_width = self.parameters["exit_band_width"]
        max_open_trades = self.parameters["max_open_trades"]
        
        # Check if we're at the position limit
        if len(self.current_positions) >= max_open_trades:
            # Only generate exit signals when at position limit
            pass_entry_signals = True
        else:
            pass_entry_signals = False
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                
            current_price = data.get("price", 0)
            if current_price > 0:
                self.price_history[symbol].append(current_price)
                
            # Skip if we don't have enough price history
            if len(self.price_history[symbol]) < lookback:
                continue
                
            # Calculate Bollinger Bands
            middle, upper, lower = self._calculate_bollinger_bands(
                self.price_history[symbol], lookback, band_width
            )
            
            if middle is None:
                continue
                
            # Calculate inner bands for exits
            inner_upper = middle + (exit_band_width * (upper - middle))
            inner_lower = middle - (exit_band_width * (middle - lower))
                
            # Current position info
            in_position = symbol in self.current_positions
            
            # Buy signal - price below lower band
            if current_price <= lower and not in_position and not pass_entry_signals:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.75,
                    reason=f"Price {current_price:.2f} touched lower Bollinger Band {lower:.2f}",
                    params={
                        "risk_percent": risk_percent,
                        "entry_price": current_price,
                        "middle_band": middle,
                        "upper_band": upper,
                        "lower_band": lower
                    }
                )
                signals.append(signal)
                self.logger.debug(f"Generated BUY signal for {symbol} at {current_price}")
                
            # Sell signal - price above upper band
            elif current_price >= upper and not in_position and not pass_entry_signals:
                # Optional: Short selling if supported
                # We'll just implement long-only for simplicity
                pass
                
            # Exit long position - price above inner upper band
            elif current_price >= inner_upper and in_position:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.75,
                    reason=f"Price {current_price:.2f} reached inner upper band {inner_upper:.2f}",
                    params={
                        "risk_percent": risk_percent,
                        "middle_band": middle,
                        "upper_band": upper,
                        "lower_band": lower
                    }
                )
                signals.append(signal)
                self.logger.debug(f"Generated SELL signal for {symbol} at {current_price}")
                
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


class PriceDeviationStrategy(Strategy):
    """
    Price Deviation mean reversion strategy.
    
    Buys when price deviates significantly below its moving average
    and sells when it returns to the mean.
    """
    
    @classmethod
    def get_parameters(cls) -> List[StrategyParameter]:
        """Define strategy parameters with mutation characteristics."""
        return [
            StrategyParameter(
                name="sma_period",
                default_value=30,
                min_value=10,
                max_value=100,
                step=5,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="entry_deviation",
                default_value=0.05,  # 5% deviation
                min_value=0.01,
                max_value=0.15,
                step=0.01,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="exit_deviation",
                default_value=0.02,  # 2% deviation 
                min_value=0.005,
                max_value=0.05,
                step=0.005,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="risk_percent",
                default_value=15.0,
                min_value=5.0,
                max_value=30.0,
                step=1.0,
                is_mutable=True,
                mutation_factor=0.2
            ),
            StrategyParameter(
                name="max_holding_periods",
                default_value=10,
                min_value=3,
                max_value=30,
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
        self.position_days = {}  # Track how long we've held each position
        
        # Current day tracking
        self.current_day = 0
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on price deviation from SMA.
        
        Args:
            market_data: Current market data keyed by symbol
                
        Returns:
            List of trading signals
        """
        signals = []
        
        # Extract parameters
        sma_period = self.parameters["sma_period"]
        entry_deviation = self.parameters["entry_deviation"]
        exit_deviation = self.parameters["exit_deviation"]
        risk_percent = self.parameters["risk_percent"]
        max_holding_periods = self.parameters["max_holding_periods"]
        
        # Update current day if available
        if market_data:
            self.current_day = next(iter(market_data.values())).get("day", self.current_day)
            
        # Process each symbol
        for symbol, data in market_data.items():
            # Skip if we don't have the necessary data
            sma_key = f"sma_{sma_period}"
            
            # Use closest available SMA if exact one not available
            if sma_key not in data:
                available_smas = [k for k in data.keys() if k.startswith("sma_")]
                if available_smas:
                    sma_key = min(available_smas, key=lambda x: abs(int(x.split("_")[1]) - sma_period))
                else:
                    continue  # No SMAs available
                    
            # Extract SMA and current price
            sma = data.get(sma_key)
            current_price = data.get("price", 0)
            
            # Skip if SMA is not available
            if sma is None or current_price <= 0:
                continue
                
            # Calculate deviation from SMA
            deviation = (current_price - sma) / sma
            
            # Current position info
            in_position = symbol in self.current_positions
            
            # Update holding period for active positions
            if in_position:
                if symbol not in self.position_days:
                    self.position_days[symbol] = 0
                self.position_days[symbol] += 1
            
            # Generate signals
            if deviation <= -entry_deviation and not in_position:
                # Buy signal - price significantly below SMA
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.7,
                    reason=f"Price deviated {deviation*100:.1f}% below {sma_period}-day SMA",
                    params={
                        "risk_percent": risk_percent,
                        "entry_price": current_price,
                        "sma": sma,
                        "deviation": deviation
                    }
                )
                signals.append(signal)
                self.logger.debug(f"Generated BUY signal for {symbol} at {current_price}")
                
            elif in_position and (
                # Sell signal conditions:
                # 1. Price returned to near SMA
                deviation >= -exit_deviation or 
                # 2. Maximum holding period reached
                self.position_days.get(symbol, 0) >= max_holding_periods
            ):
                reason = (
                    f"Price returned to within {exit_deviation*100:.1f}% of SMA" 
                    if deviation >= -exit_deviation 
                    else f"Maximum holding period ({max_holding_periods} days) reached"
                )
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.7,
                    reason=reason,
                    params={
                        "risk_percent": risk_percent,
                        "sma": sma,
                        "deviation": deviation
                    }
                )
                signals.append(signal)
                self.logger.debug(f"Generated SELL signal for {symbol} at {current_price}, reason: {reason}")
                
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
            self.position_days[symbol] = 0
            
        elif str(side) == "sell" and symbol in self.current_positions:
            # Position closed
            del self.current_positions[symbol]
            if symbol in self.position_days:
                del self.position_days[symbol]
