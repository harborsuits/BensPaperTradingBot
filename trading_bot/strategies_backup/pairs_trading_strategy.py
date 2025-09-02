import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller

from .base_strategy import Strategy, SignalType

logger = logging.getLogger(__name__)

class PairsTrading(Strategy):
    """
    Pairs trading strategy that exploits the co-movement between correlated securities
    
    Features:
    - Statistical tests for cointegration and pair suitability
    - Dynamic hedge ratio calculation
    - Spread normalization and signal generation
    - Regular rebalancing of hedge ratios
    - Distance-based position sizing
    """
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        parameters: Dict[str, Any] = None,
        name: str = "PairsTrading"
    ):
        """
        Initialize pairs trading strategy
        
        Args:
            pairs: List of symbol pairs to trade [(symbol1, symbol2), ...]
            parameters: Strategy parameters
            name: Strategy name
        """
        # Extract all unique symbols from the pairs
        all_symbols = list(set([symbol for pair in pairs for symbol in pair]))
        
        # Default parameters
        default_params = {
            "lookback_period": 100,         # Period for calculating hedge ratio
            "z_score_period": 20,           # Period for calculating z-score
            "entry_threshold": 2.0,         # Z-score threshold for entry
            "exit_threshold": 0.5,          # Z-score threshold for exit
            "max_half_life": 20,            # Maximum half-life for cointegration
            "min_half_life": 1,             # Minimum half-life for cointegration
            "hedge_ratio_method": "ols",    # Method for calculating hedge ratio: 'ols' or 'rolling'
            "rolling_window": 60,           # Window for rolling hedge ratio calculation
            "rebalance_frequency": 20,      # Bars between hedge ratio recalculations
            "cointegration_pvalue": 0.05,   # P-value threshold for cointegration test
            "risk_per_trade": 0.01,         # 1% risk per trade
            "max_position_size": 0.1,       # Maximum 10% of account in single position
            "stop_loss_z": 4.0,             # Stop loss at z-score threshold
            "position_scaling": True,       # Scale position based on z-score
            "scaling_factor": 0.5,          # Scaling factor for position size based on z-score
            "max_open_pairs": 5             # Maximum number of simultaneous pair positions
        }
        
        # Override defaults with provided parameters
        self.params = default_params.copy()
        if parameters:
            self.params.update(parameters)
        
        # Determine minimum history required
        lookback_periods = [
            self.params["lookback_period"],
            self.params["z_score_period"],
            self.params["rolling_window"]
        ]
        min_history = max(lookback_periods) + 10  # Add buffer
        
        # Initialize the strategy
        super().__init__(name, all_symbols, self.params, min_history_bars=min_history)
        
        # Store the pairs
        self.pairs = pairs
        
        # Track pair statistics
        self.pair_stats = {}
        for pair in pairs:
            self.pair_stats[pair] = {
                "hedge_ratio": 1.0,
                "half_life": None,
                "is_valid": False,
                "last_update": 0,
                "z_scores": [],
                "spread_mean": 0.0,
                "spread_std": 1.0
            }
        
        # Track active pairs
        self.active_pairs = set()
        
        logger.info(f"Initialized {name} strategy for pairs: {pairs} with parameters: {self.params}")
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame], 
        current_time: pd.Timestamp
    ) -> Dict[str, SignalType]:
        """
        Generate trading signals for pairs
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {symbol: SignalType.FLAT for symbol in self.symbols}
        
        # Check if we reached the maximum number of active pairs
        if len(self.active_pairs) >= self.params["max_open_pairs"]:
            # Only check for exit signals for currently active pairs
            for pair in self.active_pairs:
                if not self._has_pair_position(pair):
                    self.active_pairs.remove(pair)
                    continue
                
                symbol1, symbol2 = pair
                if symbol1 in data and symbol2 in data:
                    exit_signal1, exit_signal2 = self._calculate_exit_signals(pair, data)
                    signals[symbol1] = exit_signal1
                    signals[symbol2] = exit_signal2
            
            return signals
        
        # Update each pair and generate signals
        for pair in self.pairs:
            symbol1, symbol2 = pair
            
            # Skip if data for either symbol is missing
            if symbol1 not in data or symbol2 not in data:
                continue
            
            # Skip if not enough data
            if len(data[symbol1]) < self.min_history_bars or len(data[symbol2]) < self.min_history_bars:
                continue
            
            # If we already have a position in this pair, check for exit signals
            if self._has_pair_position(pair):
                exit_signal1, exit_signal2 = self._calculate_exit_signals(pair, data)
                signals[symbol1] = exit_signal1
                signals[symbol2] = exit_signal2
                continue
            
            # Update pair statistics and check if it's a valid pair
            self._update_pair_stats(pair, data)
            
            # Skip if the pair is not valid for trading
            if not self.pair_stats[pair]["is_valid"]:
                continue
            
            # Generate entry signals for the pair
            entry_signal1, entry_signal2 = self._calculate_entry_signals(pair, data)
            
            # If we have an entry signal, mark the pair as active
            if entry_signal1 != SignalType.FLAT or entry_signal2 != SignalType.FLAT:
                self.active_pairs.add(pair)
            
            signals[symbol1] = entry_signal1
            signals[symbol2] = entry_signal2
        
        return signals
    
    def _update_pair_stats(self, pair: Tuple[str, str], data: Dict[str, pd.DataFrame]) -> None:
        """
        Update statistics for a pair
        
        Args:
            pair: Symbol pair
            data: OHLCV data
        """
        symbol1, symbol2 = pair
        df1 = data[symbol1]
        df2 = data[symbol2]
        
        # Get close prices
        price1 = df1["close"]
        price2 = df2["close"]
        
        # Skip if there's not enough data
        if len(price1) < self.params["lookback_period"] or len(price2) < self.params["lookback_period"]:
            return
        
        # Get current bar index
        current_bar = len(price1) - 1
        
        # Check if we need to update the hedge ratio
        last_update = self.pair_stats[pair]["last_update"]
        if current_bar - last_update >= self.params["rebalance_frequency"]:
            # Perform cointegration test
            is_cointegrated, hedge_ratio, half_life = self._test_cointegration(price1, price2)
            
            if is_cointegrated:
                self.pair_stats[pair]["hedge_ratio"] = hedge_ratio
                self.pair_stats[pair]["half_life"] = half_life
                self.pair_stats[pair]["is_valid"] = True
                self.pair_stats[pair]["last_update"] = current_bar
                
                # Calculate spread
                spread = self._calculate_spread(price1, price2, hedge_ratio)
                
                # Calculate z-score parameters
                z_score_lookback = min(len(spread), self.params["z_score_period"])
                spread_mean = spread[-z_score_lookback:].mean()
                spread_std = spread[-z_score_lookback:].std()
                
                if spread_std > 0:
                    # Update spread statistics
                    self.pair_stats[pair]["spread_mean"] = spread_mean
                    self.pair_stats[pair]["spread_std"] = spread_std
                    
                    # Calculate current z-score
                    current_z = (spread.iloc[-1] - spread_mean) / spread_std
                    
                    # Update z-score history
                    z_scores = self.pair_stats[pair]["z_scores"]
                    z_scores.append(current_z)
                    
                    # Keep only recent history
                    if len(z_scores) > self.params["z_score_period"]:
                        z_scores = z_scores[-self.params["z_score_period"]:]
                    
                    self.pair_stats[pair]["z_scores"] = z_scores
                else:
                    # Spread has zero standard deviation, mark as invalid
                    self.pair_stats[pair]["is_valid"] = False
            else:
                # Pair is not cointegrated, mark as invalid
                self.pair_stats[pair]["is_valid"] = False
        
        # If pair is valid, update z-score
        if self.pair_stats[pair]["is_valid"]:
            hedge_ratio = self.pair_stats[pair]["hedge_ratio"]
            spread_mean = self.pair_stats[pair]["spread_mean"]
            spread_std = self.pair_stats[pair]["spread_std"]
            
            # Calculate current spread and z-score
            current_spread = price1.iloc[-1] - hedge_ratio * price2.iloc[-1]
            
            if spread_std > 0:
                current_z = (current_spread - spread_mean) / spread_std
                
                # Update z-score history
                z_scores = self.pair_stats[pair]["z_scores"]
                z_scores.append(current_z)
                
                # Keep only recent history
                if len(z_scores) > self.params["z_score_period"]:
                    z_scores = z_scores[-self.params["z_score_period"]:]
                
                self.pair_stats[pair]["z_scores"] = z_scores
    
    def _calculate_entry_signals(
        self, 
        pair: Tuple[str, str], 
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[SignalType, SignalType]:
        """
        Calculate entry signals for a pair
        
        Args:
            pair: Symbol pair
            data: OHLCV data
            
        Returns:
            Tuple of (signal for symbol1, signal for symbol2)
        """
        if not self.pair_stats[pair]["is_valid"] or not self.pair_stats[pair]["z_scores"]:
            return SignalType.FLAT, SignalType.FLAT
        
        # Get the latest z-score
        current_z = self.pair_stats[pair]["z_scores"][-1]
        
        # Entry thresholds
        entry_threshold = self.params["entry_threshold"]
        
        # Generate signals based on z-score
        if current_z > entry_threshold:
            # Spread is too high - go short symbol1, long symbol2
            return SignalType.SHORT, SignalType.LONG
        elif current_z < -entry_threshold:
            # Spread is too low - go long symbol1, short symbol2
            return SignalType.LONG, SignalType.SHORT
        else:
            # No signal
            return SignalType.FLAT, SignalType.FLAT
    
    def _calculate_exit_signals(
        self, 
        pair: Tuple[str, str], 
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[SignalType, SignalType]:
        """
        Calculate exit signals for a pair
        
        Args:
            pair: Symbol pair
            data: OHLCV data
            
        Returns:
            Tuple of (signal for symbol1, signal for symbol2)
        """
        symbol1, symbol2 = pair
        
        # If we don't have positions in both symbols, return flat
        if symbol1 not in self.positions or symbol2 not in self.positions:
            return SignalType.FLAT, SignalType.FLAT
        
        # Get current positions
        position1 = self.positions[symbol1]
        position2 = self.positions[symbol2]
        
        if not self.pair_stats[pair]["is_valid"] or not self.pair_stats[pair]["z_scores"]:
            # Pair is no longer valid, exit positions
            return SignalType.FLAT, SignalType.FLAT
        
        # Get the latest z-score
        current_z = self.pair_stats[pair]["z_scores"][-1]
        
        # Check for stop loss (z-score moved too far in the wrong direction)
        stop_loss_z = self.params["stop_loss_z"]
        
        # Assuming position1 and position2 have opposite directions
        if position1.direction == SignalType.LONG:
            # We're long symbol1, short symbol2
            # This position profits when z-score increases
            if current_z < -stop_loss_z:
                # Z-score moved too far in the wrong direction
                return SignalType.FLAT, SignalType.FLAT
        else:
            # We're short symbol1, long symbol2
            # This position profits when z-score decreases
            if current_z > stop_loss_z:
                # Z-score moved too far in the wrong direction
                return SignalType.FLAT, SignalType.FLAT
        
        # Exit threshold
        exit_threshold = self.params["exit_threshold"]
        
        # Check if z-score reverted enough to exit
        if abs(current_z) <= exit_threshold:
            # Z-score reverted to mean, exit positions
            return SignalType.FLAT, SignalType.FLAT
        
        # No exit signal, maintain positions
        return position1.direction, position2.direction
    
    def _has_pair_position(self, pair: Tuple[str, str]) -> bool:
        """
        Check if we have positions in both symbols of a pair
        
        Args:
            pair: Symbol pair
            
        Returns:
            True if we have positions in both symbols
        """
        symbol1, symbol2 = pair
        return symbol1 in self.positions and symbol2 in self.positions
    
    def _test_cointegration(
        self, 
        price1: pd.Series, 
        price2: pd.Series
    ) -> Tuple[bool, float, Optional[float]]:
        """
        Test for cointegration between two price series
        
        Args:
            price1: Price series for symbol1
            price2: Price series for symbol2
            
        Returns:
            Tuple of (is_cointegrated, hedge_ratio, half_life)
        """
        # Use only lookback period for test
        x = price2[-self.params["lookback_period"]:].values
        y = price1[-self.params["lookback_period"]:].values
        
        # Add constant to x for regression
        x_with_const = sm.add_constant(x)
        
        # Calculate hedge ratio using OLS regression
        model = sm.OLS(y, x_with_const)
        results = model.fit()
        hedge_ratio = results.params[1]
        
        # Calculate spread
        spread = y - hedge_ratio * x
        
        # Test for stationarity (ADF test)
        adf_result = adfuller(spread)
        p_value = adf_result[1]
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        # Check if cointegration is valid
        is_cointegrated = (
            p_value < self.params["cointegration_pvalue"] and
            half_life is not None and
            self.params["min_half_life"] <= half_life <= self.params["max_half_life"]
        )
        
        return is_cointegrated, hedge_ratio, half_life
    
    def _calculate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """
        Calculate half-life of mean reversion for a spread
        
        Args:
            spread: Spread series
            
        Returns:
            Half-life value or None if not mean-reverting
        """
        # Calculate the autoregression coefficient
        spread_lag = np.roll(spread, 1)
        spread_lag[0] = spread_lag[1]
        
        spread_ret = spread[1:] - spread[:-1]
        spread_lag = spread_lag[1:]
        
        # Regression without intercept
        beta = np.dot(spread_lag, spread_ret) / np.dot(spread_lag, spread_lag)
        
        # Calculate half-life
        if beta < 0:  # Mean-reverting behavior
            half_life = -np.log(2) / beta
            return half_life
        else:
            return None
    
    def _calculate_spread(
        self, 
        price1: pd.Series, 
        price2: pd.Series, 
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate spread between two price series
        
        Args:
            price1: Price series for symbol1
            price2: Price series for symbol2
            hedge_ratio: Hedge ratio
            
        Returns:
            Spread series
        """
        return price1 - hedge_ratio * price2
    
    def calculate_position_size(
        self,
        symbol: str,
        signal: SignalType,
        price: float,
        volatility: float,
        account_size: float
    ) -> float:
        """
        Calculate position size with adjustments for pairs trading
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility
            account_size: Current account size
            
        Returns:
            Position size
        """
        # Find which pair this symbol belongs to
        pair = None
        for p in self.pairs:
            if symbol in p:
                pair = p
                break
        
        if pair is None:
            # Not part of a defined pair, use default position sizing
            return super().calculate_position_size(symbol, signal, price, volatility, account_size)
        
        # Get pair statistics
        pair_stats = self.pair_stats.get(pair)
        if not pair_stats or not pair_stats["is_valid"]:
            return 0.0  # Don't trade invalid pairs
        
        # Start with the base position size calculation
        pos_size = super().calculate_position_size(symbol, signal, price, volatility, account_size)
        
        # Adjust based on hedge ratio if needed
        symbol1, symbol2 = pair
        hedge_ratio = pair_stats["hedge_ratio"]
        
        if symbol == symbol2:
            # Apply hedge ratio to the second symbol
            pos_size *= hedge_ratio
        
        # Optionally scale position based on z-score
        if self.params["position_scaling"] and pair_stats["z_scores"]:
            current_z = pair_stats["z_scores"][-1]
            
            # Scale position based on absolute z-score (more extreme = larger position)
            z_factor = min(3.0, abs(current_z)) / self.params["entry_threshold"]
            scaling = 1.0 + self.params["scaling_factor"] * (z_factor - 1.0)
            
            pos_size *= scaling
        
        return pos_size 