import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import talib
from enum import Enum

from .base_strategy import Strategy, SignalType

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset classes for macro allocation"""
    EQUITIES = "equities"       # Stock indices
    BONDS = "bonds"             # Government bonds
    COMMODITIES = "commodities" # Commodity indices or futures
    CURRENCIES = "currencies"   # Major currency pairs
    CRYPTO = "crypto"           # Cryptocurrencies
    CASH = "cash"               # Cash or money market

class MacroTrendStrategy(Strategy):
    """
    Macro trend allocation strategy that rotates between major asset classes
    
    Features:
    - Multi-asset class trend following
    - Economic indicator overlays
    - Volatility-based position sizing
    - Correlation-aware portfolio construction
    """
    
    def __init__(
        self,
        symbols: Dict[AssetClass, List[str]],
        parameters: Dict[str, Any] = None,
        name: str = "MacroTrend"
    ):
        """
        Initialize macro trend strategy
        
        Args:
            symbols: Dictionary mapping asset classes to symbols
            parameters: Strategy parameters
            name: Strategy name
        """
        # Flatten symbols list for base class
        all_symbols = [s for sym_list in symbols.values() for s in sym_list]
        
        # Default parameters
        default_params = {
            "trend_period": 60,              # Period for trend calculation
            "short_ma_period": 20,           # Short moving average period
            "long_ma_period": 50,            # Long moving average period
            "volatility_period": 20,         # Period for volatility calculation
            "min_trend_strength": 0.2,       # Minimum trend strength to take position
            "max_asset_class_allocation": 0.4, # Maximum allocation to a single asset class
            "max_single_symbol_allocation": 0.15, # Maximum allocation to a single symbol
            "risk_per_trade": 0.01,          # 1% risk per trade
            "use_correlation_filter": True,  # Whether to use correlation-aware allocation
            "correlation_lookback": 60,      # Correlation lookback period
            "min_allocation_percent": 0.05,  # Minimum allocation to include a symbol
            "cash_allocation_min": 0.1,      # Minimum cash allocation
            "rebalance_frequency": 20,       # Bars between rebalancing
            "trend_score_method": "macd",    # Method for calculating trend: 'macd', 'ma_cross', 'adx'
            "use_economic_overlay": False,   # Whether to use economic data for adjustments
            "stop_loss_atr": 2.0,            # Stop loss at 2 ATR
            "atr_period": 14                 # ATR period for stop calculation
        }
        
        # Override defaults with provided parameters
        self.params = default_params.copy()
        if parameters:
            self.params.update(parameters)
        
        # Determine minimum history required
        lookback_periods = [
            self.params["trend_period"],
            self.params["short_ma_period"],
            self.params["long_ma_period"],
            self.params["volatility_period"],
            self.params["correlation_lookback"],
            self.params["atr_period"]
        ]
        min_history = max(lookback_periods) + 10  # Add buffer
        
        # Initialize the strategy
        super().__init__(name, all_symbols, self.params, min_history_bars=min_history)
        
        # Store asset class mappings
        self.asset_class_map = {}
        for asset_class, symbol_list in symbols.items():
            for symbol in symbol_list:
                self.asset_class_map[symbol] = asset_class
        
        # Track allocation targets
        self.target_allocations = {symbol: 0.0 for symbol in all_symbols}
        self.last_rebalance_bar = 0
        
        # Track trend scores
        self.trend_scores = {symbol: 0.0 for symbol in all_symbols}
        self.asset_class_scores = {asset_class: 0.0 for asset_class in symbols.keys()}
        
        # Track correlation matrix
        self.correlation_matrix = None
        
        logger.info(f"Initialized {name} strategy with {len(all_symbols)} symbols across {len(symbols)} asset classes")
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame], 
        current_time: pd.Timestamp
    ) -> Dict[str, SignalType]:
        """
        Generate trading signals based on macro trends
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            
        Returns:
            Dictionary of symbol -> signal type
        """
        signals = {}
        
        # Skip if not enough data
        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < self.min_history_bars:
                signals[symbol] = SignalType.FLAT
                return signals  # Early return if any symbol lacks data
        
        # Get current bar index
        current_bar = len(next(iter(data.values())))
        
        # Check if it's time to rebalance
        if current_bar - self.last_rebalance_bar >= self.params["rebalance_frequency"]:
            # Calculate trend scores for each symbol
            self._calculate_trend_scores(data)
            
            # Calculate correlation matrix if using correlation filter
            if self.params["use_correlation_filter"]:
                self._calculate_correlation_matrix(data)
            
            # Generate target allocations
            self._calculate_target_allocations()
            
            # Update last rebalance bar
            self.last_rebalance_bar = current_bar
        
        # Compare current positions to target allocations and generate signals
        for symbol in self.symbols:
            current_allocation = 0.0
            if symbol in self.positions:
                # Calculate current allocation as a percentage of account
                position = self.positions[symbol]
                current_price = data[symbol].iloc[-1]['close']
                position_value = position.size * current_price
                account_value = self._get_account_value(data)
                current_allocation = position_value / account_value if account_value > 0 else 0.0
            
            target_allocation = self.target_allocations.get(symbol, 0.0)
            
            # Generate signal based on allocation difference
            if target_allocation > current_allocation + 0.02:  # 2% threshold to avoid small trades
                signals[symbol] = SignalType.LONG
            elif target_allocation < current_allocation - 0.02:
                if current_allocation > 0:
                    signals[symbol] = SignalType.FLAT  # Exit position
                else:
                    signals[symbol] = SignalType.SHORT  # Short if target is negative
            else:
                # Keep current position
                if symbol in self.positions:
                    signals[symbol] = self.positions[symbol].direction
                else:
                    signals[symbol] = SignalType.FLAT
        
        return signals
    
    def _calculate_trend_scores(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Calculate trend scores for each symbol
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
        """
        method = self.params["trend_score_method"]
        
        # Calculate trend score for each symbol
        for symbol, df in data.items():
            if method == "macd":
                score = self._calculate_macd_score(df)
            elif method == "ma_cross":
                score = self._calculate_ma_cross_score(df)
            elif method == "adx":
                score = self._calculate_adx_score(df)
            else:
                logger.warning(f"Unknown trend method: {method}, using MACD")
                score = self._calculate_macd_score(df)
            
            # Update trend score
            self.trend_scores[symbol] = score
        
        # Calculate aggregate score for each asset class
        for asset_class in self.asset_class_scores.keys():
            symbols = [s for s, ac in self.asset_class_map.items() if ac == asset_class]
            if symbols:
                scores = [self.trend_scores[s] for s in symbols if s in self.trend_scores]
                if scores:
                    self.asset_class_scores[asset_class] = np.mean(scores)
                else:
                    self.asset_class_scores[asset_class] = 0.0
    
    def _calculate_macd_score(self, df: pd.DataFrame) -> float:
        """
        Calculate trend score using MACD
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Trend score between -1 and 1
        """
        close_prices = df['close']
        
        # Calculate MACD
        macd, macd_signal, _ = talib.MACD(
            close_prices,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # Get latest values
        macd_value = macd.iloc[-1]
        signal_value = macd_signal.iloc[-1]
        
        # Calculate trend direction and strength
        if macd_value > signal_value:
            # Bullish trend
            strength = min(1.0, abs(macd_value - signal_value) / signal_value if signal_value != 0 else 0.0)
            return strength
        else:
            # Bearish trend
            strength = min(1.0, abs(signal_value - macd_value) / signal_value if signal_value != 0 else 0.0)
            return -strength
    
    def _calculate_ma_cross_score(self, df: pd.DataFrame) -> float:
        """
        Calculate trend score using moving average crossover
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Trend score between -1 and 1
        """
        close_prices = df['close']
        
        # Calculate moving averages
        short_ma = talib.EMA(close_prices, timeperiod=self.params["short_ma_period"])
        long_ma = talib.EMA(close_prices, timeperiod=self.params["long_ma_period"])
        
        # Get latest values
        short_current = short_ma.iloc[-1]
        long_current = long_ma.iloc[-1]
        
        # Calculate trend direction and strength
        if short_current > long_current:
            # Bullish trend
            strength = min(1.0, (short_current - long_current) / long_current if long_current != 0 else 0.0)
            return strength
        else:
            # Bearish trend
            strength = min(1.0, (long_current - short_current) / long_current if long_current != 0 else 0.0)
            return -strength
    
    def _calculate_adx_score(self, df: pd.DataFrame) -> float:
        """
        Calculate trend score using ADX (Average Directional Index)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Trend score between -1 and 1
        """
        # Calculate ADX and directional indicators
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Get latest values
        adx_value = adx.iloc[-1]
        plus_di_value = plus_di.iloc[-1]
        minus_di_value = minus_di.iloc[-1]
        
        # Normalize ADX to 0-1 range (ADX ranges from 0-100)
        adx_strength = min(1.0, adx_value / 50.0)
        
        # Determine trend direction
        if plus_di_value > minus_di_value:
            # Bullish trend
            return adx_strength
        else:
            # Bearish trend
            return -adx_strength
    
    def _calculate_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Calculate correlation matrix across symbols
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
        """
        # Extract return series for each symbol
        returns = {}
        for symbol, df in data.items():
            returns[symbol] = df['close'].pct_change().fillna(0)
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(returns)
        
        # Calculate correlation matrix
        lookback = min(len(returns_df), self.params["correlation_lookback"])
        corr_matrix = returns_df.iloc[-lookback:].corr()
        
        self.correlation_matrix = corr_matrix
    
    def _calculate_target_allocations(self) -> None:
        """
        Calculate target allocation for each symbol
        """
        # Reset allocations
        self.target_allocations = {symbol: 0.0 for symbol in self.symbols}
        
        # Minimum trend strength required for inclusion
        min_strength = self.params["min_trend_strength"]
        
        # Step 1: Filter symbols by trend strength
        trend_filtered_symbols = {}
        for symbol, score in self.trend_scores.items():
            if abs(score) >= min_strength:
                trend_filtered_symbols[symbol] = score
        
        # If no symbols pass the trend filter, allocate to cash
        if not trend_filtered_symbols:
            logger.info("No symbols meet trend strength criteria, allocating to cash")
            return
        
        # Step 2: Group symbols by asset class
        asset_class_symbols = {}
        for symbol, score in trend_filtered_symbols.items():
            asset_class = self.asset_class_map.get(symbol)
            if asset_class not in asset_class_symbols:
                asset_class_symbols[asset_class] = []
            asset_class_symbols[asset_class].append((symbol, score))
        
        # Step 3: Allocate capital to asset classes based on trend strength
        asset_class_allocations = {}
        total_score = sum(abs(self.asset_class_scores[ac]) for ac in asset_class_symbols.keys())
        
        # If total score is zero, allocate equally
        if total_score == 0:
            for asset_class in asset_class_symbols:
                asset_class_allocations[asset_class] = 1.0 / len(asset_class_symbols)
        else:
            for asset_class in asset_class_symbols:
                score = self.asset_class_scores.get(asset_class, 0.0)
                # Allocation proportional to absolute trend strength
                allocation = abs(score) / total_score if total_score > 0 else 0.0
                # Cap allocation per asset class
                allocation = min(allocation, self.params["max_asset_class_allocation"])
                asset_class_allocations[asset_class] = allocation
        
        # Normalize asset class allocations to sum to (1 - cash_allocation)
        total_allocation = sum(asset_class_allocations.values())
        cash_allocation = max(self.params["cash_allocation_min"], 1.0 - total_allocation)
        
        for asset_class in asset_class_allocations:
            if total_allocation > 0:
                asset_class_allocations[asset_class] *= (1.0 - cash_allocation) / total_allocation
        
        # Step 4: Allocate within each asset class
        for asset_class, allocation in asset_class_allocations.items():
            symbols_in_class = asset_class_symbols[asset_class]
            
            # Skip if no symbols in this class
            if not symbols_in_class:
                continue
            
            # Total class score (absolute value)
            class_total_score = sum(abs(score) for _, score in symbols_in_class)
            
            if class_total_score == 0:
                # Equal allocation if all scores are zero
                for symbol, _ in symbols_in_class:
                    self.target_allocations[symbol] = allocation / len(symbols_in_class)
            else:
                # Allocate proportionally to trend strength
                for symbol, score in symbols_in_class:
                    symbol_weight = abs(score) / class_total_score
                    
                    # Determine allocation direction (long/short) based on score sign
                    direction = 1 if score > 0 else -1
                    
                    # Calculate raw allocation
                    raw_allocation = direction * symbol_weight * allocation
                    
                    # Apply single symbol cap
                    capped_allocation = max(
                        -self.params["max_single_symbol_allocation"],
                        min(raw_allocation, self.params["max_single_symbol_allocation"])
                    )
                    
                    self.target_allocations[symbol] = capped_allocation
        
        # Step 5: Adjust for correlations if enabled
        if self.params["use_correlation_filter"] and self.correlation_matrix is not None:
            self._adjust_for_correlations()
        
        # Apply minimum allocation threshold
        min_allocation = self.params["min_allocation_percent"]
        for symbol in list(self.target_allocations.keys()):
            if 0 < abs(self.target_allocations[symbol]) < min_allocation:
                # Allocation too small, set to zero
                self.target_allocations[symbol] = 0.0
        
        # Final normalization to ensure sum equals 1 - cash_allocation
        active_allocations = {s: a for s, a in self.target_allocations.items() if a != 0}
        total_active = sum(abs(a) for a in active_allocations.values())
        
        if total_active > 0:
            scaling_factor = (1.0 - cash_allocation) / total_active
            for symbol in active_allocations:
                self.target_allocations[symbol] *= scaling_factor
        
        logger.info(f"Target allocations updated: {self.target_allocations}")
    
    def _adjust_for_correlations(self) -> None:
        """
        Adjust allocations based on correlation matrix to reduce risk
        """
        # Get symbols with non-zero allocations
        active_symbols = [s for s, a in self.target_allocations.items() if a != 0]
        
        if len(active_symbols) <= 1:
            # No correlation adjustments needed for single symbol
            return
        
        # Extract correlation sub-matrix for active symbols
        corr_sub = self.correlation_matrix.loc[active_symbols, active_symbols]
        
        # Calculate average correlation for each symbol
        avg_corr = corr_sub.mean()
        
        # Penalize highly correlated symbols
        for symbol in active_symbols:
            if symbol in avg_corr:
                # Get average correlation excluding self-correlation
                symbol_avg_corr = (avg_corr[symbol] * len(active_symbols) - 1) / (len(active_symbols) - 1)
                
                # Calculate penalty factor (higher correlation = bigger penalty)
                penalty = max(0, min(0.5, (symbol_avg_corr - 0.2) / 0.8))
                
                # Apply penalty to allocation
                self.target_allocations[symbol] *= (1.0 - penalty)
    
    def _get_account_value(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate current account value
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            
        Returns:
            Current account value
        """
        # This would typically retrieve the actual account value from the broker
        # For this example, we'll use a placeholder value
        account_value = 100000.0  # Placeholder
        
        return account_value
    
    def calculate_position_size(
        self,
        symbol: str,
        signal: SignalType,
        price: float,
        volatility: float,
        account_size: float
    ) -> float:
        """
        Calculate position size based on target allocation
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility
            account_size: Current account size
            
        Returns:
            Position size
        """
        # Get target allocation for this symbol
        target_allocation = self.target_allocations.get(symbol, 0.0)
        
        # Calculate position size based on allocation
        position_value = abs(target_allocation) * account_size
        
        # Convert position value to number of units/shares/contracts
        position_size = position_value / price if price > 0 else 0.0
        
        # Adjust direction based on signal and allocation sign
        if signal == SignalType.SHORT or target_allocation < 0:
            position_size = -position_size
        
        return position_size 