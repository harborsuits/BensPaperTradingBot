import pandas as pd
import numpy as np
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import joblib
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timedelta

# Local imports
from trading_bot.strategies.strategy_template import StrategyTemplate as Strategy
from trading_bot.utils.market_regime import MarketRegimeClassifier
from trading_bot.market_data.indicators import calculate_atr, calculate_rsi, calculate_macd

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enum representing different market regimes"""
    BULL_TREND = "bull_trend"               # Strong upward trend
    BEAR_TREND = "bear_trend"               # Strong downward trend
    CONSOLIDATION = "consolidation"         # Range-bound, low-volatility
    HIGH_VOLATILITY = "high_volatility"     # High volatility
    RECOVERY = "recovery"                   # Post-crash recovery
    ROTATION = "rotation"                   # Sector rotation
    RISK_OFF = "risk_off"                   # Risk aversion
    NORMAL = "normal"                       # Normal market conditions

class ParameterAdaptationType(Enum):
    """Types of parameter adaptation methods"""
    DISCRETE = "discrete"  # Discrete parameter sets for each regime
    WEIGHTED = "weighted"  # Weighted interpolation between parameter sets
    DYNAMIC = "dynamic"    # Fully dynamic adaptation based on regime metrics

class RegimeDetector:
    """
    Detects the current market regime based on market data
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        correlation_window: int = 30,
        volatility_threshold: float = 1.5,
        trend_strength_threshold: float = 0.5,
        breadth_threshold: float = 0.6,
        correlation_threshold: float = 0.7,
        lookback_period: int = 90
    ):
        """
        Initialize the regime detector
        
        Args:
            volatility_window: Period for volatility measurement
            trend_window: Period for trend detection
            correlation_window: Period for correlation calculation
            volatility_threshold: Threshold for high volatility detection
            trend_strength_threshold: Threshold for trend detection
            breadth_threshold: Market breadth threshold
            correlation_threshold: Threshold for correlation regime
            lookback_period: Maximum lookback period for data needed
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.correlation_window = correlation_window
        self.volatility_threshold = volatility_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.breadth_threshold = breadth_threshold
        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period
        
        # Store most recent regime detection
        self.current_regime = MarketRegime.NORMAL
        self.regime_start_date = None
        self.regime_data = {}
    
    def detect_regime(
        self, 
        market_data: pd.DataFrame, 
        benchmark: str = 'SPY',
        index_symbols: List[str] = None,
        economic_indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[MarketRegime, Dict[str, Any]]:
        """
        Detect the current market regime
        
        Args:
            market_data: DataFrame with market data (prices)
            benchmark: Benchmark symbol (e.g., 'SPY' for S&P 500)
            index_symbols: List of major index symbols to analyze
            economic_indicators: Optional economic indicator data
            
        Returns:
            Tuple of (MarketRegime, dict with regime metadata)
        """
        if index_symbols is None:
            index_symbols = [benchmark]
        
        # Ensure we have required data
        if benchmark not in market_data.columns:
            logger.warning(f"Benchmark {benchmark} not found in market data")
            return MarketRegime.NORMAL, {}
        
        # Calculate returns
        returns = market_data.pct_change().dropna()
        
        # Calculate volatility
        volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)  # Annualized
        current_vol = volatility.iloc[-1][benchmark]
        historical_vol = volatility.iloc[-self.lookback_period:-self.volatility_window][benchmark].mean()
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Calculate trend indicators
        sma_long = market_data.rolling(self.trend_window).mean()
        sma_short = market_data.rolling(self.trend_window // 2).mean()
        trend_direction = (sma_short.iloc[-1] / sma_long.iloc[-1] - 1) * 100  # Percentage difference
        
        # Calculate market breadth (if we have multiple indices)
        if len(index_symbols) > 1:
            symbols_above_sma = sum(1 for s in index_symbols if s in market_data.columns and 
                                  market_data.iloc[-1][s] > sma_long.iloc[-1][s])
            breadth = symbols_above_sma / len(index_symbols)
        else:
            breadth = 1.0 if market_data.iloc[-1][benchmark] > sma_long.iloc[-1][benchmark] else 0.0
        
        # Calculate correlation between assets
        if len(market_data.columns) > 1:
            correlation_matrix = returns.iloc[-self.correlation_window:].corr()
            # Get average correlation excluding self-correlations
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    correlations.append(correlation_matrix.iloc[i, j])
            avg_correlation = np.mean(correlations) if correlations else 0.0
        else:
            avg_correlation = 0.0
        
        # Store regime metadata
        regime_data = {
            'volatility': current_vol,
            'volatility_ratio': vol_ratio,
            'trend_direction': trend_direction[benchmark],
            'market_breadth': breadth,
            'average_correlation': avg_correlation,
            'detection_date': market_data.index[-1]
        }
        
        # Determine market regime
        if vol_ratio > self.volatility_threshold:
            # High volatility regime
            if trend_direction[benchmark] < -self.trend_strength_threshold:
                regime = MarketRegime.BEAR_TREND
            elif trend_direction[benchmark] > self.trend_strength_threshold:
                # Distinguish between recovery and bull market
                if self.current_regime == MarketRegime.BEAR_TREND:
                    regime = MarketRegime.RECOVERY
                else:
                    regime = MarketRegime.BULL_TREND
            else:
                regime = MarketRegime.HIGH_VOLATILITY
        else:
            # Normal or low volatility regime
            if trend_direction[benchmark] > self.trend_strength_threshold and breadth > self.breadth_threshold:
                regime = MarketRegime.BULL_TREND
            elif trend_direction[benchmark] < -self.trend_strength_threshold and breadth < (1 - self.breadth_threshold):
                regime = MarketRegime.BEAR_TREND
            elif avg_correlation > self.correlation_threshold:
                # High correlation often indicates risk-off
                regime = MarketRegime.RISK_OFF
            elif breadth < 0.4 or breadth > 0.6:
                # Some assets up, some down may indicate rotation
                regime = MarketRegime.ROTATION
            else:
                regime = MarketRegime.CONSOLIDATION
        
        # Incorporate economic indicators if available
        if economic_indicators is not None and not economic_indicators.empty:
            # Implementation depends on available economic indicators
            # For example, use recession indicators, yield curve, etc.
            pass
        
        # Update regime info
        if regime != self.current_regime:
            self.regime_start_date = market_data.index[-1]
        else:
            regime_data['days_in_regime'] = (market_data.index[-1] - self.regime_start_date).days
        
        self.current_regime = regime
        self.regime_data = regime_data
        
        return regime, regime_data

class RegimeAwareStrategy(Strategy):
    """
    A strategy that adapts its parameters based on the current market regime
    """
    
    def __init__(
        self,
        base_strategy: Strategy,
        regime_detector: Optional[RegimeDetector] = None,
        regime_parameter_sets: Optional[Dict[MarketRegime, Dict[str, Any]]] = None,
        regime_update_frequency: int = 5,  # Days between regime checks
        adaptation_delay: int = 1,  # Days to wait before adapting to a new regime
        rebalance_on_regime_change: bool = True,
        symbols: List[str] = None,
        benchmark: str = 'SPY'
    ):
        """
        Initialize the regime aware strategy wrapper
        
        Args:
            base_strategy: The base strategy to adapt
            regime_detector: Optional custom regime detector (created if None)
            regime_parameter_sets: Dictionary of parameter sets by regime
            regime_update_frequency: How often to check for regime changes (days)
            adaptation_delay: Days to wait before adapting to a new regime
            rebalance_on_regime_change: Whether to force rebalance on regime change
            symbols: List of symbols to trade
            benchmark: Benchmark symbol
        """
        super().__init__(symbols=symbols or base_strategy.symbols)
        
        self.base_strategy = base_strategy
        self.regime_detector = regime_detector or RegimeDetector()
        self.regime_parameter_sets = regime_parameter_sets or {}
        self.regime_update_frequency = regime_update_frequency
        self.adaptation_delay = adaptation_delay
        self.rebalance_on_regime_change = rebalance_on_regime_change
        self.benchmark = benchmark
        
        # State tracking
        self.current_regime = MarketRegime.NORMAL
        self.last_regime_check = None
        self.regime_transition_date = None
        self.force_rebalance = False
        
        # Ensure we have parameter sets for all regimes
        self._validate_parameter_sets()
    
    def _validate_parameter_sets(self):
        """Ensure we have parameter sets for all regimes"""
        if not self.regime_parameter_sets:
            # Initialize with default parameter sets
            self._initialize_default_parameter_sets()
        
        # Check for missing regimes
        for regime in MarketRegime:
            if regime not in self.regime_parameter_sets:
                logger.warning(f"No parameter set defined for regime {regime.value}. "
                              f"Using base strategy parameters.")
                # Use base strategy's current parameters as default
                self.regime_parameter_sets[regime] = self._get_current_parameters()
    
    def _initialize_default_parameter_sets(self):
        """Initialize default parameter sets for all regimes"""
        base_params = self._get_current_parameters()
        
        # Create conservative parameter set for high risk regimes
        conservative_params = base_params.copy()
        if 'max_position_size' in conservative_params:
            conservative_params['max_position_size'] *= 0.5
        if 'stop_loss_pct' in conservative_params:
            conservative_params['stop_loss_pct'] *= 0.8  # Tighter stops
        if 'risk_limit' in conservative_params:
            conservative_params['risk_limit'] *= 0.7
        
        # Create aggressive parameter set for bull markets
        aggressive_params = base_params.copy()
        if 'max_position_size' in aggressive_params:
            aggressive_params['max_position_size'] *= 1.2
        if 'target_profit_pct' in aggressive_params:
            aggressive_params['target_profit_pct'] *= 1.2  # Higher targets
        
        # Assign parameter sets to regimes
        self.regime_parameter_sets = {
            MarketRegime.NORMAL: base_params,
            MarketRegime.BULL_TREND: aggressive_params,
            MarketRegime.BEAR_TREND: conservative_params,
            MarketRegime.HIGH_VOLATILITY: conservative_params,
            MarketRegime.RISK_OFF: conservative_params,
            MarketRegime.CONSOLIDATION: base_params,
            MarketRegime.RECOVERY: base_params,
            MarketRegime.ROTATION: base_params
        }
        
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameters from the base strategy"""
        # This implementation depends on the base strategy's structure
        # A generic approach could be:
        params = {}
        for param_name, param_value in self.base_strategy.__dict__.items():
            # Skip built-in attributes and complex objects
            if not param_name.startswith('_') and not callable(param_value) and not isinstance(param_value, (pd.DataFrame, np.ndarray)):
                params[param_name] = param_value
        
        return params
    
    def _update_strategy_parameters(self, parameters: Dict[str, Any]):
        """Update the base strategy's parameters"""
        for param_name, param_value in parameters.items():
            if hasattr(self.base_strategy, param_name):
                setattr(self.base_strategy, param_name, param_value)
            else:
                logger.warning(f"Parameter {param_name} not found in base strategy")
    
    def _check_regime(self, market_data: pd.DataFrame) -> bool:
        """
        Check if we need to update the regime and do so if needed
        
        Args:
            market_data: Current market data
            
        Returns:
            True if regime changed, False otherwise
        """
        current_date = market_data.index[-1]
        
        # Check if we need to update regime
        if (self.last_regime_check is None or 
            (current_date - self.last_regime_check).days >= self.regime_update_frequency):
            
            # Detect current regime
            new_regime, regime_data = self.regime_detector.detect_regime(
                market_data, benchmark=self.benchmark, index_symbols=self.symbols
            )
            
            self.last_regime_check = current_date
            
            # Check if regime changed
            if new_regime != self.current_regime:
                logger.info(f"Market regime changed from {self.current_regime.value} to {new_regime.value}")
                logger.info(f"Regime data: {regime_data}")
                
                self.regime_transition_date = current_date
                self.current_regime = new_regime
                
                # Schedule parameter update after adaptation delay
                if self.adaptation_delay <= 0:
                    self._update_parameters_for_regime(new_regime)
                    
                    if self.rebalance_on_regime_change:
                        self.force_rebalance = True
                
                return True
        
        # Check if we need to apply delayed parameter update
        if (self.regime_transition_date is not None and 
            (current_date - self.regime_transition_date).days >= self.adaptation_delay):
            
            self._update_parameters_for_regime(self.current_regime)
            self.regime_transition_date = None
            
            if self.rebalance_on_regime_change:
                self.force_rebalance = True
        
        return False
    
    def _update_parameters_for_regime(self, regime: MarketRegime):
        """
        Update strategy parameters for the current regime
        
        Args:
            regime: Current market regime
        """
        if regime in self.regime_parameter_sets:
            logger.info(f"Updating strategy parameters for regime {regime.value}")
            parameters = self.regime_parameter_sets[regime]
            self._update_strategy_parameters(parameters)
        else:
            logger.warning(f"No parameter set found for regime {regime.value}")
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the current market regime
        
        Args:
            market_data: Market data with price information
            
        Returns:
            DataFrame with trading signals
        """
        # Check and update market regime if needed
        regime_changed = self._check_regime(market_data)
        
        # Generate signals using the base strategy
        signals = self.base_strategy.generate_signals(market_data)
        
        # If regime just changed and we need to rebalance, force all positions to update
        if self.force_rebalance:
            signals = self._force_signal_update(signals)
            self.force_rebalance = False
        
        return signals
    
    def _force_signal_update(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Force update of all signals after regime change
        
        Args:
            signals: Original signals DataFrame
            
        Returns:
            Updated signals DataFrame
        """
        # Implementation depends on signal format
        # This is a generic approach that assumes signals are numeric and
        # a value of 0 means no position
        for col in signals.columns:
            if col not in ['date', 'timestamp']:
                if signals.iloc[-1][col] != 0:
                    # Add a small offset to ensure the signal is seen as changed
                    signals.iloc[-1, signals.columns.get_loc(col)] += 0.0001
        
        return signals
    
    def calculate_position_size(self, signal: float, symbol: str, account_size: float) -> float:
        """
        Calculate position size based on the signal, symbol, and account size
        
        Args:
            signal: Trading signal (-1 to 1)
            symbol: The symbol to trade
            account_size: Current account size
            
        Returns:
            Position size in currency units
        """
        # Delegate to base strategy
        return self.base_strategy.calculate_position_size(signal, symbol, account_size)
    
    def add_regime_parameter_set(self, regime: MarketRegime, parameters: Dict[str, Any]):
        """
        Add or update parameter set for a specific regime
        
        Args:
            regime: MarketRegime to add parameters for
            parameters: Parameter dictionary
        """
        self.regime_parameter_sets[regime] = parameters
    
    def get_current_regime(self) -> Tuple[MarketRegime, Dict[str, Any]]:
        """
        Get the current market regime and metadata
        
        Returns:
            Tuple of (current regime, regime metadata)
        """
        return self.current_regime, self.regime_detector.regime_data

class AutoRegimeStrategy(RegimeAwareStrategy):
    """
    An extension of RegimeAwareStrategy that automatically learns optimal 
    parameters for each regime using historical performance
    """
    
    def __init__(
        self,
        base_strategy: Strategy,
        learning_rate: float = 0.1,
        min_regime_samples: int = 5,
        max_parameter_sets: int = 10,
        performance_function: Optional[Callable[[pd.Series], float]] = None,
        **kwargs
    ):
        """
        Initialize the auto regime strategy
        
        Args:
            base_strategy: The base strategy to adapt
            learning_rate: Rate to update parameters (0-1)
            min_regime_samples: Minimum samples before updating parameters
            max_parameter_sets: Maximum number of parameter sets to store per regime
            performance_function: Function to evaluate performance
            **kwargs: Additional arguments for RegimeAwareStrategy
        """
        super().__init__(base_strategy=base_strategy, **kwargs)
        
        self.learning_rate = learning_rate
        self.min_regime_samples = min_regime_samples
        self.max_parameter_sets = max_parameter_sets
        
        # Set default performance function if none provided
        if performance_function is None:
            # Default to Sharpe ratio
            self.performance_function = lambda returns: (
                returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            )
        else:
            self.performance_function = performance_function
        
        # Historical parameter sets and performance
        self.historical_parameters: Dict[MarketRegime, List[Tuple[Dict[str, Any], float]]] = {
            regime: [] for regime in MarketRegime
        }
        
        # Current test parameters
        self.test_parameters = None
        self.test_start_date = None
        self.test_returns = []
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals and learn from performance
        
        Args:
            market_data: Market data with price information
            
        Returns:
            DataFrame with trading signals
        """
        current_date = market_data.index[-1]
        
        # First, learn from past performance if we're testing parameters
        if self.test_parameters is not None and self.test_start_date is not None:
            # Calculate returns since test start
            if len(self.test_returns) > self.min_regime_samples:
                returns = pd.Series(self.test_returns)
                performance = self.performance_function(returns)
                
                # Store parameters and performance
                self._store_parameter_performance(self.current_regime, self.test_parameters, performance)
                
                # Reset test
                self.test_parameters = None
                self.test_start_date = None
                self.test_returns = []
                
                # Update parameters to best known set
                self._update_to_best_parameters(self.current_regime)
        
        # Check and update market regime if needed
        regime_changed = self._check_regime(market_data)
        
        # If regime changed, consider testing new parameters
        if regime_changed:
            # 20% chance to test new parameters (exploration)
            if np.random.random() < 0.2:
                self.test_parameters = self._generate_test_parameters(self.current_regime)
                self.test_start_date = current_date
                self._update_strategy_parameters(self.test_parameters)
            else:
                # Use best parameters for this regime
                self._update_to_best_parameters(self.current_regime)
        
        # Generate signals using the base strategy
        signals = super().generate_signals(market_data)
        
        # Store returns for performance tracking if we're testing
        if self.test_parameters is not None and len(market_data) > 1:
            # Calculate daily return
            daily_return = market_data.iloc[-1] / market_data.iloc[-2] - 1
            
            # Weight returns by signals
            weighted_return = 0
            for symbol in self.symbols:
                if symbol in signals.columns and symbol in daily_return.index:
                    signal_value = signals.iloc[-2][symbol] if len(signals) > 1 else 0
                    weighted_return += signal_value * daily_return[symbol]
            
            self.test_returns.append(weighted_return)
        
        return signals
    
    def _generate_test_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Generate test parameters based on the best known parameters for a regime
        
        Args:
            regime: The market regime
            
        Returns:
            Dictionary of test parameters
        """
        # Start with current parameters
        base_params = self._get_current_parameters()
        
        # If we have historical parameters, use the best one as base
        if regime in self.historical_parameters and self.historical_parameters[regime]:
            # Sort by performance (descending)
            sorted_params = sorted(
                self.historical_parameters[regime], 
                key=lambda x: x[1], 
                reverse=True
            )
            best_params = sorted_params[0][0]
            base_params = best_params.copy()
        
        # Randomly adjust parameters
        test_params = base_params.copy()
        for param_name, param_value in base_params.items():
            # Skip non-numeric parameters
            if not isinstance(param_value, (int, float)):
                continue
            
            # Adjust parameter with a random change
            adjustment = np.random.normal(0, 0.2)  # 20% standard deviation
            if isinstance(param_value, int):
                new_value = max(1, int(param_value * (1 + adjustment)))
                test_params[param_name] = new_value
            else:
                new_value = param_value * (1 + adjustment)
                test_params[param_name] = new_value
        
        return test_params
    
    def _store_parameter_performance(
        self, 
        regime: MarketRegime, 
        parameters: Dict[str, Any], 
        performance: float
    ):
        """
        Store parameters and their performance
        
        Args:
            regime: Market regime
            parameters: Parameter set
            performance: Performance metric value
        """
        # Add to historical parameters
        self.historical_parameters[regime].append((parameters.copy(), performance))
        
        # Keep only the top performing sets
        if len(self.historical_parameters[regime]) > self.max_parameter_sets:
            # Sort by performance (descending) and keep top sets
            self.historical_parameters[regime] = sorted(
                self.historical_parameters[regime],
                key=lambda x: x[1],
                reverse=True
            )[:self.max_parameter_sets]
        
        logger.info(f"Stored parameter performance for {regime.value}: {performance:.4f}")
    
    def _update_to_best_parameters(self, regime: MarketRegime):
        """
        Update strategy to use the best parameters for a regime
        
        Args:
            regime: Market regime
        """
        if regime in self.historical_parameters and self.historical_parameters[regime]:
            # Sort by performance (descending)
            sorted_params = sorted(
                self.historical_parameters[regime], 
                key=lambda x: x[1], 
                reverse=True
            )
            best_params = sorted_params[0][0]
            best_performance = sorted_params[0][1]
            
            logger.info(f"Updating to best parameters for {regime.value} "
                       f"(performance: {best_performance:.4f})")
            
            # Update strategy parameters
            self._update_strategy_parameters(best_params)
            
            # Update regime parameter set
            self.regime_parameter_sets[regime] = best_params 