import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects market regimes based on various metrics and classifies market periods.
    
    This module implements:
    1. Statistical regime detection using volatility, correlation, and trend metrics
    2. Multi-factor classification for identifying distinct market environments
    3. Hidden Markov Models for regime transitions
    4. Time series clustering for regime identification
    5. Regime shift detection for identifying transitions
    """
    
    def __init__(
        self,
        lookback_days: int = 120,
        volatility_window: int = 20,
        trend_window: int = 50,
        correlation_window: int = 30,
        num_regimes: int = 4,
        regime_persistence: int = 5,  # Minimum days for a regime to persist
        default_regime_labels: Dict[int, str] = None
    ):
        """
        Initialize the market regime detector.
        
        Args:
            lookback_days: Days of history to use for regime detection
            volatility_window: Window for volatility calculation
            trend_window: Window for trend calculation
            correlation_window: Window for correlation calculation
            num_regimes: Number of regimes to identify
            regime_persistence: Minimum days for a regime to persist
            default_regime_labels: Dictionary mapping regime numbers to descriptive labels
        """
        self.lookback_days = lookback_days
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.correlation_window = correlation_window
        self.num_regimes = num_regimes
        self.regime_persistence = regime_persistence
        
        # Set default regime labels if not provided
        self.regime_labels = default_regime_labels or {
            0: 'bullish',
            1: 'bearish',
            2: 'sideways',
            3: 'volatile'
        }
        
        # Initialize internal state
        self.market_data = None
        self.feature_data = None
        self.regime_history = None
        self.latest_regime = None
        self.regime_transitions = []
        
        # Initialize models
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=num_regimes, random_state=42)
        
        logger.info(f"Initialized MarketRegimeDetector with {num_regimes} regimes")
    
    def load_market_data(
        self,
        price_data: pd.DataFrame,
        symbol_col: str = 'symbol',
        date_col: str = 'date',
        price_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        benchmark_symbol: str = 'SPY'
    ) -> None:
        """
        Load market data for regime detection.
        
        Args:
            price_data: DataFrame with price data
            symbol_col: Column name for symbols
            date_col: Column name for dates
            price_col: Column name for prices
            volume_col: Column name for volume (optional)
            benchmark_symbol: Symbol to use as market benchmark
        """
        # Verify required columns
        required_cols = [symbol_col, date_col, price_col]
        if not all(col in price_data.columns for col in required_cols):
            logger.error(f"Price data missing required columns: {required_cols}")
            return
        
        # Process data
        if benchmark_symbol not in price_data[symbol_col].unique():
            logger.warning(f"Benchmark symbol {benchmark_symbol} not found in data")
        
        # Convert to wide format for easier analysis
        pivoted_data = price_data.pivot(
            index=date_col,
            columns=symbol_col,
            values=price_col
        )
        
        # Store price data
        self.market_data = {
            'prices': pivoted_data,
            'benchmark': benchmark_symbol
        }
        
        # If volume is available, pivot that too
        if volume_col and volume_col in price_data.columns:
            volume_pivoted = price_data.pivot(
                index=date_col,
                columns=symbol_col,
                values=volume_col
            )
            self.market_data['volume'] = volume_pivoted
        
        logger.info(f"Loaded market data with {len(pivoted_data)} time periods and {len(pivoted_data.columns)} symbols")
    
    def compute_features(self) -> pd.DataFrame:
        """
        Compute features for regime detection.
        
        Returns:
            DataFrame with computed features
        """
        if self.market_data is None or 'prices' not in self.market_data:
            logger.error("No market data loaded")
            return pd.DataFrame()
        
        price_data = self.market_data['prices']
        benchmark = self.market_data['benchmark']
        
        # Ensure the benchmark is in the data
        if benchmark not in price_data.columns:
            logger.error(f"Benchmark {benchmark} not found in price data")
            return pd.DataFrame()
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Feature 1: Volatility
        volatility = returns[benchmark].rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Feature 2: Trend strength (absolute value of z-score of current price vs moving average)
        ma = price_data[benchmark].rolling(window=self.trend_window).mean()
        trend_strength = ((price_data[benchmark] - ma) / price_data[benchmark].rolling(window=self.trend_window).std()).abs()
        
        # Feature 3: Average correlation between benchmark and other assets
        correlations = pd.DataFrame(index=returns.index)
        
        for symbol in returns.columns:
            if symbol != benchmark:
                correlations[symbol] = returns[benchmark].rolling(window=self.correlation_window).corr(returns[symbol])
        
        avg_correlation = correlations.mean(axis=1)
        
        # Feature 4: Return dispersion (cross-sectional standard deviation of returns)
        return_dispersion = returns.std(axis=1)
        
        # Feature 5: Mean reversion tendency (autocorrelation of returns)
        mean_reversion = returns[benchmark].rolling(window=self.correlation_window).apply(
            lambda x: pd.Series(x).autocorr(lag=1)
        )
        
        # Feature 6: Market efficiency (Hurst exponent approximation)
        hurst_exponent = returns[benchmark].rolling(window=max(30, self.correlation_window)).apply(
            lambda x: self._calculate_hurst_exponent(x)
        )
        
        # Feature 7: VIX-like measure (rolling volatility of volatility)
        vix_proxy = volatility.rolling(window=self.volatility_window).std()
        
        # Combine features
        features = pd.DataFrame({
            'volatility': volatility,
            'trend_strength': trend_strength,
            'avg_correlation': avg_correlation,
            'return_dispersion': return_dispersion,
            'mean_reversion': mean_reversion,
            'hurst_exponent': hurst_exponent,
            'vix_proxy': vix_proxy
        })
        
        # Clean up features
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Store features
        self.feature_data = features
        
        logger.info(f"Computed features for regime detection: {features.shape[0]} time periods, {features.shape[1]} features")
        
        return features
    
    def _calculate_hurst_exponent(self, returns: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent for returns series.
        
        Args:
            returns: Series of returns
            max_lag: Maximum lag for calculation
            
        Returns:
            Estimated Hurst exponent
        """
        # Ensure we have enough data
        if len(returns) < max_lag * 2:
            return np.nan
        
        # Convert to numpy array
        ts = returns.values
        
        # Calculate range/standard deviation for different lags
        lags = range(2, min(max_lag, len(ts) // 2))
        
        # Calculate statistics for each lag
        stats = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            # Calculate standard deviation of differences
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            
            # Convert to numpy array
            tau = np.array(tau)
            
            # Calculate statistics
            stats[i] = tau[i]
        
        # Avoid log(0)
        stats = stats[stats > 0]
        lags = np.array(lags)[stats > 0]
        
        if len(stats) < 2:
            return np.nan
        
        # Linear fit to estimate Hurst exponent
        try:
            hurst = np.polyfit(np.log(lags), np.log(stats), 1)[0]
            return hurst
        except:
            return np.nan
    
    def detect_regimes(self, method: str = 'kmeans') -> pd.Series:
        """
        Detect market regimes using the specified method.
        
        Args:
            method: Method for regime detection ('kmeans', 'hmm', 'manual')
            
        Returns:
            Series with detected regimes
        """
        if self.feature_data is None:
            logger.warning("No feature data available. Computing features first.")
            self.compute_features()
            
        if self.feature_data is None or self.feature_data.empty:
            logger.error("Failed to compute features for regime detection")
            return pd.Series()
        
        features = self.feature_data.copy()
        
        # Select detection method
        if method == 'kmeans':
            regimes = self._detect_regimes_kmeans(features)
        elif method == 'hmm':
            regimes = self._detect_regimes_hmm(features)
        elif method == 'manual':
            regimes = self._detect_regimes_manual(features)
        else:
            logger.error(f"Unknown regime detection method: {method}")
            return pd.Series()
        
        # Apply regime persistence filter to avoid frequent switching
        if self.regime_persistence > 1:
            regimes = self._apply_regime_persistence(regimes)
        
        # Store regime history
        self.regime_history = regimes
        
        # Store latest regime
        if not regimes.empty:
            self.latest_regime = regimes.iloc[-1]
        
        # Detect regime transitions
        self._detect_regime_transitions(regimes)
        
        logger.info(f"Detected {self.num_regimes} market regimes using {method} method")
        
        return regimes
    
    def _detect_regimes_kmeans(self, features: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes using K-means clustering.
        
        Args:
            features: DataFrame with computed features
            
        Returns:
            Series with detected regimes
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit KMeans
        labels = self.cluster_model.fit_predict(scaled_features)
        
        # Create Series with regime labels
        regimes = pd.Series(labels, index=features.index)
        
        return regimes
    
    def _detect_regimes_hmm(self, features: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes using Hidden Markov Models.
        
        Args:
            features: DataFrame with computed features
            
        Returns:
            Series with detected regimes
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.error("hmmlearn not installed. Please install it for HMM-based regime detection.")
            return pd.Series()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=self.num_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        model.fit(scaled_features)
        
        # Predict hidden states
        hidden_states = model.predict(scaled_features)
        
        # Create Series with regime labels
        regimes = pd.Series(hidden_states, index=features.index)
        
        return regimes
    
    def _detect_regimes_manual(self, features: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes using manual rules.
        
        Args:
            features: DataFrame with computed features
            
        Returns:
            Series with detected regimes
        """
        # Initialize regimes
        regimes = pd.Series(index=features.index, dtype='int')
        
        # Rule 1: High volatility indicates volatility regime
        high_vol = features['volatility'] > features['volatility'].quantile(0.75)
        high_vix = features['vix_proxy'] > features['vix_proxy'].quantile(0.75)
        regimes[high_vol & high_vix] = 3  # Volatile regime
        
        # Rule 2: Strong trend indicates bullish/bearish regime
        if 'prices' in self.market_data:
            price_data = self.market_data['prices']
            benchmark = self.market_data['benchmark']
            
            # Calculate benchmark returns
            benchmark_returns = price_data[benchmark].pct_change().dropna()
            
            # Calculate trend direction using moving average
            ma = benchmark_returns.rolling(window=self.trend_window).mean()
            
            # Strong positive trend = bullish
            strong_positive = (features['trend_strength'] > features['trend_strength'].quantile(0.75)) & (ma > 0)
            regimes[strong_positive] = 0  # Bullish regime
            
            # Strong negative trend = bearish
            strong_negative = (features['trend_strength'] > features['trend_strength'].quantile(0.75)) & (ma < 0)
            regimes[strong_negative] = 1  # Bearish regime
        
        # Rule 3: Mean reversion with low volatility indicates sideways regime
        mean_reverting = features['mean_reversion'] < features['mean_reversion'].quantile(0.25)
        low_vol = features['volatility'] < features['volatility'].quantile(0.5)
        regimes[mean_reverting & low_vol & regimes.isna()] = 2  # Sideways regime
        
        # Fill remaining NaNs with most common regime
        if regimes.isna().any():
            most_common = regimes.value_counts().idxmax()
            regimes = regimes.fillna(most_common)
        
        return regimes
    
    def _apply_regime_persistence(self, regimes: pd.Series) -> pd.Series:
        """
        Apply regime persistence filter to avoid frequent regime switching.
        
        Args:
            regimes: Series with detected regimes
            
        Returns:
            Series with filtered regimes
        """
        # Convert to numpy for faster processing
        regime_array = regimes.values
        
        # Initialize filtered array
        filtered = np.zeros_like(regime_array)
        
        # First value is always the same
        filtered[0] = regime_array[0]
        
        # Current regime and its duration
        current_regime = regime_array[0]
        current_duration = 1
        
        # Process the rest of the array
        for i in range(1, len(regime_array)):
            if regime_array[i] == current_regime:
                # Same regime, increment duration
                current_duration += 1
                filtered[i] = current_regime
            else:
                # Different regime
                if current_duration < self.regime_persistence:
                    # Too short, keep previous regime
                    filtered[i] = current_regime
                    current_duration += 1
                else:
                    # Long enough, switch to new regime
                    current_regime = regime_array[i]
                    current_duration = 1
                    filtered[i] = current_regime
        
        # Convert back to Series
        filtered_regimes = pd.Series(filtered, index=regimes.index)
        
        return filtered_regimes
    
    def _detect_regime_transitions(self, regimes: pd.Series) -> None:
        """
        Detect regime transitions and store them.
        
        Args:
            regimes: Series with detected regimes
        """
        if regimes.empty:
            return
        
        # Find regime changes
        changes = regimes.diff().fillna(0) != 0
        change_points = regimes[changes]
        
        # Record transitions
        transitions = []
        
        for date, regime in change_points.items():
            # Get regime label
            regime_label = self.regime_labels.get(regime, f"Regime {regime}")
            
            transitions.append({
                'date': date,
                'regime': int(regime),
                'regime_label': regime_label
            })
        
        self.regime_transitions = transitions
        
        if transitions:
            logger.info(f"Detected {len(transitions)} regime transitions")
    
    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get the current market regime.
        
        Returns:
            Dictionary with current regime information
        """
        if self.regime_history is None or self.latest_regime is None:
            logger.warning("No regime history available")
            return {}
        
        regime = int(self.latest_regime)
        regime_label = self.regime_labels.get(regime, f"Regime {regime}")
        
        # Calculate duration of current regime
        if self.regime_transitions:
            last_transition = self.regime_transitions[-1]['date']
            last_date = self.regime_history.index[-1]
            duration = (last_date - last_transition).days
        else:
            duration = len(self.regime_history)
        
        # Get features for current regime
        if self.feature_data is not None:
            current_features = self.feature_data.iloc[-1].to_dict()
        else:
            current_features = {}
        
        return {
            'regime': regime,
            'label': regime_label,
            'duration_days': duration,
            'features': current_features
        }
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each regime.
        
        Returns:
            Dictionary with regime statistics
        """
        if self.regime_history is None or self.market_data is None:
            logger.warning("No regime history or market data available")
            return {}
        
        price_data = self.market_data['prices']
        benchmark = self.market_data['benchmark']
        
        # Calculate benchmark returns
        returns = price_data[benchmark].pct_change().dropna()
        
        # Align returns with regimes
        aligned_data = pd.DataFrame({
            'returns': returns,
            'regime': self.regime_history
        }).dropna()
        
        # Calculate statistics by regime
        regime_stats = {}
        
        for regime in range(self.num_regimes):
            regime_returns = aligned_data[aligned_data['regime'] == regime]['returns']
            
            if len(regime_returns) > 0:
                regime_label = self.regime_labels.get(regime, f"Regime {regime}")
                
                stats = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'median_return': regime_returns.median(),
                    'volatility': regime_returns.std(),
                    'max_return': regime_returns.max(),
                    'min_return': regime_returns.min(),
                    'positive_days': (regime_returns > 0).mean(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
                }
                
                regime_stats[regime_label] = stats
        
        return regime_stats
    
    def plot_regime_history(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the regime history.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot image
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.error("Matplotlib not installed. Please install it for plotting.")
            return
        
        if self.regime_history is None:
            logger.warning("No regime history available")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create regime label mapping
        regime_colors = {
            0: 'green',      # Bullish
            1: 'red',        # Bearish
            2: 'gray',       # Sideways
            3: 'orange'      # Volatile
        }
        
        # Create scatter plot with colored points
        for regime in range(self.num_regimes):
            regime_label = self.regime_labels.get(regime, f"Regime {regime}")
            regime_data = self.regime_history[self.regime_history == regime]
            
            if not regime_data.empty:
                plt.scatter(
                    regime_data.index,
                    [regime] * len(regime_data),
                    c=regime_colors.get(regime, 'blue'),
                    label=regime_label,
                    s=30,
                    alpha=0.7
                )
        
        # Plot benchmark price if available
        if self.market_data is not None and 'prices' in self.market_data:
            price_data = self.market_data['prices']
            benchmark = self.market_data['benchmark']
            
            if benchmark in price_data.columns:
                # Normalize benchmark to fit on same scale
                benchmark_prices = price_data[benchmark].reindex(self.regime_history.index)
                normalized_prices = (benchmark_prices - benchmark_prices.min()) / (benchmark_prices.max() - benchmark_prices.min()) * (self.num_regimes - 1)
                
                # Add secondary y-axis for benchmark
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # Plot benchmark
                ax2.plot(
                    normalized_prices.index,
                    normalized_prices.values,
                    'k--',
                    alpha=0.3,
                    label=benchmark
                )
                
                # Set y-axis label
                ax2.set_ylabel(benchmark)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Market Regime')
        plt.title('Market Regime History')
        
        # Format x-axis as dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add y-axis ticks for regimes
        plt.yticks(
            range(self.num_regimes),
            [self.regime_labels.get(i, f"Regime {i}") for i in range(self.num_regimes)]
        )
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def get_regime_transition_matrix(self) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix.
        
        Returns:
            DataFrame with transition probabilities
        """
        if self.regime_history is None:
            logger.warning("No regime history available")
            return pd.DataFrame()
        
        # Get regime values
        regimes = self.regime_history.values
        
        # Initialize transition count matrix
        transition_counts = np.zeros((self.num_regimes, self.num_regimes))
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transition_counts[int(from_regime), int(to_regime)] += 1
        
        # Convert to probabilities
        transition_probs = np.zeros_like(transition_counts)
        
        for i in range(self.num_regimes):
            row_sum = transition_counts[i, :].sum()
            if row_sum > 0:
                transition_probs[i, :] = transition_counts[i, :] / row_sum
        
        # Create DataFrame
        transition_df = pd.DataFrame(
            transition_probs,
            index=[self.regime_labels.get(i, f"Regime {i}") for i in range(self.num_regimes)],
            columns=[self.regime_labels.get(i, f"Regime {i}") for i in range(self.num_regimes)]
        )
        
        return transition_df
    
    def predict_next_regime(self) -> Dict[str, Any]:
        """
        Predict the most likely next regime based on transition matrix.
        
        Returns:
            Dictionary with prediction details
        """
        if self.regime_history is None or self.latest_regime is None:
            logger.warning("No regime history available")
            return {}
        
        # Get transition matrix
        transition_matrix = self.get_regime_transition_matrix()
        
        if transition_matrix.empty:
            return {}
        
        # Current regime
        current_regime = int(self.latest_regime)
        
        # Get transition probabilities from current regime
        next_probs = transition_matrix.iloc[current_regime].values
        
        # Find most likely next regime
        next_regime = np.argmax(next_probs)
        next_prob = next_probs[next_regime]
        
        # Get regimes sorted by probability
        sorted_indices = np.argsort(next_probs)[::-1]
        sorted_probs = next_probs[sorted_indices]
        
        next_regimes = []
        
        for i, idx in enumerate(sorted_indices):
            if sorted_probs[i] > 0:
                next_regimes.append({
                    'regime': int(idx),
                    'label': self.regime_labels.get(int(idx), f"Regime {idx}"),
                    'probability': sorted_probs[i]
                })
        
        return {
            'current_regime': current_regime,
            'current_label': self.regime_labels.get(current_regime, f"Regime {current_regime}"),
            'next_regime': int(next_regime),
            'next_label': self.regime_labels.get(int(next_regime), f"Regime {next_regime}"),
            'next_probability': next_prob,
            'all_next_regimes': next_regimes
        }
    
    def save_regime_data(self, filepath: str) -> bool:
        """
        Save regime detection results to a file.
        
        Args:
            filepath: Path to save the data
            
        Returns:
            Success flag
        """
        if self.regime_history is None:
            logger.warning("No regime history to save")
            return False
        
        try:
            # Prepare data
            regime_data = pd.DataFrame({
                'regime': self.regime_history,
                'regime_label': self.regime_history.map(lambda x: self.regime_labels.get(x, f"Regime {x}"))
            })
            
            # Add features if available
            if self.feature_data is not None:
                for col in self.feature_data.columns:
                    regime_data[col] = self.feature_data[col]
            
            # Save to CSV
            regime_data.to_csv(filepath)
            
            logger.info(f"Saved regime data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving regime data: {e}")
            return False
    
    def load_regime_data(self, filepath: str) -> bool:
        """
        Load regime detection results from a file.
        
        Args:
            filepath: Path to load the data from
            
        Returns:
            Success flag
        """
        try:
            # Load data
            regime_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            if 'regime' not in regime_data.columns:
                logger.error("Loaded data does not contain regime column")
                return False
            
            # Extract regimes
            self.regime_history = regime_data['regime']
            
            # Set latest regime
            if not self.regime_history.empty:
                self.latest_regime = self.regime_history.iloc[-1]
            
            # Extract features if available
            feature_cols = [col for col in regime_data.columns if col not in ['regime', 'regime_label']]
            
            if feature_cols:
                self.feature_data = regime_data[feature_cols]
            
            # Detect regime transitions
            self._detect_regime_transitions(self.regime_history)
            
            logger.info(f"Loaded regime data from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading regime data: {e}")
            return False 