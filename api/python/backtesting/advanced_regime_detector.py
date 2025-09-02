import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from trading_bot.backtesting.regime_detector import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolatilityRegime:
    """Enum-like class for volatility regimes"""
    LOW = "low_volatility"
    NORMAL = "normal_volatility"
    HIGH = "high_volatility"
    EXTREME = "extreme_volatility"

class CorrelationRegime:
    """Enum-like class for correlation regimes"""
    HIGH_POSITIVE = "high_positive_correlation"
    MODERATE_POSITIVE = "moderate_positive_correlation"
    UNCORRELATED = "uncorrelated"
    MODERATE_NEGATIVE = "moderate_negative_correlation"
    HIGH_NEGATIVE = "high_negative_correlation"

class SectorRotationPhase:
    """Enum-like class for sector rotation phases"""
    EARLY_EXPANSION = "early_expansion"  # Consumer Discretionary, Technology, Industrials
    LATE_EXPANSION = "late_expansion"    # Energy, Materials, Industrials
    EARLY_CONTRACTION = "early_contraction"  # Healthcare, Consumer Staples, Utilities
    LATE_CONTRACTION = "late_contraction"    # Financials, Technology, Consumer Discretionary
    RECOVERY = "recovery"                # Technology, Financials, Real Estate

class AdvancedRegimeDetector(MarketRegimeDetector):
    """
    Advanced market regime detector that extends the base MarketRegimeDetector with:
    
    1. Multiple timeframe analysis to identify conflicting trends
    2. Volatility-based regime classification
    3. Correlation-based regime detection across asset classes
    4. Sector rotation analysis for equity trading
    """
    
    def __init__(
        self,
        lookback_days: int = 120,
        timeframes: List[str] = None,
        volatility_windows: Dict[str, int] = None,
        trend_windows: Dict[str, int] = None,
        correlation_window: int = 30,
        num_regimes: int = 4,
        regime_persistence: int = 5,
        vix_data: pd.Series = None,
        default_regime_labels: Dict[int, str] = None
    ):
        """
        Initialize the advanced market regime detector.
        
        Args:
            lookback_days: Days of history to use for regime detection
            timeframes: List of timeframes to analyze (e.g., ["daily", "weekly", "monthly"])
            volatility_windows: Dictionary mapping timeframes to volatility windows
            trend_windows: Dictionary mapping timeframes to trend windows
            correlation_window: Window for correlation calculation
            num_regimes: Number of regimes to identify
            regime_persistence: Minimum days for a regime to persist
            vix_data: Optional Series with VIX index values
            default_regime_labels: Dictionary mapping regime numbers to descriptive labels
        """
        # Initialize base class with longest timeframe parameters
        super().__init__(
            lookback_days=lookback_days,
            volatility_window=20,  # Default, will be overridden per timeframe
            trend_window=50,       # Default, will be overridden per timeframe 
            correlation_window=correlation_window,
            num_regimes=num_regimes,
            regime_persistence=regime_persistence,
            default_regime_labels=default_regime_labels
        )
        
        # Set default timeframes if not provided
        self.timeframes = timeframes or ["daily", "weekly", "monthly"]
        
        # Set default volatility windows if not provided
        self.volatility_windows = volatility_windows or {
            "daily": 20,
            "weekly": 12,
            "monthly": 6
        }
        
        # Set default trend windows if not provided
        self.trend_windows = trend_windows or {
            "daily": 50,
            "weekly": 20,
            "monthly": 12
        }
        
        # Store VIX data if provided
        self.vix_data = vix_data
        
        # Initialize containers for timeframe-specific data
        self.market_data_by_timeframe = {}
        self.feature_data_by_timeframe = {}
        self.regime_history_by_timeframe = {}
        
        # Initialize containers for advanced regime detection
        self.volatility_regimes = None
        self.correlation_regimes = None
        self.trend_conflict_regimes = None
        self.sector_rotation_phases = None
        
        # Sector mapping for rotation analysis
        self.sector_etf_mapping = {
            "XLK": "technology",
            "XLF": "financials",
            "XLV": "healthcare",
            "XLE": "energy",
            "XLY": "consumer_discretionary",
            "XLP": "consumer_staples",
            "XLI": "industrials",
            "XLB": "materials",
            "XLU": "utilities",
            "XLRE": "real_estate",
            "XLC": "communication_services"
        }
        
        logger.info(f"Initialized AdvancedRegimeDetector with {num_regimes} regimes across {len(self.timeframes)} timeframes")

    def load_market_data_multi_timeframe(
        self,
        price_data_by_timeframe: Dict[str, pd.DataFrame],
        symbol_col: str = 'symbol',
        date_col: str = 'date',
        price_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        benchmark_symbol: str = 'SPY'
    ) -> None:
        """
        Load market data for multiple timeframes.
        
        Args:
            price_data_by_timeframe: Dictionary mapping timeframes to price DataFrames
            symbol_col: Column name for symbols
            date_col: Column name for dates
            price_col: Column name for prices
            volume_col: Column name for volume (optional)
            benchmark_symbol: Symbol to use as market benchmark
        """
        for timeframe, price_data in price_data_by_timeframe.items():
            # Load data for this timeframe using the base class method
            super().load_market_data(
                price_data=price_data,
                symbol_col=symbol_col,
                date_col=date_col,
                price_col=price_col,
                volume_col=volume_col,
                benchmark_symbol=benchmark_symbol
            )
            
            # Store the market data for this timeframe
            self.market_data_by_timeframe[timeframe] = self.market_data.copy()
        
        # Set the primary (daily) timeframe data as the default
        primary_timeframe = self.timeframes[0]
        if primary_timeframe in self.market_data_by_timeframe:
            self.market_data = self.market_data_by_timeframe[primary_timeframe]
        
        logger.info(f"Loaded market data for {len(self.market_data_by_timeframe)} timeframes")

    def compute_features_multi_timeframe(self) -> Dict[str, pd.DataFrame]:
        """
        Compute features for all timeframes.
        
        Returns:
            Dictionary mapping timeframes to feature DataFrames
        """
        for timeframe, market_data in self.market_data_by_timeframe.items():
            # Temporarily set market data to current timeframe
            self.market_data = market_data
            
            # Set appropriate windows for this timeframe
            self.volatility_window = self.volatility_windows.get(timeframe, 20)
            self.trend_window = self.trend_windows.get(timeframe, 50)
            
            # Compute features using base class method
            features = super().compute_features()
            
            # Store features for this timeframe
            self.feature_data_by_timeframe[timeframe] = features
        
        # Reset to primary timeframe data
        primary_timeframe = self.timeframes[0]
        if primary_timeframe in self.market_data_by_timeframe:
            self.market_data = self.market_data_by_timeframe[primary_timeframe]
            self.feature_data = self.feature_data_by_timeframe[primary_timeframe]
        
        logger.info(f"Computed features for {len(self.feature_data_by_timeframe)} timeframes")
        
        return self.feature_data_by_timeframe

    def detect_regimes_multi_timeframe(self, method: str = 'kmeans') -> Dict[str, pd.Series]:
        """
        Detect regimes for all timeframes.
        
        Args:
            method: Method for regime detection ('kmeans', 'hmm', 'manual')
            
        Returns:
            Dictionary mapping timeframes to regime Series
        """
        for timeframe, feature_data in self.feature_data_by_timeframe.items():
            # Temporarily set feature data to current timeframe
            self.feature_data = feature_data
            
            # Set market data to current timeframe
            self.market_data = self.market_data_by_timeframe[timeframe]
            
            # Detect regimes using base class method
            regimes = super().detect_regimes(method=method)
            
            # Store regimes for this timeframe
            self.regime_history_by_timeframe[timeframe] = regimes
        
        # Reset to primary timeframe data
        primary_timeframe = self.timeframes[0]
        if primary_timeframe in self.feature_data_by_timeframe:
            self.feature_data = self.feature_data_by_timeframe[primary_timeframe]
            self.market_data = self.market_data_by_timeframe[primary_timeframe]
            self.regime_history = self.regime_history_by_timeframe[primary_timeframe]
        
        logger.info(f"Detected regimes for {len(self.regime_history_by_timeframe)} timeframes using {method} method")
        
        return self.regime_history_by_timeframe

    def detect_trend_conflicts(self) -> pd.DataFrame:
        """
        Detect conflicts between trends across different timeframes.
        
        Returns:
            DataFrame with trend conflict information
        """
        if not self.regime_history_by_timeframe or len(self.regime_history_by_timeframe) < 2:
            logger.warning("Need at least two timeframes to detect trend conflicts")
            return pd.DataFrame()
        
        # Get the primary timeframe (usually daily)
        primary_timeframe = self.timeframes[0]
        primary_regimes = self.regime_history_by_timeframe.get(primary_timeframe)
        
        if primary_regimes is None:
            logger.warning(f"No regime data for primary timeframe {primary_timeframe}")
            return pd.DataFrame()
        
        # Initialize conflict detection DataFrame
        trend_conflicts = pd.DataFrame(index=primary_regimes.index)
        trend_conflicts['primary_regime'] = primary_regimes
        trend_conflicts['primary_regime_label'] = primary_regimes.map(self.regime_labels)
        
        # Track bullish and bearish signals across timeframes
        bullish_timeframes = []
        bearish_timeframes = []
        neutral_timeframes = []
        
        # Identify bullish/bearish regimes for each timeframe
        for timeframe, regimes in self.regime_history_by_timeframe.items():
            # Skip primary timeframe as it's already processed
            if timeframe == primary_timeframe:
                continue
            
            # Align to primary timeframe dates
            aligned_regimes = regimes.reindex(primary_regimes.index, method='ffill')
            
            # Add to trend conflicts DataFrame
            trend_conflicts[f'{timeframe}_regime'] = aligned_regimes
            trend_conflicts[f'{timeframe}_regime_label'] = aligned_regimes.map(self.regime_labels)
            
            # Calculate direction (bullish/bearish/neutral)
            # This assumes regime 0 is bullish, 1 is bearish, and others are neutral
            # Adjust as needed based on your regime labels
            for date, regime in aligned_regimes.items():
                if date in trend_conflicts.index:
                    regime_label = self.regime_labels.get(regime, "")
                    if "bullish" in regime_label.lower():
                        if timeframe not in bullish_timeframes:
                            bullish_timeframes.append(timeframe)
                    elif "bearish" in regime_label.lower():
                        if timeframe not in bearish_timeframes:
                            bearish_timeframes.append(timeframe)
                    else:
                        if timeframe not in neutral_timeframes:
                            neutral_timeframes.append(timeframe)
        
        # Determine conflict status
        def get_conflict_status(row):
            # Count regimes by type
            bullish_count = sum(1 for col in trend_conflicts.columns if 'regime_label' in col 
                               and 'bullish' in str(row[col]).lower())
            bearish_count = sum(1 for col in trend_conflicts.columns if 'regime_label' in col 
                               and 'bearish' in str(row[col]).lower())
            
            # Clear trend if all aligned
            if bullish_count > 0 and bearish_count == 0:
                return "aligned_bullish"
            elif bearish_count > 0 and bullish_count == 0:
                return "aligned_bearish"
            # Strong conflict if we have both bullish and bearish
            elif bullish_count > 0 and bearish_count > 0:
                return "strong_conflict"
            # No clear direction
            else:
                return "indeterminate"
        
        trend_conflicts['conflict_status'] = trend_conflicts.apply(get_conflict_status, axis=1)
        
        # Store results
        self.trend_conflict_regimes = trend_conflicts
        
        logger.info(f"Detected trend conflicts across {len(self.regime_history_by_timeframe)} timeframes")
        
        return trend_conflicts

    def classify_volatility_regimes(
        self, 
        vix_thresholds: Dict[str, Tuple[float, float]] = None
    ) -> pd.Series:
        """
        Classify volatility regimes based on VIX or volatility metrics.
        
        Args:
            vix_thresholds: Dictionary mapping regime names to (lower, upper) thresholds
                            Default thresholds are based on historical VIX percentiles
            
        Returns:
            Series with volatility regime labels
        """
        # Set default VIX thresholds if not provided
        if vix_thresholds is None:
            vix_thresholds = {
                VolatilityRegime.LOW: (0, 15),
                VolatilityRegime.NORMAL: (15, 25),
                VolatilityRegime.HIGH: (25, 35),
                VolatilityRegime.EXTREME: (35, float('inf'))
            }
        
        # Get primary timeframe data
        primary_timeframe = self.timeframes[0]
        feature_data = self.feature_data_by_timeframe.get(primary_timeframe)
        
        if feature_data is None or feature_data.empty:
            logger.warning("No feature data available for volatility regime classification")
            return pd.Series()
        
        # Use VIX data if provided, otherwise use calculated volatility
        if self.vix_data is not None and not self.vix_data.empty:
            # Use external VIX data
            vix_series = self.vix_data
            # Align with feature data dates
            vix_series = vix_series.reindex(feature_data.index, method='ffill')
        else:
            # Use calculated volatility proxy
            vix_series = feature_data['vix_proxy'] * 100  # Scale to VIX-like values
        
        # Classify regimes based on thresholds
        vol_regimes = pd.Series(index=vix_series.index, dtype='object')
        
        for regime, (lower, upper) in vix_thresholds.items():
            mask = (vix_series >= lower) & (vix_series < upper)
            vol_regimes[mask] = regime
        
        # Fill any NaN values with NORMAL regime
        vol_regimes = vol_regimes.fillna(VolatilityRegime.NORMAL)
        
        # Store volatility regimes
        self.volatility_regimes = vol_regimes
        
        logger.info(f"Classified volatility regimes with {len(vol_regimes)} periods")
        
        return vol_regimes

    def detect_correlation_regimes(
        self, 
        correlation_thresholds: Dict[str, Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Detect correlation regimes between asset classes.
        
        Args:
            correlation_thresholds: Dictionary mapping regime names to (lower, upper) thresholds
            
        Returns:
            DataFrame with correlation regimes
        """
        # Set default correlation thresholds if not provided
        if correlation_thresholds is None:
            correlation_thresholds = {
                CorrelationRegime.HIGH_POSITIVE: (0.7, 1.0),
                CorrelationRegime.MODERATE_POSITIVE: (0.3, 0.7),
                CorrelationRegime.UNCORRELATED: (-0.3, 0.3),
                CorrelationRegime.MODERATE_NEGATIVE: (-0.7, -0.3),
                CorrelationRegime.HIGH_NEGATIVE: (-1.0, -0.7)
            }
        
        # Get primary timeframe data
        primary_timeframe = self.timeframes[0]
        market_data = self.market_data_by_timeframe.get(primary_timeframe)
        
        if market_data is None or 'prices' not in market_data:
            logger.warning("No market data available for correlation regime detection")
            return pd.DataFrame()
        
        price_data = market_data['prices']
        benchmark = market_data['benchmark']
        
        # Ensure benchmark is in data
        if benchmark not in price_data.columns:
            logger.error(f"Benchmark {benchmark} not found in price data")
            return pd.DataFrame()
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Map symbols to asset classes if available
        # This is a simple implementation - you might want to expand this with a more
        # comprehensive mapping of symbols to asset classes
        asset_class_returns = {}
        
        # If we have sector ETFs, use them as asset classes
        for symbol in returns.columns:
            if symbol in self.sector_etf_mapping:
                asset_class = self.sector_etf_mapping[symbol]
                asset_class_returns[asset_class] = returns[symbol]
        
        # If we don't have mapped asset classes, use all available symbols
        if not asset_class_returns:
            asset_class_returns = {col: returns[col] for col in returns.columns if col != benchmark}
        
        # Calculate rolling correlations between benchmark and each asset class
        correlations = pd.DataFrame(index=returns.index)
        
        for asset_class, asset_returns in asset_class_returns.items():
            correlations[asset_class] = returns[benchmark].rolling(window=self.correlation_window).corr(asset_returns)
        
        # Classify correlations into regimes
        correlation_regimes = pd.DataFrame(index=correlations.index)
        
        for asset_class in correlations.columns:
            asset_regimes = pd.Series(index=correlations.index, dtype='object')
            
            for regime, (lower, upper) in correlation_thresholds.items():
                mask = (correlations[asset_class] >= lower) & (correlations[asset_class] < upper)
                asset_regimes[mask] = regime
            
            correlation_regimes[asset_class] = asset_regimes
        
        # Add correlation values
        for asset_class in correlations.columns:
            correlation_regimes[f'{asset_class}_value'] = correlations[asset_class]
        
        # Store correlation regimes
        self.correlation_regimes = correlation_regimes
        
        logger.info(f"Detected correlation regimes for {len(correlation_regimes.columns) // 2} asset classes")
        
        return correlation_regimes

    def analyze_sector_rotation(self) -> pd.DataFrame:
        """
        Analyze sector rotation patterns for equity markets.
        
        Returns:
            DataFrame with sector rotation analysis
        """
        # Get primary timeframe data
        primary_timeframe = self.timeframes[0]
        market_data = self.market_data_by_timeframe.get(primary_timeframe)
        
        if market_data is None or 'prices' not in market_data:
            logger.warning("No market data available for sector rotation analysis")
            return pd.DataFrame()
        
        price_data = market_data['prices']
        
        # Extract sector ETFs
        sector_symbols = [symbol for symbol in price_data.columns if symbol in self.sector_etf_mapping]
        
        if len(sector_symbols) < 3:
            logger.warning(f"Not enough sector ETFs found (found {len(sector_symbols)}, need at least 3)")
            return pd.DataFrame()
        
        # Calculate returns for sectors
        returns = price_data[sector_symbols].pct_change().dropna()
        
        # Calculate rolling relative performance (z-score of returns)
        window = min(20, len(returns) // 4)  # Use at least 5 periods
        
        # For each date, calculate the relative performance of each sector
        relative_perf = pd.DataFrame(index=returns.index)
        
        for date in returns.index[window:]:
            period_returns = returns.loc[date-pd.Timedelta(days=window):date]
            
            # Calculate cumulative returns for the period
            cumulative_returns = (1 + period_returns).prod() - 1
            
            # Calculate z-scores
            mean_return = cumulative_returns.mean()
            std_return = cumulative_returns.std()
            
            if std_return > 0:
                z_scores = (cumulative_returns - mean_return) / std_return
                relative_perf.loc[date, sector_symbols] = z_scores
        
        # Identify top and bottom sectors for each period
        sector_rankings = pd.DataFrame(index=relative_perf.index)
        
        for date in relative_perf.index:
            if any(pd.notna(relative_perf.loc[date])):
                # Sort sectors by performance
                ranked_sectors = relative_perf.loc[date].dropna().sort_values(ascending=False)
                
                # Store top and bottom sectors
                top_sectors = [self.sector_etf_mapping.get(symbol, symbol) for symbol in ranked_sectors.index[:3]]
                bottom_sectors = [self.sector_etf_mapping.get(symbol, symbol) for symbol in ranked_sectors.index[-3:]]
                
                sector_rankings.loc[date, 'top_sectors'] = ','.join(top_sectors)
                sector_rankings.loc[date, 'bottom_sectors'] = ','.join(bottom_sectors)
        
        # Define sector rotation phases and their representative sectors
        rotation_phases = {
            SectorRotationPhase.EARLY_EXPANSION: ["consumer_discretionary", "technology", "industrials"],
            SectorRotationPhase.LATE_EXPANSION: ["energy", "materials", "industrials"],
            SectorRotationPhase.EARLY_CONTRACTION: ["healthcare", "consumer_staples", "utilities"],
            SectorRotationPhase.LATE_CONTRACTION: ["financials", "technology", "consumer_discretionary"],
            SectorRotationPhase.RECOVERY: ["technology", "financials", "real_estate"]
        }
        
        # For each date, calculate the best matching phase
        def get_phase_score(top_sectors, phase_sectors):
            top_set = set(top_sectors.split(','))
            phase_set = set(phase_sectors)
            return len(top_set.intersection(phase_set))
        
        for date in sector_rankings.index:
            if pd.notna(sector_rankings.loc[date, 'top_sectors']):
                top_sectors = sector_rankings.loc[date, 'top_sectors']
                
                # Calculate score for each phase
                phase_scores = {
                    phase: get_phase_score(top_sectors, sectors)
                    for phase, sectors in rotation_phases.items()
                }
                
                # Find best matching phase
                best_phase = max(phase_scores.items(), key=lambda x: x[1])[0]
                
                # Only assign if there's a reasonable match
                if phase_scores[best_phase] >= 1:
                    sector_rankings.loc[date, 'rotation_phase'] = best_phase
                else:
                    sector_rankings.loc[date, 'rotation_phase'] = "indeterminate"
        
        # Add sector dispersion (measure of sector performance spread)
        sector_rankings['sector_dispersion'] = relative_perf.std(axis=1)
        
        # Store sector rotation results
        self.sector_rotation_phases = sector_rankings
        
        logger.info(f"Analyzed sector rotation for {len(sector_rankings)} periods")
        
        return sector_rankings

    def get_integrated_regime_analysis(self) -> Dict[str, Any]:
        """
        Get integrated analysis combining all regime detection methods.
        
        Returns:
            Dictionary with integrated regime analysis
        """
        analysis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes_analyzed': list(self.timeframes),
            'primary_timeframe': self.timeframes[0] if self.timeframes else None
        }
        
        # Add base regime detection
        if hasattr(self, 'regime_history') and self.regime_history is not None:
            latest_regime = self.get_current_regime()
            analysis['primary_regime'] = latest_regime
        
        # Add trend conflict analysis
        if hasattr(self, 'trend_conflict_regimes') and self.trend_conflict_regimes is not None:
            if not self.trend_conflict_regimes.empty:
                latest_conflict = self.trend_conflict_regimes.iloc[-1].to_dict()
                analysis['trend_conflicts'] = {
                    'status': latest_conflict.get('conflict_status', 'unknown'),
                    'timeframe_regimes': {
                        tf: {
                            'regime': latest_conflict.get(f'{tf}_regime'),
                            'label': latest_conflict.get(f'{tf}_regime_label')
                        }
                        for tf in self.timeframes if f'{tf}_regime' in latest_conflict
                    }
                }
            else:
                analysis['trend_conflicts'] = {
                    'status': 'insufficient_data',
                    'message': 'Not enough data to detect trend conflicts'
                }
        
        # Add volatility regime
        if hasattr(self, 'volatility_regimes') and self.volatility_regimes is not None:
            if not self.volatility_regimes.empty:
                analysis['volatility_regime'] = {
                    'current': self.volatility_regimes.iloc[-1] if not self.volatility_regimes.empty else None,
                    'history': {
                        regime: len(self.volatility_regimes[self.volatility_regimes == regime])
                        for regime in set(self.volatility_regimes)
                    }
                }
            else:
                analysis['volatility_regime'] = {
                    'current': None,
                    'history': {},
                    'message': 'Insufficient data for volatility regime classification'
                }
        
        # Add correlation regime summary
        if hasattr(self, 'correlation_regimes') and self.correlation_regimes is not None:
            if not self.correlation_regimes.empty:
                # Filter only the regime columns (not the _value columns)
                regime_cols = [col for col in self.correlation_regimes.columns if '_value' not in col]
                
                # Get the latest regime for each asset class
                latest_correlation = {
                    col: self.correlation_regimes[col].iloc[-1]
                    for col in regime_cols
                }
                
                # Get the latest correlation values
                latest_values = {
                    col.replace('_value', ''): self.correlation_regimes[col].iloc[-1]
                    for col in self.correlation_regimes.columns if '_value' in col
                }
                
                analysis['correlation_regimes'] = {
                    'current': latest_correlation,
                    'correlation_values': latest_values
                }
            else:
                analysis['correlation_regimes'] = {
                    'current': {},
                    'correlation_values': {},
                    'message': 'Insufficient data for correlation regime detection'
                }
        
        # Add sector rotation analysis
        if hasattr(self, 'sector_rotation_phases') and self.sector_rotation_phases is not None:
            if not self.sector_rotation_phases.empty:
                latest_rotation = self.sector_rotation_phases.iloc[-1].to_dict()
                
                analysis['sector_rotation'] = {
                    'current_phase': latest_rotation.get('rotation_phase'),
                    'top_sectors': latest_rotation.get('top_sectors', '').split(',') if latest_rotation.get('top_sectors') else [],
                    'bottom_sectors': latest_rotation.get('bottom_sectors', '').split(',') if latest_rotation.get('bottom_sectors') else [],
                    'sector_dispersion': latest_rotation.get('sector_dispersion')
                }
            else:
                analysis['sector_rotation'] = {
                    'current_phase': None,
                    'top_sectors': [],
                    'bottom_sectors': [],
                    'message': 'Insufficient data for sector rotation analysis'
                }
        
        # Add actionable insights based on combined analysis
        analysis['actionable_insights'] = self._generate_actionable_insights(analysis)
        
        return analysis
    
    def _generate_actionable_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate actionable insights based on the combined regime analysis.
        
        Args:
            analysis: The integrated regime analysis dictionary
            
        Returns:
            List of actionable insights with description and recommendation
        """
        insights = []
        
        # Insight based on trend conflicts
        if 'trend_conflicts' in analysis and 'status' in analysis['trend_conflicts']:
            conflict_status = analysis['trend_conflicts']['status']
            
            if conflict_status == 'strong_conflict':
                insights.append({
                    'type': 'trend_conflict',
                    'description': 'Strong conflict between timeframes detected',
                    'recommendation': 'Reduce position sizes and implement tighter risk controls'
                })
            elif conflict_status == 'aligned_bullish':
                insights.append({
                    'type': 'trend_alignment',
                    'description': 'Bullish alignment across timeframes',
                    'recommendation': 'Consider longer holding periods and trend-following strategies'
                })
            elif conflict_status == 'aligned_bearish':
                insights.append({
                    'type': 'trend_alignment',
                    'description': 'Bearish alignment across timeframes',
                    'recommendation': 'Consider defensive positioning and tactical short exposure'
                })
        
        # Insight based on volatility regime
        if 'volatility_regime' in analysis and 'current' in analysis['volatility_regime']:
            vol_regime = analysis['volatility_regime']['current']
            
            if vol_regime == VolatilityRegime.HIGH or vol_regime == VolatilityRegime.EXTREME:
                insights.append({
                    'type': 'volatility',
                    'description': f'High volatility environment ({vol_regime})',
                    'recommendation': 'Reduce position sizes and consider volatility-based strategies'
                })
            elif vol_regime == VolatilityRegime.LOW:
                insights.append({
                    'type': 'volatility',
                    'description': 'Low volatility environment',
                    'recommendation': 'Consider mean-reversion strategies and selling options premium'
                })
        
        # Insight based on correlation regimes
        if 'correlation_regimes' in analysis and 'current' in analysis['correlation_regimes']:
            # Check for dominant correlation regime
            correlation_summary = analysis['correlation_regimes']['current']
            
            # Count occurrences of each regime
            regime_counts = {}
            for asset, regime in correlation_summary.items():
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                regime_counts[regime] += 1
            
            # Find dominant regime (if any)
            if regime_counts:
                dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
                dominant_count = regime_counts[dominant_regime]
                
                # Only consider it dominant if it applies to at least half the assets
                if dominant_count >= len(correlation_summary) / 2:
                    if dominant_regime == CorrelationRegime.HIGH_POSITIVE:
                        insights.append({
                            'type': 'correlation',
                            'description': 'Assets highly correlated - limited diversification benefit',
                            'recommendation': 'Consider alternative asset classes or reduced exposure'
                        })
                    elif dominant_regime in [CorrelationRegime.UNCORRELATED, CorrelationRegime.MODERATE_NEGATIVE, CorrelationRegime.HIGH_NEGATIVE]:
                        insights.append({
                            'type': 'correlation',
                            'description': 'Low correlation environment - good diversification potential',
                            'recommendation': 'Maintain diversified exposure across assets'
                        })
        
        # Insight based on sector rotation
        if 'sector_rotation' in analysis and 'current_phase' in analysis['sector_rotation']:
            rotation_phase = analysis['sector_rotation']['current_phase']
            top_sectors = analysis['sector_rotation'].get('top_sectors', [])
            
            if rotation_phase and rotation_phase != "indeterminate":
                if rotation_phase == SectorRotationPhase.EARLY_EXPANSION:
                    insights.append({
                        'type': 'sector_rotation',
                        'description': 'Early economic expansion phase detected',
                        'recommendation': f'Favor cyclical sectors: {", ".join(top_sectors)}'
                    })
                elif rotation_phase == SectorRotationPhase.LATE_EXPANSION:
                    insights.append({
                        'type': 'sector_rotation',
                        'description': 'Late economic expansion phase detected',
                        'recommendation': f'Favor inflation-sensitive sectors: {", ".join(top_sectors)}'
                    })
                elif rotation_phase == SectorRotationPhase.EARLY_CONTRACTION:
                    insights.append({
                        'type': 'sector_rotation',
                        'description': 'Early economic contraction phase detected',
                        'recommendation': f'Favor defensive sectors: {", ".join(top_sectors)}'
                    })
                elif rotation_phase == SectorRotationPhase.LATE_CONTRACTION:
                    insights.append({
                        'type': 'sector_rotation',
                        'description': 'Late economic contraction phase detected',
                        'recommendation': f'Begin positioning for recovery: {", ".join(top_sectors)}'
                    })
                elif rotation_phase == SectorRotationPhase.RECOVERY:
                    insights.append({
                        'type': 'sector_rotation',
                        'description': 'Economic recovery phase detected',
                        'recommendation': f'Favor early-cycle sectors: {", ".join(top_sectors)}'
                    })
        
        return insights

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the full regime detection pipeline.
        
        Returns:
            Integrated regime analysis dictionary
        """
        # Ensure we have data loaded and features computed
        if not self.market_data_by_timeframe:
            logger.warning("No market data loaded. Cannot run analysis.")
            return {}
        
        # Compute features for all timeframes if not already done
        if not self.feature_data_by_timeframe:
            self.compute_features_multi_timeframe()
        
        # Detect regimes across timeframes
        self.detect_regimes_multi_timeframe()
        
        # Detect trend conflicts
        self.detect_trend_conflicts()
        
        # Classify volatility regimes
        self.classify_volatility_regimes()
        
        # Detect correlation regimes
        self.detect_correlation_regimes()
        
        # Analyze sector rotation
        self.analyze_sector_rotation()
        
        # Get integrated analysis
        analysis = self.get_integrated_regime_analysis()
        
        logger.info("Completed full market regime analysis")
        
        return analysis 