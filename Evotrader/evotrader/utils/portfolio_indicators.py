"""
Portfolio-level indicators for EvoTrader.

This module provides indicators that analyze relationships between multiple assets
in a portfolio, including correlation, beta, relative strength, and other
cross-market metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
import pandas as pd
from collections import defaultdict

from .indicator_system import Indicator

logger = logging.getLogger(__name__)


class PortfolioIndicator:
    """Base class for portfolio-level indicators that analyze multiple assets."""
    
    def __init__(self, symbols: List[str], params: Dict[str, Any] = None):
        """
        Initialize portfolio indicator.
        
        Args:
            symbols: List of trading symbols to analyze
            params: Configuration parameters
        """
        self.symbols = symbols
        self.params = params or {}
        self.values: Dict[str, Any] = {}
        self.is_ready = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Data storage
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.timestamp_history: List[int] = []
    
    def update(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Update the portfolio indicator with the latest market data.
        
        Args:
            data: Dictionary mapping symbols to their market data
        """
        # Check if we have data for all required symbols
        if not all(symbol in data for symbol in self.symbols):
            missing = [s for s in self.symbols if s not in data]
            self.logger.warning(f"Missing data for symbols: {missing}")
            return
        
        # Check for timestamp consistency
        timestamps = set(data[s].get('timestamp', 0) for s in self.symbols)
        if len(timestamps) > 1:
            self.logger.warning(f"Inconsistent timestamps across symbols: {timestamps}")
        
        # Use the most common timestamp
        timestamp = max(timestamps, key=list(timestamps).count) if timestamps else 0
        self.timestamp_history.append(timestamp)
        
        # Update price histories
        for symbol in self.symbols:
            symbol_data = data[symbol]
            price = symbol_data.get('price')
            if price is not None:
                self.price_history[symbol].append(price)
            else:
                self.logger.warning(f"No price data for {symbol}")
                # Use previous price or None
                prev_price = self.price_history[symbol][-1] if self.price_history[symbol] else None
                self.price_history[symbol].append(prev_price)
        
        # Calculate indicator values
        self._calculate()
    
    def _calculate(self) -> None:
        """
        Calculate indicator values.
        Override in subclasses.
        """
        pass
    
    def get_value(self, key: str = None) -> Any:
        """
        Get indicator value for a specific key.
        
        Args:
            key: Value key, or None for all values
            
        Returns:
            Indicator value or all values if key is None
        """
        if key is None:
            return self.values
        return self.values.get(key)


class CorrelationMatrix(PortfolioIndicator):
    """
    Correlation matrix indicator.
    
    Calculates the price correlation between all assets in the portfolio.
    """
    
    def __init__(self, symbols: List[str], period: int = 20):
        """
        Initialize correlation matrix indicator.
        
        Args:
            symbols: List of trading symbols to analyze
            period: Correlation calculation period
        """
        super().__init__(symbols, {'period': period})
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.min_data_points = period
    
    def _calculate(self) -> None:
        """Calculate correlation matrix."""
        period = self.params['period']
        
        # Check if we have enough data
        min_history = min(len(history) for history in self.price_history.values())
        if min_history < period:
            return
        
        # Create a DataFrame with the latest period's price data
        price_data = {}
        for symbol, history in self.price_history.items():
            price_data[symbol] = history[-period:]
        
        df = pd.DataFrame(price_data)
        
        # Calculate correlation matrix
        try:
            corr_matrix = df.corr(method='pearson')
            self.correlation_matrix = corr_matrix
            self.values['matrix'] = corr_matrix.to_dict()
            
            # Average correlation for each symbol
            avg_corr = {}
            for symbol in self.symbols:
                # Average correlation with all other symbols
                correlations = [
                    corr for other_symbol, corr in corr_matrix[symbol].items()
                    if other_symbol != symbol and not pd.isna(corr)
                ]
                avg_corr[symbol] = sum(correlations) / len(correlations) if correlations else 0
            
            self.values['average_correlation'] = avg_corr
            
            # Highly correlated pairs (above 0.7)
            high_corr_pairs = []
            for i, symbol1 in enumerate(self.symbols):
                for symbol2 in self.symbols[i+1:]:
                    corr = corr_matrix.loc[symbol1, symbol2]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append((symbol1, symbol2, corr))
            
            self.values['high_correlation_pairs'] = high_corr_pairs
            
            # The indicator is ready once we've calculated the matrix
            self.is_ready = True
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Get correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Correlation coefficient or None if not available
        """
        if not self.correlation_matrix is not None:
            return None
            
        if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
            return self.correlation_matrix.loc[symbol1, symbol2]
        
        return None
    
    def get_uncorrelated_symbols(
        self, 
        target_symbol: str, 
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find symbols with low correlation to the target symbol.
        
        Args:
            target_symbol: Symbol to find uncorrelated assets for
            threshold: Maximum absolute correlation to consider as uncorrelated
            
        Returns:
            List of (symbol, correlation) tuples where |correlation| < threshold
        """
        if self.correlation_matrix is None or target_symbol not in self.correlation_matrix.index:
            return []
            
        result = []
        for symbol, corr in self.correlation_matrix[target_symbol].items():
            if symbol != target_symbol and abs(corr) < threshold:
                result.append((symbol, corr))
                
        return result


class RelativeStrengthIndex(PortfolioIndicator):
    """
    Relative Strength Index across assets.
    
    Measures how each asset in the portfolio is performing relative to others.
    """
    
    def __init__(self, symbols: List[str], benchmark_symbol: str = None, period: int = 14):
        """
        Initialize relative strength indicator.
        
        Args:
            symbols: List of trading symbols to analyze
            benchmark_symbol: Symbol to use as benchmark (or None to use average)
            period: Calculation period
        """
        super().__init__(symbols, {
            'benchmark_symbol': benchmark_symbol,
            'period': period
        })
        self.relative_strength: Dict[str, float] = {}
        self.percent_change: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.rs_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.min_data_points = period + 1  # Need at least period+1 for price changes
    
    def _calculate(self) -> None:
        """Calculate relative strength for each symbol."""
        period = self.params['period']
        benchmark = self.params['benchmark_symbol']
        
        # Check if we have enough data
        min_history = min(len(history) for history in self.price_history.values())
        if min_history < 2:  # Need at least 2 points for percent change
            return
        
        # Calculate percent change for each symbol
        for symbol, history in self.price_history.items():
            if len(history) >= 2 and history[-1] is not None and history[-2] is not None:
                pct_change = ((history[-1] - history[-2]) / history[-2]) * 100
                self.percent_change[symbol].append(pct_change)
            else:
                # If we don't have valid data, use 0 change
                self.percent_change[symbol].append(0)
        
        # Calculate benchmark performance
        if benchmark and benchmark in self.percent_change:
            benchmark_changes = self.percent_change[benchmark]
            if len(benchmark_changes) > 0:
                benchmark_change = benchmark_changes[-1]
            else:
                benchmark_change = 0
        else:
            # Use average of all symbols as benchmark
            all_changes = [
                changes[-1] for symbol, changes in self.percent_change.items()
                if len(changes) > 0
            ]
            benchmark_change = sum(all_changes) / len(all_changes) if all_changes else 0
        
        # Calculate relative strength
        for symbol in self.symbols:
            changes = self.percent_change[symbol]
            if len(changes) > 0:
                # Relative strength = Asset Change / Benchmark Change
                # Handle benchmark change of 0 or very small values
                if abs(benchmark_change) < 0.01:
                    rs = 1.0 if changes[-1] >= 0 else -1.0
                else:
                    rs = changes[-1] / benchmark_change
                
                self.rs_history[symbol].append(rs)
                
                # Calculate the average RS over the period
                if len(self.rs_history[symbol]) >= period:
                    avg_rs = sum(self.rs_history[symbol][-period:]) / period
                    self.relative_strength[symbol] = avg_rs
        
        # Update values
        self.values['relative_strength'] = self.relative_strength.copy()
        
        # Calculate rankings
        if self.relative_strength:
            rankings = sorted(
                self.relative_strength.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            self.values['rankings'] = rankings
            
            # Top and bottom performers
            self.values['top_performers'] = rankings[:min(3, len(rankings))]
            self.values['bottom_performers'] = rankings[-min(3, len(rankings)):]
            
            self.is_ready = True
    
    def get_strongest_symbols(self, count: int = 3) -> List[Tuple[str, float]]:
        """
        Get the strongest performing symbols.
        
        Args:
            count: Number of symbols to return
            
        Returns:
            List of (symbol, relative strength) tuples
        """
        if not self.is_ready:
            return []
            
        rankings = sorted(
            self.relative_strength.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return rankings[:min(count, len(rankings))]
    
    def get_weakest_symbols(self, count: int = 3) -> List[Tuple[str, float]]:
        """
        Get the weakest performing symbols.
        
        Args:
            count: Number of symbols to return
            
        Returns:
            List of (symbol, relative strength) tuples
        """
        if not self.is_ready:
            return []
            
        rankings = sorted(
            self.relative_strength.items(), 
            key=lambda x: x[1]
        )
        return rankings[:min(count, len(rankings))]


class MarketBreadth(PortfolioIndicator):
    """
    Market Breadth indicator.
    
    Measures the number of advancing vs declining assets in the portfolio.
    """
    
    def __init__(self, symbols: List[str], period: int = 10):
        """
        Initialize market breadth indicator.
        
        Args:
            symbols: List of trading symbols to analyze
            period: Period for moving averages
        """
        super().__init__(symbols, {'period': period})
        self.advances: List[int] = []
        self.declines: List[int] = []
        self.unchanged: List[int] = []
        self.advance_decline_ratio: List[float] = []
        self.advance_decline_line: List[float] = []
        self.mcclellan_oscillator: List[float] = []
        self.min_data_points = 2  # Need at least 2 points for price changes
    
    def _calculate(self) -> None:
        """Calculate market breadth metrics."""
        # Check if we have enough data
        min_history = min(len(history) for history in self.price_history.values())
        if min_history < 2:  # Need at least 2 points for advances/declines
            return
        
        # Count advances, declines, and unchanged
        advances = 0
        declines = 0
        unchanged = 0
        
        for symbol, history in self.price_history.items():
            if len(history) >= 2 and history[-1] is not None and history[-2] is not None:
                if history[-1] > history[-2]:
                    advances += 1
                elif history[-1] < history[-2]:
                    declines += 1
                else:
                    unchanged += 1
        
        self.advances.append(advances)
        self.declines.append(declines)
        self.unchanged.append(unchanged)
        
        # Calculate advance-decline ratio
        if declines > 0:
            ratio = advances / declines
        else:
            ratio = advances if advances > 0 else 1.0
            
        self.advance_decline_ratio.append(ratio)
        
        # Calculate advance-decline line
        if not self.advance_decline_line:
            self.advance_decline_line.append(advances - declines)
        else:
            self.advance_decline_line.append(
                self.advance_decline_line[-1] + (advances - declines)
            )
        
        # Calculate McClellan Oscillator (EMA(19) - EMA(39) of advances-declines)
        period = self.params['period']
        ad_diff = [a - d for a, d in zip(self.advances, self.declines)]
        
        if len(ad_diff) >= period * 2:
            try:
                # Simple implementation of EMA
                ema_fast = self._calculate_ema(ad_diff, period)
                ema_slow = self._calculate_ema(ad_diff, period * 2)
                oscillator = ema_fast[-1] - ema_slow[-1]
                self.mcclellan_oscillator.append(oscillator)
            except Exception as e:
                self.logger.error(f"Error calculating McClellan Oscillator: {str(e)}")
                if self.mcclellan_oscillator:
                    self.mcclellan_oscillator.append(self.mcclellan_oscillator[-1])
                else:
                    self.mcclellan_oscillator.append(0)
        elif self.mcclellan_oscillator:
            self.mcclellan_oscillator.append(self.mcclellan_oscillator[-1])
        else:
            self.mcclellan_oscillator.append(0)
        
        # Update values
        self.values['advances'] = advances
        self.values['declines'] = declines
        self.values['unchanged'] = unchanged
        self.values['advance_decline_ratio'] = ratio
        self.values['advance_decline_line'] = self.advance_decline_line[-1]
        
        if self.mcclellan_oscillator:
            self.values['mcclellan_oscillator'] = self.mcclellan_oscillator[-1]
        
        # Market is bullish if A/D line is rising and McClellan Oscillator is positive
        if len(self.advance_decline_line) >= 2:
            ad_rising = self.advance_decline_line[-1] > self.advance_decline_line[-2]
            mcclellan_positive = self.mcclellan_oscillator[-1] > 0
            
            self.values['market_breadth_bullish'] = ad_rising and mcclellan_positive
            self.values['market_breadth_bearish'] = not ad_rising and not mcclellan_positive
            
            self.is_ready = True
    
    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """
        Calculate EMA for a list of values.
        
        Args:
            values: List of input values
            period: EMA period
            
        Returns:
            List of EMA values
        """
        if len(values) < period:
            return []
            
        ema = [sum(values[:period]) / period]  # Start with SMA
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(values)):
            ema.append((values[i] - ema[-1]) * multiplier + ema[-1])
            
        return ema
    
    def is_market_bullish(self) -> bool:
        """Check if market breadth indicators are bullish."""
        return self.is_ready and self.values.get('market_breadth_bullish', False)
    
    def is_market_bearish(self) -> bool:
        """Check if market breadth indicators are bearish."""
        return self.is_ready and self.values.get('market_breadth_bearish', False)


class PortfolioVolatility(PortfolioIndicator):
    """
    Portfolio Volatility indicator.
    
    Measures the overall volatility of the portfolio and correlations
    during volatile periods.
    """
    
    def __init__(self, symbols: List[str], period: int = 20):
        """
        Initialize portfolio volatility indicator.
        
        Args:
            symbols: List of trading symbols to analyze
            period: Volatility calculation period
        """
        super().__init__(symbols, {'period': period})
        self.returns: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.volatility: Dict[str, float] = {}
        self.portfolio_volatility: float = 0.0
        self.min_data_points = period + 1  # Need period+1 prices for period returns
    
    def _calculate(self) -> None:
        """Calculate portfolio volatility metrics."""
        period = self.params['period']
        
        # Check if we have enough data
        min_history = min(len(history) for history in self.price_history.values())
        if min_history < 2:  # Need at least 2 points for returns
            return
        
        # Calculate returns for each symbol
        for symbol, history in self.price_history.items():
            if len(history) >= 2 and history[-1] is not None and history[-2] is not None:
                # Log return: ln(P1/P0)
                try:
                    if history[-2] > 0 and history[-1] > 0:
                        ret = np.log(history[-1] / history[-2])
                        self.returns[symbol].append(ret)
                    else:
                        self.returns[symbol].append(0)
                except Exception:
                    self.returns[symbol].append(0)
        
        # Calculate volatility for each symbol
        for symbol, returns in self.returns.items():
            if len(returns) >= period:
                recent_returns = returns[-period:]
                self.volatility[symbol] = np.std(recent_returns) * np.sqrt(252)  # Annualized
        
        # Calculate portfolio returns based on equal weighting
        if all(len(returns) >= period for returns in self.returns.values()):
            portfolio_returns = []
            for i in range(1, period + 1):
                # Equal-weighted portfolio return for each period
                day_returns = [returns[-i] for returns in self.returns.values()]
                portfolio_returns.append(sum(day_returns) / len(day_returns))
                
            # Calculate portfolio volatility
            self.portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        
        # Update values
        self.values['individual_volatility'] = self.volatility.copy()
        self.values['portfolio_volatility'] = self.portfolio_volatility
        
        # High volatility symbols (top 25%)
        if self.volatility:
            vol_ranking = sorted(
                self.volatility.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            high_vol_cutoff = len(vol_ranking) // 4
            self.values['high_volatility_symbols'] = vol_ranking[:max(1, high_vol_cutoff)]
            
            # Diversification ratio: Average individual vol / portfolio vol
            if self.portfolio_volatility > 0:
                avg_vol = sum(self.volatility.values()) / len(self.volatility)
                self.values['diversification_ratio'] = avg_vol / self.portfolio_volatility
            
            self.is_ready = True
    
    def get_high_volatility_symbols(self, threshold: float = None) -> List[Tuple[str, float]]:
        """
        Get symbols with high volatility.
        
        Args:
            threshold: Volatility threshold (or None to use top 25%)
            
        Returns:
            List of (symbol, volatility) tuples
        """
        if not self.is_ready:
            return []
            
        if threshold is not None:
            return [(s, v) for s, v in self.volatility.items() if v > threshold]
        else:
            vol_ranking = sorted(
                self.volatility.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            high_vol_cutoff = max(1, len(vol_ranking) // 4)
            return vol_ranking[:high_vol_cutoff]
    
    def get_low_volatility_symbols(self, threshold: float = None) -> List[Tuple[str, float]]:
        """
        Get symbols with low volatility.
        
        Args:
            threshold: Volatility threshold (or None to use bottom 25%)
            
        Returns:
            List of (symbol, volatility) tuples
        """
        if not self.is_ready:
            return []
            
        if threshold is not None:
            return [(s, v) for s, v in self.volatility.items() if v < threshold]
        else:
            vol_ranking = sorted(
                self.volatility.items(), 
                key=lambda x: x[1]
            )
            low_vol_cutoff = max(1, len(vol_ranking) // 4)
            return vol_ranking[:low_vol_cutoff]
    
    def is_well_diversified(self, threshold: float = 1.5) -> bool:
        """
        Check if the portfolio is well diversified.
        
        A well-diversified portfolio has a diversification ratio > threshold.
        
        Args:
            threshold: Diversification threshold
            
        Returns:
            True if the portfolio is well diversified
        """
        return self.is_ready and self.values.get('diversification_ratio', 0) > threshold
