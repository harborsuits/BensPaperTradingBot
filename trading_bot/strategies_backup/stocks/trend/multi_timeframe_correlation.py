#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Correlation Strategy Module

This module implements a sophisticated trading strategy that analyzes correlations
between multiple assets across different timeframes to identify trading opportunities
based on correlation divergences, statistical relationships, and relative strength.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Setup logging
logger = logging.getLogger(__name__)

class MultiTimeframeCorrelationStrategy(StrategyOptimizable):
    """
    Multi-Timeframe Correlation Strategy.
    
    Analyzes cross-asset correlations across multiple timeframes to identify trading
    opportunities based on correlation divergences, relative strength, and statistical
    relationships between related assets.
    
    Key features:
    - Systematically identifies and tracks correlations between related assets
    - Detects statistically significant divergences in correlated asset pairs
    - Measures relative strength to determine outperforming/underperforming assets
    - Calculates beta relationships for risk adjustment across correlated pairs
    - Monitors correlation regime changes that might signal emerging opportunities
    - Uses both higher and lower timeframes for confirmation and precise entries
    - Implements advanced position sizing based on correlation exposure
    
    Trading methodology:
    - Scans the universe to identify strongly correlated asset pairs
    - Monitors these pairs for divergences from their statistical relationship
    - Generates signals when divergences reach statistical significance (z-score based)
    - Takes positions anticipating reversion to the established correlation
    - Uses technical indicators on lower timeframes for entry timing
    - Manages risk using volatility-adjusted position sizing and correlation-aware limits
    - Features time-based and performance-based exit rules
    
    Ideal market conditions:
    - Markets with clear fundamental relationships between assets (e.g., sector peers)
    - Environments with established statistical correlations between instruments
    - Periods where correlations remain relatively stable but allow temporary divergences
    - Markets with sufficient liquidity across all correlated assets
    
    Limitations:
    - Requires substantial historical data to establish reliable correlations
    - Correlation relationships can break down during market regime changes
    - Computationally intensive due to matrix calculations across multiple assets
    - Performance dependent on the stability of statistical relationships
    - May generate fewer signals than single-asset strategies
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Multi-Timeframe Correlation strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # Timeframes
            "analysis_timeframe": TimeFrame.DAY_1,    # Higher timeframe for correlation analysis
            "entry_timeframe": TimeFrame.HOUR_4,      # Lower timeframe for entry signals
            
            # Correlation parameters
            "correlation_window": 60,                # Window for correlation calculation (days)
            "correlation_threshold": 0.7,            # Minimum correlation threshold
            "correlation_change_threshold": 0.3,     # Threshold for correlation regime change
            "pairs_to_analyze": 3,                   # Number of top correlated pairs to analyze
            
            # Price return parameters
            "return_window": 20,                     # Window for calculating price returns
            "relative_strength_window": 20,          # Window for relative strength calculation
            
            # Divergence parameters
            "divergence_lookback": 10,               # Period to look for divergences
            "z_score_threshold": 1.5,                # Z-score threshold for divergence
            
            # Performance metrics
            "beta_window": 60,                       # Window for beta calculation
            "alpha_window": 60,                      # Window for alpha calculation
            "sharpe_window": 60,                     # Window for Sharpe ratio calculation
            
            # Entry/exit parameters
            "rsi_period": 14,                        # RSI period
            "relative_rsi_lookback": 5,              # Lookback for relative RSI comparison
            "volatility_window": 20,                 # Window for volatility calculation
            "atr_period": 14,                        # ATR period for stop placement
            "target_atr_multiple": 2.0,              # Target as multiple of ATR
            "stop_atr_multiple": 1.0,                # Stop loss as multiple of ATR
            "max_holding_days": 14,                  # Maximum holding period in days
            
            # Risk management
            "max_correlation_exposure": 3,           # Max positions in correlated assets
            "max_risk_per_trade": 0.005,             # Maximum risk per trade (0.5%)
            "position_sizing_method": "equal",       # Method for position sizing
            
            # Basket weighting
            "use_basket_trading": False,             # Trade baskets of assets
            "basket_weighting_method": "equal",      # Method for weighting basket components
            
            # Backtest and optimization
            "in_sample_days": 180,                   # In-sample period for optimization
            "out_sample_days": 60                    # Out-of-sample period for validation
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Set appropriate timeframes for this strategy
        if metadata is None or not hasattr(metadata, 'timeframes'):
            self.timeframes = [
                TimeFrame.HOUR_4,  # Entry timeframe
                TimeFrame.DAY_1    # Correlation analysis timeframe
            ]
        
        # Initialize watched symbols
        self.correlated_pairs = {}  # Will store correlated symbol pairs
        
        logger.info(f"Initialized Multi-Timeframe Correlation strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "correlation_window": [30, 60, 90],
            "correlation_threshold": [0.6, 0.7, 0.8],
            "correlation_change_threshold": [0.2, 0.3, 0.4],
            "pairs_to_analyze": [2, 3, 5],
            "return_window": [10, 20, 30],
            "z_score_threshold": [1.0, 1.5, 2.0],
            "rsi_period": [10, 14, 21],
            "volatility_window": [10, 20, 30],
            "target_atr_multiple": [1.5, 2.0, 2.5],
            "stop_atr_multiple": [0.5, 1.0, 1.5],
            "max_holding_days": [7, 14, 21],
            "max_risk_per_trade": [0.003, 0.005, 0.007]
        }
    
    def _calculate_correlation_matrix(self, data: Dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
        """
        Calculate correlation matrix between multiple assets.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price data
            window: Rolling window size for correlation calculation
            
        Returns:
            DataFrame with correlation matrix
        """
        # Extract closing prices for all symbols
        close_prices = {}
        for symbol, df in data.items():
            if 'close' in df.columns:
                close_prices[symbol] = df['close']
        
        # Create DataFrame with all close prices
        all_prices = pd.DataFrame(close_prices)
        
        # Calculate correlation matrix
        returns = all_prices.pct_change().dropna()
        correlation_matrix = returns.rolling(window=window).corr().iloc[-window:]
        
        # Get the last correlation snapshot
        symbols = list(close_prices.keys())
        final_corr = pd.DataFrame(index=symbols, columns=symbols)
        
        for i in symbols:
            for j in symbols:
                if i == j:
                    final_corr.loc[i, j] = 1.0
                else:
                    # Get the most recent correlation value
                    corr_value = correlation_matrix.loc[(slice(None), i), j].iloc[-1]
                    final_corr.loc[i, j] = corr_value
        
        return final_corr
    
    def _find_correlated_pairs(self, corr_matrix: pd.DataFrame, threshold: float, max_pairs: int) -> List[Tuple[str, str, float]]:
        """
        Find the top correlated pairs above threshold.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Minimum correlation threshold
            max_pairs: Maximum number of pairs to return
            
        Returns:
            List of tuples (symbol1, symbol2, correlation)
        """
        pairs = []
        
        # Extract upper triangle of correlation matrix (excluding diagonal)
        for i in range(len(corr_matrix.index)):
            for j in range(i + 1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.index[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold:
                    pairs.append((symbol1, symbol2, correlation))
        
        # Sort by absolute correlation (descending)
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return pairs[:max_pairs]
    
    def _calculate_relative_strength(self, data1: pd.Series, data2: pd.Series, window: int) -> pd.Series:
        """
        Calculate relative strength between two assets.
        
        Args:
            data1: Price series for first asset
            data2: Price series for second asset
            window: Window for relative strength calculation
            
        Returns:
            Series with relative strength values
        """
        # Calculate ratio of prices
        ratio = data1 / data2
        
        # Calculate z-score of ratio
        ratio_mean = ratio.rolling(window=window).mean()
        ratio_std = ratio.rolling(window=window).std()
        z_score = (ratio - ratio_mean) / ratio_std
        
        return z_score
    
    def _calculate_beta(self, returns1: pd.Series, returns2: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling beta between two assets.
        
        Args:
            returns1: Return series for first asset
            returns2: Return series for second asset (benchmark)
            window: Rolling window size
            
        Returns:
            Series with beta values
        """
        # Calculate covariance and variance
        covariance = returns1.rolling(window=window).cov(returns2)
        variance = returns2.rolling(window=window).var()
        
        # Calculate beta
        beta = covariance / variance
        
        return beta
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _check_divergence(self, prices1: pd.Series, prices2: pd.Series, correlation: float, 
                         lookback: int, z_score_threshold: float) -> Tuple[bool, float]:
        """
        Check for price divergence between correlated assets.
        
        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset
            correlation: Expected correlation between assets
            lookback: Lookback period
            z_score_threshold: Z-score threshold for divergence
            
        Returns:
            Tuple of (divergence_exists, z_score)
        """
        if len(prices1) < lookback or len(prices2) < lookback:
            return False, 0.0
        
        # Get returns
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()
        
        # Calculate cumulative returns over lookback period
        cum_return1 = (1 + returns1[-lookback:]).cumprod()[-1] - 1
        cum_return2 = (1 + returns2[-lookback:]).cumprod()[-1] - 1
        
        # If correlation is positive, divergence happens when returns have opposite signs
        # If correlation is negative, divergence happens when returns have same sign
        if correlation > 0:
            sign_match = np.sign(cum_return1) == np.sign(cum_return2)
            divergence = not sign_match
        else:
            sign_match = np.sign(cum_return1) != np.sign(cum_return2)
            divergence = not sign_match
        
        # Calculate z-score of return difference
        return_diff = cum_return1 - (correlation * cum_return2)
        
        # If we have enough historical data, calculate z-score
        if len(returns1) > lookback * 3:
            # Get historical return differences
            hist_returns1 = prices1.pct_change().dropna()
            hist_returns2 = prices2.pct_change().dropna()
            
            hist_cum_returns = []
            for i in range(len(hist_returns1) - lookback):
                r1 = (1 + hist_returns1.iloc[i:i+lookback]).cumprod()[-1] - 1
                r2 = (1 + hist_returns2.iloc[i:i+lookback]).cumprod()[-1] - 1
                hist_cum_returns.append(r1 - (correlation * r2))
            
            # Calculate z-score
            z_score = (return_diff - np.mean(hist_cum_returns)) / np.std(hist_cum_returns)
            
            # Divergence needs to be significant
            divergence = divergence and (abs(z_score) > z_score_threshold)
            
            return divergence, z_score
        
        return divergence, 0.0
    
    def _get_unique_symbols(self, correlated_pairs: List[Tuple[str, str, float]]) -> Set[str]:
        """
        Get unique symbols from correlated pairs.
        
        Args:
            correlated_pairs: List of correlated pairs (symbol1, symbol2, correlation)
            
        Returns:
            Set of unique symbols
        """
        symbols = set()
        for s1, s2, _ in correlated_pairs:
            symbols.add(s1)
            symbols.add(s2)
        return symbols
    
    def calculate_indicators(self, data: Dict[str, Dict[TimeFrame, pd.DataFrame]]) -> Dict[str, Dict[TimeFrame, Dict[str, pd.DataFrame]]]:
        """
        Calculate indicators for all timeframes for all symbols.
        
        Args:
            data: Dictionary mapping symbols to dictionaries of timeframes to DataFrames
            
        Returns:
            Dictionary of calculated indicators for each symbol and timeframe
        """
        indicators = {}
        
        # Get parameters
        analysis_timeframe = self.parameters.get("analysis_timeframe", TimeFrame.DAY_1)
        entry_timeframe = self.parameters.get("entry_timeframe", TimeFrame.HOUR_4)
        correlation_window = self.parameters.get("correlation_window", 60)
        correlation_threshold = self.parameters.get("correlation_threshold", 0.7)
        pairs_to_analyze = self.parameters.get("pairs_to_analyze", 3)
        return_window = self.parameters.get("return_window", 20)
        rsi_period = self.parameters.get("rsi_period", 14)
        atr_period = self.parameters.get("atr_period", 14)
        
        # Process analysis timeframe data - need this to calculate correlations
        analysis_data = {}
        for symbol, timeframes in data.items():
            if analysis_timeframe in timeframes:
                df = timeframes[analysis_timeframe]
                if 'close' in df.columns:
                    analysis_data[symbol] = df
        
        # Calculate correlation matrix if we have enough symbols
        if len(analysis_data) > 1:
            try:
                # Calculate correlation matrix
                correlation_matrix = self._calculate_correlation_matrix(analysis_data, correlation_window)
                
                # Find correlated pairs
                correlated_pairs = self._find_correlated_pairs(
                    correlation_matrix, correlation_threshold, pairs_to_analyze
                )
                
                # Store correlated pairs for use in signal generation
                self.correlated_pairs = {(s1, s2): corr for s1, s2, corr in correlated_pairs}
                
                # Get list of all symbols in the correlated pairs
                symbols_to_analyze = self._get_unique_symbols(correlated_pairs)
                
                # Calculate indicators for each symbol
                for symbol in symbols_to_analyze:
                    indicators[symbol] = {}
                    
                    # Process analysis timeframe indicators
                    if analysis_timeframe in data[symbol]:
                        df_analysis = data[symbol][analysis_timeframe]
                        
                        if not all(col in df_analysis.columns for col in ['open', 'high', 'low', 'close']):
                            logger.warning(f"Required columns not found for {symbol} on {analysis_timeframe}")
                            continue
                        
                        try:
                            # Store raw price data
                            indicators[symbol][analysis_timeframe] = {
                                "price": pd.DataFrame({"price": df_analysis['close']})
                            }
                            
                            # Calculate returns
                            returns = df_analysis['close'].pct_change(return_window)
                            indicators[symbol][analysis_timeframe]["returns"] = pd.DataFrame({"returns": returns})
                            
                            # For each correlated pair, calculate relative metrics
                            for (s1, s2), corr in self.correlated_pairs.items():
                                if symbol == s1 and s2 in data:
                                    other_symbol = s2
                                    other_df = data[other_symbol][analysis_timeframe]
                                    
                                    # Calculate relative strength
                                    rs = self._calculate_relative_strength(
                                        df_analysis['close'], 
                                        other_df['close'], 
                                        return_window
                                    )
                                    indicators[symbol][analysis_timeframe][f"rel_strength_{other_symbol}"] = pd.DataFrame({
                                        f"rel_strength_{other_symbol}": rs
                                    })
                                    
                                    # Calculate beta
                                    beta = self._calculate_beta(
                                        df_analysis['close'].pct_change().dropna(),
                                        other_df['close'].pct_change().dropna(),
                                        correlation_window
                                    )
                                    indicators[symbol][analysis_timeframe][f"beta_{other_symbol}"] = pd.DataFrame({
                                        f"beta_{other_symbol}": beta
                                    })
                                
                                elif symbol == s2 and s1 in data:
                                    other_symbol = s1
                                    other_df = data[other_symbol][analysis_timeframe]
                                    
                                    # Calculate relative strength
                                    rs = self._calculate_relative_strength(
                                        df_analysis['close'], 
                                        other_df['close'], 
                                        return_window
                                    )
                                    indicators[symbol][analysis_timeframe][f"rel_strength_{other_symbol}"] = pd.DataFrame({
                                        f"rel_strength_{other_symbol}": rs
                                    })
                                    
                                    # Calculate beta
                                    beta = self._calculate_beta(
                                        df_analysis['close'].pct_change().dropna(),
                                        other_df['close'].pct_change().dropna(),
                                        correlation_window
                                    )
                                    indicators[symbol][analysis_timeframe][f"beta_{other_symbol}"] = pd.DataFrame({
                                        f"beta_{other_symbol}": beta
                                    })
                        
                        except Exception as e:
                            logger.error(f"Error calculating analysis indicators for {symbol}: {e}")
                    
                    # Process entry timeframe indicators
                    if entry_timeframe in data[symbol]:
                        df_entry = data[symbol][entry_timeframe]
                        
                        if not all(col in df_entry.columns for col in ['open', 'high', 'low', 'close']):
                            logger.warning(f"Required columns not found for {symbol} on {entry_timeframe}")
                            continue
                        
                        try:
                            # Calculate RSI
                            rsi = self._calculate_rsi(df_entry['close'], rsi_period)
                            indicators[symbol][entry_timeframe] = {
                                "rsi": pd.DataFrame({"rsi": rsi})
                            }
                            
                            # Calculate ATR
                            atr = self._calculate_atr(df_entry['high'], df_entry['low'], df_entry['close'], atr_period)
                            indicators[symbol][entry_timeframe]["atr"] = pd.DataFrame({"atr": atr})
                        
                        except Exception as e:
                            logger.error(f"Error calculating entry indicators for {symbol}: {e}")
            
            except Exception as e:
                logger.error(f"Error calculating correlation indicators: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, Dict[TimeFrame, pd.DataFrame]], indicators: Optional[Dict[str, Dict[TimeFrame, Dict[str, pd.DataFrame]]]] = None) -> Dict[str, Signal]:
        """
        Generate signals based on correlation analysis and divergences.
        
        Args:
            data: Dictionary mapping symbols to dictionaries of timeframes to DataFrames
            indicators: Pre-calculated indicators (optional, will be computed if not provided)
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators if not provided
        if indicators is None:
            indicators = self.calculate_indicators(data)
        
        # Get parameters
        analysis_timeframe = self.parameters.get("analysis_timeframe", TimeFrame.DAY_1)
        entry_timeframe = self.parameters.get("entry_timeframe", TimeFrame.HOUR_4)
        divergence_lookback = self.parameters.get("divergence_lookback", 10)
        z_score_threshold = self.parameters.get("z_score_threshold", 1.5)
        rsi_lower_bound = 35
        rsi_upper_bound = 65
        target_atr_multiple = self.parameters.get("target_atr_multiple", 2.0)
        stop_atr_multiple = self.parameters.get("stop_atr_multiple", 1.0)
        
        # Generate signals
        signals = {}
        
        # Check if we have correlated pairs
        if not self.correlated_pairs:
            return signals
        
        # Look for divergences in correlated pairs
        for (symbol1, symbol2), correlation in self.correlated_pairs.items():
            # Skip if we don't have indicators for both symbols
            if (symbol1 not in indicators or symbol2 not in indicators or
                analysis_timeframe not in indicators[symbol1] or analysis_timeframe not in indicators[symbol2] or
                entry_timeframe not in indicators[symbol1] or entry_timeframe not in indicators[symbol2]):
                continue
            
            try:
                # Get price data
                price1 = indicators[symbol1][analysis_timeframe]["price"]["price"]
                price2 = indicators[symbol2][analysis_timeframe]["price"]["price"]
                
                # Check for divergence
                divergence_exists, z_score = self._check_divergence(
                    price1, price2, correlation, divergence_lookback, z_score_threshold
                )
                
                if divergence_exists:
                    # Determine which symbol to trade based on RSI
                    rsi1 = indicators[symbol1][entry_timeframe]["rsi"]["rsi"].iloc[-1]
                    rsi2 = indicators[symbol2][entry_timeframe]["rsi"]["rsi"].iloc[-1]
                    
                    # If correlation is positive:
                    # - If one asset is underperforming (negative z-score), and its RSI is low, BUY it
                    # - If one asset is overperforming (positive z-score), and its RSI is high, SELL it
                    
                    # If correlation is negative:
                    # - If one asset is moving opposite to expected (negative z-score), and its RSI is low, BUY it
                    # - If one asset is moving opposite to expected (positive z-score), and its RSI is high, SELL it
                    
                    symbol_to_trade = None
                    signal_type = None
                    
                    if correlation > 0:  # Positively correlated
                        if z_score < 0 and rsi1 < rsi_lower_bound:  # Symbol1 underperforming
                            symbol_to_trade = symbol1
                            signal_type = SignalType.BUY
                        elif z_score > 0 and rsi1 > rsi_upper_bound:  # Symbol1 overperforming
                            symbol_to_trade = symbol1
                            signal_type = SignalType.SELL
                        elif z_score > 0 and rsi2 < rsi_lower_bound:  # Symbol2 underperforming
                            symbol_to_trade = symbol2
                            signal_type = SignalType.BUY
                        elif z_score < 0 and rsi2 > rsi_upper_bound:  # Symbol2 overperforming
                            symbol_to_trade = symbol2
                            signal_type = SignalType.SELL
                    else:  # Negatively correlated
                        if z_score < 0 and rsi1 > rsi_upper_bound:  # Symbol1 moving opposite to expected
                            symbol_to_trade = symbol1
                            signal_type = SignalType.SELL
                        elif z_score > 0 and rsi1 < rsi_lower_bound:  # Symbol1 moving opposite to expected
                            symbol_to_trade = symbol1
                            signal_type = SignalType.BUY
                        elif z_score > 0 and rsi2 > rsi_upper_bound:  # Symbol2 moving opposite to expected
                            symbol_to_trade = symbol2
                            signal_type = SignalType.SELL
                        elif z_score < 0 and rsi2 < rsi_lower_bound:  # Symbol2 moving opposite to expected
                            symbol_to_trade = symbol2
                            signal_type = SignalType.BUY
                    
                    # Generate signal if we have a valid trade
                    if symbol_to_trade and signal_type:
                        # Get current data
                        latest_entry = data[symbol_to_trade][entry_timeframe].iloc[-1]
                        latest_timestamp = latest_entry.name if isinstance(latest_entry.name, datetime) else datetime.now()
                        
                        # Calculate ATR for stop loss and target
                        atr = indicators[symbol_to_trade][entry_timeframe]["atr"]["atr"].iloc[-1]
                        
                        # Calculate stop loss and take profit
                        if signal_type == SignalType.BUY:
                            stop_loss = latest_entry['close'] - (atr * stop_atr_multiple)
                            take_profit = latest_entry['close'] + (atr * target_atr_multiple)
                        else:  # SELL
                            stop_loss = latest_entry['close'] + (atr * stop_atr_multiple)
                            take_profit = latest_entry['close'] - (atr * target_atr_multiple)
                        
                        # Calculate confidence based on z-score
                        confidence = min(0.9, abs(z_score) / 3.0)
                        
                        # Create signal
                        signals[symbol_to_trade] = Signal(
                            symbol=symbol_to_trade,
                            signal_type=signal_type,
                            price=latest_entry['close'],
                            timestamp=latest_timestamp,
                            confidence=confidence,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            timeframe=entry_timeframe,
                            metadata={
                                "analysis_timeframe": analysis_timeframe.value,
                                "entry_timeframe": entry_timeframe.value,
                                "correlated_pair": f"{symbol1}_{symbol2}",
                                "correlation": correlation,
                                "z_score": z_score,
                                "rsi": indicators[symbol_to_trade][entry_timeframe]["rsi"]["rsi"].iloc[-1],
                                "atr": atr,
                                "max_holding_days": self.parameters.get("max_holding_days", 14),
                                "strategy_type": "multi_timeframe_correlation"
                            }
                        )
                        
                        logger.info(f"Generated correlation {signal_type} signal for {symbol_to_trade}")
                        logger.info(f"Correlated with {symbol1 if symbol_to_trade == symbol2 else symbol2}, z-score: {z_score:.2f}")
                        logger.info(f"Entry: {latest_entry['close']:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol1}/{symbol2}: {e}")
        
        return signals
    
    def should_exit(self, symbol: str, position_data: Dict[str, Any], current_data: Dict[TimeFrame, pd.DataFrame], days_held: int) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.
        
        Args:
            symbol: Symbol of the position
            position_data: Data about the position
            current_data: Current price data for the symbol
            days_held: Number of days the position has been held
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Get parameters
        max_holding_days = self.parameters.get("max_holding_days", 14)
        analysis_timeframe = self.parameters.get("analysis_timeframe", TimeFrame.DAY_1)
        
        # Check if position has been held longer than max holding days
        if days_held >= max_holding_days:
            return True, "Time-based exit triggered"
        
        # Check if correlation divergence has reverted
        try:
            if "correlated_pair" in position_data["metadata"]:
                correlated_pair = position_data["metadata"]["correlated_pair"]
                symbol1, symbol2 = correlated_pair.split("_")
                
                # If the correlated symbol is not in current data, we can't check divergence
                if (symbol1 not in current_data or symbol2 not in current_data or
                    analysis_timeframe not in current_data[symbol1] or 
                    analysis_timeframe not in current_data[symbol2]):
                    return False, ""
                
                original_z_score = position_data["metadata"]["z_score"]
                
                # Get current price data
                price1 = current_data[symbol1][analysis_timeframe]['close']
                price2 = current_data[symbol2][analysis_timeframe]['close']
                
                # Get correlation
                correlation = position_data["metadata"]["correlation"]
                
                # Check if the divergence still exists
                divergence_lookback = self.parameters.get("divergence_lookback", 10)
                z_score_threshold = self.parameters.get("z_score_threshold", 1.5)
                
                divergence_exists, current_z_score = self._check_divergence(
                    price1, price2, correlation, divergence_lookback, z_score_threshold
                )
                
                # Exit if divergence has reverted (z-score crossed zero or reduced significantly)
                if (original_z_score * current_z_score < 0) or (abs(current_z_score) < 0.5):
                    return True, "Correlation divergence reversion exit triggered"
        
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
        
        return False, "" 