"""
Portfolio-Level Strategy Optimizer

Provides optimization capabilities across the entire portfolio rather
than just individual strategies, enabling coordinated strategy allocation
and risk management across all assets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import json
import os
from scipy.optimize import minimize

# Import strategy components
from trading_bot.strategies.hybrid_strategy_optimizer import HybridStrategyOptimizer
from trading_bot.ml_pipeline.ml_regime_detector import MLRegimeDetector
from trading_bot.ml_pipeline.optimizer.metrics import StrategyMetrics

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Portfolio-level optimization system
    
    Optimizes strategy allocation, risk budgeting, and position sizing
    across the entire portfolio rather than individual strategies.
    """
    
    def __init__(self, config=None):
        """
        Initialize the portfolio optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Portfolio parameters
        self.risk_budget = self.config.get('risk_budget', 0.02)  # 2% daily VaR
        self.max_drawdown_target = self.config.get('max_drawdown_target', 0.15)  # 15% max DD
        self.strategy_correlation_threshold = self.config.get('strategy_correlation_threshold', 0.6)
        self.results_dir = self.config.get('results_dir', 'optimization_results')
        
        # Sub-optimizers
        self.hybrid_optimizer = HybridStrategyOptimizer(config=self.config)
        self.regime_detector = MLRegimeDetector(config=self.config.get('regime_detector', {}))
        self.metrics = StrategyMetrics()
        
        # Portfolio allocation
        self.portfolio_weights = {}
        self.strategy_allocations = {}
        self.asset_allocations = {}
        self.risk_allocations = {}
        self.current_regime = None
        
        # Load last optimization if available
        self._load_last_optimization()
        
        logger.info("Portfolio Optimizer initialized")
    
    def _load_last_optimization(self):
        """Load the last saved optimization results if available"""
        try:
            result_files = []
            if os.path.exists(self.results_dir):
                result_files = [f for f in os.listdir(self.results_dir) 
                               if f.startswith('portfolio_allocation_') and f.endswith('.json')]
            
            if result_files:
                # Sort by timestamp (newest first)
                result_files.sort(reverse=True)
                latest_file = os.path.join(self.results_dir, result_files[0])
                
                with open(latest_file, 'r') as f:
                    allocation = json.load(f)
                
                self.portfolio_weights = allocation.get('portfolio_weights', {})
                self.strategy_allocations = allocation.get('strategy_allocations', {})
                self.asset_allocations = allocation.get('asset_allocations', {})
                self.risk_allocations = allocation.get('risk_allocations', {})
                self.current_regime = allocation.get('market_regime', None)
                
                logger.info(f"Loaded portfolio allocation from {latest_file}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error loading portfolio allocation: {e}")
            return False
    
    def _save_optimization_results(self, results: Dict[str, Any], suffix: str = ''):
        """
        Save optimization results to disk
        
        Args:
            results: Optimization results dictionary
            suffix: Optional suffix for the filename
        """
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_allocation_{timestamp}{suffix}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            # Convert numpy types to native Python types
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            # Prepare serializable results
            serializable_results = convert_numpy(results)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Saved portfolio allocation to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving portfolio allocation: {e}")
            return None
    
    def optimize_portfolio(self, symbols: List[str], 
                          strategies: List[str], 
                          historical_data: Dict[str, pd.DataFrame],
                          optimization_method: str = 'risk_parity',
                          days: int = 90,
                          regime_specific: bool = True) -> Dict[str, Any]:
        """
        Optimize portfolio-level strategy allocation
        
        Args:
            symbols: List of symbols to include
            strategies: List of strategy names to optimize
            historical_data: Dictionary of symbol -> DataFrame with price data
            optimization_method: Method to use ('risk_parity', 'sharpe_max', 'minvar')
            days: Number of days of historical data to use
            regime_specific: Whether to create regime-specific allocations
            
        Returns:
            Dict with optimization results
        """
        logger.info(f"Starting portfolio optimization with {len(symbols)} symbols and {len(strategies)} strategies")
        
        # Detect current market regime if using regime-specific optimization
        current_regime = None
        if regime_specific:
            # Use SPY or first symbol for regime detection
            regime_symbol = 'SPY' if 'SPY' in historical_data else list(historical_data.keys())[0]
            regime_info = self.regime_detector.detect_regime(historical_data[regime_symbol])
            current_regime = regime_info['regime']
            logger.info(f"Current market regime: {current_regime} (confidence: {regime_info['confidence']:.4f})")
        
        # First optimize individual strategies and collect performance data
        strategy_results = {}
        
        for strategy in strategies:
            logger.info(f"Optimizing {strategy} strategy")
            
            # Use strategy-specific optimizer (hybrid or standard)
            if strategy == 'hybrid':
                strategy_result = self.hybrid_optimizer.optimize_strategy_weights(
                    symbols=symbols,
                    timeframes=['1d'],  # Daily timeframe for portfolio optimization
                    days=days,
                    optimization_method='genetic',
                    include_regime_detection=regime_specific
                )
            else:
                # For individual strategies, optimize parameters
                strategy_result = self._optimize_individual_strategy(
                    strategy, symbols, historical_data, days, regime_specific
                )
            
            strategy_results[strategy] = strategy_result
        
        # Collect performance metrics for all strategies
        strategy_metrics = {}
        
        for strategy, result in strategy_results.items():
            # Extract performance metrics
            if 'symbol_metrics' in result:
                # Calculate average across symbols
                metrics = {}
                for metric in ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate', 'expectancy']:
                    values = [symbol_data.get(metric, 0) for symbol_data in result['symbol_metrics'].values()]
                    metrics[metric] = np.mean(values) if values else 0
                
                strategy_metrics[strategy] = metrics
            elif 'best_metrics' in result:
                strategy_metrics[strategy] = result['best_metrics']
            else:
                # Default metrics if none available
                strategy_metrics[strategy] = {
                    'sharpe_ratio': 0.5,
                    'sortino_ratio': 0.7,
                    'max_drawdown': -0.2,
                    'win_rate': 0.5,
                    'expectancy': 0.01
                }
        
        # Calculate performance correlations between strategies
        strategy_returns = self._calculate_strategy_returns(symbols, strategies, historical_data)
        correlation_matrix = self._calculate_correlation_matrix(strategy_returns)
        
        # Execute portfolio optimization method
        if optimization_method == 'risk_parity':
            portfolio_weights = self._optimize_risk_parity(strategy_metrics, correlation_matrix)
        elif optimization_method == 'sharpe_max':
            portfolio_weights = self._optimize_sharpe(strategy_metrics, strategy_returns)
        elif optimization_method == 'minvar':
            portfolio_weights = self._optimize_minimum_variance(correlation_matrix, strategy_returns)
        else:
            # Default to equal weight
            portfolio_weights = {strategy: 1.0 / len(strategies) for strategy in strategies}
        
        # Normalize weights to sum to 1
        total_weight = sum(portfolio_weights.values())
        if total_weight > 0:
            portfolio_weights = {k: v / total_weight for k, v in portfolio_weights.items()}
        
        # Calculate risk allocation
        risk_allocations = self._calculate_risk_allocation(portfolio_weights, strategy_metrics)
        
        # Optimize asset allocation within each strategy
        asset_allocations = {}
        for strategy in strategies:
            if strategy in strategy_results:
                result = strategy_results[strategy]
                
                if 'symbol_weights' in result:
                    asset_allocations[strategy] = result['symbol_weights']
                elif 'best_params' in result and 'symbol_weights' in result['best_params']:
                    asset_allocations[strategy] = result['best_params']['symbol_weights']
                else:
                    # Equal allocation across symbols
                    asset_allocations[strategy] = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        # Store results
        self.portfolio_weights = portfolio_weights
        self.strategy_allocations = strategy_results
        self.asset_allocations = asset_allocations
        self.risk_allocations = risk_allocations
        self.current_regime = current_regime
        
        # Prepare results
        results = {
            'portfolio_weights': portfolio_weights,
            'strategy_allocations': strategy_results,
            'asset_allocations': asset_allocations,
            'risk_allocations': risk_allocations,
            'correlation_matrix': correlation_matrix.to_dict() if isinstance(correlation_matrix, pd.DataFrame) else correlation_matrix,
            'strategy_metrics': strategy_metrics,
            'market_regime': current_regime,
            'optimization_method': optimization_method,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save results
        suffix = f"_{current_regime}" if current_regime else ""
        self._save_optimization_results(results, suffix)
        
        return results
    
    def _optimize_individual_strategy(self, strategy: str, 
                                     symbols: List[str],
                                     historical_data: Dict[str, pd.DataFrame],
                                     days: int,
                                     regime_specific: bool) -> Dict[str, Any]:
        """
        Optimize an individual trading strategy
        
        Args:
            strategy: Strategy name
            symbols: List of symbols
            historical_data: Historical price data
            days: Number of days to use
            regime_specific: Whether to use regime-specific optimization
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing {strategy} strategy for {len(symbols)} symbols")
        
        # Use standard grid/random parameter search
        from trading_bot.ml_pipeline.strategy_optimizer import StrategyOptimizer
        
        optimizer = StrategyOptimizer(strategy_type=strategy)
        
        # Define parameter grid based on strategy type
        if strategy == 'momentum':
            param_grid = {
                'lookback_days': [10, 20, 30, 40],
                'signal_threshold': [0.01, 0.02, 0.03, 0.05],
                'exit_threshold': [0.01, 0.02, 0.03]
            }
        elif strategy == 'mean_reversion':
            param_grid = {
                'lookback_days': [5, 10, 15, 20],
                'std_dev': [1.5, 2.0, 2.5, 3.0],
                'exit_std_dev': [0.5, 0.75, 1.0]
            }
        elif strategy == 'trend_following':
            param_grid = {
                'fast_period': [10, 20, 30],
                'slow_period': [40, 60, 80, 100],
                'signal_threshold': [0.0, 0.01, 0.02]
            }
        else:
            # Default parameters
            param_grid = {
                'lookback_days': [10, 20, 30],
                'signal_threshold': [0.01, 0.02, 0.03]
            }
        
        # Run optimization
        results = optimizer.grid_search(
            param_grid=param_grid,
            symbols=symbols,
            days=days,
            metric='sharpe_ratio'
        )
        
        return results
    
    def _calculate_strategy_returns(self, symbols: List[str], 
                                   strategies: List[str],
                                   historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Calculate historical returns for each strategy
        
        Args:
            symbols: List of symbols
            strategies: List of strategies
            historical_data: Historical price data
            
        Returns:
            Dictionary of strategy -> return series
        """
        strategy_returns = {}
        
        # Import required modules
        from trading_bot.strategies.strategy_factory import StrategyFactory
        
        # Calculate returns for each strategy
        for strategy_name in strategies:
            try:
                # Create strategy instance
                strategy = StrategyFactory.create_strategy(
                    strategy_name,
                    config={'optimize_mode': True}
                )
                
                # Get signals for each symbol and calculate returns
                all_returns = []
                
                for symbol in symbols:
                    if symbol in historical_data:
                        symbol_data = historical_data[symbol].copy()
                        
                        # Generate signals
                        signals = strategy.generate_signals(symbol_data)
                        
                        if signals and 'signals' in signals:
                            # Convert signals to position series
                            signal_series = pd.Series(index=symbol_data.index)
                            
                            for s in signals['signals']:
                                if 'date' in s and 'signal' in s:
                                    date = pd.to_datetime(s['date'])
                                    if date in signal_series.index:
                                        signal_series[date] = s['signal']
                            
                            # Forward fill positions
                            signal_series = signal_series.fillna(method='ffill').fillna(0)
                            
                            # Calculate daily returns
                            daily_returns = symbol_data['close'].pct_change()
                            
                            # Calculate strategy returns (position * next day's return)
                            strategy_return = signal_series.shift(1) * daily_returns
                            strategy_return = strategy_return.fillna(0)
                            
                            all_returns.append(strategy_return)
                
                # Combine returns across symbols (average)
                if all_returns:
                    # Align all series to the same index
                    if len(all_returns) > 1:
                        common_index = all_returns[0].index
                        for ret in all_returns[1:]:
                            common_index = common_index.intersection(ret.index)
                        
                        aligned_returns = [ret.loc[common_index] for ret in all_returns]
                        strategy_returns[strategy_name] = pd.concat(aligned_returns, axis=1).mean(axis=1)
                    else:
                        strategy_returns[strategy_name] = all_returns[0]
            
            except Exception as e:
                logger.error(f"Error calculating returns for {strategy_name}: {e}")
                # Use random returns as fallback
                dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
                strategy_returns[strategy_name] = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
        
        return strategy_returns
    
    def _calculate_correlation_matrix(self, 
                                     strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategies
        
        Args:
            strategy_returns: Dictionary of strategy -> return series
            
        Returns:
            Correlation matrix DataFrame
        """
        # Convert to DataFrame
        returns_df = pd.DataFrame(strategy_returns)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def _optimize_risk_parity(self, 
                             strategy_metrics: Dict[str, Dict[str, float]],
                             correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio using risk parity approach
        
        Args:
            strategy_metrics: Dictionary of strategy metrics
            correlation_matrix: Correlation matrix of strategy returns
            
        Returns:
            Dictionary of strategy weights
        """
        strategies = list(strategy_metrics.keys())
        
        # Extract volatility for each strategy (using 1/sharpe as proxy if not available)
        volatilities = {}
        for strategy, metrics in strategy_metrics.items():
            if 'volatility' in metrics:
                volatilities[strategy] = metrics['volatility']
            elif 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] > 0:
                # Approximate volatility as return / sharpe
                volatilities[strategy] = 0.1 / metrics['sharpe_ratio']  # Assume 10% return
            else:
                volatilities[strategy] = 0.2  # Default
        
        # Function to minimize for risk parity (equal risk contribution)
        def risk_parity_objective(weights):
            # Risk contribution should be equal
            n = len(weights)
            vol = np.array([volatilities[s] for s in strategies])
            corr = correlation_matrix.values if isinstance(correlation_matrix, pd.DataFrame) else correlation_matrix
            
            # Portfolio variance
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.outer(vol, vol) * corr, weights)))
            
            # Risk contribution
            marginal_risk = np.dot(np.outer(vol, vol) * corr, weights)
            risk_contribution = weights * marginal_risk / portfolio_vol
            
            # Target: equal risk contribution
            target_risk = portfolio_vol / n
            return ((risk_contribution - target_risk)**2).sum()
        
        # Initial guess: equal weights
        n = len(strategies)
        initial_weights = np.ones(n) / n
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Check result
        if result.success:
            # Convert to dictionary
            weights = {strategy: weight for strategy, weight in zip(strategies, result.x)}
            return weights
        else:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            # Return equal weights as fallback
            return {strategy: 1.0 / n for strategy in strategies}
    
    def _optimize_sharpe(self, 
                        strategy_metrics: Dict[str, Dict[str, float]],
                        strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Optimize portfolio to maximize Sharpe ratio
        
        Args:
            strategy_metrics: Dictionary of strategy metrics
            strategy_returns: Dictionary of strategy return series
            
        Returns:
            Dictionary of strategy weights
        """
        strategies = list(strategy_metrics.keys())
        
        # Convert returns to DataFrame and calculate mean/cov
        returns_df = pd.DataFrame(strategy_returns)
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Function to minimize (negative Sharpe ratio)
        def neg_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Initial guess: equal weights
        n = len(strategies)
        initial_weights = np.ones(n) / n
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # Optimize
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Check result
        if result.success:
            # Convert to dictionary
            weights = {strategy: weight for strategy, weight in zip(strategies, result.x)}
            return weights
        else:
            logger.warning(f"Sharpe optimization failed: {result.message}")
            # Return equal weights as fallback
            return {strategy: 1.0 / n for strategy in strategies}
    
    def _optimize_minimum_variance(self, 
                                  correlation_matrix: pd.DataFrame,
                                  strategy_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Optimize portfolio for minimum variance
        
        Args:
            correlation_matrix: Correlation matrix
            strategy_returns: Dictionary of strategy return series
            
        Returns:
            Dictionary of strategy weights
        """
        strategies = list(strategy_returns.keys())
        
        # Convert returns to DataFrame and calculate covariance
        returns_df = pd.DataFrame(strategy_returns)
        cov_matrix = returns_df.cov().values
        
        # Function to minimize (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Initial guess: equal weights
        n = len(strategies)
        initial_weights = np.ones(n) / n
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Check result
        if result.success:
            # Convert to dictionary
            weights = {strategy: weight for strategy, weight in zip(strategies, result.x)}
            return weights
        else:
            logger.warning(f"Minimum variance optimization failed: {result.message}")
            # Return equal weights as fallback
            return {strategy: 1.0 / n for strategy in strategies}
    
    def _calculate_risk_allocation(self, 
                                  portfolio_weights: Dict[str, float],
                                  strategy_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate risk allocation for each strategy
        
        Args:
            portfolio_weights: Strategy weight allocation
            strategy_metrics: Dictionary of strategy metrics
            
        Returns:
            Dictionary of risk allocation
        """
        # Extract volatility (risk) for each strategy
        volatilities = {}
        for strategy, metrics in strategy_metrics.items():
            if 'volatility' in metrics:
                volatilities[strategy] = metrics['volatility']
            elif 'max_drawdown' in metrics:
                # Use max drawdown as proxy for risk
                volatilities[strategy] = abs(metrics['max_drawdown'])
            else:
                volatilities[strategy] = 0.2  # Default
        
        # Calculate weighted risk
        risk_contribution = {}
        total_risk = 0
        
        for strategy, weight in portfolio_weights.items():
            if strategy in volatilities:
                risk = weight * volatilities[strategy]
                risk_contribution[strategy] = risk
                total_risk += risk
        
        # Normalize to percentage
        if total_risk > 0:
            risk_allocation = {
                strategy: risk / total_risk 
                for strategy, risk in risk_contribution.items()
            }
        else:
            # Equal risk allocation if no risk information
            n = len(portfolio_weights)
            risk_allocation = {
                strategy: 1.0 / n 
                for strategy in portfolio_weights.keys()
            }
        
        return risk_allocation
    
    def get_allocation_for_regime(self, regime: str) -> Dict[str, Any]:
        """
        Get optimal allocation for a specific market regime
        
        Args:
            regime: Market regime name
            
        Returns:
            Dict with allocation for the regime
        """
        # Check if we have regime-specific allocations
        result_files = []
        if os.path.exists(self.results_dir):
            result_files = [f for f in os.listdir(self.results_dir) 
                           if f.startswith('portfolio_allocation_') and 
                           f.endswith(f'_{regime}.json')]
        
        if result_files:
            # Sort by timestamp (newest first)
            result_files.sort(reverse=True)
            latest_file = os.path.join(self.results_dir, result_files[0])
            
            try:
                with open(latest_file, 'r') as f:
                    allocation = json.load(f)
                
                logger.info(f"Loaded {regime} regime allocation from {latest_file}")
                return allocation
            except Exception as e:
                logger.error(f"Error loading {regime} regime allocation: {e}")
        
        # Fall back to current allocation
        if self.portfolio_weights:
            return {
                'portfolio_weights': self.portfolio_weights,
                'strategy_allocations': self.strategy_allocations,
                'asset_allocations': self.asset_allocations,
                'risk_allocations': self.risk_allocations,
                'market_regime': self.current_regime,
                'is_fallback': True
            }
        
        # No allocation available
        return {
            'error': f"No allocation available for {regime} regime",
            'is_fallback': True
        }
    
    def apply_allocation(self, allocation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a portfolio allocation
        
        Args:
            allocation: Portfolio allocation dictionary
            
        Returns:
            Dict with application results
        """
        try:
            # Update internal state
            self.portfolio_weights = allocation.get('portfolio_weights', {})
            self.strategy_allocations = allocation.get('strategy_allocations', {})
            self.asset_allocations = allocation.get('asset_allocations', {})
            self.risk_allocations = allocation.get('risk_allocations', {})
            self.current_regime = allocation.get('market_regime', None)
            
            logger.info(f"Applied portfolio allocation for {self.current_regime} regime")
            
            return {
                'success': True,
                'message': f"Applied allocation for {self.current_regime} regime",
                'weights': self.portfolio_weights,
                'asset_allocations': self.asset_allocations
            }
        
        except Exception as e:
            logger.error(f"Error applying allocation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
