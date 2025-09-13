#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimization Module for BensBot

This module implements portfolio optimization techniques:
- Mean-Variance (Markowitz) Optimization
- Risk Parity / Equal Risk Contribution
- Maximum Diversification
- Minimum Variance
- Maximum Sharpe Ratio
- Black-Litterman Model

These algorithms enable optimal asset allocation based on different
risk preferences and market expectations.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Union, Tuple, Optional, Callable
import logging
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Portfolio optimization engine implementing various allocation methods.
    """
    
    def __init__(self, returns_data: pd.DataFrame = None, risk_free_rate: float = 0.0):
        """
        Initialize the portfolio optimizer.
        
        Args:
            returns_data: DataFrame of historical returns (assets in columns, time in rows)
            risk_free_rate: Annualized risk-free rate (decimal)
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate statistics if data is provided
        if returns_data is not None:
            self._calculate_statistics()
        
        logger.info(f"Portfolio Optimizer initialized with {returns_data.shape[1] if returns_data is not None else 0} assets")
    
    def set_returns_data(self, returns_data: pd.DataFrame):
        """
        Set historical returns data and recalculate statistics.
        
        Args:
            returns_data: DataFrame of historical returns (assets in columns, time in rows)
        """
        self.returns_data = returns_data
        self._calculate_statistics()
        
        logger.info(f"Returns data set with {len(returns_data)} observations for {returns_data.shape[1]} assets")
    
    def set_risk_free_rate(self, risk_free_rate: float):
        """
        Set the risk-free rate.
        
        Args:
            risk_free_rate: Annualized risk-free rate (decimal)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        logger.info(f"Risk-free rate set to {risk_free_rate:.2%}")
    
    def _calculate_statistics(self):
        """
        Calculate means, covariance, and correlation matrices from returns data.
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before calculating statistics")
        
        # Calculate mean returns (annualized)
        self.mean_returns = self.returns_data.mean() * 252
        
        # Calculate covariance matrix (annualized)
        self.cov_matrix = self.returns_data.cov() * 252
        
        # Calculate correlation matrix
        self.corr_matrix = self.returns_data.corr()
    
    def _portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Array of asset weights
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Calculate portfolio return
        port_return = np.sum(self.mean_returns * weights)
        
        # Calculate portfolio volatility
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility
        
        return port_return, port_volatility, sharpe_ratio
    
    def optimize_mean_variance(self, target_return: Optional[float] = None, 
                             risk_aversion: Optional[float] = None,
                             constraints: Dict = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform mean-variance optimization (Markowitz Portfolio Theory).
        
        Args:
            target_return: Target portfolio return (optional)
            risk_aversion: Risk aversion coefficient (optional)
            constraints: Additional constraints (optional)
            
        Returns:
            Dictionary with weights and performance metrics
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before optimization")
        
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix)
        
        # Set default constraints (weights sum to 1.0)
        if constraints is None:
            constraints = {}
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Add any custom constraints
        for constraint_type, constraint_func in constraints.items():
            constraints_list.append({'type': constraint_type, 'fun': constraint_func})
        
        # Set bounds for weights (default: 0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weighting)
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Define optimization objective based on inputs
        if target_return is not None:
            # Minimize volatility subject to target return constraint
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Add target return constraint
            target_return_constraint = {
                'type': 'eq',
                'fun': lambda x: np.sum(x * self.mean_returns) - target_return
            }
            constraints_list.append(target_return_constraint)
            
        elif risk_aversion is not None:
            # Maximize utility function U = E[r] - 0.5 * λ * σ²
            def objective(weights):
                port_return, port_volatility, _ = self._portfolio_performance(weights)
                utility = port_return - 0.5 * risk_aversion * (port_volatility ** 2)
                return -utility  # Minimize negative utility
        else:
            # Maximize Sharpe ratio (default)
            def objective(weights):
                port_return, port_volatility, sharpe = self._portfolio_performance(weights)
                return -sharpe  # Minimize negative Sharpe
        
        # Run optimization
        result = sco.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'disp': False}
        )
        
        # Extract optimal weights
        weights = result['x']
        
        # Calculate portfolio performance metrics
        port_return, port_volatility, sharpe_ratio = self._portfolio_performance(weights)
        
        # Create output
        optimization_result = {
            'weights': {asset: weight for asset, weight in zip(self.returns_data.columns, weights)},
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result['success']
        }
        
        logger.info(f"Mean-Variance optimization completed: return={port_return:.4f}, vol={port_volatility:.4f}, sharpe={sharpe_ratio:.4f}")
        return optimization_result
    
    def optimize_minimum_variance(self, constraints: Dict = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find the minimum variance portfolio.
        
        Args:
            constraints: Additional constraints (optional)
            
        Returns:
            Dictionary with weights and performance metrics
        """
        # The minimum variance portfolio is a special case of mean-variance with no return target
        logger.info("Calculating minimum variance portfolio")
        return self.optimize_mean_variance(constraints=constraints)
    
    def optimize_maximum_sharpe(self, constraints: Dict = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find the maximum Sharpe ratio portfolio (tangency portfolio).
        
        Args:
            constraints: Additional constraints (optional)
            
        Returns:
            Dictionary with weights and performance metrics
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before optimization")
        
        num_assets = len(self.mean_returns)
        
        # Set default constraints (weights sum to 1.0)
        if constraints is None:
            constraints = {}
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Add any custom constraints
        for constraint_type, constraint_func in constraints.items():
            constraints_list.append({'type': constraint_type, 'fun': constraint_func})
        
        # Set bounds for weights (default: 0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weighting)
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Define objective function (negative Sharpe ratio)
        def negative_sharpe_ratio(weights):
            port_return, port_volatility, sharpe = self._portfolio_performance(weights)
            return -sharpe  # Minimize negative Sharpe
        
        # Run optimization
        result = sco.minimize(
            negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'disp': False}
        )
        
        # Extract optimal weights
        weights = result['x']
        
        # Calculate portfolio performance metrics
        port_return, port_volatility, sharpe_ratio = self._portfolio_performance(weights)
        
        # Create output
        optimization_result = {
            'weights': {asset: weight for asset, weight in zip(self.returns_data.columns, weights)},
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result['success']
        }
        
        logger.info(f"Maximum Sharpe optimization completed: return={port_return:.4f}, vol={port_volatility:.4f}, sharpe={sharpe_ratio:.4f}")
        return optimization_result
    
    def optimize_risk_parity(self, constraints: Dict = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate risk parity / equal risk contribution portfolio.
        
        Args:
            constraints: Additional constraints (optional)
            
        Returns:
            Dictionary with weights and performance metrics
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before optimization")
        
        num_assets = len(self.mean_returns)
        
        # Initial guess (equal weighting)
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Set bounds for weights (default: 0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Set constraints (weights sum to 1.0)
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Risk budget for each asset (equal in risk parity)
        risk_budget = np.array([1.0 / num_assets] * num_assets)
        
        # Objective function: minimize squared error between risk contributions and target
        def risk_budget_objective(weights):
            # Calculate portfolio risk
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Calculate risk contribution of each asset
            asset_rc = np.dot(self.cov_matrix, weights) * weights / portfolio_vol
            
            # Target risk contribution (equal for risk parity)
            target_risk_contribution = portfolio_vol * risk_budget
            
            # Return sum of squared errors
            return np.sum((asset_rc - target_risk_contribution)**2)
        
        # Run optimization
        result = sco.minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'disp': False}
        )
        
        # Extract optimal weights
        weights = result['x']
        
        # Calculate portfolio performance metrics
        port_return, port_volatility, sharpe_ratio = self._portfolio_performance(weights)
        
        # Create output
        optimization_result = {
            'weights': {asset: weight for asset, weight in zip(self.returns_data.columns, weights)},
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result['success']
        }
        
        logger.info(f"Risk Parity optimization completed: return={port_return:.4f}, vol={port_volatility:.4f}, sharpe={sharpe_ratio:.4f}")
        return optimization_result
    
    def optimize_maximum_diversification(self, constraints: Dict = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate maximum diversification portfolio.
        
        Args:
            constraints: Additional constraints (optional)
            
        Returns:
            Dictionary with weights and performance metrics
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before optimization")
        
        num_assets = len(self.mean_returns)
        
        # Initial guess (equal weighting)
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Set bounds for weights (default: 0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Set constraints (weights sum to 1.0)
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Objective function: maximize diversification ratio
        def negative_diversification_ratio(weights):
            # Calculate weighted average of asset volatilities
            weighted_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix)))
            
            # Calculate portfolio volatility
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Diversification ratio = weighted sum of vols / portfolio vol
            div_ratio = weighted_vol / port_vol
            
            return -div_ratio  # Minimize negative ratio
        
        # Run optimization
        result = sco.minimize(
            negative_diversification_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'disp': False}
        )
        
        # Extract optimal weights
        weights = result['x']
        
        # Calculate portfolio performance metrics
        port_return, port_volatility, sharpe_ratio = self._portfolio_performance(weights)
        
        # Create output
        optimization_result = {
            'weights': {asset: weight for asset, weight in zip(self.returns_data.columns, weights)},
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result['success']
        }
        
        logger.info(f"Maximum Diversification optimization completed: return={port_return:.4f}, vol={port_volatility:.4f}, sharpe={sharpe_ratio:.4f}")
        return optimization_result
    
    def optimize_black_litterman(self, market_cap_weights: Dict[str, float], 
                              views: Dict[str, float],
                              view_confidences: Dict[str, float],
                              tau: float = 0.05) -> Dict[str, Union[np.ndarray, float]]:
        """
        Implement Black-Litterman portfolio optimization.
        
        Args:
            market_cap_weights: Market cap weights for each asset
            views: Dictionary of views (asset -> expected return)
            view_confidences: Confidence in each view (0-1)
            tau: Scaling factor for uncertainty in prior
            
        Returns:
            Dictionary with weights and performance metrics
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before Black-Litterman optimization")
        
        # Extract assets and convert market cap weights to array
        assets = self.returns_data.columns
        mcap_weights = np.array([market_cap_weights.get(asset, 0) for asset in assets])
        
        # Normalize market cap weights
        mcap_weights = mcap_weights / np.sum(mcap_weights)
        
        # Calculate implied returns from market caps (reverse optimization)
        implied_returns = self.risk_free_rate + np.dot(self.cov_matrix, mcap_weights)
        
        # Set up views matrix and confidences
        P = np.zeros((len(views), len(assets)))
        q = np.zeros(len(views))
        omega = np.zeros((len(views), len(views)))
        
        for i, (asset, view) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.get_loc(asset)
                P[i, asset_idx] = 1
                q[i] = view
                omega[i, i] = (1 / view_confidences.get(asset, 0.5)) * np.diag(self.cov_matrix)[asset_idx]
        
        # Calculate posterior returns using Black-Litterman formula
        cov_scaled = tau * self.cov_matrix
        temp1 = np.dot(np.dot(P, cov_scaled), P.T) + omega
        temp2 = np.dot(np.dot(cov_scaled, P.T), np.linalg.inv(temp1))
        posterior_returns = implied_returns + np.dot(temp2, q - np.dot(P, implied_returns))
        
        # Use posterior returns for mean-variance optimization
        original_returns = self.mean_returns.copy()
        self.mean_returns = pd.Series(posterior_returns, index=assets)
        
        # Run mean-variance optimization with posterior returns
        result = self.optimize_mean_variance()
        
        # Restore original returns
        self.mean_returns = original_returns
        
        logger.info(f"Black-Litterman optimization completed: return={result['return']:.4f}, vol={result['volatility']:.4f}, sharpe={result['sharpe_ratio']:.4f}")
        return result
    
    def generate_efficient_frontier(self, points: int = 100, 
                                 min_return: Optional[float] = None,
                                 max_return: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the efficient frontier.
        
        Args:
            points: Number of portfolios to generate
            min_return: Minimum return for the frontier
            max_return: Maximum return for the frontier
            
        Returns:
            Tuple of (returns, volatilities, sharpe_ratios, weights)
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before generating efficient frontier")
        
        # Find minimum and maximum possible returns if not specified
        if min_return is None or max_return is None:
            assets = self.returns_data.columns
            
            # Find asset with min and max return
            min_asset = self.mean_returns.idxmin()
            max_asset = self.mean_returns.idxmax()
            
            min_ret = self.mean_returns[min_asset]
            max_ret = self.mean_returns[max_asset]
            
            # Add some padding
            if min_return is None:
                min_return = min_ret
            if max_return is None:
                max_return = max_ret
        
        # Create range of target returns
        target_returns = np.linspace(min_return, max_return, points)
        
        # Collect results
        efficient_returns = np.zeros(points)
        efficient_volatilities = np.zeros(points)
        efficient_sharpe_ratios = np.zeros(points)
        efficient_weights = []
        
        # Generate portfolios for each target return
        for i, target_return in enumerate(target_returns):
            result = self.optimize_mean_variance(target_return=target_return)
            
            # Store results
            efficient_returns[i] = result['return']
            efficient_volatilities[i] = result['volatility']
            efficient_sharpe_ratios[i] = result['sharpe_ratio']
            efficient_weights.append(result['weights'])
        
        logger.info(f"Generated efficient frontier with {points} portfolios")
        return efficient_returns, efficient_volatilities, efficient_sharpe_ratios, efficient_weights
    
    def plot_efficient_frontier(self, points: int = 100, 
                             show_assets: bool = True,
                             show_optimal: bool = True,
                             show_cml: bool = True,
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the efficient frontier.
        
        Args:
            points: Number of portfolios to generate
            show_assets: Whether to show individual assets
            show_optimal: Whether to show the optimal portfolio (max Sharpe)
            show_cml: Whether to show the capital market line
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be set before plotting efficient frontier")
        
        # Generate efficient frontier
        returns, volatilities, sharpe_ratios, _ = self.generate_efficient_frontier(points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        ax.plot(volatilities, returns, 'b-', linewidth=3, label='Efficient Frontier')
        
        # Plot individual assets if requested
        if show_assets:
            assets = self.returns_data.columns
            asset_returns = self.mean_returns
            asset_volatilities = np.sqrt(np.diag(self.cov_matrix))
            
            ax.scatter(asset_volatilities, asset_returns, marker='o', s=100, c='red', alpha=0.5, label='Assets')
            
            # Add asset labels
            for i, asset in enumerate(assets):
                ax.annotate(asset, (asset_volatilities[i], asset_returns[i]), 
                           xytext=(10, 0), textcoords='offset points', fontsize=8)
        
        # Plot optimal portfolio (max Sharpe) if requested
        if show_optimal:
            optimal = self.optimize_maximum_sharpe()
            ax.scatter(optimal['volatility'], optimal['return'], marker='*', s=200, c='green', 
                      label=f"Max Sharpe (SR={optimal['sharpe_ratio']:.2f})")
        
        # Plot capital market line if requested
        if show_cml and show_optimal:
            # Get optimal portfolio
            opt_return = optimal['return']
            opt_vol = optimal['volatility']
            
            # Create CML points
            x_min, x_max = ax.get_xlim()
            cml_x = np.linspace(0, x_max * 1.2, 100)
            slope = (opt_return - self.risk_free_rate) / opt_vol
            cml_y = self.risk_free_rate + slope * cml_x
            
            ax.plot(cml_x, cml_y, 'g--', label='Capital Market Line')
            ax.scatter(0, self.risk_free_rate, marker='o', s=80, c='black', label=f'Risk-Free ({self.risk_free_rate:.2%})')
        
        # Set labels and title
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        ax.set_title('Efficient Frontier', fontsize=14)
        
        # Set axis ticks to percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        return fig

# Helper class for rebalancing recommendations
class PortfolioRebalancer:
    """
    Generate rebalancing recommendations for an existing portfolio.
    """
    
    def __init__(self, optimizer: PortfolioOptimizer, 
                current_weights: Dict[str, float],
                threshold: float = 0.05):
        """
        Initialize the portfolio rebalancer.
        
        Args:
            optimizer: Portfolio optimizer instance
            current_weights: Current portfolio weights
            threshold: Threshold for rebalancing (minimum deviation)
        """
        self.optimizer = optimizer
        self.current_weights = current_weights
        self.threshold = threshold
        
        logger.info(f"Portfolio Rebalancer initialized with {len(current_weights)} positions")
    
    def calculate_optimal_weights(self, method: str = 'max_sharpe', **kwargs) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights based on the selected method.
        
        Args:
            method: Optimization method
            **kwargs: Additional parameters for the optimization method
            
        Returns:
            Dictionary of optimal weights
        """
        if method == 'min_variance':
            result = self.optimizer.optimize_minimum_variance(**kwargs)
        elif method == 'max_diversification':
            result = self.optimizer.optimize_maximum_diversification(**kwargs)
        elif method == 'risk_parity':
            result = self.optimizer.optimize_risk_parity(**kwargs)
        elif method == 'black_litterman':
            result = self.optimizer.optimize_black_litterman(**kwargs)
        else:  # default is max_sharpe
            result = self.optimizer.optimize_maximum_sharpe(**kwargs)
        
        return result['weights']
    
    def generate_rebalancing_recommendations(self, method: str = 'max_sharpe', 
                                          **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Generate rebalancing recommendations.
        
        Args:
            method: Optimization method
            **kwargs: Additional parameters for the optimization method
            
        Returns:
            Dictionary with buy/sell recommendations and target weights
        """
        # Calculate optimal weights
        optimal_weights = self.calculate_optimal_weights(method, **kwargs)
        
        # Compare with current weights
        recommendations = {'buy': {}, 'sell': {}, 'hold': {}}
        
        # Set of all assets (current + optimal)
        all_assets = set(list(self.current_weights.keys()) + list(optimal_weights.keys()))
        
        for asset in all_assets:
            current = self.current_weights.get(asset, 0)
            target = optimal_weights.get(asset, 0)
            
            # Calculate deviation
            deviation = target - current
            
            # Categorize based on deviation and threshold
            if abs(deviation) > self.threshold:
                if deviation > 0:
                    recommendations['buy'][asset] = deviation
                else:
                    recommendations['sell'][asset] = abs(deviation)
            else:
                recommendations['hold'][asset] = current
        
        # Add metadata
        recommendations['metadata'] = {
            'optimization_method': method,
            'current_weights': self.current_weights,
            'target_weights': optimal_weights,
            'threshold': self.threshold
        }
        
        logger.info(f"Generated rebalancing recommendations: {len(recommendations['buy'])} buys, {len(recommendations['sell'])} sells")
        return recommendations
