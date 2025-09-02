#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Risk Analytics Module for BensBot

This module implements sophisticated risk metrics including:
- Value at Risk (VaR) - Historical, Parametric, and Monte Carlo methods
- Conditional Value at Risk (CVaR) / Expected Shortfall
- Drawdown Analysis
- Stress Testing
- Risk Attribution

These metrics provide institutional-grade risk assessment capabilities
for portfolio management and trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedRiskAnalytics:
    """
    Advanced risk analytics capabilities for portfolio and strategy risk assessment.
    
    This class provides methods for calculating various risk metrics including VaR,
    CVaR, drawdowns, and stress testing for both portfolios and individual strategies.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """
        Initialize the risk analytics engine.
        
        Args:
            confidence_level: Confidence level for VaR calculations (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days for risk metrics
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.returns_data = None
        self.portfolio_weights = None
        
        logger.info(f"Advanced Risk Analytics initialized (confidence={confidence_level}, horizon={time_horizon})")
    
    def set_portfolio_data(self, returns_data: pd.DataFrame, portfolio_weights: Optional[Dict[str, float]] = None):
        """
        Set the portfolio data for risk calculations.
        
        Args:
            returns_data: DataFrame of historical returns (assets in columns, time in rows)
            portfolio_weights: Dictionary of weights by asset (optional, defaults to equal weighting)
        """
        self.returns_data = returns_data
        
        # If weights not provided, use equal weighting
        if portfolio_weights is None:
            assets = returns_data.columns
            self.portfolio_weights = {asset: 1.0 / len(assets) for asset in assets}
        else:
            # Normalize weights to sum to 1.0
            total_weight = sum(portfolio_weights.values())
            self.portfolio_weights = {k: v / total_weight for k, v in portfolio_weights.items()}
        
        logger.info(f"Portfolio data set: {len(returns_data)} data points, {len(self.portfolio_weights)} assets")
    
    def calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate weighted portfolio returns based on asset returns and weights.
        
        Returns:
            pd.Series: Time series of portfolio returns
        """
        if self.returns_data is None or self.portfolio_weights is None:
            raise ValueError("Returns data and portfolio weights must be set first")
        
        # Only use assets that exist in both returns data and weights
        common_assets = [asset for asset in self.portfolio_weights if asset in self.returns_data.columns]
        
        # Calculate weighted sum of returns
        weights_array = np.array([self.portfolio_weights[asset] for asset in common_assets])
        asset_returns = self.returns_data[common_assets].values
        
        portfolio_returns = np.dot(asset_returns, weights_array)
        return pd.Series(portfolio_returns, index=self.returns_data.index)
    
    def calculate_historical_var(self, returns: Optional[pd.Series] = None, 
                               confidence_level: Optional[float] = None) -> float:
        """
        Calculate historical Value at Risk (VaR).
        
        Args:
            returns: Series of returns (optional, uses portfolio returns if not provided)
            confidence_level: Confidence level (optional, uses default if not provided)
        
        Returns:
            float: Historical VaR value
        """
        if returns is None:
            returns = self.calculate_portfolio_returns()
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate the percentile
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        
        # Scale by time horizon using square root of time rule
        var = var * np.sqrt(self.time_horizon)
        
        logger.info(f"Historical VaR ({confidence_level*100:.1f}%, {self.time_horizon}-day): {var:.4f}")
        return var
    
    def calculate_parametric_var(self, returns: Optional[pd.Series] = None,
                               confidence_level: Optional[float] = None) -> float:
        """
        Calculate parametric Value at Risk assuming normal distribution.
        
        Args:
            returns: Series of returns (optional, uses portfolio returns if not provided)
            confidence_level: Confidence level (optional, uses default if not provided)
        
        Returns:
            float: Parametric VaR value
        """
        if returns is None:
            returns = self.calculate_portfolio_returns()
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate mean and standard deviation
        mu = returns.mean()
        sigma = returns.std()
        
        # Calculate z-score for the confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = -(mu + sigma * z_score)
        
        # Scale by time horizon
        var = var * np.sqrt(self.time_horizon)
        
        logger.info(f"Parametric VaR ({confidence_level*100:.1f}%, {self.time_horizon}-day): {var:.4f}")
        return var
    
    def calculate_monte_carlo_var(self, returns: Optional[pd.Series] = None,
                                confidence_level: Optional[float] = None,
                                num_simulations: int = 10000) -> float:
        """
        Calculate Monte Carlo VaR by simulating potential future returns.
        
        Args:
            returns: Series of returns (optional, uses portfolio returns if not provided)
            confidence_level: Confidence level (optional, uses default if not provided)
            num_simulations: Number of Monte Carlo simulations to run
            
        Returns:
            float: Monte Carlo VaR value
        """
        if returns is None:
            returns = self.calculate_portfolio_returns()
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate mean and covariance for the multivariate normal distribution
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random samples from the normal distribution
        simulated_returns = np.random.normal(mu, sigma, num_simulations)
        
        # Calculate VaR from the simulated distribution
        var = -np.percentile(simulated_returns, 100 * (1 - confidence_level))
        
        # Scale by time horizon
        var = var * np.sqrt(self.time_horizon)
        
        logger.info(f"Monte Carlo VaR ({confidence_level*100:.1f}%, {self.time_horizon}-day): {var:.4f}")
        return var
    
    def calculate_cvar(self, returns: Optional[pd.Series] = None,
                     confidence_level: Optional[float] = None) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns (optional, uses portfolio returns if not provided)
            confidence_level: Confidence level (optional, uses default if not provided)
            
        Returns:
            float: CVaR value
        """
        if returns is None:
            returns = self.calculate_portfolio_returns()
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate VaR threshold
        var = self.calculate_historical_var(returns, confidence_level)
        
        # Find all returns that exceed VaR
        tail_returns = returns[returns <= -var]
        
        # Calculate expected value of tail returns
        cvar = -tail_returns.mean()
        
        # Scale by time horizon
        cvar = cvar * np.sqrt(self.time_horizon)
        
        logger.info(f"CVaR/Expected Shortfall ({confidence_level*100:.1f}%, {self.time_horizon}-day): {cvar:.4f}")
        return cvar
    
    def calculate_max_drawdown(self, returns: Optional[pd.Series] = None) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate the maximum drawdown and its duration.
        
        Args:
            returns: Series of returns (optional, uses portfolio returns if not provided)
            
        Returns:
            Tuple of (max_drawdown, peak_date, valley_date)
        """
        if returns is None:
            returns = self.calculate_portfolio_returns()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns / running_max) - 1
        
        # Find maximum drawdown
        max_drawdown = drawdowns.min()
        max_drawdown_idx = drawdowns.idxmin()
        
        # Find the peak (before the max drawdown)
        peak_idx = running_max.loc[:max_drawdown_idx].idxmax()
        
        logger.info(f"Maximum Drawdown: {max_drawdown:.4f}, Peak: {peak_idx}, Valley: {max_drawdown_idx}")
        return max_drawdown, peak_idx, max_drawdown_idx
    
    def stress_test(self, scenarios: Dict[str, Dict[str, float]], 
                  portfolio_value: float = 100000.0) -> Dict[str, float]:
        """
        Perform stress testing on the portfolio based on defined scenarios.
        
        Args:
            scenarios: Dictionary of scenarios, where each scenario is a dict of asset -> shock value
            portfolio_value: Current portfolio value (default: $100,000)
            
        Returns:
            Dictionary of scenario names -> portfolio impact amounts
        """
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            # Calculate the weighted impact
            impact = 0.0
            for asset, shock in shocks.items():
                if asset in self.portfolio_weights:
                    impact += shock * self.portfolio_weights[asset]
            
            # Calculate dollar impact
            dollar_impact = impact * portfolio_value
            results[scenario_name] = dollar_impact
            
            logger.info(f"Stress Test - {scenario_name}: {impact:.2%} ({dollar_impact:.2f})")
        
        return results
    
    def generate_risk_report(self, portfolio_value: float = 100000.0) -> Dict[str, Union[float, str]]:
        """
        Generate a comprehensive risk report for the portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary containing all risk metrics
        """
        if self.returns_data is None:
            raise ValueError("Portfolio data must be set first")
        
        # Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns()
        
        # Calculate various risk metrics
        hist_var = self.calculate_historical_var(portfolio_returns)
        param_var = self.calculate_parametric_var(portfolio_returns)
        mc_var = self.calculate_monte_carlo_var(portfolio_returns)
        cvar = self.calculate_cvar(portfolio_returns)
        max_dd, peak, valley = self.calculate_max_drawdown(portfolio_returns)
        
        # Calculate dollar value for each metric
        var_dollar = hist_var * portfolio_value
        cvar_dollar = cvar * portfolio_value
        max_dd_dollar = abs(max_dd) * portfolio_value
        
        # Standard deviation (annualized)
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe = (portfolio_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
        
        # Prepare the report
        report = {
            "portfolio_value": portfolio_value,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "confidence_level": self.confidence_level,
            "time_horizon": self.time_horizon,
            "historical_var_pct": hist_var,
            "historical_var_dollar": var_dollar,
            "parametric_var_pct": param_var,
            "monte_carlo_var_pct": mc_var,
            "cvar_pct": cvar,
            "cvar_dollar": cvar_dollar,
            "max_drawdown_pct": max_dd,
            "max_drawdown_dollar": max_dd_dollar,
            "max_drawdown_peak": peak.strftime("%Y-%m-%d") if isinstance(peak, pd.Timestamp) else str(peak),
            "max_drawdown_valley": valley.strftime("%Y-%m-%d") if isinstance(valley, pd.Timestamp) else str(valley),
            "annualized_volatility": annual_vol,
            "sharpe_ratio": sharpe
        }
        
        logger.info("Risk report generated successfully")
        return report
    
    def plot_var_analysis(self, returns: Optional[pd.Series] = None,
                         confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> plt.Figure:
        """
        Create a visualization of VaR at different confidence levels.
        
        Args:
            returns: Series of returns (optional, uses portfolio returns if not provided)
            confidence_levels: List of confidence levels to plot
            
        Returns:
            matplotlib Figure object
        """
        if returns is None:
            returns = self.calculate_portfolio_returns()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the histogram of returns
        n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.6, color='lightblue')
        
        # Plot kernel density estimate
        returns_range = np.linspace(returns.min(), returns.max(), 1000)
        kernel = stats.gaussian_kde(returns)
        ax.plot(returns_range, kernel(returns_range), 'k-', lw=2)
        
        # Plot VaR lines for different confidence levels
        colors = ['r', 'g', 'b']
        for i, cl in enumerate(confidence_levels):
            var = self.calculate_historical_var(returns, cl)
            ax.axvline(-var, color=colors[i % len(colors)], linestyle='--', 
                      label=f'VaR {cl*100:.1f}%: {var:.4f}')
        
        ax.set_title('Value at Risk (VaR) Analysis', fontsize=14)
        ax.set_xlabel('Returns', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        
        return fig


class PortfolioRiskManager:
    """
    Risk management system for the trading portfolio.
    
    This class provides methods for monitoring portfolio risk,
    generating alerts when risk thresholds are breached, and
    recommending risk reduction actions.
    """
    
    def __init__(self, analytics: AdvancedRiskAnalytics,
                risk_limits: Dict[str, float] = None):
        """
        Initialize the portfolio risk manager.
        
        Args:
            analytics: Risk analytics engine
            risk_limits: Dictionary of risk metric limits
        """
        self.analytics = analytics
        self.risk_limits = risk_limits or {
            "var_limit": 0.03,  # 3% VaR limit
            "cvar_limit": 0.05,  # 5% CVaR limit
            "position_limit": 0.20,  # 20% maximum position size
            "sector_limit": 0.30,  # 30% maximum sector exposure
        }
        self.alerts = []
        
        logger.info("Portfolio Risk Manager initialized")
    
    def check_risk_limits(self, portfolio_value: float = 100000.0) -> List[Dict[str, str]]:
        """
        Check if any risk limits are breached and generate alerts.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            List of alert dictionaries
        """
        # Clear previous alerts
        self.alerts = []
        
        # Get risk metrics
        risk_report = self.analytics.generate_risk_report(portfolio_value)
        
        # Check VaR limit
        if risk_report["historical_var_pct"] > self.risk_limits["var_limit"]:
            self.alerts.append({
                "level": "HIGH",
                "type": "VAR_LIMIT_BREACH",
                "message": f"VaR ({risk_report['historical_var_pct']:.2%}) exceeds limit ({self.risk_limits['var_limit']:.2%})",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check CVaR limit
        if risk_report["cvar_pct"] > self.risk_limits["cvar_limit"]:
            self.alerts.append({
                "level": "HIGH",
                "type": "CVAR_LIMIT_BREACH",
                "message": f"CVaR ({risk_report['cvar_pct']:.2%}) exceeds limit ({self.risk_limits['cvar_limit']:.2%})",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check position concentration (if weights are available)
        if self.analytics.portfolio_weights:
            max_position = max(self.analytics.portfolio_weights.values())
            max_position_asset = max(self.analytics.portfolio_weights, 
                                   key=self.analytics.portfolio_weights.get)
            
            if max_position > self.risk_limits["position_limit"]:
                self.alerts.append({
                    "level": "MEDIUM",
                    "type": "POSITION_CONCENTRATION",
                    "message": f"Position in {max_position_asset} ({max_position:.2%}) exceeds limit ({self.risk_limits['position_limit']:.2%})",
                    "timestamp": datetime.now().isoformat()
                })
        
        return self.alerts
    
    def get_risk_reduction_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate recommendations for reducing portfolio risk.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Only make recommendations if alerts exist
        if not self.alerts:
            return recommendations
        
        # Analyze which assets contribute most to risk
        if self.analytics.returns_data is not None and self.analytics.portfolio_weights is not None:
            # Calculate asset volatilities
            asset_vols = {col: self.analytics.returns_data[col].std() * np.sqrt(252) 
                        for col in self.analytics.returns_data.columns
                        if col in self.analytics.portfolio_weights}
            
            # Calculate weighted contributions to risk
            risk_contributions = {asset: asset_vols[asset] * self.analytics.portfolio_weights[asset]
                               for asset in asset_vols.keys()}
            
            # Find top 3 risk contributors
            top_contributors = sorted(risk_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for asset, contribution in top_contributors:
                recommendations.append({
                    "action": "REDUCE_POSITION",
                    "asset": asset,
                    "reason": f"High risk contribution ({contribution:.2%} of portfolio risk)",
                    "suggestion": f"Consider reducing position in {asset} by 25-50%"
                })
        
        return recommendations
    
    def run_scenario_analysis(self, scenarios: Dict[str, Dict[str, float]],
                            portfolio_value: float = 100000.0) -> Dict[str, Dict[str, float]]:
        """
        Run scenario analysis to assess portfolio vulnerability to specific market events.
        
        Args:
            scenarios: Dictionary of scenario definitions
            portfolio_value: Current portfolio value
            
        Returns:
            Dict of scenario results including dollar impact and portfolio change
        """
        results = {}
        
        # Run stress tests
        impact_values = self.analytics.stress_test(scenarios, portfolio_value)
        
        # Calculate percentage impact for each scenario
        for scenario, impact in impact_values.items():
            results[scenario] = {
                "dollar_impact": impact,
                "percentage_impact": impact / portfolio_value,
                "post_scenario_value": portfolio_value + impact
            }
            
            # Generate alert if impact is severe (more than 10% loss)
            if impact < -0.1 * portfolio_value:
                self.alerts.append({
                    "level": "HIGH",
                    "type": "SCENARIO_VULNERABILITY",
                    "message": f"High vulnerability to '{scenario}' scenario: {impact/portfolio_value:.2%} impact",
                    "timestamp": datetime.now().isoformat()
                })
        
        return results


# Common market stress scenarios for testing
DEFAULT_STRESS_SCENARIOS = {
    "Market Crash": {
        "SPY": -0.20,
        "QQQ": -0.25,
        "IWM": -0.22,
        "AAPL": -0.30,
        "MSFT": -0.28,
        "GOOGL": -0.27,
        "AMZN": -0.32
    },
    "Interest Rate Spike": {
        "TLT": -0.10,
        "IEF": -0.06,
        "LQD": -0.08,
        "SPY": -0.05,
        "QQQ": -0.08,
        "XLF": 0.03
    },
    "Energy Crisis": {
        "XLE": 0.15,
        "XOP": 0.20,
        "SPY": -0.07,
        "XLU": -0.12,
        "XLI": -0.10
    },
    "Dollar Strength": {
        "UUP": 0.08,
        "EEM": -0.12,
        "FXI": -0.15,
        "GLD": -0.10,
        "SLV": -0.12
    },
    "Tech Sector Correction": {
        "QQQ": -0.15,
        "XLK": -0.18,
        "AAPL": -0.20,
        "MSFT": -0.18,
        "GOOGL": -0.22,
        "AMZN": -0.25
    }
}
