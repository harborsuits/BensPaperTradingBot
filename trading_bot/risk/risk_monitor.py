import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

class RiskMonitor:
    """
    Monitors risk across multiple timeframes and portfolios.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.timeframes = {
            "daily": {"window": 1, "threshold": -3.0},
            "weekly": {"window": 5, "threshold": -8.0},
            "monthly": {"window": 20, "threshold": -15.0},
        }
        
        # Override timeframes if provided in config
        if "timeframes" in self.config:
            for timeframe, params in self.config["timeframes"].items():
                if timeframe in self.timeframes:
                    self.timeframes[timeframe].update(params)
        
        # Initialize portfolio tracking
        self.portfolio_values = {}
        self.portfolio_returns = {}
        self.portfolio_drawdowns = {}
        
        # Risk breach tracking
        self.risk_breaches = []
        
        # Stress test parameters
        self.stress_scenarios = {
            "market_crash": {"market": -20.0, "volatility": 50.0},
            "sector_rotation": {"market": -5.0, "volatility": 30.0},
            "volatility_spike": {"market": -10.0, "volatility": 100.0},
            "liquidity_crisis": {"market": -15.0, "volatility": 80.0}
        }
        
        # Override stress scenarios if provided in config
        if "stress_scenarios" in self.config:
            for scenario, params in self.config["stress_scenarios"].items():
                if scenario in self.stress_scenarios:
                    self.stress_scenarios[scenario].update(params)
                else:
                    self.stress_scenarios[scenario] = params
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def update_portfolio(self, portfolio_id, value, timestamp=None):
        """
        Update portfolio value and recalculate metrics.
        
        Args:
            portfolio_id: Identifier for the portfolio
            value: Current portfolio value
            timestamp: Timestamp for this update
        """
        timestamp = timestamp or datetime.now()
        
        # Initialize if first update
        if portfolio_id not in self.portfolio_values:
            self.portfolio_values[portfolio_id] = []
            self.portfolio_returns[portfolio_id] = []
        
        # Add new value
        self.portfolio_values[portfolio_id].append((timestamp, value))
        
        # Calculate return if we have previous values
        if len(self.portfolio_values[portfolio_id]) > 1:
            prev_time, prev_value = self.portfolio_values[portfolio_id][-2]
            current_return = (value / prev_value - 1) * 100
            self.portfolio_returns[portfolio_id].append((timestamp, current_return))
        
        # Update drawdown
        self._update_drawdown(portfolio_id)
        
        # Check for risk breaches
        self._check_breaches(portfolio_id, timestamp)
    
    def _update_drawdown(self, portfolio_id):
        """
        Recalculate drawdown for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
        """
        values = [v for _, v in self.portfolio_values[portfolio_id]]
        dates = [d for d, _ in self.portfolio_values[portfolio_id]]
        
        if len(values) < 2:
            return
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdown percentage
        drawdowns = [(value / peak - 1) * 100 for value, peak in zip(values, running_max)]
        
        # Store drawdown series
        self.portfolio_drawdowns[portfolio_id] = list(zip(dates, drawdowns))
    
    def _check_breaches(self, portfolio_id, timestamp):
        """
        Check for risk limit breaches across timeframes.
        
        Args:
            portfolio_id: Portfolio identifier
            timestamp: Current timestamp
        """
        returns = [(d, r) for d, r in self.portfolio_returns[portfolio_id] if d <= timestamp]
        
        if not returns:
            return
        
        # Check each timeframe
        for timeframe, params in self.timeframes.items():
            window = params["window"]
            threshold = params["threshold"]
            
            if len(returns) >= window:
                recent_returns = returns[-window:]
                cumulative_return = sum(r for _, r in recent_returns)
                
                if cumulative_return <= threshold:
                    breach = {
                        "portfolio_id": portfolio_id,
                        "timeframe": timeframe,
                        "timestamp": timestamp,
                        "threshold": threshold,
                        "actual": cumulative_return,
                        "severity": abs(cumulative_return / threshold)
                    }
                    self.risk_breaches.append(breach)
                    self.logger.warning(f"Risk breach in {portfolio_id}: {timeframe} return {cumulative_return:.2f}% exceeded threshold {threshold:.2f}%")
    
    def run_stress_test(self, portfolio_id, strategy_allocations, strategy_profiles):
        """
        Run stress tests on the current portfolio.
        
        Args:
            portfolio_id: Portfolio to test
            strategy_allocations: Current strategy allocations (percentage)
            strategy_profiles: Risk profiles for each strategy
        
        Returns:
            dict: Stress test results
        """
        if portfolio_id not in self.portfolio_values or not self.portfolio_values[portfolio_id]:
            self.logger.warning(f"No portfolio data found for {portfolio_id}")
            return {}
        
        results = {}
        
        # Get current portfolio value
        current_value = self.portfolio_values[portfolio_id][-1][1]
        
        for scenario, params in self.stress_scenarios.items():
            market_shock = params["market"]
            vol_shock = params["volatility"]
            
            # Initialize scenario impact
            scenario_impact = 0.0
            strategy_impacts = {}
            
            for strategy, allocation in strategy_allocations.items():
                if strategy in strategy_profiles:
                    # Get strategy beta and volatility sensitivity
                    beta = strategy_profiles[strategy].get("beta", 1.0)
                    vol_sensitivity = strategy_profiles[strategy].get("volatility_sensitivity", 0.5)
                    
                    # Calculate strategy impact
                    market_impact = beta * market_shock
                    vol_impact = vol_sensitivity * vol_shock / 100
                    
                    # Total strategy impact
                    strategy_impact = market_impact + vol_impact
                    
                    # Track individual strategy impact
                    strategy_impacts[strategy] = {
                        "allocation": allocation,
                        "impact_pct": strategy_impact,
                        "beta": beta,
                        "vol_sensitivity": vol_sensitivity
                    }
                    
                    # Apply allocation weight
                    scenario_impact += strategy_impact * (allocation / 100)
            
            # Store result
            results[scenario] = {
                "portfolio_impact_pct": scenario_impact,
                "portfolio_value_after": current_value * (1 + scenario_impact/100),
                "change_in_value": current_value * (scenario_impact/100),
                "strategy_impacts": strategy_impacts,
                "scenario_params": params
            }
        
        return results
    
    def get_risk_report(self, portfolio_id=None):
        """
        Generate a risk report for a portfolio or all portfolios.
        
        Args:
            portfolio_id: Optional portfolio identifier
            
        Returns:
            dict: Risk report data
        """
        if portfolio_id:
            portfolios = [portfolio_id]
        else:
            portfolios = list(self.portfolio_values.keys())
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolios": {}
        }
        
        for portfolio in portfolios:
            if portfolio not in self.portfolio_values:
                continue
                
            # Get most recent values
            current_value = self.portfolio_values[portfolio][-1][1] if self.portfolio_values[portfolio] else 0
            
            # Calculate current drawdown
            current_drawdown = self.portfolio_drawdowns[portfolio][-1][1] if self.portfolio_drawdowns[portfolio] else 0
            
            # Calculate recent returns
            recent_returns = {}
            for timeframe, params in self.timeframes.items():
                window = params["window"]
                if portfolio in self.portfolio_returns and len(self.portfolio_returns[portfolio]) >= window:
                    recent_returns[timeframe] = sum(r for _, r in self.portfolio_returns[portfolio][-window:])
                else:
                    recent_returns[timeframe] = None
            
            # Get recent breaches
            portfolio_breaches = [b for b in self.risk_breaches if b["portfolio_id"] == portfolio]
            recent_breaches = sorted(portfolio_breaches[-5:], key=lambda x: x["timestamp"], reverse=True) if portfolio_breaches else []
            
            # Add to report
            report["portfolios"][portfolio] = {
                "current_value": current_value,
                "current_drawdown": current_drawdown,
                "recent_returns": recent_returns,
                "recent_breaches": recent_breaches,
                "breach_count": len(portfolio_breaches)
            }
        
        return report
    
    def get_portfolio_metrics(self, portfolio_id):
        """
        Get risk metrics for a specific portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            dict: Portfolio risk metrics
        """
        if portfolio_id not in self.portfolio_values or not self.portfolio_values[portfolio_id]:
            return {}
        
        # Get portfolio values as pandas series
        dates = [d for d, _ in self.portfolio_values[portfolio_id]]
        values = [v for _, v in self.portfolio_values[portfolio_id]]
        portfolio_series = pd.Series(values, index=dates)
        
        # Get returns
        returns = [r for _, r in self.portfolio_returns[portfolio_id]]
        
        # Calculate metrics
        current_value = values[-1]
        max_value = max(values)
        min_value = min(values)
        current_drawdown = self.portfolio_drawdowns[portfolio_id][-1][1] if self.portfolio_drawdowns[portfolio_id] else 0
        max_drawdown = min([d for _, d in self.portfolio_drawdowns[portfolio_id]]) if self.portfolio_drawdowns[portfolio_id] else 0
        
        # Calculate volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)
        else:
            volatility = 0
        
        # Calculate VaR (Value at Risk)
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR
        else:
            var_95 = 0
            var_99 = 0
        
        return {
            "current_value": current_value,
            "max_value": max_value,
            "min_value": min_value,
            "return_pct": (current_value / values[0] - 1) * 100 if values else 0,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "var_95": var_95,
            "var_99": var_99,
            "breach_count": len([b for b in self.risk_breaches if b["portfolio_id"] == portfolio_id])
        }
        
    def reset(self):
        """Reset risk monitor state."""
        # Keep portfolio history for analysis but clear breaches
        self.risk_breaches = [] 