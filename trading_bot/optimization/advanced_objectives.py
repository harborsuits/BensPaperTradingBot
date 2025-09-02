import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional, Tuple, Union
from scipy import stats
import logging
from enum import Enum
import empyrical as ep

logger = logging.getLogger(__name__)

class ObjectiveDirection(Enum):
    """Direction for optimization objectives"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class RiskCategory(Enum):
    """Categories of risk metrics"""
    VOLATILITY = "volatility"          # Measures of dispersion (std dev, variance)
    DRAWDOWN = "drawdown"              # Measures of downside magnitude
    TAIL_RISK = "tail_risk"            # Measures of extreme events
    CONSISTENCY = "consistency"        # Measures of stability/reliability
    SENSITIVITY = "sensitivity"        # Measures of market sensitivity
    LIQUIDITY = "liquidity"            # Measures of liquidity risk
    BEHAVIORAL = "behavioral"          # Measures that account for behavioral biases


class ObjectiveFunction:
    """Base class for optimization objective functions"""
    
    def __init__(
        self,
        name: str,
        direction: ObjectiveDirection,
        risk_categories: List[RiskCategory],
        weight: float = 1.0,
        description: str = ""
    ):
        self.name = name
        self.direction = direction
        self.risk_categories = risk_categories
        self.weight = weight
        self.description = description
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute the objective value from a series of returns.
        
        Args:
            returns: Series of period returns
            **kwargs: Additional arguments for computation
            
        Returns:
            Computed objective value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __call__(self, returns: pd.Series, **kwargs) -> float:
        """Make the objective function callable."""
        return self.compute(returns, **kwargs)


# --- Standard Risk-Adjusted Metrics ---

class SharpeRatio(ObjectiveFunction):
    """Sharpe ratio objective function."""
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        period: str = "daily",
        weight: float = 1.0
    ):
        """
        Initialize Sharpe ratio objective.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            period: Return period ('daily', 'monthly', 'yearly')
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Sharpe Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.VOLATILITY],
            weight=weight,
            description="Ratio of excess returns to volatility"
        )
        self.risk_free_rate = risk_free_rate
        self.period = period
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Sharpe ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Sharpe ratio
        """
        if returns.empty:
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        # Handle too few data points
        if len(returns) < 2:
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate daily risk-free rate if needed
            if self.period == "daily":
                daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            elif self.period == "monthly":
                daily_rf = (1 + self.risk_free_rate) ** (1/12) - 1
            else:
                daily_rf = self.risk_free_rate
            
            # Use empyrical's sharpe ratio calculation if available
            if ep is not None:
                sharpe = ep.sharpe_ratio(
                    returns=returns,
                    risk_free=daily_rf,
                    period=self.period,
                    annualization=None
                )
            else:
                # Manual calculation
                excess_returns = returns - daily_rf
                sharpe = excess_returns.mean() / excess_returns.std(ddof=1)
                
                # Annualize if needed
                if self.period == "daily":
                    sharpe *= np.sqrt(252)
                elif self.period == "monthly":
                    sharpe *= np.sqrt(12)
            
            # Handle invalid sharpe ratio
            if np.isnan(sharpe) or np.isinf(sharpe):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return sharpe
            
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class SortinoRatio(ObjectiveFunction):
    """Sortino ratio objective function."""
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        period: str = "daily", 
        target_return: float = 0.0,
        weight: float = 1.0
    ):
        """
        Initialize Sortino ratio objective.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            period: Return period ('daily', 'monthly', 'yearly')
            target_return: Minimum acceptable return
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Sortino Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.DRAWDOWN],
            weight=weight,
            description="Ratio of excess returns to downside deviation"
        )
        self.risk_free_rate = risk_free_rate
        self.period = period
        self.target_return = target_return
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Sortino ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Sortino ratio
        """
        if returns.empty or len(returns) < 2:
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate period risk-free rate if needed
            if self.period == "daily":
                period_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            elif self.period == "monthly":
                period_rf = (1 + self.risk_free_rate) ** (1/12) - 1
            else:
                period_rf = self.risk_free_rate
            
            # Use empyrical's sortino ratio calculation if available
            if ep is not None:
                sortino = ep.sortino_ratio(
                    returns=returns,
                    required_return=period_rf,
                    period=self.period,
                    annualization=None
                )
            else:
                # Manual calculation
                excess_returns = returns - period_rf
                downside_returns = excess_returns[excess_returns < 0]
                
                if len(downside_returns) > 0:
                    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
                    sortino = excess_returns.mean() / downside_deviation
                else:
                    # No downside - infinite sortino ratio
                    return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
                
                # Annualize if needed
                if self.period == "daily":
                    sortino *= np.sqrt(252)
                elif self.period == "monthly":
                    sortino *= np.sqrt(12)
            
            # Handle invalid sortino ratio
            if np.isnan(sortino) or np.isinf(sortino):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return sortino
            
        except Exception as e:
            logger.warning(f"Error calculating Sortino ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class CalmarRatio(ObjectiveFunction):
    """Calmar ratio objective function."""
    
    def __init__(
        self,
        window: int = 36,
        weight: float = 1.0
    ):
        """
        Initialize Calmar ratio objective.
        
        Args:
            window: Months to use for max drawdown calculation
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Calmar Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.DRAWDOWN],
            weight=weight,
            description="Ratio of annualized return to maximum drawdown"
        )
        self.window = window
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Calmar ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Calmar ratio
        """
        if returns.empty or len(returns) < 2:
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            if max_drawdown >= 0:
                # No drawdown - infinite Calmar ratio
                return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
            
            # Calculate annualized returns
            n_years = len(returns) / 252  # Assuming daily returns
            annualized_return = (cum_returns.iloc[-1] ** (1 / n_years)) - 1
            
            # Calculate Calmar ratio
            calmar = annualized_return / abs(max_drawdown)
            
            if np.isnan(calmar) or np.isinf(calmar):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return calmar
            
        except Exception as e:
            logger.warning(f"Error calculating Calmar ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


# --- Advanced Risk-Adjusted Metrics ---

class OmegaRatio(ObjectiveFunction):
    """Omega ratio objective function."""
    
    def __init__(
        self,
        threshold: float = 0.0,
        weight: float = 1.0
    ):
        """
        Initialize Omega ratio objective.
        
        Args:
            threshold: Return threshold (minimum acceptable return)
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Omega Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.TAIL_RISK],
            weight=weight,
            description="Ratio of upside potential to downside risk relative to threshold"
        )
        self.threshold = threshold
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Omega ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Omega ratio
        """
        if returns.empty or len(returns) < 10:  # Need reasonable sample size
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Separate returns into positive and negative excess returns
            excess_returns = returns - self.threshold
            positive_excess = excess_returns[excess_returns >= 0]
            negative_excess = excess_returns[excess_returns < 0]
            
            if len(negative_excess) == 0:
                # No downside - infinite Omega ratio
                return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
            
            if len(positive_excess) == 0:
                # No upside - zero Omega ratio
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
            
            # Calculate Omega ratio
            omega = positive_excess.sum() / abs(negative_excess.sum())
            
            if np.isnan(omega) or np.isinf(omega):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return omega
            
        except Exception as e:
            logger.warning(f"Error calculating Omega ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class PainRatio(ObjectiveFunction):
    """Pain ratio objective function."""
    
    def __init__(
        self,
        weight: float = 1.0
    ):
        """
        Initialize Pain ratio objective.
        
        Args:
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Pain Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.DRAWDOWN, RiskCategory.CONSISTENCY],
            weight=weight,
            description="Ratio of annualized return to pain index (average drawdown)"
        )
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Pain ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Pain ratio
        """
        if returns.empty or len(returns) < 5:  # Need reasonable sample size
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdown series
            drawdowns = (cum_returns / running_max) - 1
            
            # Calculate pain index (average drawdown)
            pain_index = abs(drawdowns.mean())
            
            if pain_index == 0:
                # No pain - infinite Pain ratio
                return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
            
            # Calculate annualized returns
            n_years = len(returns) / 252  # Assuming daily returns
            annualized_return = (cum_returns.iloc[-1] ** (1 / n_years)) - 1
            
            # Calculate Pain ratio
            pain_ratio = annualized_return / pain_index
            
            if np.isnan(pain_ratio) or np.isinf(pain_ratio):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return pain_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating Pain ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class UlcerIndex(ObjectiveFunction):
    """Ulcer Index objective function."""
    
    def __init__(
        self,
        weight: float = 1.0
    ):
        """
        Initialize Ulcer Index objective.
        
        Args:
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Ulcer Index",
            direction=ObjectiveDirection.MINIMIZE,
            risk_categories=[RiskCategory.DRAWDOWN, RiskCategory.CONSISTENCY],
            weight=weight,
            description="Measure of drawdown severity/duration (root-mean-square of drawdowns)"
        )
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Ulcer Index from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Ulcer Index
        """
        if returns.empty or len(returns) < 5:  # Need reasonable sample size
            return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf
        
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate percentage drawdowns
            pct_drawdowns = (cum_returns / running_max) - 1
            
            # Calculate Ulcer Index (square root of mean squared drawdown)
            ulcer_index = np.sqrt((pct_drawdowns ** 2).mean())
            
            if np.isnan(ulcer_index) or np.isinf(ulcer_index):
                return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf
                
            return ulcer_index
            
        except Exception as e:
            logger.warning(f"Error calculating Ulcer Index: {e}")
            return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf


class MartinRatio(ObjectiveFunction):
    """Martin Ratio objective function (return / Ulcer Index)."""
    
    def __init__(
        self,
        weight: float = 1.0
    ):
        """
        Initialize Martin Ratio objective.
        
        Args:
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Martin Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.DRAWDOWN, RiskCategory.CONSISTENCY],
            weight=weight,
            description="Ratio of annualized return to Ulcer Index"
        )
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Martin Ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Martin Ratio
        """
        if returns.empty or len(returns) < 5:  # Need reasonable sample size
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate percentage drawdowns
            pct_drawdowns = (cum_returns / running_max) - 1
            
            # Calculate Ulcer Index (square root of mean squared drawdown)
            ulcer_index = np.sqrt((pct_drawdowns ** 2).mean())
            
            if ulcer_index == 0:
                # No drawdowns - infinite Martin Ratio
                return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
            
            # Calculate annualized returns
            n_years = len(returns) / 252  # Assuming daily returns
            annualized_return = (cum_returns.iloc[-1] ** (1 / n_years)) - 1
            
            # Calculate Martin Ratio
            martin_ratio = annualized_return / ulcer_index
            
            if np.isnan(martin_ratio) or np.isinf(martin_ratio):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return martin_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating Martin Ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class TailRiskMetric(ObjectiveFunction):
    """Value at Risk (VaR) or Conditional Value at Risk (CVaR) objective."""
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        use_cvar: bool = True,
        weight: float = 1.0
    ):
        """
        Initialize tail risk objective.
        
        Args:
            confidence_level: Confidence level (typically 0.95 or 0.99)
            use_cvar: Whether to use CVaR (True) or VaR (False)
            weight: Weight in multi-objective optimization
        """
        name = "CVaR" if use_cvar else "VaR"
        super().__init__(
            name=name,
            direction=ObjectiveDirection.MINIMIZE,
            risk_categories=[RiskCategory.TAIL_RISK],
            weight=weight,
            description=f"{'Conditional ' if use_cvar else ''}Value at Risk at {confidence_level:.0%} confidence"
        )
        self.confidence_level = confidence_level
        self.use_cvar = use_cvar
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute VaR or CVaR from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            VaR or CVaR value
        """
        if returns.empty or len(returns) < 10:  # Need reasonable sample size
            return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf
        
        try:
            # Calculate VaR
            var = returns.quantile(1 - self.confidence_level)
            
            if not self.use_cvar:
                # Return VaR
                if np.isnan(var) or np.isinf(var):
                    return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf
                return var
            
            # Calculate CVaR (mean of returns below VaR)
            cvar_returns = returns[returns <= var]
            if len(cvar_returns) > 0:
                cvar = cvar_returns.mean()
            else:
                # No returns below VaR
                cvar = var
            
            if np.isnan(cvar) or np.isinf(cvar):
                return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf
                
            return cvar
            
        except Exception as e:
            logger.warning(f"Error calculating {'CVaR' if self.use_cvar else 'VaR'}: {e}")
            return np.inf if self.direction == ObjectiveDirection.MINIMIZE else -np.inf


class InformationRatio(ObjectiveFunction):
    """Information Ratio objective function."""
    
    def __init__(
        self,
        benchmark_returns: pd.Series = None,
        weight: float = 1.0
    ):
        """
        Initialize Information Ratio objective.
        
        Args:
            benchmark_returns: Series of benchmark returns
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Information Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.CONSISTENCY, RiskCategory.SENSITIVITY],
            weight=weight,
            description="Ratio of active returns to tracking error"
        )
        self.benchmark_returns = benchmark_returns
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Information Ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Information Ratio
        """
        if returns.empty or len(returns) < 5:  # Need reasonable sample size
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Use provided benchmark returns if available
            benchmark_returns = self.benchmark_returns
            
            # If no benchmark provided, try to get it from kwargs
            if benchmark_returns is None:
                benchmark_returns = kwargs.get('benchmark_returns')
                
            # If still no benchmark, return NaN
            if benchmark_returns is None:
                logger.warning("No benchmark returns provided for Information Ratio")
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
            
            # Align benchmark to returns index
            benchmark_returns = benchmark_returns.reindex(returns.index)
            
            # Calculate active returns and tracking error
            active_returns = returns - benchmark_returns
            mean_active_return = active_returns.mean()
            tracking_error = active_returns.std(ddof=1)
            
            if tracking_error == 0:
                # No tracking error - infinite Information Ratio if positive active return
                if mean_active_return > 0:
                    return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
                else:
                    return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
            
            # Calculate Information Ratio
            information_ratio = mean_active_return / tracking_error
            
            if np.isnan(information_ratio) or np.isinf(information_ratio):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return information_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating Information Ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class GainToPainRatio(ObjectiveFunction):
    """Gain to Pain ratio objective function."""
    
    def __init__(
        self,
        weight: float = 1.0
    ):
        """
        Initialize Gain to Pain ratio objective.
        
        Args:
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Gain to Pain Ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.CONSISTENCY, RiskCategory.BEHAVIORAL],
            weight=weight,
            description="Ratio of sum of positive returns to absolute sum of negative returns"
        )
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Gain to Pain ratio from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Gain to Pain ratio
        """
        if returns.empty or len(returns) < 5:  # Need reasonable sample size
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Separate positive and negative returns
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                # No negative returns - infinite Gain to Pain ratio
                return np.inf if self.direction == ObjectiveDirection.MAXIMIZE else -np.inf
            
            # Calculate sums
            sum_positive = positive_returns.sum()
            sum_negative = abs(negative_returns.sum())
            
            # Calculate Gain to Pain ratio
            gain_to_pain = sum_positive / sum_negative
            
            if np.isnan(gain_to_pain) or np.isinf(gain_to_pain):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return gain_to_pain
            
        except Exception as e:
            logger.warning(f"Error calculating Gain to Pain ratio: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class ProspectTheoryMetric(ObjectiveFunction):
    """Prospect Theory utility objective function."""
    
    def __init__(
        self,
        lambda_loss_aversion: float = 2.25,
        alpha_value_fn: float = 0.88,
        beta_value_fn: float = 0.88,
        reference_return: float = 0.0,
        weight: float = 1.0
    ):
        """
        Initialize Prospect Theory utility objective.
        
        Args:
            lambda_loss_aversion: Loss aversion parameter (λ)
            alpha_value_fn: Value function parameter for gains (α)
            beta_value_fn: Value function parameter for losses (β)
            reference_return: Reference point for gains/losses
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Prospect Theory Utility",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.BEHAVIORAL],
            weight=weight,
            description="Behavioral utility based on Prospect Theory"
        )
        self.lambda_loss_aversion = lambda_loss_aversion
        self.alpha_value_fn = alpha_value_fn
        self.beta_value_fn = beta_value_fn
        self.reference_return = reference_return
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute Prospect Theory utility from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Prospect Theory utility
        """
        if returns.empty:
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate excess returns relative to reference point
            excess_returns = returns - self.reference_return
            
            # Separate gains and losses
            gains = excess_returns[excess_returns > 0]
            losses = excess_returns[excess_returns < 0]
            
            # Calculate utility components using prospect theory value function
            utility_gains = np.sum(gains ** self.alpha_value_fn) if len(gains) > 0 else 0
            utility_losses = np.sum((-losses) ** self.beta_value_fn) if len(losses) > 0 else 0
            
            # Apply loss aversion and calculate total utility
            prospect_utility = utility_gains - self.lambda_loss_aversion * utility_losses
            
            if np.isnan(prospect_utility) or np.isinf(prospect_utility):
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
                
            return prospect_utility
            
        except Exception as e:
            logger.warning(f"Error calculating Prospect Theory utility: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


class RegimePerformanceMetric(ObjectiveFunction):
    """Objective function for performance across different market regimes."""
    
    def __init__(
        self,
        regimes: Dict[str, pd.DatetimeIndex] = None,
        regime_weights: Dict[str, float] = None,
        base_metric: ObjectiveFunction = None,
        min_regime_coverage: float = 0.0,
        weight: float = 1.0
    ):
        """
        Initialize regime performance objective.
        
        Args:
            regimes: Dictionary mapping regime names to DatetimeIndex of dates
            regime_weights: Dictionary mapping regime names to weights
            base_metric: Base performance metric to use within each regime
            min_regime_coverage: Minimum fraction of regimes that must have data
            weight: Weight in multi-objective optimization
        """
        super().__init__(
            name="Regime-Adjusted Performance",
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=[RiskCategory.CONSISTENCY, RiskCategory.SENSITIVITY],
            weight=weight,
            description="Performance metric adjusted for different market regimes"
        )
        self.regimes = regimes or {}
        
        # Equal weights by default
        if regime_weights is None and regimes is not None:
            self.regime_weights = {k: 1.0 / len(regimes) for k in regimes.keys()}
        else:
            self.regime_weights = regime_weights or {}
        
        # Default to Sharpe ratio if no base metric provided
        self.base_metric = base_metric or SharpeRatio()
        self.min_regime_coverage = min_regime_coverage
        
        # Inherit direction from base metric
        self.direction = self.base_metric.direction
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute regime-adjusted performance from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Regime-adjusted performance metric
        """
        if returns.empty or not self.regimes:
            return self.base_metric.compute(returns, **kwargs)
        
        try:
            # Calculate performance in each regime
            regime_performance = {}
            covered_regimes = 0
            
            for regime_name, regime_dates in self.regimes.items():
                # Select returns for this regime
                regime_returns = returns.loc[returns.index.isin(regime_dates)]
                
                if not regime_returns.empty and len(regime_returns) >= 5:
                    # Calculate metric for this regime
                    regime_perf = self.base_metric.compute(regime_returns, **kwargs)
                    regime_performance[regime_name] = regime_perf
                    covered_regimes += 1
            
            # Check minimum regime coverage
            if len(self.regimes) > 0:
                coverage = covered_regimes / len(self.regimes)
                if coverage < self.min_regime_coverage:
                    return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
            
            # Calculate weighted average performance across regimes
            if not regime_performance:
                # No regimes with sufficient data
                return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for regime_name, perf in regime_performance.items():
                # Use provided weights or equal weighting
                weight = self.regime_weights.get(regime_name, 1.0 / len(regime_performance))
                weighted_sum += perf * weight
                weight_sum += weight
            
            if weight_sum > 0:
                regime_adjusted_perf = weighted_sum / weight_sum
            else:
                regime_adjusted_perf = -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
            
            return regime_adjusted_perf
            
        except Exception as e:
            logger.warning(f"Error calculating regime-adjusted performance: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf


# --- Multi-Objective Optimization ---

class CompositeObjective(ObjectiveFunction):
    """Composite objective function for multi-objective optimization."""
    
    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        weights: List[float] = None,
        aggregation_method: str = "weighted_sum",
        name: str = "Composite Objective"
    ):
        """
        Initialize composite objective.
        
        Args:
            objectives: List of objective functions
            weights: List of weights for each objective (defaults to internal weights)
            aggregation_method: Method to aggregate objectives ('weighted_sum', 'min', 'product')
            name: Name of the composite objective
        """
        # Collect risk categories from all objectives
        all_categories = []
        for obj in objectives:
            all_categories.extend(obj.risk_categories)
        
        # Remove duplicates while preserving order
        unique_categories = []
        for cat in all_categories:
            if cat not in unique_categories:
                unique_categories.append(cat)
        
        super().__init__(
            name=name,
            direction=ObjectiveDirection.MAXIMIZE,
            risk_categories=unique_categories,
            weight=1.0,
            description=f"Composite of {len(objectives)} objectives: {', '.join(obj.name for obj in objectives)}"
        )
        
        self.objectives = objectives
        
        # Use provided weights or get from objectives
        if weights is None:
            self.weights = [obj.weight for obj in objectives]
        else:
            if len(weights) != len(objectives):
                raise ValueError("Number of weights must match number of objectives")
            self.weights = weights
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
        
        self.aggregation_method = aggregation_method
    
    def compute(self, returns: pd.Series, **kwargs) -> float:
        """
        Compute composite objective value from returns.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Composite objective value
        """
        if returns.empty:
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
        
        try:
            # Calculate each objective
            objective_values = []
            normalized_values = []
            
            for i, objective in enumerate(self.objectives):
                # Compute the objective value
                value = objective.compute(returns, **kwargs)
                objective_values.append(value)
                
                # Normalize and handle direction
                if objective.direction == ObjectiveDirection.MINIMIZE:
                    # For minimization objectives, we invert the value
                    # after adding a small constant to avoid division by zero
                    if value > 0:
                        norm_value = 1.0 / (value + 1e-6)
                    else:
                        norm_value = 1.0
                else:
                    # For maximization objectives, we use the value directly
                    # but clip to avoid -inf
                    norm_value = max(value, -1e6)
                
                normalized_values.append(norm_value)
            
            # Aggregate objectives based on method
            if self.aggregation_method == "weighted_sum":
                # Weighted sum of normalized values
                composite_value = sum(v * w for v, w in zip(normalized_values, self.weights))
                
            elif self.aggregation_method == "min":
                # Minimum of normalized values (worst case)
                composite_value = min(normalized_values)
                
            elif self.aggregation_method == "product":
                # Product of normalized values (geometric mean flavor)
                # Add 1 to each value to ensure positivity
                adjusted_values = [max(0, v + 1) for v in normalized_values]
                composite_value = np.prod(adjusted_values)
                
            else:
                # Default to weighted sum
                composite_value = sum(v * w for v, w in zip(normalized_values, self.weights))
            
            return composite_value
            
        except Exception as e:
            logger.warning(f"Error calculating composite objective: {e}")
            return -np.inf if self.direction == ObjectiveDirection.MAXIMIZE else np.inf
    
    def get_component_values(self, returns: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Get individual component values for each objective.
        
        Args:
            returns: Series of period returns
            
        Returns:
            Dictionary mapping objective names to values
        """
        component_values = {}
        
        for objective in self.objectives:
            value = objective.compute(returns, **kwargs)
            component_values[objective.name] = value
        
        return component_values


# --- Create common objective instances ---

def create_sharpe_ratio() -> SharpeRatio:
    """Create default Sharpe ratio objective."""
    return SharpeRatio(risk_free_rate=0.02)

def create_sortino_ratio() -> SortinoRatio:
    """Create default Sortino ratio objective."""
    return SortinoRatio(risk_free_rate=0.02)

def create_calmar_ratio() -> CalmarRatio:
    """Create default Calmar ratio objective."""
    return CalmarRatio()

def create_omega_ratio() -> OmegaRatio:
    """Create default Omega ratio objective."""
    return OmegaRatio(threshold=0.0)

def create_gain_to_pain() -> GainToPainRatio:
    """Create default Gain to Pain ratio objective."""
    return GainToPainRatio()

def create_behavioral_objective() -> ProspectTheoryMetric:
    """Create default behavioral objective based on Prospect Theory."""
    return ProspectTheoryMetric()

def create_drawdown_adjusted() -> CompositeObjective:
    """Create drawdown-adjusted composite objective."""
    return CompositeObjective(
        objectives=[
            SharpeRatio(weight=0.5),
            CalmarRatio(weight=0.3),
            UlcerIndex(weight=0.2)
        ],
        name="Drawdown-Adjusted Performance"
    )

def create_tail_risk_adjusted() -> CompositeObjective:
    """Create tail risk adjusted composite objective."""
    return CompositeObjective(
        objectives=[
            SharpeRatio(weight=0.4),
            SortinoRatio(weight=0.3),
            TailRiskMetric(confidence_level=0.95, use_cvar=True, weight=0.3)
        ],
        name="Tail Risk-Adjusted Performance"
    )

def create_balanced_objective() -> CompositeObjective:
    """Create balanced composite objective."""
    return CompositeObjective(
        objectives=[
            SharpeRatio(weight=0.25),
            SortinoRatio(weight=0.2),
            CalmarRatio(weight=0.2),
            GainToPainRatio(weight=0.2),
            ProspectTheoryMetric(weight=0.15)
        ],
        name="Balanced Performance Metric"
    )

# Optimization functions that can be directly used as objectives

def maximize_sharpe(returns: List[float], **kwargs) -> float:
    """Maximize Sharpe ratio"""
    return SharpeRatio(**kwargs)(returns)

def maximize_sortino(returns: List[float], **kwargs) -> float:
    """Maximize Sortino ratio"""
    return SortinoRatio(**kwargs)(returns)

def maximize_calmar(returns: List[float], **kwargs) -> float:
    """Maximize Calmar ratio"""
    return CalmarRatio(**kwargs)(returns)

def minimize_drawdown(returns: List[float], **kwargs) -> float:
    """Minimize maximum drawdown (return negative for maximization)"""
    return -RiskAdjustedObjectives.max_drawdown(returns)

def maximize_gain_to_pain(returns: List[float], **kwargs) -> float:
    """Maximize Gain to Pain ratio"""
    return GainToPainRatio(**kwargs)(returns)

def maximize_multi_factor(returns: List[float], **kwargs) -> float:
    """Maximize multi-factor objective"""
    return RiskAdjustedObjectives.multi_factor_objective(returns, **kwargs)

def maximize_consistency_adjusted_returns(returns: List[float], **kwargs) -> float:
    """Maximize consistency-adjusted returns"""
    return RiskAdjustedObjectives.consistency_adjusted_returns(returns, **kwargs)

def maximize_information_ratio(
    returns: List[float], 
    benchmark_returns: List[float],
    **kwargs
) -> float:
    """Maximize information ratio relative to benchmark"""
    return RiskAdjustedObjectives.information_ratio(returns, benchmark_returns, **kwargs)

def risk_adjusted_return(backtest_results: Dict[str, Any], risk_weight: float = 1.0) -> float:
    """
    Calculate risk-adjusted return score that balances returns and risk
    
    Args:
        backtest_results: Dictionary of backtest results
        risk_weight: Weight for risk penalty (higher = more conservative)
        
    Returns:
        Risk-adjusted score (higher is better)
    """
    # Extract metrics from backtest results
    total_return = backtest_results.get("total_return", 0.0)
    max_drawdown = backtest_results.get("max_drawdown", 1.0)
    volatility = backtest_results.get("volatility", 0.01)
    
    # Avoid division by zero
    if max_drawdown < 0.0001:
        max_drawdown = 0.0001
    if volatility < 0.0001:
        volatility = 0.0001
    
    # Calculate score components
    return_component = total_return
    
    # Risk component combines drawdown and volatility
    risk_component = (max_drawdown * volatility) ** 0.5
    
    # Calculate final score
    score = return_component - (risk_weight * risk_component)
    
    return score

def sharpe_with_drawdown_constraint(
    backtest_results: Dict[str, Any], 
    max_acceptable_drawdown: float = 0.20,
    drawdown_penalty_factor: float = 10.0
) -> float:
    """
    Calculate Sharpe ratio with severe penalty for exceeding drawdown threshold
    
    Args:
        backtest_results: Dictionary of backtest results
        max_acceptable_drawdown: Maximum acceptable drawdown
        drawdown_penalty_factor: Penalty factor for exceeding max drawdown
        
    Returns:
        Modified Sharpe ratio (higher is better)
    """
    # Extract metrics from backtest results
    sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
    max_drawdown = backtest_results.get("max_drawdown", 1.0)
    
    # Calculate drawdown penalty
    if max_drawdown <= max_acceptable_drawdown:
        # No penalty if drawdown is within acceptable range
        drawdown_penalty = 1.0
    else:
        # Exponential penalty for exceeding max drawdown
        excess = max_drawdown - max_acceptable_drawdown
        drawdown_penalty = np.exp(-drawdown_penalty_factor * excess)
    
    # Apply penalty to Sharpe ratio
    adjusted_sharpe = sharpe_ratio * drawdown_penalty
    
    return adjusted_sharpe

def robust_regime_performance(
    backtest_results: Dict[str, Any],
    regime_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate performance score that rewards consistency across market regimes
    
    Args:
        backtest_results: Dictionary of backtest results, including regime-specific results
        regime_weights: Optional weights for different regimes (default: equal weights)
        
    Returns:
        Robust regime performance score (higher is better)
    """
    # Extract regime-specific results
    regime_results = backtest_results.get("regime_results", {})
    
    if not regime_results:
        # Fall back to overall Sharpe if no regime data
        return backtest_results.get("sharpe_ratio", 0.0)
    
    # Default to equal weights if not provided
    if regime_weights is None:
        regime_weights = {regime: 1.0 / len(regime_results) for regime in regime_results}
    
    # Calculate weighted regime scores
    regime_scores = []
    
    for regime, results in regime_results.items():
        # Use Sharpe ratio as the base metric for each regime
        regime_sharpe = results.get("sharpe_ratio", 0.0)
        
        # Get weight for this regime
        weight = regime_weights.get(regime, 0.0)
        
        regime_scores.append((regime_sharpe, weight))
    
    # Calculate weighted average
    total_weight = sum(weight for _, weight in regime_scores)
    
    if total_weight > 0:
        weighted_score = sum(score * weight for score, weight in regime_scores) / total_weight
    else:
        weighted_score = 0.0
    
    # Penalize variance in performance across regimes
    score_values = [score for score, _ in regime_scores]
    performance_variance = np.var(score_values) if len(score_values) > 1 else 0.0
    
    # Final score rewards high performance and consistency across regimes
    robust_score = weighted_score * (1.0 - min(0.5, performance_variance))
    
    return robust_score

def scenario_resilience_score(
    backtest_results: Dict[str, Any],
    scenario_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate score based on performance across stress scenarios
    
    Args:
        backtest_results: Dictionary of backtest results, including scenario results
        scenario_weights: Optional weights for different scenarios (default: equal weights)
        
    Returns:
        Scenario resilience score (higher is better)
    """
    # Extract scenario results
    scenario_results = backtest_results.get("scenario_results", {})
    
    if not scenario_results:
        # Fall back to overall Sharpe if no scenario data
        return backtest_results.get("sharpe_ratio", 0.0)
    
    # Default to equal weights if not provided
    if scenario_weights is None:
        scenario_weights = {scenario: 1.0 / len(scenario_results) for scenario in scenario_results}
    
    # Calculate weighted scenario scores
    total_score = 0.0
    total_weight = 0.0
    
    for scenario, results in scenario_results.items():
        # Return and drawdown for this scenario
        scenario_return = results.get("total_return", 0.0)
        scenario_max_dd = results.get("max_drawdown", 1.0)
        
        # Avoid division by zero
        if scenario_max_dd < 0.0001:
            scenario_max_dd = 0.0001
        
        # Calculate return-to-drawdown ratio for the scenario
        scenario_rtd = scenario_return / scenario_max_dd
        
        # Get weight for this scenario
        weight = scenario_weights.get(scenario, 0.0)
        
        total_score += scenario_rtd * weight
        total_weight += weight
    
    # Calculate final score
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.0

def create_multi_objective_function(
    objectives: List[Callable[[Dict[str, Any]], float]],
    weights: List[float]
) -> Callable[[Dict[str, Any]], float]:
    """
    Create a combined objective function from multiple objectives
    
    Args:
        objectives: List of objective functions
        weights: List of weights for each objective
        
    Returns:
        Combined objective function
    """
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else weights
    
    def combined_objective(backtest_results: Dict[str, Any]) -> float:
        """Combined objective function"""
        scores = []
        
        for objective, weight in zip(objectives, normalized_weights):
            # Calculate score for this objective
            score = objective(backtest_results)
            scores.append(score * weight)
        
        # Return weighted sum
        return sum(scores)
    
    return combined_objective

def create_walk_forward_objective(
    base_objective: Callable[[Dict[str, Any]], float],
    is_train_weight: float = 0.4,
    is_val_weight: float = 0.6
) -> Callable[[Dict[str, Any]], float]:
    """
    Create an objective function that balances in-sample and out-of-sample performance
    
    Args:
        base_objective: Base objective function to apply
        is_train_weight: Weight for in-sample (training) performance
        is_val_weight: Weight for out-of-sample (validation) performance
        
    Returns:
        Walk-forward objective function
    """
    def walk_forward_objective(backtest_results: Dict[str, Any]) -> float:
        """Walk-forward objective function"""
        # Extract in-sample and out-of-sample results
        is_results = backtest_results.get("in_sample_results", backtest_results)
        oos_results = backtest_results.get("out_of_sample_results", {})
        
        if not oos_results:
            # If no out-of-sample results, use just in-sample
            return base_objective(is_results)
        
        # Calculate scores for in-sample and out-of-sample
        is_score = base_objective(is_results)
        oos_score = base_objective(oos_results)
        
        # Final score is weighted average of in-sample and out-of-sample
        return (is_train_weight * is_score) + (is_val_weight * oos_score)
    
    return walk_forward_objective 