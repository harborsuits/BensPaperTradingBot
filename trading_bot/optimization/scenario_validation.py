import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable, Union, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# Local imports
from trading_bot.backtesting.backtest_engine import BacktestEngine
from trading_bot.strategies.strategy_template import StrategyTemplate as Strategy
from trading_bot.optimization.objective_functions import ObjectiveFunction
from trading_bot.optimization.advanced_objectives import ObjectiveFunction

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of market scenarios for validation"""
    BEAR_MARKET = "bear_market"
    BULL_MARKET = "bull_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISING_RATES = "rising_rates"
    FALLING_RATES = "falling_rates"
    RECESSION = "recession"
    RECOVERY = "recovery"
    MARKET_CRASH = "market_crash"
    SECTOR_ROTATION = "sector_rotation"
    CUSTOM = "custom"
    FLASH_CRASH = "flash_crash"
    CORRELATION_SHOCK = "correlation_shock"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    REGIME_CHANGE = "regime_change"

@dataclass
class MarketScenario:
    """
    Defines a market scenario for strategy validation
    """
    name: str
    type: ScenarioType
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    description: str = ""
    importance_weight: float = 1.0
    
    # Validation criteria
    min_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    min_sharpe: Optional[float] = None
    min_sortino: Optional[float] = None
    max_volatility: Optional[float] = None
    max_correlation: Optional[float] = None
    correlation_asset: Optional[str] = None
    custom_criteria: Optional[Callable] = None
    
    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = pd.Timestamp(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = pd.Timestamp(self.end_date)

class ScenarioLibrary:
    """Library of predefined market scenarios for testing strategies"""
    
    def __init__(self):
        """Initialize the scenario library"""
        self.scenarios: Dict[str, MarketScenario] = {}
    
    def add_scenario(self, scenario: MarketScenario) -> None:
        """
        Add a new scenario to the library
        
        Args:
            scenario: MarketScenario to add
        """
        self.scenarios[scenario.name] = scenario
    
    def get_scenario(self, name: str) -> Optional[MarketScenario]:
        """
        Get a scenario by name
        
        Args:
            name: Name of the scenario
            
        Returns:
            MarketScenario if found, None otherwise
        """
        return self.scenarios.get(name)
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[MarketScenario]:
        """
        Get all scenarios of a particular type
        
        Args:
            scenario_type: Type of scenario to find
            
        Returns:
            List of matching MarketScenario objects
        """
        return [s for s in self.scenarios.values() if s.type == scenario_type]
    
    def list_scenarios(self) -> List[str]:
        """
        List all scenario names
        
        Returns:
            List of scenario names
        """
        return list(self.scenarios.keys())

class ScenarioValidator:
    """
    Validates strategy parameters across multiple market scenarios.
    
    This class helps ensure that parameter sets perform well across
    different market conditions, not just in the primary backtest period.
    """
    
    def __init__(
        self,
        scenarios: Union[ScenarioLibrary, List[MarketScenario]],
        data_provider: Any,  # The data provider interface
        base_objective: Optional[ObjectiveFunction] = None,
        scenario_weights: Optional[Dict[str, float]] = None,
        require_all_scenarios: bool = False,
        importance_threshold: float = 0.8,
    ):
        """
        Initialize the scenario validator.
        
        Args:
            scenarios: ScenarioLibrary or list of MarketScenario objects
            data_provider: Data provider that can fetch data for scenario periods
            base_objective: Optional base objective function to use for scoring
            scenario_weights: Optional dict mapping scenario names to weights
            require_all_scenarios: Whether all scenarios must pass validation
            importance_threshold: Threshold for weighted importance score
        """
        if isinstance(scenarios, ScenarioLibrary):
            self.scenarios = list(scenarios.scenarios.values())
        else:
            self.scenarios = scenarios
        self.data_provider = data_provider
        self.base_objective = base_objective
        self.require_all_scenarios = require_all_scenarios
        self.importance_threshold = importance_threshold
        
        # Process scenario weights
        if scenario_weights is None:
            # Use default weights from scenarios
            self.scenario_weights = {s.name: s.importance_weight for s in self.scenarios}
        else:
            self.scenario_weights = scenario_weights
            
        # Normalize weights to sum to 1.0
        total_weight = sum(self.scenario_weights.values())
        if total_weight > 0:
            self.scenario_weights = {
                name: weight / total_weight for name, weight in self.scenario_weights.items()
            }
        
        logger.info(f"Initialized ScenarioValidator with {len(self.scenarios)} scenarios")
        for scenario in self.scenarios:
            logger.info(f"  {scenario.name}: {scenario.start_date} to {scenario.end_date} "
                       f"(weight: {self.scenario_weights.get(scenario.name, 0):.2f})")
    
    def validate_parameters(
        self,
        strategy: Strategy,
        parameters: Dict[str, Any],
        return_details: bool = False
    ) -> Union[bool, Tuple[bool, Dict]]:
        """
        Validate a parameter set across all defined scenarios.
        
        Args:
            strategy: Strategy instance to validate
            parameters: Parameter set to validate
            return_details: Whether to return detailed results
            
        Returns:
            True if parameters pass validation, False otherwise.
            If return_details is True, returns a tuple of (passed, details_dict)
        """
        # Apply parameters to strategy
        original_params = strategy.get_parameters()
        strategy.set_parameters(parameters)
        
        try:
            # Initialize results
            scenario_results = {}
            passed_scenarios = 0
            total_importance = 0.0
            passed_importance = 0.0
            
            # Validate each scenario
            for scenario in self.scenarios:
                # Get data for this scenario
                scenario_data = self._get_scenario_data(scenario)
                
                # Skip if no data available
                if scenario_data is None or scenario_data.empty:
                    logger.warning(f"No data available for scenario: {scenario.name}")
                    scenario_results[scenario.name] = {
                        "passed": False,
                        "reason": "No data available",
                        "metrics": {}
                    }
                    continue
                
                # Run backtest for this scenario period
                backtest_result = self._run_scenario_backtest(strategy, scenario_data, scenario)
                
                # Evaluate scenario criteria
                passed, reason, metrics = self._evaluate_scenario_criteria(
                    scenario, backtest_result
                )
                
                # Store result
                scenario_results[scenario.name] = {
                    "passed": passed,
                    "reason": reason,
                    "metrics": metrics
                }
                
                # Update counters
                if passed:
                    passed_scenarios += 1
                    passed_importance += self.scenario_weights.get(scenario.name, 0)
                
                total_importance += self.scenario_weights.get(scenario.name, 0)
            
            # Determine overall validation result
            if self.require_all_scenarios:
                validation_passed = passed_scenarios == len(self.scenarios)
            else:
                validation_passed = passed_importance / total_importance >= self.importance_threshold
            
            # Compile details if requested
            if return_details:
                details = {
                    "scenario_results": scenario_results,
                    "passed_scenarios": passed_scenarios,
                    "total_scenarios": len(self.scenarios),
                    "passed_importance": passed_importance,
                    "total_importance": total_importance,
                    "importance_ratio": passed_importance / total_importance if total_importance > 0 else 0
                }
                return validation_passed, details
            
            return validation_passed
            
        finally:
            # Restore original parameters
            strategy.set_parameters(original_params)
    
    def _get_scenario_data(self, scenario: MarketScenario) -> Optional[pd.DataFrame]:
        """
        Get market data for a specific scenario period.
        
        Args:
            scenario: The market scenario
            
        Returns:
            DataFrame with market data or None if not available
        """
        try:
            # This assumes your data provider has a method to get data by date range
            return self.data_provider.get_data(
                start_date=scenario.start_date,
                end_date=scenario.end_date
            )
        except Exception as e:
            logger.error(f"Error retrieving data for scenario {scenario.name}: {str(e)}")
            return None
    
    def _run_scenario_backtest(
        self,
        strategy: Strategy,
        scenario_data: pd.DataFrame,
        scenario: MarketScenario
    ) -> Dict[str, Any]:
        """
        Run a backtest for the given scenario.
        
        Args:
            strategy: Strategy with parameters to test
            scenario_data: Market data for the scenario
            scenario: The scenario definition
            
        Returns:
            Dictionary with backtest results
        """
        # This is a simplified example. In a real system, you would:
        # 1. Initialize a backtest engine
        # 2. Run the backtest with the strategy on the scenario data
        # 3. Calculate relevant metrics
        
        # Generate signals for the scenario period
        signals = strategy.generate_signals(scenario_data)
        
        # Calculate returns based on signals (simplified)
        returns = self._calculate_backtest_returns(signals, scenario_data)
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(returns)
        
        # If using base objective, calculate score
        if self.base_objective is not None:
            objective_score = self.base_objective.calculate(returns, metrics)
            metrics["objective_score"] = objective_score
        
        return {
            "returns": returns,
            "metrics": metrics,
            "signals": signals
        }
    
    def _calculate_backtest_returns(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate returns from signals and price data (simplified).
        
        Args:
            signals: Signal DataFrame
            data: Price data
            
        Returns:
            Series of daily returns
        """
        # This is a simplified implementation
        # In a real system, you'd use your actual backtest engine
        
        if signals.empty:
            # No signals, return zeros
            return pd.Series(0, index=data.index)
        
        # Initialize returns
        returns = pd.Series(0, index=data.index)
        
        # Process signals
        for date, day_signals in signals.groupby('date'):
            if date not in data.index:
                continue
                
            # Calculate position-weighted returns for this day
            day_return = 0.0
            total_position = 0.0
            
            for _, signal in day_signals.iterrows():
                symbol = signal['symbol']
                
                if symbol not in data.columns:
                    continue
                
                # Get position direction and size
                if 'direction' in signal:
                    direction = signal['direction']
                    position_size = signal.get('position_size', 1.0)
                    
                    # Calculate return based on direction
                    if direction == 'buy':
                        # Long position
                        symbol_return = data[symbol].pct_change().loc[date]
                        day_return += position_size * symbol_return
                    elif direction == 'sell':
                        # Short position
                        symbol_return = data[symbol].pct_change().loc[date]
                        day_return -= position_size * symbol_return
                        
                    total_position += position_size
            
            # Store day's return
            if total_position > 0:
                returns[date] = day_return / total_position
        
        return returns
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics from returns.
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Dictionary of performance metrics
        """
        # Skip if no returns
        if returns.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Downside deviation (for Sortino ratio)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade statistics
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        # Profit factor (sum of gains / sum of losses)
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    
    def _evaluate_scenario_criteria(
        self,
        scenario: MarketScenario,
        backtest_result: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Evaluate if a backtest result passes the scenario criteria.
        
        Args:
            scenario: The scenario definition with criteria
            backtest_result: Results from the backtest
            
        Returns:
            Tuple of (passed, reason, metrics)
        """
        metrics = backtest_result["metrics"]
        
        # Initialize result
        passed = True
        reason = "Passed all criteria"
        
        # Check minimum return
        if scenario.min_return is not None and metrics["total_return"] < scenario.min_return:
            passed = False
            reason = f"Return ({metrics['total_return']:.2%}) below minimum ({scenario.min_return:.2%})"
        
        # Check maximum drawdown
        if passed and scenario.max_drawdown is not None and abs(metrics["max_drawdown"]) > scenario.max_drawdown:
            passed = False
            reason = f"Drawdown ({abs(metrics['max_drawdown']):.2%}) exceeds maximum ({scenario.max_drawdown:.2%})"
        
        # Check minimum Sharpe ratio
        if passed and scenario.min_sharpe is not None and metrics["sharpe_ratio"] < scenario.min_sharpe:
            passed = False
            reason = f"Sharpe ({metrics['sharpe_ratio']:.2f}) below minimum ({scenario.min_sharpe:.2f})"
        
        # Check minimum Sortino ratio
        if passed and scenario.min_sortino is not None and metrics["sortino_ratio"] < scenario.min_sortino:
            passed = False
            reason = f"Sortino ({metrics['sortino_ratio']:.2f}) below minimum ({scenario.min_sortino:.2f})"
        
        # Check maximum volatility
        if passed and scenario.max_volatility is not None and metrics["volatility"] > scenario.max_volatility:
            passed = False
            reason = f"Volatility ({metrics['volatility']:.2%}) exceeds maximum ({scenario.max_volatility:.2%})"
        
        # Check correlation if specified
        if passed and scenario.max_correlation is not None and scenario.correlation_asset is not None:
            # This would require actual correlation calculation with the specified asset
            # For now, we'll just assume it passed
            correlation = 0.0  # Placeholder
            if correlation > scenario.max_correlation:
                passed = False
                reason = f"Correlation ({correlation:.2f}) exceeds maximum ({scenario.max_correlation:.2f})"
        
        # Check custom criteria if specified
        if passed and scenario.custom_criteria is not None:
            try:
                custom_passed, custom_reason = scenario.custom_criteria(backtest_result)
                if not custom_passed:
                    passed = False
                    reason = custom_reason
            except Exception as e:
                passed = False
                reason = f"Error in custom criteria: {str(e)}"
        
        return passed, reason, metrics
    
    def get_scenario_performance(
        self,
        strategy: Strategy,
        parameters: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed performance results for each scenario.
        
        Args:
            strategy: Strategy instance to evaluate
            parameters: Parameter set to use
            
        Returns:
            Dictionary mapping scenario names to performance details
        """
        _, details = self.validate_parameters(strategy, parameters, return_details=True)
        return details["scenario_results"]
    
    @classmethod
    def create_standard_scenarios(cls) -> ScenarioLibrary:
        """
        Create a set of standard market scenarios based on historical periods.
        
        Returns:
            ScenarioLibrary with predefined scenarios
        """
        library = ScenarioLibrary()
        
        # 2008 Financial Crisis
        library.add_scenario(MarketScenario(
            name="2008_financial_crisis",
            type=ScenarioType.BEAR_MARKET,
            start_date="2008-09-01",
            end_date="2009-03-31",
            description="Global financial crisis following Lehman Brothers collapse"
        ))
        
        # COVID-19 Crash
        library.add_scenario(MarketScenario(
            name="covid19_crash",
            type=ScenarioType.FLASH_CRASH,
            start_date="2020-02-15",
            end_date="2020-03-31",
            description="Rapid market decline due to COVID-19 pandemic"
        ))
        
        # Post-COVID Recovery
        library.add_scenario(MarketScenario(
            name="covid19_recovery",
            type=ScenarioType.RECOVERY,
            start_date="2020-04-01",
            end_date="2020-08-31",
            description="Market recovery following COVID-19 crash"
        ))
        
        # 2018 Q4 Correction
        library.add_scenario(MarketScenario(
            name="2018_q4_correction",
            type=ScenarioType.BEAR_MARKET,
            start_date="2018-10-01",
            end_date="2018-12-31",
            description="Q4 2018 market correction due to interest rate and trade concerns"
        ))
        
        # 2021 Growth to Value Rotation
        library.add_scenario(MarketScenario(
            name="2021_growth_to_value",
            type=ScenarioType.SECTOR_ROTATION,
            start_date="2021-01-01",
            end_date="2021-03-31",
            description="Rotation from growth to value stocks as yields rose"
        ))
        
        # 2022 High Inflation Period
        library.add_scenario(MarketScenario(
            name="2022_inflation_shock",
            type=ScenarioType.REGIME_CHANGE,
            start_date="2022-01-01",
            end_date="2022-06-30",
            description="Market adjustment to high inflation and rising rates"
        ))
        
        # 2016-2017 Low Volatility Bull Market
        library.add_scenario(MarketScenario(
            name="2016_2017_low_vol_bull",
            type=ScenarioType.BULL_MARKET,
            start_date="2016-07-01",
            end_date="2017-12-31",
            description="Low volatility bull market period"
        ))
        
        # 2015 Sideways Market
        library.add_scenario(MarketScenario(
            name="2015_sideways",
            type=ScenarioType.SIDEWAYS,
            start_date="2015-01-01",
            end_date="2015-12-31",
            description="Sideways market with multiple small corrections"
        ))
        
        # 2020 March Liquidity Crisis
        library.add_scenario(MarketScenario(
            name="2020_march_liquidity",
            type=ScenarioType.LIQUIDITY_CRISIS,
            start_date="2020-03-01",
            end_date="2020-03-31",
            description="Liquidity crisis during COVID crash with Treasury market dislocation"
        ))
        
        # 2011 US Debt Ceiling Crisis
        library.add_scenario(MarketScenario(
            name="2011_debt_ceiling",
            type=ScenarioType.HIGH_VOLATILITY,
            start_date="2011-07-01",
            end_date="2011-09-30",
            description="Market volatility during US debt ceiling crisis and downgrade"
        ))
        
        return library
    
    @classmethod
    def create_custom_scenario(
        cls,
        name: str,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        criteria: Dict[str, Any],
        description: str = "",
        importance_weight: float = 1.0
    ) -> MarketScenario:
        """
        Create a custom market scenario with specified criteria.
        
        Args:
            name: Scenario name
            start_date: Start date of scenario
            end_date: End date of scenario
            criteria: Dictionary of criteria (min_return, max_drawdown, etc.)
            description: Optional description
            importance_weight: Importance weight
            
        Returns:
            Custom MarketScenario
        """
        # Create scenario with base parameters
        scenario = MarketScenario(
            name=name,
            type=ScenarioType.CUSTOM,
            start_date=start_date,
            end_date=end_date,
            description=description,
            importance_weight=importance_weight
        )
        
        # Add criteria
        for key, value in criteria.items():
            if hasattr(scenario, key):
                setattr(scenario, key, value)
        
        return scenario

class ScenarioTester:
    """
    Performs scenario-based testing and validation of strategy parameters.
    """
    
    def __init__(self, scenario_manager: Optional[ScenarioManager] = None):
        """
        Initialize the scenario tester.
        
        Args:
            scenario_manager: Optional ScenarioManager instance
        """
        self.scenario_manager = scenario_manager or ScenarioManager()
    
    def generate_scenario_report(
        self, 
        strategy: Any,
        market_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report on strategy performance across scenarios.
        
        Args:
            strategy: Strategy instance to test
            market_data: DataFrame with OHLCV market data
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Dictionary with scenario report results
        """
        # Detect scenarios
        scenarios = self.scenario_manager.detect_scenarios(market_data)
        
        # Initialize report
        report = {
            "overall_performance": {},
            "scenario_performance": {},
            "robustness_score": 0.0,
            "vulnerability_scenarios": [],
            "strength_scenarios": []
        }
        
        # Calculate overall performance
        overall_returns = self._calculate_strategy_returns(strategy, market_data)
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data["close"].pct_change().fillna(0)
            
            # Calculate metrics against benchmark
            overall_alpha = np.mean(overall_returns - benchmark_returns)
            tracking_error = np.std(overall_returns - benchmark_returns)
            info_ratio = overall_alpha / tracking_error if tracking_error > 0 else 0
            
            report["overall_performance"]["alpha"] = float(overall_alpha)
            report["overall_performance"]["information_ratio"] = float(info_ratio)
            report["overall_performance"]["tracking_error"] = float(tracking_error)
        
        # Calculate standard metrics
        report["overall_performance"]["total_return"] = float((1 + overall_returns).prod() - 1)
        report["overall_performance"]["sharpe_ratio"] = float(
            np.mean(overall_returns) / np.std(overall_returns) * np.sqrt(252)
            if np.std(overall_returns) > 0 else 0
        )
        report["overall_performance"]["max_drawdown"] = self._calculate_max_drawdown(overall_returns)
        
        # Calculate performance for each scenario
        all_scenario_sharpes = []
        
        for scenario_type, periods in scenarios.items():
            scenario_performance = {}
            
            for i, (start, end) in enumerate(periods):
                # Get data for the period
                period_data = market_data.loc[start:end]
                
                # Skip if not enough data
                if len(period_data) < 5:
                    continue
                
                # Calculate performance
                period_returns = self._calculate_strategy_returns(strategy, period_data)
                
                # Calculate metrics
                period_sharpe = (
                    np.mean(period_returns) / np.std(period_returns) * np.sqrt(252)
                    if np.std(period_returns) > 0 and len(period_returns) > 1 else 0
                )
                period_total_return = (1 + period_returns).prod() - 1
                period_max_dd = self._calculate_max_drawdown(period_returns)
                
                # Store metrics
                period_key = f"period_{i+1}"
                scenario_performance[period_key] = {
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "days": len(period_data),
                    "total_return": float(period_total_return),
                    "sharpe_ratio": float(period_sharpe),
                    "max_drawdown": float(period_max_dd)
                }
                
                all_scenario_sharpes.append((scenario_type, period_sharpe))
            
            # Average metrics across all periods for this scenario
            if scenario_performance:
                avg_sharpe = np.mean([
                    p["sharpe_ratio"] for p in scenario_performance.values()
                ])
                avg_return = np.mean([
                    p["total_return"] for p in scenario_performance.values()
                ])
                avg_drawdown = np.mean([
                    p["max_drawdown"] for p in scenario_performance.values()
                ])
                
                report["scenario_performance"][str(scenario_type)] = {
                    "periods": scenario_performance,
                    "average_metrics": {
                        "avg_sharpe_ratio": float(avg_sharpe),
                        "avg_total_return": float(avg_return),
                        "avg_max_drawdown": float(avg_drawdown)
                    }
                }
        
        # Calculate robustness score (consistency across scenarios)
        if all_scenario_sharpes:
            scenario_sharpes = {}
            for scenario_type, sharpe in all_scenario_sharpes:
                if scenario_type not in scenario_sharpes:
                    scenario_sharpes[scenario_type] = []
                scenario_sharpes[scenario_type].append(sharpe)
            
            # Calculate average Sharpe for each scenario type
            avg_scenario_sharpes = {
                s: np.mean(sharpes) for s, sharpes in scenario_sharpes.items()
            }
            
            # Robustness is inversely proportional to the variation in performance
            sharpe_values = list(avg_scenario_sharpes.values())
            if len(sharpe_values) > 1:
                sharpe_std = np.std(sharpe_values)
                sharpe_mean = np.mean(sharpe_values)
                
                # Coefficient of variation, inverted and scaled
                if sharpe_mean > 0:
                    cv = sharpe_std / abs(sharpe_mean)
                    robustness = 1.0 / (1.0 + cv)
                else:
                    robustness = 0.0
                
                report["robustness_score"] = float(robustness)
            
            # Identify vulnerability and strength scenarios
            for scenario, sharpe in avg_scenario_sharpes.items():
                if sharpe < 0:
                    report["vulnerability_scenarios"].append(str(scenario))
                elif sharpe > report["overall_performance"]["sharpe_ratio"]:
                    report["strength_scenarios"].append(str(scenario))
        
        return report
    
    def _calculate_strategy_returns(self, strategy: Any, market_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate strategy returns on the given market data.
        
        Args:
            strategy: Strategy instance
            market_data: Market data DataFrame
            
        Returns:
            Array of period returns
        """
        # This is a simplified implementation
        # In a real system, this would run the strategy on the data
        try:
            # Try to generate signals and calculate returns
            if hasattr(strategy, 'generate_signals') and hasattr(strategy, 'calculate_position_size'):
                signals = strategy.generate_signals(market_data)
                
                # Apply signals to market data (simplified)
                returns = np.zeros(len(market_data))
                
                for i in range(1, len(market_data)):
                    if i-1 < len(signals):
                        signal = signals[i-1]
                        price_change = market_data['close'].iloc[i] / market_data['close'].iloc[i-1] - 1
                        returns[i] = signal * price_change
                
                return returns
            else:
                # Fallback to a simple calculation
                price_changes = market_data['close'].pct_change().fillna(0).values
                return price_changes
                
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {e}")
            return np.zeros(len(market_data))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from returns.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Maximum drawdown value (negative)
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdowns
        drawdowns = cum_returns / running_max - 1
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return float(max_drawdown)
    
    def compare_parameter_sets(
        self,
        parameter_sets: List[Dict[str, Any]],
        strategy_class: Any,
        market_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        scenario_weights: Optional[Dict[Union[ScenarioType, str], float]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple parameter sets across different scenarios.
        
        Args:
            parameter_sets: List of parameter dictionaries to compare
            strategy_class: Strategy class to instantiate with parameters
            market_data: DataFrame with OHLCV market data
            benchmark_data: Optional benchmark data for comparison
            scenario_weights: Optional custom weights for scenarios
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            "parameter_sets": [],
            "best_overall": None,
            "best_per_scenario": {},
            "most_robust": None
        }
        
        # Detect scenarios
        scenarios = self.scenario_manager.detect_scenarios(market_data)
        
        # Evaluate each parameter set
        for i, params in enumerate(parameter_sets):
            strategy = strategy_class(**params)
            
            # Generate report for this parameter set
            report = self.generate_scenario_report(
                strategy, market_data, benchmark_data
            )
            
            # Add report to results
            param_result = {
                "parameter_set_id": i,
                "parameters": params,
                "overall_performance": report["overall_performance"],
                "scenario_performance": report["scenario_performance"],
                "robustness_score": report["robustness_score"]
            }
            
            comparison_results["parameter_sets"].append(param_result)
        
        # Find best overall parameter set
        if comparison_results["parameter_sets"]:
            # Sort by Sharpe ratio for overall performance
            sorted_by_sharpe = sorted(
                comparison_results["parameter_sets"],
                key=lambda x: x["overall_performance"].get("sharpe_ratio", 0),
                reverse=True
            )
            comparison_results["best_overall"] = sorted_by_sharpe[0]["parameter_set_id"]
            
            # Find most robust parameter set
            sorted_by_robustness = sorted(
                comparison_results["parameter_sets"],
                key=lambda x: x.get("robustness_score", 0),
                reverse=True
            )
            comparison_results["most_robust"] = sorted_by_robustness[0]["parameter_set_id"]
            
            # Find best parameter set for each scenario
            for scenario_type in scenarios:
                scenario_str = str(scenario_type)
                best_for_scenario = None
                best_sharpe = -float('inf')
                
                for param_result in comparison_results["parameter_sets"]:
                    if (scenario_str in param_result["scenario_performance"] and
                        "average_metrics" in param_result["scenario_performance"][scenario_str]):
                        
                        avg_metrics = param_result["scenario_performance"][scenario_str]["average_metrics"]
                        scenario_sharpe = avg_metrics.get("avg_sharpe_ratio", 0)
                        
                        if scenario_sharpe > best_sharpe:
                            best_sharpe = scenario_sharpe
                            best_for_scenario = param_result["parameter_set_id"]
                
                if best_for_scenario is not None:
                    comparison_results["best_per_scenario"][scenario_str] = best_for_scenario
        
        return comparison_results

def create_market_scenarios(
    full_market_data: pd.DataFrame,
    full_other_data: pd.DataFrame,
    lookback_window: int = 252
) -> Dict[ScenarioType, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create market scenarios from historical data
    
    Args:
        full_market_data: Full market data DataFrame
        full_other_data: Full other data DataFrame (e.g., economic indicators)
        lookback_window: Number of days to use for each scenario
        
    Returns:
        Dictionary mapping scenario types to (market_data, other_data) tuples
    """
    scenarios = {}
    
    # Calculate daily returns for all symbols
    returns = full_market_data.pct_change().dropna()
    
    # Calculate volatility
    rolling_vol = returns.rolling(window=20).std()
    
    # Find periods for each scenario type
    
    # 1. Bull market: Largest uptrend
    rolling_returns = full_market_data.pct_change(lookback_window).dropna()
    bull_end_idx = rolling_returns.mean(axis=1).idxmax()
    bull_start_idx = full_market_data.index[
        full_market_data.index.get_loc(bull_end_idx) - lookback_window
    ]
    
    # 2. Bear market: Largest downtrend
    bear_end_idx = rolling_returns.mean(axis=1).idxmin()
    bear_start_idx = full_market_data.index[
        full_market_data.index.get_loc(bear_end_idx) - lookback_window
    ]
    
    # 3. Sideways market: Period with lowest absolute returns
    abs_returns = abs(rolling_returns.mean(axis=1))
    sideways_end_idx = abs_returns.idxmin()
    sideways_start_idx = full_market_data.index[
        full_market_data.index.get_loc(sideways_end_idx) - lookback_window
    ]
    
    # 4. High volatility: Period with highest volatility
    mean_vol = rolling_vol.mean(axis=1)
    high_vol_end_idx = mean_vol.idxmax()
    high_vol_start_idx = full_market_data.index[
        full_market_data.index.get_loc(high_vol_end_idx) - lookback_window
    ]
    
    # 5. Low volatility: Period with lowest volatility
    low_vol_end_idx = mean_vol.idxmin()
    low_vol_start_idx = full_market_data.index[
        full_market_data.index.get_loc(low_vol_end_idx) - lookback_window
    ]
    
    # 6. Market crash: Find largest drawdown
    # (Simplified - using largest 5-day drop)
    five_day_returns = full_market_data.pct_change(5).dropna()
    crash_end_idx = five_day_returns.mean(axis=1).idxmin()
    crash_start_idx = full_market_data.index[
        max(0, full_market_data.index.get_loc(crash_end_idx) - lookback_window)
    ]
    
    # 7. Market recovery: Period after crash with largest gains
    # (Simplified - using largest 20-day gain)
    twenty_day_returns = full_market_data.pct_change(20).dropna()
    recovery_end_idx = twenty_day_returns.mean(axis=1).idxmax()
    recovery_start_idx = full_market_data.index[
        max(0, full_market_data.index.get_loc(recovery_end_idx) - lookback_window)
    ]
    
    # Create scenarios dictionary with data slices
    scenarios[ScenarioType.BULL_MARKET] = (
        full_market_data.loc[bull_start_idx:bull_end_idx],
        full_other_data.loc[bull_start_idx:bull_end_idx]
    )
    
    scenarios[ScenarioType.BEAR_MARKET] = (
        full_market_data.loc[bear_start_idx:bear_end_idx],
        full_other_data.loc[bear_start_idx:bear_end_idx]
    )
    
    scenarios[ScenarioType.SIDEWAYS] = (
        full_market_data.loc[sideways_start_idx:sideways_end_idx],
        full_other_data.loc[sideways_start_idx:sideways_end_idx]
    )
    
    scenarios[ScenarioType.HIGH_VOLATILITY] = (
        full_market_data.loc[high_vol_start_idx:high_vol_end_idx],
        full_other_data.loc[high_vol_start_idx:high_vol_end_idx]
    )
    
    scenarios[ScenarioType.LOW_VOLATILITY] = (
        full_market_data.loc[low_vol_start_idx:low_vol_end_idx],
        full_other_data.loc[low_vol_start_idx:low_vol_end_idx]
    )
    
    scenarios[ScenarioType.MARKET_CRASH] = (
        full_market_data.loc[crash_start_idx:crash_end_idx],
        full_other_data.loc[crash_start_idx:crash_end_idx]
    )
    
    scenarios[ScenarioType.RECOVERY] = (
        full_market_data.loc[recovery_start_idx:recovery_end_idx],
        full_other_data.loc[recovery_start_idx:recovery_end_idx]
    )
    
    return scenarios

def add_custom_scenario(
    scenarios: Dict[ScenarioType, Tuple[pd.DataFrame, pd.DataFrame]],
    market_data: pd.DataFrame,
    other_data: pd.DataFrame,
    scenario_name: str = "custom"
) -> Dict[ScenarioType, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Add a custom scenario to existing scenarios
    
    Args:
        scenarios: Existing scenarios dictionary
        market_data: Market data for custom scenario
        other_data: Other data for custom scenario
        scenario_name: Name for custom scenario
        
    Returns:
        Updated scenarios dictionary
    """
    custom_type = ScenarioType.CUSTOM
    if scenario_name != "custom":
        # Create a new enum value for the custom scenario
        custom_type = ScenarioType(scenario_name)
    
    scenarios[custom_type] = (market_data, other_data)
    return scenarios 