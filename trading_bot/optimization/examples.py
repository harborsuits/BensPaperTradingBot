import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

from .parameter_space import ParameterSpace
from .search_methods import GridSearch, RandomSearch, BayesianOptimization, GeneticAlgorithm
from .optimizer import ParameterOptimizer, OptimizationMetric, RegimeWeight, WalkForwardMethod

from ..backtesting.order_book_simulator import (
    OrderBookSimulator, MarketRegimeDetector, OrderSide
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_moving_average_example():
    """Example of optimizing moving average crossover strategy parameters"""
    # 1. Define parameter space
    param_space = ParameterSpace()
    param_space.add_integer_parameter("short_ma", 5, 50, 20, "Short moving average window")
    param_space.add_integer_parameter("long_ma", 20, 200, 50, "Long moving average window")
    param_space.add_range_parameter("stop_loss", 0.01, 0.1, 0.05, "Stop loss percentage")
    param_space.add_range_parameter("take_profit", 0.02, 0.2, 0.1, "Take profit percentage")
    param_space.add_boolean_parameter("use_trailing_stop", False, "Whether to use trailing stop")
    
    # 2. Choose search method (Grid, Random, Bayesian, or Genetic)
    search_method = GridSearch(param_space, num_points=5)
    # Alternative search methods:
    # search_method = RandomSearch(param_space, num_iterations=100)
    # search_method = BayesianOptimization(param_space, num_iterations=50)
    # search_method = GeneticAlgorithm(param_space, population_size=20, generations=5)
    
    # 3. Create optimizer
    optimizer = ParameterOptimizer(
        parameter_space=param_space,
        search_method=search_method,
        objective_metric=OptimizationMetric.SHARPE_RATIO,
        weight_strategy=RegimeWeight.EQUAL,
        is_maximizing=True,
        use_walk_forward=True,
        walk_forward_method=WalkForwardMethod.ROLLING,
        train_size=200,
        test_size=50,
        output_dir="optimization_results",
        verbose=True
    )
    
    # 4. Load data
    # Load your price data here, for example:
    prices = pd.Series(np.random.random(1000) * 100, index=pd.date_range("2020-01-01", periods=1000))
    
    # 5. Define regime detector for regime-aware optimization
    regime_detector = MarketRegimeDetector()
    optimizer.set_regime_detector(regime_detector)
    
    # 6. Define strategy evaluator function
    def evaluate_ma_strategy(params: Dict[str, Any], indices: List[int]) -> Dict[str, float]:
        """
        Evaluate moving average crossover strategy with given parameters
        
        Args:
            params: Strategy parameters
            indices: Data indices to use
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract parameters
        short_ma_window = params["short_ma"]
        long_ma_window = params["long_ma"]
        stop_loss = params["stop_loss"]
        take_profit = params["take_profit"]
        use_trailing_stop = params["use_trailing_stop"]
        
        # Extract price slice for this evaluation
        price_slice = prices.iloc[indices]
        
        # Calculate indicators
        short_ma = price_slice.rolling(window=short_ma_window).mean()
        long_ma = price_slice.rolling(window=long_ma_window).mean()
        
        # Calculate signals
        signals = pd.Series(0, index=price_slice.index)
        signals[short_ma > long_ma] = 1    # Buy signal
        signals[short_ma < long_ma] = -1   # Sell signal
        
        # Calculate returns
        position = signals.shift(1, fill_value=0)
        returns = position * price_slice.pct_change()
        
        # Apply stop loss and take profit
        # In a real implementation, this would be more sophisticated
        
        # Calculate performance metrics
        total_return = (1 + returns.fillna(0)).prod() - 1
        daily_returns = returns.fillna(0)
        
        sharpe_ratio = 0
        sortino_ratio = 0
        max_drawdown = 0
        
        if len(daily_returns) > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns / peak) - 1
            max_drawdown = drawdown.min()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
        }
    
    # 7. Run optimization
    result = optimizer.optimize(
        strategy_evaluator=evaluate_ma_strategy,
        prices=prices,
        max_evaluations=100
    )
    
    # 8. Print optimization results
    print("Optimization Results:")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best objective: {result['best_objective']}")
    print(f"Number of evaluations: {result['n_evaluations']}")
    print(f"Duration: {result['duration_seconds']:.2f} seconds")
    
    # 9. Generate visualizations
    optimizer.plot_walk_forward_results(result)
    optimizer.plot_regime_performance(result)
    
    return result

def run_volatility_based_sizing_example():
    """Example of optimizing volatility-based position sizing parameters"""
    # 1. Define parameter space
    param_space = ParameterSpace()
    param_space.add_range_parameter("target_volatility", 0.05, 0.30, 0.15, "Target annualized volatility")
    param_space.add_integer_parameter("vol_window", 10, 60, 20, "Window for volatility calculation")
    param_space.add_range_parameter("max_leverage", 1.0, 3.0, 2.0, "Maximum allowed leverage")
    param_space.add_boolean_parameter("use_atr", False, "Whether to use ATR-based sizing")
    param_space.add_range_parameter("atr_risk_factor", 1.0, 4.0, 2.0, "Risk factor for ATR-based sizing")
    
    # 2. Choose search method
    search_method = BayesianOptimization(param_space, num_iterations=30)
    
    # 3. Create optimizer
    optimizer = ParameterOptimizer(
        parameter_space=param_space,
        search_method=search_method,
        objective_metric=OptimizationMetric.SHARPE_RATIO,
        weight_strategy=RegimeWeight.DURATION,  # Weight regimes by their duration
        is_maximizing=True,
        use_walk_forward=True,
        walk_forward_method=WalkForwardMethod.EXPANDING,
        train_size=400,
        test_size=100,
        output_dir="volatility_optimization_results",
        verbose=True
    )
    
    # 4. Load data
    # For this example, we'll create synthetic OHLCV data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000)
    closes = np.random.random(1000) * 100 + 50
    
    # Add some trends and volatility regimes
    trend = np.linspace(0, 40, 1000) 
    closes = closes + trend
    
    # Add volatility regimes
    volatility = np.ones(1000)
    volatility[300:500] = 2.5  # Higher volatility period
    volatility[700:900] = 3.0  # Even higher volatility period
    
    # Create price movements
    for i in range(1, 1000):
        closes[i] = closes[i-1] * (1 + np.random.normal(0, 0.01 * volatility[i]))
    
    # Create OHLCV dataframe
    ohlcv = pd.DataFrame({
        'open': closes * (1 + np.random.normal(0, 0.005, 1000)),
        'high': closes * (1 + np.random.normal(0.01, 0.01, 1000)),
        'low': closes * (1 - np.random.normal(0.01, 0.01, 1000)),
        'close': closes,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Fix high/low values to be consistent
    for i in range(len(ohlcv)):
        ohlcv.iloc[i, 1] = max(ohlcv.iloc[i, [0, 1, 2, 3]])  # high
        ohlcv.iloc[i, 2] = min(ohlcv.iloc[i, [0, 1, 2, 3]])  # low
    
    # 5. Set up simulator and regime detector
    simulator = OrderBookSimulator(symbols=["SPY"])
    regime_detector = MarketRegimeDetector()
    optimizer.set_regime_detector(regime_detector)
    
    # 6. Define strategy evaluator function
    def evaluate_volatility_sizing(params: Dict[str, Any], indices: List[int]) -> Dict[str, float]:
        """
        Evaluate volatility-based position sizing with given parameters
        
        Args:
            params: Strategy parameters
            indices: Data indices to use
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract parameters
        target_volatility = params["target_volatility"]
        vol_window = params["vol_window"]
        max_leverage = params["max_leverage"]
        use_atr = params["use_atr"]
        atr_risk_factor = params["atr_risk_factor"]
        
        # Extract data slice for this evaluation
        data_slice = ohlcv.iloc[indices]
        
        # Initialize position and equity
        position = 0
        equity = 100000.0
        equity_curve = [equity]
        
        # Strategy logic - simple moving average crossover
        # with volatility-based position sizing
        short_ma = data_slice['close'].rolling(window=20).mean()
        long_ma = data_slice['close'].rolling(window=50).mean()
        
        # Calculate historical volatility
        returns = data_slice['close'].pct_change().fillna(0)
        vol = returns.rolling(window=vol_window).std() * np.sqrt(252)
        
        # Calculate ATR if used
        if use_atr:
            true_ranges = []
            for i in range(1, len(data_slice)):
                high = data_slice['high'].iloc[i]
                low = data_slice['low'].iloc[i]
                prev_close = data_slice['close'].iloc[i-1]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)
            
            atr = pd.Series(true_ranges, index=data_slice.index[1:]).rolling(window=14).mean()
            atr = atr.reindex(data_slice.index).fillna(0)
        
        # Simulate trading
        for i in range(vol_window + 1, len(data_slice)):
            if short_ma.iloc[i] > long_ma.iloc[i] and position <= 0:
                # Buy signal
                # Calculate position size based on volatility
                if use_atr:
                    atr_pct = atr.iloc[i] / data_slice['close'].iloc[i]
                    position_size = target_volatility / (atr_pct * atr_risk_factor * np.sqrt(252))
                    position_size = min(position_size, max_leverage)
                else:
                    if vol.iloc[i] > 0:
                        position_size = target_volatility / vol.iloc[i]
                        position_size = min(position_size, max_leverage)
                    else:
                        position_size = 1.0
                
                # Enter long position
                position = position_size
                
            elif short_ma.iloc[i] < long_ma.iloc[i] and position >= 0:
                # Sell signal
                # Exit position
                position = 0
            
            # Calculate equity change
            price_change = data_slice['close'].iloc[i] / data_slice['close'].iloc[i-1] - 1
            equity_change = position * price_change * equity
            equity += equity_change
            equity_curve.append(equity)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().fillna(0)
        
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown < 0 else 0
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "volatility": returns.std() * np.sqrt(252)
        }
    
    # 7. Run optimization
    result = optimizer.optimize(
        strategy_evaluator=evaluate_volatility_sizing,
        prices=ohlcv['close'],
        max_evaluations=30
    )
    
    # 8. Print optimization results
    print("Volatility Sizing Optimization Results:")
    print(f"Best parameters: {result['best_params']}")
    print(f"Best objective: {result['best_objective']}")
    print(f"Number of evaluations: {result['n_evaluations']}")
    print(f"Duration: {result['duration_seconds']:.2f} seconds")
    
    # 9. Generate visualizations
    optimizer.plot_walk_forward_results(result)
    optimizer.plot_regime_performance(result)
    
    return result

if __name__ == "__main__":
    print("Running Moving Average Optimization Example...")
    ma_result = run_moving_average_example()
    
    print("\nRunning Volatility-Based Sizing Optimization Example...")
    vol_result = run_volatility_based_sizing_example() 