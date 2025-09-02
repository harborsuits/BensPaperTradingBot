import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from trading_bot.strategies.strategy_ensemble import StrategyEnsemble, WeightingMethod, DynamicEnsemble
from trading_bot.strategies.strategy_template import StrategyTemplate as Strategy
from trading_bot.strategies.macro_trend_strategy import MacroTrendStrategy
from trading_bot.strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime, RegimeDetector
from trading_bot.backtesting.performance_metrics import calculate_comprehensive_metrics
from trading_bot.backtesting.visualizations import plot_equity_curve, plot_drawdowns, plot_returns_distribution

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a simple mock strategy for testing
class MockStrategy(Strategy):
    def __init__(self, name: str, signal_generator, symbols: List[str] = None):
        super().__init__(symbols=symbols or ["SPY", "QQQ", "IWM", "GLD"])
        self.name = name
        self.signal_generator = signal_generator
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        return self.signal_generator(market_data)
    
    def calculate_position_size(self, signal: float, symbol: str, account_size: float) -> float:
        return signal * account_size * 0.1  # Use 10% of account per position


def generate_market_data(days: int = 252, symbols: List[str] = None):
    """Generate market data for testing"""
    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SHY"]
    
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Create DataFrame with dates as index
    data = pd.DataFrame(index=pd.DatetimeIndex(dates))
    
    # Generate prices for each symbol with different characteristics
    for symbol in symbols:
        # Base trend and volatility based on symbol
        if symbol == "SPY":
            trend = 0.00035  # ~8% annual return
            vol = 0.01
        elif symbol == "QQQ":
            trend = 0.00045  # ~12% annual return
            vol = 0.015
        elif symbol == "IWM":
            trend = 0.0003  # ~7% annual return
            vol = 0.013
        elif symbol == "GLD":
            trend = 0.0002  # ~5% annual return
            vol = 0.008
        elif symbol == "TLT":
            trend = 0.0001  # ~2.5% annual return
            vol = 0.009
        else:
            trend = 0.00005  # ~1% annual return
            vol = 0.001
        
        # Add regime changes
        prices = [100]
        for i in range(1, days):
            # Create regimes
            if i < days * 0.3:  # Bull market
                day_trend = trend * 1.5
                day_vol = vol * 0.8
            elif i < days * 0.5:  # Bear market
                day_trend = -trend
                day_vol = vol * 1.5
            elif i < days * 0.7:  # Sideways market
                day_trend = trend * 0.2
                day_vol = vol * 1.2
            else:  # Recovery
                day_trend = trend * 2
                day_vol = vol
            
            # Calculate price with trend and random noise
            price = prices[-1] * (1 + day_trend + np.random.normal(0, day_vol))
            prices.append(price)
        
        # Add to DataFrame
        data[symbol] = prices
    
    return data


def test_ensemble_strategy():
    """Test the StrategyEnsemble class with mock strategies"""
    # Generate market data
    market_data = generate_market_data(days=504, symbols=["SPY", "QQQ", "IWM", "GLD", "TLT", "SHY"])
    
    # Create mock strategies with different characteristics
    def trend_signal_generator(data):
        # Create signals based on 20-day and 50-day moving averages
        signals = pd.DataFrame(index=[data.index[-1]], columns=["SPY", "QQQ", "IWM", "GLD"])
        for symbol in ["SPY", "QQQ", "IWM", "GLD"]:
            if symbol in data.columns:
                ma_fast = data[symbol].rolling(20).mean()
                ma_slow = data[symbol].rolling(50).mean()
                signal = 1 if ma_fast.iloc[-1] > ma_slow.iloc[-1] else -1
                signals[symbol] = signal
        return signals
    
    def momentum_signal_generator(data):
        # Create signals based on 14-day RSI
        signals = pd.DataFrame(index=[data.index[-1]], columns=["SPY", "QQQ", "IWM", "GLD"])
        for symbol in ["SPY", "QQQ", "IWM", "GLD"]:
            if symbol in data.columns:
                # Calculate returns
                returns = data[symbol].pct_change()
                # Calculate 14-day RSI (simplified)
                gains = returns.copy()
                losses = returns.copy()
                gains[gains < 0] = 0
                losses[losses > 0] = 0
                avg_gain = gains.rolling(14).mean()
                avg_loss = -losses.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Generate signal based on RSI
                if rsi.iloc[-1] > 70:
                    signal = -0.5  # Overbought
                elif rsi.iloc[-1] < 30:
                    signal = 1  # Oversold
                else:
                    signal = 0  # Neutral
                
                signals[symbol] = signal
        return signals
    
    def mean_reversion_signal_generator(data):
        # Create signals based on Bollinger Bands
        signals = pd.DataFrame(index=[data.index[-1]], columns=["SPY", "QQQ", "IWM", "GLD"])
        for symbol in ["SPY", "QQQ", "IWM", "GLD"]:
            if symbol in data.columns:
                # Calculate 20-day moving average and standard deviation
                ma = data[symbol].rolling(20).mean()
                std = data[symbol].rolling(20).std()
                
                # Calculate Bollinger Bands
                upper_band = ma + 2 * std
                lower_band = ma - 2 * std
                
                # Generate signal based on price relative to bands
                price = data[symbol].iloc[-1]
                if price > upper_band.iloc[-1]:
                    signal = -1  # Sell
                elif price < lower_band.iloc[-1]:
                    signal = 1  # Buy
                else:
                    signal = 0  # Hold
                
                signals[symbol] = signal
        return signals
    
    # Create mock strategies
    trend_strategy = MockStrategy("Trend Strategy", trend_signal_generator)
    momentum_strategy = MockStrategy("Momentum Strategy", momentum_signal_generator)
    mean_rev_strategy = MockStrategy("Mean Reversion Strategy", mean_reversion_signal_generator)
    
    # Try each weighting method
    weighting_methods = [
        WeightingMethod.EQUAL,
        WeightingMethod.PERFORMANCE,
        WeightingMethod.VOLATILITY,
        WeightingMethod.ADAPTIVE
    ]
    
    results = {}
    
    for method in weighting_methods:
        logger.info(f"Testing ensemble with {method.value} weighting method")
        
        # Create ensemble strategy
        ensemble = StrategyEnsemble(
            strategies=[trend_strategy, momentum_strategy, mean_rev_strategy],
            weighting_method=method,
            performance_window=60,
            rebalance_frequency=20,
            correlation_threshold=0.7,
            min_weight=0.1,
            max_weight=0.6
        )
        
        # Backtest the ensemble
        backtest_results = backtest_strategy(ensemble, market_data)
        
        # Store results
        results[method.value] = backtest_results
    
    # Create and test a dynamic ensemble
    logger.info("Testing dynamic ensemble")
    dynamic_ensemble = DynamicEnsemble(
        strategies=[trend_strategy, momentum_strategy, mean_rev_strategy],
        min_active_strategies=1,
        max_active_strategies=3,
        activation_threshold=0.2,
        deactivation_threshold=-0.1,
        weighting_method=WeightingMethod.PERFORMANCE
    )
    
    # Backtest the dynamic ensemble
    dynamic_results = backtest_strategy(dynamic_ensemble, market_data)
    results["dynamic"] = dynamic_results
    
    # Compare results
    compare_results(results)


def backtest_strategy(strategy, market_data):
    """Simple backtester for strategy testing"""
    # Initialize portfolio
    initial_capital = 100000
    portfolio_value = initial_capital
    portfolio_history = [portfolio_value]
    returns_history = [0]
    weights_history = []
    
    # Track positions
    positions = {symbol: 0 for symbol in strategy.symbols}
    
    # Start from 50th day to have enough data for indicators
    start_idx = 50
    
    # Iterate through each day
    for i in range(start_idx, len(market_data) - 1):
        # Get data up to current day
        current_data = market_data.iloc[:i+1]
        
        # Generate signals
        signals = strategy.generate_signals(current_data)
        
        if signals.empty:
            continue
        
        # Record weights if using StrategyEnsemble
        if hasattr(strategy, 'get_strategy_weights'):
            weights_history.append(strategy.get_strategy_weights())
        
        # Update positions based on signals
        day_pnl = 0
        
        for symbol in strategy.symbols:
            if symbol in signals.columns and symbol in market_data.columns:
                signal = signals[symbol].iloc[-1]
                
                # Calculate new position
                new_position = strategy.calculate_position_size(signal, symbol, portfolio_value)
                
                # Calculate change in position
                position_change = new_position - positions[symbol]
                
                # Update position
                positions[symbol] = new_position
                
                # Calculate P&L for next day
                price_change = market_data[symbol].iloc[i+1] / market_data[symbol].iloc[i] - 1
                position_pnl = positions[symbol] * price_change
                day_pnl += position_pnl
        
        # Update portfolio value
        portfolio_value += day_pnl
        portfolio_history.append(portfolio_value)
        
        # Calculate daily return
        daily_return = day_pnl / portfolio_history[-2]
        returns_history.append(daily_return)
    
    # Calculate performance metrics
    returns = np.array(returns_history[1:])  # Skip first 0 return
    equity_curve = np.array(portfolio_history)
    
    metrics = calculate_comprehensive_metrics(
        returns=returns,
        equity_curve=equity_curve,
        risk_free_rate=0.02,  # 2% annual risk-free rate
        annualization_factor=252  # Daily returns
    )
    
    # Add portfolio history and weights
    metrics['portfolio_history'] = portfolio_history
    metrics['returns_history'] = returns_history
    metrics['weights_history'] = weights_history
    
    return metrics


def compare_results(results):
    """Compare backtest results from different weighting methods"""
    # Extract key metrics for comparison
    comparison = pd.DataFrame(index=list(results.keys()))
    
    # Common metrics to compare
    metrics_to_compare = [
        'annualized_return',
        'annualized_volatility',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio',
        'win_rate',
        'profit_factor',
        'avg_win_loss_ratio'
    ]
    
    # Collect metrics
    for method, result in results.items():
        for metric in metrics_to_compare:
            if metric in result:
                comparison.loc[method, metric] = result[metric]
    
    # Print comparison table
    logger.info("Performance Comparison:")
    logger.info("\n" + str(comparison))
    
    # Plot equity curves
    plt.figure(figsize=(12, 8))
    
    for method, result in results.items():
        portfolio_history = result['portfolio_history']
        plt.plot(portfolio_history, label=method)
    
    plt.title('Equity Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_comparison.png')
    
    # Plot weight changes for the adaptive method
    if 'adaptive' in results and results['adaptive']['weights_history']:
        weights = results['adaptive']['weights_history']
        weights_df = pd.DataFrame(weights)
        
        plt.figure(figsize=(12, 6))
        for col in weights_df.columns:
            plt.plot(weights_df[col], label=col)
        
        plt.title('Strategy Weight Changes (Adaptive Method)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('weight_changes.png')
    
    logger.info("Plots saved as 'equity_comparison.png' and 'weight_changes.png'")


def test_with_real_strategies():
    """Test the ensemble with actual strategy implementations"""
    # Generate market data
    market_data = generate_market_data(days=504, symbols=["SPY", "QQQ", "IWM", "GLD", "TLT", "SHY"])
    
    # Create the MacroTrendStrategy
    macro_trend = MacroTrendStrategy(
        symbols=["SPY", "QQQ", "IWM", "GLD", "TLT"],
        trend_method="macd",
        trend_periods={"fast": 12, "slow": 26, "signal": 9},
        ma_periods={"short": 50, "medium": 100, "long": 200},
        volatility_periods=20,
        rebalance_frequency=20,
        min_allocation=0.0,
        max_allocation=0.3
    )
    
    # Create a RegimeDetector
    regime_detector = RegimeDetector()
    
    # Create parameter sets for different regimes
    regime_parameter_sets = {
        MarketRegime.BULL_TREND: {
            "trend_method": "macd",
            "trend_periods": {"fast": 8, "slow": 21, "signal": 9},
            "min_allocation": 0.0,
            "max_allocation": 0.4
        },
        MarketRegime.BEAR_TREND: {
            "trend_method": "adx",
            "trend_periods": {"adx_period": 14, "threshold": 25},
            "min_allocation": 0.0,
            "max_allocation": 0.2
        },
        MarketRegime.HIGH_VOLATILITY: {
            "trend_method": "ma_cross",
            "ma_periods": {"short": 20, "medium": 50, "long": 200},
            "min_allocation": 0.0,
            "max_allocation": 0.15
        },
        MarketRegime.LOW_VOLATILITY: {
            "trend_method": "macd",
            "trend_periods": {"fast": 12, "slow": 26, "signal": 9},
            "min_allocation": 0.0,
            "max_allocation": 0.3
        }
    }
    
    # Create the RegimeAwareStrategy
    regime_aware = RegimeAwareStrategy(
        base_strategy=macro_trend,
        regime_detector=regime_detector,
        regime_parameter_sets=regime_parameter_sets,
        check_regime_frequency=10,
        symbols=["SPY", "QQQ", "IWM", "GLD", "TLT"]
    )
    
    # Create a mean reversion strategy using the MockStrategy
    mean_rev = MockStrategy("Mean Reversion", mean_reversion_signal_generator, 
                           symbols=["SPY", "QQQ", "IWM", "GLD", "TLT"])
    
    # Create the ensemble strategy
    ensemble = StrategyEnsemble(
        strategies=[macro_trend, regime_aware, mean_rev],
        weighting_method=WeightingMethod.ADAPTIVE,
        performance_window=60,
        rebalance_frequency=20,
        correlation_threshold=0.7,
        min_weight=0.1,
        max_weight=0.6
    )
    
    # Backtest the ensemble
    results = backtest_strategy(ensemble, market_data)
    
    # Log key performance metrics
    logger.info(f"Ensemble Strategy Results:")
    logger.info(f"Annualized Return: {results['annualized_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 8))
    plt.plot(results['portfolio_history'])
    plt.title('Ensemble Strategy Equity Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ensemble_equity.png')
    
    # Plot weight changes
    if results['weights_history']:
        weights = results['weights_history']
        weights_df = pd.DataFrame(weights)
        
        plt.figure(figsize=(12, 6))
        for col in weights_df.columns:
            plt.plot(weights_df[col], label=col)
        
        plt.title('Strategy Weight Changes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('ensemble_weights.png')
    
    logger.info("Plots saved as 'ensemble_equity.png' and 'ensemble_weights.png'")


if __name__ == "__main__":
    logger.info("Starting strategy ensemble tests")
    
    # Run basic tests with mock strategies
    test_ensemble_strategy()
    
    # Test with real strategy implementations
    test_with_real_strategies()
    
    logger.info("Tests completed") 