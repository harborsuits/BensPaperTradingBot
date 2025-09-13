import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.strategies.strategy_ensemble import StrategyEnsemble, WeightingMethod, DynamicEnsemble
from trading_bot.strategies.strategy_template import StrategyTemplate as Strategy
from trading_bot.strategies.macro_trend_strategy import MacroTrendStrategy
from trading_bot.strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime, RegimeDetector
from trading_bot.backtesting.performance_metrics import calculate_comprehensive_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple mean reversion strategy
class MeanReversionStrategy(Strategy):
    def __init__(
        self,
        symbols: list,
        lookback_period: int = 20,
        z_score_threshold: float = 1.5,
        name: str = "Mean Reversion"
    ):
        """
        Initialize mean reversion strategy
        
        Args:
            symbols: List of symbols to trade
            lookback_period: Period for calculating mean and std dev
            z_score_threshold: Z-score threshold for entry/exit
            name: Strategy name
        """
        super().__init__(symbols=symbols)
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.name = name
        
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on mean reversion principle
        
        Args:
            market_data: Market data with symbols as columns
            
        Returns:
            DataFrame with signals for each symbol
        """
        signals = pd.DataFrame(index=[market_data.index[-1]], columns=self.symbols)
        
        for symbol in self.symbols:
            if symbol in market_data.columns and len(market_data) > self.lookback_period:
                # Calculate mean and std dev
                price_series = market_data[symbol]
                mean = price_series.rolling(window=self.lookback_period).mean()
                std = price_series.rolling(window=self.lookback_period).std()
                
                # Calculate z-score
                current_price = price_series.iloc[-1]
                z_score = (current_price - mean.iloc[-1]) / std.iloc[-1]
                
                # Generate signal based on z-score
                if z_score < -self.z_score_threshold:
                    # Price is below expected range, buy signal
                    signals[symbol] = 1.0
                elif z_score > self.z_score_threshold:
                    # Price is above expected range, sell signal
                    signals[symbol] = -1.0
                else:
                    # Price is within expected range, neutral
                    signals[symbol] = 0.0
            else:
                signals[symbol] = 0.0
                
        return signals
    
    def calculate_position_size(self, signal: float, symbol: str, account_size: float) -> float:
        """
        Calculate position size based on signal strength
        
        Args:
            signal: Signal value (-1 to 1)
            symbol: Symbol to trade
            account_size: Current account size
            
        Returns:
            Position size in units of account currency
        """
        # Use a simple position sizing of 10% of account per full signal
        position = account_size * 0.1 * abs(signal)
        
        # Return signed position
        return position if signal > 0 else -position


# Simple momentum strategy
class MomentumStrategy(Strategy):
    def __init__(
        self,
        symbols: list,
        momentum_period: int = 90,
        smoothing_period: int = 20,
        name: str = "Momentum"
    ):
        """
        Initialize momentum strategy
        
        Args:
            symbols: List of symbols to trade
            momentum_period: Period for momentum calculation
            smoothing_period: Period for smoothing momentum
            name: Strategy name
        """
        super().__init__(symbols=symbols)
        self.momentum_period = momentum_period
        self.smoothing_period = smoothing_period
        self.name = name
        
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on price momentum
        
        Args:
            market_data: Market data with symbols as columns
            
        Returns:
            DataFrame with signals for each symbol
        """
        signals = pd.DataFrame(index=[market_data.index[-1]], columns=self.symbols)
        
        for symbol in self.symbols:
            if symbol in market_data.columns and len(market_data) > self.momentum_period:
                # Calculate momentum (price change over period)
                price_series = market_data[symbol]
                momentum = price_series / price_series.shift(self.momentum_period) - 1
                
                # Smooth momentum
                if len(momentum) > self.smoothing_period:
                    smoothed_momentum = momentum.rolling(window=self.smoothing_period).mean()
                    
                    # Generate signal based on momentum
                    current_momentum = smoothed_momentum.iloc[-1]
                    
                    if current_momentum > 0.05:  # Strong positive momentum
                        signals[symbol] = 1.0
                    elif current_momentum < -0.05:  # Strong negative momentum
                        signals[symbol] = -1.0
                    elif current_momentum > 0:  # Weak positive momentum
                        signals[symbol] = 0.5
                    elif current_momentum < 0:  # Weak negative momentum
                        signals[symbol] = -0.5
                    else:
                        signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.0
            else:
                signals[symbol] = 0.0
                
        return signals
    
    def calculate_position_size(self, signal: float, symbol: str, account_size: float) -> float:
        """
        Calculate position size based on signal strength
        
        Args:
            signal: Signal value (-1 to 1)
            symbol: Symbol to trade
            account_size: Current account size
            
        Returns:
            Position size in units of account currency
        """
        # Use a simple position sizing of 15% of account per full signal
        position = account_size * 0.15 * abs(signal)
        
        # Return signed position
        return position if signal > 0 else -position


def load_or_generate_data(symbols, start_date=None, end_date=None, generate=False):
    """
    Load historical data or generate synthetic data for testing
    
    Args:
        symbols: List of symbols
        start_date: Start date for data
        end_date: End date for data
        generate: Whether to generate synthetic data
        
    Returns:
        DataFrame with market data
    """
    if generate:
        # Generate synthetic data
        if start_date is None:
            start_date = datetime.now() - timedelta(days=1000)
        if end_date is None:
            end_date = datetime.now()
        
        days = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        data = pd.DataFrame(index=pd.DatetimeIndex(dates))
        
        # Generate prices for each symbol with different characteristics
        for symbol in symbols:
            # Base trend and volatility based on symbol
            if symbol == "SPY" or symbol == "VOO":
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
                trend = 0.00025  # ~6% annual return
                vol = 0.011
            
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
    else:
        # Placeholder for loading actual market data
        # In a real implementation, you would load data from a source
        logger.warning("Data loading not implemented. Using synthetic data instead.")
        return load_or_generate_data(symbols, start_date, end_date, generate=True)


def run_backtest(strategy, market_data):
    """
    Run a simple backtest for the strategy
    
    Args:
        strategy: Strategy to test
        market_data: Market data for testing
        
    Returns:
        Dictionary with backtest results
    """
    # Initialize portfolio
    initial_capital = 100000
    portfolio_value = initial_capital
    portfolio_history = [portfolio_value]
    returns_history = [0]
    weights_history = []
    
    # Track positions
    positions = {symbol: 0 for symbol in strategy.symbols}
    
    # Start from 100th day to have enough data for indicators
    start_idx = 100
    
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


def compare_strategies(results):
    """
    Compare backtest results from different strategies
    
    Args:
        results: Dictionary mapping strategy names to backtest results
    """
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
    for name, result in results.items():
        for metric in metrics_to_compare:
            if metric in result:
                comparison.loc[name, metric] = result[metric]
    
    # Print comparison table
    logger.info("Performance Comparison:")
    logger.info("\n" + str(comparison))
    
    # Plot equity curves
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        portfolio_history = result['portfolio_history']
        plt.plot(portfolio_history, label=name)
    
    plt.title('Equity Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_comparison.png')
    plt.show()
    
    # Plot weight changes for the adaptive ensemble
    if 'Adaptive Ensemble' in results and results['Adaptive Ensemble']['weights_history']:
        weights = results['Adaptive Ensemble']['weights_history']
        weights_df = pd.DataFrame(weights)
        
        plt.figure(figsize=(12, 6))
        for col in weights_df.columns:
            plt.plot(weights_df[col], label=col)
        
        plt.title('Strategy Weight Changes (Adaptive Ensemble)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('weight_changes.png')
        plt.show()
    
    logger.info("Plots saved as 'equity_comparison.png' and 'weight_changes.png'")


def main():
    """Run the ensemble example"""
    # Define symbols to trade
    symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SHY"]
    
    # Load or generate market data
    market_data = load_or_generate_data(
        symbols=symbols,
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2023, 12, 31),
        generate=True
    )
    
    logger.info(f"Loaded market data with {len(market_data)} days for {len(symbols)} symbols")
    
    # Create individual strategies
    
    # 1. Macro Trend Strategy
    macro_trend = MacroTrendStrategy(
        symbols=symbols,
        trend_method="macd",
        trend_periods={"fast": 12, "slow": 26, "signal": 9},
        ma_periods={"short": 50, "medium": 100, "long": 200},
        volatility_periods=20,
        rebalance_frequency=20,
        min_allocation=0.0,
        max_allocation=0.3
    )
    
    # 2. Mean Reversion Strategy
    mean_reversion = MeanReversionStrategy(
        symbols=symbols,
        lookback_period=20,
        z_score_threshold=1.5
    )
    
    # 3. Momentum Strategy
    momentum = MomentumStrategy(
        symbols=symbols,
        momentum_period=90,
        smoothing_period=20
    )
    
    # 4. Create a RegimeAwareStrategy
    regime_detector = RegimeDetector()
    
    # Define parameter sets for different regimes
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
    
    regime_aware = RegimeAwareStrategy(
        base_strategy=macro_trend,
        regime_detector=regime_detector,
        regime_parameter_sets=regime_parameter_sets,
        check_regime_frequency=10,
        symbols=symbols
    )
    
    # Create ensemble strategies
    
    # 1. Equal weighted ensemble
    equal_ensemble = StrategyEnsemble(
        strategies=[macro_trend, mean_reversion, momentum, regime_aware],
        weighting_method=WeightingMethod.EQUAL,
        strategy_names=["Macro Trend", "Mean Reversion", "Momentum", "Regime Aware"]
    )
    
    # 2. Performance weighted ensemble
    performance_ensemble = StrategyEnsemble(
        strategies=[macro_trend, mean_reversion, momentum, regime_aware],
        weighting_method=WeightingMethod.PERFORMANCE,
        performance_window=60,
        rebalance_frequency=20,
        strategy_names=["Macro Trend", "Mean Reversion", "Momentum", "Regime Aware"]
    )
    
    # 3. Adaptive ensemble
    adaptive_ensemble = StrategyEnsemble(
        strategies=[macro_trend, mean_reversion, momentum, regime_aware],
        weighting_method=WeightingMethod.ADAPTIVE,
        performance_window=60,
        rebalance_frequency=20,
        correlation_threshold=0.7,
        min_weight=0.1,
        max_weight=0.6,
        strategy_names=["Macro Trend", "Mean Reversion", "Momentum", "Regime Aware"]
    )
    
    # 4. Dynamic ensemble
    dynamic_ensemble = DynamicEnsemble(
        strategies=[macro_trend, mean_reversion, momentum, regime_aware],
        strategy_names=["Macro Trend", "Mean Reversion", "Momentum", "Regime Aware"],
        min_active_strategies=2,
        max_active_strategies=4,
        activation_threshold=0.2,
        deactivation_threshold=-0.1,
        weighting_method=WeightingMethod.PERFORMANCE,
        performance_window=60,
        rebalance_frequency=20
    )
    
    # Run backtests
    results = {}
    
    # Backtest individual strategies
    logger.info("Backtesting Macro Trend Strategy...")
    results["Macro Trend"] = run_backtest(macro_trend, market_data)
    
    logger.info("Backtesting Mean Reversion Strategy...")
    results["Mean Reversion"] = run_backtest(mean_reversion, market_data)
    
    logger.info("Backtesting Momentum Strategy...")
    results["Momentum"] = run_backtest(momentum, market_data)
    
    logger.info("Backtesting Regime Aware Strategy...")
    results["Regime Aware"] = run_backtest(regime_aware, market_data)
    
    # Backtest ensemble strategies
    logger.info("Backtesting Equal Weighted Ensemble...")
    results["Equal Ensemble"] = run_backtest(equal_ensemble, market_data)
    
    logger.info("Backtesting Performance Weighted Ensemble...")
    results["Performance Ensemble"] = run_backtest(performance_ensemble, market_data)
    
    logger.info("Backtesting Adaptive Ensemble...")
    results["Adaptive Ensemble"] = run_backtest(adaptive_ensemble, market_data)
    
    logger.info("Backtesting Dynamic Ensemble...")
    results["Dynamic Ensemble"] = run_backtest(dynamic_ensemble, market_data)
    
    # Compare results
    compare_strategies(results)


if __name__ == "__main__":
    main() 