#!/usr/bin/env python3
"""
Multi-Factor Strategy Example

This script demonstrates how to use the MultiFactor trading strategy
with our simulation framework, including parameter optimization,
market regime simulation, and performance analysis.
"""

import os
import sys
import logging
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

from trading_bot.simulation.trading_simulator import TradingSimulator, SimulationConfig, SimulationMode, MarketScenario
from trading_bot.simulation.parameter_optimizer import ParameterOptimizer, OptimizationConfig, ParameterSpace
from trading_bot.simulation.volatility_regime_simulator import VolatilityRegimeSimulator, VolatilityRegime, MarketTrend
from trading_bot.data_providers.yahoo_finance_provider import YahooFinanceProvider
from trading_bot.risk_manager import RiskManager, RiskLevel
from trading_bot.strategies.multi_factor_strategy import MultiFactor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_factor_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rich console for better output
console = Console()

def create_strategy(symbol, data_provider, risk_manager, **params):
    """Factory function to create a multi-factor strategy"""
    return MultiFactor(
        symbol=symbol,
        data_provider=data_provider,
        risk_manager=risk_manager,
        params=params
    )

def create_risk_manager(data_provider, max_position_size=0.05, stop_loss_pct=0.02, **kwargs):
    """Factory function to create a risk manager"""
    return RiskManager(
        multi_asset_adapter=data_provider,
        global_limits={
            "max_portfolio_risk": RiskLevel.MODERATE,
            "max_drawdown_pct": 0.25,
            "max_position_size": max_position_size,
            "stop_loss_pct": stop_loss_pct
        }
    )

def load_config():
    """Load configuration from file or return default"""
    try:
        config_path = os.path.join(script_dir, "config/multi_factor_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading config: {str(e)}, using default configuration")
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "initial_capital": 100000,
            "start_date": (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "data_frequency": "1d",
            "market_scenario": "NORMAL",
            "parameter_optimization": True,
            "test_multi_regime": True,  # Test strategy across different market regimes
            "regimes_to_test": ["NORMAL", "HIGH_VOLATILITY", "LOW_VOLATILITY", "TRENDING_UP", "TRENDING_DOWN"],
            "optimization_metric": "sharpe_ratio"
        }

def run_parameter_optimization(data_provider, config):
    """Run parameter optimization for the multi-factor strategy"""
    console.print(Panel("[bold blue]Running Multi-Factor Strategy Parameter Optimization[/bold blue]"))
    
    # Create base simulation config
    start_date = datetime.datetime.strptime(config["start_date"], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(config["end_date"], "%Y-%m-%d")
    
    base_config = SimulationConfig(
        mode=SimulationMode.BACKTEST,
        start_date=start_date,
        end_date=end_date,
        initial_capital=config["initial_capital"],
        symbols=config["symbols"],
        market_scenario=MarketScenario.NORMAL,
        data_frequency=config["data_frequency"],
        slippage_model="percentage",
        slippage_value=0.1,
        commission_model="fixed",
        commission_value=1.0
    )
    
    # Define parameter spaces
    # Multi-factor strategy has many parameters - focusing on key ones for optimization
    param_spaces = [
        # Moving average parameters
        ParameterSpace(
            name="strategy_fast_ma_period",
            values=[10, 15, 20, 25],
            description="Fast moving average period"
        ),
        ParameterSpace(
            name="strategy_slow_ma_period",
            values=[40, 50, 60, 70],
            description="Slow moving average period"
        ),
        # RSI parameters
        ParameterSpace(
            name="strategy_rsi_period",
            values=[9, 14, 21],
            description="RSI period"
        ),
        ParameterSpace(
            name="strategy_rsi_overbought",
            values=[70, 75, 80],
            description="RSI overbought threshold"
        ),
        ParameterSpace(
            name="strategy_rsi_oversold",
            values=[20, 25, 30],
            description="RSI oversold threshold"
        ),
        # Signal weighting
        ParameterSpace(
            name="strategy_ma_signal_weight",
            values=[0.2, 0.25, 0.3],
            description="Moving average signal weight"
        ),
        ParameterSpace(
            name="strategy_rsi_signal_weight",
            values=[0.15, 0.2, 0.25],
            description="RSI signal weight"
        ),
        ParameterSpace(
            name="strategy_macd_signal_weight",
            values=[0.2, 0.25, 0.3],
            description="MACD signal weight"
        ),
        # Risk parameters
        ParameterSpace(
            name="strategy_minimum_conviction",
            values=[0.5, 0.6, 0.7],
            description="Minimum conviction threshold for entry"
        ),
        ParameterSpace(
            name="strategy_stop_loss_atr_mult",
            values=[1.5, 2.0, 2.5],
            description="Stop loss ATR multiplier"
        ),
        # Position sizing
        ParameterSpace(
            name="strategy_base_position_size",
            values=[0.01, 0.02, 0.03],
            description="Base position size as % of portfolio"
        ),
        ParameterSpace(
            name="risk_max_position_size",
            values=[0.03, 0.05, 0.07],
            description="Maximum position size as % of portfolio"
        )
    ]
    
    # Create optimization config
    opt_config = OptimizationConfig(
        parameter_spaces=param_spaces,
        target_metric=config.get("optimization_metric", "sharpe_ratio"),
        higher_is_better=True,
        max_parallel_jobs=4,
        save_all_results=True,
        output_dir="optimization_results/multi_factor",
        resample_method="timeseries",
        resample_count=3
    )
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        base_simulation_config=base_config,
        optimization_config=opt_config,
        data_provider=data_provider,
        strategy_factory=create_strategy,
        risk_manager_factory=create_risk_manager
    )
    
    # Run optimization
    with Progress() as progress:
        task = progress.add_task("[green]Optimizing Multi-Factor Strategy parameters...", total=None)
        
        try:
            results = optimizer.run_grid_search()
            progress.update(task, completed=100)
            
            # Display results
            table = Table(title="Multi-Factor Strategy Optimization Results")
            table.add_column("Parameter", style="cyan")
            table.add_column("Optimal Value", style="green")
            
            for param, value in results.best_parameters.items():
                table.add_row(param, str(value))
                
            table.add_row("Best Score", f"{results.best_score:.4f}")
            console.print(table)
            
            return results.best_parameters
            
        except Exception as e:
            progress.update(task, completed=100)
            logger.error(f"Parameter optimization failed: {str(e)}")
            console.print(f"[bold red]Optimization failed:[/bold red] {str(e)}")
            
            # Return default parameters
            return {
                "strategy_fast_ma_period": 20,
                "strategy_slow_ma_period": 50,
                "strategy_rsi_period": 14,
                "strategy_rsi_overbought": 70,
                "strategy_rsi_oversold": 30,
                "strategy_ma_signal_weight": 0.25,
                "strategy_rsi_signal_weight": 0.20,
                "strategy_macd_signal_weight": 0.25,
                "strategy_minimum_conviction": 0.6,
                "strategy_stop_loss_atr_mult": 2.0,
                "strategy_base_position_size": 0.02,
                "risk_max_position_size": 0.05
            }

def run_simulation(data_provider, params, config, market_scenario=MarketScenario.NORMAL):
    """Run simulation with the multi-factor strategy"""
    scenario_name = market_scenario.name if isinstance(market_scenario, MarketScenario) else market_scenario
    console.print(Panel(f"[bold green]Running Multi-Factor Strategy Simulation - {scenario_name}[/bold green]"))
    
    # Convert string to enum if needed
    if isinstance(market_scenario, str):
        try:
            market_scenario = getattr(MarketScenario, market_scenario)
        except AttributeError:
            logger.warning(f"Invalid market scenario: {market_scenario}, using NORMAL")
            market_scenario = MarketScenario.NORMAL
    
    # Create simulation config
    start_date = datetime.datetime.strptime(config["start_date"], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(config["end_date"], "%Y-%m-%d")
    
    sim_config = SimulationConfig(
        mode=SimulationMode.BACKTEST,
        start_date=start_date,
        end_date=end_date,
        initial_capital=config["initial_capital"],
        symbols=config["symbols"],
        market_scenario=market_scenario,
        data_frequency=config["data_frequency"],
        slippage_model="percentage",
        slippage_value=0.1,
        commission_model="fixed",
        commission_value=1.0
    )
    
    # Create risk manager with parameters
    risk_params = {k: v for k, v in params.items() if k.startswith('risk_')}
    risk_manager = create_risk_manager(data_provider, **risk_params)
    
    # Create simulator
    simulator = TradingSimulator(
        config=sim_config,
        data_provider=data_provider,
        risk_manager=risk_manager,
        strategy_factory=lambda symbol, data_provider, risk_manager: create_strategy(
            symbol=symbol,
            data_provider=data_provider,
            risk_manager=risk_manager,
            **{k.replace('strategy_', ''): v for k, v in params.items() if k.startswith('strategy_')}
        ),
        output_dir=f"simulation_results/multi_factor/{scenario_name.lower()}"
    )
    
    # Run simulation with progress reporting
    with Progress() as progress:
        task = progress.add_task("[green]Running simulation...", total=100)
        
        # Update progress periodically
        progress.update(task, completed=25)
        
        # Run the simulation
        results = simulator.run_simulation()
        
        progress.update(task, completed=100)
    
    # Display performance metrics
    metrics = results["performance_metrics"]
    
    table = Table(title=f"Multi-Factor Strategy Performance - {scenario_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Return", f"{metrics['total_return']:.2%}")
    table.add_row("Annualized Return", f"{metrics['annualized_return']:.2%}")
    table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    table.add_row("Win Rate", f"{metrics['win_rate']:.2%}")
    table.add_row("Total Trades", str(metrics['total_trades']))
    
    console.print(table)
    
    # Generate visualization
    save_path = os.path.join(f"simulation_results/multi_factor/{scenario_name.lower()}", 
                           f"performance_{results['simulation_id']}.png")
    simulator.plot_portfolio_performance(save_path=save_path)
    console.print(f"Performance chart saved to: {save_path}")
    
    return results

def generate_regime_comparison_chart(results_by_regime):
    """Generate a comparison chart of strategy performance across different market regimes"""
    if not results_by_regime:
        return
        
    # Extract key metrics
    regimes = list(results_by_regime.keys())
    total_returns = [results_by_regime[r]["performance_metrics"]["total_return"] * 100 for r in regimes]
    sharpe_ratios = [results_by_regime[r]["performance_metrics"]["sharpe_ratio"] for r in regimes]
    max_drawdowns = [results_by_regime[r]["performance_metrics"]["max_drawdown"] * 100 for r in regimes]
    win_rates = [results_by_regime[r]["performance_metrics"]["win_rate"] * 100 for r in regimes]
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Factor Strategy Performance Across Market Regimes', fontsize=16)
    
    # Plot total returns
    axs[0, 0].bar(regimes, total_returns, color='green')
    axs[0, 0].set_title('Total Return (%)')
    axs[0, 0].set_ylabel('Percent (%)')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot Sharpe ratios
    axs[0, 1].bar(regimes, sharpe_ratios, color='blue')
    axs[0, 1].set_title('Sharpe Ratio')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot max drawdowns
    axs[1, 0].bar(regimes, max_drawdowns, color='red')
    axs[1, 0].set_title('Maximum Drawdown (%)')
    axs[1, 0].set_ylabel('Percent (%)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot win rates
    axs[1, 1].bar(regimes, win_rates, color='purple')
    axs[1, 1].set_title('Win Rate (%)')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Rotate x-labels for better readability
    for ax in axs.flat:
        ax.set_xticklabels(regimes, rotation=45)
    
    plt.tight_layout()
    save_path = "simulation_results/multi_factor/regime_comparison.png"
    plt.savefig(save_path)
    plt.close()
    
    console.print(f"Regime comparison chart saved to: {save_path}")

def test_volatility_simulator():
    """Run a test of the volatility regime simulator"""
    console.print(Panel("[bold magenta]Testing Volatility Regime Simulator[/bold magenta]"))
    
    # Create a simulator with a few assets
    assets = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
    simulator = VolatilityRegimeSimulator(assets=assets)
    
    # Generate data for different regimes
    regimes_to_test = {
        "Low Volatility": (VolatilityRegime.LOW, MarketTrend.SIDEWAYS),
        "Normal Bull Market": (VolatilityRegime.NORMAL, MarketTrend.BULLISH),
        "High Volatility Bear": (VolatilityRegime.HIGH, MarketTrend.BEARISH),
        "Extreme Volatility": (VolatilityRegime.EXTREME, MarketTrend.SIDEWAYS)
    }
    
    all_data = {}
    
    for regime_name, (vol_regime, trend) in regimes_to_test.items():
        console.print(f"Generating {regime_name} data...")
        
        # Set regime
        simulator.set_current_state(vol_regime, trend)
        
        # Generate data (100 periods)
        data = simulator.generate_data(periods=100, with_regime_changes=False)
        all_data[regime_name] = data
        
        # Plot first asset
        first_asset = assets[0]
        plt.figure(figsize=(10, 6))
        plt.plot(data[first_asset].index, data[first_asset]['close'])
        plt.title(f"{first_asset} Price in {regime_name} Regime")
        plt.grid(True)
        save_path = f"simulation_results/multi_factor/volatility_test_{regime_name.replace(' ', '_').lower()}.png"
        plt.savefig(save_path)
        plt.close()
        
        console.print(f"Plot saved to: {save_path}")
    
    console.print("[bold green]Volatility simulation testing complete![/bold green]")
    return all_data

def main():
    """Main function to run the multi-factor strategy example"""
    console.print(Panel.fit(
        "[bold cyan]Multi-Factor Trading Strategy Example[/bold cyan]\n\n"
        "This example demonstrates the multi-factor trading strategy with:\n"
        "- Parameter optimization\n"
        "- Testing across different market regimes\n"
        "- Performance analysis and visualization",
        title="Trading Bot"
    ))
    
    try:
        # Create output directories
        os.makedirs("simulation_results/multi_factor", exist_ok=True)
        os.makedirs("optimization_results/multi_factor", exist_ok=True)
        
        # Load configuration
        config = load_config()
        console.print(f"Loaded configuration for symbols: {', '.join(config['symbols'])}")
        
        # Initialize data provider
        data_provider = YahooFinanceProvider()
        console.print(f"Using data provider: {data_provider.__class__.__name__}")
        
        # Test volatility simulator if requested
        if config.get("test_volatility_simulator", False):
            vol_sim_data = test_volatility_simulator()
        
        # Run parameter optimization if enabled
        params = {}
        if config.get("parameter_optimization", True):
            params = run_parameter_optimization(data_provider, config)
        else:
            # Use default parameters
            params = {
                "strategy_fast_ma_period": 20,
                "strategy_slow_ma_period": 50,
                "strategy_rsi_period": 14,
                "strategy_rsi_overbought": 70,
                "strategy_rsi_oversold": 30,
                "strategy_ma_signal_weight": 0.25,
                "strategy_rsi_signal_weight": 0.20,
                "strategy_macd_signal_weight": 0.25,
                "strategy_minimum_conviction": 0.6,
                "strategy_stop_loss_atr_mult": 2.0,
                "strategy_base_position_size": 0.02,
                "risk_max_position_size": 0.05
            }
        
        # Run simulations across different market regimes
        results_by_regime = {}
        
        if config.get("test_multi_regime", True):
            for regime in config.get("regimes_to_test", ["NORMAL"]):
                try:
                    results = run_simulation(data_provider, params, config, market_scenario=regime)
                    results_by_regime[regime] = results
                except Exception as e:
                    logger.error(f"Error running simulation for {regime} regime: {str(e)}")
                    console.print(f"[bold red]Error in {regime} regime simulation:[/bold red] {str(e)}")
        else:
            # Run just one simulation with NORMAL regime
            results = run_simulation(data_provider, params, config)
            results_by_regime["NORMAL"] = results
        
        # Generate regime comparison chart if multiple regimes were tested
        if len(results_by_regime) > 1:
            generate_regime_comparison_chart(results_by_regime)
        
        console.print("[bold green]Multi-Factor Strategy Example completed successfully![/bold green]")
        
    except Exception as e:
        logger.error(f"Error in Multi-Factor Strategy Example: {str(e)}")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 