#!/usr/bin/env python3
"""
Enhanced Paper Trading Simulation Example

This script demonstrates an enhanced paper trading simulation
with parameter optimization, error handling, volatility regime simulation,
and advanced visualization.
"""

import os
import sys
import logging
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from trading_bot.data_providers.yahoo_finance_provider import YahooFinanceProvider
from trading_bot.data_providers.alpha_vantage_provider import AlphaVantageProvider
from trading_bot.ml.market_anomaly_detector import MarketAnomalyDetector
from trading_bot.risk_manager import RiskManager, RiskLevel
from trading_bot.strategies.moving_average_strategy import MovingAverageStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rich console for better output
console = Console()

class SimulationError(Exception):
    """Custom exception for simulation errors"""
    pass

def create_strategy(symbol, data_provider, risk_manager, fast_ma=20, slow_ma=50, **kwargs):
    """Factory function to create a trading strategy"""
    try:
        strategy = MovingAverageStrategy(
            symbol=symbol,
            data_provider=data_provider,
            risk_manager=risk_manager,
            fast_ma_period=fast_ma,
            slow_ma_period=slow_ma
        )
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy for {symbol}: {str(e)}")
        raise SimulationError(f"Strategy creation failed: {str(e)}")

def create_risk_manager(data_provider, max_position_size=0.05, stop_loss_pct=0.02, **kwargs):
    """Factory function to create a risk manager"""
    try:
        risk_manager = RiskManager(
            multi_asset_adapter=data_provider,
            global_limits={
                "max_portfolio_risk": RiskLevel.MODERATE,
                "max_drawdown_pct": 0.25,
                "max_position_size": max_position_size,
                "stop_loss_pct": stop_loss_pct
            }
        )
        return risk_manager
    except Exception as e:
        logger.error(f"Error creating risk manager: {str(e)}")
        raise SimulationError(f"Risk manager creation failed: {str(e)}")

def load_config():
    """Load configuration from file"""
    try:
        config_path = os.path.join(script_dir, "config/simulation_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading config: {str(e)}, using default configuration")
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "initial_capital": 100000,
            "start_date": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "data_frequency": "1d",
            "market_scenario": "NORMAL",
            "parameter_optimization": True,
            "visualization": "plotly",  # Options: matplotlib, plotly
            "enable_anomaly_detection": True
        }

def setup_data_providers(config):
    """Set up and initialize data providers with failover"""
    providers = []
    
    # Primary provider
    try:
        yahoo = YahooFinanceProvider()
        providers.append(yahoo)
        logger.info("Initialized Yahoo Finance provider")
    except Exception as e:
        logger.error(f"Error initializing Yahoo Finance provider: {str(e)}")
    
    # Secondary provider for failover
    try:
        # Note: requires API key in environment
        alpha_vantage = AlphaVantageProvider()
        providers.append(alpha_vantage)
        logger.info("Initialized Alpha Vantage provider")
    except Exception as e:
        logger.warning(f"Error initializing Alpha Vantage provider: {str(e)}")
    
    if not providers:
        raise SimulationError("Failed to initialize any data providers")
    
    # Use the first available provider
    return providers[0]

def setup_anomaly_detector(symbol, data_provider, config):
    """Set up market anomaly detector"""
    if not config.get("enable_anomaly_detection", False):
        return None
        
    try:
        detector = MarketAnomalyDetector(
            symbol=symbol,
            lookback_window=20,
            alert_threshold=0.95,
            model_dir="models/anomaly_detectors"
        )
        
        # Load model or train if not available
        if not detector.load_model():
            logger.info(f"Training new anomaly detection model for {symbol}")
            
            # Get historical data for training
            start_date = datetime.datetime.now() - datetime.timedelta(days=365)
            end_date = datetime.datetime.now()
            
            try:
                training_data = data_provider.get_historical_data(
                    symbol=symbol, 
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )
                
                if len(training_data) > 50:  # Ensure enough data for training
                    detector.train_model(training_data)
                    detector.save_model()
                else:
                    logger.warning(f"Insufficient training data for {symbol} anomaly detector")
                    return None
            except Exception as e:
                logger.error(f"Error fetching training data for anomaly detector: {str(e)}")
                return None
                
        return detector
    except Exception as e:
        logger.error(f"Error setting up anomaly detector: {str(e)}")
        return None

def run_parameter_optimization(data_provider, config):
    """Run parameter optimization to find optimal strategy parameters"""
    console.print(Panel("[bold blue]Running Parameter Optimization[/bold blue]"))
    
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
    param_spaces = [
        ParameterSpace(
            name="strategy_fast_ma",
            values=[5, 10, 15, 20, 25, 30],
            description="Fast moving average period"
        ),
        ParameterSpace(
            name="strategy_slow_ma",
            values=[30, 40, 50, 60, 70, 80],
            description="Slow moving average period"
        ),
        ParameterSpace(
            name="risk_max_position_size",
            values=[0.02, 0.05, 0.1, 0.15],
            description="Maximum position size as fraction of portfolio"
        ),
        ParameterSpace(
            name="risk_stop_loss_pct",
            values=[0.01, 0.02, 0.03, 0.05],
            description="Stop loss percentage"
        )
    ]
    
    # Create optimization config
    opt_config = OptimizationConfig(
        parameter_spaces=param_spaces,
        target_metric="sharpe_ratio",
        higher_is_better=True,
        max_parallel_jobs=4,
        save_all_results=True,
        output_dir="optimization_results",
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
        task = progress.add_task("[green]Optimizing parameters...", total=None)
        
        try:
            results = optimizer.run_grid_search()
            progress.update(task, completed=100)
            
            # Display results
            table = Table(title="Optimization Results")
            table.add_column("Parameter", style="cyan")
            table.add_column("Optimal Value", style="green")
            
            for param, value in results.best_parameters.items():
                table.add_row(param, str(value))
                
            table.add_row("Best Score (Sharpe)", f"{results.best_score:.4f}")
            console.print(table)
            
            return results.best_parameters
            
        except Exception as e:
            progress.update(task, completed=100)
            logger.error(f"Parameter optimization failed: {str(e)}")
            console.print(f"[bold red]Optimization failed:[/bold red] {str(e)}")
            
            # Return default parameters
            return {
                "strategy_fast_ma": 20,
                "strategy_slow_ma": 50,
                "risk_max_position_size": 0.05,
                "risk_stop_loss_pct": 0.02
            }

def run_simulation(data_provider, params, config):
    """Run the trading simulation with the provided parameters"""
    console.print(Panel(f"[bold green]Running Trading Simulation[/bold green]"))
    
    try:
        # Extract parameters
        fast_ma = params.get("strategy_fast_ma", 20)
        slow_ma = params.get("strategy_slow_ma", 50)
        max_position_size = params.get("risk_max_position_size", 0.05)
        stop_loss_pct = params.get("risk_stop_loss_pct", 0.02)
        
        # Create risk manager
        risk_manager = create_risk_manager(
            data_provider=data_provider,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct
        )
        
        # Create simulation config
        start_date = datetime.datetime.strptime(config["start_date"], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(config["end_date"], "%Y-%m-%d")
        
        # Map string scenario to enum
        scenario_map = {
            "NORMAL": MarketScenario.NORMAL,
            "HIGH_VOLATILITY": MarketScenario.HIGH_VOLATILITY,
            "LOW_LIQUIDITY": MarketScenario.LOW_LIQUIDITY,
            "FLASH_CRASH": MarketScenario.FLASH_CRASH,
            "SIDEWAYS": MarketScenario.SIDEWAYS
        }
        market_scenario = scenario_map.get(config["market_scenario"], MarketScenario.NORMAL)
        
        sim_config = SimulationConfig(
            mode=SimulationMode.PAPER_TRADING,
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
        
        # Create simulator
        simulator = TradingSimulator(
            config=sim_config,
            data_provider=data_provider,
            risk_manager=risk_manager,
            strategy_factory=lambda symbol, data_provider, risk_manager: create_strategy(
                symbol=symbol,
                data_provider=data_provider,
                risk_manager=risk_manager,
                fast_ma=fast_ma,
                slow_ma=slow_ma
            ),
            output_dir="simulation_results"
        )
        
        # Setup anomaly detectors for each symbol
        anomaly_detectors = {}
        for symbol in config["symbols"]:
            detector = setup_anomaly_detector(symbol, data_provider, config)
            if detector:
                anomaly_detectors[symbol] = detector
        
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
        
        table = Table(title="Simulation Performance Metrics")
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
        if config.get("visualization") == "plotly":
            create_plotly_dashboard(results, anomaly_detectors)
        else:
            # Default to matplotlib
            save_path = os.path.join("simulation_results", f"performance_{results['simulation_id']}.png")
            simulator.plot_portfolio_performance(save_path=save_path)
            console.print(f"Performance chart saved to: {save_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        console.print(f"[bold red]Simulation failed:[/bold red] {str(e)}")
        raise SimulationError(f"Simulation failed: {str(e)}")

def create_plotly_dashboard(results, anomaly_detectors=None):
    """Create an interactive Plotly dashboard with simulation results"""
    try:
        # Convert portfolio history to DataFrame
        portfolio_history = pd.DataFrame(results["portfolio_history"])
        portfolio_history.set_index("timestamp", inplace=True)
        
        # Convert trades to DataFrame
        trades = pd.DataFrame(results["trades"]) if results["trades"] else pd.DataFrame()
        
        # Calculate additional metrics
        if not portfolio_history.empty:
            portfolio_history["daily_return"] = portfolio_history["total_value"].pct_change().fillna(0)
            portfolio_history["cumulative_return"] = (portfolio_history["total_value"] / portfolio_history["total_value"].iloc[0]) - 1
            
            # Calculate drawdowns
            portfolio_history["peak"] = portfolio_history["total_value"].cummax()
            portfolio_history["drawdown"] = (portfolio_history["total_value"] - portfolio_history["peak"]) / portfolio_history["peak"]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Portfolio Value", "Drawdown", "Daily Returns"),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Add portfolio value trace
        fig.add_trace(
            go.Scatter(
                x=portfolio_history.index,
                y=portfolio_history["total_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        
        # Add buy/sell markers if trades exist
        if not trades.empty:
            # Buy trades
            buy_trades = trades[trades["action"] == "BUY"]
            if not buy_trades.empty:
                # Map trade timestamps to portfolio_history timestamps
                buy_times = []
                buy_values = []
                
                for _, trade in buy_trades.iterrows():
                    trade_time = trade["time"]
                    # Find closest time in portfolio_history
                    if trade_time in portfolio_history.index:
                        buy_times.append(trade_time)
                        buy_values.append(portfolio_history.loc[trade_time, "total_value"])
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_values,
                        mode="markers",
                        name="Buy",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color="green"
                        ),
                        hoverinfo="text",
                        text=[f"Buy: {trades.loc[trades['time'] == t, 'symbol'].iloc[0]}" for t in buy_times]
                    ),
                    row=1, col=1
                )
            
            # Sell trades
            sell_trades = trades[trades["action"] == "SELL"]
            if not sell_trades.empty:
                # Map trade timestamps to portfolio_history timestamps
                sell_times = []
                sell_values = []
                
                for _, trade in sell_trades.iterrows():
                    trade_time = trade["time"]
                    # Find closest time in portfolio_history
                    if trade_time in portfolio_history.index:
                        sell_times.append(trade_time)
                        sell_values.append(portfolio_history.loc[trade_time, "total_value"])
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_values,
                        mode="markers",
                        name="Sell",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color="red"
                        ),
                        hoverinfo="text",
                        text=[f"Sell: {trades.loc[trades['time'] == t, 'symbol'].iloc[0]}" for t in sell_times]
                    ),
                    row=1, col=1
                )
        
        # Add drawdown trace
        fig.add_trace(
            go.Scatter(
                x=portfolio_history.index,
                y=portfolio_history["drawdown"],
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red", width=1)
            ),
            row=2, col=1
        )
        
        # Add daily returns trace
        colors = ["green" if r >= 0 else "red" for r in portfolio_history["daily_return"]]
        fig.add_trace(
            go.Bar(
                x=portfolio_history.index,
                y=portfolio_history["daily_return"],
                name="Daily Returns",
                marker_color=colors
            ),
            row=3, col=1
        )
        
        # Add anomaly markers if detectors were used
        if anomaly_detectors:
            anomaly_times = []
            anomaly_values = []
            anomaly_texts = []
            
            for symbol, detector in anomaly_detectors.items():
                if hasattr(detector, 'anomalies') and detector.anomalies:
                    for anomaly_time, anomaly_score in detector.anomalies.items():
                        if anomaly_time in portfolio_history.index:
                            anomaly_times.append(anomaly_time)
                            anomaly_values.append(portfolio_history.loc[anomaly_time, "total_value"])
                            anomaly_texts.append(f"Anomaly in {symbol}: {anomaly_score:.2f}")
            
            if anomaly_times:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_times,
                        y=anomaly_values,
                        mode="markers",
                        name="Market Anomalies",
                        marker=dict(
                            symbol="x",
                            size=12,
                            color="purple",
                            line=dict(width=1, color="black")
                        ),
                        hoverinfo="text",
                        text=anomaly_texts
                    ),
                    row=1, col=1
                )
        
        # Add performance metrics as annotations
        metrics = results["performance_metrics"]
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}<br>"
            f"Annualized: {metrics['annualized_return']:.2%}<br>"
            f"Sharpe: {metrics['sharpe_ratio']:.2f}<br>"
            f"Max DD: {metrics['max_drawdown']:.2%}<br>"
            f"Win Rate: {metrics['win_rate']:.2%}<br>"
            f"Trades: {metrics['total_trades']}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            text=metrics_text,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        # Update layout
        fig.update_layout(
            title=f"Simulation Results (ID: {results['simulation_id']})",
            height=800,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)
        
        # Save figure
        output_path = os.path.join("simulation_results", f"dashboard_{results['simulation_id']}.html")
        fig.write_html(output_path)
        console.print(f"Interactive dashboard saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating Plotly dashboard: {str(e)}")
        console.print(f"[bold yellow]Warning:[/bold yellow] Failed to create interactive dashboard: {str(e)}")

def main():
    """Main function to run the enhanced paper trading simulation"""
    console.print(Panel.fit(
        "[bold cyan]Enhanced Paper Trading Simulation[/bold cyan]\n\n"
        "This example demonstrates paper trading with parameter optimization, "
        "error handling, volatility regime simulation, and advanced visualization.",
        title="Trading Bot"
    ))
    
    try:
        # Load configuration
        config = load_config()
        console.print(f"Loaded configuration for symbols: {', '.join(config['symbols'])}")
        
        # Set up data providers
        data_provider = setup_data_providers(config)
        console.print(f"Using data provider: {data_provider.__class__.__name__}")
        
        # Run parameter optimization if enabled
        params = {}
        if config.get("parameter_optimization", False):
            params = run_parameter_optimization(data_provider, config)
        else:
            # Use default parameters
            params = {
                "strategy_fast_ma": 20,
                "strategy_slow_ma": 50,
                "risk_max_position_size": 0.05,
                "risk_stop_loss_pct": 0.02
            }
        
        # Run simulation
        results = run_simulation(data_provider, params, config)
        
        console.print("[bold green]Simulation completed successfully![/bold green]")
        
    except SimulationError as e:
        console.print(f"[bold red]Simulation Error:[/bold red] {str(e)}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}")
        logger.exception("Unexpected error in main function")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 