#!/usr/bin/env python3
"""
ML-Powered Trading Strategy Example

This script demonstrates how to use machine learning components for trading strategy execution.
It showcases market condition classification, price prediction with confidence scores,
parameter optimization for different market regimes, and anomaly detection for risk management.
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Configure rich for better console output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading system components
from trading_bot.multi_asset_adapter import MultiAssetAdapter
from trading_bot.risk_manager import RiskManager, RiskLevel
from trading_bot.ml.ml_integration import MLIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml_trading_strategy")

# Create console for rich output
console = Console()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML-Powered Trading Strategy")
    
    parser.add_argument('--config', type=str, default='../config/config.json',
                        help="Path to configuration file")
    
    parser.add_argument('--assets', type=str, nargs='+', default=['SPY', 'QQQ', 'AAPL'],
                        help="Assets to trade")
    
    parser.add_argument('--backtest', action='store_true',
                        help="Run in backtest mode")
    
    parser.add_argument('--start-date', type=str, default=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
                        help="Start date for backtesting (format: YYYY-MM-DD)")
    
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help="End date for backtesting (format: YYYY-MM-DD)")
    
    parser.add_argument('--timeframe', type=str, default='1d',
                        help="Timeframe for data (e.g., 1h, 1d)")
    
    parser.add_argument('--plot', action='store_true', 
                        help="Plot results")
                        
    parser.add_argument('--force-training', action='store_true',
                        help="Force ML model training")
                        
    parser.add_argument('--risk-level', type=str, choices=['low', 'moderate', 'high', 'extreme'],
                        default='moderate', help="Risk level for trading")
    
    return parser.parse_args()

def display_market_conditions(market_classifier, data, title="Current Market Condition"):
    """Display the current market conditions."""
    condition = market_classifier.predict_condition(data)
    
    if not condition:
        console.print(Panel("[yellow]Market condition analysis unavailable[/yellow]"))
        return
    
    table = Table(title=title)
    table.add_column("Condition", style="cyan")
    table.add_column("Probability", style="magenta")
    table.add_column("Description", style="green")
    
    # Add the primary condition
    condition_desc = {
        "bullish": "Strong uptrend with positive momentum",
        "bearish": "Downtrend with negative momentum",
        "sideways": "Range-bound market with low directional conviction",
        "volatile": "High volatility with unpredictable price action"
    }
    
    table.add_row(
        condition["condition"].upper(),
        f"{condition['probability']:.2f}",
        condition_desc.get(condition["condition"], "Unknown condition")
    )
    
    # Add probability for other conditions if available
    if "other_probabilities" in condition:
        for cond, prob in condition["other_probabilities"].items():
            if cond != condition["condition"]:
                table.add_row(
                    cond.upper(),
                    f"{prob:.2f}",
                    condition_desc.get(cond, "Unknown condition")
                )
    
    console.print(table)

def display_price_predictions(predictions, title="Price Movement Predictions"):
    """Display price movement predictions with confidence scores."""
    if not predictions:
        console.print(Panel("[yellow]Price predictions unavailable[/yellow]"))
        return
    
    table = Table(title=title)
    table.add_column("Asset", style="cyan")
    table.add_column("Direction", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Signal", style="yellow")
    
    for asset, prediction in predictions.items():
        direction = prediction.get("direction", "neutral")
        confidence = prediction.get("confidence", 0)
        
        direction_str = direction.upper()
        direction_color = "green" if direction == "up" else "red" if direction == "down" else "yellow"
        
        # Determine signal based on confidence
        signal = "HOLD"
        if confidence >= 0.8:
            signal = "STRONG BUY" if direction == "up" else "STRONG SELL" if direction == "down" else "HOLD"
        elif confidence >= 0.65:
            signal = "BUY" if direction == "up" else "SELL" if direction == "down" else "HOLD"
        
        signal_color = "green" if "BUY" in signal else "red" if "SELL" in signal else "yellow"
        
        table.add_row(
            asset,
            f"[{direction_color}]{direction_str}[/{direction_color}]",
            f"{confidence:.2f}",
            f"[{signal_color}]{signal}[/{signal_color}]"
        )
    
    console.print(table)

def display_optimized_parameters(parameters, condition, title="Optimized Strategy Parameters"):
    """Display optimized parameters for different strategies."""
    if not parameters:
        console.print(Panel(f"[yellow]No optimized parameters available for {condition} market[/yellow]"))
        return
    
    table = Table(title=f"{title} for {condition.upper()} Market")
    table.add_column("Strategy", style="cyan")
    table.add_column("Parameter", style="magenta")
    table.add_column("Value", style="green")
    
    for strategy, params in parameters.items():
        for i, (param, value) in enumerate(params.items()):
            # Only add strategy name on the first row for each strategy
            if i == 0:
                table.add_row(strategy, param, f"{value}")
            else:
                table.add_row("", param, f"{value}")
    
    console.print(table)

def display_anomaly_status(anomaly_status, title="Market Anomaly Detection"):
    """Display market anomaly detection status."""
    if not anomaly_status:
        console.print(Panel("[yellow]Anomaly detection unavailable[/yellow]"))
        return
    
    is_anomaly = anomaly_status.get("is_anomaly", False)
    score = anomaly_status.get("score", 0)
    
    if is_anomaly:
        panel_title = f"[bold red]{title}: ANOMALY DETECTED[/bold red]"
        content = f"[red]Anomaly Score: {score:.2f}[/red]\n\n"
        
        # Add contributing features if available
        if "features" in anomaly_status:
            content += "[red]Contributing Factors:[/red]\n"
            for feature, value in anomaly_status["features"].items():
                content += f"• {feature}: {value:.2f}\n"
            
        # Add recommendations
        content += "\n[yellow]Risk Management Recommendations:[/yellow]\n"
        content += "• Reduce position sizes by 50%\n"
        content += "• Tighten stop losses\n"
        content += "• Avoid new entries in affected markets\n"
        
        console.print(Panel(content, title=panel_title, border_style="red"))
    else:
        panel_title = f"[bold green]{title}: Normal Market Conditions[/bold green]"
        content = f"[green]Anomaly Score: {score:.2f} (below threshold)[/green]\n"
        content += "[green]Market microstructure appears normal[/green]"
        
        console.print(Panel(content, title=panel_title, border_style="green"))

def execute_trading_decisions(ml_integration, adapter, risk_manager):
    """Execute trading decisions based on ML insights."""
    console.print("\n[bold cyan]Executing Trading Decisions...[/bold cyan]")
    
    # Apply ML insights to generate actions
    result = ml_integration.apply_insights(adapter, risk_manager)
    
    if result["status"] != "success":
        console.print(f"[red]Error: {result.get('message', 'Unknown error')}[/red]")
        return
    
    actions = result.get("actions", {})
    
    if not actions:
        console.print("[yellow]No trading actions generated based on current conditions[/yellow]")
        return
    
    # Display trading actions
    table = Table(title="Trading Actions")
    table.add_column("Asset", style="cyan")
    table.add_column("Action", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("Strategy", style="yellow")
    table.add_column("Confidence", style="blue")
    
    for asset, details in actions.items():
        action = details.get("direction", "HOLD")
        size = details.get("size", 1.0)
        strategy = details.get("strategy", "default")
        confidence = details.get("confidence", 0)
        
        action_color = "green" if action == "BUY" else "red" if action == "SELL" else "yellow"
        
        table.add_row(
            asset,
            f"[{action_color}]{action}[/{action_color}]",
            f"{size:.2f}",
            strategy,
            f"{confidence:.2f}"
        )
    
    console.print(table)
    
    # Execute orders (simulated)
    with Progress() as progress:
        task = progress.add_task("[cyan]Executing orders...", total=len(actions))
        
        for asset, details in actions.items():
            # Here you would actually execute the order
            action = details.get("direction", "HOLD")
            size = details.get("size", 1.0)
            
            try:
                # Simulated execution
                logger.info(f"Executing {action} order for {asset} with size {size}")
                progress.advance(task)
            except Exception as e:
                logger.error(f"Error executing order for {asset}: {e}")
    
    console.print("[green]Trading execution completed[/green]")

def plot_results(ml_integration, data, predictions):
    """Plot the results of the ML analysis."""
    if not data or data.empty:
        console.print("[yellow]No data available for plotting[/yellow]")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot 1: Price with predictions
    symbol = data.index.name or "Asset"
    ax1.set_title(f"{symbol} Price with ML Predictions")
    
    # Plot close price
    ax1.plot(data.index, data['close'], label='Close Price', color='black')
    
    # Plot predictions if available
    if predictions:
        buy_signals = []
        sell_signals = []
        
        for timestamp, row in data.iterrows():
            if timestamp in predictions:
                pred = predictions[timestamp]
                confidence = pred.get('confidence', 0)
                direction = pred.get('direction', 'neutral')
                
                if direction == 'up' and confidence >= 0.65:
                    buy_signals.append((timestamp, row['close']))
                elif direction == 'down' and confidence >= 0.65:
                    sell_signals.append((timestamp, row['close']))
        
        # Plot signals
        if buy_signals:
            buy_x, buy_y = zip(*buy_signals)
            ax1.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy Signal')
        
        if sell_signals:
            sell_x, sell_y = zip(*sell_signals)
            ax1.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell Signal')
    
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot 2: Market Conditions
    if hasattr(ml_integration, 'market_classifier') and ml_integration.market_classifier:
        conditions = ml_integration.market_classifier.predict_condition_series(data)
        
        if conditions is not None and not conditions.empty:
            # Convert conditions to numeric for plotting
            condition_map = {'bullish': 1, 'sideways': 0, 'bearish': -1, 'volatile': 0.5}
            numeric_conditions = conditions['condition'].map(condition_map)
            
            ax2.plot(conditions.index, numeric_conditions, label='Market Condition', color='purple', linewidth=2)
            ax2.set_yticks([-1, 0, 0.5, 1])
            ax2.set_yticklabels(['Bearish', 'Sideways', 'Volatile', 'Bullish'])
            ax2.set_ylabel('Condition')
            ax2.grid(True)
    
    # Plot 3: Anomaly Detection
    if hasattr(ml_integration, 'anomaly_detector') and ml_integration.anomaly_detector:
        anomalies = ml_integration.anomaly_detector.detect_anomaly_series(data)
        
        if anomalies is not None and not anomalies.empty:
            ax3.plot(anomalies.index, anomalies['score'], label='Anomaly Score', color='orange', linewidth=1.5)
            ax3.axhline(y=ml_integration.anomaly_detector.anomaly_threshold, color='red', linestyle='--', label='Threshold')
            
            # Highlight anomaly periods
            anomaly_periods = anomalies[anomalies['is_anomaly'] == True]
            for idx in anomaly_periods.index:
                ax3.axvspan(idx - timedelta(days=1), idx + timedelta(days=1), color='red', alpha=0.2)
            
            ax3.set_ylabel('Anomaly Score')
            ax3.set_ylim(0, 1)
            ax3.legend(loc='upper left')
            ax3.grid(True)
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def run_backtest(ml_integration, adapter, risk_manager, args):
    """Run a backtest of the ML-powered trading strategy."""
    console.print("\n[bold cyan]Running Backtest...[/bold cyan]")
    
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
        return None
    
    assets = args.assets
    timeframe = args.timeframe
    
    # Get historical data for backtesting
    with Progress() as progress:
        data_task = progress.add_task("[cyan]Fetching historical data...", total=len(assets))
        
        asset_data = {}
        for asset in assets:
            try:
                data = adapter.get_historical_data(
                    asset, 
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None and not data.empty:
                    asset_data[asset] = data
                    logger.info(f"Loaded {len(data)} bars for {asset}")
                else:
                    logger.warning(f"No data available for {asset}")
            except Exception as e:
                logger.error(f"Error loading data for {asset}: {e}")
            
            progress.advance(data_task)
    
    if not asset_data:
        console.print("[red]No historical data available for backtesting[/red]")
        return None
    
    # Run backtest
    console.print("[cyan]Processing historical data with ML models...[/cyan]")
    
    results = {}
    signals_by_date = {}
    trades = []
    
    with Progress() as progress:
        # Create a progress bar for each asset
        asset_tasks = {asset: progress.add_task(f"[cyan]Processing {asset}...", total=len(data)) 
                      for asset, data in asset_data.items()}
        
        # Process each asset
        for asset, data in asset_data.items():
            results[asset] = {
                'equity_curve': [1.0],  # Start with $1
                'dates': [data.index[0]],
                'positions': [0],
                'trades': []
            }
            
            position = 0
            equity = 1.0
            
            # Process each date
            for i in range(100, len(data)):  # Skip first 100 bars for initial training
                current_date = data.index[i]
                
                # Extract training data
                train_data = data.iloc[i-100:i].copy()
                
                # Generate predictions
                prediction = ml_integration.price_predictor.predict(train_data, asset)
                
                if prediction:
                    # Record prediction
                    if current_date not in signals_by_date:
                        signals_by_date[current_date] = {}
                    
                    signals_by_date[current_date][asset] = prediction
                    
                    # Generate trading signal
                    direction = prediction.get('direction', 'neutral')
                    confidence = prediction.get('confidence', 0)
                    
                    # Trading logic
                    if direction == 'up' and confidence >= 0.65 and position <= 0:
                        # Buy signal
                        old_position = position
                        position = 1
                        entry_price = data.iloc[i]['close']
                        
                        # Record trade
                        trade = {
                            'asset': asset,
                            'type': 'BUY',
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'position_size': 1,
                            'confidence': confidence
                        }
                        
                        trades.append(trade)
                        results[asset]['trades'].append(trade)
                        
                        logger.info(f"BUY {asset} at {entry_price} on {current_date}")
                        
                    elif direction == 'down' and confidence >= 0.65 and position >= 0:
                        # Sell signal
                        old_position = position
                        position = -1
                        entry_price = data.iloc[i]['close']
                        
                        # Record trade
                        trade = {
                            'asset': asset,
                            'type': 'SELL',
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'position_size': 1,
                            'confidence': confidence
                        }
                        
                        trades.append(trade)
                        results[asset]['trades'].append(trade)
                        
                        logger.info(f"SELL {asset} at {entry_price} on {current_date}")
                
                # Update equity
                if i > 0:
                    price_change = data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1
                    equity_change = position * price_change  # Simple return calculation
                    equity *= (1 + equity_change)
                
                # Record results
                results[asset]['equity_curve'].append(equity)
                results[asset]['dates'].append(current_date)
                results[asset]['positions'].append(position)
                
                # Update progress bar
                progress.advance(asset_tasks[asset])
    
    # Process results
    for asset, result in results.items():
        # Calculate metrics
        equity_curve = np.array(result['equity_curve'])
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = equity_curve[-1] / equity_curve[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
        
        result['metrics'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(result['trades'])
        }
    
    # Display results
    for asset, result in results.items():
        metrics = result['metrics']
        
        console.print(f"\n[bold cyan]Backtest Results for {asset}[/bold cyan]")
        
        table = Table(title=f"{asset} Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Return", f"{metrics['total_return']*100:.2f}%")
        table.add_row("Annual Return", f"{metrics['annual_return']*100:.2f}%")
        table.add_row("Volatility", f"{metrics['volatility']*100:.2f}%")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        table.add_row("Number of Trades", f"{metrics['num_trades']}")
        
        console.print(table)
    
    return results, signals_by_date, trades

def plot_backtest_results(results):
    """Plot the results of the backtest."""
    if not results:
        console.print("[yellow]No backtest results available for plotting[/yellow]")
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curves for each asset
    ax1.set_title("Equity Curves")
    
    for asset, result in results.items():
        equity_curve = result['equity_curve']
        dates = result['dates']
        ax1.plot(dates, equity_curve, label=f"{asset} (Sharpe: {result['metrics']['sharpe_ratio']:.2f})")
    
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot drawdowns
    ax2.set_title("Drawdowns")
    
    for asset, result in results.items():
        equity_curve = np.array(result['equity_curve'])
        dates = result['dates']
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        
        ax2.plot(dates, drawdown, label=asset)
        
        # Highlight major drawdowns
        major_dd_threshold = 0.1  # 10%
        major_dd = drawdown >= major_dd_threshold
        
        for i in range(1, len(major_dd)):
            if major_dd[i] and not major_dd[i-1]:  # Start of major drawdown
                start_idx = i
            elif not major_dd[i] and major_dd[i-1]:  # End of major drawdown
                ax2.axvspan(dates[start_idx], dates[i], color='red', alpha=0.2)
    
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(0, np.max([np.max(drawdown) for asset, result in results.items() for drawdown in [(np.maximum.accumulate(np.array(result['equity_curve'])) - np.array(result['equity_curve'])) / np.maximum.accumulate(np.array(result['equity_curve']))]]) * 1.1)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    console.print(Panel("[bold cyan]ML-Powered Trading Strategy[/bold cyan]", 
                         subtitle="Combining ML insights for effective trading"))
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        console.print(f"[red]Configuration file not found at {config_path}[/red]")
        config_path = None
    
    # Initialize components
    try:
        # Initialize MultiAssetAdapter
        console.print("[cyan]Initializing trading system components...[/cyan]")
        adapter = MultiAssetAdapter(config_path=config_path)
        
        # Initialize RiskManager
        risk_level_map = {
            'low': RiskLevel.LOW,
            'moderate': RiskLevel.MODERATE,
            'high': RiskLevel.HIGH,
            'extreme': RiskLevel.EXTREME
        }
        risk_manager = RiskManager(
            multi_asset_adapter=adapter,
            config_path=config_path,
            default_risk_level=risk_level_map.get(args.risk_level, RiskLevel.MODERATE)
        )
        
        # Initialize ML Integration
        ml_integration = MLIntegration(config_path=config_path)
        ml_integration.initialize(adapter, risk_manager)
        
        console.print("[green]Trading system components initialized successfully[/green]")
        
        # Force training if requested
        if args.force_training:
            console.print("[cyan]Force training ML models...[/cyan]")
            ml_integration.update(force_training=True)
        else:
            console.print("[cyan]Updating ML models...[/cyan]")
            ml_integration.update()
        
        # Get latest data for analysis
        assets = args.assets
        timeframe = args.timeframe
        
        latest_data = {}
        for asset in assets:
            data = adapter.get_latest_data(asset, timeframe=timeframe, bars=100)
            if data is not None and not data.empty:
                latest_data[asset] = data
                logger.info(f"Loaded {len(data)} bars of {timeframe} data for {asset}")
            else:
                logger.warning(f"No data available for {asset}")
        
        if not latest_data:
            console.print("[red]No data available for analysis[/red]")
            return
        
        # Display market conditions
        console.print("\n[bold cyan]Market Condition Analysis[/bold cyan]")
        display_market_conditions(
            ml_integration.market_classifier, 
            latest_data.get("SPY"), 
            "Current Market Regime"
        )
        
        # Get and display price predictions
        console.print("\n[bold cyan]Price Movement Predictions[/bold cyan]")
        predictions = {}
        for asset, data in latest_data.items():
            prediction = ml_integration.price_predictor.predict(data, asset)
            if prediction:
                predictions[asset] = prediction
        
        display_price_predictions(predictions)
        
        # Display optimized parameters
        console.print("\n[bold cyan]Strategy Parameter Optimization[/bold cyan]")
        if ml_integration.current_market_condition:
            display_optimized_parameters(
                ml_integration.optimized_parameters,
                ml_integration.current_market_condition
            )
        else:
            console.print("[yellow]Market condition not available for parameter optimization[/yellow]")
        
        # Display anomaly status
        console.print("\n[bold cyan]Market Anomaly Detection[/bold cyan]")
        display_anomaly_status(ml_integration.anomaly_status)
        
        # Execute trading decisions or run backtest
        if args.backtest:
            # Run backtest
            backtest_results, signals, trades = run_backtest(ml_integration, adapter, risk_manager, args)
            
            # Plot backtest results if requested
            if args.plot and backtest_results:
                plot_backtest_results(backtest_results)
        else:
            # Execute trading decisions
            execute_trading_decisions(ml_integration, adapter, risk_manager)
            
            # Plot results if requested
            if args.plot and "SPY" in latest_data:
                # Get historical predictions for plotting
                historical_predictions = {}
                for i in range(len(latest_data["SPY"])):
                    prediction = ml_integration.price_predictor.predict(
                        latest_data["SPY"].iloc[:i+1], "SPY"
                    )
                    if prediction:
                        historical_predictions[latest_data["SPY"].index[i]] = prediction
                
                plot_results(ml_integration, latest_data["SPY"], historical_predictions)
        
        # Save ML integration state
        ml_integration.save_state()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 