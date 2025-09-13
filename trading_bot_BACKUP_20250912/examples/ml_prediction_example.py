#!/usr/bin/env python3
"""
ML Prediction Example

This script demonstrates how to use the ML components with the trading bot,
showing price predictions, market classification, parameter optimization,
and anomaly detection in action.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

# Add parent directory to path to allow importing from trading_bot
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.ml.ml_integration import MLIntegration
from trading_bot.multi_asset_adapter import MultiAssetAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_example')

# Initialize rich console
console = Console()

def main():
    """Main function to demonstrate ML integration."""
    console.print(Panel.fit(
        "[bold blue]ML Prediction Example[/bold blue]\n"
        "Demonstrating machine learning components for trading bot",
        title="Trading Bot ML"
    ))
    
    # Step 1: Load configuration and initialize MultiAssetAdapter
    console.print("[bold]Step 1:[/bold] Loading configuration...")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "config", "config.json")
    
    try:
        # Initialize the MultiAssetAdapter
        console.print("Initializing MultiAssetAdapter...")
        adapter = MultiAssetAdapter(config_path=config_path)
        console.print("[green]✓[/green] MultiAssetAdapter initialized successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing MultiAssetAdapter: {str(e)}")
        console.print("Using mock data for demonstration")
        adapter = None
    
    # Step 2: Initialize ML Integration
    console.print("\n[bold]Step 2:[/bold] Initializing ML components...")
    
    ml_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "config", "ml_config.json")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "models")
    
    # Make models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        ml_integration = MLIntegration(
            multi_asset_adapter=adapter,
            models_dir=models_dir,
            config_path=ml_config_path
        )
        console.print("[green]✓[/green] ML Integration initialized successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing ML Integration: {str(e)}")
        sys.exit(1)
    
    # Step 3: Select symbols for demonstration
    test_symbols = ["AAPL", "BTC/USD", "EUR/USD", "ES"]
    if adapter:
        try:
            # Get first 5 symbols from each asset class
            stocks = adapter.get_symbols_by_asset_class("stock")[:2]
            forex = adapter.get_symbols_by_asset_class("forex")[:1]
            crypto = adapter.get_symbols_by_asset_class("crypto")[:1]
            futures = adapter.get_symbols_by_asset_class("future")[:1]
            
            test_symbols = stocks + forex + crypto + futures
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Using default test symbols: {e}")
    
    console.print(f"\n[bold]Step 3:[/bold] Selected test symbols: {', '.join(test_symbols)}")
    
    # Step 4: Generate or load example data
    console.print("\n[bold]Step 4:[/bold] Loading market data...")
    
    # Dictionary to store data for each symbol
    market_data = {}
    
    for symbol in test_symbols:
        with console.status(f"Loading data for {symbol}..."):
            if adapter:
                try:
                    # Get historical data from adapter
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)  # One year of data
                    
                    data = adapter.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe='1d'
                    )
                    
                    if len(data) < 30:
                        console.print(f"[yellow]⚠[/yellow] Insufficient data for {symbol}, using mock data")
                        data = generate_mock_data(symbol)
                    else:
                        console.print(f"[green]✓[/green] Loaded {len(data)} data points for {symbol}")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Error loading data for {symbol}: {str(e)}")
                    data = generate_mock_data(symbol)
            else:
                data = generate_mock_data(symbol)
            
            market_data[symbol] = data
    
    # Step 5: Train ML models for each symbol if needed
    console.print("\n[bold]Step 5:[/bold] Training ML models...")
    
    with Progress() as progress:
        train_task = progress.add_task("[cyan]Training models...", total=len(test_symbols))
        
        for symbol in test_symbols:
            progress.update(train_task, description=f"[cyan]Training {symbol}...")
            
            # Check if models exist or need retraining
            price_pred_needed = ml_integration.should_retrain(symbol, "price_prediction")
            market_class_needed = ml_integration.should_retrain(symbol, "market_classification")
            param_opt_needed = ml_integration.should_retrain(symbol, "parameter_optimization")
            anomaly_needed = ml_integration.should_retrain(symbol, "anomaly_detection")
            
            if price_pred_needed or market_class_needed or param_opt_needed or anomaly_needed:
                try:
                    # Train just this symbol
                    results = ml_integration.train_ml_components([symbol])
                    
                    # Check success
                    if symbol in results["details"] and results["details"][symbol]["components_trained"] > 0:
                        progress.update(train_task, advance=1)
                    else:
                        console.print(f"[yellow]⚠[/yellow] Training failed for {symbol}")
                        progress.update(train_task, advance=1)
                        
                except Exception as e:
                    console.print(f"[red]✗[/red] Error training models for {symbol}: {str(e)}")
                    progress.update(train_task, advance=1)
            else:
                console.print(f"[green]✓[/green] Models for {symbol} are up to date")
                progress.update(train_task, advance=1)
    
    # Step 6: Generate predictions for each symbol
    console.print("\n[bold]Step 6:[/bold] Generating predictions...")
    
    all_insights = {}
    
    for symbol in test_symbols:
        console.print(f"\n[bold]Predictions for {symbol}:[/bold]")
        
        try:
            # Get latest portion of data for predictions
            latest_data = market_data[symbol].iloc[-60:].copy()
            
            # Get consolidated insights
            insights = ml_integration.get_consolidated_ml_insights(symbol, latest_data)
            all_insights[symbol] = insights
            
            # Display predictions
            display_insights(insights)
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error generating predictions for {symbol}: {str(e)}")
    
    # Step 7: Visualize results
    console.print("\n[bold]Step 7:[/bold] Visualizing results...")
    
    try:
        plot_predictions(all_insights, market_data)
        console.print("[green]✓[/green] Generated visualization")
    except Exception as e:
        console.print(f"[red]✗[/red] Error visualizing results: {str(e)}")
    
    console.print("\n[bold green]Done![/bold green] ML prediction example completed successfully.")

def generate_mock_data(symbol, days=365):
    """Generate mock OHLCV data for demonstration purposes."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random price series with trend and volatility
    base_price = 100
    if 'BTC' in symbol:
        base_price = 30000
    elif 'EUR' in symbol:
        base_price = 1.1
    
    # Add a trend component
    trend_factor = np.linspace(0, 0.3, len(dates))
    if np.random.random() < 0.5:  # 50% chance of downtrend
        trend_factor = -trend_factor
    
    # Add a cyclical component
    cycle = 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    
    # Generate price series
    closes = base_price * (1 + trend_factor + cycle + 0.01 * np.random.randn(len(dates)))
    
    # Generate other OHLCV data
    daily_volatility = 0.015
    volumes = np.random.normal(1000000, 300000, size=len(dates))
    
    data = pd.DataFrame({
        'open': closes * (1 - 0.5 * daily_volatility * np.random.rand(len(dates))),
        'high': closes * (1 + daily_volatility * np.random.rand(len(dates))),
        'low': closes * (1 - daily_volatility * np.random.rand(len(dates))),
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    return data

def display_insights(insights):
    """Display ML insights in a formatted table."""
    console = Console()
    
    # Display price predictions
    if "price_predictions" in insights and insights["price_predictions"]:
        console.print("\n[bold cyan]Price Predictions:[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time Horizon")
        table.add_column("Direction")
        table.add_column("Confidence")
        table.add_column("Target Price")
        
        for horizon, pred in insights["price_predictions"].items():
            direction = pred["direction"]
            direction_colored = f"[green]{direction}[/green]" if direction == "UP" else f"[red]{direction}[/red]"
            
            confidence = f"{pred['confidence']:.2f}"
            target_price = f"{pred.get('target_price', 'N/A')}"
            
            table.add_row(
                horizon,
                direction_colored,
                confidence,
                target_price
            )
        
        console.print(table)
    
    # Display market regime
    if "market_regime" in insights:
        regime = insights["market_regime"]
        prob = insights.get("regime_probability", 0)
        
        regime_color = "yellow"
        if regime == "Bullish Trend":
            regime_color = "green"
        elif regime == "Bearish Trend":
            regime_color = "red"
        
        console.print(f"\n[bold cyan]Market Regime:[/bold cyan] [{regime_color}]{regime}[/{regime_color}] (Confidence: {prob:.2f})")
    
    # Display optimal parameters
    if "optimal_parameters" in insights and insights["optimal_parameters"]:
        console.print("\n[bold cyan]Optimal Strategy Parameters:[/bold cyan]")
        
        params = insights["optimal_parameters"]
        for param, value in params.items():
            console.print(f"  {param}: {value}")
    
    # Display anomalies
    if "anomalies" in insights and insights["anomalies"].get("detected", False):
        console.print("\n[bold red]Market Anomalies Detected:[/bold red]")
        
        for explanation in insights["anomalies"].get("explanation", []):
            console.print(f"  • {explanation}")
    
    # Display trading signals
    if "trading_signals" in insights:
        signals = insights["trading_signals"]
        signal = signals["overall_signal"]
        confidence = signals["confidence"]
        
        signal_color = "yellow"
        if signal == "LONG":
            signal_color = "green"
        elif signal == "SHORT":
            signal_color = "red"
        
        console.print(f"\n[bold cyan]Trading Signal:[/bold cyan] [{signal_color}]{signal}[/{signal_color}] (Confidence: {confidence:.2f})")
        
        if signals["reasoning"]:
            console.print("[bold]Reasoning:[/bold]")
            for reason in signals["reasoning"]:
                console.print(f"  • {reason}")

def plot_predictions(all_insights, market_data):
    """Create visualization of predictions and market data."""
    n_symbols = len(all_insights)
    
    # Only proceed if we have insights
    if n_symbols == 0:
        return
    
    fig, axes = plt.subplots(n_symbols, 1, figsize=(12, 4*n_symbols), dpi=100)
    if n_symbols == 1:
        axes = [axes]
    
    for i, (symbol, insights) in enumerate(all_insights.items()):
        ax = axes[i]
        
        # Get price data
        if symbol in market_data:
            prices = market_data[symbol]
            
            # Plot price
            ax.plot(prices.index, prices['close'], label='Close Price', color='blue', alpha=0.7)
            
            # Highlight last 30 days
            last_30 = prices.iloc[-30:].index
            ax.axvspan(last_30[0], last_30[-1], alpha=0.2, color='gray')
            
            # Add signals from insights if available
            if "trading_signals" in insights:
                signal = insights["trading_signals"]["overall_signal"]
                
                last_date = prices.index[-1]
                y_pos = prices['close'].iloc[-1]
                
                if signal == "LONG":
                    ax.plot(last_date, y_pos, 'g^', markersize=10, label='LONG Signal')
                elif signal == "SHORT":
                    ax.plot(last_date, y_pos, 'rv', markersize=10, label='SHORT Signal')
                
                # Add predictions if available
                if "price_predictions" in insights and insights["price_predictions"]:
                    # Get the max horizon prediction
                    horizons = sorted(insights["price_predictions"].keys())
                    
                    if horizons:
                        furthest_horizon = horizons[-1]
                        pred = insights["price_predictions"][furthest_horizon]
                        
                        days_ahead = int(furthest_horizon.split('_')[0])
                        future_date = last_date + timedelta(days=days_ahead)
                        
                        # Use target price if available, otherwise project based on direction
                        if pred.get('target_price'):
                            future_price = pred['target_price']
                        else:
                            pct_change = 0.02 if pred['direction'] == 'UP' else -0.02
                            future_price = y_pos * (1 + pct_change)
                        
                        color = 'green' if pred['direction'] == 'UP' else 'red'
                        ax.plot([last_date, future_date], [y_pos, future_price], 
                                color=color, linestyle='--', alpha=0.7)
                        ax.plot(future_date, future_price, 'o', color=color, 
                                alpha=0.7, label=f"{furthest_horizon} Prediction")
            
            # Add market regime if available
            if "market_regime" in insights:
                regime = insights["market_regime"]
                text_color = 'green' if regime == 'Bullish Trend' else 'red' if regime == 'Bearish Trend' else 'orange'
                ax.text(0.02, 0.05, f"Regime: {regime}", transform=ax.transAxes, 
                        fontsize=10, color=text_color, bbox=dict(facecolor='white', alpha=0.7))
            
            # Add anomaly flags if available
            if "anomalies" in insights and insights["anomalies"].get("detected", False):
                ax.text(0.02, 0.12, "⚠️ Anomalies Detected", transform=ax.transAxes, 
                        fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f"{symbol} Price and Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"ml_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_path)
    
    # Return path for display
    return output_path

if __name__ == "__main__":
    main() 