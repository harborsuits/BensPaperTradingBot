#!/usr/bin/env python3
"""
Market Microstructure Anomaly Detection Example

This script demonstrates how to use the MarketAnomalyDetector to identify
unusual patterns in market data that could indicate market dislocations,
manipulation, or trading opportunities.

Usage:
    python market_anomaly_detection_example.py [--symbol SYMBOL] [--days DAYS]
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the anomaly detector
from ml.market_anomaly_detector import MarketAnomalyDetector
from multi_asset_adapter import MultiAssetAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a console for rich output
console = Console()

def load_config():
    """Load configuration from file."""
    try:
        # Try to load from config.json in the parent directory
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/config.json'))
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            console.print("[yellow]No config file found, using default settings[/yellow]")
            return {}
    except Exception as e:
        console.print(f"[red]Error loading config: {str(e)}[/red]")
        return {}

def fetch_historical_data(symbol, days, adapter=None):
    """
    Fetch historical market data for the specified symbol.
    
    Args:
        symbol: Trading symbol
        days: Number of days of historical data to fetch
        adapter: Optional MultiAssetAdapter instance
        
    Returns:
        DataFrame with OHLCV data
    """
    console.print(f"Fetching {days} days of historical data for {symbol}...")
    
    try:
        if adapter:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data using the adapter
            data = adapter.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1h'  # Hourly data for better anomaly detection
            )
            
            console.print(f"[green]Successfully fetched {len(data)} data points[/green]")
            return data
        else:
            # Generate mock data if no adapter is available
            return generate_mock_data(symbol, days)
    except Exception as e:
        console.print(f"[red]Error fetching historical data: {str(e)}[/red]")
        console.print("[yellow]Falling back to mock data[/yellow]")
        return generate_mock_data(symbol, days)

def generate_mock_data(symbol, days):
    """
    Generate mock market data for testing.
    
    Args:
        symbol: Trading symbol
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    console.print("[yellow]Generating mock data for demonstration...[/yellow]")
    
    # Number of hours in the requested days
    periods = days * 24
    
    # Create date range
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(hours=periods)
    date_range = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Generate price data with some trends and volatility
    base_price = 100.0
    
    # Random walk with drift
    returns = np.random.normal(0.0001, 0.001, periods).cumsum()
    
    # Add some seasonality
    seasonality = 0.01 * np.sin(np.linspace(0, 15 * np.pi, periods))
    
    # Combine components
    price_series = base_price * (1 + returns + seasonality)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': price_series * (1 + 0.001 * np.random.randn(periods)),
        'high': price_series * (1 + 0.003 + 0.002 * np.random.rand(periods)),
        'low': price_series * (1 - 0.003 - 0.002 * np.random.rand(periods)),
        'close': price_series,
        'volume': np.random.lognormal(10, 1, periods),
    }, index=date_range)
    
    # Add some bid-ask data if available in a real scenario
    data['bid'] = data['close'] * (1 - 0.0005 - 0.0003 * np.random.rand(periods))
    data['ask'] = data['close'] * (1 + 0.0005 + 0.0003 * np.random.rand(periods))
    data['bid_size'] = np.random.lognormal(8, 1, periods)
    data['ask_size'] = np.random.lognormal(8, 1, periods)
    
    # Add some anomalies
    add_anomalies(data)
    
    console.print(f"[green]Generated {len(data)} mock data points for {symbol}[/green]")
    return data

def add_anomalies(data):
    """Add synthetic anomalies to the mock data."""
    if len(data) < 100:
        return
    
    # Number of anomalies to add (around 1% of data points)
    num_anomalies = max(3, int(len(data) * 0.01))
    
    # Choose random indices for anomalies, but not too close to the beginning or end
    buffer = 10
    indices = np.random.choice(
        range(buffer, len(data) - buffer), 
        size=num_anomalies, 
        replace=False
    )
    
    for idx in indices:
        # Randomly choose anomaly type
        anomaly_type = np.random.choice([
            'price_spike', 'volume_spike', 'spread_widening', 
            'order_imbalance', 'mini_flash_crash'
        ])
        
        if anomaly_type == 'price_spike':
            # Sudden price jump
            direction = np.random.choice([-1, 1])
            factor = 1 + direction * np.random.uniform(0.02, 0.05)
            data.iloc[idx, data.columns.get_indexer(['open', 'high', 'low', 'close'])] *= factor
            
        elif anomaly_type == 'volume_spike':
            # Unusual volume
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= np.random.uniform(5, 10)
            
        elif anomaly_type == 'spread_widening':
            # Bid-ask spread suddenly widens
            if 'bid' in data.columns and 'ask' in data.columns:
                data.iloc[idx, data.columns.get_indexer(['bid'])] *= 0.98
                data.iloc[idx, data.columns.get_indexer(['ask'])] *= 1.02
                
        elif anomaly_type == 'order_imbalance':
            # Significant imbalance in order book
            if 'bid_size' in data.columns and 'ask_size' in data.columns:
                if np.random.random() > 0.5:
                    # More bids than asks
                    data.iloc[idx, data.columns.get_indexer(['bid_size'])] *= np.random.uniform(3, 5)
                else:
                    # More asks than bids
                    data.iloc[idx, data.columns.get_indexer(['ask_size'])] *= np.random.uniform(3, 5)
                    
        elif anomaly_type == 'mini_flash_crash':
            # Brief but sharp price drop followed by recovery
            if idx + 3 < len(data):
                # Drop
                crash_factor = np.random.uniform(0.9, 0.95)
                data.iloc[idx, data.columns.get_indexer(['close', 'low'])] *= crash_factor
                
                # Partial recovery in next candle
                recovery = 1 + np.random.uniform(0.01, 0.03)
                data.iloc[idx+1, data.columns.get_indexer(['open', 'close'])] *= recovery
                
                # Full recovery in a couple of candles
                recovery = 1 / crash_factor
                data.iloc[idx+2:idx+4, data.columns.get_indexer(['close'])] *= recovery

def train_anomaly_detector(data, symbol):
    """
    Train an anomaly detector on the provided data.
    
    Args:
        data: DataFrame with market data
        symbol: Trading symbol
        
    Returns:
        Trained MarketAnomalyDetector instance
    """
    console.print(Panel(f"Training anomaly detector for {symbol}", style="blue"))
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=100)
        
        # Initialize detector
        detector = MarketAnomalyDetector(
            symbol=symbol,
            lookback_window=20,
            alert_threshold=0.85,
            model_dir="models/anomaly_detection",
            use_autoencoder=True,
            contamination=0.01
        )
        
        progress.update(task, completed=30)
        
        # Train the detector
        training_results = detector.train(data)
        
        progress.update(task, completed=100)
    
    # Display training results
    if "error" in training_results:
        console.print(f"[red]Error during training: {training_results['error']}[/red]")
    else:
        table = Table(title=f"Training Results for {symbol}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Samples", str(training_results.get("num_samples", 0)))
        table.add_row("Features", str(training_results.get("num_features", 0)))
        table.add_row("Anomalies Found", str(training_results.get("num_anomalies", 0)))
        table.add_row("Anomaly Threshold", f"{training_results.get('anomaly_threshold', 0):.4f}")
        
        # Add top features
        if "top_features" in training_results:
            top_features = ", ".join([f"{f[0]}" for f in training_results["top_features"]])
            table.add_row("Top Features", top_features)
        
        console.print(table)
    
    return detector

def detect_anomalies(detector, data):
    """
    Detect anomalies in the provided data.
    
    Args:
        detector: Trained MarketAnomalyDetector instance
        data: DataFrame with market data
        
    Returns:
        Dictionary with detection results
    """
    console.print(Panel("Detecting anomalies in market data", style="blue"))
    
    # Detect anomalies
    results = detector.detect_anomalies(data)
    
    if "error" in results:
        console.print(f"[red]Error during detection: {results['error']}[/red]")
        return results
    
    # Display results
    table = Table(title="Anomaly Detection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Data Points", str(len(data)))
    table.add_row("Anomalies Detected", str(results.get("num_anomalies", 0)))
    table.add_row("Maximum Anomaly Score", f"{results.get('max_anomaly_score', 0):.4f}")
    table.add_row("Latest Point Score", f"{results.get('latest_score', 0):.4f}")
    table.add_row("Alert Threshold", f"{detector.alert_threshold:.4f}")
    
    console.print(table)
    
    # If anomalies were detected, display details
    if results.get("num_anomalies", 0) > 0:
        console.print("\n[bold yellow]Anomaly Details:[/bold yellow]")
        for idx in results.get("anomaly_indices", [])[-5:]:  # Show the last 5 anomalies
            if idx < len(data):
                date = data.index[idx]
                console.print(f"Anomaly at {date}: Score {results.get('latest_score', 0):.4f}")
                
                # Show the actual data point
                point = data.iloc[idx]
                console.print(f"  Price: {point['close']:.2f}, Volume: {point['volume']:.0f}")
                
                if "latest_point" in results and "contributing_features" in results["latest_point"]:
                    features = results["latest_point"]["contributing_features"]
                    if features:
                        console.print("  Contributing factors:")
                        for feature, contrib in features:
                            console.print(f"    - {feature}: {contrib:.4f}")
    
    # Generate an alert message for the latest point
    if results.get("latest_score", 0) > detector.alert_threshold:
        alert = detector.get_alert_message(data, results)
        console.print(Panel(alert, title="[bold red]ANOMALY ALERT[/bold red]", border_style="red"))
    
    return results

def visualize_results(data, results, detector):
    """
    Visualize the anomaly detection results.
    
    Args:
        data: DataFrame with market data
        results: Anomaly detection results
        detector: MarketAnomalyDetector instance
    """
    console.print(Panel("Visualizing anomaly detection results", style="blue"))
    
    try:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data on top subplot
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
        
        # Highlight anomalies on the price chart
        anomaly_indices = results.get("anomaly_indices", [])
        if anomaly_indices:
            anomaly_dates = [data.index[i] for i in anomaly_indices if i < len(data)]
            anomaly_prices = [data['close'].iloc[i] for i in anomaly_indices if i < len(data)]
            ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=50, label='Anomalies')
        
        # Add volume as bars on the bottom of price chart
        ax1v = ax1.twinx()
        ax1v.bar(data.index, data['volume'], alpha=0.3, color='gray', label='Volume')
        ax1v.set_ylabel('Volume')
        
        # Set up legends and labels
        ax1.set_title(f'Price Chart with Anomaly Detection for {detector.symbol}')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1v.legend(loc='upper right')
        
        # Plot anomaly scores on bottom subplot
        # We need to calculate scores for each point
        features = detector.engineer_features(data)
        if not features.empty:
            X = detector.scaler.transform(features)
            scores = detector._calculate_anomaly_scores(X)
            
            ax2.plot(data.index[-len(scores):], scores, color='purple', label='Anomaly Score')
            ax2.axhline(y=detector.alert_threshold, color='red', linestyle='--', label=f'Alert Threshold ({detector.alert_threshold})')
            
            # Highlight anomalies
            for i in anomaly_indices:
                if i < len(scores):
                    ax2.axvline(x=data.index[i], color='red', alpha=0.3)
            
            ax2.set_title('Anomaly Scores')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1.1)
            ax2.legend()
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save figure
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"anomaly_detection_{detector.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename)
        
        console.print(f"[green]Visualization saved to {filename}[/green]")
        
        # Show figure if running in interactive mode
        plt.show()
        
    except Exception as e:
        console.print(f"[red]Error visualizing results: {str(e)}[/red]")

def simulate_live_monitoring(detector, initial_data, num_periods=10):
    """
    Simulate live market monitoring with the anomaly detector.
    
    Args:
        detector: Trained MarketAnomalyDetector
        initial_data: Initial market data
        num_periods: Number of periods to simulate
    """
    console.print(Panel(f"Simulating live monitoring for {num_periods} periods", style="blue"))
    
    # Copy initial data
    data = initial_data.copy()
    
    # Get the last timestamp
    last_timestamp = data.index[-1]
    
    # Create progress bar
    with Progress() as progress:
        task = progress.add_task("[cyan]Monitoring...", total=num_periods)
        
        for i in range(num_periods):
            # Generate a new data point
            new_timestamp = last_timestamp + timedelta(hours=1)
            last_close = data['close'].iloc[-1]
            
            # Generate new price (random walk with drift)
            price_change = np.random.normal(0.0001, 0.001)
            new_close = last_close * (1 + price_change)
            
            # Generate new OHLCV data
            new_point = pd.DataFrame({
                'open': [new_close * (1 - 0.001 * np.random.randn())],
                'high': [new_close * (1 + 0.002 + 0.001 * np.random.rand())],
                'low': [new_close * (1 - 0.002 - 0.001 * np.random.rand())],
                'close': [new_close],
                'volume': [np.random.lognormal(10, 1)],
                'bid': [new_close * (1 - 0.0005 - 0.0001 * np.random.rand())],
                'ask': [new_close * (1 + 0.0005 + 0.0001 * np.random.rand())],
                'bid_size': [np.random.lognormal(8, 1)],
                'ask_size': [np.random.lognormal(8, 1)]
            }, index=[new_timestamp])
            
            # Add anomaly with 20% probability
            if np.random.random() < 0.2:
                # Add a price spike
                spike_factor = 1 + np.random.choice([-1, 1]) * np.random.uniform(0.01, 0.03)
                new_point['close'] *= spike_factor
                console.print(f"[yellow]Injected anomaly at {new_timestamp}: price spike {spike_factor:.4f}[/yellow]")
            
            # Append new data
            data = pd.concat([data, new_point])
            
            # Run detection on latest data (using a window)
            window_size = 100
            window_data = data.iloc[-window_size:] if len(data) > window_size else data
            
            results = detector.detect_anomalies(window_data)
            
            # Check if an anomaly was detected in the latest point
            latest_score = results.get("latest_score", 0)
            
            # Update console with status
            status = f"Period {i+1}/{num_periods} - {new_timestamp}: Score {latest_score:.4f}"
            if latest_score > detector.alert_threshold:
                console.print(f"[bold red]{status} - ANOMALY DETECTED![/bold red]")
                
                # Display alert
                alert = detector.get_alert_message(window_data, results)
                console.print(Panel(alert, title="[bold red]ANOMALY ALERT[/bold red]", border_style="red"))
            else:
                console.print(f"[green]{status} - Normal[/green]")
            
            # Update progress
            progress.update(task, advance=1)
            
            # Update last timestamp
            last_timestamp = new_timestamp
    
    console.print("[bold green]Live monitoring simulation completed[/bold green]")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Market Microstructure Anomaly Detection Example")
    parser.add_argument("--symbol", default="SPY", help="Trading symbol to analyze")
    parser.add_argument("--days", type=int, default=30, help="Number of days of historical data")
    args = parser.parse_args()
    
    console.print(Panel(f"Market Microstructure Anomaly Detection Example", 
                        subtitle=f"Symbol: {args.symbol}, Days: {args.days}", 
                        style="bold blue"))
    
    # Load configuration
    config = load_config()
    
    # Initialize MultiAssetAdapter if possible
    adapter = None
    try:
        # Only attempt to initialize if dependencies/config are available
        adapter = MultiAssetAdapter(config.get("data_providers", {}))
        console.print("[green]Initialized MultiAssetAdapter for data fetching[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not initialize MultiAssetAdapter: {str(e)}[/yellow]")
        console.print("[yellow]Will use mock data for demonstration[/yellow]")
    
    # Fetch historical data
    data = fetch_historical_data(args.symbol, args.days, adapter)
    
    if data is None or len(data) < 50:
        console.print("[red]Not enough data to proceed. Exiting.[/red]")
        return
    
    # Display data summary
    console.print(Panel(f"Data Summary for {args.symbol}", style="green"))
    console.print(f"Time Range: {data.index[0]} to {data.index[-1]}")
    console.print(f"Number of data points: {len(data)}")
    console.print(f"Price Range: {data['low'].min():.2f} - {data['high'].max():.2f}")
    
    # Train the anomaly detector
    detector = train_anomaly_detector(data, args.symbol)
    
    # Detect anomalies
    results = detect_anomalies(detector, data)
    
    # Visualize results
    visualize_results(data, results, detector)
    
    # Simulate live monitoring
    simulate_live_monitoring(detector, data, num_periods=10)
    
    console.print(Panel("Example completed successfully!", style="bold green"))

if __name__ == "__main__":
    main() 