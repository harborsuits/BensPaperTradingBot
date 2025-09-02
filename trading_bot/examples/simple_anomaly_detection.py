#!/usr/bin/env python3
"""
Simple Market Anomaly Detection Example

This script demonstrates the basic usage of MarketAnomalyDetector
without integrating with other trading components.

Usage:
    python simple_anomaly_detection.py --symbol BTC/USD
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the anomaly detector
from ml.market_anomaly_detector import MarketAnomalyDetector

# Rich console utilities for better output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

def load_data(symbol, days=30):
    """
    Load historical data for the specified symbol.
    This is a simplified version that generates mock data.
    
    Args:
        symbol: Trading symbol
        days: Number of days of data
        
    Returns:
        DataFrame with OHLCV data
    """
    console.print(f"Generating mock data for {symbol} over {days} days...")
    
    # Number of hours in the requested days
    periods = days * 24
    
    # Create date range
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(hours=periods)
    date_range = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate price data with some trends and volatility
    base_price = 100.0
    if "BTC" in symbol:
        base_price = 30000.0
    elif "ETH" in symbol:
        base_price = 2000.0
    
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
    
    console.print(f"[green]Generated {len(data)} data points for {symbol}[/green]")
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
    
    # Add specific anomalies to highlight at the end
    # Last 10% of the data - for demonstration purposes
    end_section = int(len(data) * 0.9)
    if end_section > buffer:
        indices = np.append(indices, [end_section, end_section + 10, end_section + 20])
    
    anomaly_descriptions = []
    
    for idx in indices:
        if idx >= len(data):
            continue
            
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
            anomaly_descriptions.append(f"Price spike at {data.index[idx]} - {factor:.4f}x")
            
        elif anomaly_type == 'volume_spike':
            # Unusual volume
            factor = np.random.uniform(5, 10)
            data.iloc[idx, data.columns.get_indexer(['volume'])] *= factor
            anomaly_descriptions.append(f"Volume spike at {data.index[idx]} - {factor:.1f}x normal")
            
        elif anomaly_type == 'spread_widening':
            # Bid-ask spread suddenly widens
            if 'bid' in data.columns and 'ask' in data.columns:
                data.iloc[idx, data.columns.get_indexer(['bid'])] *= 0.98
                data.iloc[idx, data.columns.get_indexer(['ask'])] *= 1.02
                anomaly_descriptions.append(f"Spread widening at {data.index[idx]}")
                
        elif anomaly_type == 'order_imbalance':
            # Significant imbalance in order book
            if 'bid_size' in data.columns and 'ask_size' in data.columns:
                if np.random.random() > 0.5:
                    # More bids than asks
                    factor = np.random.uniform(3, 5)
                    data.iloc[idx, data.columns.get_indexer(['bid_size'])] *= factor
                    anomaly_descriptions.append(f"Bid imbalance at {data.index[idx]} - {factor:.1f}x")
                else:
                    # More asks than bids
                    factor = np.random.uniform(3, 5)
                    data.iloc[idx, data.columns.get_indexer(['ask_size'])] *= factor
                    anomaly_descriptions.append(f"Ask imbalance at {data.index[idx]} - {factor:.1f}x")
                    
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
                
                anomaly_descriptions.append(f"Mini flash crash at {data.index[idx]} - {crash_factor:.2f}x drop")
    
    # Print anomaly descriptions
    if anomaly_descriptions:
        console.print("[yellow]Injected anomalies:[/yellow]")
        for desc in anomaly_descriptions[-5:]:  # Show the last 5 anomalies
            console.print(f"  - {desc}")
    
    return data

def setup_anomaly_detector(symbol):
    """
    Initialize the anomaly detector with reasonable default parameters.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        MarketAnomalyDetector instance
    """
    console.print(Panel("Setting up Market Anomaly Detector", style="blue"))
    
    # Create models directory if it doesn't exist
    models_dir = "models/anomaly_detection"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize the anomaly detector
    detector = MarketAnomalyDetector(
        symbol=symbol,
        lookback_window=20,  # Use 20 periods of history for feature engineering
        alert_threshold=0.75,  # Alert when anomaly score is above 0.75
        model_dir=models_dir,
        use_autoencoder=True,  # Use both Isolation Forest and Autoencoder
        contamination=0.01  # Assume about 1% of data points are anomalies
    )
    
    console.print("[green]Anomaly detector initialized successfully[/green]")
    return detector

def train_detector(detector, data):
    """
    Train the anomaly detector on the provided data.
    
    Args:
        detector: MarketAnomalyDetector instance
        data: DataFrame with OHLCV data
        
    Returns:
        Training results dictionary
    """
    console.print(Panel(f"Training anomaly detector for {detector.symbol}", style="blue"))
    
    try:
        # Train the detector
        training_results = detector.train(data)
        
        # Display training results
        if "error" in training_results:
            console.print(f"[red]Error during training: {training_results['error']}[/red]")
        else:
            table = Table(title=f"Training Results for {detector.symbol}")
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
            
            # Save the model
            detector.save_model()
            console.print("[green]Model saved successfully[/green]")
        
        return training_results
        
    except Exception as e:
        console.print(f"[red]Error training detector: {str(e)}[/red]")
        return {"error": str(e)}

def detect_anomalies(detector, data):
    """
    Detect anomalies in the provided data.
    
    Args:
        detector: MarketAnomalyDetector instance
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary with detection results
    """
    console.print(Panel("Detecting anomalies in market data", style="blue"))
    
    try:
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
            # Get all anomalies
            anomaly_indices = results.get("anomaly_indices", [])
            
            # Show only the top 5 anomalies by score if there are too many
            if len(anomaly_indices) > 5:
                anomaly_scores = [results.get("anomaly_scores", [])[i] for i in range(len(anomaly_indices))]
                # Get indices of top 5 anomalies by score
                top_indices = sorted(range(len(anomaly_scores)), key=lambda i: anomaly_scores[i], reverse=True)[:5]
                top_anomalies = [anomaly_indices[i] for i in top_indices]
            else:
                top_anomalies = anomaly_indices
            
            for idx in top_anomalies:
                if idx < len(data):
                    date = data.index[idx]
                    score = results.get("anomaly_scores", [])[list(anomaly_indices).index(idx)]
                    console.print(f"Anomaly at {date}: Score {score:.4f}")
                    
                    # Show the actual data point
                    point = data.iloc[idx]
                    console.print(f"  Price: {point['close']:.2f}, Volume: {point['volume']:.0f}")
                    
                    # Show bid-ask spread if available
                    if 'bid' in point and 'ask' in point:
                        spread = (point['ask'] - point['bid']) / point['close'] * 10000  # in basis points
                        console.print(f"  Spread: {spread:.2f} bps")
                    
                    # Show imbalance if available
                    if 'bid_size' in point and 'ask_size' in point:
                        imbalance = point['bid_size'] / (point['bid_size'] + point['ask_size'])
                        console.print(f"  Order book imbalance: {imbalance:.2f} (0.5 is balanced)")
        
        # Generate an alert message for the latest point
        if results.get("latest_score", 0) > detector.alert_threshold:
            alert = detector.get_alert_message(data, results)
            console.print(Panel(alert, title="[bold red]ANOMALY ALERT[/bold red]", border_style="red"))
        
        return results
    
    except Exception as e:
        console.print(f"[red]Error detecting anomalies: {str(e)}[/red]")
        return {"error": str(e)}

def visualize_anomalies(data, results, detector):
    """
    Visualize the anomaly detection results.
    
    Args:
        data: DataFrame with OHLCV data
        results: Dictionary with detection results
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
        anomaly_scores = results.get("all_scores", [])
        if anomaly_scores:
            ax2.plot(data.index[-len(anomaly_scores):], anomaly_scores, color='purple', label='Anomaly Score')
            ax2.axhline(y=detector.alert_threshold, color='red', linestyle='--', label=f'Alert Threshold ({detector.alert_threshold})')
            
            # Highlight anomalies
            for i in anomaly_indices:
                if i < len(data):
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
        filename = os.path.join(output_dir, f"simple_anomaly_{detector.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename)
        
        console.print(f"[green]Visualization saved to {filename}[/green]")
        
        # Show figure
        plt.show()
        
    except Exception as e:
        console.print(f"[red]Error visualizing anomalies: {str(e)}[/red]")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Market Anomaly Detection Example")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol to analyze")
    parser.add_argument("--days", type=int, default=30, help="Number of days of historical data")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--threshold", type=float, default=0.75, help="Anomaly alert threshold (0-1)")
    args = parser.parse_args()
    
    console.print(Panel(f"Simple Market Anomaly Detection Example",
                        subtitle=f"Symbol: {args.symbol}, Days: {args.days}", 
                        style="bold blue"))
    
    # Load data
    data = load_data(args.symbol, args.days)
    
    if data is None or len(data) < 50:
        console.print("[red]Not enough data to proceed. Exiting.[/red]")
        return
    
    # Display data summary
    console.print(Panel(f"Data Summary for {args.symbol}", style="green"))
    console.print(f"Time Range: {data.index[0]} to {data.index[-1]}")
    console.print(f"Number of data points: {len(data)}")
    console.print(f"Price Range: {data['low'].min():.2f} - {data['high'].max():.2f}")
    
    # Setup anomaly detector
    detector = setup_anomaly_detector(args.symbol)
    
    # Override default threshold if specified
    if args.threshold != 0.75:
        detector.alert_threshold = args.threshold
        console.print(f"[yellow]Alert threshold set to {args.threshold}[/yellow]")
    
    # Train the anomaly detector
    train_detector(detector, data)
    
    # Detect anomalies
    results = detect_anomalies(detector, data)
    
    # Visualize results unless --no-plot is specified
    if not args.no_plot:
        visualize_anomalies(data, results, detector)
    
    # Summarize findings
    console.print(Panel("Summary of Findings", style="bold green"))
    
    anomaly_count = results.get("num_anomalies", 0)
    latest_score = results.get("latest_score", 0)
    
    if anomaly_count > 0:
        console.print(f"Found {anomaly_count} anomalies in the data")
        
        # Categorize the types of anomalies if possible
        console.print("\nPotential anomaly categories:")
        
        # This is a simplified categorization just for demonstration
        # In a real system, this would be more sophisticated
        if any(results.get("feature_contributions", {}).get("price_volatility", [])):
            console.print("- [yellow]Price volatility anomalies[/yellow]")
        
        if any(results.get("feature_contributions", {}).get("volume_spike", [])):
            console.print("- [yellow]Volume spike anomalies[/yellow]")
        
        if any(results.get("feature_contributions", {}).get("spread_widening", [])):
            console.print("- [yellow]Bid-ask spread anomalies[/yellow]")
        
        if any(results.get("feature_contributions", {}).get("order_imbalance", [])):
            console.print("- [yellow]Order book imbalance anomalies[/yellow]")
    else:
        console.print("[green]No significant anomalies detected in the data[/green]")
    
    # Report on the latest market state
    console.print("\n[bold]Latest Market State:[/bold]")
    console.print(f"Anomaly Score: {latest_score:.4f} (Threshold: {detector.alert_threshold})")
    
    if latest_score > detector.alert_threshold:
        console.print("[bold red]ALERT: Current market conditions show anomalous behavior[/bold red]")
        console.print("Recommended Action: Exercise caution in trading decisions")
    else:
        console.print("[green]Current market conditions appear normal[/green]")
    
    console.print("\n[bold green]Analysis completed successfully![/bold green]")

if __name__ == "__main__":
    main() 