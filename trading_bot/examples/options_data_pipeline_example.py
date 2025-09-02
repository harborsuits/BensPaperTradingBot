#!/usr/bin/env python3
"""
Options Data Pipeline Example

This script demonstrates how to use the options data pipeline with the MultiAssetAdapter.
It shows:
1. How to initialize the options market data component
2. How to integrate multiple data sources
3. How to handle data gaps
4. How to use the options data with the MultiAssetAdapter
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from trading_bot.adapters.multi_asset_adapter import MultiAssetAdapter
from trading_bot.data.options_market_data import OptionsMarketData, DataSourcePriority

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("options_pipeline_example")

# Initialize rich console for pretty output
console = Console()

def main():
    """
    Main function demonstrating the options data pipeline integration.
    """
    console.print(Panel.fit(
        "Options Data Pipeline Integration Example",
        style="bold green"
    ))

    # Step 1: Load configuration
    try:
        config_path = "../config/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        console.print("[green]✓[/green] Successfully loaded configuration")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[red]✗[/red] Error loading configuration: {str(e)}")
        console.print("Using default configuration instead")
        config = {}

    # Step 2: Initialize the MultiAssetAdapter
    try:
        adapter = MultiAssetAdapter(
            account_id="demo_account",
            config=config
        )
        console.print("[green]✓[/green] Successfully initialized MultiAssetAdapter")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing MultiAssetAdapter: {str(e)}")
        console.print("Exiting due to adapter initialization failure")
        return

    # Step 3: Set up options data integration
    try:
        options_data = OptionsMarketData(
            cache_dir="../data/options_cache", 
            enable_local_cache=True,
            cache_ttl=3600  # 1 hour cache
        )
        console.print("[green]✓[/green] Successfully initialized OptionsMarketData")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing OptionsMarketData: {str(e)}")
        options_data = None
    
    # Step 4: Register additional data sources
    if options_data:
        try:
            # Register a mock data source for demonstration/fallback
            options_data.register_mock_data_source(
                source_name="demo_source",
                priority=DataSourcePriority.FALLBACK
            )
            console.print("[green]✓[/green] Registered mock data source")
            
            # Integration with MultiAssetAdapter
            adapter.register_options_data_provider(options_data)
            console.print("[green]✓[/green] Integrated options data with MultiAssetAdapter")
        except Exception as e:
            console.print(f"[red]✗[/red] Error registering data sources: {str(e)}")
    
    # Step 5: Fetch options chain data
    symbol = "AAPL"
    console.print(f"\nFetching options chain for [bold]{symbol}[/bold]...")
    
    try:
        options_chain = adapter.get_options_chain(symbol)
        if options_chain and "_is_mock" not in options_chain:
            display_options_chain_summary(options_chain, symbol)
        else:
            console.print("[yellow]Using mock options data as real data fetch failed[/yellow]")
            display_mock_options_data(symbol)
    except Exception as e:
        console.print(f"[red]✗[/red] Error fetching options chain: {str(e)}")
        display_mock_options_data(symbol)
    
    # Step 6: Get IV history
    console.print(f"\nFetching IV history for [bold]{symbol}[/bold] (last 30 days)...")
    
    try:
        today = datetime.now()
        start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        iv_history = adapter.get_implied_volatility_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if isinstance(iv_history, pd.DataFrame) and not iv_history.empty:
            display_iv_history(iv_history, symbol)
        else:
            console.print("[yellow]No IV history data available, showing mock data[/yellow]")
            display_mock_iv_history(symbol, start_date, end_date)
    except Exception as e:
        console.print(f"[red]✗[/red] Error fetching IV history: {str(e)}")
        display_mock_iv_history(symbol, start_date, end_date)
    
    # Step 7: Display data quality metrics
    console.print("\nData Quality Metrics:")
    display_data_quality_metrics(symbol, adapter)


def display_options_chain_summary(options_chain, symbol):
    """Display a summary of the options chain data."""
    if not options_chain or "options" not in options_chain:
        console.print("[yellow]Empty options chain data[/yellow]")
        return
    
    options = options_chain.get("options", {}).get("option", [])
    if not options:
        console.print("[yellow]No options contracts in data[/yellow]")
        return
    
    # Get summary info
    summary = options_chain.get("options", {}).get("summary", {})
    underlying_price = summary.get("underlying_price", "N/A")
    
    # Get unique expirations and strikes
    expirations = set()
    strikes = set()
    call_count = 0
    put_count = 0
    
    for option in options:
        exp = option.get("expiration", "")
        strike = option.get("strike", 0)
        opt_type = option.get("type", "").lower()
        
        if exp:
            expirations.add(exp)
        if strike:
            strikes.add(strike)
        
        if opt_type == "call":
            call_count += 1
        elif opt_type == "put":
            put_count += 1
    
    # Create a table for display
    table = Table(title=f"Options Chain Summary for {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Underlying Price", f"${underlying_price}")
    table.add_row("Total Contracts", str(len(options)))
    table.add_row("Calls / Puts", f"{call_count} / {put_count}")
    table.add_row("Expirations", str(len(expirations)))
    table.add_row("Strike Prices", str(len(strikes)))
    
    # Add some example contracts if available
    if options:
        table.add_section()
        table.add_row("Sample Contracts:", "")
        
        for i, option in enumerate(options[:3]):
            symbol = option.get("symbol", "N/A")
            strike = option.get("strike", "N/A")
            exp = option.get("expiration", "N/A")
            opt_type = option.get("type", "N/A").upper()
            last = option.get("last", "N/A")
            iv = option.get("greeks", {}).get("iv", "N/A")
            
            table.add_row(
                f"Contract {i+1}",
                f"{symbol} - {exp} {strike} {opt_type} - Last: {last} - IV: {iv}%"
            )
    
    console.print(table)


def display_mock_options_data(symbol):
    """Display mock options data for demonstration."""
    console.print(Panel.fit(
        f"[bold]Mock Options Data for {symbol}[/bold]\n\n"
        f"This is simulated data for demonstration purposes.\n"
        f"In a production environment, real options data would be fetched from market data providers.",
        style="yellow"
    ))
    
    # Create a mock table
    table = Table(title=f"Mock Options Chain for {symbol}")
    table.add_column("Expiration", style="cyan")
    table.add_column("Strike", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Last", style="yellow")
    table.add_column("IV%", style="red")
    
    # Mock data rows
    expirations = [
        (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
        (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
        (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    ]
    
    strikes = [180, 185, 190, 195, 200]
    
    # Generate some mock rows
    for exp in expirations[:2]:
        for strike in strikes[:3]:
            # Call
            table.add_row(
                exp, 
                str(strike), 
                "CALL",
                f"${np.random.uniform(5, 15):.2f}",
                f"{np.random.uniform(20, 40):.1f}"
            )
            # Put
            table.add_row(
                exp, 
                str(strike), 
                "PUT",
                f"${np.random.uniform(4, 12):.2f}",
                f"{np.random.uniform(25, 45):.1f}"
            )
    
    console.print(table)


def display_iv_history(iv_history, symbol):
    """Display the implied volatility history."""
    if iv_history is None or iv_history.empty:
        console.print("[yellow]No IV history data available[/yellow]")
        return
    
    # Create a table summary
    table = Table(title=f"IV History Summary for {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    iv_mean = iv_history['iv'].mean() if 'iv' in iv_history.columns else 0
    iv_std = iv_history['iv'].std() if 'iv' in iv_history.columns else 0
    iv_min = iv_history['iv'].min() if 'iv' in iv_history.columns else 0
    iv_max = iv_history['iv'].max() if 'iv' in iv_history.columns else 0
    iv_current = iv_history['iv'].iloc[-1] if 'iv' in iv_history.columns and len(iv_history) > 0 else 0
    
    table.add_row("Period", f"{iv_history['date'].min()} to {iv_history['date'].max()}")
    table.add_row("Data Points", str(len(iv_history)))
    table.add_row("Current IV", f"{iv_current:.2f}%")
    table.add_row("Mean IV", f"{iv_mean:.2f}%")
    table.add_row("Min / Max IV", f"{iv_min:.2f}% / {iv_max:.2f}%")
    table.add_row("IV Std Dev", f"{iv_std:.2f}%")
    
    if 'iv_percentile' in iv_history.columns and len(iv_history) > 0:
        iv_percentile = iv_history['iv_percentile'].iloc[-1]
        table.add_row("Current IV Percentile", f"{iv_percentile:.1f}")
    
    console.print(table)
    
    # Try to plot the IV history if matplotlib is available
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(iv_history['date']), iv_history['iv'], marker='o', linestyle='-')
        plt.title(f'Implied Volatility History for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('IV (%)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot to a file
        output_dir = "../output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{symbol}_iv_history.png"
        plt.savefig(output_file)
        
        console.print(f"[green]✓[/green] IV history chart saved to {output_file}")
    except Exception as e:
        console.print(f"[yellow]Could not generate IV history chart: {str(e)}[/yellow]")


def display_mock_iv_history(symbol, start_date, end_date):
    """Display mock IV history data for demonstration."""
    console.print(Panel.fit(
        f"[bold]Mock IV History for {symbol}[/bold]\n\n"
        f"This is simulated IV history data for demonstration purposes.\n"
        f"In a production environment, real IV history would be fetched from market data providers.",
        style="yellow"
    ))
    
    # Generate a mock IV history dataframe
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start=start, end=end)
    
    # Generate some random IV values with a trend
    n_days = len(date_range)
    base_iv = 25.0
    trend = np.linspace(-5, 5, n_days)  # Linear trend from -5 to +5
    noise = np.random.normal(0, 2, n_days)  # Random noise
    
    iv_values = base_iv + trend + noise
    iv_values = np.maximum(10, iv_values)  # Ensure minimum IV of 10%
    
    # Create dataframe
    mock_df = pd.DataFrame({
        'date': date_range.strftime('%Y-%m-%d'),
        'iv': iv_values,
        'iv_percentile': np.random.uniform(0, 100, n_days)
    })
    
    # Display the mock data
    display_iv_history(mock_df, symbol)


def display_data_quality_metrics(symbol, adapter):
    """Display data quality metrics for the options data pipeline."""
    table = Table(title="Data Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Check options chain data completeness
    try:
        options_chain = adapter.get_options_chain(symbol)
        if options_chain and "options" in options_chain and options_chain["options"].get("option"):
            options_count = len(options_chain["options"].get("option", []))
            status = "[green]✓[/green]" if options_count > 0 else "[yellow]⚠[/yellow]"
            table.add_row("Options Chain Completeness", status, f"{options_count} contracts available")
        else:
            table.add_row("Options Chain Completeness", "[red]✗[/red]", "No options data available")
    except Exception:
        table.add_row("Options Chain Completeness", "[red]✗[/red]", "Error fetching options data")
    
    # Check IV history data quality
    try:
        today = datetime.now()
        start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        iv_history = adapter.get_implied_volatility_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if isinstance(iv_history, pd.DataFrame) and not iv_history.empty:
            # Check for gaps in the data
            expected_days = (today - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
            actual_days = len(iv_history)
            completeness = (actual_days / expected_days) * 100
            
            status = "[green]✓[/green]" if completeness > 95 else "[yellow]⚠[/yellow]"
            table.add_row("IV History Completeness", status, f"{completeness:.1f}% ({actual_days}/{expected_days} days)")
            
            # Check for extreme values
            if 'iv' in iv_history.columns:
                has_extremes = any(iv_history['iv'] > 100) or any(iv_history['iv'] < 5)
                status = "[yellow]⚠[/yellow]" if has_extremes else "[green]✓[/green]"
                table.add_row("IV Values Range", status, f"Min: {iv_history['iv'].min():.1f}, Max: {iv_history['iv'].max():.1f}")
        else:
            table.add_row("IV History Completeness", "[red]✗[/red]", "No IV history data available")
    except Exception as e:
        table.add_row("IV History Completeness", "[red]✗[/red]", f"Error: {str(e)}")
    
    # Check data source status
    table.add_row("Primary Data Source", "[yellow]⚠[/yellow]", "Using fallback/mock data for demonstration")
    
    # Check data freshness
    cache_status = "[green]✓[/green]" if hasattr(adapter, 'get_options_last_updated') else "[yellow]⚠[/yellow]"
    table.add_row("Data Freshness", cache_status, "Cache TTL set to 3600 seconds")
    
    console.print(table)


if __name__ == "__main__":
    main() 