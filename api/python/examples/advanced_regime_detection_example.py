#!/usr/bin/env python3
"""
Advanced Market Regime Detection Example

This example demonstrates how to use the AdvancedRegimeDetector to:
1. Analyze multiple timeframe trends and detect conflicts
2. Classify volatility-based regimes (low/normal/high/extreme)
3. Detect correlation-based regimes across asset classes
4. Analyze sector rotation for equity trading

The example uses historical price data for several ETFs representing different asset classes and sectors.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Adjust path to import from trading_bot package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_bot.backtesting.advanced_regime_detector import (
    AdvancedRegimeDetector, 
    VolatilityRegime,
    CorrelationRegime,
    SectorRotationPhase
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console for better output formatting
console = Console()

def load_sample_market_data():
    """
    Load or generate sample market data for multiple timeframes.
    
    Returns:
        Dictionary mapping timeframes to price DataFrames
    """
    # Check if we have real data in the expected location
    data_path = Path(__file__).parent.parent.parent / 'data' / 'market'
    
    if data_path.exists():
        # Try to load real data if available
        daily_file = data_path / 'daily_prices.csv'
        weekly_file = data_path / 'weekly_prices.csv'
        monthly_file = data_path / 'monthly_prices.csv'
        
        data_by_timeframe = {}
        
        # Load daily data if available
        if daily_file.exists():
            daily_data = pd.read_csv(daily_file)
            daily_data['date'] = pd.to_datetime(daily_data['date'])
            data_by_timeframe['daily'] = daily_data
        
        # Load weekly data if available
        if weekly_file.exists():
            weekly_data = pd.read_csv(weekly_file)
            weekly_data['date'] = pd.to_datetime(weekly_data['date'])
            data_by_timeframe['weekly'] = weekly_data
        
        # Load monthly data if available
        if monthly_file.exists():
            monthly_data = pd.read_csv(monthly_file)
            monthly_data['date'] = pd.to_datetime(monthly_data['date'])
            data_by_timeframe['monthly'] = monthly_data
        
        if data_by_timeframe:
            logger.info(f"Loaded actual market data from: {data_path}")
            return data_by_timeframe
    
    # If no real data, generate synthetic data
    logger.info("Generating synthetic market data for demonstration")
    
    # List of symbols to include in synthetic data
    symbols = [
        # Major indices
        'SPY',     # S&P 500
        'QQQ',     # Nasdaq 100
        'IWM',     # Russell 2000
        
        # Sector ETFs
        'XLK',     # Technology
        'XLF',     # Financials 
        'XLV',     # Healthcare
        'XLE',     # Energy
        'XLY',     # Consumer Discretionary
        'XLP',     # Consumer Staples
        'XLI',     # Industrials
        'XLB',     # Materials
        'XLU',     # Utilities
        'XLRE',    # Real Estate
        
        # Other asset classes
        'GLD',     # Gold
        'SLV',     # Silver
        'TLT',     # 20+ Year Treasury
        'HYG',     # High Yield Corporate Bonds
        'UUP'      # US Dollar
    ]
    
    # Generate synthetic price data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365*2)  # 2 years of history
    
    # Create date ranges for different timeframes
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='BM')
    
    # Initialize data containers
    daily_data = []
    weekly_data = []
    monthly_data = []
    
    # Generate data for each symbol with realistic correlations
    np.random.seed(42)  # For reproducibility
    
    # Base S&P 500 price series (other assets will correlate to this)
    spy_daily_returns = np.random.normal(0.0005, 0.01, size=len(daily_dates))
    spy_daily_prices = 100 * np.cumprod(1 + spy_daily_returns)
    
    # Correlation matrix for asset returns
    correlations = {
        'QQQ': 0.9,        # Tech-heavy, high correlation to S&P
        'IWM': 0.8,        # Small caps, strong but not perfect correlation
        'XLK': 0.85,       # Tech sector, highly correlated to S&P
        'XLF': 0.75,       # Financials, moderately high correlation
        'XLV': 0.6,        # Healthcare, moderate correlation
        'XLE': 0.4,        # Energy, lower correlation
        'XLY': 0.8,        # Consumer Discretionary, high correlation
        'XLP': 0.3,        # Consumer Staples, defensive sector, low correlation
        'XLI': 0.7,        # Industrials, moderately high correlation
        'XLB': 0.6,        # Materials, moderate correlation
        'XLU': 0.2,        # Utilities, defensive sector, low correlation
        'XLRE': 0.5,       # Real Estate, moderate correlation
        'GLD': -0.2,       # Gold, slightly negative correlation
        'SLV': -0.1,       # Silver, slightly negative correlation
        'TLT': -0.3,       # Long-term Treasuries, negative correlation
        'HYG': 0.5,        # High Yield Bonds, moderate correlation
        'UUP': -0.2        # US Dollar, slightly negative correlation
    }
    
    # Generate price data for each symbol
    for symbol in symbols:
        # For SPY, we already generated the base series
        if symbol == 'SPY':
            price_series = spy_daily_prices
        else:
            # Generate correlated returns
            correlation = correlations.get(symbol, 0.5)
            independent_returns = np.random.normal(0.0003, 0.012, size=len(daily_dates))
            correlated_returns = (correlation * spy_daily_returns + 
                                 np.sqrt(1 - correlation**2) * independent_returns)
            
            # Add some autocorrelation
            for i in range(1, len(correlated_returns)):
                correlated_returns[i] = 0.1 * correlated_returns[i-1] + 0.9 * correlated_returns[i]
            
            # Convert returns to prices
            price_series = 100 * np.cumprod(1 + correlated_returns)
        
        # Create daily data entries
        for i, date in enumerate(daily_dates):
            daily_data.append({
                'date': date,
                'symbol': symbol,
                'close': price_series[i],
                'volume': np.random.randint(100000, 10000000)
            })
        
        # Create weekly data (use Friday's values)
        weekly_indices = [i for i, date in enumerate(daily_dates) if date.dayofweek == 4 and date in weekly_dates]
        for i in weekly_indices:
            if i < len(price_series):
                weekly_data.append({
                    'date': daily_dates[i],
                    'symbol': symbol,
                    'close': price_series[i],
                    'volume': np.random.randint(500000, 50000000)
                })
        
        # Create monthly data (use month-end values)
        monthly_indices = [i for i, date in enumerate(daily_dates) if date in monthly_dates]
        for i in monthly_indices:
            if i < len(price_series):
                monthly_data.append({
                    'date': daily_dates[i],
                    'symbol': symbol,
                    'close': price_series[i],
                    'volume': np.random.randint(2000000, 200000000)
                })
    
    # Convert to DataFrames
    daily_df = pd.DataFrame(daily_data)
    weekly_df = pd.DataFrame(weekly_data)
    monthly_df = pd.DataFrame(monthly_data)
    
    # Create a dictionary mapping timeframes to price DataFrames
    data_by_timeframe = {
        'daily': daily_df,
        'weekly': weekly_df,
        'monthly': monthly_df
    }
    
    return data_by_timeframe

def generate_vix_data(daily_data):
    """
    Generate synthetic VIX data based on S&P 500 returns.
    
    Args:
        daily_data: DataFrame with daily price data
        
    Returns:
        Series with synthetic VIX values
    """
    # Extract SPY data
    spy_data = daily_data[daily_data['symbol'] == 'SPY'].copy()
    spy_data = spy_data.sort_values('date')
    
    # Calculate daily returns
    spy_data['return'] = spy_data['close'].pct_change()
    
    # Calculate rolling volatility (20-day)
    spy_data['volatility'] = spy_data['return'].rolling(window=20).std() * np.sqrt(252)
    
    # Generate synthetic VIX (base level around 15, scaling factor of 100, add some noise)
    base_vix = 15
    scale_factor = 100
    
    vix_values = base_vix + scale_factor * spy_data['volatility']
    
    # Add some noise to make it less perfectly correlated
    noise = np.random.normal(0, 2, size=len(vix_values))
    vix_values = vix_values + noise
    
    # Ensure minimum VIX level
    vix_values = np.maximum(vix_values, 9)
    
    # Create VIX Series
    vix_series = pd.Series(vix_values.values, index=spy_data['date'])
    
    return vix_series

def run_advanced_regime_detection():
    """
    Run the advanced market regime detection example.
    """
    console.print(Panel.fit(
        "[bold green]Advanced Market Regime Detection Example[/bold green]\n"
        "Demonstrating multiple timeframe analysis, volatility regimes, correlation regimes, and sector rotation"
    ))
    
    # Step 1: Load market data
    console.print("[bold cyan]Step 1: Loading market data...[/bold cyan]")
    market_data = load_sample_market_data()
    
    # Print data summary
    for timeframe, data in market_data.items():
        symbol_count = len(data['symbol'].unique())
        date_range = f"{data['date'].min().date()} to {data['date'].max().date()}"
        console.print(f"  - {timeframe.capitalize()} data: {len(data)} rows, {symbol_count} symbols, {date_range}")
    
    # Generate synthetic VIX data
    vix_data = generate_vix_data(market_data['daily'])
    console.print(f"  - Generated synthetic VIX data with {len(vix_data)} values")
    
    # Step 2: Initialize the Advanced Regime Detector
    console.print("\n[bold cyan]Step 2: Initializing Advanced Regime Detector...[/bold cyan]")
    detector = AdvancedRegimeDetector(
        lookback_days=252,  # Use 1 year of trading days
        timeframes=['daily', 'weekly', 'monthly'],
        volatility_windows={'daily': 20, 'weekly': 10, 'monthly': 6},
        trend_windows={'daily': 50, 'weekly': 20, 'monthly': 12},
        correlation_window=30,
        num_regimes=4,
        regime_persistence=5,
        vix_data=vix_data
    )
    
    # Step 3: Load market data into the detector
    console.print("\n[bold cyan]Step 3: Loading market data into detector...[/bold cyan]")
    detector.load_market_data_multi_timeframe(
        price_data_by_timeframe=market_data,
        symbol_col='symbol',
        date_col='date',
        price_col='close',
        volume_col='volume',
        benchmark_symbol='SPY'
    )
    
    # Step 4: Run the full analysis
    console.print("\n[bold cyan]Step 4: Running full regime analysis...[/bold cyan]")
    analysis = detector.run_full_analysis()
    
    # Step A: Display primary regime information
    console.print("\n[bold cyan]Results: Primary Market Regime[/bold cyan]")
    
    if 'primary_regime' in analysis:
        regime_info = analysis['primary_regime']
        regime_table = Table(show_header=True, header_style="bold magenta")
        regime_table.add_column("Attribute")
        regime_table.add_column("Value")
        
        regime_table.add_row("Regime Number", str(regime_info.get('regime', 'Unknown')))
        regime_table.add_row("Regime Label", str(regime_info.get('label', 'Unknown')))
        regime_table.add_row("Duration (days)", str(regime_info.get('duration_days', 'Unknown')))
        
        # Add selected features
        features = regime_info.get('features', {})
        if features:
            regime_table.add_row("Volatility", f"{features.get('volatility', 0):.4f}")
            regime_table.add_row("Trend Strength", f"{features.get('trend_strength', 0):.4f}")
            regime_table.add_row("Avg Correlation", f"{features.get('avg_correlation', 0):.4f}")
        
        console.print(regime_table)
    else:
        console.print("[yellow]No primary regime information available[/yellow]")
    
    # Step B: Display multi-timeframe trend conflicts
    console.print("\n[bold cyan]Results: Multi-Timeframe Trend Analysis[/bold cyan]")
    
    if 'trend_conflicts' in analysis:
        conflict_info = analysis['trend_conflicts']
        conflict_status = conflict_info.get('status', 'unknown')
        
        status_color = {
            'aligned_bullish': 'green',
            'aligned_bearish': 'red',
            'strong_conflict': 'yellow',
            'indeterminate': 'white'
        }.get(conflict_status, 'white')
        
        console.print(f"Trend Conflict Status: [{status_color}]{conflict_status}[/{status_color}]")
        
        # Display regime by timeframe
        timeframe_table = Table(show_header=True, header_style="bold magenta")
        timeframe_table.add_column("Timeframe")
        timeframe_table.add_column("Regime")
        timeframe_table.add_column("Label")
        
        for timeframe, regime_data in conflict_info.get('timeframe_regimes', {}).items():
            regime_label = regime_data.get('label', 'Unknown')
            label_color = 'green' if 'bullish' in str(regime_label).lower() else 'red' if 'bearish' in str(regime_label).lower() else 'white'
            
            timeframe_table.add_row(
                timeframe.capitalize(),
                str(regime_data.get('regime', 'Unknown')),
                f"[{label_color}]{regime_label}[/{label_color}]"
            )
        
        console.print(timeframe_table)
    else:
        console.print("[yellow]No trend conflict information available[/yellow]")
    
    # Step C: Display volatility regime information
    console.print("\n[bold cyan]Results: Volatility Regime Analysis[/bold cyan]")
    
    if 'volatility_regime' in analysis:
        vol_info = analysis['volatility_regime']
        current_regime = vol_info.get('current')
        
        # Color coding based on volatility regime
        vol_color = {
            VolatilityRegime.LOW: 'green',
            VolatilityRegime.NORMAL: 'blue',
            VolatilityRegime.HIGH: 'yellow',
            VolatilityRegime.EXTREME: 'red'
        }.get(current_regime, 'white')
        
        console.print(f"Current Volatility Regime: [{vol_color}]{current_regime}[/{vol_color}]")
        
        # Display history summary
        history = vol_info.get('history', {})
        if history:
            vol_table = Table(show_header=True, header_style="bold magenta")
            vol_table.add_column("Regime")
            vol_table.add_column("Days")
            vol_table.add_column("Percentage")
            
            total_days = sum(history.values())
            
            for regime, days in history.items():
                regime_color = {
                    VolatilityRegime.LOW: 'green',
                    VolatilityRegime.NORMAL: 'blue',
                    VolatilityRegime.HIGH: 'yellow',
                    VolatilityRegime.EXTREME: 'red'
                }.get(regime, 'white')
                
                percentage = (days / total_days) * 100 if total_days > 0 else 0
                
                vol_table.add_row(
                    f"[{regime_color}]{regime}[/{regime_color}]",
                    str(days),
                    f"{percentage:.1f}%"
                )
            
            console.print(vol_table)
    else:
        console.print("[yellow]No volatility regime information available[/yellow]")
    
    # Step D: Display correlation regimes
    console.print("\n[bold cyan]Results: Correlation Regime Analysis[/bold cyan]")
    
    if 'correlation_regimes' in analysis and 'current' in analysis['correlation_regimes']:
        corr_regimes = analysis['correlation_regimes']['current']
        corr_values = analysis['correlation_regimes'].get('correlation_values', {})
        
        if corr_regimes:
            corr_table = Table(show_header=True, header_style="bold magenta")
            corr_table.add_column("Asset Class")
            corr_table.add_column("Correlation Regime")
            corr_table.add_column("Correlation Value")
            
            for asset, regime in corr_regimes.items():
                # Color coding based on correlation regime
                regime_color = {
                    CorrelationRegime.HIGH_POSITIVE: 'red',
                    CorrelationRegime.MODERATE_POSITIVE: 'yellow',
                    CorrelationRegime.UNCORRELATED: 'green',
                    CorrelationRegime.MODERATE_NEGATIVE: 'cyan',
                    CorrelationRegime.HIGH_NEGATIVE: 'blue'
                }.get(regime, 'white')
                
                corr_value = corr_values.get(asset, 0)
                
                corr_table.add_row(
                    asset,
                    f"[{regime_color}]{regime}[/{regime_color}]",
                    f"{corr_value:.4f}"
                )
            
            console.print(corr_table)
        else:
            console.print("[yellow]No correlation regimes detected[/yellow]")
    else:
        console.print("[yellow]No correlation regime information available[/yellow]")
    
    # Step E: Display sector rotation analysis
    console.print("\n[bold cyan]Results: Sector Rotation Analysis[/bold cyan]")
    
    if 'sector_rotation' in analysis:
        rotation_info = analysis['sector_rotation']
        current_phase = rotation_info.get('current_phase')
        top_sectors = rotation_info.get('top_sectors', [])
        bottom_sectors = rotation_info.get('bottom_sectors', [])
        
        if current_phase:
            phase_color = {
                SectorRotationPhase.EARLY_EXPANSION: 'green',
                SectorRotationPhase.LATE_EXPANSION: 'yellow',
                SectorRotationPhase.EARLY_CONTRACTION: 'red',
                SectorRotationPhase.LATE_CONTRACTION: 'magenta',
                SectorRotationPhase.RECOVERY: 'cyan'
            }.get(current_phase, 'white')
            
            console.print(f"Current Economic Phase: [{phase_color}]{current_phase}[/{phase_color}]")
            
            # Display top and bottom sectors
            sector_table = Table(show_header=True, header_style="bold magenta")
            sector_table.add_column("Category")
            sector_table.add_column("Sectors")
            
            sector_table.add_row(
                "[green]Top Performing[/green]",
                ", ".join(top_sectors) if top_sectors else "None detected"
            )
            
            sector_table.add_row(
                "[red]Bottom Performing[/red]",
                ", ".join(bottom_sectors) if bottom_sectors else "None detected"
            )
            
            console.print(sector_table)
            
            # Display sector dispersion
            dispersion = rotation_info.get('sector_dispersion')
            if dispersion is not None:
                console.print(f"Sector Performance Dispersion: {dispersion:.4f}")
        else:
            console.print("[yellow]No sector rotation phase detected[/yellow]")
    else:
        console.print("[yellow]No sector rotation information available[/yellow]")
    
    # Step F: Display actionable insights
    console.print("\n[bold cyan]Results: Actionable Insights[/bold cyan]")
    
    if 'actionable_insights' in analysis and analysis['actionable_insights']:
        insights = analysis['actionable_insights']
        
        insight_table = Table(show_header=True, header_style="bold magenta")
        insight_table.add_column("Type")
        insight_table.add_column("Description")
        insight_table.add_column("Recommendation")
        
        # Type colors
        type_colors = {
            'trend_conflict': 'yellow',
            'trend_alignment': 'green',
            'volatility': 'red',
            'correlation': 'blue',
            'sector_rotation': 'magenta'
        }
        
        for insight in insights:
            insight_type = insight.get('type', 'unknown')
            type_color = type_colors.get(insight_type, 'white')
            
            insight_table.add_row(
                f"[{type_color}]{insight_type.replace('_', ' ').title()}[/{type_color}]",
                insight.get('description', ''),
                insight.get('recommendation', '')
            )
        
        console.print(insight_table)
    else:
        console.print("[yellow]No actionable insights available[/yellow]")
    
    # Final summary
    console.print("\n[bold green]Advanced Market Regime Detection Analysis Complete[/bold green]")
    console.print("Use these insights to adapt your trading strategies to current market conditions")

if __name__ == "__main__":
    run_advanced_regime_detection() 