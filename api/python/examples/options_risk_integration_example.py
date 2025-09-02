#!/usr/bin/env python3
"""
Options Risk Integration Example

This script demonstrates how to integrate the OptionsRiskManager with the
MultiAssetAdapter to manage risk for options positions and strategies.

The example shows:
1. Setting up the options risk manager with the multi-asset adapter
2. Analyzing risk for individual options positions
3. Managing portfolio Greeks and risk limits
4. Analyzing multi-leg option strategies
5. Handling expiration risk and position adjustments
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from trading_bot.adapters.multi_asset_adapter import MultiAssetAdapter
from trading_bot.options_risk_manager import OptionsRiskManager
from trading_bot.option_strategy_risk_manager import OptionStrategyRiskManager
from trading_bot.data.options_market_data import OptionsMarketData, DataSourcePriority

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("options_risk_example")

# Initialize rich console for pretty output
console = Console()

def main():
    """
    Main function demonstrating options risk management integration with MultiAssetAdapter.
    """
    console.print(Panel.fit(
        "Options Risk Management Integration Example",
        style="bold blue"
    ))

    # Step 1: Initialize the MultiAssetAdapter
    try:
        console.print("\n[bold]Step 1: Initializing MultiAssetAdapter...[/bold]")
        adapter = initialize_adapter()
        console.print("[green]✓[/green] Successfully initialized MultiAssetAdapter")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing adapter: {str(e)}")
        return

    # Step 2: Set up the OptionsRiskManager
    try:
        console.print("\n[bold]Step 2: Setting up OptionsRiskManager...[/bold]")
        risk_manager = initialize_risk_manager(adapter)
        console.print("[green]✓[/green] Successfully initialized OptionsRiskManager")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing risk manager: {str(e)}")
        return

    # Step 3: Set up the OptionStrategyRiskManager
    try:
        console.print("\n[bold]Step 3: Setting up OptionStrategyRiskManager...[/bold]")
        strategy_risk_manager = initialize_strategy_risk_manager(risk_manager)
        console.print("[green]✓[/green] Successfully initialized OptionStrategyRiskManager")
    except Exception as e:
        console.print(f"[red]✗[/red] Error initializing strategy risk manager: {str(e)}")
        return

    # Step 4: Analyze individual option positions
    console.print("\n[bold]Step 4: Analyzing Individual Option Positions...[/bold]")
    analyze_individual_options(risk_manager)

    # Step 5: Analyze portfolio Greeks
    console.print("\n[bold]Step 5: Analyzing Portfolio Greeks...[/bold]")
    analyze_portfolio_greeks(risk_manager)

    # Step 6: Analyze option strategies
    console.print("\n[bold]Step 6: Analyzing Option Strategies...[/bold]")
    analyze_option_strategies(strategy_risk_manager)

    # Step 7: Expiration risk management
    console.print("\n[bold]Step 7: Expiration Risk Management...[/bold]")
    analyze_expiration_risk(risk_manager)

    # Step 8: Stress testing
    console.print("\n[bold]Step 8: Stress Testing Option Positions...[/bold]")
    stress_test_options(risk_manager, strategy_risk_manager)

    console.print("\n[bold green]Options Risk Management Example Complete![/bold green]")


def initialize_adapter():
    """Initialize and configure the MultiAssetAdapter"""
    
    # Try to load configuration from file
    config_path = "../config/config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        console.print("[yellow]Warning: Configuration file not found or invalid. Using defaults.[/yellow]")
        config = {}
    
    # Initialize the adapter with demo settings
    adapter = MultiAssetAdapter(
        account_id="demo_account",
        config=config
    )
    
    # Set up options market data
    options_data = OptionsMarketData(
        cache_dir="../data/options_cache",
        enable_local_cache=True
    )
    
    # Register a mock data source for demo purposes
    options_data.register_mock_data_source(
        source_name="demo_source",
        priority=DataSourcePriority.PRIMARY
    )
    
    # Register options data with adapter
    adapter.register_options_data_provider(options_data)
    
    return adapter


def initialize_risk_manager(adapter):
    """Initialize and configure the OptionsRiskManager"""
    
    # Create the options risk manager
    risk_manager = OptionsRiskManager(
        adapter=adapter,
        config_path=None,  # Use default config
        max_position_size=100,
        max_portfolio_delta=1000,
        max_portfolio_gamma=5000,
        max_vega_exposure=10000
    )
    
    # Add some demo positions to the risk manager
    add_demo_positions(risk_manager)
    
    return risk_manager


def initialize_strategy_risk_manager(risk_manager):
    """Initialize and configure the OptionStrategyRiskManager"""
    
    # Create the strategy risk manager
    strategy_manager = OptionStrategyRiskManager(
        options_risk_manager=risk_manager,
        config=None  # Use default config
    )
    
    return strategy_manager


def add_demo_positions(risk_manager):
    """Add demo positions to the risk manager for analysis"""
    
    # Mock positions for demonstration
    positions = [
        # Apple long call
        {
            "symbol": "AAPL230616C180000",
            "underlying": "AAPL",
            "quantity": 10,
            "strike": 180,
            "expiration": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            "option_type": "call",
            "position_type": "long",
            "entry_price": 5.60,
            "current_price": 6.20,
            "greeks": {
                "delta": 0.45,
                "gamma": 0.05,
                "theta": -0.12,
                "vega": 0.15,
                "rho": 0.03
            }
        },
        # Microsoft long put
        {
            "symbol": "MSFT230616P320000",
            "underlying": "MSFT",
            "quantity": 5,
            "strike": 320,
            "expiration": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            "option_type": "put",
            "position_type": "long",
            "entry_price": 8.30,
            "current_price": 7.80,
            "greeks": {
                "delta": -0.38,
                "gamma": 0.04,
                "theta": -0.15,
                "vega": 0.18,
                "rho": -0.04
            }
        },
        # Tesla short call
        {
            "symbol": "TSLA230616C220000",
            "underlying": "TSLA",
            "quantity": -3,
            "strike": 220,
            "expiration": (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
            "option_type": "call",
            "position_type": "short",
            "entry_price": 4.50,
            "current_price": 3.80,
            "greeks": {
                "delta": 0.52,
                "gamma": 0.06,
                "theta": -0.22,
                "vega": 0.20,
                "rho": 0.05
            }
        }
    ]
    
    # Add positions to risk manager
    for position in positions:
        risk_manager.add_option_position(
            symbol=position["symbol"],
            quantity=position["quantity"],
            option_data=position
        )
    
    return positions


def analyze_individual_options(risk_manager):
    """Analyze risk metrics for individual option positions"""
    
    # Get the positions from the risk manager
    positions = risk_manager.get_all_options_positions()
    
    if not positions:
        console.print("[yellow]No option positions found for analysis[/yellow]")
        return
    
    # Create a table for option position analysis
    table = Table(title="Individual Option Position Analysis")
    table.add_column("Symbol", style="cyan")
    table.add_column("Position", style="green")
    table.add_column("Delta", style="yellow")
    table.add_column("Gamma", style="magenta")
    table.add_column("Theta", style="red")
    table.add_column("Vega", style="blue")
    table.add_column("PnL", style="green")
    table.add_column("Risk Score", style="yellow")
    
    # Analyze each position
    for symbol, position in positions.items():
        # Get position analytics
        analytics = risk_manager.analyze_option_position(symbol)
        
        # Check if we have valid analytics
        if not analytics:
            continue
            
        # Format position type and quantity
        position_type = position.get("position_type", "long")
        quantity = position.get("quantity", 0)
        position_desc = f"{position_type.upper()} {abs(quantity)}"
        
        # Get PnL
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        pnl = (current_price - entry_price) * quantity
        pnl_formatted = f"${pnl:.2f}"
        if pnl > 0:
            pnl_formatted = f"[green]{pnl_formatted}[/green]"
        elif pnl < 0:
            pnl_formatted = f"[red]{pnl_formatted}[/red]"
        
        # Get Greeks
        greeks = analytics.get("greeks", {})
        delta = greeks.get("delta", 0) * quantity
        gamma = greeks.get("gamma", 0) * quantity
        theta = greeks.get("theta", 0) * quantity
        vega = greeks.get("vega", 0) * quantity
        
        # Calculate risk score (simplified example)
        days_to_expiry = analytics.get("days_to_expiry", 30)
        if days_to_expiry <= 0:
            risk_score = "EXTREME"
        elif days_to_expiry < 7:
            risk_score = "HIGH"
        elif days_to_expiry < 14:
            risk_score = "MEDIUM"
        else:
            risk_score = "LOW"
        
        # Add row to table
        table.add_row(
            symbol,
            position_desc,
            f"{delta:.2f}",
            f"{gamma:.4f}",
            f"{theta:.2f}",
            f"{vega:.2f}",
            pnl_formatted,
            risk_score
        )
    
    console.print(table)


def analyze_portfolio_greeks(risk_manager):
    """Analyze portfolio-level Greeks and risk exposures"""
    
    # Get portfolio Greeks
    portfolio_greeks = risk_manager.get_portfolio_greeks()
    
    if not portfolio_greeks:
        console.print("[yellow]No portfolio Greeks available for analysis[/yellow]")
        return
    
    # Create a table for portfolio Greeks
    table = Table(title="Portfolio Greeks Analysis")
    table.add_column("Greek", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("% of Max", style="yellow")
    table.add_column("Status", style="bold")
    
    # Add Greeks to table
    for greek, value in portfolio_greeks.items():
        if greek == "delta":
            max_value = risk_manager.max_portfolio_delta
            pct = abs(value) / max_value * 100 if max_value > 0 else 0
            status = get_risk_status(pct)
            table.add_row(
                greek.capitalize(),
                f"{value:.2f}",
                f"{pct:.1f}%",
                status
            )
        elif greek == "gamma":
            max_value = risk_manager.max_portfolio_gamma
            pct = abs(value) / max_value * 100 if max_value > 0 else 0
            status = get_risk_status(pct)
            table.add_row(
                greek.capitalize(),
                f"{value:.4f}",
                f"{pct:.1f}%",
                status
            )
        elif greek == "theta":
            # No specific limit for theta
            table.add_row(
                greek.capitalize(),
                f"{value:.2f}",
                "N/A",
                "[yellow]MONITOR[/yellow]"
            )
        elif greek == "vega":
            max_value = risk_manager.max_vega_exposure
            pct = abs(value) / max_value * 100 if max_value > 0 else 0
            status = get_risk_status(pct)
            table.add_row(
                greek.capitalize(),
                f"{value:.2f}",
                f"{pct:.1f}%",
                status
            )
    
    console.print(table)
    
    # Display Greek exposures by underlying
    console.print("\n[bold]Greek Exposures by Underlying:[/bold]")
    underlying_greeks = risk_manager.get_greeks_by_underlying()
    
    if underlying_greeks:
        table2 = Table(title="Greek Exposures by Underlying")
        table2.add_column("Underlying", style="cyan")
        table2.add_column("Delta", style="green")
        table2.add_column("Gamma", style="magenta")
        table2.add_column("Theta", style="red")
        table2.add_column("Vega", style="blue")
        
        for underlying, greeks in underlying_greeks.items():
            table2.add_row(
                underlying,
                f"{greeks.get('delta', 0):.2f}",
                f"{greeks.get('gamma', 0):.4f}",
                f"{greeks.get('theta', 0):.2f}",
                f"{greeks.get('vega', 0):.2f}"
            )
        
        console.print(table2)


def analyze_option_strategies(strategy_risk_manager):
    """Analyze multi-leg option strategies"""
    
    # Define some example strategies
    strategies = [
        # Vertical spread
        {
            "underlying": "AAPL",
            "legs": [
                {
                    "option_type": "call",
                    "strike": 180,
                    "expiration": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    "position": "long",
                    "quantity": 1,
                    "price": 5.60
                },
                {
                    "option_type": "call",
                    "strike": 190,
                    "expiration": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    "position": "short",
                    "quantity": 1,
                    "price": 2.10
                }
            ],
            "strategy_type": "vertical_spread"
        },
        # Iron condor
        {
            "underlying": "SPY",
            "legs": [
                {
                    "option_type": "put",
                    "strike": 400,
                    "expiration": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
                    "position": "short",
                    "quantity": 1,
                    "price": 3.50
                },
                {
                    "option_type": "put",
                    "strike": 390,
                    "expiration": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
                    "position": "long",
                    "quantity": 1,
                    "price": 2.20
                },
                {
                    "option_type": "call",
                    "strike": 430,
                    "expiration": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
                    "position": "short",
                    "quantity": 1,
                    "price": 3.80
                },
                {
                    "option_type": "call",
                    "strike": 440,
                    "expiration": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
                    "position": "long",
                    "quantity": 1,
                    "price": 2.30
                }
            ],
            "strategy_type": "iron_condor"
        }
    ]
    
    # Analyze each strategy
    for i, strategy in enumerate(strategies):
        strategy_id = strategy_risk_manager.add_option_strategy(
            underlying=strategy["underlying"],
            legs=strategy["legs"],
            strategy_type=strategy.get("strategy_type")
        )
        
        # Get strategy analysis
        analysis = strategy_risk_manager.analyze_strategy_risk(strategy_id)
        
        # Display strategy analysis
        display_strategy_analysis(strategy, analysis, i+1)


def display_strategy_analysis(strategy, analysis, index):
    """Display analysis for an option strategy"""
    
    strategy_type = strategy.get("strategy_type", "custom").replace("_", " ").title()
    
    console.print(f"\n[bold cyan]Strategy {index}: {strategy['underlying']} {strategy_type}[/bold cyan]")
    
    # Create a table for strategy metrics
    table = Table(title=f"{strategy['underlying']} {strategy_type} Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add key metrics to table
    if analysis:
        # Max gain/loss
        max_gain = analysis.get("max_gain", "Unknown")
        max_loss = analysis.get("max_loss", "Unknown")
        
        if isinstance(max_gain, (int, float, Decimal)):
            table.add_row("Max Gain", f"${max_gain:.2f}")
        else:
            table.add_row("Max Gain", str(max_gain))
            
        if isinstance(max_loss, (int, float, Decimal)):
            table.add_row("Max Loss", f"${max_loss:.2f}")
        else:
            table.add_row("Max Loss", str(max_loss))
        
        # Risk/reward ratio
        if isinstance(max_gain, (int, float, Decimal)) and isinstance(max_loss, (int, float, Decimal)) and max_loss != 0:
            risk_reward = abs(max_gain / max_loss) if max_loss != 0 else float('inf')
            table.add_row("Risk/Reward Ratio", f"{risk_reward:.2f}")
        
        # Breakeven points
        breakevens = analysis.get("breakevens", [])
        if breakevens:
            be_points = ", ".join([f"${be:.2f}" for be in breakevens])
            table.add_row("Breakeven Points", be_points)
        
        # Probability of profit
        pop = analysis.get("probability_of_profit")
        if pop:
            table.add_row("Probability of Profit", f"{pop:.1f}%")
        
        # Net Greeks
        net_greeks = analysis.get("net_greeks", {})
        if net_greeks:
            table.add_row("Net Delta", f"{net_greeks.get('delta', 0):.2f}")
            table.add_row("Net Gamma", f"{net_greeks.get('gamma', 0):.4f}")
            table.add_row("Net Theta", f"{net_greeks.get('theta', 0):.2f}")
            table.add_row("Net Vega", f"{net_greeks.get('vega', 0):.2f}")
        
        # Days to expiration
        dte = analysis.get("days_to_expiration", "Unknown")
        table.add_row("Days to Expiration", str(dte))
        
        console.print(table)
        
        # Display strategy recommendation if available
        recommendation = analysis.get("recommendation", "")
        if recommendation:
            console.print(f"[bold]Recommendation:[/bold] {recommendation}")
    else:
        console.print("[yellow]No analysis available for this strategy[/yellow]")


def analyze_expiration_risk(risk_manager):
    """Analyze expiration risk for options positions"""
    
    # Get expiration data
    expiration_data = risk_manager.get_expiration_risk_analysis()
    
    if not expiration_data:
        console.print("[yellow]No expiration risk data available[/yellow]")
        return
    
    # Create a table for expiration risk
    table = Table(title="Options Expiration Risk Analysis")
    table.add_column("Period", style="cyan")
    table.add_column("Contracts", style="green")
    table.add_column("Notional Value", style="yellow")
    table.add_column("Net Theta", style="red")
    table.add_column("Risk Level", style="bold")
    
    # Categorize by expiration period
    periods = {
        "This Week": [],
        "Next Week": [],
        "This Month": [],
        "Next Month": [],
        "Later": []
    }
    
    current_date = datetime.now().date()
    
    # Categorize positions
    for symbol, data in expiration_data.items():
        try:
            expiry = datetime.strptime(data.get("expiration", ""), "%Y-%m-%d").date()
            days_to_expiry = (expiry - current_date).days
            
            if days_to_expiry <= 7:
                periods["This Week"].append(data)
            elif days_to_expiry <= 14:
                periods["Next Week"].append(data)
            elif days_to_expiry <= 30:
                periods["This Month"].append(data)
            elif days_to_expiry <= 60:
                periods["Next Month"].append(data)
            else:
                periods["Later"].append(data)
        except (ValueError, TypeError):
            # Skip positions with invalid expiration dates
            continue
    
    # Process each period
    for period_name, positions in periods.items():
        if not positions:
            continue
            
        # Calculate metrics for this period
        contract_count = len(positions)
        notional_value = sum(p.get("notional_value", 0) for p in positions)
        net_theta = sum(p.get("theta", 0) for p in positions)
        
        # Determine risk level based on days to expiry and theta
        if period_name == "This Week":
            risk_level = "[red]HIGH[/red]"
        elif period_name == "Next Week":
            risk_level = "[yellow]MEDIUM[/yellow]"
        else:
            risk_level = "[green]LOW[/green]"
        
        # Add to table
        table.add_row(
            period_name,
            str(contract_count),
            f"${notional_value:.2f}",
            f"${net_theta:.2f}/day",
            risk_level
        )
    
    console.print(table)
    
    # Show positions expiring this week
    this_week = periods["This Week"]
    if this_week:
        console.print("\n[bold red]Positions Expiring This Week:[/bold red]")
        
        expiring_table = Table()
        expiring_table.add_column("Symbol", style="cyan")
        expiring_table.add_column("Type", style="green")
        expiring_table.add_column("Strike", style="yellow")
        expiring_table.add_column("Expiration", style="red")
        expiring_table.add_column("Status", style="bold")
        
        for position in this_week:
            symbol = position.get("symbol", "")
            option_type = position.get("option_type", "").upper()
            strike = position.get("strike", 0)
            expiration = position.get("expiration", "")
            
            # Determine ITM/OTM/ATM status
            underlying_price = position.get("underlying_price", 0)
            if option_type == "CALL":
                if underlying_price > strike:
                    status = "[green]ITM[/green]"
                elif underlying_price == strike:
                    status = "[yellow]ATM[/yellow]"
                else:
                    status = "[red]OTM[/red]"
            else:  # PUT
                if underlying_price < strike:
                    status = "[green]ITM[/green]"
                elif underlying_price == strike:
                    status = "[yellow]ATM[/yellow]"
                else:
                    status = "[red]OTM[/red]"
            
            expiring_table.add_row(
                symbol,
                option_type,
                f"${strike:.2f}",
                expiration,
                status
            )
        
        console.print(expiring_table)


def stress_test_options(risk_manager, strategy_risk_manager):
    """Perform stress testing on option positions"""
    
    # Define stress scenarios
    scenarios = [
        {"name": "Market Down 10%", "price_change": -0.10, "volatility_change": 0.30},
        {"name": "Market Down 5%", "price_change": -0.05, "volatility_change": 0.15},
        {"name": "Market Up 5%", "price_change": 0.05, "volatility_change": -0.10},
        {"name": "Market Up 10%", "price_change": 0.10, "volatility_change": -0.15},
        {"name": "Volatility Spike", "price_change": -0.03, "volatility_change": 0.50},
        {"name": "Volatility Crush", "price_change": 0.02, "volatility_change": -0.30}
    ]
    
    # Run stress tests
    stress_results = {}
    
    for scenario in track(scenarios, description="Running stress tests..."):
        # Run portfolio-level stress test
        result = risk_manager.run_stress_test(
            price_change=scenario["price_change"],
            volatility_change=scenario["volatility_change"],
            interest_rate_change=0.001  # 10 basis points
        )
        
        stress_results[scenario["name"]] = result
    
    # Display stress test results
    display_stress_test_results(stress_results)


def display_stress_test_results(stress_results):
    """Display the results of stress testing"""
    
    if not stress_results:
        console.print("[yellow]No stress test results available[/yellow]")
        return
    
    # Create a table for stress test results
    table = Table(title="Options Portfolio Stress Test Results")
    table.add_column("Scenario", style="cyan")
    table.add_column("P&L Impact", style="green")
    table.add_column("Delta Change", style="yellow")
    table.add_column("Gamma Impact", style="magenta")
    table.add_column("Vega Impact", style="blue")
    table.add_column("Risk Assessment", style="bold")
    
    # Add data for each scenario
    for scenario_name, result in stress_results.items():
        if not result:
            continue
            
        # Extract data
        pnl_impact = result.get("pnl_impact", 0)
        delta_change = result.get("delta_change", 0)
        gamma_impact = result.get("gamma_impact", 0)
        vega_impact = result.get("vega_impact", 0)
        
        # Format P&L
        if pnl_impact >= 0:
            pnl_formatted = f"[green]+${pnl_impact:.2f}[/green]"
        else:
            pnl_formatted = f"[red]-${abs(pnl_impact):.2f}[/red]"
        
        # Determine risk assessment based on P&L impact
        account_size = 100000  # Example account size
        pnl_pct = pnl_impact / account_size * 100
        
        if pnl_pct <= -15:
            risk = "[red]SEVERE[/red]"
        elif pnl_pct <= -10:
            risk = "[red]HIGH[/red]"
        elif pnl_pct <= -5:
            risk = "[yellow]MODERATE[/yellow]"
        elif pnl_pct <= -2:
            risk = "[green]LOW[/green]"
        else:
            risk = "[green]MINIMAL[/green]"
        
        # Add to table
        table.add_row(
            scenario_name,
            pnl_formatted,
            f"{delta_change:+.2f}",
            f"{gamma_impact:+.4f}",
            f"{vega_impact:+.2f}",
            risk
        )
    
    console.print(table)


def get_risk_status(percentage):
    """Get a colored status based on percentage of max"""
    if percentage >= 90:
        return "[red]CRITICAL[/red]"
    elif percentage >= 75:
        return "[red]HIGH[/red]"
    elif percentage >= 50:
        return "[yellow]MEDIUM[/yellow]"
    else:
        return "[green]LOW[/green]"


if __name__ == "__main__":
    main() 