#!/usr/bin/env python3
"""
Gradual Rollout Demo Script

This script demonstrates how to use the gradual rollout capabilities of the feature flag system
to selectively enable features for specific asset classes, time windows, or percentages of trades.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import random

# Add parent directory to path to import trading_bot modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from trading_bot.feature_flags import (
    get_feature_flag_service,
    FlagCategory,
    AssetClass,
)
from trading_bot.feature_flags.dashboard import FeatureFlagDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up Rich console
console = Console()

def print_section(title):
    """Print a section title."""
    console.print(f"\n[bold cyan]{'=' * 20} {title} {'=' * 20}[/bold cyan]\n")

def create_gradual_rollout_flags():
    """Create various feature flags with gradual rollout settings."""
    print_section("CREATING GRADUAL ROLLOUT FLAGS")
    
    # Get feature flag service
    service = get_feature_flag_service()
    
    # Create a percentage-based rollout flag
    success, message = service.create_flag(
        id="new_pricing_model",
        name="New Asset Pricing Model",
        description="Advanced pricing model for more accurate valuations",
        category=FlagCategory.STRATEGY,
        rollout_percentage=25,  # Start with 25% of trades
        requires_confirmation=True
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create an asset class-specific flag
    success, message = service.create_flag(
        id="crypto_risk_limits",
        name="Crypto Risk Limits",
        description="Additional risk limits for crypto trading",
        category=FlagCategory.RISK,
        applicable_asset_classes={"CRYPTO"},
        default=True
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create a flag with time window rule
    success, message = service.create_flag(
        id="market_hours_only",
        name="Market Hours Only Trading",
        description="Only execute trades during market hours",
        category=FlagCategory.EXECUTION,
        default=True
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Add a time window rule
    success, message = service.add_context_rule(
        flag_id="market_hours_only",
        rule_type="time_window",
        parameters={
            "start_time": "09:30",
            "end_time": "16:00"
        }
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create a flag with account value rule
    success, message = service.create_flag(
        id="advanced_strategies",
        name="Advanced Trading Strategies",
        description="Complex trading strategies for larger accounts",
        category=FlagCategory.STRATEGY
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Add account value rule
    success, message = service.add_context_rule(
        flag_id="advanced_strategies",
        rule_type="account_value",
        parameters={
            "min_value": 50000.0
        }
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create a flag with market condition rule
    success, message = service.create_flag(
        id="volatility_protection",
        name="Volatility Protection",
        description="Additional protection during high volatility",
        category=FlagCategory.RISK,
        default=True
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Add market condition rule
    success, message = service.add_context_rule(
        flag_id="volatility_protection",
        rule_type="market_condition",
        parameters={
            "conditions": ["HIGH_VOLATILITY", "EXTREME_VOLATILITY"]
        }
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create a flag with multiple rules and asset classes
    success, message = service.create_flag(
        id="complex_feature",
        name="Complex Feature",
        description="Feature with multiple conditions",
        category=FlagCategory.EXPERIMENTAL,
        applicable_asset_classes={"FOREX", "EQUITY"},
        rollout_percentage=50
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Add time window rule
    success, message = service.add_context_rule(
        flag_id="complex_feature",
        rule_type="time_window",
        parameters={
            "start_time": "08:00",
            "end_time": "17:00"
        }
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Add account value rule
    success, message = service.add_context_rule(
        flag_id="complex_feature",
        rule_type="account_value",
        parameters={
            "min_value": 10000.0,
            "max_value": 1000000.0
        }
    )
    console.print(f"[bold]{message}[/bold]")
    
    console.print("\n[green]All gradual rollout flags created successfully![/green]")

def simulate_flag_checks():
    """Simulate checking flags in various contexts."""
    print_section("SIMULATING FLAG CHECKS")
    
    service = get_feature_flag_service()
    
    # Sample contexts for simulation
    asset_classes = ["FOREX", "CRYPTO", "EQUITY", "FUTURES", "INDICES"]
    account_values = [5000, 25000, 75000, 200000, 1000000]
    market_conditions = ["NORMAL", "LOW_VOLATILITY", "HIGH_VOLATILITY", "EXTREME_VOLATILITY"]
    symbols = ["EURUSD", "BTCUSD", "AAPL", "ES", "SPX"]
    
    # Set up table to display results
    table = Table(title="Feature Flag Check Results")
    table.add_column("Flag ID", style="cyan")
    table.add_column("Context", style="yellow")
    table.add_column("Enabled", style="bold")
    table.add_column("Explanation")
    
    # Current time for time-based rules
    current_time = datetime.now()
    market_hours = current_time.replace(hour=12, minute=30)  # Middle of trading day
    after_hours = current_time.replace(hour=20, minute=0)    # After market close
    
    # Test percentage-based rollout
    for i in range(5):
        symbol = random.choice(symbols)
        context = {"symbol": symbol}
        enabled = service.is_enabled("new_pricing_model", context)
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        explanation = "Random selection based on consistent hashing of symbol"
        table.add_row("new_pricing_model", f"Symbol: {symbol}", status, explanation)
    
    # Test asset class rule
    for asset_class in ["CRYPTO", "FOREX", "EQUITY"]:
        context = {"asset_class": asset_class}
        enabled = service.is_enabled("crypto_risk_limits", context)
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        explanation = "Only enabled for CRYPTO asset class"
        table.add_row("crypto_risk_limits", f"Asset Class: {asset_class}", status, explanation)
    
    # Test time window rule
    for test_time in [market_hours, after_hours]:
        context = {"current_time": test_time}
        enabled = service.is_enabled("market_hours_only", context)
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        time_str = test_time.strftime("%H:%M")
        explanation = "Only enabled during market hours (09:30-16:00)"
        table.add_row("market_hours_only", f"Time: {time_str}", status, explanation)
    
    # Test account value rule
    for account_value in [10000, 60000]:
        context = {"account_value": account_value}
        enabled = service.is_enabled("advanced_strategies", context)
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        explanation = "Only enabled for accounts >= $50,000"
        table.add_row("advanced_strategies", f"Account: ${account_value}", status, explanation)
    
    # Test market condition rule
    for condition in ["NORMAL", "HIGH_VOLATILITY"]:
        context = {"market_condition": condition}
        enabled = service.is_enabled("volatility_protection", context)
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        explanation = "Only enabled during high/extreme volatility"
        table.add_row("volatility_protection", f"Market: {condition}", status, explanation)
    
    # Test complex rule combination
    contexts = [
        {
            "asset_class": "FOREX",
            "current_time": market_hours,
            "account_value": 50000,
            "symbol": "EURUSD"
        },
        {
            "asset_class": "CRYPTO",
            "current_time": market_hours,
            "account_value": 50000,
            "symbol": "BTCUSD"
        },
        {
            "asset_class": "EQUITY",
            "current_time": after_hours,
            "account_value": 50000,
            "symbol": "AAPL"
        }
    ]
    
    for context in contexts:
        enabled = service.is_enabled("complex_feature", context)
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        context_str = f"Asset: {context['asset_class']}, Time: {context['current_time'].strftime('%H:%M')}"
        explanation = "Requires FOREX/EQUITY, during business hours, account $10K-$1M, and 50% rollout"
        table.add_row("complex_feature", context_str, status, explanation)
    
    console.print(table)

def demonstrate_rollout_adjustments():
    """Demonstrate adjusting rollout percentages and monitoring impact."""
    print_section("GRADUAL ROLLOUT ADJUSTMENTS")
    
    service = get_feature_flag_service()
    
    console.print("[bold]Simulating gradual increase of rollout percentage[/bold]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Increasing rollout...", total=100)
        
        # Start at 25%
        current = 25
        progress.update(task, completed=current)
        
        # Show initial success rate
        success_rate = simulate_success_rate("new_pricing_model", 100)
        console.print(f"Rollout: {current}% -> Success rate: {success_rate}%")
        
        # Increase to 50%
        time.sleep(1)
        current = 50
        service.update_flag_rollout("new_pricing_model", current)
        progress.update(task, completed=current)
        
        # Show updated success rate
        success_rate = simulate_success_rate("new_pricing_model", 100)
        console.print(f"Rollout: {current}% -> Success rate: {success_rate}%")
        
        # Increase to 75%
        time.sleep(1)
        current = 75
        service.update_flag_rollout("new_pricing_model", current)
        progress.update(task, completed=current)
        
        # Show updated success rate
        success_rate = simulate_success_rate("new_pricing_model", 100)
        console.print(f"Rollout: {current}% -> Success rate: {success_rate}%")
        
        # Finally to 100%
        time.sleep(1)
        current = 100
        service.update_flag_rollout("new_pricing_model", current)
        progress.update(task, completed=current)
        
        # Show final success rate
        success_rate = simulate_success_rate("new_pricing_model", 100)
        console.print(f"Rollout: {current}% -> Success rate: {success_rate}%")
    
    console.print("\n[green]Rollout complete! Feature is now enabled for all users.[/green]")

def simulate_success_rate(flag_id, num_attempts):
    """Simulate success rate for a flag with the current rollout percentage.
    
    Args:
        flag_id: The flag ID to check
        num_attempts: Number of simulated attempts
        
    Returns:
        int: Percentage of successful checks
    """
    service = get_feature_flag_service()
    
    success_count = 0
    for i in range(num_attempts):
        # Create a unique context for each attempt
        context = {
            "symbol": f"SYM{i % 10}",
            "account_id": f"ACC{i % 20}"
        }
        
        if service.is_enabled(flag_id, context):
            success_count += 1
    
    return int(success_count / num_attempts * 100)

def display_rollout_dashboard():
    """Display the dashboard for gradual rollout flags."""
    print_section("GRADUAL ROLLOUT DASHBOARD")
    
    dashboard = FeatureFlagDashboard(console)
    dashboard.display_gradual_rollout_dashboard()
    
    # Display each flag in detail
    service = get_feature_flag_service()
    gradual_flags = [
        flag for flag in service.list_flags() 
        if flag.rollout_percentage < 100 or 
           "ALL" not in flag.applicable_asset_classes or 
           flag.context_rules
    ]
    
    for flag in gradual_flags[:2]:  # Show just a couple for brevity
        print_section(f"FLAG DETAILS: {flag.id}")
        dashboard.display_flag_details(flag.id)

def main():
    """Run the gradual rollout demo."""
    # Ensure the data directory exists
    os.makedirs("data/feature_flags", exist_ok=True)
    
    try:
        console.print(Panel(
            "This demo shows how to use gradual rollout capabilities in the feature flag system.\n"
            "It demonstrates percentage-based rollouts, asset class targeting, time windows,\n"
            "account value rules, and market condition filtering.",
            title="Gradual Rollout Demo",
            border_style="cyan"
        ))
        
        # Create flags with gradual rollout settings
        create_gradual_rollout_flags()
        
        # Simulate checking flags in different contexts
        simulate_flag_checks()
        
        # Demonstrate adjusting rollout percentages
        demonstrate_rollout_adjustments()
        
        # Display the dashboard
        display_rollout_dashboard()
        
        print_section("DEMO COMPLETE")
        
    except KeyboardInterrupt:
        console.print("[yellow]Demo stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in demo: {e}[/red]")
        logger.exception("Error in gradual rollout demo")
    finally:
        # Clean up
        service = get_feature_flag_service()
        service.cleanup()

if __name__ == "__main__":
    main() 