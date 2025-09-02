#!/usr/bin/env python3
"""
Feature Flags Demo Script

This script demonstrates how to use the feature flag system to selectively
enable/disable features in the trading bot without requiring a full deployment.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import trading_bot modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel

from trading_bot.feature_flags import (
    get_feature_flag_service,
    FlagCategory,
    FlagChangeEvent
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

def flag_change_callback(event: FlagChangeEvent):
    """Callback for flag changes."""
    state = "ENABLED" if event.enabled else "DISABLED"
    console.print(f"[yellow]FLAG CHANGED:[/yellow] {event.flag_id} -> {state} by {event.changed_by}")
    if event.reason:
        console.print(f"  Reason: {event.reason}")

def demo_feature_management():
    """Demonstrate the feature flag management system."""
    print_section("FEATURE FLAG DEMO")
    
    # Get feature flag service
    service = get_feature_flag_service()
    dashboard = FeatureFlagDashboard(console)
    
    # Register callback for flag changes
    service.register_callback(flag_change_callback)
    
    # Create a basic demo strategy flag
    success, message = service.create_flag(
        id="demo_strategy",
        name="Demo Trading Strategy",
        description="A demonstration trading strategy for testing the feature flag system",
        category=FlagCategory.STRATEGY,
        requires_confirmation=True
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create another flag with rollback
    success, message = service.create_flag(
        id="market_volatility_protection",
        name="Market Volatility Protection",
        description="Enables additional risk checks during high market volatility",
        category=FlagCategory.RISK,
        default=True,
        rollback_after_seconds=300  # Auto-disable after 5 minutes
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Create a dependent flag
    success, message = service.create_flag(
        id="advanced_risk_metrics",
        name="Advanced Risk Metrics",
        description="Enables more CPU-intensive risk calculations",
        category=FlagCategory.RISK,
        dependent_flags={"risk_limits"}
    )
    console.print(f"[bold]{message}[/bold]")
    
    # Display flag summary
    print_section("FLAG SUMMARY")
    dashboard.display_flag_summary()
    
    # Display detailed flag info
    print_section("FLAG DETAILS")
    dashboard.display_flags_by_category()
    
    # Demonstrate enabling and disabling flags
    print_section("FLAG OPERATIONS")
    
    # Enable the demo strategy
    console.print("[bold]Enabling demo_strategy...[/bold]")
    success, message = service.set_flag(
        flag_id="demo_strategy",
        enabled=True,
        changed_by="admin",
        reason="Demo activated for testing"
    )
    console.print(f"[bold]{message}[/bold]")
    time.sleep(1)  # Pause to see the callback
    
    # Display flag details
    dashboard.display_flag_details("demo_strategy")
    
    # Disable a risk feature
    console.print("\n[bold]Disabling risk_limits...[/bold]")
    success, message = service.set_flag(
        flag_id="risk_limits",
        enabled=False,
        changed_by="admin",
        reason="Testing dependent flag behavior"
    )
    console.print(f"[bold]{message}[/bold]")
    time.sleep(1)  # Pause to see the callback
    
    # Should automatically disable dependent flags
    console.print("\n[bold]Checking advanced_risk_metrics (should be auto-disabled)...[/bold]")
    dashboard.display_flag_details("advanced_risk_metrics")
    
    # Show recent changes
    print_section("RECENT CHANGES")
    dashboard.display_recent_changes()
    
    # Save the flags
    console.print("\n[bold]Saving flags to disk...[/bold]")
    service.save()
    
    # Reset the demo strategy flag
    console.print("\n[bold]Resetting demo_strategy to default...[/bold]")
    success, message = service.reset_flag("demo_strategy")
    console.print(f"[bold]{message}[/bold]")
    
    print_section("FEATURE FLAG DEMO COMPLETE")

def use_feature_flags_in_code():
    """Demonstrate how to use feature flags in your code."""
    print_section("USING FEATURE FLAGS IN CODE")
    
    service = get_feature_flag_service()
    
    # Simple flag check
    if service.is_enabled("demo_strategy"):
        console.print("[green]Demo strategy is enabled[/green]")
        
        # Your feature code would go here
        console.print(Panel(
            "def execute_demo_strategy():\n"
            "    # This code only runs when the feature flag is enabled\n"
            "    logger.info('Executing demo strategy')\n"
            "    position = calculate_position()\n"
            "    return execute_trade(position)",
            title="Feature-Flagged Code Example",
            border_style="green"
        ))
    else:
        console.print("[red]Demo strategy is disabled[/red]")
    
    # Check for a system feature with risk implications
    if service.is_enabled("emergency_stop"):
        console.print("[yellow]⚠️ Emergency stop is active - trading is paused[/yellow]")
    else:
        console.print("[green]Trading is active - emergency stop is disabled[/green]")
    
    # Example of a feature with volatile market protection
    if service.is_enabled("market_volatility_protection"):
        console.print("[yellow]Market volatility protection is active[/yellow]")
        console.print("This flag will automatically disable after 5 minutes")
        
        # Show how this would be used in real code
        console.print(Panel(
            "def execute_order(order):\n"
            "    if service.is_enabled('market_volatility_protection'):\n"
            "        # Apply additional risk checks\n"
            "        if order.size > max_order_size * 0.5:  # 50% normal limit\n"
            "            logger.warning('Order size reduced due to volatility protection')\n"
            "            order.size = max_order_size * 0.5\n"
            "    \n"
            "    # Execute the order\n"
            "    return broker.place_order(order)",
            title="Market Volatility Protection Example",
            border_style="yellow"
        ))

def main():
    """Run the feature flag demo."""
    # Ensure the data directory exists
    os.makedirs("data/feature_flags", exist_ok=True)
    
    try:
        # Run the demo
        demo_feature_management()
        
        # Show how to use flags in code
        use_feature_flags_in_code()
        
    except KeyboardInterrupt:
        console.print("[yellow]Demo stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in demo: {e}[/red]")
        logger.exception("Error in feature flag demo")
    finally:
        # Clean up
        service = get_feature_flag_service()
        service.cleanup()

if __name__ == "__main__":
    main() 