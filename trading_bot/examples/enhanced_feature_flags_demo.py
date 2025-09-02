#!/usr/bin/env python3
"""
Enhanced Feature Flags Demo

This script demonstrates the complete enhanced feature flag system with:
1. Metrics integration
2. A/B testing framework
3. Automated rollbacks
4. Dependency visualization

It shows how these components work together to provide a comprehensive
feature management system for a trading bot.
"""

import logging
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import trading_bot modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.progress import Progress

from trading_bot.feature_flags import (
    get_feature_flag_service,
    FlagCategory,
    FlagChangeEvent
)
from trading_bot.feature_flags.metrics import get_metrics_collector
from trading_bot.feature_flags.ab_testing import (
    get_ab_testing_service,
    ExperimentStatus,
    RollbackRule
)
from trading_bot.feature_flags.auto_rollback import (
    get_auto_rollback_service,
    ThresholdDirection,
    AlertSeverity
)
from trading_bot.feature_flags.visualizer import get_visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up Rich console
console = Console()

# Global metrics for simulation
simulated_metrics = {
    "pnl_daily": 0.0,
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "win_rate": 0.0,
    "trade_count": 0,
    "avg_trade_duration": 0.0,
    "volatility": 0.0,
    "transaction_costs": 0.0
}

def print_section(title):
    """Print a section title."""
    console.print(f"\n[bold cyan]{'=' * 20} {title} {'=' * 20}[/bold cyan]\n")

def get_performance_metrics() -> Dict[str, float]:
    """Callback to get current performance metrics.
    
    This simulates what would be a real metrics collector in production that 
    would fetch actual performance data from the trading system.
    
    Returns:
        Dict[str, float]: Current performance metrics
    """
    global simulated_metrics
    
    # In a real system, we would fetch actual metrics from the trading platform
    # For this demo, we'll generate simulated metrics with some randomness
    
    # Add some random changes to metrics
    simulated_metrics["pnl_daily"] += random.uniform(-2.0, 3.0)
    simulated_metrics["sharpe_ratio"] = max(0, simulated_metrics["sharpe_ratio"] + random.uniform(-0.1, 0.15))
    simulated_metrics["max_drawdown"] = min(0, simulated_metrics["max_drawdown"] + random.uniform(-0.5, 0.3))
    simulated_metrics["win_rate"] = max(0, min(1, simulated_metrics["win_rate"] + random.uniform(-0.05, 0.05)))
    simulated_metrics["trade_count"] += random.randint(-5, 10)
    simulated_metrics["avg_trade_duration"] += random.uniform(-5, 5)
    simulated_metrics["volatility"] = max(0, simulated_metrics["volatility"] + random.uniform(-0.2, 0.3))
    simulated_metrics["transaction_costs"] += random.uniform(-0.5, 1.0)
    
    # Ensure valid values
    simulated_metrics["trade_count"] = max(0, simulated_metrics["trade_count"])
    
    return dict(simulated_metrics)

def setup_feature_flags():
    """Set up feature flags for the demo."""
    print_section("SETTING UP FEATURE FLAGS")
    
    service = get_feature_flag_service()
    
    # Set up metrics collector with our performance metrics callback
    metrics_collector = get_metrics_collector(get_performance_metrics)
    
    # Create a set of feature flags with different characteristics
    flags_to_create = [
        {
            "id": "new_strategy_v1",
            "name": "New Trading Strategy v1",
            "description": "New mean-reversion strategy for high volatility markets",
            "category": FlagCategory.STRATEGY,
            "requires_confirmation": True,
            "dependent_flags": set()
        },
        {
            "id": "enhanced_risk_controls",
            "name": "Enhanced Risk Controls",
            "description": "Additional risk controls for volatile markets",
            "category": FlagCategory.RISK,
            "requires_confirmation": True,
            "default": True,
            "dependent_flags": set()
        },
        {
            "id": "alt_data_provider",
            "name": "Alternative Data Provider",
            "description": "Use alternative data provider for market data",
            "category": FlagCategory.DATA,
            "rollout_percentage": 50,
            "applicable_asset_classes": {"CRYPTO", "FOREX"}
        },
        {
            "id": "experimental_ml_signals",
            "name": "Experimental ML Signals",
            "description": "Use machine learning signals for entry/exit",
            "category": FlagCategory.EXPERIMENTAL,
            "rollback_after_seconds": 3600,  # Auto-disable after 1 hour
            "dependent_flags": {"enhanced_risk_controls"}
        },
        {
            "id": "order_optimization",
            "name": "Order Execution Optimization",
            "description": "Optimize order execution to reduce costs",
            "category": FlagCategory.EXECUTION,
            "applicable_asset_classes": {"EQUITY", "FUTURES"}
        }
    ]
    
    # Create the flags
    for flag_data in flags_to_create:
        flag_id = flag_data.pop("id")
        
        # Check if flag exists
        if service.get_flag(flag_id):
            console.print(f"Flag '{flag_id}' already exists, updating...")
            continue
        
        # Create flag
        success, message = service.create_flag(id=flag_id, **flag_data)
        console.print(f"[{'green' if success else 'red'}]{message}[/{'green' if success else 'red'}]")
    
    console.print("\n[green]Feature flags set up successfully![/green]")
    
    # Take initial performance snapshot
    metrics_collector.record_performance_snapshot()
    
    return service

def setup_ab_testing():
    """Set up A/B testing for the new strategy."""
    print_section("SETTING UP A/B TESTING")
    
    ab_service = get_ab_testing_service()
    
    # Create an experiment for the new strategy
    success, message, experiment = ab_service.create_experiment(
        name="New Strategy Evaluation",
        description="Testing the effectiveness of the new trading strategy",
        flag_id="new_strategy_v1",
        metrics=["pnl_daily", "sharpe_ratio", "max_drawdown", "win_rate"],
        created_by="demo_script",
        asset_classes={"FOREX", "CRYPTO"},
        control_group_size=50,  # 50% control, 50% treatment
        min_sample_size=30,
        duration_days=7,  # One week test
        auto_start=True
    )
    
    console.print(f"[{'green' if success else 'red'}]{message}[/{'green' if success else 'red'}]")
    
    if success:
        console.print(f"Created experiment with ID: {experiment.id}")
        
        # Show experiment details
        experiment_table = Table(title="A/B Test Experiment")
        experiment_table.add_column("Property", style="cyan")
        experiment_table.add_column("Value")
        
        experiment_table.add_row("ID", experiment.id)
        experiment_table.add_row("Name", experiment.name)
        experiment_table.add_row("Status", experiment.status.value)
        experiment_table.add_row("Start Date", str(experiment.start_date))
        experiment_table.add_row("End Date", str(experiment.end_date))
        experiment_table.add_row("Metrics", ", ".join(experiment.metrics))
        experiment_table.add_row("Asset Classes", ", ".join(experiment.asset_classes))
        
        console.print(experiment_table)
    
    return ab_service

def setup_auto_rollback():
    """Set up auto-rollback for the experimental ML signals."""
    print_section("SETTING UP AUTO-ROLLBACK")
    
    rollback_service = get_auto_rollback_service()
    
    # Create rollback rules for the experimental ML signals
    rules = [
        RollbackRule(
            id="drawdown_limit",
            metric_name="max_drawdown",
            threshold=-10.0,  # 10% drawdown
            direction=ThresholdDirection.BELOW,
            severity=AlertSeverity.CRITICAL,
            evaluation_window=5,  # 5 minutes
            cooldown_period=60,  # 60 minutes
            description="Maximum drawdown exceeded 10%"
        ),
        RollbackRule(
            id="sharpe_ratio_limit",
            metric_name="sharpe_ratio",
            threshold=0.5,  # Sharpe ratio below 0.5
            direction=ThresholdDirection.BELOW,
            severity=AlertSeverity.WARNING,
            evaluation_window=10,  # 10 minutes
            cooldown_period=30,  # 30 minutes
            description="Sharpe ratio below acceptable threshold"
        ),
        RollbackRule(
            id="transaction_cost_limit",
            metric_name="transaction_costs",
            threshold=15.0,  # Transaction costs too high
            direction=ThresholdDirection.ABOVE,
            severity=AlertSeverity.WARNING,
            evaluation_window=5,  # 5 minutes
            cooldown_period=30,  # 30 minutes
            description="Transaction costs exceeded threshold"
        )
    ]
    
    # Add rollback configuration
    success, message = rollback_service.add_rollback_config(
        flag_id="experimental_ml_signals",
        rules=rules,
        auto_rollback=True,
        require_multiple_triggers=True  # Require at least 2 rules to trigger
    )
    
    console.print(f"[{'green' if success else 'red'}]{message}[/{'green' if success else 'red'}]")
    
    # Show rollback rules
    rules_table = Table(title="Auto-Rollback Rules for experimental_ml_signals")
    rules_table.add_column("Rule ID", style="cyan")
    rules_table.add_column("Metric", style="blue")
    rules_table.add_column("Threshold", style="magenta")
    rules_table.add_column("Direction")
    rules_table.add_column("Severity", style="yellow")
    rules_table.add_column("Description")
    
    for rule in rules:
        direction_str = "Below" if rule.direction == ThresholdDirection.BELOW else "Above"
        rules_table.add_row(
            rule.id,
            rule.metric_name,
            str(rule.threshold),
            direction_str,
            rule.severity.value,
            rule.description
        )
    
    console.print(rules_table)
    
    return rollback_service

def generate_visualizations():
    """Generate visualizations for the feature flags."""
    print_section("GENERATING VISUALIZATIONS")
    
    visualizer = get_visualizer()
    
    # Generate basic dependency graph
    console.print("Generating dependency graph...")
    dep_file = visualizer.visualize_dependencies(
        filename="demo_dependencies",
        show=False,
        highlight_flags=["experimental_ml_signals", "enhanced_risk_controls"]
    )
    console.print(f"[green]Saved dependency graph to {dep_file}[/green]")
    
    # Generate simulated impact data for heatmap
    console.print("Generating impact heatmap...")
    impact_data = {
        "new_strategy_v1": {
            "pnl_daily": 5.2,
            "sharpe_ratio": 0.8,
            "max_drawdown": -2.5,
            "win_rate": 3.1
        },
        "enhanced_risk_controls": {
            "pnl_daily": -1.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": 4.5,
            "win_rate": -0.7
        },
        "alt_data_provider": {
            "pnl_daily": 2.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": -1.2,
            "win_rate": 1.5
        },
        "experimental_ml_signals": {
            "pnl_daily": 7.8,
            "sharpe_ratio": -1.1,
            "max_drawdown": -3.5,
            "win_rate": 4.2
        },
        "order_optimization": {
            "pnl_daily": 0.9,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.7,
            "win_rate": 0.5
        }
    }
    
    heatmap_file = visualizer.visualize_impact_heatmap(
        metric_impacts=impact_data,
        filename="demo_impact_heatmap",
        show=False
    )
    console.print(f"[green]Saved impact heatmap to {heatmap_file}[/green]")
    
    # Generate simulated history data for timeline
    console.print("Generating status timeline...")
    history_data = {}
    
    # Create fake history data
    now = datetime.now()
    for flag_id in ["new_strategy_v1", "enhanced_risk_controls", "alt_data_provider", 
                    "experimental_ml_signals", "order_optimization"]:
        history = []
        
        # Generate some random changes over the past 14 days
        for i in range(10):
            days_ago = random.randint(0, 14)
            hours_ago = random.randint(0, 23)
            timestamp = now - timedelta(days=days_ago, hours=hours_ago)
            
            history.append({
                "timestamp": timestamp.isoformat(),
                "enabled": random.choice([True, False]),
                "changed_by": random.choice(["admin", "system", "auto_rollback"])
            })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        # Add to history data
        history_data[flag_id] = history
    
    timeline_file = visualizer.visualize_flag_status_timeline(
        flag_history=history_data,
        filename="demo_status_timeline",
        show=False,
        days=14
    )
    console.print(f"[green]Saved status timeline to {timeline_file}[/green]")
    
    return visualizer

def simulate_ab_test_scenario():
    """Simulate an A/B test scenario."""
    print_section("SIMULATING A/B TEST SCENARIO")
    
    service = get_feature_flag_service()
    ab_service = get_ab_testing_service()
    metrics_collector = get_metrics_collector()
    
    # Get the experiment
    experiments = ab_service.list_experiments(status=ExperimentStatus.RUNNING)
    if not experiments:
        console.print("[yellow]No running experiments found[/yellow]")
        return
    
    experiment = experiments[0]
    console.print(f"Simulating data for experiment: {experiment.name} ({experiment.id})")
    
    # Simulate performance data collection for control and treatment groups
    with Progress() as progress:
        task = progress.add_task("[cyan]Collecting performance data...", total=10)
        
        for i in range(10):
            # Simulate different contexts (trades)
            for j in range(5):
                # Create a context for this check
                context = {
                    "asset_class": random.choice(list(experiment.asset_classes)),
                    "symbol": f"SYM{j}",
                    "account_id": f"ACC{j % 3}"
                }
                
                # Get variant for this context
                variant_id, variant = ab_service.get_variant_for_context(experiment.id, context)
                
                if variant is None:
                    continue
                
                # Check if the flag should be enabled for this variant
                flag_enabled = service.is_enabled(variant.flag_id, context)
                
                # Record flag usage
                metrics_collector.record_flag_usage(variant.flag_id, context, flag_enabled)
                
                # Bias performance metrics based on variant
                # In this simulation, the new strategy (treatment) performs better
                global simulated_metrics
                if variant_id == "treatment":
                    simulated_metrics["pnl_daily"] += random.uniform(0.5, 1.5)
                    simulated_metrics["sharpe_ratio"] += random.uniform(0.05, 0.1)
                else:
                    simulated_metrics["pnl_daily"] += random.uniform(-0.2, 0.8)
                    simulated_metrics["sharpe_ratio"] += random.uniform(-0.05, 0.05)
            
            # Record a performance snapshot
            metrics_collector.record_performance_snapshot()
            
            # Update progress
            progress.update(task, advance=1)
            time.sleep(0.5)
    
    # Analyze the experiment results
    console.print("\n[bold]Experiment Results:[/bold]")
    results = ab_service.analyze_experiment(experiment.id)
    
    # Show results
    metrics_table = Table(title=f"A/B Test Results: {experiment.name}")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Control (Mean)")
    metrics_table.add_column("Treatment (Mean)")
    metrics_table.add_column("Difference", style="magenta")
    metrics_table.add_column("% Change")
    metrics_table.add_column("Significant?")
    
    for metric, data in results.get("results", {}).items():
        variants = data.get("variants", {})
        comparison = data.get("comparison", {})
        
        # Skip metrics without comparison data
        if not comparison or not variants.get("control") or not variants.get("treatment"):
            continue
        
        control_mean = variants.get("control", {}).get("mean", "N/A")
        treatment_mean = variants.get("treatment", {}).get("mean", "N/A")
        
        abs_diff = comparison.get("absolute_difference", "N/A")
        percent_diff = comparison.get("percentage_difference", "N/A")
        
        if isinstance(percent_diff, (int, float)):
            percent_diff = f"{percent_diff:.2f}%"
        
        significant = comparison.get("significant", False)
        significant_text = "[green]Yes[/green]" if significant else "[red]No[/red]"
        
        metrics_table.add_row(
            metric,
            f"{control_mean:.4f}" if isinstance(control_mean, (int, float)) else str(control_mean),
            f"{treatment_mean:.4f}" if isinstance(treatment_mean, (int, float)) else str(treatment_mean),
            f"{abs_diff:.4f}" if isinstance(abs_diff, (int, float)) else str(abs_diff),
            str(percent_diff),
            significant_text
        )
    
    conclusion = results.get("summary", {}).get("conclusion", "No conclusion available")
    console.print(metrics_table)
    console.print(f"\n[bold]Conclusion:[/bold] {conclusion}")

def simulate_auto_rollback_scenario():
    """Simulate an auto-rollback scenario."""
    print_section("SIMULATING AUTO-ROLLBACK SCENARIO")
    
    service = get_feature_flag_service()
    rollback_service = get_auto_rollback_service()
    metrics_collector = get_metrics_collector()
    
    # Enable the experimental ML signals flag
    success, message = service.set_flag(
        flag_id="experimental_ml_signals",
        enabled=True,
        changed_by="demo_script",
        reason="Testing auto-rollback"
    )
    
    console.print(f"[{'green' if success else 'red'}]{message}[/{'green' if success else 'red'}]")
    
    # Set up a panel to track metrics
    console.print("\n[bold]Monitoring metrics for auto-rollback trigger:[/bold]")
    console.print("The flag will be rolled back if max_drawdown < -10.0 AND (sharpe_ratio < 0.5 OR transaction_costs > 15.0)")
    
    # Simulate deteriorating metrics
    with Progress() as progress:
        task = progress.add_task("[cyan]Simulating deteriorating metrics...", total=10)
        
        for i in range(10):
            # Update metrics to gradually deteriorate
            global simulated_metrics
            if i >= 5:  # After halfway, make metrics worse
                simulated_metrics["max_drawdown"] -= random.uniform(1.0, 2.0)
                simulated_metrics["sharpe_ratio"] -= random.uniform(0.1, 0.2)
                simulated_metrics["transaction_costs"] += random.uniform(1.0, 3.0)
            
            # Record a performance snapshot
            metrics_collector.record_performance_snapshot()
            
            # Get current metrics
            metrics = get_performance_metrics()
            
            # Display current metrics
            metrics_str = ", ".join([f"{k}: {v:.2f}" for k, v in {
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "transaction_costs": metrics["transaction_costs"]
            }.items()])
            
            # Update progress description with metrics
            progress.update(task, description=f"[cyan]Monitoring metrics: {metrics_str}")
            
            # Check if we should trigger rollback
            if i >= 7:  # After 70% of the way, check for rollback
                action_taken, triggered_rules = rollback_service.check_flag_metrics(
                    flag_id="experimental_ml_signals",
                    metrics=metrics
                )
                
                if action_taken:
                    progress.update(task, completed=10)
                    console.print("\n[bold red]Auto-rollback triggered![/bold red]")
                    console.print(f"Rules triggered: {', '.join(rule.id for rule in triggered_rules)}")
                    break
            
            # Update progress
            progress.update(task, advance=1)
            time.sleep(0.5)
    
    # Check if flag was rolled back
    if not service.is_enabled("experimental_ml_signals"):
        console.print("\n[green]Flag 'experimental_ml_signals' was successfully rolled back[/green]")
    else:
        console.print("\n[yellow]Flag 'experimental_ml_signals' was not rolled back[/yellow]")

def demonstrate_flag_dependencies():
    """Demonstrate flag dependencies in action."""
    print_section("DEMONSTRATING FLAG DEPENDENCIES")
    
    service = get_feature_flag_service()
    
    console.print("Flag 'experimental_ml_signals' depends on 'enhanced_risk_controls'")
    console.print("This means if 'enhanced_risk_controls' is disabled, 'experimental_ml_signals' should also be disabled")
    
    # Re-enable both flags first
    service.set_flag("enhanced_risk_controls", True, "demo_script", "Demonstrating dependencies")
    service.set_flag("experimental_ml_signals", True, "demo_script", "Demonstrating dependencies")
    
    # Show current state
    console.print("\n[bold]Initial state:[/bold]")
    console.print(f"enhanced_risk_controls: {'Enabled' if service.is_enabled('enhanced_risk_controls') else 'Disabled'}")
    console.print(f"experimental_ml_signals: {'Enabled' if service.is_enabled('experimental_ml_signals') else 'Disabled'}")
    
    # Disable the dependency
    console.print("\n[bold]Disabling enhanced_risk_controls:[/bold]")
    service.set_flag("enhanced_risk_controls", False, "demo_script", "Demonstrating dependencies")
    
    # Show new state
    console.print("\n[bold]New state:[/bold]")
    console.print(f"enhanced_risk_controls: {'Enabled' if service.is_enabled('enhanced_risk_controls') else 'Disabled'}")
    console.print(f"experimental_ml_signals: {'Enabled' if service.is_enabled('experimental_ml_signals') else 'Disabled'}")
    
    # Conclusion
    if not service.is_enabled("experimental_ml_signals"):
        console.print("\n[green]Flag dependencies working correctly![/green]")
    else:
        console.print("\n[red]Flag dependencies not working as expected![/red]")

def main():
    """Run the enhanced feature flags demo."""
    # Ensure the data directory exists
    os.makedirs("data/feature_flags", exist_ok=True)
    
    try:
        console.print(Panel(
            "This demo shows the enhanced feature flag system with metrics integration,\n"
            "A/B testing, automated rollbacks, and dependency visualization.",
            title="Enhanced Feature Flags Demo",
            border_style="cyan"
        ))
        
        # Set up all components
        service = setup_feature_flags()
        ab_service = setup_ab_testing()
        rollback_service = setup_auto_rollback()
        visualizer = generate_visualizations()
        
        # Demonstrate A/B testing
        simulate_ab_test_scenario()
        
        # Demonstrate auto-rollback
        simulate_auto_rollback_scenario()
        
        # Demonstrate flag dependencies
        demonstrate_flag_dependencies()
        
        print_section("DEMO COMPLETE")
        console.print("[green]All feature flag enhancements have been demonstrated successfully![/green]")
        
    except KeyboardInterrupt:
        console.print("[yellow]Demo stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in demo: {e}[/red]")
        logger.exception("Error in enhanced feature flags demo")
    finally:
        # Clean up
        service = get_feature_flag_service()
        service.cleanup()
        
        metrics_collector = get_metrics_collector()
        metrics_collector.cleanup()
        
        rollback_service = get_auto_rollback_service()
        rollback_service.cleanup()

if __name__ == "__main__":
    main() 