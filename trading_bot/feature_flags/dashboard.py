"""
Feature Flag Dashboard Component

Provides visualizations and interactive components for the trading bot's feature flags.
This module is responsible for displaying the current state of feature flags
in the bot's dashboard UI.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED
from rich.console import Group
from rich.progress_bar import ProgressBar

from .service import FlagCategory, FeatureFlag, get_feature_flag_service, ContextRule

logger = logging.getLogger(__name__)

class FeatureFlagDashboard:
    """Displays feature flag information in the bot dashboard."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the feature flag dashboard.
        
        Args:
            console: Rich console to use for output
        """
        self.console = console or Console()
        self.service = get_feature_flag_service()
    
    def display_flag_summary(self):
        """Display a summary of all feature flags."""
        all_flags = self.service.list_flags()
        
        # Count flags by category and state
        categories = {}
        enabled_count = 0
        disabled_count = 0
        
        for flag in all_flags:
            category_name = flag.category.name
            if category_name not in categories:
                categories[category_name] = {"enabled": 0, "disabled": 0}
            
            if flag.enabled:
                categories[category_name]["enabled"] += 1
                enabled_count += 1
            else:
                categories[category_name]["disabled"] += 1
                disabled_count += 1
        
        # Create summary table
        table = Table(title="Feature Flag Summary", box=ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Enabled", style="green")
        table.add_column("Disabled", style="red")
        table.add_column("Total", style="blue")
        
        for category, counts in sorted(categories.items()):
            total = counts["enabled"] + counts["disabled"]
            table.add_row(
                category,
                str(counts["enabled"]),
                str(counts["disabled"]),
                str(total)
            )
        
        table.add_row(
            "TOTAL",
            str(enabled_count),
            str(disabled_count),
            str(enabled_count + disabled_count),
            style="bold"
        )
        
        self.console.print(table)
    
    def display_flags_by_category(self, category: Optional[FlagCategory] = None):
        """Display detailed information about flags in a category.
        
        Args:
            category: The category to display flags for, or None for all
        """
        flags = self.service.list_flags(category)
        
        # Group flags by category
        grouped_flags = {}
        for flag in flags:
            category_name = flag.category.name
            if category_name not in grouped_flags:
                grouped_flags[category_name] = []
            grouped_flags[category_name].append(flag)
        
        # Display each category
        for category_name, category_flags in sorted(grouped_flags.items()):
            table = Table(title=f"{category_name} Flags", box=ROUNDED)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="blue")
            table.add_column("Status", style="bold")
            table.add_column("Modified", style="magenta")
            table.add_column("Description")
            
            for flag in sorted(category_flags, key=lambda f: f.id):
                status_text = Text("ENABLED", "green bold") if flag.enabled else Text("DISABLED", "red")
                modified_ago = datetime.now() - flag.modified_at
                modified_text = self._format_time_ago(modified_ago)
                
                table.add_row(
                    flag.id,
                    flag.name,
                    status_text,
                    modified_text,
                    flag.description[:50] + "..." if len(flag.description) > 50 else flag.description
                )
            
            self.console.print(table)
    
    def display_flag_details(self, flag_id: str):
        """Display detailed information about a specific flag.
        
        Args:
            flag_id: ID of the flag to display
        """
        flag = self.service.get_flag(flag_id)
        if not flag:
            self.console.print(f"[red]Flag '{flag_id}' not found[/red]")
            return
        
        # Create flag details panel
        status = "[green]ENABLED[/green]" if flag.enabled else "[red]DISABLED[/red]"
        title = f"{flag.name} ({flag.id}) - {status}"
        
        content = []
        content.append(f"[bold cyan]Description:[/bold cyan] {flag.description}")
        content.append(f"[bold cyan]Category:[/bold cyan] {flag.category.name}")
        content.append(f"[bold cyan]Created:[/bold cyan] {flag.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"[bold cyan]Last Modified:[/bold cyan] {flag.modified_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if flag.default != flag.enabled:
            content.append(f"[bold yellow]Current state differs from default ({flag.default})[/bold yellow]")
        
        if flag.requires_confirmation:
            content.append("[bold magenta]Requires confirmation to change[/bold magenta]")
        
        if flag.rollback_after_seconds:
            hours = flag.rollback_after_seconds // 3600
            minutes = (flag.rollback_after_seconds % 3600) // 60
            rollback_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            content.append(f"[bold red]Auto-rollback after: {rollback_text}[/bold red]")
        
        if flag.dependent_flags:
            content.append(f"[bold cyan]Dependencies:[/bold cyan] {', '.join(flag.dependent_flags)}")
        
        # Add gradual rollout info
        content.append("\n[bold cyan]Gradual Rollout Settings:[/bold cyan]")
        
        # Add rollout percentage with progress bar
        progress_text = f"[bold cyan]Rollout Percentage:[/bold cyan] {flag.rollout_percentage}%"
        content.append(progress_text)
        
        # Add applicable asset classes
        asset_classes = list(flag.applicable_asset_classes)
        if asset_classes:
            if "ALL" in asset_classes:
                content.append(f"[bold cyan]Asset Classes:[/bold cyan] All asset classes")
            else:
                content.append(f"[bold cyan]Asset Classes:[/bold cyan] {', '.join(asset_classes)}")
        
        # Add context rules if any exist
        if flag.context_rules:
            content.append("\n[bold cyan]Context Rules:[/bold cyan]")
            for i, rule in enumerate(flag.context_rules):
                rule_desc = self._format_context_rule(rule)
                content.append(f"  {i+1}. {rule_desc}")
        
        # Show history in a separate table
        history_table = Table(title="Change History", box=ROUNDED)
        history_table.add_column("Time", style="magenta")
        history_table.add_column("State", style="bold")
        history_table.add_column("Changed By", style="cyan")
        history_table.add_column("Reason")
        
        for event in sorted(flag.history, key=lambda e: e.timestamp, reverse=True)[:10]:
            state_text = Text("ENABLED", "green bold") if event.enabled else Text("DISABLED", "red")
            history_table.add_row(
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                state_text,
                event.changed_by,
                event.reason or ""
            )
        
        # Create a rollout percentage progress bar
        progress_bar = ProgressBar(
            total=100,
            completed=flag.rollout_percentage,
            width=50
        )
        
        panel_content = Group(
            Text("\n".join(content)),
            progress_bar,
            Text("\n"),
            history_table
        )
        
        panel = Panel(
            panel_content,
            title=title,
            border_style="blue"
        )
        
        self.console.print(panel)
    
    def display_recent_changes(self, limit: int = 20):
        """Display recent flag changes across all flags.
        
        Args:
            limit: Maximum number of changes to display
        """
        # Collect all events from all flags
        all_events = []
        for flag in self.service.list_flags():
            for event in flag.history:
                all_events.append((flag, event))
        
        # Sort by timestamp (newest first) and limit
        all_events.sort(key=lambda x: x[1].timestamp, reverse=True)
        all_events = all_events[:limit]
        
        # Create table
        table = Table(title="Recent Flag Changes", box=ROUNDED)
        table.add_column("Time", style="magenta")
        table.add_column("Flag", style="cyan")
        table.add_column("Category", style="blue")
        table.add_column("Changed By", style="yellow")
        table.add_column("New State", style="bold")
        table.add_column("Reason")
        
        for flag, event in all_events:
            state_text = Text("ENABLED", "green bold") if event.enabled else Text("DISABLED", "red")
            table.add_row(
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f"{flag.name} ({flag.id})",
                flag.category.name,
                event.changed_by,
                state_text,
                event.reason or ""
            )
        
        self.console.print(table)
    
    def _format_time_ago(self, delta: timedelta) -> str:
        """Format a timedelta as a human-readable string.
        
        Args:
            delta: Time difference
            
        Returns:
            str: Formatted time ago string
        """
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
    
    def _format_context_rule(self, rule: ContextRule) -> str:
        """Format a context rule for display.
        
        Args:
            rule: The context rule to format
            
        Returns:
            str: Formatted rule description
        """
        if rule.rule_type == "asset_class":
            asset_classes = rule.parameters.get("asset_classes", [])
            return f"Asset Class Rule: Apply to {', '.join(asset_classes)}"
        
        elif rule.rule_type == "time_window":
            start = rule.parameters.get("start_time", "00:00")
            end = rule.parameters.get("end_time", "23:59")
            return f"Time Window Rule: Active between {start} and {end}"
        
        elif rule.rule_type == "account_value":
            min_val = rule.parameters.get("min_value", "any")
            max_val = rule.parameters.get("max_value", "any")
            return f"Account Value Rule: {min_val} to {max_val}"
        
        elif rule.rule_type == "market_condition":
            conditions = rule.parameters.get("conditions", [])
            return f"Market Condition Rule: Active during {', '.join(conditions)}"
        
        else:
            return f"Unknown Rule Type: {rule.rule_type}"
    
    def display_gradual_rollout_dashboard(self):
        """Display a dashboard of flags with gradual rollout settings."""
        all_flags = self.service.list_flags()
        
        # Filter to flags with non-default rollout settings
        gradual_flags = [
            flag for flag in all_flags 
            if flag.rollout_percentage < 100 or 
               "ALL" not in flag.applicable_asset_classes or 
               flag.context_rules
        ]
        
        if not gradual_flags:
            self.console.print("[yellow]No flags with gradual rollout settings found[/yellow]")
            return
        
        # Create table
        table = Table(title="Gradual Rollout Dashboard", box=ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Rollout %", style="magenta")
        table.add_column("Asset Classes")
        table.add_column("Rules", justify="right")
        
        for flag in sorted(gradual_flags, key=lambda f: f.id):
            status_text = Text("ENABLED", "green bold") if flag.enabled else Text("DISABLED", "red")
            
            # Format asset classes
            asset_classes = list(flag.applicable_asset_classes)
            if "ALL" in asset_classes:
                asset_text = "All"
            else:
                asset_text = ", ".join(asset_classes)
            
            # Count of rules
            rules_count = len(flag.context_rules)
            rules_text = str(rules_count) if rules_count > 0 else "-"
            
            table.add_row(
                flag.id,
                flag.name,
                status_text,
                f"{flag.rollout_percentage}%",
                asset_text,
                rules_text
            )
        
        self.console.print(table) 