"""
Enhanced Trading Dashboard

A powerful dashboard for monitoring trading system performance,
controlling trading operations, and analyzing trade data.

This dashboard integrates with the trading bot API and provides:
- Interactive charts and visualizations
- Real-time data monitoring
- Advanced analytics and statistics
- Comprehensive configuration options
- Notifications and alerts
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

# UI libraries
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.align import Align
from rich import box, print

# Import custom modules
from trading_bot.config_manager import ConfigManager
from trading_bot.notification_manager import NotificationManager
from trading_bot.api_client import APIClient, APIEndpoint
from trading_bot.dashboard_charts import DashboardCharts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("EnhancedDashboard")

class EnhancedDashboard:
    """
    Enhanced trading dashboard with interactive features, live data updates,
    visual charts, and comprehensive control over the trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced dashboard.
        
        Args:
            config_path: Path to dashboard configuration file
        """
        # Initialize console
        self.console = Console()
        self.charts_initialized = False
        
        # Load configuration
        logger.info("Loading dashboard configuration...")
        self.config_manager = ConfigManager(
            config_dir=os.path.expanduser("~/.trading_bot"),
            default_config_path=os.path.join(os.path.dirname(__file__), "defaults", "dashboard_config.json")
        )
        
        if config_path and os.path.exists(config_path):
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.load_config()
        
        # Extract key configuration values
        self.api_url = self.config.get("api", {}).get("url", "http://localhost:5000")
        self.refresh_interval = self.config.get("dashboard", {}).get("refresh_interval", 10)
        self.dark_mode = self.config.get("dashboard", {}).get("color_theme", "dark") == "dark"
        
        # Initialize API client
        logger.info(f"Connecting to API at {self.api_url}...")
        self.api_client = APIClient(
            base_url=self.api_url,
            auth_token=self.config.get("api", {}).get("authentication", {}).get("token"),
            timeout=self.config.get("api", {}).get("timeout", 10),
            max_retries=self.config.get("api", {}).get("retry_count", 3)
        )
        
        # Initialize notification manager
        logger.info("Setting up notification system...")
        self.notification_manager = NotificationManager()
        
        # Update notification config from dashboard config
        if "notifications" in self.config:
            self.notification_manager.update_config(self.config["notifications"])
        
        # Initialize chart generator
        logger.info("Preparing chart renderer...")
        self.charts = DashboardCharts(
            output_dir=os.path.join(os.path.expanduser("~/.trading_bot"), "charts"),
            dark_mode=self.dark_mode
        )
        
        # Data store for dashboard
        self.data = {
            "trades": [],
            "open_positions": [],
            "metrics": {},
            "recommendations": [],
            "account": {
                "balance": 0.0,
                "equity": 0.0
            },
            "bot_status": {
                "status": "Unknown",
                "market_environment": "Unknown"
            },
            "chart_data": {}
        }
        
        # Threading
        self.running = False
        self.data_thread = None
        self.chart_thread = None
        self.data_lock = threading.Lock()
        
        # Dashboard state
        self.active_panel = "summary"
        self.notification_history = []
        self.last_refresh = datetime.now()
        
        logger.info("Dashboard initialization complete")
    
    def _fetch_data(self):
        """Background thread to periodically fetch data from API."""
        logger.info("Starting data fetch thread")
        
        while self.running:
            try:
                self._update_data()
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in data fetch thread: {e}")
                self.notification_manager.send_notification(
                    "Data Fetch Error",
                    f"Error fetching data: {str(e)}",
                    level="ERROR"
                )
                time.sleep(self.refresh_interval)
    
    def _update_data(self):
        """Fetch fresh data from the API."""
        with self.data_lock:
            # Update last refresh timestamp
            self.last_refresh = datetime.now()
            
            # Fetch open positions
            open_trades_response = self.api_client.get_open_trades()
            if open_trades_response.success:
                self.data["open_positions"] = open_trades_response.data.get("open_trades", [])
            
            # Fetch recent trades (last 50)
            trades_response = self.api_client.get_journal_trades()
            if trades_response.success:
                self.data["trades"] = trades_response.data.get("trades", [])
                
                # Sort by timestamp, most recent first
                if self.data["trades"]:
                    self.data["trades"] = sorted(
                        self.data["trades"],
                        key=lambda t: t.get("timestamp", ""),
                        reverse=True
                    )
            
            # Fetch performance metrics
            metrics_response = self.api_client.get_journal_metrics()
            if metrics_response.success:
                self.data["metrics"] = metrics_response.data.get("metrics", {})
            
            # Fetch recommendations
            recommendations_response = self.api_client.get_trade_recommendations()
            if recommendations_response.success:
                self.data["recommendations"] = recommendations_response.data.get("recommendations", [])
            
            # Fetch system status
            strategy_status_response = self.api_client.get_strategy_status()
            if strategy_status_response.success:
                self.data["bot_status"] = {
                    "status": "Running",
                    "market_environment": strategy_status_response.data.get("market_environment", "Unknown"),
                    "optimal_strategies": strategy_status_response.data.get("optimal_strategies", [])
                }
            
            # Check API health
            api_health = self.api_client.health_check()
            if api_health["status"] != "healthy":
                self.notification_manager.send_notification(
                    "API Health Warning",
                    f"API health check failed: {len(api_health['endpoints'])} endpoints affected",
                    level="WARNING",
                    metadata={"health_data": api_health}
                )
    
    def _update_charts(self):
        """Background thread to periodically update charts."""
        logger.info("Starting chart update thread")
        
        # Wait a few seconds for initial data fetch
        time.sleep(5)
        
        while self.running:
            try:
                with self.data_lock:
                    # Check if we have enough data for charts
                    if len(self.data["trades"]) > 0:
                        # Generate dashboard charts
                        self.data["chart_data"] = self.charts.generate_dashboard_charts(
                            self.data["trades"],
                            self.data["metrics"],
                            output_prefix="dashboard_",
                            save_files=True
                        )
                        self.charts_initialized = True
                        logger.info(f"Generated {len(self.data['chart_data'])} charts")
            except Exception as e:
                logger.error(f"Error updating charts: {e}")
            
            # Charts don't need to update as frequently as data
            time.sleep(60)  # Update charts every minute
    
    def _format_pnl(self, pnl: Union[float, str], include_color: bool = True) -> str:
        """Format P&L value with colors."""
        try:
            pnl_value = float(pnl) if isinstance(pnl, str) else pnl
            formatted = f"${pnl_value:.2f}"
            
            if include_color:
                if pnl_value > 0:
                    return f"[green]{formatted}[/green]"
                elif pnl_value < 0:
                    return f"[red]{formatted}[/red]"
                else:
                    return formatted
            else:
                return formatted
        except (ValueError, TypeError):
            return str(pnl)
    
    def show_header(self):
        """Display dashboard header with key information."""
        now = datetime.now()
        header_panel = Panel(
            f"[bold]Trading Bot Dashboard[/bold] | {now.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"API Status: {self.api_client.health_check()['status']} | "
            f"Last Update: {(now - self.last_refresh).seconds}s ago | "
            f"Market: {self.data['bot_status']['market_environment']}",
            style="white on blue",
            border_style="blue"
        )
        self.console.print(header_panel)
    
    def show_summary_panel(self):
        """Display summary panel with key metrics."""
        # Create a layout with two columns
        layout = Layout()
        layout.split_column(
            Layout(name="title"),
            Layout(name="content", ratio=5)
        )
        
        layout["content"].split_row(
            Layout(name="stats", ratio=2),
            Layout(name="positions", ratio=3)
        )
        
        # Add title
        layout["title"].update(Panel(Align.center("[bold]Dashboard Summary[/bold]")))
        
        # Create metrics panel
        metrics = self.data["metrics"]
        stats_table = Table(box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Win Rate", f"{metrics.get('win_rate', 0) * 100:.1f}%")
        stats_table.add_row("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        stats_table.add_row("Total P&L", self._format_pnl(metrics.get("total_pnl", 0)))
        stats_table.add_row("Average Win", self._format_pnl(metrics.get("avg_win", 0)))
        stats_table.add_row("Average Loss", self._format_pnl(metrics.get("avg_loss", 0)))
        stats_table.add_row("Largest Win", self._format_pnl(metrics.get("largest_win", 0)))
        stats_table.add_row("Largest Loss", self._format_pnl(metrics.get("largest_loss", 0)))
        stats_table.add_row("Total Trades", str(metrics.get("total_trades", 0)))
        
        # Create open positions table
        positions_table = Table(title="Open Positions", box=box.ROUNDED)
        positions_table.add_column("Symbol", style="cyan")
        positions_table.add_column("Strategy", style="magenta")
        positions_table.add_column("Entry", justify="right")
        positions_table.add_column("Current", justify="right")
        positions_table.add_column("P&L", justify="right")
        positions_table.add_column("Trade ID", style="dim")
        
        for position in self.data["open_positions"][:5]:  # Show first 5 positions
            symbol = position.get("symbol", "N/A")
            strategy = position.get("strategy", "N/A")
            entry_price = f"${position.get('entry_price', 0):.2f}"
            current_price = f"${position.get('current_price', 0):.2f}" if position.get('current_price') else "N/A"
            
            # Calculate P&L if we have both prices
            pnl = "N/A"
            if position.get('entry_price') and position.get('current_price'):
                pnl_value = (position['current_price'] - position['entry_price']) * position.get('position_size', 0)
                pnl = self._format_pnl(pnl_value)
            
            trade_id = position.get("trade_id", "N/A")
            
            positions_table.add_row(symbol, strategy, entry_price, current_price, pnl, trade_id)
        
        if not self.data["open_positions"]:
            positions_table.add_row("No open positions", "", "", "", "", "")
        
        # Update layout with tables
        layout["stats"].update(stats_table)
        layout["positions"].update(positions_table)
        
        # Print the layout
        self.console.print(layout)
    
    def show_positions_panel(self):
        """Display detailed information about open positions."""
        positions_table = Table(title="All Open Positions", box=box.ROUNDED)
        positions_table.add_column("Symbol", style="cyan")
        positions_table.add_column("Strategy", style="magenta")
        positions_table.add_column("Entry", justify="right")
        positions_table.add_column("Stop", justify="right", style="red")
        positions_table.add_column("Target", justify="right", style="green")
        positions_table.add_column("Size", justify="right")
        positions_table.add_column("Duration", justify="right")
        positions_table.add_column("Trade ID", style="dim")
        
        for position in self.data["open_positions"]:
            symbol = position.get("symbol", "N/A")
            strategy = position.get("strategy", "N/A")
            entry_price = f"${position.get('entry_price', 0):.2f}"
            stop_price = f"${position.get('stop_price', 0):.2f}"
            target_price = f"${position.get('target_price', 0):.2f}" if position.get('target_price') else "N/A"
            position_size = str(round(position.get('position_size', 0), 2))
            
            # Calculate duration if we have entry timestamp
            duration = "N/A"
            if "timestamp" in position:
                entry_time = datetime.fromisoformat(position["timestamp"])
                duration_sec = (datetime.now() - entry_time).total_seconds()
                if duration_sec < 3600:
                    duration = f"{duration_sec / 60:.1f}m"
                else:
                    duration = f"{duration_sec / 3600:.1f}h"
            
            trade_id = position.get("trade_id", "N/A")
            
            positions_table.add_row(
                symbol, strategy, entry_price, stop_price, target_price, 
                position_size, duration, trade_id
            )
        
        if not self.data["open_positions"]:
            positions_table.add_row("No open positions", "", "", "", "", "", "", "")
        
        self.console.print(positions_table)
        
        # Add action buttons
        self.console.print("\n[bold]Actions:[/bold]")
        self.console.print("[1] Exit a position   [2] Modify stop/target   [3] Back to summary")
    
    def show_trades_panel(self):
        """Display recent trades and performance metrics."""
        if not self.data["trades"]:
            self.console.print(Panel("No trade history available"))
            return
        
        # Show most recent trades
        trades_table = Table(title="Recent Trades", box=box.ROUNDED)
        trades_table.add_column("Date", style="dim")
        trades_table.add_column("Symbol", style="cyan")
        trades_table.add_column("Strategy", style="magenta")
        trades_table.add_column("Direction", style="blue")
        trades_table.add_column("Entry", justify="right")
        trades_table.add_column("Exit", justify="right")
        trades_table.add_column("P&L", justify="right")
        trades_table.add_column("Duration", justify="right")
        
        for trade in self.data["trades"][:10]:  # Show first 10 trades
            date = trade.get("timestamp", "").split("T")[0] if "timestamp" in trade else "N/A"
            symbol = trade.get("symbol", "N/A")
            strategy = trade.get("strategy_name", "N/A")
            direction = trade.get("trade_direction", "N/A")
            entry_price = f"${trade.get('entry_price', 0):.2f}"
            exit_price = f"${trade.get('exit_price', 0):.2f}" if "exit_price" in trade else "Open"
            pnl = self._format_pnl(trade.get("pnl_dollars", 0))
            duration = trade.get("duration", "N/A")
            
            trades_table.add_row(
                date, symbol, strategy, direction, entry_price, exit_price, pnl, duration
            )
        
        self.console.print(trades_table)
        
        # Add filter options
        self.console.print("\n[bold]Filter Options:[/bold]")
        self.console.print("[1] By strategy   [2] By symbol   [3] By date range   [4] Back to summary")
    
    def show_charts_panel(self):
        """Display interactive charts panel."""
        if not self.charts_initialized:
            with Panel("Charts are being generated, please wait...") as panel:
                self.console.print(panel)
            return
        
        chart_types = {
            "1": "Equity Curve",
            "2": "Win/Loss Ratio",
            "3": "Strategy Performance",
            "4": "P&L Distribution",
            "5": "Drawdown Analysis",
            "6": "Day of Week Analysis",
            "7": "Trade Duration"
        }
        
        self.console.print(Panel("[bold]Chart Selection[/bold]"))
        for key, name in chart_types.items():
            self.console.print(f"[{key}] {name}")
        
        # Show chart paths
        self.console.print("\n[bold]Generated Charts:[/bold]")
        for chart_name, chart in self.data["chart_data"].items():
            self.console.print(f"- {chart_name}: ~/.trading_bot/charts/dashboard_{chart_name}.html")
        
        self.console.print("\nEnter chart number to open in browser, or 'b' to go back to summary")
    
    def show_notifications_panel(self):
        """Display recent notifications and alerts."""
        notifications = self.notification_manager.get_notification_history(limit=20)
        
        if not notifications:
            self.console.print(Panel("No notifications yet"))
            return
        
        notifications_table = Table(title="Recent Notifications", box=box.ROUNDED)
        notifications_table.add_column("Time", style="dim")
        notifications_table.add_column("Level", style="cyan")
        notifications_table.add_column("Title", style="bold")
        notifications_table.add_column("Message")
        
        for notification in notifications:
            time_str = datetime.fromisoformat(notification["timestamp"]).strftime("%H:%M:%S")
            level = notification["level"]
            title = notification["title"]
            message = notification["message"]
            
            # Apply color based on level
            level_style = {
                "INFO": "green",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red"
            }.get(level, "white")
            
            notifications_table.add_row(
                time_str,
                f"[{level_style}]{level}[/{level_style}]",
                title,
                message
            )
        
        self.console.print(notifications_table)
    
    def show_settings_panel(self):
        """Display and modify dashboard settings."""
        settings_table = Table(title="Dashboard Settings", box=box.ROUNDED)
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="green")
        settings_table.add_column("Description")
        
        # Display key settings
        settings_table.add_row(
            "API URL",
            self.api_url,
            "Trading bot API endpoint"
        )
        settings_table.add_row(
            "Refresh Interval",
            f"{self.refresh_interval} seconds",
            "How often to refresh data"
        )
        settings_table.add_row(
            "Dark Mode",
            str(self.dark_mode),
            "Dark mode for charts"
        )
        settings_table.add_row(
            "Notifications",
            str(self.notification_manager.config["enabled"]),
            "Enable/disable notifications"
        )
        
        self.console.print(settings_table)
        
        # Add setting modification options
        self.console.print("\n[bold]Modify Settings:[/bold]")
        self.console.print("[1] API URL   [2] Refresh Interval   [3] Toggle Dark Mode   " +
                          "[4] Toggle Notifications   [5] Back to summary")
    
    def handle_user_input(self, panel: str, command: str) -> tuple:
        """
        Handle user input based on active panel.
        
        Args:
            panel: Active panel name
            command: User command
            
        Returns:
            Tuple of (new_panel, continue_running)
        """
        next_panel = panel
        continue_running = True
        
        # Handle global commands
        if command == "q":
            if Confirm.ask("Are you sure you want to exit?"):
                return panel, False
            else:
                return panel, True
        
        # Handle panel-specific commands
        if panel == "summary":
            if command == "1":
                next_panel = "positions"
            elif command == "2":
                next_panel = "trades"
            elif command == "3":
                next_panel = "charts"
            elif command == "4":
                next_panel = "notifications"
            elif command == "5":
                next_panel = "settings"
            elif command == "r":
                self._update_data()
        
        elif panel == "positions":
            if command == "1":
                self._handle_exit_position()
            elif command == "2":
                self._handle_modify_position()
            elif command == "3":
                next_panel = "summary"
        
        elif panel == "trades":
            if command == "1":
                self._handle_filter_by_strategy()
            elif command == "2":
                self._handle_filter_by_symbol()
            elif command == "3":
                self._handle_filter_by_date()
            elif command == "4":
                next_panel = "summary"
        
        elif panel == "charts":
            if command == "b":
                next_panel = "summary"
            elif command in ["1", "2", "3", "4", "5", "6", "7"]:
                self._handle_open_chart(command)
        
        elif panel == "settings":
            if command == "1":
                self._handle_change_api_url()
            elif command == "2":
                self._handle_change_refresh_interval()
            elif command == "3":
                self._handle_toggle_dark_mode()
            elif command == "4":
                self._handle_toggle_notifications()
            elif command == "5":
                next_panel = "summary"
        
        return next_panel, continue_running
    
    def _handle_exit_position(self):
        """Handle exiting a position."""
        if not self.data["open_positions"]:
            self.console.print("[yellow]No open positions to exit[/yellow]")
            return
        
        # Show open positions for reference
        positions_table = Table()
        positions_table.add_column("#", style="cyan")
        positions_table.add_column("Symbol", style="bold")
        positions_table.add_column("Strategy")
        positions_table.add_column("Entry Price")
        positions_table.add_column("Trade ID", style="dim")
        
        for i, position in enumerate(self.data["open_positions"]):
            positions_table.add_row(
                str(i+1),
                position.get("symbol", "N/A"),
                position.get("strategy", "N/A"),
                f"${position.get('entry_price', 0):.2f}",
                position.get("trade_id", "N/A")
            )
        
        self.console.print(positions_table)
        
        # Prompt for position to exit
        position_num = Prompt.ask(
            "Enter position number to exit (or 'c' to cancel)",
            choices=[str(i+1) for i in range(len(self.data["open_positions"]))] + ["c"]
        )
        
        if position_num == "c":
            return
        
        # Get position details
        idx = int(position_num) - 1
        position = self.data["open_positions"][idx]
        trade_id = position.get("trade_id")
        symbol = position.get("symbol")
        
        # Prompt for exit price (optional)
        exit_price_str = Prompt.ask(
            "Enter exit price (leave empty for market price)"
        )
        
        exit_price = None
        if exit_price_str:
            try:
                exit_price = float(exit_price_str)
            except ValueError:
                self.console.print("[red]Invalid price format. Using market price.[/red]")
        
        # Confirm exit
        if not Confirm.ask(f"Confirm exit for {symbol} (ID: {trade_id})?"):
            return
        
        # Execute exit
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task("Exiting position...", total=None)
            response = self.api_client.exit_trade(trade_id, exit_price=exit_price)
        
        if response.success:
            self.console.print(f"[green]Successfully exited {symbol} (ID: {trade_id})[/green]")
            self.notification_manager.send_notification(
                f"Exited {symbol}",
                f"Trade {trade_id} exited successfully" + 
                (f" at ${exit_price:.2f}" if exit_price else " at market price"),
                level="SUCCESS"
            )
            
            # Refresh open positions
            self._update_data()
        else:
            self.console.print(f"[red]Error exiting position: {response.error_message}[/red]")
            self.notification_manager.send_notification(
                "Exit Error",
                f"Failed to exit {symbol} (ID: {trade_id}): {response.error_message}",
                level="ERROR"
            )
    
    def _handle_modify_position(self):
        """Handle modifying a position's stop/target."""
        # Implementation omitted for brevity
        self.console.print("[yellow]Feature not yet implemented[/yellow]")
    
    def _handle_filter_by_strategy(self):
        """Handle filtering trades by strategy."""
        # Implementation omitted for brevity
        self.console.print("[yellow]Feature not yet implemented[/yellow]")
    
    def _handle_filter_by_symbol(self):
        """Handle filtering trades by symbol."""
        # Implementation omitted for brevity
        self.console.print("[yellow]Feature not yet implemented[/yellow]")
    
    def _handle_filter_by_date(self):
        """Handle filtering trades by date range."""
        # Implementation omitted for brevity
        self.console.print("[yellow]Feature not yet implemented[/yellow]")
    
    def _handle_open_chart(self, chart_num: str):
        """Handle opening a chart in the browser."""
        chart_map = {
            "1": "equity_curve",
            "2": "win_loss",
            "3": "strategy_performance",
            "4": "pnl_distribution",
            "5": "drawdown",
            "6": "day_of_week",
            "7": "trade_duration"
        }
        
        chart_name = chart_map.get(chart_num)
        if not chart_name or chart_name not in self.data["chart_data"]:
            self.console.print(f"[yellow]Chart {chart_num} not available[/yellow]")
            return
        
        # Get chart path
        chart_path = os.path.join(
            os.path.expanduser("~/.trading_bot/charts"),
            f"dashboard_{chart_name}.html"
        )
        
        if not os.path.exists(chart_path):
            self.console.print(f"[yellow]Chart file not found: {chart_path}[/yellow]")
            return
        
        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{chart_path}")
        self.console.print(f"[green]Opened {chart_name} chart in browser[/green]")
    
    def _handle_change_api_url(self):
        """Handle changing the API URL."""
        current_url = self.api_url
        new_url = Prompt.ask(f"Enter new API URL (current: {current_url})")
        
        if new_url and new_url != current_url:
            # Update configuration
            self.api_url = new_url
            self.config_manager.set_value("api.url", new_url)
            
            # Update API client
            self.api_client = APIClient(
                base_url=new_url,
                auth_token=self.config.get("api", {}).get("authentication", {}).get("token"),
                timeout=self.config.get("api", {}).get("timeout", 10),
                max_retries=self.config.get("api", {}).get("retry_count", 3)
            )
            
            self.console.print(f"[green]API URL updated to {new_url}[/green]")
            
            # Refresh data
            self._update_data()
    
    def _handle_change_refresh_interval(self):
        """Handle changing the refresh interval."""
        current_interval = self.refresh_interval
        new_interval_str = Prompt.ask(
            f"Enter new refresh interval in seconds (current: {current_interval})"
        )
        
        try:
            new_interval = int(new_interval_str)
            if new_interval < 1:
                self.console.print("[red]Refresh interval must be at least 1 second[/red]")
                return
            
            # Update configuration
            self.refresh_interval = new_interval
            self.config_manager.set_value("dashboard.refresh_interval", new_interval)
            
            self.console.print(f"[green]Refresh interval updated to {new_interval} seconds[/green]")
        except ValueError:
            self.console.print("[red]Invalid interval format. Must be an integer.[/red]")
    
    def _handle_toggle_dark_mode(self):
        """Handle toggling dark mode."""
        self.dark_mode = not self.dark_mode
        self.config_manager.set_value("dashboard.color_theme", "dark" if self.dark_mode else "light")
        
        # Update chart renderer
        self.charts.dark_mode = self.dark_mode
        self.charts.template = "plotly_dark" if self.dark_mode else "plotly_white"
        
        self.console.print(f"[green]Dark mode {'enabled' if self.dark_mode else 'disabled'}[/green]")
        self.console.print("[yellow]Charts will be updated with the new theme on the next refresh[/yellow]")
    
    def _handle_toggle_notifications(self):
        """Handle toggling notifications."""
        enabled = self.notification_manager.config["enabled"]
        self.notification_manager.update_config({"enabled": not enabled})
        
        # Update configuration
        self.config_manager.set_value("notifications.enabled", not enabled)
        
        self.console.print(f"[green]Notifications {'disabled' if enabled else 'enabled'}[/green]")
    
    def display_dashboard(self):
        """Display the dashboard based on active panel."""
        # Clear screen
        self.console.clear()
        
        # Show header
        self.show_header()
        
        # Show active panel
        if self.active_panel == "summary":
            self.show_summary_panel()
            self.console.print("\n[bold]Navigation:[/bold]")
            self.console.print("[1] Open Positions   [2] Trade History   [3] Charts   " +
                              "[4] Notifications   [5] Settings   [r] Refresh   [q] Quit")
        
        elif self.active_panel == "positions":
            self.show_positions_panel()
        
        elif self.active_panel == "trades":
            self.show_trades_panel()
        
        elif self.active_panel == "charts":
            self.show_charts_panel()
        
        elif self.active_panel == "notifications":
            self.show_notifications_panel()
            self.console.print("\n[bold]Press any key to return to summary[/bold]")
            self.active_panel = "summary"  # Auto-return to summary after viewing
        
        elif self.active_panel == "settings":
            self.show_settings_panel()
    
    def run(self):
        """Run the dashboard main loop."""
        try:
            self.running = True
            
            # Initial data fetch
            self._update_data()
            
            # Start background threads
            self.data_thread = threading.Thread(target=self._fetch_data)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            self.chart_thread = threading.Thread(target=self._update_charts)
            self.chart_thread.daemon = True
            self.chart_thread.start()
            
            # Show welcome notification
            self.notification_manager.send_notification(
                "Dashboard Started",
                "Trading dashboard is now running",
                level="INFO"
            )
            
            # Main display loop
            while self.running:
                self.display_dashboard()
                
                # Get user input
                command = Prompt.ask("Enter command")
                
                # Handle input
                self.active_panel, self.running = self.handle_user_input(self.active_panel, command)
            
            # Graceful shutdown
            self.console.print("[yellow]Shutting down dashboard...[/yellow]")
            
        except KeyboardInterrupt:
            self.console.print("[yellow]Dashboard terminated by user[/yellow]")
        except Exception as e:
            logger.error(f"Error in dashboard: {e}")
            self.console.print(f"[red]Error in dashboard: {str(e)}[/red]")
        finally:
            # Clean up
            self.running = False
            
            if self.data_thread:
                self.data_thread.join(timeout=0.5)
            
            if self.chart_thread:
                self.chart_thread.join(timeout=0.5)
            
            self.console.print("[green]Dashboard shut down successfully[/green]")

if __name__ == "__main__":
    # Get command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Trading Dashboard")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Start dashboard
    dashboard = EnhancedDashboard(config_path=args.config)
    dashboard.run() 