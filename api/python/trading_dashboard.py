"""
Trading Dashboard

A real-time dashboard for monitoring trading bot activities and performance.
Provides interactive control and visualization of trading strategies, open positions,
and performance metrics.
"""

import json
import os
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align

console = Console()

# Configuration: Load from environment variables or default to localhost
BOT_API_URL = os.getenv("BOT_API_URL", "http://localhost:5000")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "10"))

class TradingDashboard:
    def __init__(self):
        self.trades = []
        self.metrics = {}
        self.recommendations = []
        self.bot_status = {}
        self.account_balance = {}
        self.open_positions = []

    def concurrent_get(self, endpoint):
        """Helper function to concurrently GET data from bot endpoints."""
        url = f"{BOT_API_URL}/{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return endpoint, response.json()
        except Exception as e:
            console.log(f"[red]Error fetching {endpoint}: {e}[/red]")
            return endpoint, None

    def fetch_data(self):
        """Fetch live data from the trading bot concurrently."""
        endpoints = {
            "trades": "journal/trades",
            "metrics": "journal/metrics", 
            "recommendations": "journal/recommendations", 
            "status": "strategy_status",
            "open_positions": "open-trades"
        }
        
        fetched_data = {}
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            future_to_ep = {executor.submit(self.concurrent_get, url): key for key, url in endpoints.items()}
            for future in as_completed(future_to_ep):
                key = future_to_ep[future]
                _, data = future.result()
                fetched_data[key] = data

        # Update instance variables if data is available
        if fetched_data.get("trades") and fetched_data["trades"].get("status") == "success":
            self.trades = fetched_data["trades"].get("trades", [])
        
        if fetched_data.get("metrics") and fetched_data["metrics"].get("status") == "success":
            self.metrics = fetched_data["metrics"].get("metrics", {})
        
        if fetched_data.get("recommendations") and fetched_data["recommendations"].get("status") == "success":
            self.recommendations = fetched_data["recommendations"].get("recommendations", [])
        
        if fetched_data.get("status") and fetched_data["status"].get("status") == "success":
            self.bot_status = {
                "status": "Running",
                "market_environment": fetched_data["status"].get("market_environment", "Unknown"),
                "optimal_strategies": fetched_data["status"].get("optimal_strategies", [])
            }
        
        if fetched_data.get("open_positions") and fetched_data["open_positions"].get("status") == "success":
            self.open_positions = fetched_data["open_positions"].get("open_trades", [])
        
        # Get account balance (separate call)
        try:
            response = requests.post(f"{BOT_API_URL}/update-account-size", json={"account_size": 0}, timeout=5)
            if response.ok:
                current_size = str(response.json().get("message", "")).replace("Account size updated to $", "").replace(",", "")
                try:
                    self.account_balance = {
                        "balance": float(current_size) if current_size else 0.0,
                        "equity": float(current_size) if current_size else 0.0
                    }
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            console.log(f"[red]Error fetching account balance: {e}[/red]")

    def show_header(self):
        """Display a custom header with system info."""
        header_text = f"[bold blue]Trading Dashboard[/bold blue] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        header_panel = Panel(Align.center(header_text), style="bold green", border_style="green")
        console.print(header_panel)

    def show_summary(self):
        """Display a summary of system status and account info."""
        market_env = self.bot_status.get('market_environment', 'Unknown')
        summary_panel = Panel.fit(
            f"[bold]Bot Status:[/bold] {self.bot_status.get('status', 'Unknown')}\n"
            f"[bold]Market Environment:[/bold] {market_env}\n"
            f"[bold]Account Balance:[/bold] ${self.account_balance.get('balance', '0.00'):,.2f}",
            title="[bold]System Summary[/bold]",
            border_style="cyan",
        )
        console.print(summary_panel)

        # Display top strategies
        if self.metrics.get("strategy_performance"):
            table = Table(title="Top Strategies (by Win Rate)")
            table.add_column("Strategy", style="magenta")
            table.add_column("Win Rate", style="green")
            table.add_column("Profit Factor", justify="right", style="yellow")
            table.add_column("Avg Win", justify="right", style="cyan")
            table.add_column("Avg Loss", justify="right", style="red")
            
            # Sort strategies by win rate
            sorted_strats = sorted(
                self.metrics["strategy_performance"].items(), 
                key=lambda x: x[1].get("win_rate", 0), 
                reverse=True
            )
            
            for strat_name, metrics in sorted_strats[:5]:  # Show top 5
                win_rate = metrics.get("win_rate", 0) * 100
                table.add_row(
                    strat_name,
                    f"{win_rate:.1f}%",
                    f"{metrics.get('profit_factor', 0):.2f}",
                    f"${metrics.get('avg_win', 0):.2f}",
                    f"${metrics.get('avg_loss', 0):.2f}"
                )
            console.print(table)
        else:
            console.print("[italic]No strategy metrics available yet.[/italic]")

        # Display optimal strategies based on current market regime
        if self.bot_status.get("optimal_strategies"):
            console.print(f"[bold green]Optimal Strategies for Current Market:[/bold green] {', '.join(self.bot_status['optimal_strategies'])}")

    def show_recent_trades(self):
        """Display a table of recent trades (last 5)."""
        table = Table(title="Recent Trades (last 5)")
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="bold")
        table.add_column("Strategy", style="magenta")
        table.add_column("Direction", style="cyan")
        table.add_column("PnL", justify="right")
        table.add_column("Emotional Flag", style="red")

        # Sort trades by timestamp descending
        sorted_trades = sorted(self.trades, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        for trade in sorted_trades[:5]:
            timestamp = trade.get("timestamp", "").split("T")[0] if "timestamp" in trade else "N/A"
            symbol = trade.get("symbol", "N/A")
            strat = trade.get("strategy_name", "N/A")
            direction = trade.get("trade_direction", "N/A")
            
            # Format PnL
            pnl = trade.get("pnl_dollars", 0)
            pnl_formatted = f"[green]${pnl:.2f}[/green]" if pnl > 0 else f"[red]-${abs(pnl):.2f}[/red]"
            
            emo_flag = trade.get("emotional_flag", "none")
            emo_formatted = f"[red]{emo_flag}[/red]" if emo_flag != "none" else "none"
            
            table.add_row(timestamp, symbol, strat, direction, pnl_formatted, emo_formatted)
        
        console.print(table)

    def show_open_positions(self):
        """Display a table of current open positions."""
        if not self.open_positions:
            console.print("[italic]No open positions at this time.[/italic]")
            return
            
        table = Table(title="Open Positions")
        table.add_column("Symbol", style="bold")
        table.add_column("Strategy", style="magenta")
        table.add_column("Entry Price", justify="right")
        table.add_column("Stop Price", justify="right", style="red")
        table.add_column("Target", justify="right", style="green")
        table.add_column("Size", justify="right")
        table.add_column("Trade ID", style="dim")

        for position in self.open_positions:
            table.add_row(
                position.get("symbol", "N/A"),
                position.get("strategy", "N/A"),
                f"${position.get('entry_price', 0):.2f}",
                f"${position.get('stop_price', 0):.2f}",
                f"${position.get('target_price', 0):.2f}" if position.get('target_price') else "N/A",
                str(round(position.get('position_size', 0), 2)),
                position.get("trade_id", "N/A")
            )
        console.print(table)

    def show_recommendations(self):
        """Display AI-driven trading recommendations."""
        if not self.recommendations:
            console.print("[italic]No recommendations at this time.[/italic]")
            return

        rec_panel = Panel(title="Trading Recommendations", border_style="cyan")
        rec_messages = ""
        for rec in self.recommendations:
            action = rec.get("action", "No action")
            reason = rec.get("reason", "")
            priority = rec.get("priority", "medium")
            
            # Color code by priority
            if priority == "high":
                rec_messages += f"[bold red]● {action}[/bold red] - {reason}\n"
            elif priority == "medium":
                rec_messages += f"[bold yellow]● {action}[/bold yellow] - {reason}\n"
            else:
                rec_messages += f"[bold green]● {action}[/bold green] - {reason}\n"
        
        rec_panel = Panel(rec_messages, title="[bold]AI Recommendations[/bold]", border_style="cyan")
        console.print(rec_panel)

    def show_bot_commands(self):
        """Display bot control commands and process user choice."""
        commands_panel = Panel.fit(
            "[bold magenta]Bot Control Commands[/bold magenta]\n"
            "[1] Exit Trade\n"
            "[2] Update Account Size\n"
            "[3] Generate Daily Report\n"
            "[4] Generate Weekly Report\n"
            "[5] Refresh Data Manually\n"
            "[6] Exit Dashboard",
            border_style="magenta"
        )
        console.print(commands_panel)

        choice = Prompt.ask("Enter a command", choices=["1", "2", "3", "4", "5", "6"])
        if choice == "1":
            self.exit_trade()
        elif choice == "2":
            self.update_account_size()
        elif choice == "3":
            self.generate_report("daily")
        elif choice == "4":
            self.generate_report("weekly")
        elif choice == "5":
            console.print("[bold cyan]Manual refresh initiated...[/bold cyan]")
            self.fetch_data()
            self.display_all()
        elif choice == "6":
            console.print("[bold red]Exiting Dashboard...[/bold red]")
            exit()

    def exit_trade(self):
        """Exit a specific trade by Trade ID."""
        # Show open positions for reference
        self.show_open_positions()
        
        if not self.open_positions:
            console.print("[yellow]No open positions to exit.[/yellow]")
            return
            
        trade_id = Prompt.ask("Enter the Trade ID to exit")
        exit_price = Prompt.ask("Enter exit price (leave blank for current market price)", default="")
        
        try:
            payload = {"symbol": ""}
            if exit_price:
                payload["exit_price"] = float(exit_price)
                
            # Find the symbol for this trade ID
            for position in self.open_positions:
                if position.get("trade_id") == trade_id:
                    payload["symbol"] = position.get("symbol", "")
                    break
                    
            if not payload["symbol"]:
                console.print(f"[red]Trade ID {trade_id} not found in open positions.[/red]")
                return
                
            response = requests.post(f"{BOT_API_URL}/exit_trade/{trade_id}", json=payload, timeout=10)
            if response.ok:
                console.print(f"[bold green]Trade {trade_id} exited successfully.[/bold green]")
                self.fetch_data()  # Refresh data
            else:
                console.print(f"[red]Failed to exit trade {trade_id}. Status: {response.status_code}[/red]")
                console.print(response.text)
        except Exception as e:
            console.print(f"[red]Error exiting trade: {e}[/red]")

    def update_account_size(self):
        """Update the account size."""
        current = self.account_balance.get('balance', 0)
        new_size = Prompt.ask(f"Enter new account size (current: ${current:,.2f})")
        
        try:
            new_size_float = float(new_size.replace(',', ''))
            if new_size_float <= 0:
                console.print("[red]Account size must be positive.[/red]")
                return
                
            response = requests.post(f"{BOT_API_URL}/update-account-size", json={"account_size": new_size_float}, timeout=5)
            if response.ok:
                console.print(f"[bold green]{response.json().get('message', 'Account size updated successfully')}[/bold green]")
                self.fetch_data()  # Refresh data
            else:
                console.print(f"[red]Failed to update account size. Status: {response.status_code}[/red]")
        except ValueError:
            console.print("[red]Invalid account size. Please enter a valid number.[/red]")
        except Exception as e:
            console.print(f"[red]Error updating account size: {e}[/red]")

    def generate_report(self, report_type):
        """Generate a trading report."""
        try:
            # For daily report
            if report_type == "daily":
                # Prepare data for daily report
                today = datetime.now().strftime("%Y-%m-%d")
                payload = {
                    "trade_data": self.trades,
                    "performance_metrics": self.metrics,
                    "market_context": {
                        "market_environment": self.bot_status.get("market_environment", "Unknown"),
                        "date": today
                    }
                }
                
                response = requests.post(f"{BOT_API_URL}/reports/generate-daily", json=payload, timeout=15)
            # For weekly report
            else:
                # Prepare data for weekly report
                payload = {
                    "weekly_trade_data": self.trades,
                    "weekly_performance": self.metrics
                }
                
                response = requests.post(f"{BOT_API_URL}/reports/generate-weekly", json=payload, timeout=15)
            
            if response.ok:
                report_path = response.json().get("report_path", "Unknown")
                console.print(f"[bold green]Report generated successfully at: {report_path}[/bold green]")
            else:
                console.print(f"[red]Failed to generate report. Status: {response.status_code}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")

    def display_all(self):
        """Displays all dashboard sections sequentially."""
        os.system("cls" if os.name == "nt" else "clear")
        self.show_header()
        self.show_summary()
        self.show_open_positions()
        self.show_recent_trades()
        self.show_recommendations()
        self.show_bot_commands()

    def launch(self):
        """Continuously refresh the dashboard every REFRESH_INTERVAL seconds."""
        while True:
            try:
                self.fetch_data()
                self.display_all()
                console.rule("[bold green]Refreshing data automatically")
                for i in Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True).track(range(REFRESH_INTERVAL), description="Next refresh in..."):
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("[bold yellow]Keyboard interrupt detected. Exiting dashboard...[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]Error in dashboard: {e}[/bold red]")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        console.print("[bold green]Initializing Trading Dashboard...[/bold green]")
        dashboard = TradingDashboard()
        dashboard.launch()
    except KeyboardInterrupt:
        console.print("[bold yellow]Keyboard interrupt detected. Exiting dashboard...[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Critical error in dashboard: {e}[/bold red]") 