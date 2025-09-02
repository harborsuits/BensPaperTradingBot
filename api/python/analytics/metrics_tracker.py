import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Track and visualize trading performance metrics.
    """
    
    def __init__(self, log_dir="logs"):
        """Initialize the metrics tracker with a directory for log files."""
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        self.trades_file = os.path.join(log_dir, "trades.csv")
        self.positions_file = os.path.join(log_dir, "positions.csv")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics if file doesn't exist
        if not os.path.exists(self.metrics_file):
            self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize the metrics file with default values."""
        default_metrics = {
            "last_updated": datetime.now().isoformat(),
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "drawdown": {
                "current": 0.0,
                "max": 0.0,
                "start_date": None,
                "end_date": None
            },
            "daily_pnl": {},
            "strategy_performance": {}
        }
        
        with open(self.metrics_file, "w") as f:
            json.dump(default_metrics, f, indent=2)
    
    def update_metrics(self):
        """
        Update all trading metrics based on the latest trade data.
        
        This should be called regularly to keep metrics current.
        """
        if not os.path.exists(self.trades_file) or not os.path.exists(self.positions_file):
            logger.warning("Trade or position data not found, can't update metrics")
            return
        
        try:
            # Load trade and position data
            trades_df = pd.read_csv(self.trades_file)
            positions_df = pd.read_csv(self.positions_file)
            
            # Convert timestamps to datetime
            if not trades_df.empty and "timestamp" in trades_df.columns:
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            
            # Filter to closed positions
            closed_positions = positions_df[positions_df["status"] == "closed"].copy()
            if not closed_positions.empty:
                closed_positions["exit_date"] = pd.to_datetime(closed_positions["exit_date"])
            
            # Calculate basic metrics
            metrics = {
                "last_updated": datetime.now().isoformat(),
                "total_trades": len(closed_positions) if not closed_positions.empty else 0
            }
            
            if not closed_positions.empty:
                # Win rate
                winning_trades = len(closed_positions[closed_positions["pnl"] > 0])
                metrics["win_rate"] = winning_trades / len(closed_positions)
                
                # Calculate profit factor
                gross_profit = closed_positions[closed_positions["pnl"] > 0]["pnl"].sum()
                gross_loss = abs(closed_positions[closed_positions["pnl"] < 0]["pnl"].sum())
                metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Total P&L
                metrics["total_pnl"] = closed_positions["pnl"].sum()
                
                # Calculate equity curve and drawdown
                equity_curve, drawdown_data = self._calculate_equity_curve(closed_positions)
                metrics["drawdown"] = drawdown_data
                
                # Calculate daily P&L
                daily_pnl = self._calculate_daily_pnl(closed_positions)
                metrics["daily_pnl"] = daily_pnl
                
                # Strategy performance comparison
                strategy_performance = self._calculate_strategy_performance(closed_positions)
                metrics["strategy_performance"] = strategy_performance
            
            # Save updated metrics
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Trading metrics updated")
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            return None
    
    def _calculate_equity_curve(self, closed_positions):
        """
        Calculate the equity curve and drawdown metrics.
        
        Args:
            closed_positions (pd.DataFrame): DataFrame of closed positions
        
        Returns:
            tuple: (equity_curve, drawdown_data)
        """
        if closed_positions.empty:
            return {}, {"current": 0.0, "max": 0.0, "start_date": None, "end_date": None}
        
        # Sort positions by exit date
        closed_positions = closed_positions.sort_values("exit_date")
        
        # Initialize equity curve
        initial_capital = 10000  # Assume $10k starting capital
        equity = initial_capital
        equity_curve = {"dates": [], "equity": []}
        
        # Track drawdown
        running_max = equity
        current_drawdown = 0
        max_drawdown = 0
        drawdown_start = None
        drawdown_end = None
        
        # Calculate equity curve
        for _, position in closed_positions.iterrows():
            equity += position["pnl"]
            equity_curve["dates"].append(position["exit_date"].strftime("%Y-%m-%d"))
            equity_curve["equity"].append(equity)
            
            # Update drawdown calculations
            if equity > running_max:
                running_max = equity
            
            drawdown = (running_max - equity) / running_max if running_max > 0 else 0
            
            if drawdown > current_drawdown:
                current_drawdown = drawdown
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    drawdown_end = position["exit_date"].strftime("%Y-%m-%d")
            
            if drawdown == 0 and current_drawdown > 0:
                current_drawdown = 0
                drawdown_start = None
            
            if drawdown > 0 and drawdown_start is None:
                drawdown_start = position["exit_date"].strftime("%Y-%m-%d")
        
        drawdown_data = {
            "current": current_drawdown,
            "max": max_drawdown,
            "start_date": drawdown_start,
            "end_date": drawdown_end
        }
        
        return equity_curve, drawdown_data
    
    def _calculate_daily_pnl(self, closed_positions):
        """
        Calculate daily P&L from closed positions.
        
        Args:
            closed_positions (pd.DataFrame): DataFrame of closed positions
        
        Returns:
            dict: Daily P&L as a dictionary with dates as keys
        """
        if closed_positions.empty:
            return {}
        
        # Group by exit date and sum P&L
        closed_positions["exit_date_str"] = closed_positions["exit_date"].dt.strftime("%Y-%m-%d")
        daily_pnl = closed_positions.groupby("exit_date_str")["pnl"].sum().to_dict()
        
        return daily_pnl
    
    def _calculate_strategy_performance(self, closed_positions):
        """
        Calculate performance metrics by strategy.
        
        Args:
            closed_positions (pd.DataFrame): DataFrame of closed positions
        
        Returns:
            dict: Performance metrics by strategy
        """
        if closed_positions.empty:
            return {}
        
        # Handle positions with multiple strategies (comma-separated)
        all_strategies = []
        for strategies in closed_positions["strategy"]:
            all_strategies.extend(strategies.split(","))
        unique_strategies = list(set(all_strategies))
        
        strategy_performance = {}
        
        for strategy in unique_strategies:
            strategy = strategy.strip()
            # Filter positions that include this strategy
            strategy_positions = closed_positions[closed_positions["strategy"].str.contains(strategy)]
            
            if strategy_positions.empty:
                continue
            
            # Calculate metrics for this strategy
            total_trades = len(strategy_positions)
            winning_trades = len(strategy_positions[strategy_positions["pnl"] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = strategy_positions["pnl"].sum()
            avg_pnl = strategy_positions["pnl"].mean()
            
            gross_profit = strategy_positions[strategy_positions["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(strategy_positions[strategy_positions["pnl"] < 0]["pnl"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            strategy_performance[strategy] = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "profit_factor": profit_factor
            }
        
        return strategy_performance
    
    def get_metrics(self):
        """
        Get the current trading metrics.
        
        Returns:
            dict: Current trading metrics
        """
        if not os.path.exists(self.metrics_file):
            self._initialize_metrics()
        
        with open(self.metrics_file, "r") as f:
            metrics = json.load(f)
        
        return metrics
    
    def generate_equity_curve_chart(self, as_base64=True):
        """
        Generate an equity curve chart based on closed positions.
        
        Args:
            as_base64 (bool): If True, return the chart as a base64-encoded string
        
        Returns:
            str or plt.Figure: Equity curve chart as base64 string or matplotlib figure
        """
        metrics = self.get_metrics()
        
        if metrics.get("total_trades", 0) == 0:
            logger.warning("No trades available for equity curve chart")
            return None
        
        # Get equity curve data from the latest metrics
        equity_data = self._calculate_equity_curve(pd.read_csv(self.positions_file))[0]
        
        if not equity_data or "dates" not in equity_data or not equity_data["dates"]:
            logger.warning("No equity curve data available")
            return None
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        plt.plot(equity_data["dates"], equity_data["equity"], marker="o", markersize=3)
        
        # Set titles and labels
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Highlight max drawdown period if available
        drawdown = metrics.get("drawdown", {})
        if drawdown.get("start_date") and drawdown.get("end_date"):
            start_idx = equity_data["dates"].index(drawdown["start_date"]) if drawdown["start_date"] in equity_data["dates"] else None
            end_idx = equity_data["dates"].index(drawdown["end_date"]) if drawdown["end_date"] in equity_data["dates"] else None
            
            if start_idx is not None and end_idx is not None:
                plt.axvspan(start_idx, end_idx, alpha=0.2, color="red")
                plt.annotate(f"Max DD: {drawdown['max']:.2%}", 
                             xy=(end_idx, equity_data["equity"][end_idx]),
                             xytext=(10, -30),
                             textcoords="offset points",
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.tight_layout()
        
        if as_base64:
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()
            return image_base64
        else:
            return plt.gcf()
    
    def generate_daily_pnl_chart(self, as_base64=True):
        """
        Generate a daily P&L chart based on closed positions.
        
        Args:
            as_base64 (bool): If True, return the chart as a base64-encoded string
        
        Returns:
            str or plt.Figure: Daily P&L chart as base64 string or matplotlib figure
        """
        metrics = self.get_metrics()
        daily_pnl = metrics.get("daily_pnl", {})
        
        if not daily_pnl:
            logger.warning("No daily P&L data available")
            return None
        
        # Convert to lists for plotting
        dates = list(daily_pnl.keys())
        pnl_values = list(daily_pnl.values())
        
        # Create a color list based on P&L (green for positive, red for negative)
        colors = ["green" if pnl >= 0 else "red" for pnl in pnl_values]
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(dates, pnl_values, color=colors)
        
        # Set titles and labels
        plt.title("Daily P&L")
        plt.xlabel("Date")
        plt.ylabel("P&L ($)")
        plt.grid(True, axis="y")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add profit/loss totals
        total_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        total_loss = sum(pnl for pnl in pnl_values if pnl < 0)
        plt.figtext(0.01, 0.01, f"Total Profit: ${total_profit:.2f} | Total Loss: ${total_loss:.2f}")
        
        plt.tight_layout()
        
        if as_base64:
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()
            return image_base64
        else:
            return plt.gcf()
    
    def generate_strategy_comparison_chart(self, as_base64=True):
        """
        Generate a strategy comparison chart based on closed positions.
        
        Args:
            as_base64 (bool): If True, return the chart as a base64-encoded string
        
        Returns:
            str or plt.Figure: Strategy comparison chart as base64 string or matplotlib figure
        """
        metrics = self.get_metrics()
        strategy_performance = metrics.get("strategy_performance", {})
        
        if not strategy_performance:
            logger.warning("No strategy performance data available")
            return None
        
        # Extract data for plotting
        strategies = list(strategy_performance.keys())
        win_rates = [strategy_performance[s]["win_rate"] for s in strategies]
        profit_factors = [min(strategy_performance[s]["profit_factor"], 5) for s in strategies]  # Cap at 5 for readability
        total_pnls = [strategy_performance[s]["total_pnl"] for s in strategies]
        
        # Create the chart with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # Plot win rates
        ax1.bar(strategies, win_rates, color="blue", alpha=0.7)
        ax1.set_title("Win Rate by Strategy")
        ax1.set_ylabel("Win Rate")
        ax1.set_ylim(0, 1)
        ax1.grid(True, axis="y")
        
        # Plot profit factors
        ax2.bar(strategies, profit_factors, color="green", alpha=0.7)
        ax2.set_title("Profit Factor by Strategy")
        ax2.set_ylabel("Profit Factor")
        ax2.grid(True, axis="y")
        
        # Plot total P&L
        bars = ax3.bar(strategies, total_pnls, color=["green" if pnl >= 0 else "red" for pnl in total_pnls], alpha=0.7)
        ax3.set_title("Total P&L by Strategy")
        ax3.set_ylabel("Total P&L ($)")
        ax3.grid(True, axis="y")
        
        # Rotate strategy names for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if as_base64:
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()
            return image_base64
        else:
            return fig 