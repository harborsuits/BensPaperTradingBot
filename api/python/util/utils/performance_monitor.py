"""
Performance Monitoring System

This module provides monitoring capabilities for the strategy rotation system,
tracking performance metrics and sending alerts when thresholds are crossed.
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from trading_bot.utils.db_models import AllocationDatabase

# Configure logging
logger = logging.getLogger("performance_monitor")

class PerformanceMonitor:
    """
    Monitors the performance of trading strategies and sends alerts when thresholds are crossed.
    """
    
    def __init__(self, 
                strategies: List[str],
                metrics_db_path: Optional[str] = None,
                telegram_token: Optional[str] = None,
                telegram_chat_id: Optional[str] = None,
                email_config: Optional[Dict[str, str]] = None):
        """
        Initialize the performance monitor.
        
        Args:
            strategies: List of strategies to monitor
            metrics_db_path: Path to the metrics database
            telegram_token: Telegram bot token for alerts
            telegram_chat_id: Telegram chat ID for alerts
            email_config: Email configuration for alerts
        """
        self.strategies = strategies
        self.db = AllocationDatabase(metrics_db_path)
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.email_config = email_config
        
        # Default thresholds
        self.thresholds = {
            "drawdown_alert": -0.10,  # 10% drawdown
            "daily_loss_alert": -0.05,  # 5% daily loss
            "strategy_underperformance_days": 14,  # Days of underperformance to trigger alert
            "volatility_alert": 0.25,  # 25% annualized volatility
        }
        
        # Track performance history
        self.performance_history = {strategy: [] for strategy in strategies}
        
        logger.info(f"Performance monitor initialized for {len(strategies)} strategies")
    
    def set_threshold(self, threshold_name: str, value: float) -> None:
        """
        Set a monitoring threshold.
        
        Args:
            threshold_name: Name of the threshold
            value: Threshold value
        """
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            logger.info(f"Set {threshold_name} threshold to {value}")
        else:
            logger.warning(f"Unknown threshold: {threshold_name}")
    
    def update_metrics(self, 
                     strategy: str, 
                     metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy: Strategy name
            metrics: Performance metrics dictionary
        """
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy}")
            return
        
        # Add timestamp to metrics
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Add to history
        self.performance_history[strategy].append(metrics)
        
        # Save to database
        self.db.save_performance_metrics(strategy, metrics)
        
        # Check for threshold violations
        self._check_thresholds(strategy, metrics)
        
        logger.info(f"Updated performance metrics for {strategy}")
    
    def _check_thresholds(self, 
                        strategy: str, 
                        metrics: Dict[str, float]) -> None:
        """
        Check if any thresholds have been violated.
        
        Args:
            strategy: Strategy name
            metrics: Performance metrics
        """
        alerts = []
        
        # Check drawdown
        if metrics.get("max_drawdown", 0) <= self.thresholds["drawdown_alert"]:
            alerts.append(f"Drawdown threshold violated: {metrics['max_drawdown']:.2%}")
        
        # Check daily return
        if metrics.get("daily_return", 0) <= self.thresholds["daily_loss_alert"]:
            alerts.append(f"Daily loss threshold violated: {metrics['daily_return']:.2%}")
        
        # Check volatility
        if metrics.get("volatility", 0) >= self.thresholds["volatility_alert"]:
            alerts.append(f"Volatility threshold violated: {metrics['volatility']:.2%}")
        
        # Check for persistent underperformance
        if len(self.performance_history[strategy]) >= self.thresholds["strategy_underperformance_days"]:
            recent_metrics = self.performance_history[strategy][-int(self.thresholds["strategy_underperformance_days"]):]
            avg_return = np.mean([m.get("daily_return", 0) for m in recent_metrics])
            
            if avg_return < 0:
                days = int(self.thresholds["strategy_underperformance_days"])
                alerts.append(f"Strategy underperforming for {days} days: Avg return {avg_return:.2%}")
        
        # Send alerts if any thresholds violated
        if alerts:
            self._send_alert(strategy, alerts)
    
    def _send_alert(self, 
                   strategy: str, 
                   alerts: List[str]) -> None:
        """
        Send alerts for threshold violations.
        
        Args:
            strategy: Strategy name
            alerts: List of alert messages
        """
        title = f"⚠️ Strategy Alert: {strategy}"
        message = "\n".join(alerts)
        
        logger.warning(f"{title}\n{message}")
        
        # Send Telegram alert if configured
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram_alert(title, message)
        
        # Send email alert if configured
        if self.email_config:
            self._send_email_alert(title, message)
    
    def _send_telegram_alert(self, 
                           title: str, 
                           message: str) -> bool:
        """
        Send an alert via Telegram.
        
        Args:
            title: Alert title
            message: Alert message
            
        Returns:
            Boolean indicating success
        """
        try:
            text = f"{title}\n\n{message}"
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {str(e)}")
            return False
    
    def _send_email_alert(self, 
                        title: str, 
                        message: str) -> bool:
        """
        Send an alert via email.
        
        Args:
            title: Alert title
            message: Alert message
            
        Returns:
            Boolean indicating success
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            if not self.email_config:
                logger.error("Email configuration missing")
                return False
            
            sender = self.email_config.get("sender")
            receiver = self.email_config.get("receiver")
            smtp_server = self.email_config.get("smtp_server")
            smtp_port = int(self.email_config.get("smtp_port", 587))
            username = self.email_config.get("username")
            password = self.email_config.get("password")
            
            if not all([sender, receiver, smtp_server, username, password]):
                logger.error("Incomplete email configuration")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = receiver
            msg["Subject"] = title
            
            # Add message body
            msg.attach(MIMEText(message, "plain"))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def generate_performance_report(self, 
                                  days: int = 30,
                                  save_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a performance report for all strategies.
        
        Args:
            days: Number of days to include in the report
            save_path: Path to save the report (None to return as string)
            
        Returns:
            Report as a string if save_path is None, otherwise path to saved report
        """
        # Fetch performance data for each strategy
        strategies_data = {}
        start_date = datetime.now() - timedelta(days=days)
        
        for strategy in self.strategies:
            metrics_history = []
            
            # Fetch from DB (in a real implementation)
            # This is a simplified version
            for entry in self.performance_history[strategy]:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                if entry_date >= start_date:
                    metrics_history.append(entry)
            
            if metrics_history:
                strategies_data[strategy] = metrics_history
        
        if not strategies_data:
            logger.warning("No performance data available for report")
            return None
        
        # Generate the report
        report = [
            f"Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Period: Last {days} days",
            "",
            "Summary:",
        ]
        
        # Calculate summary metrics
        summary_table = []
        for strategy, history in strategies_data.items():
            returns = [entry.get("daily_return", 0) for entry in history]
            cumulative_return = (1 + pd.Series(returns)).prod() - 1
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            max_dd = min([entry.get("max_drawdown", 0) for entry in history], default=0)
            
            summary_table.append({
                "Strategy": strategy,
                "Return": cumulative_return,
                "Volatility": volatility,
                "Max Drawdown": max_dd,
                "Sharpe": cumulative_return / volatility if volatility > 0 else 0
            })
        
        # Add summary table to report
        report.append("\nStrategy Performance:")
        report.append("-" * 80)
        report.append(f"{'Strategy':<20} {'Return':>10} {'Volatility':>12} {'Max Drawdown':>15} {'Sharpe':>10}")
        report.append("-" * 80)
        
        for entry in summary_table:
            report.append(
                f"{entry['Strategy']:<20} "
                f"{entry['Return']:>9.2%} "
                f"{entry['Volatility']:>11.2%} "
                f"{entry['Max Drawdown']:>14.2%} "
                f"{entry['Sharpe']:>9.2f}"
            )
        
        report.append("-" * 80)
        report.append("")
        
        # Add allocation history if available
        allocation_history = self._get_allocation_history(days)
        if allocation_history:
            report.append("\nAllocation Changes:")
            report.append("-" * 80)
            report.append(f"{'Date':<12} {'Market Regime':<15} {'Allocation Changes'}")
            report.append("-" * 80)
            
            for entry in allocation_history:
                date = entry.get("timestamp", "").split("T")[0]
                regime = entry.get("market_regime", "unknown")
                allocations = entry.get("allocations", {})
                
                # Format allocation summary
                alloc_summary = ", ".join([f"{s}: {a:.1f}%" for s, a in allocations.items()])
                
                report.append(f"{date:<12} {regime:<15} {alloc_summary}")
            
            report.append("-" * 80)
        
        # Join the report
        report_text = "\n".join(report)
        
        # Save or return the report
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)
            return save_path
        else:
            return report_text
    
    def _get_allocation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get allocation history for the past N days.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of allocation history entries
        """
        return self.db.get_allocation_history(limit=10)
    
    def plot_strategy_performance(self, 
                                days: int = 30,
                                save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot strategy performance over time.
        
        Args:
            days: Number of days to include in the plot
            save_path: Path to save the plot (None to display)
            
        Returns:
            Matplotlib Figure if save_path is provided, None otherwise
        """
        # Fetch performance data for each strategy
        start_date = datetime.now() - timedelta(days=days)
        
        # Prepare data for plotting
        strategy_returns = {}
        strategy_drawdowns = {}
        
        for strategy in self.strategies:
            returns = []
            drawdowns = []
            dates = []
            
            for entry in self.performance_history[strategy]:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                if entry_date >= start_date:
                    returns.append(entry.get("daily_return", 0))
                    drawdowns.append(entry.get("max_drawdown", 0))
                    dates.append(entry_date)
            
            if dates:
                strategy_returns[strategy] = (dates, returns)
                strategy_drawdowns[strategy] = (dates, drawdowns)
        
        if not strategy_returns:
            logger.warning("No performance data available for plotting")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot cumulative returns
        ax1.set_title(f"Strategy Performance - Last {days} Days")
        ax1.set_ylabel("Cumulative Return (%)")
        
        for strategy, (dates, returns) in strategy_returns.items():
            # Convert to pandas for easier manipulation
            returns_df = pd.Series(returns, index=pd.DatetimeIndex(dates))
            
            # Calculate cumulative returns
            cumulative = (1 + returns_df).cumprod() - 1
            
            # Plot
            ax1.plot(cumulative.index, cumulative * 100, label=strategy)
        
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdowns
        ax2.set_title("Maximum Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        
        for strategy, (dates, drawdowns) in strategy_drawdowns.items():
            # Convert to pandas
            drawdowns_df = pd.Series(drawdowns, index=pd.DatetimeIndex(dates))
            
            # Plot
            ax2.plot(drawdowns_df.index, drawdowns_df * 100, label=strategy)
        
        ax2.legend()
        ax2.grid(True)
        
        # Format the plot
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return fig
        else:
            plt.show()
            return None

class StrategyMonitor:
    """
    Higher-level monitor that integrates with trading system to provide
    real-time performance monitoring and alerts.
    """
    
    def __init__(self, 
                strategy_rotator,
                db_path: Optional[str] = None,
                config_path: Optional[str] = None):
        """
        Initialize the strategy monitor.
        
        Args:
            strategy_rotator: IntegratedStrategyRotator instance
            db_path: Path to the database
            config_path: Path to monitor configuration
        """
        self.rotator = strategy_rotator
        self.strategies = strategy_rotator.strategies
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Initialize the performance monitor
        self.monitor = PerformanceMonitor(
            strategies=self.strategies,
            metrics_db_path=db_path,
            telegram_token=config.get("telegram_token", os.getenv("TELEGRAM_BOT_TOKEN")),
            telegram_chat_id=config.get("telegram_chat_id", os.getenv("TELEGRAM_CHAT_ID")),
            email_config=config.get("email_config")
        )
        
        # Set custom thresholds if provided
        for threshold, value in config.get("thresholds", {}).items():
            self.monitor.set_threshold(threshold, value)
        
        logger.info("Strategy monitor initialized")
    
    def update_metrics_from_rotator(self) -> None:
        """
        Update performance metrics from the strategy rotator.
        """
        # Get performance data from rotator
        performance_data = self.rotator.get_strategy_performance()
        
        # Update each strategy's metrics in the monitor
        for strategy, metrics in performance_data.items():
            self.monitor.update_metrics(strategy, metrics)
        
        logger.info("Updated metrics from rotator")
    
    def monitor_rotation(self, 
                        rotation_result: Dict[str, Any]) -> None:
        """
        Monitor a strategy rotation event.
        
        Args:
            rotation_result: Result from strategy rotation
        """
        if not rotation_result.get("rotated", False):
            return
        
        # Update metrics
        self.update_metrics_from_rotator()
        
        # Record the allocation change
        regime = rotation_result.get("regime", "unknown")
        allocations = rotation_result.get("new_allocations", {})
        
        # Log significant allocation changes
        significant_changes = []
        for strategy, allocation in allocations.items():
            prev_allocation = rotation_result.get("previous_allocations", {}).get(strategy, 0)
            change = allocation - prev_allocation
            
            if abs(change) >= 10:  # 10% change threshold
                significant_changes.append(f"{strategy}: {prev_allocation:.1f}% → {allocation:.1f}% ({change:+.1f}%)")
        
        if significant_changes:
            logger.info(f"Significant allocation changes in {regime} regime:")
            for change in significant_changes:
                logger.info(f"  {change}")
    
    def generate_daily_report(self, 
                            save_dir: str = "./reports") -> Optional[str]:
        """
        Generate a daily performance report.
        
        Args:
            save_dir: Directory to save the report
            
        Returns:
            Path to the saved report or None if generation failed
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate report filename
        today = datetime.now().strftime("%Y-%m-%d")
        report_path = f"{save_dir}/performance_report_{today}.txt"
        
        # Generate and save the report
        return self.monitor.generate_performance_report(days=30, save_path=report_path) 