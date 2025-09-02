#!/usr/bin/env python3
"""
Test Status Tracker

This module implements tracking and monitoring of strategy performance during
forward testing, with special focus on proprietary trading firm requirements.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_status_tracker')


class TestStatusTracker:
    """
    Tracks and monitors strategy performance during forward testing.
    
    This class maintains performance metrics, monitors compliance with
    proprietary trading firm limits, and generates alerts when thresholds
    are approached or exceeded.
    """
    
    def __init__(self, 
                risk_profile_path: Optional[str] = None,
                output_dir: str = "./forward_test_results",
                strategy_id: Optional[str] = None):
        """
        Initialize the test status tracker.
        
        Args:
            risk_profile_path: Path to risk profile YAML file
            output_dir: Directory for output files
            strategy_id: Optional strategy identifier
        """
        self.strategy_id = strategy_id
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default thresholds (will be overridden by risk profile)
        self.thresholds = {
            'max_drawdown': 5.0,
            'daily_loss_limit': 3.0,
            'profit_target': 8.0,
            'drawdown_warning': 3.0,
            'drawdown_critical': 4.0,
            'daily_loss_warning': 2.0,
            'daily_loss_critical': 2.5
        }
        
        # Load risk profile if provided
        if risk_profile_path and os.path.exists(risk_profile_path):
            self._load_risk_profile(risk_profile_path)
        
        # Initialize performance tracking
        self.starting_equity = 100.0  # 100% as base
        self.current_equity = self.starting_equity
        self.equity_curve = {}
        self.daily_pnl = {}
        self.drawdowns = {}
        self.max_drawdown = 0.0
        self.peak_equity = self.starting_equity
        self.daily_metrics = {}
        self.trades = []
        self.test_start_date = None
        self.last_update_date = None
        self.profitable_days = 0
        self.losing_days = 0
        self.alerts = []
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if strategy_id:
            self.log_file = os.path.join(output_dir, f"{strategy_id}_status_{timestamp}.log")
        else:
            self.log_file = os.path.join(output_dir, f"test_status_{timestamp}.log")
        
        # Add file handler to logger
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("Test status tracker initialized")
    
    def _load_risk_profile(self, profile_path: str):
        """
        Load risk profile from YAML file.
        
        Args:
            profile_path: Path to risk profile file
        """
        try:
            with open(profile_path, 'r') as f:
                risk_profile = yaml.safe_load(f)
            
            # Load thresholds from risk profile
            if 'criteria' in risk_profile:
                for key, value in risk_profile['criteria'].items():
                    if key in self.thresholds:
                        self.thresholds[key] = value
            
            # Load alert thresholds
            if 'forward_testing' in risk_profile and 'alert_thresholds' in risk_profile['forward_testing']:
                alert_thresholds = risk_profile['forward_testing']['alert_thresholds']
                
                for key, value in alert_thresholds.items():
                    if key in self.thresholds:
                        self.thresholds[key] = value
            
            logger.info(f"Loaded risk profile from {profile_path}")
            
        except Exception as e:
            logger.error(f"Failed to load risk profile: {e}")
    
    def start_test(self, initial_equity: float = 100.0):
        """
        Start a new forward test.
        
        Args:
            initial_equity: Initial equity value (100.0 = 100%)
        """
        self.starting_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.equity_curve = {}
        self.daily_pnl = {}
        self.drawdowns = {}
        self.max_drawdown = 0.0
        self.daily_metrics = {}
        self.trades = []
        self.test_start_date = datetime.now().date()
        self.last_update_date = self.test_start_date
        self.profitable_days = 0
        self.losing_days = 0
        self.alerts = []
        
        logger.info(f"Started new forward test with initial equity: {initial_equity}")
        
        # Record initial equity
        self.track_equity(self.test_start_date, initial_equity)
    
    def track_equity(self, day: Union[datetime, date], equity: float) -> None:
        """
        Track equity value for a specific day.
        
        Args:
            day: Date to record equity for
            equity: Equity value
        """
        if isinstance(day, datetime):
            day = day.date()
        
        # Record equity
        self.equity_curve[day] = equity
        self.current_equity = equity
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        drawdown_pct = self._calculate_drawdown(equity)
        self.drawdowns[day] = drawdown_pct
        
        # Update maximum drawdown
        if drawdown_pct > self.max_drawdown:
            self.max_drawdown = drawdown_pct
            logger.info(f"New maximum drawdown: {self.max_drawdown:.2f}%")
        
        # Update last update date
        self.last_update_date = day
        
        # Check drawdown thresholds
        self._check_drawdown_thresholds(drawdown_pct)
    
    def _calculate_drawdown(self, current_equity: float) -> float:
        """
        Calculate current drawdown percentage.
        
        Args:
            current_equity: Current equity value
            
        Returns:
            Drawdown percentage
        """
        if self.peak_equity <= 0:
            return 0.0
        
        drawdown_pct = 100 * (self.peak_equity - current_equity) / self.peak_equity
        return drawdown_pct
    
    def _check_drawdown_thresholds(self, drawdown_pct: float) -> None:
        """
        Check if drawdown has reached warning or critical thresholds.
        
        Args:
            drawdown_pct: Current drawdown percentage
        """
        # Check critical threshold first
        if drawdown_pct >= self.thresholds['drawdown_critical']:
            self._generate_alert(
                level="CRITICAL",
                alert_type="DRAWDOWN",
                message=f"Drawdown has reached CRITICAL level: {drawdown_pct:.2f}% "
                        f"(threshold: {self.thresholds['drawdown_critical']}%)",
                value=drawdown_pct,
                threshold=self.thresholds['drawdown_critical']
            )
        # Check warning threshold
        elif drawdown_pct >= self.thresholds['drawdown_warning']:
            self._generate_alert(
                level="WARNING",
                alert_type="DRAWDOWN",
                message=f"Drawdown has reached WARNING level: {drawdown_pct:.2f}% "
                        f"(threshold: {self.thresholds['drawdown_warning']}%)",
                value=drawdown_pct,
                threshold=self.thresholds['drawdown_warning']
            )
    
    def track_daily_pnl(self, day: Union[datetime, date], pnl_pct: float) -> None:
        """
        Track daily profit and loss.
        
        Args:
            day: Date of P&L
            pnl_pct: Percentage P&L for the day
        """
        if isinstance(day, datetime):
            day = day.date()
        
        # Record daily P&L
        self.daily_pnl[day] = pnl_pct
        
        # Update profitable/losing days counts
        if pnl_pct > 0:
            self.profitable_days += 1
        elif pnl_pct < 0:
            self.losing_days += 1
        
        # Calculate new equity
        prev_day = day - timedelta(days=1)
        prev_equity = self.equity_curve.get(prev_day, self.starting_equity)
        new_equity = prev_equity * (1 + pnl_pct / 100)
        
        # Update equity curve
        self.track_equity(day, new_equity)
        
        # Check daily loss threshold
        if pnl_pct < 0:
            abs_loss = abs(pnl_pct)
            self._check_daily_loss_thresholds(abs_loss)
        
        logger.info(f"Day {(day - self.test_start_date).days + 1}: "
                   f"P&L = {pnl_pct:.2f}%, Equity = {new_equity:.2f}, "
                   f"Drawdown = {self.drawdowns[day]:.2f}%")
    
    def _check_daily_loss_thresholds(self, loss_pct: float) -> None:
        """
        Check if daily loss has reached warning or critical thresholds.
        
        Args:
            loss_pct: Daily loss percentage (positive value)
        """
        # Check critical threshold first
        if loss_pct >= self.thresholds['daily_loss_critical']:
            self._generate_alert(
                level="CRITICAL",
                alert_type="DAILY_LOSS",
                message=f"Daily loss has reached CRITICAL level: {loss_pct:.2f}% "
                        f"(threshold: {self.thresholds['daily_loss_critical']}%)",
                value=loss_pct,
                threshold=self.thresholds['daily_loss_critical']
            )
        # Check warning threshold
        elif loss_pct >= self.thresholds['daily_loss_warning']:
            self._generate_alert(
                level="WARNING",
                alert_type="DAILY_LOSS",
                message=f"Daily loss has reached WARNING level: {loss_pct:.2f}% "
                        f"(threshold: {self.thresholds['daily_loss_warning']}%)",
                value=loss_pct,
                threshold=self.thresholds['daily_loss_warning']
            )
    
    def _generate_alert(self, 
                       level: str, 
                       alert_type: str, 
                       message: str,
                       value: float,
                       threshold: float) -> None:
        """
        Generate an alert and log it.
        
        Args:
            level: Alert level (INFO, WARNING, CRITICAL)
            alert_type: Type of alert
            message: Alert message
            value: Current value
            threshold: Threshold value
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'type': alert_type,
            'message': message,
            'value': value,
            'threshold': threshold
        }
        
        self.alerts.append(alert)
        
        if level == "WARNING":
            logger.warning(message)
        elif level == "CRITICAL":
            logger.critical(message)
        else:
            logger.info(message)
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a completed trade.
        
        Args:
            trade: Trade information dictionary
        """
        self.trades.append(trade)
        
        # Log trade
        direction = "LONG" if trade.get('direction', 1) > 0 else "SHORT"
        pnl = trade.get('pnl', 0)
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        
        logger.info(f"Trade {len(self.trades)}: {direction} {trade.get('symbol', '')} "
                   f"P&L = {pnl_str}, Return = {trade.get('return_pct', 0):.2f}%")
    
    def check_thresholds(self) -> Dict[str, bool]:
        """
        Check if strategy is compliant with prop firm thresholds.
        
        Returns:
            Dictionary of threshold compliance results
        """
        # Calculate test duration
        if self.test_start_date is None or self.last_update_date is None:
            logger.warning("Cannot check thresholds: Test not started or no data recorded")
            return {'valid': False}
        
        test_duration = (self.last_update_date - self.test_start_date).days + 1
        
        # Calculate current gain/loss
        if self.starting_equity > 0:
            total_return_pct = 100 * (self.current_equity - self.starting_equity) / self.starting_equity
        else:
            total_return_pct = 0
        
        # Calculate profitable days percentage
        total_days = self.profitable_days + self.losing_days
        profitable_days_pct = 100 * self.profitable_days / total_days if total_days > 0 else 0
        
        # Check thresholds
        threshold_results = {
            'meets_max_drawdown': self.max_drawdown <= self.thresholds['max_drawdown'],
            'meets_daily_loss': all(abs(pnl) <= self.thresholds['daily_loss_limit'] 
                                   for pnl in self.daily_pnl.values() if pnl < 0),
            'meets_profit_target': total_return_pct >= self.thresholds['profit_target'],
            'current_return': total_return_pct,
            'max_drawdown': self.max_drawdown,
            'test_duration_days': test_duration,
            'profitable_days_pct': profitable_days_pct,
            'valid': True
        }
        
        # Log threshold check
        logger.info(f"Threshold check: Return = {total_return_pct:.2f}%, "
                   f"Max Drawdown = {self.max_drawdown:.2f}%, "
                   f"Duration = {test_duration} days, "
                   f"Profitable Days = {profitable_days_pct:.2f}%")
        
        return threshold_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a complete performance summary.
        
        Returns:
            Summary dictionary
        """
        if not self.equity_curve:
            return {'valid': False, 'message': 'No data recorded'}
        
        # Calculate metrics
        test_duration = (self.last_update_date - self.test_start_date).days + 1
        total_return_pct = 100 * (self.current_equity - self.starting_equity) / self.starting_equity
        
        # Calculate average daily return
        daily_returns = list(self.daily_pnl.values())
        avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        
        # Calculate return volatility
        return_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = 0
        if return_volatility > 0:
            sharpe_ratio = (avg_daily_return / return_volatility) * np.sqrt(252)
        
        # Calculate profitable days percentage
        total_days = self.profitable_days + self.losing_days
        profitable_days_pct = 100 * self.profitable_days / total_days if total_days > 0 else 0
        
        # Calculate worst day
        worst_day = min(self.daily_pnl.items(), key=lambda x: x[1]) if self.daily_pnl else (None, 0)
        
        # Calculate best day
        best_day = max(self.daily_pnl.items(), key=lambda x: x[1]) if self.daily_pnl else (None, 0)
        
        # Generate summary
        summary = {
            'test_start_date': self.test_start_date.isoformat() if self.test_start_date else None,
            'test_end_date': self.last_update_date.isoformat() if self.last_update_date else None,
            'test_duration_days': test_duration,
            'starting_equity': self.starting_equity,
            'ending_equity': self.current_equity,
            'total_return_pct': total_return_pct,
            'max_drawdown': self.max_drawdown,
            'avg_daily_return': avg_daily_return,
            'return_volatility': return_volatility,
            'sharpe_ratio': sharpe_ratio,
            'profitable_days': self.profitable_days,
            'losing_days': self.losing_days,
            'profitable_days_pct': profitable_days_pct,
            'worst_day': {
                'date': worst_day[0].isoformat() if worst_day[0] else None,
                'loss': worst_day[1]
            },
            'best_day': {
                'date': best_day[0].isoformat() if best_day[0] else None,
                'gain': best_day[1]
            },
            'threshold_check': self.check_thresholds(),
            'alert_count': len(self.alerts),
            'valid': True
        }
        
        # Add trade statistics if trades are recorded
        if self.trades:
            trade_pnls = [trade.get('pnl', 0) for trade in self.trades]
            winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
            losing_trades = sum(1 for pnl in trade_pnls if pnl < 0)
            
            trade_stats = {
                'total_trades': len(self.trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': 100 * winning_trades / len(self.trades) if self.trades else 0,
                'avg_winner': sum(pnl for pnl in trade_pnls if pnl > 0) / winning_trades if winning_trades else 0,
                'avg_loser': sum(pnl for pnl in trade_pnls if pnl < 0) / losing_trades if losing_trades else 0
            }
            
            # Calculate profit factor
            total_gain = sum(pnl for pnl in trade_pnls if pnl > 0)
            total_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
            
            trade_stats['profit_factor'] = total_gain / total_loss if total_loss else float('inf')
            
            summary['trade_statistics'] = trade_stats
        
        return summary
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save test results to a file.
        
        Args:
            filename: Optional filename for results
            
        Returns:
            Path to saved file
        """
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.strategy_id:
                filename = f"{self.strategy_id}_results_{timestamp}.json"
            else:
                filename = f"test_results_{timestamp}.json"
        
        # Full path to results file
        results_path = os.path.join(self.output_dir, filename)
        
        # Generate summary
        summary = self.generate_summary()
        
        # Convert dates to strings in equity curve and drawdowns
        equity_curve_str = {date.isoformat(): value for date, value in self.equity_curve.items()}
        drawdowns_str = {date.isoformat(): value for date, value in self.drawdowns.items()}
        daily_pnl_str = {date.isoformat(): value for date, value in self.daily_pnl.items()}
        
        # Prepare results data
        results = {
            'summary': summary,
            'equity_curve': equity_curve_str,
            'drawdowns': drawdowns_str,
            'daily_pnl': daily_pnl_str,
            'alerts': self.alerts,
            'trades': self.trades
        }
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved test results to {results_path}")
        
        return results_path
    
    def plot_equity_curve(self, filename: Optional[str] = None) -> str:
        """
        Generate and save equity curve plot.
        
        Args:
            filename: Optional filename for plot
            
        Returns:
            Path to saved file
        """
        if not self.equity_curve:
            logger.warning("Cannot plot equity curve: No data recorded")
            return ""
        
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.strategy_id:
                filename = f"{self.strategy_id}_equity_{timestamp}.png"
            else:
                filename = f"equity_curve_{timestamp}.png"
        
        # Full path to plot file
        plot_path = os.path.join(self.output_dir, filename)
        
        # Sort dates
        sorted_dates = sorted(self.equity_curve.keys())
        equity_values = [self.equity_curve[date] for date in sorted_dates]
        drawdown_values = [self.drawdowns[date] for date in sorted_dates]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(sorted_dates, equity_values, 'b-', linewidth=2)
        ax1.set_title('Forward Test Equity Curve')
        ax1.set_ylabel('Equity (%)')
        ax1.grid(True)
        
        # Add starting value line
        ax1.axhline(y=self.starting_equity, color='g', linestyle='--', alpha=0.6)
        
        # Plot drawdown
        ax2.fill_between(sorted_dates, 0, drawdown_values, color='r', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        # Rotate date labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved equity curve plot to {plot_path}")
        
        return plot_path


# Main execution for testing
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Test the status tracker")
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=10,
        help="Number of days to simulate"
    )
    
    parser.add_argument(
        "--risk-profile", 
        type=str, 
        default=None,
        help="Path to risk profile YAML file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./forward_test_results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = TestStatusTracker(args.risk_profile, args.output, "test_strategy")
    
    # Start test
    tracker.start_test()
    
    # Simulate days
    start_date = datetime.now().date() - timedelta(days=args.days)
    
    for i in range(args.days):
        day = start_date + timedelta(days=i)
        
        # Generate random daily P&L between -3% and +2%
        daily_pnl = random.uniform(-3.0, 2.0)
        
        # Record daily P&L
        tracker.track_daily_pnl(day, daily_pnl)
        
        # Simulate some trades
        for _ in range(random.randint(0, 3)):
            trade = {
                'symbol': random.choice(['ES', 'NQ', 'CL', 'GC']),
                'entry_time': datetime.combine(day, datetime.min.time()) + timedelta(hours=random.randint(9, 16)),
                'exit_time': datetime.combine(day, datetime.min.time()) + timedelta(hours=random.randint(9, 16), minutes=random.randint(30, 120)),
                'direction': random.choice([1, -1]),
                'entry_price': round(random.uniform(100, 5000), 2),
                'exit_price': round(random.uniform(100, 5000), 2),
                'pnl': round(random.uniform(-2.0, 3.0), 2),
                'return_pct': round(random.uniform(-1.0, 1.5), 2)
            }
            
            tracker.record_trade(trade)
    
    # Generate and save summary
    summary = tracker.generate_summary()
    print(json.dumps(summary, indent=2))
    
    # Save results
    results_path = tracker.save_results()
    
    # Generate equity curve plot
    plot_path = tracker.plot_equity_curve()
