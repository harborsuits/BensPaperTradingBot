#!/usr/bin/env python
"""
Run Nightly Recap

This script runs the nightly performance recap process after market close.
It analyzes trading performance, generates reports, sends email summaries,
and can trigger optimization jobs if configured to do so.

Usage:
    python -m trading_bot.monitoring.run_nightly_recap [--date YYYY-MM-DD] [--config CONFIG_PATH]
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import components
from trading_bot.monitoring.analyze_strategy import analyze_strategy_performance, generate_insights
from trading_bot.monitoring.recap_reporting import create_performance_report, send_email_report, generate_html_report
from trading_bot.dashboard.paper_trading_dashboard import PaperTradingDashboard
from trading_bot.execution.adaptive_paper_integration import get_paper_trading_instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'nightly_recap_runner.log'))
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        'email': {
            'enabled': True,
            'server': 'smtp.gmail.com',
            'port': 587,
            'username': '',
            'password': '',
            'recipients': []
        },
        'benchmarks': ['SPY', 'VIX'],
        'thresholds': {
            'sharpe_ratio': 0.5,
            'win_rate': 45.0,  # percentage
            'max_drawdown': -10.0,  # percentage
            'rolling_windows': [5, 10, 20, 60]  # days
        },
        'optimization': {
            'auto_optimize': False,
            'optimization_threshold': -20.0  # percentage deterioration
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
            return default_config
    else:
        logger.info("Using default configuration")
        return default_config

def run_nightly_recap(force_date: str = None, config_path: str = None):
    """
    Run the nightly recap process
    
    Args:
        force_date: Optional date to generate recap for (YYYY-MM-DD)
        config_path: Path to configuration file
    
    Returns:
        Tuple of (success, message)
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting nightly recap at {start_time}")
        
        # Load configuration
        config = load_config(config_path)
        
        # Create output directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('reports/nightly', exist_ok=True)
        
        # Initialize components
        dashboard = PaperTradingDashboard()
        paper_trading = get_paper_trading_instance()
        
        # Load trading data
        if force_date:
            logger.info(f"Using forced date: {force_date}")
            # TODO: Implement loading historical data for specific date
            success = dashboard.load_data()
        else:
            # Load latest data
            success = dashboard.load_live_data()
            if not success:
                success = dashboard.load_data()
        
        if not success:
            return False, "Failed to load trading data"
        
        # Get performance data
        equity_history = dashboard.equity_history
        trade_history = dashboard.trade_history
        
        if equity_history is None or equity_history.empty:
            return False, "No equity data available"
        
        # Calculate today's results
        today_results = calculate_today_results(equity_history, trade_history)
        
        # Fetch benchmark data
        benchmark_data = fetch_benchmark_data(today_results['date'], config['benchmarks'])
        
        # Get strategy metrics
        strategy_metrics = calculate_strategy_metrics(dashboard, config['thresholds'])
        
        # Analyze strategy performance
        alerts = analyze_strategy_performance(strategy_metrics, config['thresholds'])
        
        # Generate insights
        suggestions = generate_insights(alerts, config['optimization']['optimization_threshold'])
        
        # Create performance report
        report_path = create_performance_report(
            today_results, 
            benchmark_data, 
            alerts, 
            suggestions
        )
        
        # Send email report if enabled
        if config['email']['enabled'] and config['email']['recipients']:
            # Generate HTML report
            html_report = generate_html_report(
                today_results,
                benchmark_data,
                alerts,
                suggestions,
                today_results['date']
            )
            
            # Send email
            email_sent = send_email_report(
                config['email'],
                html_report,
                f"Trading Performance Recap - {today_results['date'].strftime('%Y-%m-%d')}"
            )
            
            if not email_sent:
                logger.warning("Failed to send email report")
        
        # Auto-optimize if configured
        if (config['optimization']['auto_optimize'] and 
            any(alert['action_required'] for alert in alerts)):
            logger.info("Triggering optimization jobs...")
            trigger_optimization_jobs(alerts, suggestions)
        
        end_time = datetime.now()
        logger.info(f"Nightly recap completed in {end_time - start_time}")
        
        return True, "Nightly recap completed successfully"
        
    except Exception as e:
        error_msg = f"Error in nightly recap: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def calculate_today_results(equity_history, trade_history):
    """
    Calculate today's trading results
    
    Args:
        equity_history: Equity history DataFrame
        trade_history: Trade history DataFrame
        
    Returns:
        Dictionary with today's results
    """
    date_column = 'timestamp' if 'timestamp' in equity_history.columns else equity_history.columns[0]
    
    # Get most recent date
    most_recent_date = equity_history[date_column].max()
    
    # Filter for today's data
    today_equity = equity_history[equity_history[date_column].dt.date == most_recent_date.date()]
    
    # Calculate daily metrics
    if not today_equity.empty:
        starting_equity = today_equity['equity'].iloc[0]
        ending_equity = today_equity['equity'].iloc[-1]
        daily_pnl = ending_equity - starting_equity
        daily_return = daily_pnl / starting_equity * 100
        
        # Get today's trades
        if trade_history is not None and not trade_history.empty:
            if 'exit_time' in trade_history.columns:
                today_trades = trade_history[trade_history['exit_time'].dt.date == most_recent_date.date()]
                trades_count = len(today_trades)
                winning_trades = len(today_trades[today_trades['pnl'] > 0])
                win_rate = winning_trades / trades_count * 100 if trades_count > 0 else 0
            else:
                trades_count = 0
                winning_trades = 0
                win_rate = 0
        else:
            trades_count = 0
            winning_trades = 0
            win_rate = 0
        
        return {
            'date': most_recent_date.date(),
            'starting_equity': starting_equity,
            'ending_equity': ending_equity,
            'daily_pnl': daily_pnl,
            'daily_return': daily_return,
            'trades': trades_count,
            'winning_trades': winning_trades,
            'win_rate': win_rate
        }
    else:
        # Default values if no data
        return {
            'date': datetime.now().date(),
            'starting_equity': 0,
            'ending_equity': 0,
            'daily_pnl': 0,
            'daily_return': 0,
            'trades': 0,
            'winning_trades': 0,
            'win_rate': 0
        }

def fetch_benchmark_data(target_date, benchmark_symbols):
    """
    Fetch benchmark data for comparison
    
    Args:
        target_date: Target date for data
        benchmark_symbols: List of benchmark symbols
        
    Returns:
        Dictionary with benchmark data
    """
    try:
        # Try to import yfinance
        import yfinance as yf
        
        benchmark_data = {}
        
        for symbol in benchmark_symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=target_date - timedelta(days=5), end=target_date + timedelta(days=1))
            
            if not hist.empty:
                # Get data for target date or most recent available
                target_data = hist[hist.index.date <= target_date.date()]
                if not target_data.empty:
                    most_recent = target_data.iloc[-1]
                    daily_return = most_recent['Close'] / most_recent['Open'] - 1 if most_recent['Open'] > 0 else 0
                    
                    benchmark_data[symbol] = {
                        'date': most_recent.name.date(),
                        'close': most_recent['Close'],
                        'daily_return': daily_return * 100  # percentage
                    }
        
        return benchmark_data
    
    except ImportError:
        logger.warning("yfinance not installed. Using mock benchmark data.")
        return {
            'SPY': {
                'date': target_date,
                'close': 450.0,
                'daily_return': 0.3
            },
            'VIX': {
                'date': target_date,
                'close': 15.0,
                'daily_return': -1.2
            }
        }
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
        return {}

def calculate_strategy_metrics(dashboard, thresholds):
    """
    Calculate performance metrics for all strategies
    
    Args:
        dashboard: PaperTradingDashboard instance
        thresholds: Performance thresholds
        
    Returns:
        Dictionary with strategy metrics
    """
    # This would typically extract metrics from the dashboard
    # For now, just return a sample with key metric fields
    
    # Try to get metrics from the dashboard
    if hasattr(dashboard, 'strategy_metrics') and dashboard.strategy_metrics:
        return dashboard.strategy_metrics
    
    # As a fallback, generate some sample metrics
    # In a real implementation, these would be calculated from strategy data
    return {
        'Momentum Strategy': {
            'sharpe_ratio': 1.7,
            'win_rate': 68.0,
            'max_drawdown': -4.5,
            'total_pnl': 3850.25,
            'trades_total': 28,
            'current_weight': 0.35,
            'sharpe_ratio_5d': 1.8,
            'win_rate_5d': 70.0,
            'max_drawdown_5d': -3.2
        },
        'Strangle Strategy': {
            'sharpe_ratio': 0.42,
            'win_rate': 48.0,
            'max_drawdown': -8.7,
            'total_pnl': 1250.18,
            'trades_total': 15,
            'current_weight': 0.25,
            'sharpe_ratio_5d': 0.38,
            'win_rate_5d': 33.0,
            'max_drawdown_5d': -7.5
        },
        'Calendar Spread Strategy': {
            'sharpe_ratio': 0.95,
            'win_rate': 62.0,
            'max_drawdown': -12.5,
            'total_pnl': 850.45,
            'trades_total': 12,
            'current_weight': 0.15,
            'sharpe_ratio_5d': 0.92,
            'win_rate_5d': 60.0,
            'max_drawdown_5d': -8.2
        }
    }

def trigger_optimization_jobs(alerts, suggestions):
    """
    Trigger optimization jobs for strategies requiring action
    
    Args:
        alerts: List of strategy alerts
        suggestions: List of strategy suggestions
    """
    # Get strategies requiring action
    strategies_to_optimize = [alert['strategy'] for alert in alerts if alert['action_required']]
    
    # Log optimization trigger
    if strategies_to_optimize:
        logger.info(f"Triggering optimization for strategies: {', '.join(strategies_to_optimize)}")
        
        # TODO: Implement actual optimization job trigger
        # This would typically call an optimization script or API
        for strategy in strategies_to_optimize:
            logger.info(f"Would optimize {strategy} here")

def main():
    """Main function to parse arguments and run recap"""
    parser = argparse.ArgumentParser(description='Run nightly performance recap')
    
    parser.add_argument(
        '--date',
        type=str,
        help='Generate recap for specific date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Run nightly recap
    success, message = run_nightly_recap(args.date, args.config)
    
    if not success:
        logger.error(message)
        sys.exit(1)
    
    logger.info(message)
    sys.exit(0)

if __name__ == "__main__":
    main()
