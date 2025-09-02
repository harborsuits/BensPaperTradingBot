#!/usr/bin/env python
"""
Recap Reporting Module for Nightly Recap System

This module contains functions for generating performance reports
and sending email notifications with trading performance insights.
"""

import os
import sys
import logging
import json
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

def create_performance_report(
    today_results: Dict[str, Any],
    benchmark_performance: Dict[str, Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    output_dir: str = 'reports/nightly'
) -> str:
    """
    Create and save performance report as JSON
    
    Args:
        today_results: Dictionary with today's performance
        benchmark_performance: Dictionary with benchmark performance
        alerts: List of strategy alerts
        suggestions: List of strategy suggestions
        output_dir: Output directory path
    
    Returns:
        Path to the generated report file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report data
        report_data = {
            'date': today_results['date'].strftime('%Y-%m-%d') if isinstance(today_results['date'], datetime) else today_results['date'],
            'daily_pnl': today_results.get('daily_pnl', 0),
            'daily_return': today_results.get('daily_return', 0),
            'equity_value': today_results.get('ending_equity', 0),
            'total_trades': today_results.get('trades', 0),
            'win_rate': today_results.get('win_rate', 0),
            'benchmarks': benchmark_performance,
            'alerts': [{
                'strategy': alert['strategy'],
                'severity': alert['severity'],
                'message': alert['alerts'][0]['message'] if alert['alerts'] else "No message",
                'action_required': alert['action_required']
            } for alert in alerts],
            'suggestions': [{
                'strategy': item['strategy'],
                'action': item['suggestion']['action'],
                'current_weight': item['suggestion']['current_weight'],
                'suggested_weight': item['suggestion']['suggested_weight'],
                'reason': item['suggestion']['reason']
            } for item in suggestions]
        }
        
        # Generate report filename
        date_str = datetime.now().strftime('%Y%m%d')
        if isinstance(today_results['date'], datetime):
            date_str = today_results['date'].strftime('%Y%m%d')
        elif isinstance(today_results['date'], str):
            try:
                date_obj = datetime.strptime(today_results['date'], '%Y-%m-%d')
                date_str = date_obj.strftime('%Y%m%d')
            except ValueError:
                pass
        
        report_path = os.path.join(output_dir, f"{date_str}_recap_data.json")
        
        # Write report to file
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error creating performance report: {e}")
        return ""

def generate_html_report(
    today_results: Dict[str, Any],
    benchmark_performance: Dict[str, Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    report_date: Optional[datetime] = None
) -> str:
    """
    Generate HTML report for email
    
    Args:
        today_results: Dictionary with today's performance
        benchmark_performance: Dictionary with benchmark performance
        alerts: List of strategy alerts
        suggestions: List of strategy suggestions
        report_date: Report date
    
    Returns:
        HTML content for email
    """
    if not report_date:
        report_date = datetime.now()
    
    date_str = report_date.strftime("%A, %B %d, %Y")
    
    # Start building HTML report
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
            }}
            .header {{
                background-color: #0066cc;
                color: white;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                padding: 20px;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .summary {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
            }}
            .metrics {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
            }}
            .metric {{
                padding: 10px;
                margin: 5px;
                background-color: #f9f9f9;
                border-radius: 5px;
                width: 30%;
                box-sizing: border-box;
            }}
            .metric h3 {{
                margin: 0;
                font-size: 14px;
                color: #666;
            }}
            .metric p {{
                margin: 5px 0 0;
                font-size: 18px;
                font-weight: bold;
            }}
            .positive {{
                color: #00aa00;
            }}
            .negative {{
                color: #cc0000;
            }}
            .neutral {{
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .warning {{
                background-color: #fff3cd;
            }}
            .critical {{
                background-color: #f8d7da;
            }}
            .action {{
                background-color: #d4edda;
            }}
            .footer {{
                font-size: 12px;
                color: #999;
                text-align: center;
                margin-top: 30px;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Trading Performance Recap</h1>
            <p>{date_str}</p>
        </div>
        <div class="container">
            <div class="section summary">
                <h2>Daily Performance Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Daily P&L</h3>
                        <p class="{get_value_class(today_results.get('daily_pnl', 0))}">
                            ${format_number(today_results.get('daily_pnl', 0))}
                        </p>
                    </div>
                    <div class="metric">
                        <h3>Daily Return</h3>
                        <p class="{get_value_class(today_results.get('daily_return', 0))}">
                            {format_percentage(today_results.get('daily_return', 0))}
                        </p>
                    </div>
                    <div class="metric">
                        <h3>Win Rate</h3>
                        <p class="{get_win_rate_class(today_results.get('win_rate', 0))}">
                            {format_percentage(today_results.get('win_rate', 0))}
                        </p>
                    </div>
                </div>
            </div>
    """
    
    # Add benchmark comparison if available
    if benchmark_performance:
        html += """
            <div class="section">
                <h2>Benchmark Comparison</h2>
                <table>
                    <tr>
                        <th>Benchmark</th>
                        <th>Daily Return</th>
                        <th>Your Performance</th>
                    </tr>
        """
        
        for symbol, data in benchmark_performance.items():
            performance_diff = today_results.get('daily_return', 0) - data.get('daily_return', 0)
            html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td class="{get_value_class(data.get('daily_return', 0))}">
                            {format_percentage(data.get('daily_return', 0))}
                        </td>
                        <td class="{get_value_class(performance_diff)}">
                            {'+' if performance_diff > 0 else ''}{format_percentage(performance_diff)} vs {symbol}
                        </td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
    
    # Add alerts section if there are alerts
    if alerts:
        html += """
            <div class="section">
                <h2>Strategy Alerts</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Severity</th>
                        <th>Issue</th>
                    </tr>
        """
        
        for alert in alerts:
            severity_class = 'warning' if alert['severity'] == 'warning' else 'critical'
            html += f"""
                    <tr class="{severity_class}">
                        <td><strong>{alert['strategy']}</strong></td>
                        <td>{alert['severity'].title()}</td>
                        <td>{alert['alerts'][0]['message'] if alert['alerts'] else 'No message'}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
    
    # Add suggestions section if there are suggestions
    if suggestions:
        html += """
            <div class="section">
                <h2>Recommended Actions</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Current Weight</th>
                        <th>Suggested Weight</th>
                        <th>Reason</th>
                    </tr>
        """
        
        for item in suggestions:
            suggestion = item['suggestion']
            html += f"""
                    <tr class="action">
                        <td><strong>{item['strategy']}</strong></td>
                        <td>{format_percentage(suggestion['current_weight'] * 100)}</td>
                        <td>{format_percentage(suggestion['suggested_weight'] * 100)}</td>
                        <td>{suggestion['reason']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
    
    # Close HTML
    html += """
            <div class="footer">
                <p>This is an automated performance recap email. Please do not reply to this email.</p>
                <p>Visit your trading dashboard for more detailed analysis and actions.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def send_email_report(
    email_config: Dict[str, Any],
    report_html: str,
    subject: str = "Daily Trading Performance Recap",
    attachments: List[str] = None
) -> bool:
    """
    Send email report
    
    Args:
        email_config: Email configuration
        report_html: HTML content for the email
        subject: Email subject
        attachments: List of file paths to attach
    
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Check if email is enabled
        if not email_config.get('enabled', False):
            logger.info("Email reporting is disabled in config")
            return False
        
        # Check for recipients
        recipients = email_config.get('recipients', [])
        if not recipients:
            logger.warning("No email recipients specified in config")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = email_config.get('username', 'trading-bot@example.com')
        msg['To'] = ', '.join(recipients)
        
        # Attach HTML body
        msg.attach(MIMEText(report_html, 'html'))
        
        # Attach files if any
        if attachments:
            for file_path in attachments:
                if not os.path.exists(file_path):
                    logger.warning(f"Attachment file not found: {file_path}")
                    continue
                
                with open(file_path, 'rb') as f:
                    file_name = os.path.basename(file_path)
                    
                    # Handle different file types
                    if file_path.endswith(('.png', '.jpg', '.jpeg')):
                        img = MIMEImage(f.read())
                        img.add_header('Content-Disposition', 'attachment', filename=file_name)
                        msg.attach(img)
                    else:
                        attachment = MIMEText(f.read())
                        attachment.add_header('Content-Disposition', 'attachment', filename=file_name)
                        msg.attach(attachment)
        
        # Connect to server and send
        server = smtplib.SMTP(email_config.get('server', 'smtp.gmail.com'), email_config.get('port', 587))
        server.starttls()
        
        # Login if username and password provided
        if 'username' in email_config and 'password' in email_config and email_config['username'] and email_config['password']:
            server.login(email_config['username'], email_config['password'])
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email report sent to {', '.join(recipients)}")
        return True
    
    except Exception as e:
        logger.error(f"Error sending email report: {e}")
        return False

def generate_performance_visualizations(
    today_results: Dict[str, Any],
    benchmark_performance: Dict[str, Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    output_dir: str = 'reports/nightly/charts'
) -> List[str]:
    """
    Generate performance visualizations for the report
    
    Args:
        today_results: Dictionary with today's performance
        benchmark_performance: Dictionary with benchmark performance
        alerts: List of strategy alerts
        output_dir: Output directory path
    
    Returns:
        List of paths to generated chart files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        chart_paths = []
        
        # TODO: Implement chart generation for performance metrics
        # This would typically include:
        # 1. Equity curve
        # 2. Strategy performance chart
        # 3. Benchmark comparison
        
        return chart_paths
    
    except Exception as e:
        logger.error(f"Error generating performance visualizations: {e}")
        return []

# Helper function to format numbers
def format_number(value: float) -> str:
    """Format a number with commas for thousands"""
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.2f}"

# Helper function to format percentages
def format_percentage(value: float) -> str:
    """Format a value as percentage"""
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.2f}%"

# Helper function to get class for values
def get_value_class(value: float) -> str:
    """Get CSS class based on value"""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"

# Helper function to get class for win rate
def get_win_rate_class(win_rate: float) -> str:
    """Get CSS class based on win rate"""
    if win_rate >= 60:
        return "positive"
    if win_rate >= 45:
        return "neutral"
    return "negative"
