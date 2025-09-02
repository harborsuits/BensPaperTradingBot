#!/usr/bin/env python3
"""
Report Generator - Creates HTML and PDF reports for strategy evaluation

This module generates detailed performance reports for trading strategies,
focusing on metrics relevant to proprietary trading firm evaluations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('report_generator')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Using Matplotlib for visualizations.")
    PLOTLY_AVAILABLE = False


class ReportGenerator:
    """
    Generates detailed HTML and PDF reports for strategy performance.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for report output
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize templates
        self._init_templates()
    
    def _init_templates(self):
        """Initialize HTML templates for reports."""
        # Simple HTML template with placeholders
        self.html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 10px; margin-bottom: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .metric {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; background-color: #f8f9fa; }}
                .metric h3 {{ margin-top: 0; }}
                .metric.positive {{ border-left: 4px solid green; }}
                .metric.negative {{ border-left: 4px solid red; }}
                .chart {{ margin-bottom: 30px; padding: 10px; background-color: white; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .footer {{ margin-top: 30px; padding: 10px; background-color: #f8f9fa; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on: {date}</p>
            </div>
            <div class="container">
                <div class="summary">
                    {summary_metrics}
                </div>
                <div class="chart">
                    <h2>Equity Curve</h2>
                    {equity_chart}
                </div>
                <div class="chart">
                    <h2>Drawdown</h2>
                    {drawdown_chart}
                </div>
                <div class="chart">
                    <h2>Trade Analysis</h2>
                    {trade_analysis}
                </div>
                <div class="chart">
                    <h2>Monthly Performance</h2>
                    {monthly_performance}
                </div>
                <div class="chart">
                    <h2>Trade Log</h2>
                    {trade_log}
                </div>
                <div class="chart">
                    <h2>Strategy Parameters</h2>
                    {strategy_params}
                </div>
                <div class="chart">
                    <h2>Prop Firm Compliance</h2>
                    {compliance_status}
                </div>
            </div>
            <div class="footer">
                EvoTrader Report Generator | {date}
            </div>
        </body>
        </html>
        """
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for HTML display."""
        html = ""
        
        # Core metrics with classifications
        core_metrics = {
            'total_return_pct': {'label': 'Total Return', 'format': '{:.2f}%', 'is_positive': lambda x: x > 0},
            'max_drawdown': {'label': 'Max Drawdown', 'format': '{:.2f}%', 'is_positive': lambda x: x < 5},
            'sharpe_ratio': {'label': 'Sharpe Ratio', 'format': '{:.2f}', 'is_positive': lambda x: x > 1},
            'win_rate': {'label': 'Win Rate', 'format': '{:.2f}%', 'is_positive': lambda x: x > 50},
            'profit_factor': {'label': 'Profit Factor', 'format': '{:.2f}', 'is_positive': lambda x: x > 1.5},
            'avg_trade': {'label': 'Avg Trade', 'format': '{:.2f}%', 'is_positive': lambda x: x > 0},
            'max_consecutive_wins': {'label': 'Max Consecutive Wins', 'format': '{:d}', 'is_positive': lambda x: x > 3},
            'max_consecutive_losses': {'label': 'Max Consecutive Losses', 'format': '{:d}', 'is_positive': lambda x: x < 5}
        }
        
        for key, config in core_metrics.items():
            if key in metrics:
                value = metrics[key]
                formatted_value = config['format'].format(value)
                css_class = "positive" if config['is_positive'](value) else "negative"
                
                html += f"""
                <div class="metric {css_class}">
                    <h3>{config['label']}</h3>
                    <div class="value">{formatted_value}</div>
                </div>
                """
        
        return html
    
    def generate_equity_chart(self, equity_curve: pd.Series, drawdowns: pd.Series) -> str:
        """
        Generate equity curve chart with drawdown overlay.
        
        Args:
            equity_curve: Series of equity values
            drawdowns: Series of drawdown values
            
        Returns:
            HTML for chart
        """
        if PLOTLY_AVAILABLE:
            # Create plotly figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               row_heights=[0.7, 0.3], 
                               vertical_spacing=0.05)
            
            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index, 
                    y=equity_curve.values, 
                    name="Equity Curve",
                    mode="lines",
                    line=dict(color="blue", width=2)
                ),
                row=1, col=1
            )
            
            # Add drawdown
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index, 
                    y=-drawdowns.values, 
                    name="Drawdown",
                    fill="tozeroy",
                    mode="lines",
                    line=dict(color="red", width=1)
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                title="Equity Curve with Drawdown",
                xaxis_title="Date",
                yaxis_title="Equity (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update yaxis for drawdown to be inverted
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            # Export to HTML
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
            
        else:
            # Matplotlib fallback
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=2)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity (%)')
            ax1.grid(True)
            
            # Plot drawdown
            ax2.fill_between(drawdowns.index, 0, -drawdowns.values, color='r', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            # Format date
            plt.gcf().autofmt_xdate()
            
            # Save to file
            chart_file = os.path.join(self.output_dir, f"equity_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.tight_layout()
            plt.savefig(chart_file, dpi=100)
            plt.close()
            
            return f'<img src="{os.path.basename(chart_file)}" width="100%" />'
    
    def _format_trade_table(self, trades: List[Dict[str, Any]]) -> str:
        """Format trades as HTML table."""
        if not trades:
            return "<p>No trades available</p>"
        
        html = """
        <table>
            <tr>
                <th>#</th>
                <th>Entry Time</th>
                <th>Exit Time</th>
                <th>Symbol</th>
                <th>Direction</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>P&L</th>
                <th>Return (%)</th>
                <th>Duration</th>
            </tr>
        """
        
        for i, trade in enumerate(trades[:100]):  # Limit to first 100 trades
            direction = "Long" if trade.get('direction', 1) > 0 else "Short"
            pnl = trade.get('pnl', 0)
            pnl_class = 'style="color:green"' if pnl > 0 else 'style="color:red"'
            
            # Format dates
            entry_time = trade.get('entry_time', '')
            exit_time = trade.get('exit_time', '')
            
            if isinstance(entry_time, datetime):
                entry_time = entry_time.strftime('%Y-%m-%d %H:%M')
            
            if isinstance(exit_time, datetime):
                exit_time = exit_time.strftime('%Y-%m-%d %H:%M')
            
            html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{entry_time}</td>
                <td>{exit_time}</td>
                <td>{trade.get('symbol', '')}</td>
                <td>{direction}</td>
                <td>{trade.get('entry_price', 0):.4f}</td>
                <td>{trade.get('exit_price', 0):.4f}</td>
                <td {pnl_class}>{pnl:.2f}</td>
                <td {pnl_class}>{trade.get('return_pct', 0):.2f}%</td>
                <td>{trade.get('duration', '')}</td>
            </tr>
            """
        
        html += "</table>"
        
        if len(trades) > 100:
            html += f"<p>Showing 100 of {len(trades)} trades.</p>"
        
        return html
    
    def _format_compliance(self, compliance_results: Dict[str, Any]) -> str:
        """Format compliance results as HTML."""
        if not compliance_results:
            return "<p>No compliance results available</p>"
        
        is_compliant = compliance_results.get('compliant', False)
        status_color = "green" if is_compliant else "red"
        status_text = "COMPLIANT" if is_compliant else "NON-COMPLIANT"
        
        html = f"""
        <div style="border-left: 4px solid {status_color}; padding: 10px;">
            <h3 style="color: {status_color};">Status: {status_text}</h3>
        """
        
        # Add specifics for each check
        checks = compliance_results.get('checks', {})
        html += "<table>"
        html += "<tr><th>Compliance Check</th><th>Status</th></tr>"
        
        for check, result in checks.items():
            check_color = "green" if result else "red"
            check_text = "PASS" if result else "FAIL"
            html += f'<tr><td>{check.replace("_", " ").title()}</td><td style="color:{check_color}">{check_text}</td></tr>'
        
        html += "</table>"
        
        # Add reasons for non-compliance
        if not is_compliant and 'non_compliant_reasons' in compliance_results:
            reasons = compliance_results['non_compliant_reasons']
            html += "<h4>Reasons for Non-Compliance:</h4><ul>"
            
            for reason in reasons:
                html += f"<li>{reason}</li>"
            
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def generate_report(self, 
                       strategy_name: str,
                       backtest_results: Dict[str, Any],
                       compliance_results: Optional[Dict[str, Any]] = None,
                       output_format: str = "html") -> str:
        """
        Generate a complete performance report.
        
        Args:
            strategy_name: Name of the strategy
            backtest_results: Dictionary with backtest results
            compliance_results: Optional compliance check results
            output_format: Output format (html or pdf)
            
        Returns:
            Path to generated report
        """
        try:
            # Extract metrics and data
            metrics = backtest_results.get('metrics', {})
            trades = backtest_results.get('trades', [])
            
            # Extract or calculate equity curve and drawdowns
            if 'equity_curve' in backtest_results:
                equity_curve = pd.Series(backtest_results['equity_curve'])
            else:
                # If not provided, calculate simple equity curve from trades
                equity_curve = self._calculate_equity_curve(trades)
            
            if 'drawdowns' in backtest_results:
                drawdowns = pd.Series(backtest_results['drawdowns'])
            else:
                # Calculate drawdowns from equity curve
                drawdowns = self._calculate_drawdowns(equity_curve)
            
            # Format components
            summary_metrics = self._format_metrics(metrics)
            equity_chart = self.generate_equity_chart(equity_curve, drawdowns)
            trade_log = self._format_trade_table(trades)
            
            # Format strategy parameters
            strategy_params = "<ul>"
            for param, value in backtest_results.get('parameters', {}).items():
                strategy_params += f"<li><strong>{param}:</strong> {value}</li>"
            strategy_params += "</ul>"
            
            # Format compliance results
            compliance_status = self._format_compliance(compliance_results) if compliance_results else ""
            
            # Fill template
            report_html = self.html_template.format(
                title=f"Strategy Report: {strategy_name}",
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                summary_metrics=summary_metrics,
                equity_chart=equity_chart,
                drawdown_chart="", # Placeholder
                trade_analysis="", # Placeholder
                monthly_performance="", # Placeholder
                trade_log=trade_log,
                strategy_params=strategy_params,
                compliance_status=compliance_status
            )
            
            # Generate file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = ''.join(c if c.isalnum() else '_' for c in strategy_name)
            report_file = os.path.join(self.output_dir, f"{safe_name}_{timestamp}.html")
            
            with open(report_file, 'w') as f:
                f.write(report_html)
            
            # Generate PDF if requested
            if output_format == "pdf":
                try:
                    from weasyprint import HTML
                    pdf_file = report_file.replace('.html', '.pdf')
                    HTML(report_file).write_pdf(pdf_file)
                    return pdf_file
                except ImportError:
                    logger.warning("WeasyPrint not available. Using HTML output instead.")
                    return report_file
            
            return report_file
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _calculate_equity_curve(self, trades: List[Dict[str, Any]]) -> pd.Series:
        """Calculate equity curve from trades."""
        if not trades:
            return pd.Series([0])
        
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.get('exit_time', datetime.now()))
        
        # Extract daily P&L
        daily_pnl = {}
        
        for trade in sorted_trades:
            exit_time = trade.get('exit_time')
            if exit_time:
                date_key = exit_time.date() if isinstance(exit_time, datetime) else exit_time
                pnl = trade.get('pnl', 0)
                
                if date_key in daily_pnl:
                    daily_pnl[date_key] += pnl
                else:
                    daily_pnl[date_key] = pnl
        
        # Convert to Series and calculate cumulative returns
        if daily_pnl:
            daily_series = pd.Series(daily_pnl)
            equity_curve = 100 * (1 + daily_series.cumsum() / 10000)  # Assuming $10k starting capital
            return equity_curve
        
        return pd.Series([100])
    
    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdowns from equity curve."""
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown in percentage terms
        drawdown = 100 * (equity_curve - running_max) / running_max
        
        return -drawdown  # Convert to positive values


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reports for trading strategies")
    
    parser.add_argument(
        "--backtest", 
        type=str, 
        required=True,
        help="Path to backtest results JSON file"
    )
    
    parser.add_argument(
        "--compliance", 
        type=str, 
        default=None,
        help="Path to compliance results JSON file"
    )
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True,
        help="Strategy name"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./reports",
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["html", "pdf"],
        default="html",
        help="Output format (html or pdf)"
    )
    
    args = parser.parse_args()
    
    # Load backtest results
    with open(args.backtest, 'r') as f:
        backtest_results = json.load(f)
    
    # Load compliance results if provided
    compliance_results = None
    if args.compliance:
        with open(args.compliance, 'r') as f:
            compliance_results = json.load(f)
    
    # Generate report
    report_generator = ReportGenerator(args.output)
    report_file = report_generator.generate_report(
        args.strategy,
        backtest_results,
        compliance_results,
        args.format
    )
    
    print(f"Report generated: {report_file}")
