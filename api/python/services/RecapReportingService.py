import os
import json
import logging
import threading
from collections import deque
from datetime import datetime, timedelta
import pytz

from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.security.credentials_manager import CredentialsManager
from trading_bot.security.secure_logger import SecureLogger
from trading_bot.analysis.transaction_cost_analyzer import TransactionCostAnalyzer

# Import existing recap reporting functions for reuse
from trading_bot.monitoring.recap_reporting import (
    create_performance_report, 
    generate_html_report,
    send_email_report,
    generate_performance_visualizations
)


class RecapReportingService:
    """
    Service to collect trade events and generate end-of-day recap reports.
    Supports real-time tracking and batch (EOD) reporting via EventBus.
    
    Features:
    - Event-driven architecture integrated with EventBus
    - Secure credential handling and sensitive data masking
    - Market hours detection for auto-reporting
    - Transaction cost analysis
    - HTML email reports with visualizations
    """
    def __init__(self, report_dir='./reports', symbols=None, send_emails=True):
        # Setup directories
        self.report_dir = report_dir
        self.charts_dir = os.path.join(report_dir, 'charts')
        
        # State tracking
        self.symbols = symbols or ["SPY", "AAPL", "MSFT", "QQQ", "AMZN"]
        self.trades = deque(maxlen=1000)  # Limit to prevent memory issues
        self.pnl = 0.0
        self.starting_equity = 0.0
        self.current_equity = 0.0
        self.alerts = []
        self.suggestions = []
        self.performance_metrics = {}
        self.send_emails = send_emails
        
        # Security and logging
        self.credentials = CredentialsManager()
        self.logger = SecureLogger(name=__name__)
        
        # Analytics components
        self.tca = TransactionCostAnalyzer()
        
        # EventBus setup
        self.event_bus = get_global_event_bus()
        
        # Create directories
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        self._initialize_event_listeners()

    def _initialize_event_listeners(self):
        """Subscribe to relevant events on the event bus"""
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.on_trade_executed)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self.on_order_filled)
        self.event_bus.subscribe(EventType.RISK_ALERT, self.on_risk_alert)
        self.event_bus.subscribe(EventType.DRAWDOWN_ALERT, self.on_risk_alert)
        self.event_bus.subscribe(EventType.DRAWDOWN_THRESHOLD_EXCEEDED, self.on_risk_alert)
        self.event_bus.subscribe(EventType.END_OF_DAY, self.on_end_of_day)
        
        # Portfolio metrics events
        self.event_bus.subscribe(EventType.CAPITAL_ADJUSTED, self.on_capital_adjusted)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self.on_position_update)
        
        self.logger.info("RecapReportingService initialized and event listeners registered")

    def on_trade_executed(self, event: Event):
        """Process trade execution events"""
        trade = event.data
        self.trades.append(trade)
        
        # Update PnL tracking
        trade_pnl = trade.get('pnl', 0.0)
        self.pnl += trade_pnl
        
        # Log with secure logger to mask any sensitive information
        self.logger.info(f"Trade executed: {trade['symbol']} | PnL: {trade_pnl:.2f}")
        
        # Update transaction cost metrics if we have enough trades
        if len(self.trades) > 0:
            self.performance_metrics['transaction_costs'] = self.tca.analyze_transaction_costs(list(self.trades))

    def on_order_filled(self, event: Event):
        """Process order fill events"""
        order = event.data
        self.logger.info(f"Order filled: {order.get('symbol', 'Unknown')} | Qty: {order.get('quantity', 0)} | Price: {order.get('price', 0.0)}")

    def on_risk_alert(self, event: Event):
        """Process risk alert events"""
        alert = event.data
        self.alerts.append(alert)
        self.logger.warning(f"Risk alert: {alert.get('message', 'Unknown alert')} | Severity: {alert.get('severity', 'Unknown')}")

    def on_capital_adjusted(self, event: Event):
        """Track capital adjustments for accurate equity calculations"""
        capital_data = event.data
        self.current_equity = capital_data.get('new_capital', self.current_equity)
        
        # If this is the first capital adjustment of the day, set as starting equity
        if self.starting_equity == 0:
            self.starting_equity = self.current_equity
            self.logger.info(f"Starting equity set to {self.starting_equity:.2f}")
    
    def on_position_update(self, event: Event):
        """Track position updates for portfolio metrics"""
        position_data = event.data
        # Update relevant metrics that might be needed for reports
        if 'total_equity' in position_data:
            self.current_equity = position_data['total_equity']
    
    def on_end_of_day(self, event: Event):
        """Handle end of day processing and report generation"""
        self.logger.info("End of day event received. Generating report...")
        
        # Run in a separate thread to avoid blocking the main event loop
        threading.Thread(target=self._generate_and_distribute_reports).start()
    
    def _generate_and_distribute_reports(self):
        """Generate and distribute reports in a separate thread"""
        try:
            # Generate reports
            json_report = self._generate_daily_report()
            
            # Send email if configured
            if self.send_emails and json_report:
                self._send_email_report(json_report)
                
            # Reset state after successful report generation
            self._reset_state()
        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")

    def is_market_closed(self):
        """Check if the market is closed based on time and day"""
        # Get current time in Eastern timezone (US markets)
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Check if weekend
        if now.weekday() > 4:  # 5=Saturday, 6=Sunday
            return True
        
        # Regular trading hours end at 4:00 PM Eastern
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Consider market closed after 4:00 PM Eastern
        return now >= market_close
    
    def _format_today_results(self):
        """Format trading results in the structure expected by report generation functions"""
        win_count = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        win_rate = (win_count / len(self.trades)) * 100 if self.trades else 0
        
        return {
            'date': datetime.now(),
            'daily_pnl': self.pnl,
            'daily_return': (self.pnl / self.starting_equity * 100) if self.starting_equity else 0,
            'ending_equity': self.current_equity,
            'starting_equity': self.starting_equity,
            'trades': len(self.trades),
            'win_rate': win_rate,
            'transaction_costs': self.performance_metrics.get('transaction_costs', {})
        }
    
    def _get_benchmark_data(self):
        """Get benchmark comparison data"""
        # This could be enhanced to pull real benchmark data from an API
        return {
            'SPY': {
                'daily_return': 0.1,  # Example value
                'comparison': self.pnl - 0.1
            }
        }
    
    def _generate_daily_report(self):
        """Generate daily performance report and save as JSON"""
        try:
            # Format date for filename
            date_str = datetime.now().strftime('%Y%m%d')
            report_file = os.path.join(self.report_dir, f"daily_report_{date_str}.json")
            
            # Prepare report data using format similar to original functions
            today_results = self._format_today_results()
            benchmark_data = self._get_benchmark_data()
            
            # Build final report structure
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "total_trades": len(self.trades),
                "pnl": round(self.pnl, 2),
                "equity": round(self.current_equity, 2),
                "daily_return": round((self.pnl / self.starting_equity * 100) if self.starting_equity else 0, 2),
                "alerts": self.alerts,
                "suggestions": self.suggestions,
                "trade_details": list(self.trades),
                "performance_metrics": self.performance_metrics,
                "transaction_costs": self.performance_metrics.get('transaction_costs', {})
            }
            
            # Save to file
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            # Generate visualization charts using existing function
            chart_files = generate_performance_visualizations(
                today_results=today_results,
                benchmark_performance=benchmark_data,
                alerts=self.alerts,
                output_dir=self.charts_dir
            )
            
            # Use original function to create structured report
            create_performance_report(
                today_results=today_results,
                benchmark_performance=benchmark_data,
                alerts=self.alerts,
                suggestions=self.suggestions,
                output_dir=self.report_dir
            )
            
            self.logger.info(f"Daily report generated: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {str(e)}")
            return None

    def _send_email_report(self, report_file):
        """Send email report with performance details"""
        try:
            # Get email configuration from credentials manager
            email_config = {
                'server': self.credentials.get('EMAIL_SERVER', 'smtp.gmail.com'),
                'port': int(self.credentials.get('EMAIL_PORT', 587)),
                'username': self.credentials.get('EMAIL_USERNAME'),
                'password': self.credentials.get('EMAIL_PASSWORD'),
                'recipients': self.credentials.get('EMAIL_RECIPIENTS', '').split(',')
            }
            
            # Only proceed if we have credentials
            if not email_config['username'] or not email_config['password']:
                self.logger.warning("Email credentials not found, skipping email report")
                return
                
            # Generate HTML report using existing function
            html_content = generate_html_report(
                today_results=self._format_today_results(),
                benchmark_performance=self._get_benchmark_data(),
                alerts=self.alerts,
                suggestions=self.suggestions
            )
            
            # Get chart files
            chart_files = [f for f in os.listdir(self.charts_dir) 
                          if f.endswith(('.png', '.jpg')) and 
                          os.path.getmtime(os.path.join(self.charts_dir, f)) > 
                          (datetime.now() - timedelta(minutes=10)).timestamp()]
            chart_paths = [os.path.join(self.charts_dir, f) for f in chart_files]
            
            # Build email subject with key metrics
            subject = f"Trading Recap: {datetime.now().strftime('%Y-%m-%d')} | " \
                     f"PnL: {'📈+' if self.pnl >= 0 else '📉'}{abs(self.pnl):.2f} | " \
                     f"Trades: {len(self.trades)}"
                     
            # Send email with HTML report and attachments
            send_email_report(
                email_config=email_config,
                report_html=html_content,
                subject=subject,
                attachments=[report_file] + chart_paths
            )
            
            self.logger.info(f"Email report sent to {', '.join(email_config['recipients'])}")
            
        except Exception as e:
            self.logger.error(f"Error sending email report: {str(e)}")
    
    def _reset_state(self):
        """Reset service state for the next trading day"""
        self.trades.clear()
        self.pnl = 0.0
        self.starting_equity = 0.0
        self.alerts = []
        self.suggestions = []
        self.performance_metrics = {}
        self.logger.info("RecapReportingService state reset for next trading day")


    def trigger_end_of_day_manually(self):
        """Manually trigger end of day report generation (for testing or ad-hoc reports)"""
        self.logger.info("Manually triggering end of day report")
        self.on_end_of_day(Event(
            event_type=EventType.END_OF_DAY,
            data={'manual_trigger': True},
            source='manual'
        ))

if __name__ == '__main__':
    # For standalone testing
    from trading_bot.core.constants import EventType
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create service instance
    service = RecapReportingService(report_dir="./reports/test")
    
    # Simulate some trades and events
    event_bus = get_global_event_bus()
    
    # Set starting equity
    event_bus.create_and_publish(
        EventType.CAPITAL_ADJUSTED,
        data={'new_capital': 100000.0}
    )
    
    # Add some test trades
    for i in range(5):
        event_bus.create_and_publish(
            EventType.TRADE_EXECUTED,
            data={
                'symbol': 'AAPL',
                'quantity': 10,
                'price': 150.0 + i,
                'side': 'buy' if i % 2 == 0 else 'sell',
                'pnl': 100.0 if i % 2 == 0 else -50.0,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    # Add a risk alert
    event_bus.create_and_publish(
        EventType.RISK_ALERT,
        data={
            'severity': 'medium',
            'message': 'Drawdown approaching daily limit',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    # Trigger end of day manually for testing
    service.trigger_end_of_day_manually()
