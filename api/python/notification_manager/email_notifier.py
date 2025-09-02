import os
import smtplib
import logging
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

class EmailNotifier:
    """
    Handles sending of email notifications for the trading system.
    
    This class provides methods for sending various types of email notifications
    including trade alerts, error notifications, daily summaries, and performance reports.
    """
    
    def __init__(self, 
                 smtp_server: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 sender_email: str,
                 recipient_emails: Union[str, List[str]],
                 log_dir: str = None,
                 template_dir: str = None,
                 debug: bool = False):
        """
        Initialize the EmailNotifier with SMTP server details and credentials.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            sender_email: Email address to send from
            recipient_emails: Email address(es) to send to
            log_dir: Directory to store email logs
            template_dir: Directory containing email templates
            debug: Enable debug mode for verbose logging
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = sender_email
        
        # Convert single recipient to list
        if isinstance(recipient_emails, str):
            self.recipient_emails = [recipient_emails]
        else:
            self.recipient_emails = recipient_emails
            
        # Setup logging
        self.logger = logging.getLogger('EmailNotifier')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        # Create log directory if it doesn't exist
        if log_dir:
            self.log_dir = Path(log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = None
            
        # Set up template directory
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to a templates subdirectory
            self.template_dir = Path(__file__).parent / 'templates'
            
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Keep track of sent emails for rate limiting and logging
        self.sent_emails = []
        self.max_log_entries = 100

    def _create_base_message(self, subject: str, recipient_emails: List[str] = None) -> MIMEMultipart:
        """Create a base email message with headers and recipients."""
        if recipient_emails is None:
            recipient_emails = self.recipient_emails
            
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(recipient_emails)
        msg['Subject'] = subject
        
        return msg
        
    def _send_email(self, msg: MIMEMultipart, recipient_emails: List[str] = None) -> bool:
        """
        Send an email message via SMTP.
        
        Args:
            msg: The email message to send
            recipient_emails: Override default recipients
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if recipient_emails is None:
            recipient_emails = self.recipient_emails

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.sender_email, recipient_emails, msg.as_string())
                
            # Log the sent email
            email_log = {
                'timestamp': datetime.datetime.now().isoformat(),
                'subject': msg['Subject'],
                'recipients': recipient_emails,
                'status': 'sent'
            }
            self.sent_emails.append(email_log)
            
            # Trim log if it gets too large
            if len(self.sent_emails) > self.max_log_entries:
                self.sent_emails = self.sent_emails[-self.max_log_entries:]
                
            # Log to file if configured
            if self.log_dir:
                log_file = self.log_dir / 'email_log.json'
                try:
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            logs = json.load(f)
                    else:
                        logs = []
                        
                    logs.append(email_log)
                    
                    # Keep log file from growing too large
                    if len(logs) > 1000:
                        logs = logs[-1000:]
                        
                    with open(log_file, 'w') as f:
                        json.dump(logs, f, indent=2)
                except Exception as e:
                    self.logger.error(f"Failed to write to email log file: {e}")
            
            self.logger.info(f"Email sent: {msg['Subject']} to {recipient_emails}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            
            # Log the failed email
            email_log = {
                'timestamp': datetime.datetime.now().isoformat(),
                'subject': msg['Subject'],
                'recipients': recipient_emails,
                'status': 'failed',
                'error': str(e)
            }
            self.sent_emails.append(email_log)
            
            return False
    
    def _load_template(self, template_name: str) -> str:
        """Load an email template from the template directory."""
        template_path = self.template_dir / f"{template_name}.html"
        
        if not template_path.exists():
            self.logger.warning(f"Template {template_name} not found, using default")
            return "<html><body>{{content}}</body></html>"
            
        with open(template_path, 'r') as f:
            return f.read()
    
    def send_trade_alert(self, 
                        trade_data: Dict[str, Any], 
                        alert_type: str = 'entry',
                        attachments: List[str] = None) -> bool:
        """
        Send an alert for a trade entry or exit.
        
        Args:
            trade_data: Dictionary containing trade details
            alert_type: Type of alert ('entry', 'exit', 'cancelled')
            attachments: List of file paths to attach
            
        Returns:
            bool: True if email was sent successfully
        """
        if alert_type == 'entry':
            subject = f"Trade Alert: New {trade_data.get('direction', 'unknown')} position in {trade_data.get('symbol', 'unknown')}"
            template = self._load_template('trade_entry')
        elif alert_type == 'exit':
            pnl = trade_data.get('pnl', 0)
            pnl_str = f"{'+' if pnl >= 0 else ''}{pnl}"
            subject = f"Trade Alert: Closed position in {trade_data.get('symbol', 'unknown')} ({pnl_str})"
            template = self._load_template('trade_exit')
        else:
            subject = f"Trade Alert: {alert_type.capitalize()} - {trade_data.get('symbol', 'unknown')}"
            template = self._load_template('trade_alert')
            
        # Format the trade data into HTML
        trade_details = []
        for key, value in trade_data.items():
            if key in ['pnl', 'price', 'stop_loss', 'take_profit', 'entry_price', 'exit_price']:
                trade_details.append(f"<tr><td><strong>{key.replace('_', ' ').title()}</strong></td><td>{value:.2f}</td></tr>")
            else:
                trade_details.append(f"<tr><td><strong>{key.replace('_', ' ').title()}</strong></td><td>{value}</td></tr>")
                
        trade_table = f"<table border='1' cellpadding='5' cellspacing='0'><tbody>{''.join(trade_details)}</tbody></table>"
        
        # Replace placeholders in template
        html_content = template.replace('{{trade_table}}', trade_table)
        html_content = html_content.replace('{{alert_type}}', alert_type.capitalize())
        html_content = html_content.replace('{{timestamp}}', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Create email
        msg = self._create_base_message(subject)
        msg.attach(MIMEText(html_content, 'html'))
        
        # Add attachments if provided
        if attachments:
            for attachment_path in attachments:
                try:
                    with open(attachment_path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment_filename = os.path.basename(attachment_path)
                        attachment.add_header('Content-Disposition', 'attachment', filename=attachment_filename)
                        msg.attach(attachment)
                except Exception as e:
                    self.logger.error(f"Failed to attach file {attachment_path}: {e}")
        
        return self._send_email(msg)
    
    def send_error_notification(self, 
                               error_message: str, 
                               error_details: Dict[str, Any] = None,
                               priority: str = 'medium') -> bool:
        """
        Send a notification about an error in the trading system.
        
        Args:
            error_message: Main error message
            error_details: Dictionary with detailed error information
            priority: Priority level ('low', 'medium', 'high', 'critical')
            
        Returns:
            bool: True if email was sent successfully
        """
        priority_prefixes = {
            'low': '[INFO]',
            'medium': '[WARNING]',
            'high': '[ERROR]',
            'critical': '[CRITICAL]'
        }
        
        subject_prefix = priority_prefixes.get(priority.lower(), '[WARNING]')
        subject = f"{subject_prefix} Trading System Error: {error_message[:50]}{'...' if len(error_message) > 50 else ''}"
        
        template = self._load_template('error_notification')
        
        # Format error details
        error_details_html = ""
        if error_details:
            error_rows = []
            for key, value in error_details.items():
                if isinstance(value, dict) or isinstance(value, list):
                    value = json.dumps(value, indent=2)
                error_rows.append(f"<tr><td><strong>{key}</strong></td><td><pre>{value}</pre></td></tr>")
            
            error_details_html = f"<table border='1' cellpadding='5' cellspacing='0'><tbody>{''.join(error_rows)}</tbody></table>"
        
        # Replace placeholders in template
        html_content = template.replace('{{error_message}}', error_message)
        html_content = html_content.replace('{{error_details}}', error_details_html)
        html_content = html_content.replace('{{timestamp}}', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        html_content = html_content.replace('{{priority}}', priority.upper())
        
        # Create email
        msg = self._create_base_message(subject)
        msg.attach(MIMEText(html_content, 'html'))
        
        # For critical errors, may want to send to additional recipients
        recipient_emails = self.recipient_emails
        if priority.lower() == 'critical' and hasattr(self, 'critical_recipients'):
            recipient_emails = list(set(recipient_emails + self.critical_recipients))
            
        return self._send_email(msg, recipient_emails)
    
    def send_daily_summary(self, 
                          summary_data: Dict[str, Any],
                          chart_paths: List[str] = None) -> bool:
        """
        Send a daily summary of trading activity.
        
        Args:
            summary_data: Dictionary containing summary statistics
            chart_paths: List of file paths to charts to attach
            
        Returns:
            bool: True if email was sent successfully
        """
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        subject = f"Trading Summary: {date_str}"
        
        template = self._load_template('daily_summary')
        
        # Format the summary data
        summary_rows = []
        for section, data in summary_data.items():
            if isinstance(data, dict):
                summary_rows.append(f"<tr><td colspan='2'><h3>{section}</h3></td></tr>")
                for key, value in data.items():
                    if isinstance(value, float):
                        summary_rows.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
                    else:
                        summary_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            else:
                if isinstance(data, float):
                    summary_rows.append(f"<tr><td><strong>{section}</strong></td><td>{data:.2f}</td></tr>")
                else:
                    summary_rows.append(f"<tr><td><strong>{section}</strong></td><td>{data}</td></tr>")
                
        summary_table = f"<table border='1' cellpadding='5' cellspacing='0'><tbody>{''.join(summary_rows)}</tbody></table>"
        
        # Replace placeholders in template
        html_content = template.replace('{{summary_table}}', summary_table)
        html_content = html_content.replace('{{date}}', date_str)
        
        # Create email
        msg = self._create_base_message(subject)
        msg.attach(MIMEText(html_content, 'html'))
        
        # Add chart attachments
        if chart_paths:
            for chart_path in chart_paths:
                try:
                    with open(chart_path, 'rb') as f:
                        chart = MIMEApplication(f.read())
                        chart_filename = os.path.basename(chart_path)
                        chart.add_header('Content-Disposition', 'attachment', filename=chart_filename)
                        msg.attach(chart)
                        
                        # Also embed the image in the email if it's an image file
                        if chart_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            html_content = html_content.replace('</body>',
                                f"<div><img src='cid:{chart_filename}' width='100%' /></div></body>")
                            chart.add_header('Content-ID', f"<{chart_filename}>")
                except Exception as e:
                    self.logger.error(f"Failed to attach chart {chart_path}: {e}")
        
        return self._send_email(msg)
    
    def send_performance_report(self,
                               performance_data: Dict[str, Any],
                               report_period: str = 'monthly',
                               report_date: datetime.date = None,
                               attachments: List[str] = None) -> bool:
        """
        Send a performance report for a specific time period.
        
        Args:
            performance_data: Dictionary containing performance metrics
            report_period: 'daily', 'weekly', 'monthly', or 'quarterly'
            report_date: Date for the report (defaults to today)
            attachments: List of file paths to attach
            
        Returns:
            bool: True if email was sent successfully
        """
        if report_date is None:
            report_date = datetime.date.today()
            
        if report_period == 'daily':
            date_format = '%Y-%m-%d'
            subject = f"Daily Performance Report: {report_date.strftime(date_format)}"
        elif report_period == 'weekly':
            # Calculate start of week
            start_of_week = report_date - datetime.timedelta(days=report_date.weekday())
            end_of_week = start_of_week + datetime.timedelta(days=6)
            date_format = '%Y-%m-%d'
            subject = f"Weekly Performance Report: {start_of_week.strftime(date_format)} to {end_of_week.strftime(date_format)}"
        elif report_period == 'monthly':
            date_format = '%B %Y'
            subject = f"Monthly Performance Report: {report_date.strftime(date_format)}"
        elif report_period == 'quarterly':
            quarter = (report_date.month - 1) // 3 + 1
            subject = f"Quarterly Performance Report: Q{quarter} {report_date.year}"
        else:
            date_format = '%Y-%m-%d'
            subject = f"Performance Report: {report_date.strftime(date_format)}"
            
        template = self._load_template('performance_report')
        
        # Format the performance metrics into tables by category
        performance_html = ""
        for category, metrics in performance_data.items():
            if isinstance(metrics, dict):
                category_rows = []
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        category_rows.append(f"<tr><td>{metric_name}</td><td>{metric_value:.4f}</td></tr>")
                    else:
                        category_rows.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")
                
                performance_html += f"""
                <h3>{category}</h3>
                <table border='1' cellpadding='5' cellspacing='0'>
                    <tbody>{''.join(category_rows)}</tbody>
                </table>
                <br>
                """
        
        # Replace placeholders in template
        html_content = template.replace('{{performance_metrics}}', performance_html)
        html_content = html_content.replace('{{report_period}}', report_period.capitalize())
        html_content = html_content.replace('{{report_date}}', report_date.strftime(date_format))
        
        # Create email
        msg = self._create_base_message(subject)
        msg.attach(MIMEText(html_content, 'html'))
        
        # Add attachments if provided
        if attachments:
            for attachment_path in attachments:
                try:
                    with open(attachment_path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment_filename = os.path.basename(attachment_path)
                        attachment.add_header('Content-Disposition', 'attachment', filename=attachment_filename)
                        msg.attach(attachment)
                except Exception as e:
                    self.logger.error(f"Failed to attach file {attachment_path}: {e}")
        
        return self._send_email(msg)
    
    def send_custom_notification(self,
                                subject: str,
                                content: str,
                                is_html: bool = True,
                                recipient_emails: List[str] = None,
                                attachments: List[str] = None) -> bool:
        """
        Send a custom notification with arbitrary content.
        
        Args:
            subject: Email subject
            content: Email content
            is_html: Whether the content is HTML
            recipient_emails: Override default recipients
            attachments: List of file paths to attach
            
        Returns:
            bool: True if email was sent successfully
        """
        msg = self._create_base_message(subject, recipient_emails)
        
        # Add content
        if is_html:
            msg.attach(MIMEText(content, 'html'))
        else:
            msg.attach(MIMEText(content, 'plain'))
            
        # Add attachments if provided
        if attachments:
            for attachment_path in attachments:
                try:
                    with open(attachment_path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment_filename = os.path.basename(attachment_path)
                        attachment.add_header('Content-Disposition', 'attachment', filename=attachment_filename)
                        msg.attach(attachment)
                except Exception as e:
                    self.logger.error(f"Failed to attach file {attachment_path}: {e}")
        
        return self._send_email(msg, recipient_emails)
    
    def get_email_history(self, 
                        limit: int = None, 
                        filter_subject: str = None, 
                        start_date: datetime.datetime = None,
                        end_date: datetime.datetime = None) -> List[Dict[str, Any]]:
        """
        Retrieve email sending history with optional filtering.
        
        Args:
            limit: Maximum number of records to return
            filter_subject: Filter emails by subject containing this string
            start_date: Only include emails after this date
            end_date: Only include emails before this date
            
        Returns:
            List of email history records
        """
        filtered_history = self.sent_emails
        
        # Apply filters
        if filter_subject:
            filtered_history = [e for e in filtered_history if filter_subject.lower() in e.get('subject', '').lower()]
            
        if start_date:
            filtered_history = [e for e in filtered_history if datetime.datetime.fromisoformat(e.get('timestamp', '')) >= start_date]
            
        if end_date:
            filtered_history = [e for e in filtered_history if datetime.datetime.fromisoformat(e.get('timestamp', '')) <= end_date]
            
        # Apply limit
        if limit and limit > 0:
            filtered_history = filtered_history[-limit:]
            
        return filtered_history
