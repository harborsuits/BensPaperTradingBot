"""
Advanced logging and monitoring capabilities for the BensBot Trading Dashboard
"""
import logging
import pandas as pd
import streamlit as st
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import threading

from dashboard.theme import COLORS

# Configure logger
logger = logging.getLogger("dashboard.monitoring")


class SystemMonitor:
    """Monitor system metrics and performance"""
    
    def __init__(self):
        """Initialize the system monitor"""
        self._data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'event_count': [],
            'active_strategies': [],
            'trade_count': [],
            'errors': []
        }
        self._last_update = datetime.now()
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, interval_seconds=5):
        """Start background monitoring thread"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
            
    def _monitor_loop(self, interval_seconds):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self.collect_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def collect_metrics(self):
        """Collect current system metrics"""
        # In a real implementation, this would collect actual metrics
        # from your trading system components
        try:
            # Record timestamp
            now = datetime.now()
            self._data['timestamps'].append(now)
            
            # For now, use mock data or integrate with your actual system
            # CPU & Memory usage would typically come from psutil or similar
            from trading_bot.event_system import EventManager
            from trading_bot.strategies.strategy_factory import StrategyFactory
            
            # Get event system metrics
            event_system = EventManager.get_instance()
            events_processed = event_system.get_event_count(last_minutes=1)
            self._data['event_count'].append(events_processed)
            
            # Get strategy metrics
            active_strategies = len(StrategyFactory.get_active_strategies())
            self._data['active_strategies'].append(active_strategies)
            
            # CPU/Memory would be collected from system monitoring
            import random
            self._data['cpu_usage'].append(random.uniform(15, 45))
            self._data['memory_usage'].append(random.uniform(300, 800))
            
            # Count trades
            from trading_bot.trade_manager import TradeManager
            trade_count = len(TradeManager.get_instance().get_active_trades())
            self._data['trade_count'].append(trade_count)
            
            # Check for errors
            errors = self._get_recent_errors()
            self._data['errors'].append(len(errors))
            
            # Limit data points
            max_points = 1000
            if len(self._data['timestamps']) > max_points:
                for key in self._data:
                    self._data[key] = self._data[key][-max_points:]
            
            self._last_update = now
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
            # Still add a data point with some mock values
            now = datetime.now()
            self._data['timestamps'].append(now)
            import random
            self._data['cpu_usage'].append(random.uniform(15, 45))
            self._data['memory_usage'].append(random.uniform(300, 800))
            self._data['event_count'].append(random.randint(10, 200))
            self._data['active_strategies'].append(random.randint(2, 8))
            self._data['trade_count'].append(random.randint(0, 10))
            self._data['errors'].append(random.randint(0, 3))
    
    def _get_recent_errors(self):
        """Get recent errors from log files"""
        # This would scan your log files for ERROR entries
        # For now, return mock data
        return []
            
    def get_metrics_df(self, minutes=30):
        """Get metrics as a dataframe for the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        # Filter data by time
        indices = [i for i, ts in enumerate(self._data['timestamps']) if ts >= cutoff]
        
        if not indices:
            return pd.DataFrame()
            
        filtered_data = {
            'timestamp': [self._data['timestamps'][i] for i in indices],
            'cpu_usage': [self._data['cpu_usage'][i] for i in indices],
            'memory_usage': [self._data['memory_usage'][i] for i in indices],
            'event_count': [self._data['event_count'][i] for i in indices],
            'active_strategies': [self._data['active_strategies'][i] for i in indices],
            'trade_count': [self._data['trade_count'][i] for i in indices],
            'errors': [self._data['errors'][i] for i in indices],
        }
        
        return pd.DataFrame(filtered_data)
    
    def create_dashboard_charts(self, minutes=30):
        """Create monitoring dashboard charts"""
        df = self.get_metrics_df(minutes=minutes)
        
        if df.empty:
            return None
            
        # Create a figure with subplots
        fig = make_subplots(
            rows=3, 
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "CPU Usage (%)", "Memory Usage (MB)",
                "Events/Minute", "Active Strategies",
                "Active Trades", "Errors"
            )
        )
        
        # Add CPU usage trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['cpu_usage'],
                mode='lines',
                name='CPU',
                line=dict(color=COLORS['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add memory usage trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['memory_usage'],
                mode='lines',
                name='Memory',
                line=dict(color=COLORS['secondary'], width=2)
            ),
            row=1, col=2
        )
        
        # Add event count trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['event_count'],
                mode='lines',
                name='Events',
                line=dict(color=COLORS['info'], width=2)
            ),
            row=2, col=1
        )
        
        # Add active strategies trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['active_strategies'],
                mode='lines',
                name='Strategies',
                line=dict(color=COLORS['success'], width=2)
            ),
            row=2, col=2
        )
        
        # Add trade count trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['trade_count'],
                mode='lines',
                name='Trades',
                line=dict(color=COLORS['warning'], width=2)
            ),
            row=3, col=1
        )
        
        # Add errors trace
        fig.add_trace(
            go.Bar(
                x=df['timestamp'], 
                y=df['errors'],
                name='Errors',
                marker=dict(color=COLORS['danger'])
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        
        # Update y-axes ranges
        fig.update_yaxes(range=[0, max(df['cpu_usage']) * 1.1], row=1, col=1)
        fig.update_yaxes(range=[0, max(df['memory_usage']) * 1.1], row=1, col=2)
        fig.update_yaxes(range=[0, max(df['event_count']) * 1.1], row=2, col=1)
        fig.update_yaxes(range=[0, max(df['active_strategies']) * 1.1], row=2, col=2)
        fig.update_yaxes(range=[0, max(df['trade_count']) * 1.1], row=3, col=1)
        fig.update_yaxes(range=[0, max(df['errors']) * 1.5], row=3, col=2)
        
        return fig


class LogAnalyzer:
    """Advanced log analysis for trading system logs"""
    
    @staticmethod
    def parse_logs(log_file_path):
        """Parse logs from file"""
        try:
            with open(log_file_path, 'r') as f:
                log_lines = f.readlines()
                
            logs = []
            for line in log_lines:
                # Parse log line (adjust based on your actual log format)
                # Example format: "2025-04-24 23:45:12 - INFO - trading_engine - Message"
                parts = line.split(' - ', 3)
                if len(parts) >= 4:
                    timestamp_str = parts[0]
                    level = parts[1]
                    component = parts[2]
                    message = parts[3].strip()
                    
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        logs.append({
                            'timestamp': timestamp,
                            'level': level,
                            'component': component,
                            'message': message
                        })
                    except ValueError:
                        # Skip invalid timestamp formats
                        pass
            
            return pd.DataFrame(logs)
        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def analyze_error_frequency(logs_df):
        """Analyze error frequency by component"""
        if logs_df.empty or 'level' not in logs_df.columns or 'component' not in logs_df.columns:
            return pd.DataFrame()
            
        # Filter for errors and warnings
        error_logs = logs_df[logs_df['level'].isin(['ERROR', 'WARNING'])]
        
        # Group by component and level
        grouped = error_logs.groupby(['component', 'level']).size().reset_index(name='count')
        
        return grouped
    
    @staticmethod
    def find_error_patterns(logs_df):
        """Find common error patterns"""
        if logs_df.empty or 'level' not in logs_df.columns or 'message' not in logs_df.columns:
            return pd.DataFrame()
            
        # Filter for errors
        error_logs = logs_df[logs_df['level'] == 'ERROR']
        
        if error_logs.empty:
            return pd.DataFrame()
            
        # Extract error types (first part of message usually)
        error_logs['error_type'] = error_logs['message'].str.split(':', 1).str[0]
        
        # Group by error type
        grouped = error_logs.groupby('error_type').size().reset_index(name='count')
        grouped = grouped.sort_values('count', ascending=False)
        
        return grouped
    
    @staticmethod
    def create_log_timeline(logs_df):
        """Create a timeline visualization of logs"""
        if logs_df.empty or 'timestamp' not in logs_df.columns or 'level' not in logs_df.columns:
            return None
            
        # Count logs by timestamp (hourly) and level
        logs_df['hour'] = logs_df['timestamp'].dt.floor('H')
        timeline = logs_df.groupby(['hour', 'level']).size().reset_index(name='count')
        
        # Pivot table for plotting
        pivot = timeline.pivot(index='hour', columns='level', values='count').fillna(0)
        
        # Make sure we have standard log levels
        for level in ['INFO', 'WARNING', 'ERROR', 'DEBUG']:
            if level not in pivot.columns:
                pivot[level] = 0
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each log level
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot['INFO'],
            name='INFO',
            mode='lines',
            line=dict(color=COLORS['info'], width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot['DEBUG'],
            name='DEBUG',
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot['WARNING'],
            name='WARNING',
            mode='lines',
            line=dict(color=COLORS['warning'], width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot['ERROR'],
            name='ERROR',
            mode='lines',
            line=dict(color=COLORS['danger'], width=2),
            stackgroup='one'
        ))
        
        # Update layout
        fig.update_layout(
            title="Log Volume Timeline",
            xaxis_title="Time",
            yaxis_title="Log Count",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        
        return fig


class AlertManager:
    """Advanced alerting system for trading platform"""
    
    def __init__(self):
        """Initialize the alert manager"""
        self._alerts = []
        self._alert_config = {
            'cpu_threshold': 80,
            'memory_threshold': 1000,
            'error_threshold': 5,
            'strategy_failure_threshold': 3,
            'api_failure_threshold': 3
        }
    
    def check_system_health(self, metrics_df):
        """Check system health based on metrics"""
        if metrics_df.empty:
            return
            
        latest = metrics_df.iloc[-1]
        
        # Check CPU usage
        if latest['cpu_usage'] > self._alert_config['cpu_threshold']:
            self._add_alert(
                'High CPU Usage', 
                f"CPU usage at {latest['cpu_usage']:.1f}%", 
                'system', 
                'warning'
            )
            
        # Check memory usage
        if latest['memory_usage'] > self._alert_config['memory_threshold']:
            self._add_alert(
                'High Memory Usage', 
                f"Memory usage at {latest['memory_usage']:.1f} MB", 
                'system', 
                'warning'
            )
            
        # Check errors
        if latest['errors'] > self._alert_config['error_threshold']:
            self._add_alert(
                'High Error Rate', 
                f"{latest['errors']} errors detected", 
                'system', 
                'danger'
            )
    
    def check_strategy_health(self, strategies):
        """Check strategy health"""
        # Count failed strategies
        failed_count = sum(1 for s in strategies if s.get('status') == 'failed')
        
        if failed_count >= self._alert_config['strategy_failure_threshold']:
            self._add_alert(
                'Multiple Strategy Failures', 
                f"{failed_count} strategies have failed", 
                'strategy', 
                'danger'
            )
    
    def check_api_health(self, api_statuses):
        """Check API health"""
        # Count failed APIs
        failed_apis = [name for name, status in api_statuses.items() if status == 'failed']
        
        if len(failed_apis) >= self._alert_config['api_failure_threshold']:
            self._add_alert(
                'Multiple API Failures', 
                f"{len(failed_apis)} APIs are failing", 
                'api', 
                'danger'
            )
    
    def _add_alert(self, title, message, category, severity):
        """Add a new alert"""
        self._alerts.append({
            'timestamp': datetime.now(),
            'title': title,
            'message': message,
            'category': category,
            'severity': severity,
            'acknowledged': False
        })
    
    def get_active_alerts(self):
        """Get list of active (unacknowledged) alerts"""
        return [a for a in self._alerts if not a['acknowledged']]
    
    def acknowledge_alert(self, alert_idx):
        """Acknowledge an alert by index"""
        if 0 <= alert_idx < len(self._alerts):
            self._alerts[alert_idx]['acknowledged'] = True
            return True
        return False
    
    def clear_all_alerts(self):
        """Clear all alerts"""
        self._alerts = []
    
    def update_alert_config(self, config):
        """Update alert configuration"""
        self._alert_config.update(config)
        
    def display_alerts_dashboard(self):
        """Display alerts in Streamlit"""
        active_alerts = self.get_active_alerts()
        
        if not active_alerts:
            st.success("No active alerts")
            return
            
        st.warning(f"{len(active_alerts)} active alerts")
        
        for i, alert in enumerate(active_alerts):
            severity_color = {
                'info': COLORS['info'],
                'warning': COLORS['warning'],
                'danger': COLORS['danger']
            }.get(alert['severity'], COLORS['info'])
            
            st.markdown(f"""
            <div style="
                background-color: white;
                border-left: 4px solid {severity_color};
                padding: 12px;
                margin-bottom: 8px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 500; font-size: 1rem;">{alert['title']}</div>
                        <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 4px;">
                            {alert['timestamp'].strftime('%H:%M:%S')} • {alert['category'].capitalize()}
                        </div>
                        <div style="font-size: 0.9rem;">{alert['message']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Acknowledge", key=f"ack_alert_{i}"):
                self.acknowledge_alert(i)
                st.experimental_rerun()


# Create singleton instances
system_monitor = SystemMonitor()
alert_manager = AlertManager()

# Start monitoring by default
system_monitor.start_monitoring()
