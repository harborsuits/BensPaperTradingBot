"""
Risk Monitoring Panel Component

This component displays real-time risk metrics and emergency controls status,
including kill switch state, position limits, and data quality alerts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.risk.emergency_controls import EmergencyControls
from trading_bot.data.data_validator import DataValidator


class RiskMonitoringPanel:
    """Panel for monitoring risk controls and data quality."""
    
    def __init__(self, emergency_controls=None, data_validator=None):
        """
        Initialize the risk monitoring panel.
        
        Args:
            emergency_controls: Optional EmergencyControls instance
            data_validator: Optional DataValidator instance
        """
        self.event_bus = get_global_event_bus()
        
        # Use provided instances or create new ones
        self.emergency_controls = emergency_controls or EmergencyControls()
        self.data_validator = data_validator or DataValidator()
        
        # State for tracking events
        self.data_quality_alerts = []
        self.kill_switch_events = []
        self.position_updates = {}
        self.recent_orders = []
        
        # Subscribe to relevant events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus."""
        self.event_bus.subscribe(EventType.DATA_QUALITY_ALERT, self._on_data_quality_alert)
        self.event_bus.subscribe(EventType.KILL_SWITCH_ACTIVATED, self._on_kill_switch)
        self.event_bus.subscribe(EventType.KILL_SWITCH_DEACTIVATED, self._on_kill_switch)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.subscribe(EventType.ORDER_CREATED, self._on_order_event)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_event)
        self.event_bus.subscribe(EventType.ORDER_REJECTED, self._on_order_event)
    
    def _on_data_quality_alert(self, event):
        """Handle data quality alert events."""
        self.data_quality_alerts.append({
            'timestamp': datetime.now(),
            'data': event.data
        })
        
        # Keep only the most recent 100 alerts
        if len(self.data_quality_alerts) > 100:
            self.data_quality_alerts = self.data_quality_alerts[-100:]
    
    def _on_kill_switch(self, event):
        """Handle kill switch events."""
        self.kill_switch_events.append({
            'timestamp': datetime.now(),
            'event_type': event.event_type,
            'data': event.data
        })
        
        # Keep only the most recent 20 events
        if len(self.kill_switch_events) > 20:
            self.kill_switch_events = self.kill_switch_events[-20:]
    
    def _on_position_update(self, event):
        """Handle position update events."""
        position_data = event.data
        symbol = position_data.get('symbol', 'unknown')
        self.position_updates[symbol] = {
            'timestamp': datetime.now(),
            'data': position_data
        }
    
    def _on_order_event(self, event):
        """Handle order related events."""
        self.recent_orders.append({
            'timestamp': datetime.now(),
            'event_type': event.event_type,
            'data': event.data
        })
        
        # Keep only the most recent 50 orders
        if len(self.recent_orders) > 50:
            self.recent_orders = self.recent_orders[-50:]
    
    def render(self):
        """Render the risk monitoring panel."""
        st.header("Risk Control & Data Quality Dashboard")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_kill_switch_status()
        
        with col2:
            self._render_data_quality_summary()
        
        with col3:
            self._render_account_metrics()
        
        # Create tabs for detailed views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Emergency Controls", 
            "Data Quality Alerts", 
            "Position Limits", 
            "Recent Orders"
        ])
        
        with tab1:
            self._render_emergency_controls_tab()
        
        with tab2:
            self._render_data_quality_tab()
        
        with tab3:
            self._render_position_limits_tab()
        
        with tab4:
            self._render_orders_tab()
    
    def _render_kill_switch_status(self):
        """Render kill switch status card."""
        st.subheader("Kill Switch Status")
        
        # Get current kill switch status
        kill_switch_active = self.emergency_controls.kill_switch_activated
        
        # Create a styled card based on kill switch status
        if kill_switch_active:
            st.error("⚠️ KILL SWITCH ACTIVE ⚠️")
            
            # Show reason if available
            if self.kill_switch_events:
                latest_event = self.kill_switch_events[-1]
                reason = latest_event['data'].get('reason', 'Unknown reason')
                st.write(f"Reason: {reason}")
                st.write(f"Activated: {latest_event['timestamp'].strftime('%H:%M:%S')}")
            
            # Add deactivation button
            if st.button("Deactivate Kill Switch (Override)"):
                st.warning("This will deactivate the kill switch. Use with caution!")
                # In a real implementation, this would call the API to deactivate the kill switch
        else:
            st.success("✅ Kill Switch Inactive")
            st.write("Trading system is operating normally")
            
            # Add manual activation button
            if st.button("Activate Kill Switch (Emergency)"):
                st.warning("This will halt all trading immediately!")
                # In a real implementation, this would call the API to activate the kill switch
    
    def _render_data_quality_summary(self):
        """Render data quality summary card."""
        st.subheader("Data Quality Status")
        
        # Count alerts by type in the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.data_quality_alerts if a['timestamp'] > one_hour_ago]
        
        # Count alerts by severity
        error_count = sum(1 for a in recent_alerts if a['data'].get('severity') == 'error')
        warning_count = sum(1 for a in recent_alerts if a['data'].get('severity') == 'warning')
        info_count = sum(1 for a in recent_alerts if a['data'].get('severity') == 'info')
        
        total_count = len(recent_alerts)
        
        # Create a styled card based on alert counts
        if error_count > 0:
            st.error(f"⚠️ {error_count} Data Quality Errors")
        elif warning_count > 0:
            st.warning(f"⚠️ {warning_count} Data Quality Warnings")
        else:
            st.success("✅ Data Quality Good")
        
        st.write(f"Last hour: {error_count} errors, {warning_count} warnings, {info_count} info alerts")
        
        # Create a mini time series chart of alerts
        if total_count > 0:
            # Group by 5-minute buckets
            times = [a['timestamp'] for a in recent_alerts]
            min_time = min(times) if times else one_hour_ago
            
            # Create 5-minute buckets
            buckets = {}
            current_time = min_time
            while current_time <= datetime.now():
                bucket_key = current_time.strftime('%H:%M')
                buckets[bucket_key] = 0
                current_time += timedelta(minutes=5)
            
            # Count alerts in each bucket
            for alert in recent_alerts:
                bucket_key = alert['timestamp'].strftime('%H:%M')
                if bucket_key in buckets:
                    buckets[bucket_key] += 1
            
            # Create chart data
            chart_data = pd.DataFrame({
                'Time': list(buckets.keys()),
                'Alerts': list(buckets.values())
            })
            
            # Show mini chart if we have any data
            if not chart_data.empty and chart_data['Alerts'].sum() > 0:
                st.line_chart(chart_data.set_index('Time'))
    
    def _render_account_metrics(self):
        """Render account metrics card."""
        st.subheader("Account Risk Metrics")
        
        # Get status from emergency controls
        status = self.emergency_controls.get_status_report()
        
        # Calculate daily P&L metrics
        daily_pnl = status.get('daily_pnl', 0.0)
        daily_pnl_pct = status.get('daily_pnl_pct', 0.0) * 100  # Convert to percentage
        
        # Display key metrics
        if daily_pnl >= 0:
            st.write(f"Daily P&L: 📈 ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)")
        else:
            st.write(f"Daily P&L: 📉 ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)")
        
        # Drawdown metrics
        drawdown_pct = status.get('current_drawdown_pct', 0.0) * 100  # Convert to percentage
        max_drawdown_pct = status.get('limits', {}).get('max_drawdown_pct', 0.10) * 100
        
        # Create a progress bar for drawdown
        drawdown_ratio = drawdown_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        st.write(f"Drawdown: {drawdown_pct:.2f}% (Max: {max_drawdown_pct:.2f}%)")
        st.progress(min(drawdown_ratio, 1.0))
        
        # Position count
        position_count = status.get('position_count', 0)
        st.write(f"Active positions: {position_count}")
    
    def _render_emergency_controls_tab(self):
        """Render emergency controls detailed tab."""
        st.subheader("Emergency Controls")
        
        # Get status from emergency controls
        status = self.emergency_controls.get_status_report()
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Risk Limits")
            
            # Format limits as a dataframe
            limits = status.get('limits', {})
            limits_df = pd.DataFrame({
                'Parameter': [
                    'Max Daily Loss',
                    'Max Position Size',
                    'Max Strategy Loss',
                    'Max Drawdown'
                ],
                'Value': [
                    f"{limits.get('max_daily_loss_pct', 0.0) * 100:.1f}%",
                    f"{limits.get('max_position_pct', 0.0) * 100:.1f}%",
                    f"{limits.get('max_strategy_loss_pct', 0.0) * 100:.1f}%",
                    f"{limits.get('max_drawdown_pct', 0.0) * 100:.1f}%"
                ]
            })
            
            st.table(limits_df)
            
            # Controls section
            st.write("### Controls")
            
            # Toggle for emergency controls
            enabled = status.get('enabled', True)
            if st.checkbox("Emergency Controls Enabled", value=enabled):
                if not enabled:
                    # If was disabled and now enabled
                    st.info("This would enable emergency controls")
                    # In a real implementation: self.emergency_controls.enable()
            else:
                if enabled:
                    # If was enabled and now disabled
                    st.warning("⚠️ Disabling emergency controls removes safety protections!")
                    # In a real implementation: self.emergency_controls.disable()
        
        with col2:
            st.write("### Disabled Strategies")
            
            # Show list of disabled strategies
            disabled_strategies = status.get('disabled_strategies', {})
            if disabled_strategies:
                for strategy_id, reason in disabled_strategies.items():
                    st.error(f"**{strategy_id}**")
                    st.write(f"Reason: {reason}")
                    if st.button(f"Re-enable {strategy_id}"):
                        st.info(f"This would re-enable {strategy_id}")
                        # In a real implementation: self.emergency_controls.enable_strategy(strategy_id)
            else:
                st.success("No strategies are currently disabled")
            
            # Recent kill switch events
            st.write("### Recent Kill Switch Events")
            
            if self.kill_switch_events:
                for event in reversed(self.kill_switch_events[-5:]):
                    event_type = event['event_type']
                    timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    if event_type == EventType.KILL_SWITCH_ACTIVATED:
                        reason = event['data'].get('reason', 'Unknown reason')
                        st.error(f"{timestamp}: Activated - {reason}")
                    elif event_type == EventType.KILL_SWITCH_DEACTIVATED:
                        override = event['data'].get('override_reason', 'Manual deactivation')
                        st.warning(f"{timestamp}: Deactivated - {override}")
            else:
                st.info("No recent kill switch events")
    
    def _render_data_quality_tab(self):
        """Render data quality alerts detailed tab."""
        st.subheader("Data Quality Alerts")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by severity
            severity_options = ["All", "Error", "Warning", "Info"]
            selected_severity = st.selectbox("Severity", severity_options)
        
        with col2:
            # Filter by alert type
            all_alert_types = set()
            for alert in self.data_quality_alerts:
                alert_type = alert['data'].get('alert_type')
                if alert_type:
                    all_alert_types.add(alert_type)
            
            alert_type_options = ["All"] + sorted(list(all_alert_types))
            selected_alert_type = st.selectbox("Alert Type", alert_type_options)
        
        with col3:
            # Filter by symbol
            all_symbols = set()
            for alert in self.data_quality_alerts:
                symbol = alert['data'].get('symbol')
                if symbol:
                    all_symbols.add(symbol)
            
            symbol_options = ["All"] + sorted(list(all_symbols))
            selected_symbol = st.selectbox("Symbol", symbol_options)
        
        # Apply filters
        filtered_alerts = self.data_quality_alerts
        
        if selected_severity != "All":
            filtered_alerts = [a for a in filtered_alerts if a['data'].get('severity', '').lower() == selected_severity.lower()]
        
        if selected_alert_type != "All":
            filtered_alerts = [a for a in filtered_alerts if a['data'].get('alert_type') == selected_alert_type]
        
        if selected_symbol != "All":
            filtered_alerts = [a for a in filtered_alerts if a['data'].get('symbol') == selected_symbol]
        
        # Display alerts
        if filtered_alerts:
            for alert in reversed(filtered_alerts):
                data = alert['data']
                timestamp = alert['timestamp'].strftime('%H:%M:%S')
                severity = data.get('severity', 'unknown')
                message = data.get('message', 'No message')
                symbol = data.get('symbol', 'N/A')
                
                # Format alert based on severity
                if severity == 'error':
                    st.error(f"**{timestamp} | {symbol}** - {message}")
                elif severity == 'warning':
                    st.warning(f"**{timestamp} | {symbol}** - {message}")
                else:
                    st.info(f"**{timestamp} | {symbol}** - {message}")
                
                # Show detailed data in an expander
                with st.expander("Details"):
                    # Remove some fields to clean up the display
                    detail_data = data.copy()
                    if 'message' in detail_data:
                        del detail_data['message']
                    if 'severity' in detail_data:
                        del detail_data['severity']
                    
                    st.json(detail_data)
        else:
            st.info("No data quality alerts matching the selected filters")
    
    def _render_position_limits_tab(self):
        """Render position limits detailed tab."""
        st.subheader("Position Limits & Current Positions")
        
        # Get status from emergency controls
        status = self.emergency_controls.get_status_report()
        positions = status.get('positions', {})
        
        if positions:
            # Create a DataFrame for positions
            position_data = []
            for symbol, pos in positions.items():
                size = pos.get('size', 0)
                notional = pos.get('notional', 0)
                max_size = pos.get('max_size', 'unlimited')
                max_notional = pos.get('max_notional', 'unlimited')
                
                # Calculate utilization percentage for size
                if isinstance(max_size, (int, float)) and max_size > 0:
                    size_util = abs(size) / max_size * 100
                else:
                    size_util = 0
                
                # Calculate utilization percentage for notional
                if isinstance(max_notional, (int, float)) and max_notional > 0:
                    notional_util = notional / max_notional * 100
                else:
                    notional_util = 0
                
                position_data.append({
                    'Symbol': symbol,
                    'Size': size,
                    'Notional': f"${notional:,.2f}",
                    'Max Size': max_size,
                    'Max Notional': f"${max_notional:,.2f}" if isinstance(max_notional, (int, float)) else max_notional,
                    'Size Util (%)': f"{size_util:.1f}%",
                    'Notional Util (%)': f"{notional_util:.1f}%"
                })
            
            # Create the DataFrame
            df = pd.DataFrame(position_data)
            
            # Display the table
            st.dataframe(df)
            
            # Add option to adjust position limits
            st.write("### Adjust Position Limits")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_symbol = st.selectbox("Symbol", ["Select..."] + list(positions.keys()))
            
            if selected_symbol != "Select...":
                with col2:
                    current_max_size = positions[selected_symbol].get('max_size', 0)
                    if not isinstance(current_max_size, (int, float)):
                        current_max_size = 0
                    new_max_size = st.number_input("Max Size (Units)", min_value=0, value=int(current_max_size))
                
                with col3:
                    current_max_notional = positions[selected_symbol].get('max_notional', 0)
                    if not isinstance(current_max_notional, (int, float)):
                        current_max_notional = 0
                    new_max_notional = st.number_input("Max Notional Value ($)", min_value=0, value=int(current_max_notional))
                
                if st.button("Update Position Limits"):
                    st.info(f"This would update position limits for {selected_symbol}")
                    # In a real implementation: self.emergency_controls.set_position_limit(selected_symbol, new_max_size, new_max_notional)
        else:
            st.info("No positions are currently tracked")
            
            # Form to add a new position limit
            st.write("### Add New Position Limit")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_symbol = st.text_input("Symbol")
            
            with col2:
                new_max_size = st.number_input("Max Size (Units)", min_value=0, value=100)
            
            with col3:
                new_max_notional = st.number_input("Max Notional Value ($)", min_value=0, value=10000)
            
            if st.button("Add Position Limit"):
                if new_symbol:
                    st.info(f"This would add a position limit for {new_symbol}")
                    # In a real implementation: self.emergency_controls.set_position_limit(new_symbol, new_max_size, new_max_notional)
                else:
                    st.error("Symbol is required")
    
    def _render_orders_tab(self):
        """Render recent orders detailed tab."""
        st.subheader("Recent Order Activity")
        
        if not self.recent_orders:
            st.info("No recent order activity")
            return
        
        # Create a DataFrame for orders
        order_data = []
        for order_event in self.recent_orders:
            data = order_event['data']
            timestamp = order_event['timestamp'].strftime('%H:%M:%S')
            event_type = order_event['event_type'].split('_')[-1].capitalize()  # Extract last part (e.g., "created")
            
            order_data.append({
                'Time': timestamp,
                'Event': event_type,
                'Order ID': data.get('order_id', 'N/A'),
                'Symbol': data.get('symbol', 'N/A'),
                'Side': data.get('side', 'N/A').upper(),
                'Quantity': data.get('quantity', 0),
                'Price': f"${data.get('price', 0):,.2f}" if data.get('price') else 'N/A',
                'Status': self._get_order_status(order_event)
            })
        
        # Create and display the DataFrame
        df = pd.DataFrame(order_data)
        st.dataframe(df)
        
        # Order rejection chart
        st.write("### Order Rejection Rate")
        
        # Calculate rejection rate over time
        total_orders = len(self.recent_orders)
        rejected_orders = sum(1 for o in self.recent_orders if o['event_type'] == EventType.ORDER_REJECTED)
        
        if total_orders > 0:
            rejection_rate = rejected_orders / total_orders * 100
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rejection_rate,
                title={'text': "Rejection Rate (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 30], 'color': "lightyellow"},
                        {'range': [30, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Warning if rejection rate is high
            if rejection_rate > 30:
                st.warning("⚠️ High order rejection rate detected. This could trigger circuit breakers.")
            elif rejection_rate > 10:
                st.info("Order rejection rate is elevated but within acceptable limits.")
        else:
            st.info("Not enough order data to calculate rejection rate")
    
    def _get_order_status(self, order_event):
        """Get a formatted status string for an order."""
        event_type = order_event['event_type']
        data = order_event['data']
        
        if event_type == EventType.ORDER_REJECTED:
            reason = data.get('reason', 'Unknown reason')
            return f"❌ Rejected: {reason}"
        elif event_type == EventType.ORDER_FILLED:
            return "✅ Filled"
        elif event_type == EventType.ORDER_CREATED:
            return "⏳ Created"
        else:
            return "Unknown"


# Demo function to run the panel standalone
def main():
    """Run the risk monitoring panel as a standalone app."""
    st.set_page_config(
        page_title="Risk Monitoring Dashboard",
        page_icon="🛡️",
        layout="wide"
    )
    
    panel = RiskMonitoringPanel()
    panel.render()
    
    # Auto-refresh every 10 seconds
    st.empty()
    st.write("Dashboard auto-refreshes every 10 seconds")
    time.sleep(10)
    st.experimental_rerun()


if __name__ == "__main__":
    main()
