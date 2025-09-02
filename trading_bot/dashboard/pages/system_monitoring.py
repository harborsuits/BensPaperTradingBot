"""
System Monitoring Dashboard

This dashboard provides real-time monitoring of all trading system components,
including data quality, emergency controls, and system health.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType, TradingMode
from trading_bot.risk.emergency_controls import EmergencyControls
from trading_bot.data.data_validator import DataValidator


class SystemMonitoringDashboard:
    """Dashboard for monitoring the trading system health and status."""
    
    def __init__(self):
        """Initialize the dashboard components."""
        self.event_bus = get_global_event_bus()
        self.emergency_controls = EmergencyControls()
        self.data_validator = DataValidator()
        
        # State tracking
        self.component_status = self._get_initial_component_status()
        self.data_quality_alerts = []
        self.system_alerts = []
        self.last_refresh = datetime.now()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus."""
        self.event_bus.subscribe(EventType.DATA_QUALITY_ALERT, self._on_data_quality_alert)
        self.event_bus.subscribe(EventType.SYSTEM_ALERT, self._on_system_alert)
        self.event_bus.subscribe(EventType.COMPONENT_STATUS, self._on_component_status)
        self.event_bus.subscribe(EventType.KILL_SWITCH_ACTIVATED, self._on_emergency_event)
        self.event_bus.subscribe(EventType.KILL_SWITCH_DEACTIVATED, self._on_emergency_event)
    
    def _get_initial_component_status(self):
        """Initialize component status tracking."""
        return {
            "market_data": {"status": "healthy", "latency": 120, "last_update": datetime.now()},
            "order_management": {"status": "healthy", "latency": 85, "last_update": datetime.now()},
            "risk_management": {"status": "healthy", "latency": 45, "last_update": datetime.now()},
            "strategy_engine": {"status": "healthy", "latency": 55, "last_update": datetime.now()},
            "position_tracking": {"status": "healthy", "latency": 40, "last_update": datetime.now()},
            "reporting": {"status": "healthy", "latency": 130, "last_update": datetime.now()},
            "data_validation": {"status": "healthy", "latency": 35, "last_update": datetime.now()},
            "emergency_controls": {"status": "healthy", "latency": 30, "last_update": datetime.now()},
            "exchange_connectivity": {"status": "healthy", "latency": 150, "last_update": datetime.now()}
        }
    
    def _on_data_quality_alert(self, event):
        """Handle data quality alert events."""
        self.data_quality_alerts.append({
            "timestamp": datetime.now(),
            "data": event.data
        })
        
        # Keep only the most recent 100 alerts
        if len(self.data_quality_alerts) > 100:
            self.data_quality_alerts = self.data_quality_alerts[-100:]
    
    def _on_system_alert(self, event):
        """Handle system alert events."""
        self.system_alerts.append({
            "timestamp": datetime.now(),
            "data": event.data
        })
        
        # Keep only the most recent 100 alerts
        if len(self.system_alerts) > 100:
            self.system_alerts = self.system_alerts[-100:]
    
    def _on_component_status(self, event):
        """Handle component status update events."""
        component_data = event.data
        component_id = component_data.get("component_id")
        
        if component_id and component_id in self.component_status:
            self.component_status[component_id] = {
                "status": component_data.get("status", "unknown"),
                "latency": component_data.get("latency", 0),
                "last_update": datetime.now()
            }
    
    def _on_emergency_event(self, event):
        """Handle emergency control events."""
        # Add to system alerts with high priority
        self.system_alerts.append({
            "timestamp": datetime.now(),
            "data": {
                "severity": "critical",
                "component": "emergency_controls",
                "message": f"Emergency event: {event.event_type}",
                "details": event.data
            }
        })
    
    def render(self):
        """Render the system monitoring dashboard."""
        st.title("Trading System Monitoring")
        
        # Display refresh information
        refresh_time = datetime.now()
        st.write(f"Last refreshed: {refresh_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Manual refresh button
        if st.button("🔄 Refresh Dashboard"):
            st.experimental_rerun()
        
        # Create tabs for different monitoring views
        tab1, tab2, tab3, tab4 = st.tabs([
            "System Health", 
            "Data Quality", 
            "Emergency Controls",
            "Alerts & Logs"
        ])
        
        with tab1:
            self._render_system_health_tab()
        
        with tab2:
            self._render_data_quality_tab()
        
        with tab3:
            self._render_emergency_controls_tab()
        
        with tab4:
            self._render_alerts_tab()
    
    def _render_system_health_tab(self):
        """Render the system health monitoring tab."""
        st.header("System Health Overview")
        
        # Overall system status
        healthy_components = sum(1 for comp in self.component_status.values() if comp["status"] == "healthy")
        total_components = len(self.component_status)
        
        # Create columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health_percentage = (healthy_components / total_components) * 100
            st.metric("System Health", f"{health_percentage:.0f}%")
            
            # Progress bar for health
            health_color = "normal"
            if health_percentage < 80:
                health_color = "warning"
            if health_percentage < 50:
                health_color = "off"
            
            st.progress(health_percentage / 100, text=f"{healthy_components}/{total_components} components healthy")
        
        with col2:
            # Average latency
            avg_latency = sum(comp["latency"] for comp in self.component_status.values()) / total_components
            st.metric("Avg Latency", f"{avg_latency:.0f} ms")
            
            # Latency threshold indicators
            if avg_latency < 100:
                st.success("✅ Latency optimal (< 100ms)")
            elif avg_latency < 200:
                st.warning("⚠️ Latency acceptable (< 200ms)")
            else:
                st.error("❌ Latency high (> 200ms)")
        
        with col3:
            # Trading mode
            trading_mode = TradingMode.PAPER  # This would be retrieved from configuration
            st.metric("Trading Mode", trading_mode.name)
            
            if trading_mode == TradingMode.PAPER:
                st.info("🧪 Paper trading mode active")
            elif trading_mode == TradingMode.LIVE:
                st.error("💵 Live trading mode active")
            elif trading_mode == TradingMode.STOPPED:
                st.warning("⛔ Trading is stopped")
            else:
                st.success("📊 Backtest mode active")
        
        # Component status table
        st.subheader("Component Status")
        
        # Prepare data for the table
        component_data = []
        for component_id, status in self.component_status.items():
            # Calculate time since last update
            time_since_update = (datetime.now() - status["last_update"]).total_seconds()
            time_display = "Just now"
            
            if time_since_update > 3600:
                time_display = f"{time_since_update/3600:.1f} hours ago"
            elif time_since_update > 60:
                time_display = f"{time_since_update/60:.0f} min ago"
            elif time_since_update > 5:
                time_display = f"{time_since_update:.0f} sec ago"
            
            # Format component ID for display
            display_name = component_id.replace("_", " ").title()
            
            component_data.append({
                "Component": display_name,
                "Status": status["status"].title(),
                "Latency": f"{status['latency']} ms",
                "Last Update": time_display
            })
        
        # Create DataFrame
        df = pd.DataFrame(component_data)
        
        # Apply styling based on status
        def style_status(val):
            if val == "Healthy":
                return "background-color: #c6efce; color: #006100"
            elif val == "Degraded":
                return "background-color: #ffeb9c; color: #9c5700"
            else:
                return "background-color: #ffc7ce; color: #9c0006"
        
        # Display the styled table
        st.dataframe(df.style.applymap(style_status, subset=["Status"]), use_container_width=True)
        
        # System health chart over time
        st.subheader("System Health Trend")
        
        # Create sample data for the chart (in a real implementation, this would use historical data)
        hours = 24
        timestamps = [(datetime.now() - timedelta(hours=i)) for i in range(hours, 0, -1)]
        
        # Generate random health percentages that trend downward then recover
        health_values = []
        for i in range(hours):
            # Start high, dip in the middle, then recover
            if i < hours/3:
                base = 100
            elif i < 2*hours/3:
                base = 85
            else:
                base = 95
            
            # Add some noise
            health = max(0, min(100, base + np.random.normal(0, 5)))
            health_values.append(health)
        
        # Create a DataFrame for the chart
        health_df = pd.DataFrame({
            "Timestamp": timestamps,
            "Health (%)": health_values
        })
        
        # Display the chart
        st.line_chart(health_df.set_index("Timestamp"))
    
    def _render_data_quality_tab(self):
        """Render the data quality monitoring tab."""
        st.header("Data Quality Monitoring")
        
        # Create columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        # Process alerts for metrics
        recent_alerts = [a for a in self.data_quality_alerts 
                         if (datetime.now() - a["timestamp"]).total_seconds() < 3600]  # Last hour
        
        # Count by severity
        error_count = sum(1 for a in recent_alerts if a["data"].get("severity") == "error")
        warning_count = sum(1 for a in recent_alerts if a["data"].get("severity") == "warning")
        info_count = sum(1 for a in recent_alerts if a["data"].get("severity") == "info")
        
        with col1:
            st.metric("Data Errors (1h)", error_count)
            if error_count == 0:
                st.success("✅ No data errors detected")
            else:
                st.error(f"⚠️ {error_count} data errors detected")
        
        with col2:
            st.metric("Data Warnings (1h)", warning_count)
            if warning_count == 0:
                st.success("✅ No data warnings")
            else:
                st.warning(f"⚠️ {warning_count} data warnings")
        
        with col3:
            total_alerts = len(recent_alerts)
            st.metric("Total Alerts (1h)", total_alerts)
            
            if total_alerts == 0:
                st.success("✅ Data quality excellent")
            elif total_alerts < 5:
                st.info("ℹ️ Data quality good")
            else:
                st.warning("⚠️ Data quality needs attention")
        
        # Alert types breakdown
        st.subheader("Alert Types")
        
        # Count alerts by type
        alert_types = {}
        for alert in recent_alerts:
            alert_type = alert["data"].get("alert_type", "unknown")
            if alert_type in alert_types:
                alert_types[alert_type] += 1
            else:
                alert_types[alert_type] = 1
        
        if alert_types:
            # Create pie chart of alert types
            alert_df = pd.DataFrame({
                "Alert Type": list(alert_types.keys()),
                "Count": list(alert_types.values())
            })
            
            fig = go.Figure(data=[go.Pie(
                labels=alert_df["Alert Type"],
                values=alert_df["Count"],
                hole=.3,
                textinfo="percent+label",
                marker=dict(colors=["#FF6B6B", "#4ECDC4", "#FFD166", "#F8A4D8", "#6699CC"])
            )])
            
            fig.update_layout(title_text="Distribution of Alert Types")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data quality alerts in the last hour")
        
        # Symbol data quality breakdown
        st.subheader("Data Quality by Symbol")
        
        # Count alerts by symbol
        symbol_metrics = {}
        for alert in self.data_quality_alerts:
            symbol = alert["data"].get("symbol", "unknown")
            alert_type = alert["data"].get("alert_type", "unknown")
            severity = alert["data"].get("severity", "info")
            
            if symbol not in symbol_metrics:
                symbol_metrics[symbol] = {"error": 0, "warning": 0, "info": 0, "total": 0}
            
            symbol_metrics[symbol][severity] += 1
            symbol_metrics[symbol]["total"] += 1
        
        if symbol_metrics:
            # Create DataFrame for symbol metrics
            symbol_data = []
            for symbol, metrics in symbol_metrics.items():
                symbol_data.append({
                    "Symbol": symbol,
                    "Errors": metrics["error"],
                    "Warnings": metrics["warning"],
                    "Info": metrics["info"],
                    "Total Alerts": metrics["total"]
                })
            
            # Create DataFrame and sort by total alerts descending
            symbol_df = pd.DataFrame(symbol_data)
            symbol_df = symbol_df.sort_values("Total Alerts", ascending=False)
            
            # Display the table
            st.dataframe(symbol_df, use_container_width=True)
            
            # Bar chart of symbols with most alerts
            top_symbols = symbol_df.head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_symbols["Symbol"], y=top_symbols["Errors"], name="Errors", marker_color="crimson"))
            fig.add_trace(go.Bar(x=top_symbols["Symbol"], y=top_symbols["Warnings"], name="Warnings", marker_color="gold"))
            fig.add_trace(go.Bar(x=top_symbols["Symbol"], y=top_symbols["Info"], name="Info", marker_color="royalblue"))
            
            fig.update_layout(
                title="Symbols with Most Data Quality Alerts",
                xaxis_title="Symbol",
                yaxis_title="Number of Alerts",
                barmode="stack"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No symbol-specific data quality metrics available")
        
        # Recent alerts table
        st.subheader("Recent Data Quality Alerts")
        
        if recent_alerts:
            recent_data = []
            for alert in recent_alerts:
                data = alert["data"]
                timestamp = alert["timestamp"].strftime("%H:%M:%S")
                
                recent_data.append({
                    "Time": timestamp,
                    "Symbol": data.get("symbol", "N/A"),
                    "Type": data.get("alert_type", "unknown"),
                    "Severity": data.get("severity", "info").title(),
                    "Message": data.get("message", "No details")
                })
            
            # Create DataFrame
            alerts_df = pd.DataFrame(recent_data)
            
            # Apply styling based on severity
            def style_severity(val):
                if val == "Error":
                    return "background-color: #ffc7ce; color: #9c0006"
                elif val == "Warning":
                    return "background-color: #ffeb9c; color: #9c5700"
                else:
                    return "background-color: #ddebf7; color: #2f5496"
            
            # Display the styled table
            st.dataframe(alerts_df.style.applymap(style_severity, subset=["Severity"]), use_container_width=True)
        else:
            st.success("✅ No recent data quality alerts")
    
    def _render_emergency_controls_tab(self):
        """Render the emergency controls monitoring tab."""
        st.header("Emergency Controls Status")
        
        # Get emergency controls status
        kill_switch_active = self.emergency_controls.kill_switch_activated
        
        # Create columns for key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Kill Switch Status")
            
            if kill_switch_active:
                st.error("⚠️ KILL SWITCH ACTIVE ⚠️")
                st.write("All trading has been halted")
                
                # Add button to deactivate
                if st.button("Deactivate Kill Switch (Override)"):
                    st.warning("This would deactivate the kill switch")
                    # In a real implementation: self.emergency_controls.deactivate_kill_switch()
            else:
                st.success("✅ Kill Switch Inactive")
                st.write("Trading system is operating normally")
                
                # Add button to activate
                if st.button("Activate Kill Switch (Emergency)"):
                    st.warning("This will immediately halt all trading!")
                    # In a real implementation: self.emergency_controls.activate_kill_switch("Manual activation from dashboard")
        
        with col2:
            st.subheader("Risk Parameters")
            
            # Get risk limits from emergency controls
            risk_params = {
                "Max Daily Loss": "5%",
                "Max Position Size": "20%",
                "Max Drawdown": "10%",
                "Circuit Breaker Threshold": "50%"
            }
            
            # Display as a table
            risk_df = pd.DataFrame({
                "Parameter": list(risk_params.keys()),
                "Value": list(risk_params.values())
            })
            
            st.table(risk_df)
            
            # Emergency controls toggle
            ec_enabled = True  # Would be retrieved from emergency controls
            if st.checkbox("Emergency Controls Enabled", value=ec_enabled):
                if not ec_enabled:
                    st.info("This would enable emergency controls")
                    # In a real implementation: self.emergency_controls.enable()
            else:
                if ec_enabled:
                    st.warning("⚠️ Disabling emergency controls removes critical safety protections!")
                    # In a real implementation: self.emergency_controls.disable()
        
        # Position limits section
        st.subheader("Position Limits")
        
        # Sample position data (in a real implementation, this would be retrieved from emergency controls)
        positions = [
            {"Symbol": "AAPL", "Current Size": 100, "Max Size": 200, "Current Value": "$17,235.00", "Max Value": "$50,000.00", "Utilization": "34.5%"},
            {"Symbol": "MSFT", "Current Size": 75, "Max Size": 150, "Current Value": "$21,573.75", "Max Value": "$50,000.00", "Utilization": "43.1%"},
            {"Symbol": "GOOGL", "Current Size": 40, "Max Size": 100, "Current Value": "$5,530.00", "Max Value": "$50,000.00", "Utilization": "11.1%"},
            {"Symbol": "AMZN", "Current Size": 30, "Max Size": 80, "Current Value": "$4,134.00", "Max Value": "$50,000.00", "Utilization": "8.3%"},
            {"Symbol": "TSLA", "Current Size": 50, "Max Size": 100, "Current Value": "$10,172.50", "Max Value": "$50,000.00", "Utilization": "20.3%"}
        ]
        
        # Display position table
        pos_df = pd.DataFrame(positions)
        st.dataframe(pos_df, use_container_width=True)
        
        # Circuit breaker status section
        st.subheader("Circuit Breaker Status")
        
        # Sample circuit breaker data
        circuit_breakers = [
            {"Type": "Order Rejection Rate", "Threshold": "50%", "Current Value": "12%", "Status": "Normal"},
            {"Type": "Trade Frequency", "Threshold": "100 trades/min", "Current Value": "37 trades/min", "Status": "Normal"},
            {"Type": "Daily Loss", "Threshold": "5% of capital", "Current Value": "1.2% of capital", "Status": "Normal"},
            {"Type": "Drawdown", "Threshold": "10% from peak", "Current Value": "2.3% from peak", "Status": "Normal"}
        ]
        
        # Create DataFrame
        cb_df = pd.DataFrame(circuit_breakers)
        
        # Apply styling based on status
        def style_cb_status(val):
            if val == "Normal":
                return "background-color: #c6efce; color: #006100"
            elif val == "Warning":
                return "background-color: #ffeb9c; color: #9c5700"
            else:
                return "background-color: #ffc7ce; color: #9c0006"
        
        # Display the styled table
        st.dataframe(cb_df.style.applymap(style_cb_status, subset=["Status"]), use_container_width=True)
    
    def _render_alerts_tab(self):
        """Render the system alerts and logs tab."""
        st.header("System Alerts & Logs")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by severity
            severity_options = ["All", "Critical", "Error", "Warning", "Info"]
            selected_severity = st.selectbox("Severity", severity_options)
        
        with col2:
            # Filter by component
            component_options = ["All"] + list(self.component_status.keys())
            selected_component = st.selectbox("Component", component_options)
        
        # Get recent alerts
        recent_alerts = self.system_alerts
        
        # Apply filters
        if selected_severity != "All":
            recent_alerts = [a for a in recent_alerts 
                            if a["data"].get("severity", "").lower() == selected_severity.lower()]
        
        if selected_component != "All":
            recent_alerts = [a for a in recent_alerts 
                            if a["data"].get("component", "") == selected_component]
        
        # Recent alerts section
        st.subheader("Recent Alerts")
        
        if recent_alerts:
            alert_data = []
            for alert in recent_alerts:
                data = alert["data"]
                timestamp = alert["timestamp"].strftime("%H:%M:%S")
                
                alert_data.append({
                    "Time": timestamp,
                    "Component": data.get("component", "system").replace("_", " ").title(),
                    "Severity": data.get("severity", "info").title(),
                    "Message": data.get("message", "No details")
                })
            
            # Create DataFrame
            alerts_df = pd.DataFrame(alert_data)
            
            # Apply styling based on severity
            def style_alert_severity(val):
                if val == "Critical":
                    return "background-color: #7e0025; color: white"
                elif val == "Error":
                    return "background-color: #ffc7ce; color: #9c0006"
                elif val == "Warning":
                    return "background-color: #ffeb9c; color: #9c5700"
                else:
                    return "background-color: #ddebf7; color: #2f5496"
            
            # Display the styled table
            st.dataframe(alerts_df.style.applymap(style_alert_severity, subset=["Severity"]), use_container_width=True)
            
            # Allow clearing alerts
            if st.button("Clear All Alerts"):
                st.warning("This would clear all system alerts")
                # In a real implementation: self.system_alerts = []
        else:
            st.success("✅ No system alerts matching the selected filters")
        
        # System logs section
        st.subheader("System Logs")
        
        # In a real implementation, this would retrieve actual logs
        logs = [
            {"timestamp": "22:30:15", "level": "INFO", "source": "strategy_engine", "message": "Strategy 'MeanReversion' executed successfully"},
            {"timestamp": "22:29:45", "level": "INFO", "source": "order_management", "message": "Order 'ORD-1234' filled at $152.30"},
            {"timestamp": "22:28:36", "level": "WARNING", "source": "data_validation", "message": "Slightly stale data detected for TSLA"},
            {"timestamp": "22:25:12", "level": "INFO", "source": "position_tracking", "message": "Position updated for AAPL: 100 shares"},
            {"timestamp": "22:22:58", "level": "INFO", "source": "market_data", "message": "Received market data update for 32 symbols"},
            {"timestamp": "22:20:30", "level": "ERROR", "source": "exchange_connectivity", "message": "Temporary connection issue, reconnecting..."},
            {"timestamp": "22:18:15", "level": "INFO", "source": "reporting", "message": "Generated hourly performance report"},
            {"timestamp": "22:15:42", "level": "INFO", "source": "emergency_controls", "message": "Position limits checked and enforced"}
        ]
        
        # Create DataFrame
        logs_df = pd.DataFrame(logs)
        
        # Apply styling based on log level
        def style_log_level(val):
            if val == "ERROR":
                return "background-color: #ffc7ce; color: #9c0006"
            elif val == "WARNING":
                return "background-color: #ffeb9c; color: #9c5700"
            elif val == "INFO":
                return "background-color: #ddebf7; color: #2f5496"
            else:
                return ""
        
        # Display the styled table
        st.dataframe(logs_df.style.applymap(style_log_level, subset=["level"]), use_container_width=True)


def main():
    """Run the system monitoring dashboard."""
    st.set_page_config(
        page_title="System Monitoring",
        page_icon="🔍",
        layout="wide"
    )
    
    dashboard = SystemMonitoringDashboard()
    dashboard.render()
    
    # For demonstration, add auto-refresh hint
    st.sidebar.write("Dashboard updates in real-time")
    st.sidebar.info("You can also click the 'Refresh Dashboard' button to manually refresh")


if __name__ == "__main__":
    main()
