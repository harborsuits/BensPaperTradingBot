#!/usr/bin/env python3
"""
Risk Control Dashboard Panel

This component provides monitoring and control for the risk management system,
including margin status, circuit breaker status, and manual trading resume capability.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from trading_bot.dashboard.services.data_service import DataService
from trading_bot.core.service_registry import ServiceRegistry


class RiskControlPanel:
    """
    Dashboard component for risk management monitoring and control
    
    Displays:
    - Margin usage across brokers
    - Circuit breaker status
    - Manual trading pause/resume controls
    - Forced exit history
    """
    
    def __init__(self, data_service: DataService):
        """
        Initialize the risk control panel
        
        Args:
            data_service: Data service for retrieving backend data
        """
        self.data_service = data_service
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self.margin_status_cache = None
        self.margin_cache_time = None
        self.circuit_breaker_cache = None
        self.circuit_breaker_cache_time = None
        
        # Cache TTL in seconds
        self.cache_ttl = 15
    
    def _get_risk_manager(self):
        """Get the risk manager from the service registry"""
        return ServiceRegistry.get("risk_manager")
    
    def _get_circuit_breaker(self):
        """Get the circuit breaker from the service registry"""
        return ServiceRegistry.get("circuit_breaker")
    
    def _get_orchestrator(self):
        """Get the orchestrator from the service registry"""
        return ServiceRegistry.get("orchestrator")
    
    def _get_margin_status(self):
        """Get current margin status for all brokers"""
        now = datetime.now()
        
        # Return cached data if still valid
        if (self.margin_status_cache is not None and 
            self.margin_cache_time is not None and
            (now - self.margin_cache_time).total_seconds() < self.cache_ttl):
            return self.margin_status_cache
        
        # Otherwise fetch fresh data
        risk_manager = self._get_risk_manager()
        if risk_manager:
            try:
                margin_status = risk_manager.get_margin_status()
                
                # Cache the result
                self.margin_status_cache = margin_status
                self.margin_cache_time = now
                
                return margin_status
            except Exception as e:
                self.logger.error(f"Error getting margin status: {str(e)}")
                return {}
        
        # Fall back to mock data if no risk manager available
        return self.data_service.get_mock_margin_status()
    
    def _get_circuit_breaker_status(self):
        """Get current circuit breaker status"""
        now = datetime.now()
        
        # Return cached data if still valid
        if (self.circuit_breaker_cache is not None and 
            self.circuit_breaker_cache_time is not None and
            (now - self.circuit_breaker_cache_time).total_seconds() < self.cache_ttl):
            return self.circuit_breaker_cache
        
        # Otherwise fetch fresh data
        circuit_breaker = self._get_circuit_breaker()
        if circuit_breaker:
            try:
                status = circuit_breaker.get_status()
                
                # Cache the result
                self.circuit_breaker_cache = status
                self.circuit_breaker_cache_time = now
                
                return status
            except Exception as e:
                self.logger.error(f"Error getting circuit breaker status: {str(e)}")
                return {}
        
        # Fall back to mock data if no circuit breaker available
        return self.data_service.get_mock_circuit_breaker_status()
    
    def _get_trading_paused_status(self):
        """Get current trading pause status"""
        orchestrator = self._get_orchestrator()
        if orchestrator:
            try:
                return {
                    "paused": orchestrator.trading_paused,
                    "reason": orchestrator.pause_reason,
                    "pause_time": orchestrator.pause_time
                }
            except Exception as e:
                self.logger.error(f"Error getting trading pause status: {str(e)}")
                return {"paused": False}
        
        # Fall back to mock data if no orchestrator available
        return self.data_service.get_mock_trading_pause_status()
    
    def _resume_trading(self):
        """Resume trading after pause"""
        orchestrator = self._get_orchestrator()
        if orchestrator:
            try:
                orchestrator.resume_trading()
                
                # Reset circuit breakers if needed
                circuit_breaker = self._get_circuit_breaker()
                if circuit_breaker:
                    circuit_breaker.reset_breaker()
                
                st.success("Trading resumed successfully")
                
                # Clear caches to refresh data
                self.margin_cache_time = None
                self.circuit_breaker_cache_time = None
                
                return True
            except Exception as e:
                self.logger.error(f"Error resuming trading: {str(e)}")
                st.error(f"Failed to resume trading: {str(e)}")
                return False
        else:
            st.warning("No orchestrator available to resume trading")
            return False
    
    def _render_margin_status(self, margin_status):
        """Render margin status visualization"""
        if not margin_status:
            st.info("No margin data available")
            return
        
        st.subheader("Margin Status")
        
        # Create columns for broker margin cards
        cols = st.columns(min(len(margin_status), 3))
        
        # Create a margin status card for each broker
        for i, (broker_key, status) in enumerate(margin_status.items()):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                # Calculate margin usage percentage
                maintenance_req = status.get("maintenance_requirement", 0)
                margin_used = status.get("margin_used", 0)
                
                if maintenance_req > 0:
                    usage_pct = (margin_used / maintenance_req) * 100
                else:
                    usage_pct = 0
                
                # Determine color based on usage
                if usage_pct >= 90:
                    color = "red"
                elif usage_pct >= 70:
                    color = "orange"
                elif usage_pct >= 50:
                    color = "yellow"
                else:
                    color = "green"
                
                # Create styled card
                st.markdown(f"""
                <div style="border: 1px solid {color}; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                    <h3 style="color: {color};">{broker_key.upper()}</h3>
                    <p><b>Margin Usage:</b> {usage_pct:.1f}%</p>
                    <p><b>Cash:</b> ${status.get('cash', 0):,.2f}</p>
                    <p><b>Margin Used:</b> ${margin_used:,.2f}</p>
                    <p><b>Buying Power:</b> ${status.get('buying_power', 0):,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create margin usage chart
        chart_data = []
        for broker_key, status in margin_status.items():
            maintenance_req = status.get("maintenance_requirement", 0)
            margin_used = status.get("margin_used", 0)
            
            # Skip brokers with no maintenance requirement
            if maintenance_req <= 0:
                continue
                
            usage_pct = (margin_used / maintenance_req) * 100
            remaining_pct = 100 - usage_pct
            
            chart_data.append({
                "broker": broker_key.upper(),
                "type": "Used",
                "percentage": usage_pct
            })
            chart_data.append({
                "broker": broker_key.upper(),
                "type": "Available",
                "percentage": remaining_pct
            })
        
        if chart_data:
            df = pd.DataFrame(chart_data)
            fig = px.bar(
                df, 
                x="broker", 
                y="percentage", 
                color="type",
                color_discrete_map={"Used": "#ff6666", "Available": "#66b366"},
                barmode="stack",
                labels={"percentage": "Percentage", "broker": "Broker"},
                title="Margin Usage by Broker"
            )
            
            # Add a horizontal line at 80% for warning threshold
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(margin_status) - 0.5,
                y0=80,
                y1=80,
                line=dict(color="orange", width=2, dash="dash"),
            )
            
            # Add a horizontal line at 100% for max threshold
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(margin_status) - 0.5,
                y0=100,
                y1=100,
                line=dict(color="red", width=2),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_circuit_breaker_status(self, breaker_status, paused_status):
        """Render circuit breaker status"""
        st.subheader("Circuit Breakers")
        
        # Get status information
        is_triggered = breaker_status.get("is_triggered", False)
        active_breakers = breaker_status.get("active_breakers", [])
        
        # Show trading pause status
        is_paused = paused_status.get("paused", False)
        pause_reason = paused_status.get("reason", "Unknown")
        pause_time = paused_status.get("pause_time")
        
        if is_paused:
            pause_duration = ""
            if pause_time:
                duration = datetime.now() - pause_time
                hours, remainder = divmod(duration.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                pause_duration = f" for {int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            st.error(f"⚠️ Trading is PAUSED{pause_duration} due to: {pause_reason}")
            
            # Add resume button
            if st.button("✅ Resume Trading", key="resume_trading_button"):
                self._resume_trading()
        else:
            st.success("✅ Trading is ACTIVE")
        
        # Circuit breaker summary
        st.markdown("### Breaker Status")
        
        # Create columns for each type of circuit breaker
        col1, col2, col3 = st.columns(3)
        
        # Intraday drawdown
        with col1:
            intraday_active = "intraday" in active_breakers
            color = "red" if intraday_active else "green"
            intraday_dd = breaker_status.get("intraday_drawdown", 0) * 100
            intraday_threshold = breaker_status.get("intraday_threshold", 0) * 100
            
            st.markdown(f"""
            <div style="border: 1px solid {color}; border-radius: 5px; padding: 10px; text-align: center;">
                <h4>Intraday Drawdown</h4>
                <p style="font-size: 24px; color: {color};">{intraday_dd:.1f}%</p>
                <p>Threshold: {intraday_threshold:.1f}%</p>
                <p>Status: {"❌ TRIGGERED" if intraday_active else "✅ Normal"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall drawdown
        with col2:
            overall_active = "overall" in active_breakers
            color = "red" if overall_active else "green"
            overall_dd = breaker_status.get("overall_drawdown", 0) * 100
            overall_threshold = breaker_status.get("overall_threshold", 0) * 100
            
            st.markdown(f"""
            <div style="border: 1px solid {color}; border-radius: 5px; padding: 10px; text-align: center;">
                <h4>Overall Drawdown</h4>
                <p style="font-size: 24px; color: {color};">{overall_dd:.1f}%</p>
                <p>Threshold: {overall_threshold:.1f}%</p>
                <p>Status: {"❌ TRIGGERED" if overall_active else "✅ Normal"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Volatility
        with col3:
            vol_active = "volatility" in active_breakers
            color = "red" if vol_active else "green"
            # Mock volatility data
            current_vol = breaker_status.get("current_volatility", 0.15) * 100
            vol_threshold = breaker_status.get("volatility_threshold", 0.25) * 100
            
            st.markdown(f"""
            <div style="border: 1px solid {color}; border-radius: 5px; padding: 10px; text-align: center;">
                <h4>Volatility</h4>
                <p style="font-size: 24px; color: {color};">{current_vol:.1f}%</p>
                <p>Threshold: {vol_threshold:.1f}%</p>
                <p>Status: {"❌ TRIGGERED" if vol_active else "✅ Normal"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_forced_exits(self):
        """Render history of forced exits"""
        st.subheader("Recent Forced Exits")
        
        # Get forced exit history (mock data for now)
        forced_exits = self.data_service.get_forced_exit_history()
        
        if not forced_exits:
            st.info("No forced exits in the past 24 hours")
            return
        
        # Create a dataframe for display
        df = pd.DataFrame(forced_exits)
        
        # Format the dataframe
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Rename columns for display
        df = df.rename(columns={
            'timestamp': 'Time',
            'symbol': 'Symbol',
            'qty': 'Quantity',
            'reason': 'Reason',
            'price': 'Price',
            'pnl': 'P&L'
        })
        
        # Format P&L with colors
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        # Display the table with styling
        st.dataframe(df.style.applymap(color_pnl, subset=['P&L']), use_container_width=True)
    
    def render(self):
        """Render the risk control panel"""
        st.title("Risk Management")
        
        # Get current status data
        margin_status = self._get_margin_status()
        circuit_breaker_status = self._get_circuit_breaker_status()
        paused_status = self._get_trading_paused_status()
        
        # Render components
        self._render_margin_status(margin_status)
        self._render_circuit_breaker_status(circuit_breaker_status, paused_status)
        self._render_forced_exits()


def create_risk_control_panel(data_service):
    """Factory function to create the risk control panel"""
    return RiskControlPanel(data_service)
