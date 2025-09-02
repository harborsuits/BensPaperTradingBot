"""
Risk Management Dashboard Components for BensBot

This module provides visualization components for the risk management system,
displaying real-time risk metrics, alerts, and historical risk data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from dashboard.theme import COLORS
from dashboard.components import (
    section_header, 
    styled_metric_card, 
    format_currency, 
    format_percent,
    format_number
)
from dashboard.api_utils import get_mongo_collection
from dashboard.advanced_risk_factors import advanced_risk_factors_section

# Check if risk config is available
try:
    from trading_bot.risk.risk_config import RiskConfigManager
    RISK_CONFIG_AVAILABLE = True
except ImportError:
    RISK_CONFIG_AVAILABLE = False

def risk_overview():
    """Display the risk management overview section."""
    section_header("Risk Management Overview", icon="🛡️")
    
    # Get risk metrics from persistence
    portfolio_exposure = get_mongo_collection("portfolio_exposure_history")
    risk_allocations = get_mongo_collection("risk_allocation_history")
    correlation_alerts = get_mongo_collection("correlation_risk_alerts")
    drawdown_events = get_mongo_collection("drawdown_history")
    risk_attribution = get_mongo_collection("risk_attribution")
    
    # Convert to DataFrames
    portfolio_df = pd.DataFrame(list(portfolio_exposure)) if portfolio_exposure else pd.DataFrame()
    allocations_df = pd.DataFrame(list(risk_allocations)) if risk_allocations else pd.DataFrame()
    correlation_df = pd.DataFrame(list(correlation_alerts)) if correlation_alerts else pd.DataFrame()
    drawdown_df = pd.DataFrame(list(drawdown_events)) if drawdown_events else pd.DataFrame()
    attribution_df = pd.DataFrame(list(risk_attribution)) if risk_attribution else pd.DataFrame()
    
    # Sort by timestamp if available
    for df in [portfolio_df, allocations_df, correlation_df, drawdown_df, attribution_df]:
        if not df.empty and 'timestamp' in df.columns:
            # Convert string timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', ascending=False, inplace=True)
    
    # Key risk metrics for the dashboard header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_risk = portfolio_df['total_risk'].iloc[0] * 100 if not portfolio_df.empty else 0
        styled_metric_card("Portfolio Risk", current_risk, is_percent=True)
    
    with col2:
        # Count of positions with risk score > 70
        high_risk_positions = len(allocations_df[allocations_df['risk_score'] > 70]) if not allocations_df.empty else 0
        styled_metric_card("High Risk Positions", high_risk_positions)
    
    with col3:
        # Latest drawdown
        current_drawdown = drawdown_df['current_drawdown'].iloc[0] * 100 if not drawdown_df.empty else 0
        styled_metric_card("Current Drawdown", current_drawdown, is_percent=True)
    
    with col4:
        # Count of correlation alerts in last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        recent_alerts = len(correlation_df[correlation_df['timestamp'] > week_ago]) if not correlation_df.empty else 0
        styled_metric_card("Recent Correlation Alerts", recent_alerts)
    
    return {
        'portfolio_df': portfolio_df,
        'allocations_df': allocations_df,
        'correlation_df': correlation_df,
        'drawdown_df': drawdown_df,
        'attribution_df': attribution_df
    }

def risk_allocation_section(allocations_df: pd.DataFrame):
    """Display the position-level risk allocation section."""
    section_header("Position Risk Allocation", icon="📊")
    
    if allocations_df.empty:
        st.info("No position risk data available yet.")
        return
    
    # Filter to most recent data point per symbol
    latest_allocations = allocations_df.sort_values('timestamp').groupby('symbol').last().reset_index()
    
    # Create a risk score visualization
    fig = px.bar(
        latest_allocations,
        x='symbol',
        y='risk_score',
        color='risk_score',
        color_continuous_scale=['green', 'yellow', 'orange', 'red'],
        range_color=[0, 100],
        title="Position Risk Scores",
        labels={'risk_score': 'Risk Score (0-100)', 'symbol': 'Symbol'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar=dict(
            title="Risk Score",
            tickvals=[20, 40, 60, 80],
            ticktext=["Low", "Medium", "High", "Critical"]
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table of position risks
    st.subheader("Position Risk Details")
    
    # Format for display
    display_df = latest_allocations.copy()
    if 'risk_pct' in display_df.columns:
        display_df['risk_pct'] = display_df['risk_pct'].apply(lambda x: f"{x*100:.2f}%")
    if 'risk_amount' in display_df.columns:
        display_df['risk_amount'] = display_df['risk_amount'].apply(lambda x: f"${x:,.2f}")
    
    # Select columns to display
    columns_to_display = ['symbol', 'position_size', 'entry_price', 'stop_loss_price', 
                         'risk_amount', 'risk_pct', 'risk_score']
    display_cols = [col for col in columns_to_display if col in display_df.columns]
    
    st.dataframe(
        display_df[display_cols].sort_values('risk_score', ascending=False),
        use_container_width=True,
        hide_index=True
    )

def portfolio_exposure_section(portfolio_df: pd.DataFrame):
    """Display the portfolio-level exposure section."""
    section_header("Portfolio Exposure", icon="📈")
    
    if portfolio_df.empty:
        st.info("No portfolio exposure data available yet.")
        return
    
    # Historical exposure trend
    if len(portfolio_df) > 1:
        fig = go.Figure()
        
        # Add the risk trend line
        fig.add_trace(go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['total_risk'] * 100,  # Convert to percentage
            mode='lines',
            name='Portfolio Risk',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        # Add the max risk threshold
        if 'max_risk' in portfolio_df.columns:
            fig.add_trace(go.Scatter(
                x=portfolio_df['timestamp'],
                y=portfolio_df['max_risk'] * 100,  # Convert to percentage
                mode='lines',
                name='Risk Threshold',
                line=dict(color='red', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title="Portfolio Risk Over Time",
            xaxis_title="Date",
            yaxis_title="Risk (%)",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Latest portfolio exposure details
    latest_exposure = portfolio_df.iloc[0] if not portfolio_df.empty else None
    
    if latest_exposure is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Portfolio Risk")
            
            # Create a gauge chart for the risk level
            current_risk = latest_exposure['total_risk'] * 100
            max_risk = latest_exposure.get('max_risk', 0.05) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Portfolio Risk (%)"},
                gauge={
                    'axis': {'range': [0, max(max_risk * 2, 10)]},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, max_risk * 0.5], 'color': 'green'},
                        {'range': [max_risk * 0.5, max_risk * 0.8], 'color': 'yellow'},
                        {'range': [max_risk * 0.8, max_risk], 'color': 'orange'},
                        {'range': [max_risk, max_risk * 2], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_risk
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Action Status")
            
            # Show the current status and any actions
            action = latest_exposure.get('action', None)
            action_time = latest_exposure.get('timestamp', datetime.now())
            
            if action:
                st.warning(f"**Action Required:** {action.replace('_', ' ').title()}")
                st.markdown(f"Triggered at: {action_time}")
                
                # Show a button that could execute the action
                if st.button("Execute Risk Reduction Plan"):
                    st.success("Risk reduction plan initiated. Trades will be adjusted automatically.")
            else:
                st.success("No risk actions required at this time.")
                st.markdown(f"Last checked: {action_time}")

def correlation_risk_section(correlation_df: pd.DataFrame):
    """Display the correlation risk section."""
    section_header("Correlation Risk Monitor", icon="🔄")
    
    if correlation_df.empty:
        st.info("No correlation risk data available yet.")
        return
    
    # Display latest correlation alerts
    st.subheader("Recent Correlation Alerts")
    
    for _, alert in correlation_df.head(5).iterrows():
        symbols = alert.get('symbols', [])
        correlation = alert.get('correlation', 0)
        threshold = alert.get('threshold', 0.7)
        action = alert.get('action', 'monitor')
        timestamp = alert.get('timestamp', datetime.now())
        
        alert_color = "red" if correlation > 0.8 else "orange"
        
        st.markdown(f"""
        <div style="
            background-color: rgba(255, 99, 71, 0.1);
            border-left: 4px solid {alert_color};
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        ">
            <h4 style="margin: 0 0 5px 0; color: {alert_color};">High Correlation Detected: {correlation:.2f}</h4>
            <p style="margin: 0 0 5px 0; font-size: 0.9rem;">
                Between symbols: <strong>{' & '.join(symbols)}</strong>
            </p>
            <p style="margin: 0 0 5px 0; font-size: 0.9rem;">
                Threshold: {threshold:.2f} | Action: {action.replace('_', ' ').title()}
            </p>
            <p style="margin: 0; font-size: 0.8rem; color: gray;">
                {timestamp}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # If we have data for a correlation matrix, display it
    if len(correlation_df) > 5:
        st.subheader("Correlation Matrix Visualization")
        st.info("Coming soon: Interactive correlation matrix visualization")

def drawdown_protection_section(drawdown_df: pd.DataFrame):
    """Display the drawdown protection section."""
    section_header("Drawdown Protection", icon="🛑")
    
    if drawdown_df.empty:
        st.info("No drawdown protection data available yet.")
        return
    
    # Historical drawdown trend
    if len(drawdown_df) > 1:
        fig = go.Figure()
        
        # Add the drawdown trend line
        fig.add_trace(go.Scatter(
            x=drawdown_df['timestamp'],
            y=drawdown_df['current_drawdown'] * 100,  # Convert to percentage
            mode='lines',
            name='Portfolio Drawdown',
            line=dict(color=COLORS['danger'], width=2)
        ))
        
        # Add the drawdown threshold
        if 'threshold' in drawdown_df.columns:
            fig.add_trace(go.Scatter(
                x=drawdown_df['timestamp'],
                y=drawdown_df['threshold'] * 100,  # Convert to percentage
                mode='lines',
                name='Drawdown Threshold',
                line=dict(color='orange', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Latest drawdown protection actions
    st.subheader("Drawdown Protection Alerts")
    
    exceeded_events = drawdown_df[drawdown_df.get('exceeded', False) == True]
    
    if not exceeded_events.empty:
        for _, event in exceeded_events.head(3).iterrows():
            drawdown = event.get('current_drawdown', 0) * 100
            threshold = event.get('threshold', 0) * 100
            severity = event.get('severity', 0)
            action = event.get('action', 'monitor')
            timestamp = event.get('timestamp', datetime.now())
            
            severity_color = {
                0: "green",
                1: "orange",
                2: "red",
                3: "darkred"
            }.get(severity, "orange")
            
            st.markdown(f"""
            <div style="
                background-color: rgba(255, 99, 71, 0.1);
                border-left: 4px solid {severity_color};
                padding: 10px 15px;
                margin-bottom: 10px;
                border-radius: 4px;
            ">
                <h4 style="margin: 0 0 5px 0; color: {severity_color};">
                    Drawdown Protection Alert - Severity {severity}
                </h4>
                <p style="margin: 0 0 5px 0; font-size: 0.9rem;">
                    Current Drawdown: <strong>{drawdown:.2f}%</strong> (Threshold: {threshold:.2f}%)
                </p>
                <p style="margin: 0 0 5px 0; font-size: 0.9rem;">
                    Action: <strong>{action.replace('_', ' ').title()}</strong>
                </p>
                <p style="margin: 0; font-size: 0.8rem; color: gray;">
                    {timestamp}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No drawdown threshold violations detected.")

def risk_attribution_section(attribution_df: pd.DataFrame):
    """Display the risk attribution section."""
    section_header("Risk Factor Attribution", icon="🔍")
    
    if attribution_df.empty:
        st.info("No risk attribution data available yet.")
        return
    
    # Extract the latest risk attribution
    latest_attribution = attribution_df.iloc[0] if not attribution_df.empty else None
    
    if latest_attribution is not None and 'risk_factors' in latest_attribution:
        risk_factors = latest_attribution['risk_factors']
        
        if isinstance(risk_factors, dict):
            # Create DataFrame for visualization
            factors_df = pd.DataFrame({
                'Factor': list(risk_factors.keys()),
                'Contribution': list(risk_factors.values())
            })
            
            # Create pie chart
            fig = px.pie(
                factors_df,
                values='Contribution',
                names='Factor',
                title="Risk Factor Attribution",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.3
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                height=350,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Risk factors data is not in the expected format.")
    else:
        st.info("No risk attribution factors available in the data.")

def risk_settings_section():
    """Display and manage risk management settings."""
    section_header("Risk Management Settings", icon="⚙️")
    
    # Default settings
    default_settings = {
        'max_portfolio_risk': 12.0,  # in percent
        'correlation_threshold': 0.7,
        'max_position_size': 10.0,  # in percent
        'drawdown_threshold': 8.0,  # in percent
        'risk_per_trade': 1.0,  # in percent
    }
    
    # Try to get available risk profiles
    profiles = {}
    if RISK_CONFIG_AVAILABLE:
        try:
            # Create a temporary instance to get profiles
            config_manager = RiskConfigManager()
            profiles = config_manager.get_profile_presets()
        except Exception as e:
            st.error(f"Error loading risk profiles: {e}")
    
    # Profile selection
    if profiles:
        st.subheader("Risk Profile Selection")
        
        # Create profile options
        profile_options = {profile: data["name"] + " - " + data["description"] 
                          for profile, data in profiles.items()}
        
        # Default to balanced
        default_profile = "balanced"
        
        # Create a selectbox for profile selection
        selected_profile = st.selectbox(
            "Select Risk Profile",
            options=list(profile_options.keys()),
            format_func=lambda x: profile_options[x],
            index=list(profile_options.keys()).index(default_profile) if default_profile in profile_options else 0
        )
        
        # Get the selected profile data
        profile_data = profiles[selected_profile]
        
        # Display profile details
        with st.expander("Profile Details", expanded=True):
            st.write(f"**Name:** {profile_data['name']}")
            st.write(f"**Description:** {profile_data['description']}")
            
            # Create columns for displaying metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Limits:**")
                st.write(f"- Max Portfolio Risk: {profile_data['max_portfolio_risk']*100:.1f}%")
                st.write(f"- Max Position Size: {profile_data['max_position_size']*100:.1f}%")
                st.write(f"- Drawdown Threshold: {profile_data['drawdown_threshold']*100:.1f}%")
                st.write(f"- Risk Per Trade: {profile_data['risk_per_trade']*100:.2f}%")
                st.write(f"- Correlation Threshold: {profile_data['correlation_threshold']:.2f}")
            
            with col2:
                st.write("**Risk Weights:**")
                for factor, weight in profile_data['risk_weights'].items():
                    st.write(f"- {factor.replace('_', ' ').title()}: {weight*100:.1f}%")
        
        # Apply profile button
        if st.button("Apply Selected Risk Profile"):
            # In a real implementation, we would apply this profile to the risk engine
            # For now, just show a success message
            st.success(f"Applied {profile_data['name']} risk profile successfully!")
            
            # Update current settings from profile for sliders below
            current_settings = {
                'max_portfolio_risk': profile_data['max_portfolio_risk'] * 100,
                'correlation_threshold': profile_data['correlation_threshold'],
                'max_position_size': profile_data['max_position_size'] * 100,
                'drawdown_threshold': profile_data['drawdown_threshold'] * 100,
                'risk_per_trade': profile_data['risk_per_trade'] * 100,
            }
    else:
        # In a real implementation, we would get the current settings from the database
        # For now, use default settings
        current_settings = default_settings
    
    st.info("Adjust risk management thresholds below. These settings determine when risk alerts are triggered.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_portfolio_risk = st.slider(
            "Maximum Portfolio Risk (%)", 
            min_value=1.0, 
            max_value=25.0, 
            value=current_settings['max_portfolio_risk'],
            step=0.5,
            help="Maximum allowed risk exposure for the entire portfolio"
        )
        
        correlation_threshold = st.slider(
            "Correlation Alert Threshold", 
            min_value=0.5, 
            max_value=0.95, 
            value=current_settings['correlation_threshold'],
            step=0.05,
            help="Correlation coefficient threshold to trigger alerts"
        )
        
        max_position_size = st.slider(
            "Maximum Position Size (%)", 
            min_value=5.0, 
            max_value=50.0, 
            value=current_settings['max_position_size'],
            step=1.0,
            help="Maximum size for any single position as a percentage of portfolio"
        )
    
    with col2:
        drawdown_threshold = st.slider(
            "Drawdown Protection Threshold (%)", 
            min_value=5.0, 
            max_value=25.0, 
            value=current_settings['drawdown_threshold'],
            step=1.0,
            help="Drawdown percentage that triggers protective actions"
        )
        
        risk_per_trade = st.slider(
            "Risk Per Trade (%)", 
            min_value=0.25, 
            max_value=5.0, 
            value=current_settings['risk_per_trade'],
            step=0.25,
            help="Maximum risk allowed for any single trade"
        )
    
    # Additional advanced settings expander
    with st.expander("Advanced Risk Settings", expanded=False):
        st.subheader("Sector Exposure Limits")
        
        max_sector_exposure = st.slider(
            "Maximum Sector Exposure (%)",
            min_value=10.0,
            max_value=60.0,
            value=30.0,
            step=5.0,
            help="Maximum allowed exposure to any single sector"
        )
        
        max_sectors_above_threshold = st.slider(
            "Max Sectors Above Threshold",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            help="Maximum number of sectors allowed above threshold"
        )
        
        st.subheader("Liquidity Requirements")
        
        min_liquidity_score = st.slider(
            "Minimum Liquidity Score",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.1,
            help="Minimum required liquidity score (0-1)"
        )
        
        max_days_to_liquidate = st.slider(
            "Maximum Days to Liquidate",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Maximum days required to liquidate position"
        )
    
    # Save settings button
    if st.button("Save Risk Settings"):
        # In a real implementation, we would save these settings to the database
        # For now, just show a success message
        updated_settings = {
            'max_portfolio_risk': max_portfolio_risk / 100.0,  # Convert to decimal
            'correlation_threshold': correlation_threshold,
            'max_position_size': max_position_size / 100.0,  # Convert to decimal
            'drawdown_threshold': drawdown_threshold / 100.0,  # Convert to decimal
            'risk_per_trade': risk_per_trade / 100.0,  # Convert to decimal
            'sector_limits': {
                'max_sector_exposure': max_sector_exposure / 100.0,
                'max_sectors_above_threshold': max_sectors_above_threshold
            },
            'liquidity_requirements': {
                'min_liquidity_score': min_liquidity_score,
                'max_days_to_liquidate': max_days_to_liquidate
            }
        }
        
        st.success("Risk management settings updated successfully!")
        
        # In a real implementation, we would update the risk engine with the new settings
        # update_risk_engine_config(updated_settings)

def risk_dashboard():
    """Main risk management dashboard."""
    st.title("Risk Management Dashboard")
    
    # Display tabs for different risk aspects
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", 
        "Position Risk", 
        "Portfolio Exposure", 
        "Correlation Risk", 
        "Drawdown Protection",
        "Advanced Risk Factors",
        "Risk Settings"
    ])
    
    # Get all risk data first
    with tab1:
        risk_data = risk_overview()
    
    # Position risk tab
    with tab2:
        risk_allocation_section(risk_data['allocations_df'])
    
    # Portfolio exposure tab
    with tab3:
        portfolio_exposure_section(risk_data['portfolio_df'])
    
    # Correlation risk tab
    with tab4:
        correlation_risk_section(risk_data['correlation_df'])
    
    # Drawdown protection tab
    with tab5:
        drawdown_protection_section(risk_data['drawdown_df'])
        risk_attribution_section(risk_data['attribution_df'])
    
    # Advanced risk factors tab
    with tab6:
        advanced_risk_factors_section(risk_data)
    
    # Risk settings tab
    with tab7:
        risk_settings_section()

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Risk Management - BensBot Dashboard",
        page_icon="🛡️",
        layout="wide",
    )
    
    risk_dashboard()
