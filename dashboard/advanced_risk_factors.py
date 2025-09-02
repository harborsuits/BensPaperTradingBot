"""
Advanced Risk Factors Dashboard Module

This module provides visualization components for the advanced risk factors,
including liquidity risk, factor tilts, sector exposures, and concentration risk.
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

def advanced_risk_factors_section(risk_data=None):
    """Display the advanced risk factors section."""
    section_header("Advanced Risk Factors", icon="🧪")
    
    # Get enhanced risk factor data
    liquidity_risk = get_mongo_collection("liquidity_risk_history")
    factor_exposures = get_mongo_collection("factor_exposures_history")
    sector_exposures = get_mongo_collection("sector_exposures_history")
    concentration_risk = get_mongo_collection("concentration_risk_history")
    geographic_risk = get_mongo_collection("geographic_risk_history")
    
    # Convert to DataFrames
    liquidity_df = pd.DataFrame(list(liquidity_risk)) if liquidity_risk else pd.DataFrame()
    factor_df = pd.DataFrame(list(factor_exposures)) if factor_exposures else pd.DataFrame()
    sector_df = pd.DataFrame(list(sector_exposures)) if sector_exposures else pd.DataFrame()
    concentration_df = pd.DataFrame(list(concentration_risk)) if concentration_risk else pd.DataFrame()
    geographic_df = pd.DataFrame(list(geographic_risk)) if geographic_risk else pd.DataFrame()
    
    # Sort by timestamp if available
    for df in [liquidity_df, factor_df, sector_df, concentration_df, geographic_df]:
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', ascending=False, inplace=True)
    
    # Create accordion sections for each advanced risk factor
    with st.expander("Liquidity Risk Analysis", expanded=True):
        if liquidity_df.empty:
            st.info("No liquidity risk data available yet.")
        else:
            # Get the latest liquidity risk data
            latest_liquidity = liquidity_df.iloc[0]
            
            # Display key liquidity metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                styled_metric_card("Portfolio Liquidity Score", 
                                  latest_liquidity.get('portfolio_liquidity_score', 0.0),
                                  is_percent=False)
            with col2:
                styled_metric_card("Days to Liquidate", 
                                  latest_liquidity.get('days_to_liquidate', 0.0),
                                  is_percent=False)
            with col3:
                styled_metric_card("Liquidity Risk Level", 
                                  latest_liquidity.get('liquidity_risk_level', 'Low'),
                                  is_percent=False)
            
            # Create positions at risk table
            positions_at_risk = latest_liquidity.get('positions_at_risk', [])
            if positions_at_risk:
                st.subheader("Positions with Liquidity Risk")
                positions_df = pd.DataFrame(positions_at_risk)
                st.dataframe(positions_df)
            
            # Historical liquidity chart
            if len(liquidity_df) > 1:
                st.subheader("Historical Liquidity Trend")
                fig = px.line(liquidity_df, 
                             x='timestamp', 
                             y='portfolio_liquidity_score',
                             title="Portfolio Liquidity Score Over Time")
                st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Factor Exposures", expanded=False):
        if factor_df.empty:
            st.info("No factor exposure data available yet.")
        else:
            # Get the latest factor exposure data
            latest_factor = factor_df.iloc[0]
            
            # Create a radar chart for factor exposures
            st.subheader("Factor Exposure Analysis")
            
            # Extract factor exposures
            factor_exposures = latest_factor.get('factor_exposures', {})
            
            if factor_exposures:
                # Create a radar chart using plotly
                categories = list(factor_exposures.keys())
                values = list(factor_exposures.values())
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Current Factor Exposures'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(values) * 1.2]
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display factor exposures in a table
                st.subheader("Factor Exposures Detail")
                factor_df = pd.DataFrame([factor_exposures]).T.reset_index()
                factor_df.columns = ['Factor', 'Exposure']
                factor_df['Exposure'] = factor_df['Exposure'].round(4)
                st.dataframe(factor_df)
                
                # Display risk level and factors at risk
                col1, col2 = st.columns(2)
                with col1:
                    styled_metric_card("Factor Risk Level", 
                                      latest_factor.get('factor_risk_level', 'Low'),
                                      is_percent=False)
                
                # Factors at risk
                factors_at_risk = latest_factor.get('factors_at_risk', [])
                if factors_at_risk:
                    with col2:
                        styled_metric_card("Factors Above Threshold", 
                                          len(factors_at_risk),
                                          is_percent=False)
                    
                    st.subheader("Factors Above Threshold")
                    factors_df = pd.DataFrame(factors_at_risk)
                    st.dataframe(factors_df)
    
    with st.expander("Sector Exposures", expanded=False):
        if sector_df.empty:
            st.info("No sector exposure data available yet.")
        else:
            # Get the latest sector exposure data
            latest_sector = sector_df.iloc[0]
            
            # Create a pie chart for sector exposures
            st.subheader("Sector Allocation Analysis")
            
            # Extract sector exposures
            sector_exposures = latest_sector.get('sector_exposures', {})
            
            if sector_exposures:
                # Create a pie chart
                sector_names = list(sector_exposures.keys())
                sector_values = list(sector_exposures.values())
                
                fig = px.pie(
                    values=sector_values,
                    names=sector_names,
                    title="Sector Allocation"
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sector risk level
                col1, col2 = st.columns(2)
                with col1:
                    styled_metric_card("Sector Risk Level", 
                                      latest_sector.get('sector_risk_level', 'Low'),
                                      is_percent=False)
                
                # Sectors at risk
                sectors_at_risk = latest_sector.get('sectors_at_risk', [])
                if sectors_at_risk:
                    with col2:
                        styled_metric_card("Sectors Above Threshold", 
                                          len(sectors_at_risk),
                                          is_percent=False)
                    
                    st.subheader("Sectors Above Threshold")
                    sectors_df = pd.DataFrame(sectors_at_risk)
                    st.dataframe(sectors_df)
    
    with st.expander("Geographic Exposure", expanded=False):
        if geographic_df.empty:
            st.info("No geographic exposure data available yet.")
        else:
            # Display geographic exposure data
            latest_geo = geographic_df.iloc[0]
            
            # Extract country exposures
            country_exposures = latest_geo.get('country_exposures', {})
            
            if country_exposures:
                st.subheader("Geographic Allocation")
                
                # Create a bar chart for country exposures
                country_df = pd.DataFrame(list(country_exposures.items()), 
                                         columns=['Country', 'Exposure'])
                country_df = country_df.sort_values('Exposure', ascending=False)
                
                fig = px.bar(
                    country_df,
                    x='Country',
                    y='Exposure',
                    title="Geographic Allocation",
                    color='Exposure'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display geographic risk level
                styled_metric_card("Geographic Risk Level", 
                                  latest_geo.get('geographic_risk_level', 'Low'),
                                  is_percent=False)
    
    with st.expander("Concentration Risk", expanded=False):
        if concentration_df.empty:
            st.info("No concentration risk data available yet.")
        else:
            # Display concentration risk data
            latest_conc = concentration_df.iloc[0]
            
            # Display concentration metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                styled_metric_card("Concentration Score", 
                                  latest_conc.get('concentration_score', 0.0),
                                  is_percent=False)
            with col2:
                styled_metric_card("Top 5 Positions %", 
                                  latest_conc.get('top_5_concentration', 0.0) * 100,
                                  is_percent=True)
            with col3:
                styled_metric_card("Concentration Risk Level", 
                                  latest_conc.get('concentration_risk_level', 'Low'),
                                  is_percent=False)
            
            # Display positions contributing to concentration
            positions = latest_conc.get('top_positions', [])
            if positions:
                st.subheader("Top Positions by Concentration")
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df)
