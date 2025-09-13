"""
Market Context Component

This component displays market context, regime, and sentiment indicators.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def render_market_context(data_service, account_type: str = None):
    """
    Render the market context and sentiment component.
    
    Args:
        data_service: Data service for fetching market data
    """
    # Get market context data filtered by account type
    market_data = data_service.get_market_context(account_type=account_type)
    
    if not market_data:
        st.warning("No market context data available.")
        return
    
    # Top cards row with key indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Market regime card
        regime = market_data.get('market_regime', 'Neutral')
        regime_color = "#4CAF50" if regime == "Bullish" else "#F44336" if regime == "Bearish" else "#9E9E9E"
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 5px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12); border: 1px solid #ddd;">
            <h3 style="margin-top: 0; color: black;">Market Regime</h3>
            <div style="font-size: 2em; font-weight: bold; color: {regime_color};">{regime}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # VIX card
        vix = market_data.get('vix', 20)
        vix_color = "#F44336" if vix > 25 else "#FF9800" if vix > 20 else "#4CAF50"
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 5px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12); border: 1px solid #ddd;">
            <h3 style="margin-top: 0; color: black;">VIX</h3>
            <div style="font-size: 2em; font-weight: bold; color: {vix_color};">{vix:.2f}</div>
            <div style="color: black;">{'High Volatility' if vix > 25 else 'Moderate Volatility' if vix > 15 else 'Low Volatility'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Fear & Greed Index card
        fg_index = market_data.get('fear_greed_index', 50)
        
        # Determine color and label
        if fg_index <= 25:
            fg_color = "#F44336"
            fg_label = "Extreme Fear"
        elif fg_index <= 40:
            fg_color = "#FF9800"
            fg_label = "Fear"
        elif fg_index <= 60:
            fg_color = "#9E9E9E"
            fg_label = "Neutral"
        elif fg_index <= 75:
            fg_color = "#8BC34A"
            fg_label = "Greed" 
        else:
            fg_color = "#4CAF50"
            fg_label = "Extreme Greed"
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 5px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12); border: 1px solid #ddd;">
            <h3 style="margin-top: 0; color: black;">Fear & Greed Index</h3>
            <div style="font-size: 2em; font-weight: bold; color: {fg_color};">{fg_index}</div>
            <div style="color: {fg_color}; font-weight: bold;">{fg_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Market tabs
    tabs = st.tabs(["Major Indices", "Crypto", "Forex"])
    
    with tabs[0]:
        st.subheader("Major Indices")
        
        # Get indices data
        indices = market_data.get('major_indices', [])
        
        if indices:
            # Create a table
            indices_df = pd.DataFrame(indices)
            
            # Format the dataframe for display
            for idx, row in indices_df.iterrows():
                change_pct = row.get('change_pct', 0)
                price = row.get('price', 0)
                name = row.get('name', f"Index {idx}")
                
                # Set colors based on change
                change_color = "#4CAF50" if change_pct > 0 else "#F44336" if change_pct < 0 else "#9E9E9E"
                
                # Create a card for each index
                st.markdown(f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <div>
                        <h4 style="margin: 0;">{name}</h4>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2em; font-weight: bold;">{price:,.2f}</div>
                        <div style="color: {change_color};">{change_pct:+.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a bar chart of percentage changes
            fig = go.Figure()
            
            for idx, row in indices_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row.get('name', f"Index {idx}")],
                    y=[row.get('change_pct', 0)],
                    name=row.get('name', f"Index {idx}"),
                    marker_color='#4CAF50' if row.get('change_pct', 0) >= 0 else '#F44336'
                ))
            
            fig.update_layout(
                title="Daily Change (%)",
                height=300,
                showlegend=False,
                yaxis_title="Change (%)",
                xaxis_title=None
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No indices data available.")
    
    with tabs[1]:
        st.subheader("Cryptocurrency")
        
        # Get crypto data
        crypto = market_data.get('crypto', [])
        
        if crypto:
            # Create a table
            crypto_df = pd.DataFrame(crypto)
            
            # Format the dataframe for display
            for idx, row in crypto_df.iterrows():
                change_pct = row.get('change_pct', 0)
                price = row.get('price', 0)
                name = row.get('name', f"Crypto {idx}")
                
                # Set colors based on change
                change_color = "#4CAF50" if change_pct > 0 else "#F44336" if change_pct < 0 else "#9E9E9E"
                
                # Create a card for each crypto
                st.markdown(f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <div>
                        <h4 style="margin: 0;">{name}</h4>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2em; font-weight: bold;">${price:,.2f}</div>
                        <div style="color: {change_color};">{change_pct:+.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a bar chart of percentage changes
            fig = go.Figure()
            
            for idx, row in crypto_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row.get('name', f"Crypto {idx}")],
                    y=[row.get('change_pct', 0)],
                    name=row.get('name', f"Crypto {idx}"),
                    marker_color='#4CAF50' if row.get('change_pct', 0) >= 0 else '#F44336'
                ))
            
            fig.update_layout(
                title="Daily Change (%)",
                height=300,
                showlegend=False,
                yaxis_title="Change (%)",
                xaxis_title=None
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cryptocurrency data available.")
    
    with tabs[2]:
        st.subheader("Forex")
        
        # Get forex data
        forex = market_data.get('forex', [])
        
        if forex:
            # Create a table
            forex_df = pd.DataFrame(forex)
            
            # Format the dataframe for display
            for idx, row in forex_df.iterrows():
                change_pct = row.get('change_pct', 0)
                price = row.get('price', 0)
                name = row.get('name', f"Pair {idx}")
                
                # Set colors based on change
                change_color = "#4CAF50" if change_pct > 0 else "#F44336" if change_pct < 0 else "#9E9E9E"
                
                # Create a card for each forex pair
                st.markdown(f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <div>
                        <h4 style="margin: 0;">{name}</h4>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2em; font-weight: bold;">{price:.4f}</div>
                        <div style="color: {change_color};">{change_pct:+.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a bar chart of percentage changes
            fig = go.Figure()
            
            for idx, row in forex_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row.get('name', f"Pair {idx}")],
                    y=[row.get('change_pct', 0)],
                    name=row.get('name', f"Pair {idx}"),
                    marker_color='#4CAF50' if row.get('change_pct', 0) >= 0 else '#F44336'
                ))
            
            fig.update_layout(
                title="Daily Change (%)",
                height=300,
                showlegend=False,
                yaxis_title="Change (%)",
                xaxis_title=None
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forex data available.")
    
    # Market commentary (to be added from news API in the future)
    st.subheader("Market Commentary")
    st.info("Real-time market commentary will be integrated in a future update.")
