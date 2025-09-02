"""
Hybrid Strategy UI Integration

This module integrates the hybrid strategy system with the Streamlit UI,
providing visualization and controls for the strategy hybridization components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import os

# Import our hybrid strategy components
from trading_bot.strategies.hybrid_strategy_system import HybridStrategySystem
from trading_bot.strategies.strategy_factory import StrategyFactory

logger = logging.getLogger(__name__)

class HybridStrategyUI:
    """
    UI Integration for the Hybrid Strategy System
    
    Provides visualizations and controls for managing the hybrid strategy
    voting system, combining traditional, ML, and WeightedAvgPeak strategies.
    """
    
    def __init__(self, config=None):
        """
        Initialize the hybrid strategy UI
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strategy_factory = StrategyFactory()
        
        # Set up session state if needed
        if 'hybrid_strategy_weights' not in st.session_state:
            st.session_state.hybrid_strategy_weights = {
                'technical': 0.4,
                'ml': 0.4,
                'weighted_avg_peak': 0.2
            }
        
        # Set up the hybrid strategy
        if 'hybrid_strategy' not in st.session_state:
            try:
                # Create hybrid strategy instance using factory
                hybrid_strategy_config = self.config.copy()
                
                # Add current weights from session state
                hybrid_strategy_config['strategy_weights'] = st.session_state.hybrid_strategy_weights
                
                # Create the strategy
                st.session_state.hybrid_strategy = self.strategy_factory.create_strategy(
                    'hybrid',
                    config=hybrid_strategy_config
                )
                logger.info("Initialized hybrid strategy for UI")
            except Exception as e:
                logger.error(f"Error initializing hybrid strategy: {e}")
                st.session_state.hybrid_strategy = None
        
        # Initialize history for visualization
        if 'hybrid_votes_history' not in st.session_state:
            st.session_state.hybrid_votes_history = []
    
    def render_hybrid_strategy_controls(self):
        """Render controls for the hybrid strategy system"""
        st.subheader("Hybrid Strategy Controls", anchor=False)
        
        # Strategy component weight controls
        with st.expander("Strategy Component Weights", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                technical_weight = st.slider(
                    "Technical Strategies Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.hybrid_strategy_weights['technical'],
                    step=0.05,
                    key="technical_weight"
                )
            
            with col2:
                ml_weight = st.slider(
                    "ML Strategies Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.hybrid_strategy_weights['ml'],
                    step=0.05,
                    key="ml_weight"
                )
            
            with col3:
                weighted_avg_weight = st.slider(
                    "WeightedAvgPeak Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.hybrid_strategy_weights['weighted_avg_peak'],
                    step=0.05,
                    key="weighted_avg_weight"
                )
            
            # Normalize weights to sum to 1.0
            total_weight = technical_weight + ml_weight + weighted_avg_weight
            
            if total_weight > 0:
                normalized_weights = {
                    'technical': technical_weight / total_weight,
                    'ml': ml_weight / total_weight,
                    'weighted_avg_peak': weighted_avg_weight / total_weight
                }
                
                # Update the weights
                if st.button("Update Strategy Weights"):
                    st.session_state.hybrid_strategy_weights = normalized_weights
                    
                    if st.session_state.hybrid_strategy:
                        try:
                            # Update weights in the strategy
                            st.session_state.hybrid_strategy.set_component_weights(normalized_weights)
                            st.success("Strategy weights updated successfully")
                            logger.info(f"Updated hybrid strategy weights: {normalized_weights}")
                        except Exception as e:
                            st.error(f"Error updating strategy weights: {e}")
                            logger.error(f"Error updating strategy weights: {e}")
            
            # Show current weights
            st.caption("Current normalized weights:")
            st.write(f"Technical: {st.session_state.hybrid_strategy_weights['technical']:.2f} • "
                    f"ML: {st.session_state.hybrid_strategy_weights['ml']:.2f} • "
                    f"WeightedAvgPeak: {st.session_state.hybrid_strategy_weights['weighted_avg_peak']:.2f}")
    
    def render_strategy_votes_visualization(self):
        """Render visualization of strategy votes"""
        st.subheader("Strategy Voting Visualization", anchor=False)
        
        # Check if we have data
        if (not st.session_state.hybrid_strategy or 
            not hasattr(st.session_state.hybrid_strategy, 'get_strategy_votes')):
            st.warning("Hybrid strategy not available or doesn't support vote visualization")
            return
            
        try:
            # Get recent votes
            votes = st.session_state.hybrid_strategy.get_strategy_votes()
            
            if not votes:
                st.info("No recent strategy votes available")
                return
                
            # Add to history
            st.session_state.hybrid_votes_history.extend(votes)
            
            # Limit history size to prevent memory issues
            if len(st.session_state.hybrid_votes_history) > 100:
                st.session_state.hybrid_votes_history = st.session_state.hybrid_votes_history[-100:]
                
            # Create visualization
            votes_df = self._prepare_votes_dataframe(votes)
            
            if votes_df.empty:
                st.info("No vote data available for visualization")
                return
                
            # Show votes visualization
            self._render_votes_chart(votes_df)
            
            # Show recent signals
            with st.expander("Recent Strategy Signals", expanded=False):
                self._render_recent_signals_table(votes)
        except Exception as e:
            st.error(f"Error visualizing strategy votes: {e}")
            logger.error(f"Error visualizing strategy votes: {e}", exc_info=True)
    
    def render_strategy_performance(self):
        """Render strategy performance metrics"""
        st.subheader("Strategy Performance", anchor=False)
        
        # Check if we have data
        if (not st.session_state.hybrid_strategy or 
            not hasattr(st.session_state.hybrid_strategy, 'get_strategy_performance')):
            st.warning("Hybrid strategy not available or doesn't support performance metrics")
            return
            
        try:
            # Get performance metrics
            performance = st.session_state.hybrid_strategy.get_strategy_performance()
            
            if not performance:
                st.info("No performance data available yet")
                return
                
            # Create performance visualization
            self._render_performance_metrics(performance)
            
            # Option to export performance history
            with st.expander("Export/Import Strategy History", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    export_path = st.text_input(
                        "Export Path",
                        value=os.path.join(os.getcwd(), "strategy_history.json"),
                        key="export_path"
                    )
                    
                    if st.button("Export Strategy History"):
                        try:
                            st.session_state.hybrid_strategy.export_strategy_history(export_path)
                            st.success(f"Strategy history exported to {export_path}")
                        except Exception as e:
                            st.error(f"Error exporting strategy history: {e}")
                
                with col2:
                    import_path = st.text_input(
                        "Import Path",
                        value=os.path.join(os.getcwd(), "strategy_history.json"),
                        key="import_path"
                    )
                    
                    if st.button("Import Strategy History"):
                        try:
                            st.session_state.hybrid_strategy.import_strategy_history(import_path)
                            st.success(f"Strategy history imported from {import_path}")
                        except Exception as e:
                            st.error(f"Error importing strategy history: {e}")
        except Exception as e:
            st.error(f"Error rendering strategy performance: {e}")
            logger.error(f"Error rendering strategy performance: {e}", exc_info=True)
    
    def _prepare_votes_dataframe(self, votes):
        """
        Prepare dataframe for votes visualization
        
        Args:
            votes: List of vote dictionaries
            
        Returns:
            DataFrame with vote data
        """
        if not votes:
            return pd.DataFrame()
            
        # Extract vote data
        vote_data = []
        for vote in votes:
            if not vote.get('votes'):
                continue
                
            vote_entry = {
                'timestamp': vote.get('timestamp', datetime.now()),
                'ticker': vote.get('ticker', 'Unknown'),
                'timeframe': vote.get('timeframe', '1d'),
                'action': vote.get('action', 'hold'),
                'confidence': vote.get('confidence', 0.0),
                'buy_votes': vote.get('votes', {}).get('buy', 0.0),
                'sell_votes': vote.get('votes', {}).get('sell', 0.0),
                'hold_votes': vote.get('votes', {}).get('hold', 0.0)
            }
            
            vote_data.append(vote_entry)
        
        # Create dataframe
        df = pd.DataFrame(vote_data)
        
        # Convert timestamp if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _render_votes_chart(self, votes_df):
        """
        Render votes visualization chart
        
        Args:
            votes_df: DataFrame with vote data
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # Add buy votes
            fig.add_trace(go.Scatter(
                x=votes_df['timestamp'],
                y=votes_df['buy_votes'],
                mode='lines+markers',
                name='Buy Votes',
                line=dict(color='green', width=2),
                marker=dict(color='green', size=8)
            ))
            
            # Add sell votes
            fig.add_trace(go.Scatter(
                x=votes_df['timestamp'],
                y=votes_df['sell_votes'],
                mode='lines+markers',
                name='Sell Votes',
                line=dict(color='red', width=2),
                marker=dict(color='red', size=8)
            ))
            
            # Add hold votes
            fig.add_trace(go.Scatter(
                x=votes_df['timestamp'],
                y=votes_df['hold_votes'],
                mode='lines+markers',
                name='Hold Votes',
                line=dict(color='gray', width=2),
                marker=dict(color='gray', size=8)
            ))
            
            # Update layout
            fig.update_layout(
                title='Strategy Votes Over Time',
                xaxis_title='Time',
                yaxis_title='Vote Score',
                template='plotly_dark',
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add signals markers
            signal_fig = go.Figure()
            
            # Color map for signals
            color_map = {'buy': 'green', 'sell': 'red', 'hold': 'gray'}
            
            # Add signals
            signal_fig.add_trace(go.Scatter(
                x=votes_df['timestamp'],
                y=votes_df['confidence'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=[color_map.get(action, 'gray') for action in votes_df['action']],
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                name='Signal',
                text=[f"Action: {row['action']}<br>Confidence: {row['confidence']:.2f}<br>Ticker: {row['ticker']}"
                      for _, row in votes_df.iterrows()],
                hoverinfo='text'
            ))
            
            # Update layout
            signal_fig.update_layout(
                title='Final Signals with Confidence',
                xaxis_title='Time',
                yaxis_title='Confidence',
                template='plotly_dark',
                height=200,
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            # Display chart
            st.plotly_chart(signal_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering votes chart: {e}")
            logger.error(f"Error rendering votes chart: {e}", exc_info=True)
    
    def _render_recent_signals_table(self, votes):
        """
        Render table with recent signals
        
        Args:
            votes: List of vote dictionaries
        """
        if not votes:
            st.info("No recent signals available")
            return
            
        # Create dataframe for display
        display_data = []
        for vote in votes[-10:]:  # Show only the most recent 10
            display_data.append({
                'Timestamp': vote.get('timestamp', '-'),
                'Ticker': vote.get('ticker', '-'),
                'Action': vote.get('action', 'hold').upper(),
                'Confidence': f"{vote.get('confidence', 0.0):.2f}",
                'Explanation': vote.get('explanation', '-')
            })
        
        # Reverse to show newest first
        display_data.reverse()
        
        # Create dataframe
        display_df = pd.DataFrame(display_data)
        
        # Format timestamp if needed
        if 'Timestamp' in display_df.columns:
            display_df['Timestamp'] = pd.to_datetime(display_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display table
        st.table(display_df)
    
    def _render_performance_metrics(self, performance):
        """
        Render performance metrics visualization
        
        Args:
            performance: Dictionary with performance metrics
        """
        # Extract performance data
        tech_perf = performance.get('technical', {})
        ml_perf = performance.get('ml', {})
        wap_perf = performance.get('weighted_avg_peak', {})
        combined_perf = performance.get('combined', {})
        
        # Create metrics visualization
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Technical Signals", f"{tech_perf.get('total_signals', 0)}")
            
            # Signal breakdown
            st.write("Signal Breakdown:")
            st.write(f"Buy: {tech_perf.get('buy_signals', 0)} ({tech_perf.get('buy_pct', 0):.1%})")
            st.write(f"Sell: {tech_perf.get('sell_signals', 0)} ({tech_perf.get('sell_pct', 0):.1%})")
            st.write(f"Hold: {tech_perf.get('hold_signals', 0)} ({tech_perf.get('hold_pct', 0):.1%})")
        
        with col2:
            st.metric("ML Signals", f"{ml_perf.get('total_signals', 0)}")
            
            # Signal breakdown
            st.write("Signal Breakdown:")
            st.write(f"Buy: {ml_perf.get('buy_signals', 0)} ({ml_perf.get('buy_pct', 0):.1%})")
            st.write(f"Sell: {ml_perf.get('sell_signals', 0)} ({ml_perf.get('sell_pct', 0):.1%})")
            st.write(f"Hold: {ml_perf.get('hold_signals', 0)} ({ml_perf.get('hold_pct', 0):.1%})")
        
        with col3:
            st.metric("WeightedAvgPeak Signals", f"{wap_perf.get('total_signals', 0)}")
            
            # Signal breakdown
            st.write("Signal Breakdown:")
            st.write(f"Buy: {wap_perf.get('buy_signals', 0)} ({wap_perf.get('buy_pct', 0):.1%})")
            st.write(f"Sell: {wap_perf.get('sell_signals', 0)} ({wap_perf.get('sell_pct', 0):.1%})")
            st.write(f"Hold: {wap_perf.get('hold_signals', 0)} ({wap_perf.get('hold_pct', 0):.1%})")
        
        with col4:
            st.metric("Combined Signals", f"{combined_perf.get('total_signals', 0)}")
            
            # Signal breakdown
            st.write("Signal Breakdown:")
            st.write(f"Buy: {combined_perf.get('buy_signals', 0)} ({combined_perf.get('buy_pct', 0):.1%})")
            st.write(f"Sell: {combined_perf.get('sell_signals', 0)} ({combined_perf.get('sell_pct', 0):.1%})")
            st.write(f"Hold: {combined_perf.get('hold_signals', 0)} ({combined_perf.get('hold_pct', 0):.1%})")
        
        # Create signal distribution chart
        st.subheader("Signal Distribution", anchor=False)
        
        # Prepare data for chart
        strategies = ['Technical', 'ML', 'WeightedAvgPeak', 'Combined']
        buy_signals = [
            tech_perf.get('buy_pct', 0), 
            ml_perf.get('buy_pct', 0),
            wap_perf.get('buy_pct', 0),
            combined_perf.get('buy_pct', 0)
        ]
        sell_signals = [
            tech_perf.get('sell_pct', 0), 
            ml_perf.get('sell_pct', 0),
            wap_perf.get('sell_pct', 0),
            combined_perf.get('sell_pct', 0)
        ]
        hold_signals = [
            tech_perf.get('hold_pct', 0), 
            ml_perf.get('hold_pct', 0),
            wap_perf.get('hold_pct', 0),
            combined_perf.get('hold_pct', 0)
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=strategies,
            y=buy_signals,
            name='Buy',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            x=strategies,
            y=sell_signals,
            name='Sell',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            x=strategies,
            y=hold_signals,
            name='Hold',
            marker_color='gray'
        ))
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            title='Signal Distribution by Strategy Type',
            xaxis_title='Strategy',
            yaxis_title='Signal Proportion',
            template='plotly_dark',
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)


# Function to add hybrid strategy section to dashboard
def add_hybrid_strategy_section(config=None):
    """
    Add hybrid strategy section to dashboard
    
    Args:
        config: Configuration dictionary
    """
    try:
        # Initialize the hybrid strategy UI
        hybrid_ui = HybridStrategyUI(config)
        
        # Render strategy components
        hybrid_ui.render_hybrid_strategy_controls()
        hybrid_ui.render_strategy_votes_visualization()
        hybrid_ui.render_strategy_performance()
    except Exception as e:
        st.error(f"Error rendering hybrid strategy section: {e}")
        logger.error(f"Error rendering hybrid strategy section: {e}", exc_info=True)
