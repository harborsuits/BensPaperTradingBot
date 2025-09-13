#!/usr/bin/env python3
"""
Streamlit Dashboard Component for Broker ML Predictions

Visualizes machine learning predictions for broker performance,
including anomaly detection, failure prediction, and risk assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# For type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_bot.dashboard.services.data_service import DataService


def render_anomaly_detection(broker_id: str, anomaly_data: pd.DataFrame, anomaly_pct: float):
    """
    Render anomaly detection visualization
    
    Args:
        broker_id: Broker ID
        anomaly_data: DataFrame with anomaly flags and scores
        anomaly_pct: Percentage of anomalies in recent data
    """
    st.subheader(f"🔍 Anomaly Detection: {broker_id}")
    
    # Display anomaly percentage
    col1, col2 = st.columns([1, 3])
    with col1:
        # Create a gauge chart for anomaly percentage
        severity = "Low" if anomaly_pct < 0.05 else "Medium" if anomaly_pct < 0.2 else "High"
        color = "green" if anomaly_pct < 0.05 else "orange" if anomaly_pct < 0.2 else "red"
        
        st.metric(
            "Anomaly %", 
            f"{anomaly_pct:.1%}", 
            delta=None,
            delta_color="inverse"
        )
        
        st.markdown(f"**Severity**: <span style='color:{color}'>{severity}</span>", unsafe_allow_html=True)
        
        # If we have anomalies, show count
        if anomaly_pct > 0:
            anomaly_count = (anomaly_data['anomaly'] == -1).sum()
            st.metric("Anomaly Count", f"{anomaly_count}")
    
    with col2:
        if not anomaly_data.empty and 'anomaly_score' in anomaly_data.columns:
            # Plot anomaly scores over time
            fig = go.Figure()
            
            # Get metric columns for plotting
            metric_cols = []
            for col in anomaly_data.columns:
                if col.startswith(('latency_', 'reliability_', 'execution_quality_', 'cost_')) and \
                   not col.endswith(('_lag', '_roll_', '_diff', '_pct')):
                    metric_cols.append(col)
            
            # Allow user to select metric to display
            if metric_cols:
                selected_metric = st.selectbox(
                    "Select metric to display:",
                    options=metric_cols,
                    index=0,
                    key=f"anomaly_metric_{broker_id}"
                )
                
                # Normalize metric for better visualization
                if selected_metric in anomaly_data.columns:
                    normalized_metric = (anomaly_data[selected_metric] - anomaly_data[selected_metric].mean()) / anomaly_data[selected_metric].std()
                    
                    fig.add_trace(go.Scatter(
                        x=anomaly_data.index,
                        y=normalized_metric,
                        mode='lines',
                        name=selected_metric,
                        line=dict(color='blue', width=1)
                    ))
            
            # Add anomaly scores
            fig.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data['anomaly_score'],
                mode='lines',
                name='Anomaly Score',
                line=dict(color='orange', width=1)
            ))
            
            # Highlight anomalies
            anomalies = anomaly_data[anomaly_data['anomaly'] == -1]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies['anomaly_score'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=8, symbol='circle')
                ))
            
            # Add a threshold line
            threshold = anomaly_data['anomaly_score'].quantile(0.05)  # Approximation of the decision boundary
            fig.add_shape(
                type="line",
                x0=anomaly_data.index[0],
                y0=threshold,
                x1=anomaly_data.index[-1],
                y1=threshold,
                line=dict(color="red", width=1, dash="dash"),
            )
            
            fig.update_layout(
                title="Anomaly Detection Results",
                xaxis_title="Time",
                yaxis_title="Score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomalies table
            if not anomalies.empty:
                with st.expander("View Anomalies Table", expanded=False):
                    # Select subset of columns for display
                    display_cols = ['anomaly_score']
                    if selected_metric in anomaly_data.columns:
                        display_cols.append(selected_metric)
                    
                    # Add basic metrics if available
                    for col in ['latency_mean_ms', 'reliability_errors', 'reliability_pct', 'score_overall']:
                        if col in anomaly_data.columns:
                            display_cols.append(col)
                    
                    st.dataframe(anomalies[display_cols])


def render_failure_prediction(broker_id: str, failure_data: pd.DataFrame, failure_prob: float, prediction_window: int):
    """
    Render failure prediction visualization
    
    Args:
        broker_id: Broker ID
        failure_data: DataFrame with failure predictions and probabilities
        failure_prob: Overall failure probability
        prediction_window: Prediction window in hours
    """
    st.subheader(f"⚠️ Failure Prediction: {broker_id}")
    
    # Display failure probability
    col1, col2 = st.columns([1, 3])
    with col1:
        # Create gauge for failure probability
        severity = "Low" if failure_prob < 0.3 else "Medium" if failure_prob < 0.7 else "High"
        color = "green" if failure_prob < 0.3 else "orange" if failure_prob < 0.7 else "red"
        
        st.metric(
            "Failure Probability", 
            f"{failure_prob:.1%}", 
            delta=None,
            delta_color="inverse"
        )
        
        st.markdown(f"**Risk Level**: <span style='color:{color}'>{severity}</span>", unsafe_allow_html=True)
        st.markdown(f"**Prediction Window**: Next {prediction_window} hours")
        
        # If we have a high probability, show prediction time
        if failure_prob > 0.5:
            prediction_time = datetime.now() + timedelta(hours=prediction_window)
            st.markdown(f"**Potential Failure By**: {prediction_time.strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        if not failure_data.empty and 'failure_probability' in failure_data.columns:
            # Plot failure probability over time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=failure_data.index,
                y=failure_data['failure_probability'],
                mode='lines',
                name='Failure Probability',
                line=dict(color='orange', width=2)
            ))
            
            # Add a reference line at 0.5 (50% probability)
            fig.add_shape(
                type="line",
                x0=failure_data.index[0],
                y0=0.5,
                x1=failure_data.index[-1],
                y1=0.5,
                line=dict(color="red", width=1, dash="dash"),
            )
            
            # Add a reference line at 0.3 (30% probability)
            fig.add_shape(
                type="line",
                x0=failure_data.index[0],
                y0=0.3,
                x1=failure_data.index[-1],
                y1=0.3,
                line=dict(color="orange", width=1, dash="dash"),
            )
            
            fig.update_layout(
                title=f"Failure Probability (Within {prediction_window} Hours)",
                xaxis_title="Time",
                yaxis_title="Probability",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
                height=300,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Get factors contributing to prediction
            if failure_prob > 0.3:
                # Get recent data points with high probability
                recent_high_prob = failure_data[failure_data['failure_probability'] > 0.3].iloc[-5:]
                
                with st.expander("Contributing Factors", expanded=False):
                    # Select metrics that might be contributing
                    potential_factors = []
                    for col in failure_data.columns:
                        if col.startswith(('latency_', 'reliability_', 'execution_quality_', 'cost_')) and \
                           not col.endswith(('_lag', '_roll_', '_diff', '_pct')):
                            # Calculate correlation with failure probability
                            if not recent_high_prob.empty:
                                corr = recent_high_prob[col].corr(recent_high_prob['failure_probability'])
                                if abs(corr) > 0.3:  # Only include if there's some correlation
                                    potential_factors.append((col, corr))
                    
                    # Display potential factors
                    if potential_factors:
                        potential_factors.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        for factor, corr in potential_factors[:5]:  # Show top 5
                            direction = "increase" if corr > 0 else "decrease"
                            st.markdown(f"- **{factor}**: {abs(corr):.2f} correlation with failure ({direction})")
                    else:
                        st.info("No clear individual factors identified. The prediction may be based on a combination of factors or temporal patterns.")


def render_broker_risk_summary(broker_id: str, risk_data: Dict[str, Any]):
    """
    Render broker risk summary
    
    Args:
        broker_id: Broker ID
        risk_data: Risk assessment data
    """
    st.subheader(f"🚦 Risk Assessment: {broker_id}")
    
    # Extract risk level and other data
    risk_level = risk_data.get('level', 'unknown')
    action_recommended = risk_data.get('action_recommended', False)
    
    # Determine color
    color = "green" if risk_level == "low" else "orange" if risk_level == "medium" else "red"
    
    # Create summary box
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {'#ffebee' if risk_level == 'high' else '#fff8e1' if risk_level == 'medium' else '#e8f5e9'};">
            <span style="font-size: 1.2em; font-weight: bold;">Overall Risk: <span style="color:{color};">{risk_level.upper()}</span></span>
            {'<br><span style="color:red; font-weight:bold;">ACTION RECOMMENDED</span>' if action_recommended else ''}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display recommendations if any
    if action_recommended:
        st.markdown("### Recommended Actions")
        
        if risk_level == "high":
            st.markdown("""
            1. **Consider Broker Failover** - Initiate failover procedures to alternate brokers
            2. **Increase Monitoring** - Set monitoring to highest frequency and sensitivity
            3. **Diagnostic Investigation** - Investigate underlying causes
            """)
        elif risk_level == "medium":
            st.markdown("""
            1. **Prepare for Potential Issues** - Review failover procedures
            2. **Increase Monitoring** - Set more frequent checks
            3. **Alert Operations Team** - Notify relevant personnel
            """)


def render_model_status(broker_id: str, model_info: Dict[str, Any]):
    """
    Render model status information
    
    Args:
        broker_id: Broker ID
        model_info: Model information
    """
    with st.expander("Model Information", expanded=False):
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("### Anomaly Detection Model")
            if 'anomaly_detection' in model_info:
                anomaly_info = model_info['anomaly_detection']
                st.markdown(f"**Last Updated**: {anomaly_info.get('last_updated', 'Unknown')}")
                st.markdown(f"**Training Period**: {anomaly_info.get('lookback_days', 0)} days")
            else:
                st.markdown("*No anomaly detection model available*")
        
        with cols[1]:
            st.markdown("### Failure Prediction Model")
            if 'failure_prediction' in model_info:
                failure_info = model_info['failure_prediction']
                st.markdown(f"**Last Updated**: {failure_info.get('last_updated', 'Unknown')}")
                st.markdown(f"**Training Period**: {failure_info.get('lookback_days', 0)} days")
                st.markdown(f"**Accuracy**: {failure_info.get('accuracy', 0):.2f}")
                
                # Show failure definition if available
                if 'failure_definition' in failure_info:
                    def_metric = failure_info['failure_definition'].get('metric', '')
                    def_threshold = failure_info['failure_definition'].get('threshold', '')
                    def_op = failure_info['failure_definition'].get('op', '')
                    
                    op_symbol = '>' if def_op == 'gt' else '<' if def_op == 'lt' else '='
                    st.markdown(f"**Failure Definition**: {def_metric} {op_symbol} {def_threshold}")
            else:
                st.markdown("*No failure prediction model available*")


def render_broker_ml_prediction(data_service, broker_id: str):
    """
    Render ML predictions for a single broker
    
    Args:
        data_service: DataService instance
        broker_id: Broker ID to show predictions for
    """
    # Get prediction data
    prediction_data = data_service.get_broker_ml_prediction_data(broker_id)
    
    if not prediction_data:
        st.info(f"No ML prediction data available for broker {broker_id}")
        
        # Show button to build models
        if st.button(f"Build Prediction Models for {broker_id}", key=f"build_{broker_id}"):
            with st.spinner(f"Building prediction models for {broker_id}..."):
                # Trigger model building via API
                success = data_service.trigger_build_broker_models(broker_id)
                if success:
                    st.success(f"Models built successfully for {broker_id}")
                else:
                    st.error(f"Failed to build models for {broker_id}")
        return
    
    # Extract data components
    anomaly_data = prediction_data.get('anomaly_data', pd.DataFrame())
    anomaly_pct = prediction_data.get('anomaly_pct', 0.0)
    failure_data = prediction_data.get('failure_data', pd.DataFrame())
    failure_prob = prediction_data.get('failure_prob', 0.0)
    prediction_window = prediction_data.get('prediction_window', 24)
    risk_assessment = prediction_data.get('risk_assessment', {})
    model_info = prediction_data.get('model_info', {})
    
    # Create tabs for different prediction components
    tabs = st.tabs(["Risk Summary", "Anomaly Detection", "Failure Prediction", "Model Info"])
    
    with tabs[0]:
        render_broker_risk_summary(broker_id, risk_assessment)
    
    with tabs[1]:
        render_anomaly_detection(broker_id, anomaly_data, anomaly_pct)
    
    with tabs[2]:
        render_failure_prediction(broker_id, failure_data, failure_prob, prediction_window)
    
    with tabs[3]:
        render_model_status(broker_id, model_info)


def render_broker_ml_predictions_panel(data_service):
    """
    Render broker ML predictions panel
    
    Args:
        data_service: DataService instance
    """
    st.header("🧠 Broker ML Predictions")
    
    # Get all brokers
    brokers = data_service.get_all_brokers()
    
    if not brokers:
        st.info("No brokers available")
        return
    
    # Allow selecting a broker
    selected_broker = st.selectbox(
        "Select Broker",
        options=[broker['broker_id'] for broker in brokers],
        index=0,
        key="ml_selected_broker"
    )
    
    # Render predictions for selected broker
    render_broker_ml_prediction(data_service, selected_broker)
    
    # Add a section for all broker risk summary
    st.header("🔍 Broker Risk Overview")
    
    # Get risk summary for all brokers
    risk_summary = data_service.get_all_broker_risk_summary()
    
    if not risk_summary:
        st.info("No risk data available for brokers")
        return
    
    # Create summary table
    risk_data = []
    for broker_id, data in risk_summary.items():
        risk_level = data.get('level', 'unknown')
        risk_data.append({
            'Broker': broker_id,
            'Risk Level': risk_level.capitalize(),
            'Anomaly %': data.get('anomaly_pct', 0) * 100,
            'Failure Probability': data.get('failure_prob', 0) * 100,
            'Action Recommended': data.get('action_recommended', False)
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Color-code risk levels
    def highlight_risk(val):
        color = '#e8f5e9'  # light green
        if isinstance(val, str) and val.lower() == 'high':
            color = '#ffebee'  # light red
        elif isinstance(val, str) and val.lower() == 'medium':
            color = '#fff8e1'  # light yellow
        
        return f'background-color: {color}'
    
    # Display styled dataframe
    st.dataframe(risk_df.style.applymap(highlight_risk, subset=['Risk Level']))
    
    # Show high risk brokers more prominently if any
    high_risk_brokers = [b['Broker'] for b in risk_data if b['Risk Level'].lower() == 'high']
    if high_risk_brokers:
        st.warning(f"⚠️ High risk detected for brokers: {', '.join(high_risk_brokers)}")
    
    # Add option to rebuild all models
    with st.expander("Rebuild Models", expanded=False):
        st.markdown("Rebuild prediction models for all brokers or specific brokers.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rebuild All Models"):
                with st.spinner("Rebuilding all broker prediction models..."):
                    # Trigger model rebuilding via API
                    success = data_service.trigger_rebuild_all_broker_models()
                    if success:
                        st.success("All broker models rebuilt successfully")
                    else:
                        st.error("Failed to rebuild broker models")
        
        with col2:
            broker_to_rebuild = st.selectbox(
                "Select Broker to Rebuild",
                options=[broker['broker_id'] for broker in brokers],
                key="rebuild_selected_broker"
            )
            
            if st.button(f"Rebuild {broker_to_rebuild} Models"):
                with st.spinner(f"Rebuilding models for {broker_to_rebuild}..."):
                    # Trigger model rebuilding via API
                    success = data_service.trigger_build_broker_models(broker_to_rebuild)
                    if success:
                        st.success(f"Models for {broker_to_rebuild} rebuilt successfully")
                    else:
                        st.error(f"Failed to rebuild models for {broker_to_rebuild}")


if __name__ == "__main__":
    # For local testing
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    
    from trading_bot.dashboard.services.data_service import DataService
    
    st.set_page_config(
        page_title="Broker ML Predictions",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize data service with mock data
    data_service = DataService(use_mock_data=True)
    
    # Render panel
    render_broker_ml_predictions_panel(data_service)
