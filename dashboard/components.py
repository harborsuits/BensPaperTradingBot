"""
UI Components for the BensBot Trading Dashboard
"""
import streamlit as st
from typing import Dict, List, Optional, Any
import pandas as pd

from dashboard.theme import COLORS

def section_header(title: str, icon: str = "") -> None:
    """Display a section header with an optional icon."""
    st.markdown(f"<div class='section-header'>{icon}  {title}</div>", unsafe_allow_html=True)

def format_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}" if value is not None else "-"

def format_percent(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:.2f}%" if value is not None else "-"

def format_number(value: float) -> str:
    """Format a number with commas."""
    return f"{value:,.2f}" if value is not None else "-"

def styled_metric_card(label: str, value: Any, delta: Optional[float] = None, 
                       prefix: str = "", suffix: str = "", is_currency: bool = False,
                       is_percent: bool = False):
    """Display a styled metric card with a label and value."""
    formatted_value = value
    delta_color: str | None = None
    
    if is_currency and isinstance(value, (int, float)):
        formatted_value = f"${value:,.2f}"
    elif is_percent and isinstance(value, (int, float)):
        formatted_value = f"{value:.2f}%"
    
    if delta is not None:
        # Use Streamlit's default (normal) coloring for deltas
        delta_color = "normal"
            
    metric_kwargs = dict(label=label, value=f"{prefix}{formatted_value}{suffix}")
    if delta is not None:
        metric_kwargs["delta"] = delta
        if delta_color:
            metric_kwargs["delta_color"] = delta_color

    st.metric(**metric_kwargs)

def strategy_status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    colors = {
        "active": "#28a745",
        "pending_win": "#ffc107",
        "experimental": "#17a2b8",
        "failed": "#dc3545"
    }
    status_display = {
        "active": "Active",
        "pending_win": "Pending Approval",
        "experimental": "Experimental",
        "failed": "Failed"
    }
    color = colors.get(status, "#6c757d")
    display = status_display.get(status, status.capitalize())
    
    return f"""
    <div style="
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        background-color: {color};
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
    ">
        {display}
    </div>
    """

def strategy_card(strategy: Dict, action: Optional[str] = None):
    """Display a styled card for a strategy with optional action buttons."""
    from dashboard.api_utils import approve_strategy, delete_strategy
    
    with st.container():
        # Card container with dark theme styling
        st.markdown(f"""
        <div style="background-color: #1E293B; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); margin-bottom: 1.2rem;">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Strategy name and status
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                <h3 style="margin: 0; font-size: 1.3rem; color: #F8FAFC;">{strategy.get('name', 'Unnamed Strategy')}</h3>
                <div style="margin-left: 12px;">
                    {strategy_status_badge(strategy.get('status', ''))}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Description
            st.markdown(f"""
            <p style="margin: 0 0 0.8rem 0; color: #E2E8F0;">
                {strategy.get('description', 'No description available')}
            </p>
            """, unsafe_allow_html=True)
            
            # Strategy details
            metrics = []
            if strategy.get('win_rate') is not None:
                metrics.append(f"Win Rate: {strategy['win_rate']:.1f}%" if isinstance(strategy['win_rate'], (int, float)) else "Win Rate: Not Available")
            if strategy.get('profit_factor') is not None:
                metrics.append(f"Profit Factor: {strategy['profit_factor']:.2f}" if isinstance(strategy['profit_factor'], (int, float)) else "Profit Factor: Not Available")
            if strategy.get('sharpe') is not None:
                metrics.append(f"Sharpe: {strategy['sharpe']:.2f}" if isinstance(strategy['sharpe'], (int, float)) else "Sharpe: Not Available")
            if strategy.get('trades') is not None:
                metrics.append(f"Trades: {strategy['trades']}")
                
            if metrics:
                st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 0.5rem;">
                    {''.join([f'<div style="background: #334155; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; color: #E2E8F0;">{metric}</div>' for metric in metrics])}
                </div>
                """, unsafe_allow_html=True)
            
            # Display strategy parameters if available
            if 'parameters' in strategy and strategy['parameters']:
                with st.expander("Strategy Parameters", expanded=False):
                    params = strategy['parameters']
                    cols = st.columns(2)
                    for i, (param_name, param_value) in enumerate(params.items()):
                        with cols[i % 2]:
                            if isinstance(param_value, (int, float)):
                                # For numerical parameters, allow editing with a slider
                                min_val = param_value * 0.5
                                max_val = param_value * 1.5
                                step = (max_val - min_val) / 100
                                
                                if param_name.endswith('_pct') or 'percent' in param_name:
                                    # Handle percentage values
                                    new_value = st.slider(f"{param_name.replace('_', ' ').title()} (%)", 
                                                    min_value=float(min_val), 
                                                    max_value=float(max_val), 
                                                    value=float(param_value),
                                                    step=float(step),
                                                    format="%.2f%%",
                                                    key=f"{strategy.get('id', strategy.get('name', 'unknown'))}_{param_name}_pct")
                                elif param_name.endswith('_period') or 'period' in param_name:
                                    # Handle period values as integers
                                    new_value = st.slider(f"{param_name.replace('_', ' ').title()}", 
                                                    min_value=int(max(1, min_val)), 
                                                    max_value=int(max_val), 
                                                    value=int(param_value),
                                                    key=f"{strategy.get('id', strategy.get('name', 'unknown'))}_{param_name}_period")
                                else:
                                    new_value = st.slider(f"{param_name.replace('_', ' ').title()}", 
                                                    min_value=float(min_val), 
                                                    max_value=float(max_val), 
                                                    value=float(param_value),
                                                    step=float(step),
                                                    key=f"{strategy.get('id', strategy.get('name', 'unknown'))}_{param_name}")
                            else:
                                # For non-numerical parameters, just display them
                                st.text_input(f"{param_name.replace('_', ' ').title()}", value=str(param_value), key=f"{strategy.get('id', strategy.get('name', 'unknown'))}_{param_name}_text")
                    
                    if strategy.get('status') == 'template':
                        if st.button("Test Strategy with These Parameters", key=f"test-params-{strategy['id']}"):
                            st.toast("Testing strategy with custom parameters...", icon="🧪")
                            # This would call backend to test the strategy with user's parameters
                            st.success("Test initiated! Results will be available in the Backtesting tab.")
        
        with col2:
            if action == "approve":
                if st.button("✅ Approve", key=f"approve-{strategy['id']}-{hash(strategy.get('name', ''))}", use_container_width=True):
                    ok = approve_strategy(strategy['id'])
                    if ok:
                        st.toast("Strategy approved!", icon="🎉")
                        st.rerun()
            elif action == "delete":
                if st.button("🗑️ Delete", key=f"delete-{strategy['id']}-{hash(strategy.get('name', ''))}", use_container_width=True):
                    ok = delete_strategy(strategy['id'])
                    if ok:
                        st.toast("Strategy deleted!", icon="✅")
                        st.rerun()
            elif action == "view":
                st.button("📊 Details", key=f"view-{strategy['id']}-{hash(strategy.get('name', ''))}", use_container_width=True)
                
            # Add activate button for templates
            if strategy.get('status') == 'template':
                st.button("🚀 Activate", key=f"activate-{strategy['id']}-{hash(strategy.get('name', ''))}", use_container_width=True,
                          help="Deploy this strategy template to the trading system")
                
            # Add backtest button
            if 'backtest_results' not in strategy:
                st.button("🔍 Backtest", key=f"backtest-{strategy['id']}-{hash(strategy.get('name', ''))}", use_container_width=True,
                          help="Run historical backtest on this strategy")
            else:
                st.button("📈 View Results", key=f"results-{strategy['id']}-{hash(strategy.get('name', ''))}", use_container_width=True,
                          help="View detailed backtest results")
        
        # Close the card container
        st.markdown("</div>", unsafe_allow_html=True)

def strategy_lane(status: str, *, title: str, icon: str, action: str | None = None) -> None:
    """Display a lane of strategy cards with the same status."""
    from dashboard.api_utils import get_strategies
    
    section_header(title, icon)
    strategies = get_strategies(status=status)
    
    if not strategies:
        st.info("Nothing here right now 🚀")
        return
    
    # Display each strategy as a card
    for strat in strategies:
        strategy_card(strat, action)

def event_system_status_card():
    """Display a card showing event system status"""
    from dashboard.api_utils import get_event_bus_status
    
    event_status = get_event_bus_status()
    
    with st.container():
        st.markdown("""
        <div style="background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <h3 style="margin-top: 0; font-size: 1.2rem;">Event System Status</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Default status values in case the real ones aren't available
        active_channels = event_status.get('active_channels', 0)
        registered_listeners = event_status.get('registered_listeners', 0)
        events_processed = event_status.get('events_processed_1min', 0)
        
        with col1:
            styled_metric_card("Active Channels", active_channels)
            
        with col2:
            styled_metric_card("Registered Listeners", registered_listeners)
            
        with col3:
            styled_metric_card("Events/min", events_processed)
            
        # Event queue status
        queues = event_status.get('queues', {})
        if queues:
            st.markdown("<h4 style='margin-top: 1rem; font-size: 1rem;'>Message Queues</h4>", unsafe_allow_html=True)
            
            queue_data = []
            for queue_name, queue_info in queues.items():
                queue_data.append({
                    "name": queue_name,
                    "size": queue_info.get('size', 0),
                    "processed": queue_info.get('processed_total', 0),
                    "consumers": queue_info.get('consumers', 0)
                })
                
            if queue_data:
                queue_df = pd.DataFrame(queue_data)
                st.dataframe(queue_df, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def trading_mode_card():
    """Display a card showing trading mode status"""
    from dashboard.api_utils import get_trading_modes
    
    trading_modes = get_trading_modes()
    
    with st.container():
        st.markdown("""
        <div style="background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <h3 style="margin-top: 0; font-size: 1.2rem;">Trading Modes</h3>
        """, unsafe_allow_html=True)
        
        if trading_modes:
            # Show active trading mode
            active_mode = next((m for m in trading_modes if m.get('active', False)), None)
            
            if active_mode:
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span style="font-weight: 500;">Active Mode: </span>
                    <span style="padding: 0.25rem 0.5rem; border-radius: 4px; background-color: {COLORS['success']}; color: white; font-size: 0.9rem;">
                        {active_mode.get('name', 'Unknown')}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Show mode details if available
                description = active_mode.get('description', '')
                if description:
                    st.markdown(f"<p style='margin-bottom: 1rem;'>{description}</p>", unsafe_allow_html=True)
                
                # Show mode parameters if available
                params = active_mode.get('parameters', {})
                if params:
                    param_items = []
                    for param_name, param_value in params.items():
                        param_items.append(f"<li><b>{param_name}:</b> {param_value}</li>")
                    
                    if param_items:
                        st.markdown(f"""
                        <p style='margin-bottom: 0.5rem;'><b>Parameters:</b></p>
                        <ul style='margin-top: 0;'>
                            {''.join(param_items)}
                        </ul>
                        """, unsafe_allow_html=True)
            
            # Show available trading modes for selection
            st.markdown("<h4 style='margin-top: 1rem; font-size: 1rem;'>Available Modes</h4>", unsafe_allow_html=True)
            
            mode_names = [mode.get('name', f"Mode {i}") for i, mode in enumerate(trading_modes)]
            selected_mode = st.selectbox("Select Trading Mode", mode_names)
            
            if st.button("Activate Mode"):
                st.toast(f"Activating {selected_mode} mode...", icon="🔄")
                # This would need to call a function to activate the mode
                # Since it's just UI for now, we'll show a success message
                st.success(f"{selected_mode} activated!")
        else:
            st.info("No trading modes available")
            
        st.markdown("</div>", unsafe_allow_html=True)
