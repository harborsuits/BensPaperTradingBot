"""
Manual Override Panel Component

This component provides emergency controls for the trading system.
"""
import streamlit as st
import time

def render_manual_override(api_service, account_type: str = None):
    """
    Render the manual override panel with emergency controls.
    
    Args:
        api_service: API service for system control actions
    """
    # Emergency stop button
    if st.button("EMERGENCY STOP", key="emergency_stop", use_container_width=True, 
                help="Immediately pause all strategies and stop all trading"):
        with st.spinner("Emergency stop in progress..."):
            # First pause all strategies
            pause_result = api_service.pause_all_strategies()
            
            # Then disable trading system-wide
            toggle_result = api_service.toggle_trading(enabled=False)
            
            combined_success = pause_result.get("success", False) and toggle_result.get("success", False)
            
            if combined_success:
                st.success("Emergency stop successful - All trading stopped")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Emergency stop failed - Check system status")
    
    # Close all positions button
    if st.button("Close All Positions", key="close_all", use_container_width=True,
                help="Close all open positions across all strategies"):
        # Add a confirmation dialog
        confirmation = st.checkbox("Confirm: Close ALL positions across all strategies?", key="confirm_close_all")
        
        if confirmation:
            with st.spinner("Closing all positions..."):
                result = api_service.close_all_positions()
                
                if result.get("success", False):
                    st.success(f"Closed {result.get('closed_count', 'all')} positions successfully")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed to close positions"))
    
    # Trading system toggle
    system_status = st.session_state.get("system_status", {})
    trading_enabled = system_status.get("trading_enabled", True)
    
    toggle_label = "Disable Trading" if trading_enabled else "Enable Trading"
    toggle_help = "Temporarily disable all trading" if trading_enabled else "Re-enable trading system"
    
    if st.button(toggle_label, key="toggle_trading", use_container_width=True, help=toggle_help):
        with st.spinner(f"{'Disabling' if trading_enabled else 'Enabling'} trading..."):
            result = api_service.toggle_trading(enabled=not trading_enabled)
            
            if result.get("success", False):
                new_status = "disabled" if trading_enabled else "enabled"
                st.success(f"Trading {new_status} successfully")
                time.sleep(1)
                st.rerun()
            else:
                st.error(result.get("error", f"Failed to {toggle_label.lower()}"))
