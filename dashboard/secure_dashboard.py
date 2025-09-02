#!/usr/bin/env python3
"""
Secure Trading Dashboard

This application provides a Streamlit dashboard for managing broker credentials,
viewing audit logs, and monitoring trading system security.
"""

import streamlit as st
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import dashboard components
from dashboard.components.credential_management import render_credential_management
from dashboard.components.audit_log_viewer import render_audit_log_viewer

# Import trading bot modules
from trading_bot.brokers.auth_manager import initialize_auth_system, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("secure_dashboard")


def setup_page():
    """Set up the Streamlit page configuration"""
    st.set_page_config(
        page_title="Secure Trading Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .status-ok {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-warning {
        color: #FF9800;
        font-weight: bold;
    }
    .status-error {
        color: #F44336;
        font-weight: bold;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


def header_section():
    """Render the dashboard header section"""
    st.markdown('<h1 class="main-header">Secure Trading Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("## 🔒 Security and Compliance Center")
    
    # Show last update time
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def get_config_path():
    """Get the configuration file path"""
    # Check for custom config path in query parameters
    query_params = st.experimental_get_query_params()
    config_path = query_params.get("config", ["config/broker_config.json"])[0]
    
    # Validate the config path
    if not os.path.exists(config_path):
        # Try to find the config file in standard locations
        standard_paths = [
            "config/broker_config.json",
            "../config/broker_config.json",
            "broker_config.json"
        ]
        
        for path in standard_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    return config_path


def system_status_section(config_path):
    """Render the system status section"""
    st.markdown('<h2 class="sub-header">System Status</h2>', unsafe_allow_html=True)
    
    try:
        # Load configuration and initialize auth system
        config = load_config(config_path)
        credential_store, audit_log, audit_listener = initialize_auth_system(config)
        
        col1, col2, col3 = st.columns(3)
        
        # Credential Store Status
        with col1:
            st.markdown("### Credential Store")
            if credential_store:
                store_type = "Encrypted" if hasattr(credential_store, "encrypt") else "YAML"
                broker_count = len(credential_store.list_brokers())
                
                st.markdown(f"**Type:** {store_type}")
                st.markdown(f"**Brokers:** {broker_count}")
                st.markdown('<div class="status-ok">✓ OPERATIONAL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">✗ NOT AVAILABLE</div>', unsafe_allow_html=True)
        
        # Audit Log Status
        with col2:
            st.markdown("### Audit Log")
            if audit_log:
                log_type = "SQLite" if hasattr(audit_log, "conn") else "JSON"
                
                # Get count of recent events
                recent_count = 0
                try:
                    recent_events = audit_log.query_events(
                        start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    )
                    recent_count = len(recent_events)
                except Exception:
                    pass
                
                st.markdown(f"**Type:** {log_type}")
                st.markdown(f"**Today's Events:** {recent_count}")
                st.markdown('<div class="status-ok">✓ OPERATIONAL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">✗ NOT AVAILABLE</div>', unsafe_allow_html=True)
        
        # Event Listener Status
        with col3:
            st.markdown("### Event Listener")
            if audit_listener:
                st.markdown("**Status:** Connected to Event Bus")
                st.markdown('<div class="status-ok">✓ OPERATIONAL</div>', unsafe_allow_html=True)
            else:
                st.markdown("**Status:** Not Connected")
                if audit_log:
                    st.markdown('<div class="status-warning">⚠ MANUAL LOGGING ONLY</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">✗ NOT AVAILABLE</div>', unsafe_allow_html=True)
        
        return credential_store, audit_log, audit_listener
        
    except Exception as e:
        st.error(f"Error initializing system components: {str(e)}")
        logger.error(f"Error initializing system components: {str(e)}")
        return None, None, None


def main():
    """Main dashboard application"""
    # Setup page configuration
    setup_page()
    
    # Header section
    header_section()
    
    # Get configuration path
    config_path = get_config_path()
    
    # System status section
    credential_store, audit_log, audit_listener = system_status_section(config_path)
    
    # Main dashboard tabs
    tabs = st.tabs([
        "Credential Management",
        "Audit Log",
        "System Configuration"
    ])
    
    # Tab 1: Credential Management
    with tabs[0]:
        render_credential_management(config_path)
    
    # Tab 2: Audit Log
    with tabs[1]:
        render_audit_log_viewer(config_path)
    
    # Tab 3: System Configuration
    with tabs[2]:
        st.markdown('<h2 class="sub-header">System Configuration</h2>', unsafe_allow_html=True)
        
        try:
            # Load and display configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create a sanitized version of config (without sensitive data)
            sanitized_config = config.copy()
            
            # Remove sensitive fields
            if 'credential_store' in sanitized_config and 'master_password' in sanitized_config['credential_store']:
                sanitized_config['credential_store']['master_password'] = '********'
            
            if 'brokers' in sanitized_config:
                for broker_id, broker_config in sanitized_config['brokers'].items():
                    for key in ['api_key', 'api_secret', 'consumer_key', 'consumer_secret', 'access_token', 'refresh_token']:
                        if key in broker_config:
                            broker_config[key] = '********'
            
            # Display as JSON
            st.json(sanitized_config)
            
            # Configuration file info
            st.markdown(f"**Configuration File:** `{config_path}`")
            
            # Download configuration
            st.download_button(
                "Download Configuration",
                data=json.dumps(sanitized_config, indent=2),
                file_name="broker_config_export.json",
                mime="application/json"
            )
            
            # Upload new configuration
            st.markdown("### Upload New Configuration")
            st.warning("⚠️ Uploading a new configuration will replace your current settings.")
            
            uploaded_config = st.file_uploader("Upload configuration file", type=["json"])
            if uploaded_config is not None:
                try:
                    new_config = json.loads(uploaded_config.getvalue().decode())
                    
                    # Validate basic structure
                    if 'credential_store' not in new_config or 'audit_log' not in new_config:
                        st.error("Invalid configuration file format")
                    else:
                        if st.button("Apply New Configuration"):
                            with open(config_path, 'w') as f:
                                json.dump(new_config, f, indent=2)
                            st.success("Configuration updated successfully! Please refresh the page.")
                except Exception as e:
                    st.error(f"Error processing configuration file: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
    
    # Footer
    st.divider()
    st.markdown(
        "🔒 **Secure Trading System Dashboard** | "
        "Built with ❤️ using Streamlit | "
        f"Version 1.0.0 | {datetime.now().year}"
    )


if __name__ == "__main__":
    main()
