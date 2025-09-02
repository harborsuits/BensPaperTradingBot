"""
Broker Intelligence Configuration Panel

A Streamlit component that allows real-time modification of broker intelligence
parameters, weights, and thresholds. Integrates with the config_manager
to apply changes without requiring system restarts.
"""

import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load config from {config_path}: {str(e)}")
        return {}


def save_config(config_path: str, config: Dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        # Create backup of current config
        backup_path = f"{config_path}.bak"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                current_config = f.read()
            
            with open(backup_path, 'w') as f:
                f.write(current_config)
        
        # Write new config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save config: {str(e)}")
        return False


def render_broker_intelligence_config_panel(data_service: Any = None):
    """
    Render broker intelligence configuration panel
    
    Args:
        data_service: DataService instance for API calls (optional)
    """
    st.title("Broker Intelligence Configuration")
    
    # Default config path - can be overridden in settings
    default_config_path = "../../config/broker_intelligence_config.json"
    config_path = st.session_state.get("broker_intelligence_config_path", default_config_path)
    
    # File selector for config path
    with st.expander("Configuration Settings", expanded=False):
        new_config_path = st.text_input(
            "Configuration File Path", 
            value=config_path,
            help="Path to broker intelligence configuration file"
        )
        
        if new_config_path != config_path:
            st.session_state.broker_intelligence_config_path = new_config_path
            config_path = new_config_path
            st.experimental_rerun()
        
        # Load from API vs. file system
        use_api = st.checkbox(
            "Use API for configuration",
            value=st.session_state.get("use_broker_intelligence_api", False),
            help="When checked, loads and saves configuration via API instead of direct file access"
        )
        
        if use_api != st.session_state.get("use_broker_intelligence_api", False):
            st.session_state.use_broker_intelligence_api = use_api
    
    # Load configuration
    if use_api and data_service:
        # Use API to get config
        config = data_service.get_broker_intelligence_config()
        if not config:
            st.error("Failed to load configuration from API")
            return
    else:
        # Load from file
        config = load_config(config_path)
        if not config:
            st.warning(f"Could not load config from {config_path}. This may be a new configuration.")
            config = {
                "factor_weights": {
                    "latency": 0.25,
                    "reliability": 0.35,
                    "execution_quality": 0.25,
                    "cost": 0.15
                },
                "circuit_breaker_thresholds": {
                    "error_count": 5,
                    "error_rate": 0.3,
                    "availability_min": 90.0,
                    "reset_after_seconds": 300
                },
                "failover_threshold": 20.0,
                "health_threshold_normal": 80.0,
                "health_threshold_warning": 60.0
            }
    
    # Tabbed interface for different config sections
    tabs = st.tabs(["General Settings", "Factor Weights", "Circuit Breakers", "Asset Class Overrides", "A/B Testing"])
    
    # Track if config was modified
    config_modified = False
    
    # Tab 1: General Settings
    with tabs[0]:
        st.header("General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Failover threshold
            failover_threshold = st.slider(
                "Failover Threshold",
                min_value=5.0,
                max_value=50.0,
                value=float(config.get("failover_threshold", 20.0)),
                step=1.0,
                help="Score difference required to recommend broker failover"
            )
            
            if failover_threshold != config.get("failover_threshold"):
                config["failover_threshold"] = failover_threshold
                config_modified = True
        
        with col2:
            # Health thresholds
            health_normal = st.slider(
                "Normal Health Threshold",
                min_value=50.0,
                max_value=95.0,
                value=float(config.get("health_threshold_normal", 80.0)),
                step=1.0,
                help="Minimum score to consider broker health NORMAL"
            )
            
            health_warning = st.slider(
                "Warning Health Threshold",
                min_value=30.0,
                max_value=health_normal - 5.0,
                value=float(config.get("health_threshold_warning", 60.0)),
                step=1.0,
                help="Minimum score to consider broker health WARNING (below is CRITICAL)"
            )
            
            if health_normal != config.get("health_threshold_normal"):
                config["health_threshold_normal"] = health_normal
                config_modified = True
            
            if health_warning != config.get("health_threshold_warning"):
                config["health_threshold_warning"] = health_warning
                config_modified = True
    
    # Tab 2: Factor Weights
    with tabs[1]:
        st.header("Performance Factor Weights")
        st.info("These weights determine how different performance factors contribute to the overall broker score. Total must equal 1.0.")
        
        # Get current weights
        factor_weights = config.get("factor_weights", {
            "latency": 0.25,
            "reliability": 0.35,
            "execution_quality": 0.25,
            "cost": 0.15
        })
        
        # Create a form for factor weights
        with st.form("factor_weights_form"):
            # Dynamic slider for each factor
            new_weights = {}
            cols = st.columns(2)
            
            with cols[0]:
                new_weights["latency"] = st.slider(
                    "Latency Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(factor_weights.get("latency", 0.25)),
                    step=0.05,
                    help="Weight for latency performance factor"
                )
                
                new_weights["reliability"] = st.slider(
                    "Reliability Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(factor_weights.get("reliability", 0.35)),
                    step=0.05,
                    help="Weight for reliability performance factor"
                )
            
            with cols[1]:
                new_weights["execution_quality"] = st.slider(
                    "Execution Quality Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(factor_weights.get("execution_quality", 0.25)),
                    step=0.05,
                    help="Weight for execution quality performance factor"
                )
                
                new_weights["cost"] = st.slider(
                    "Cost Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(factor_weights.get("cost", 0.15)),
                    step=0.05,
                    help="Weight for cost performance factor"
                )
            
            # Calculate total
            total_weight = sum(new_weights.values())
            
            # Display total with warning if not 1.0
            st.metric(
                "Total Weight", 
                f"{total_weight:.2f}",
                delta=f"{total_weight - 1.0:.2f}" if total_weight != 1.0 else None
            )
            
            if total_weight != 1.0:
                st.warning(f"Total weight should equal 1.0, current total is {total_weight:.2f}")
            
            # Submit button
            submitted = st.form_submit_button("Update Factor Weights")
            
            if submitted:
                # If weights don't sum to 1.0, normalize them
                if total_weight != 1.0:
                    normalize = st.checkbox(
                        "Normalize weights to sum to 1.0?",
                        value=True
                    )
                    
                    if normalize:
                        # Normalize weights
                        for factor in new_weights:
                            new_weights[factor] = new_weights[factor] / total_weight
                    else:
                        st.error("Cannot update weights that don't sum to 1.0 without normalization")
                        return
                
                # Update config
                config["factor_weights"] = new_weights
                config_modified = True
                
                st.success("Factor weights updated")
    
    # Tab 3: Circuit Breakers
    with tabs[2]:
        st.header("Circuit Breaker Thresholds")
        st.info("These thresholds determine when circuit breakers are tripped due to broker failures or performance issues.")
        
        # Get current thresholds
        circuit_breaker_thresholds = config.get("circuit_breaker_thresholds", {
            "error_count": 5,
            "error_rate": 0.3,
            "availability_min": 90.0,
            "reset_after_seconds": 300
        })
        
        # Create a form for circuit breaker thresholds
        with st.form("circuit_breaker_form"):
            cols = st.columns(2)
            
            with cols[0]:
                error_count = st.number_input(
                    "Error Count Threshold",
                    min_value=1,
                    max_value=20,
                    value=int(circuit_breaker_thresholds.get("error_count", 5)),
                    help="Number of consecutive errors before tripping circuit breaker"
                )
                
                error_rate = st.slider(
                    "Error Rate Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(circuit_breaker_thresholds.get("error_rate", 0.3)),
                    step=0.05,
                    help="Error rate (0-1) that will trip circuit breaker"
                )
            
            with cols[1]:
                availability_min = st.slider(
                    "Minimum Availability (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=float(circuit_breaker_thresholds.get("availability_min", 90.0)),
                    step=1.0,
                    help="Minimum availability percentage before tripping circuit breaker"
                )
                
                reset_after_seconds = st.number_input(
                    "Reset After (seconds)",
                    min_value=30,
                    max_value=86400,  # 24 hours
                    value=int(circuit_breaker_thresholds.get("reset_after_seconds", 300)),
                    help="Time in seconds before automatically resetting circuit breaker"
                )
            
            # Submit button
            submitted = st.form_submit_button("Update Circuit Breaker Thresholds")
            
            if submitted:
                # Update config
                new_thresholds = {
                    "error_count": error_count,
                    "error_rate": error_rate,
                    "availability_min": availability_min,
                    "reset_after_seconds": reset_after_seconds
                }
                
                config["circuit_breaker_thresholds"] = new_thresholds
                config_modified = True
                
                st.success("Circuit breaker thresholds updated")
    
    # Tab 4: Asset Class Overrides
    with tabs[3]:
        st.header("Asset Class Specific Settings")
        st.info("Configure different factor weights for specific asset classes.")
        
        # Get current asset class settings
        asset_class_weights = config.get("asset_class_weights", {})
        
        # Predefined asset classes
        asset_classes = ["equities", "forex", "futures", "options", "crypto", "bonds"]
        
        # Asset class selector
        selected_asset_class = st.selectbox(
            "Select Asset Class",
            options=["New Asset Class"] + asset_classes + list(asset_class_weights.keys())
        )
        
        if selected_asset_class == "New Asset Class":
            # Allow user to enter a new asset class
            selected_asset_class = st.text_input("Asset Class Name")
        
        if selected_asset_class:
            # Get existing weights for this asset class or use defaults
            class_weights = asset_class_weights.get(selected_asset_class, {})
            if not class_weights:
                class_weights = config.get("factor_weights", {}).copy()
            
            # Create a form for asset class weights
            with st.form(f"asset_class_form_{selected_asset_class}"):
                st.subheader(f"Factor Weights for {selected_asset_class.capitalize()}")
                
                # Dynamic slider for each factor
                new_class_weights = {}
                cols = st.columns(2)
                
                with cols[0]:
                    new_class_weights["latency"] = st.slider(
                        f"Latency Weight ({selected_asset_class})",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(class_weights.get("latency", 0.25)),
                        step=0.05
                    )
                    
                    new_class_weights["reliability"] = st.slider(
                        f"Reliability Weight ({selected_asset_class})",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(class_weights.get("reliability", 0.35)),
                        step=0.05
                    )
                
                with cols[1]:
                    new_class_weights["execution_quality"] = st.slider(
                        f"Execution Quality Weight ({selected_asset_class})",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(class_weights.get("execution_quality", 0.25)),
                        step=0.05
                    )
                    
                    new_class_weights["cost"] = st.slider(
                        f"Cost Weight ({selected_asset_class})",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(class_weights.get("cost", 0.15)),
                        step=0.05
                    )
                
                # Calculate total
                total_weight = sum(new_class_weights.values())
                
                # Display total with warning if not 1.0
                st.metric(
                    "Total Weight", 
                    f"{total_weight:.2f}",
                    delta=f"{total_weight - 1.0:.2f}" if total_weight != 1.0 else None
                )
                
                if total_weight != 1.0:
                    st.warning(f"Total weight should equal 1.0, current total is {total_weight:.2f}")
                
                # Submit and delete buttons
                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("Update Asset Class Weights")
                
                with col2:
                    delete = st.form_submit_button("Delete Asset Class Override")
                
                if submitted:
                    # If weights don't sum to 1.0, normalize them
                    if total_weight != 1.0:
                        normalize = st.checkbox(
                            "Normalize asset class weights to sum to 1.0?",
                            value=True
                        )
                        
                        if normalize:
                            # Normalize weights
                            for factor in new_class_weights:
                                new_class_weights[factor] = new_class_weights[factor] / total_weight
                        else:
                            st.error("Cannot update weights that don't sum to 1.0 without normalization")
                            return
                    
                    # Update config
                    if "asset_class_weights" not in config:
                        config["asset_class_weights"] = {}
                    
                    config["asset_class_weights"][selected_asset_class] = new_class_weights
                    config_modified = True
                    
                    st.success(f"Asset class weights for {selected_asset_class} updated")
                
                if delete and selected_asset_class in config.get("asset_class_weights", {}):
                    # Remove asset class override
                    del config["asset_class_weights"][selected_asset_class]
                    config_modified = True
                    
                    st.success(f"Asset class override for {selected_asset_class} deleted")
                    time.sleep(1)  # Give time for message to be seen
                    st.experimental_rerun()
    
    # Tab 5: A/B Testing
    with tabs[4]:
        st.header("A/B Testing Configuration")
        st.info("Create and manage A/B test profiles for broker intelligence parameters.")
        
        # Get A/B test profiles
        ab_test_profiles = config.get("ab_test_profiles", {})
        
        # Enable A/B testing
        ab_testing_enabled = st.checkbox(
            "Enable A/B Testing",
            value=config.get("ab_testing_enabled", False),
            help="When enabled, randomly selects configuration profiles based on weights"
        )
        
        if ab_testing_enabled != config.get("ab_testing_enabled", False):
            config["ab_testing_enabled"] = ab_testing_enabled
            config_modified = True
        
        # Profile selector
        profile_options = list(ab_test_profiles.keys())
        selected_profile = st.selectbox(
            "Select Profile",
            options=["Create New Profile"] + profile_options,
            index=0
        )
        
        if selected_profile == "Create New Profile":
            # Allow user to create a new profile
            new_profile_name = st.text_input("New Profile Name")
            
            if new_profile_name:
                if st.button("Create Profile"):
                    if new_profile_name in ab_test_profiles:
                        st.error(f"Profile '{new_profile_name}' already exists")
                    else:
                        # Create new profile with current settings as base
                        new_profile = {
                            "description": f"A/B Test Profile: {new_profile_name}",
                            "weight": 1.0,
                            "active": True,
                            "config": {
                                "factor_weights": config.get("factor_weights", {}).copy(),
                                "circuit_breaker_thresholds": config.get("circuit_breaker_thresholds", {}).copy(),
                                "failover_threshold": config.get("failover_threshold", 20.0),
                                "health_threshold_normal": config.get("health_threshold_normal", 80.0),
                                "health_threshold_warning": config.get("health_threshold_warning", 60.0)
                            }
                        }
                        
                        # Add to config
                        if "ab_test_profiles" not in config:
                            config["ab_test_profiles"] = {}
                        
                        config["ab_test_profiles"][new_profile_name] = new_profile
                        config_modified = True
                        
                        st.success(f"Created new A/B test profile: {new_profile_name}")
                        time.sleep(1)  # Give time for message to be seen
                        st.experimental_rerun()
        
        elif selected_profile in ab_test_profiles:
            # Edit existing profile
            profile = ab_test_profiles[selected_profile]
            
            with st.form(f"profile_form_{selected_profile}"):
                st.subheader(f"Edit Profile: {selected_profile}")
                
                # Basic profile settings
                description = st.text_area(
                    "Description",
                    value=profile.get("description", ""),
                    help="Human-readable description of this profile"
                )
                
                weight = st.slider(
                    "Selection Weight",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(profile.get("weight", 1.0)),
                    step=0.1,
                    help="Higher weight means this profile is selected more often"
                )
                
                active = st.checkbox(
                    "Active",
                    value=profile.get("active", True),
                    help="When checked, this profile is included in A/B testing"
                )
                
                # Profile configuration (simplified)
                st.subheader("Profile Configuration")
                st.info("These settings will be used when this profile is selected")
                
                profile_config = profile.get("config", {})
                
                # Display key settings
                factor_weights = profile_config.get("factor_weights", config.get("factor_weights", {}))
                
                cols = st.columns(4)
                cols[0].metric("Latency Weight", f"{factor_weights.get('latency', 0.25):.2f}")
                cols[1].metric("Reliability Weight", f"{factor_weights.get('reliability', 0.35):.2f}")
                cols[2].metric("Execution Quality Weight", f"{factor_weights.get('execution_quality', 0.25):.2f}")
                cols[3].metric("Cost Weight", f"{factor_weights.get('cost', 0.15):.2f}")
                
                # Advanced configuration
                with st.expander("Advanced Configuration", expanded=False):
                    st.warning("Editing profile configuration directly is not supported in this interface. Use the file editor instead.")
                
                # Submit and delete buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    submitted = st.form_submit_button("Update Profile")
                
                with col2:
                    delete = st.form_submit_button("Delete Profile")
                
                with col3:
                    make_default = st.form_submit_button("Make Default")
                
                if submitted:
                    # Update profile
                    profile["description"] = description
                    profile["weight"] = weight
                    profile["active"] = active
                    
                    # Update config
                    config["ab_test_profiles"][selected_profile] = profile
                    config_modified = True
                    
                    st.success(f"Profile {selected_profile} updated")
                
                if delete:
                    # Remove profile
                    del config["ab_test_profiles"][selected_profile]
                    config_modified = True
                    
                    st.success(f"Profile {selected_profile} deleted")
                    time.sleep(1)  # Give time for message to be seen
                    st.experimental_rerun()
                
                if make_default:
                    # Make this profile the default by copying its config to main config
                    if "config" in profile:
                        # Copy top-level keys from profile config to main config
                        for key, value in profile["config"].items():
                            config[key] = value
                        
                        config_modified = True
                        st.success(f"Profile {selected_profile} applied as default configuration")
    
    # Save changes if modified
    if config_modified:
        if st.button("Save Configuration"):
            if use_api and data_service:
                # Save via API
                success = data_service.save_broker_intelligence_config(config)
                if success:
                    st.success("Configuration saved successfully via API")
                else:
                    st.error("Failed to save configuration via API")
            else:
                # Save to file
                if save_config(config_path, config):
                    st.success(f"Configuration saved to {config_path}")
                    
                    # Show notification about hot reload
                    st.info("Configuration changes will be automatically applied due to hot reload functionality.")


if __name__ == "__main__":
    # For local testing
    render_broker_intelligence_config_panel()
