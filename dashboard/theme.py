"""
Theme configuration for the BensBot Trading Dashboard
"""
import streamlit as st

# Color palette
COLORS = {
    "primary": "#2E7DFF",       # Main accent color
    "secondary": "#6C63FF",     # Secondary accent
    "success": "#28a745",       # Success states
    "warning": "#ffc107",       # Warning states
    "danger": "#dc3545",        # Error/danger states
    "info": "#17a2b8",          # Info states
    "light": "#f8f9fa",         # Light backgrounds
    "dark": "#1E293B",          # Dark text/elements (darker navy)
    "bg": "#0F172A",            # Background (dark navy)
    "text": "#E2E8F0",          # Main text (light gray)
    "card_bg": "#1E293B",       # Card background (darker navy)
    "card_text": "#F8FAFC"      # Card text (almost white)
}

def apply_custom_styling():
    """Apply custom styling to the Streamlit dashboard"""
    st.markdown("""
    <style>
    /* Overall App Styling */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }
    
    /* Card styling */
    .st-emotion-cache-1r6slb0 {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        color: #F8FAFC;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        text-align: center;
        color: #F8FAFC;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Success button */
    .success-btn {
        background-color: #28a745 !important;
        color: white !important;
    }
    
    /* Danger button */
    .danger-btn {
        background-color: #dc3545 !important;
        color: white !important;
    }
    
    /* Table styling */
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    
    /* General text and widgets */
    div.stText p {
        color: #E2E8F0 !important;
    }
    
    .stTextInput label, .stNumberInput label, .stDateInput label, .stSelectbox label {
        color: #E2E8F0 !important;
    }
    
    div.data_frame tbody {
        color: #E2E8F0 !important;
    }
    
    div.data_frame th {
        background-color: #334155 !important;
        color: #F8FAFC !important;
    }
    
    /* Plotly chart adjustments */
    .stPlotlyChart {
        background-color: #1E293B !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
    }
    
    .status-pending {
        background-color: #ffc107;
    }
    
    .status-failed {
        background-color: #dc3545;
    }
    
    .status-experimental {
        background-color: #6C63FF;
    }

    /* Dashboard header */
    .dashboard-header {
        padding: 0.5rem 0 1.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .dashboard-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        margin-top: 0;
    }
    .status-indicators {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .status-badge.success {
        background-color: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.4);
    }
    .status-badge.warning {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.4);
    }
    .status-badge.error {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid rgba(220, 53, 69, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)
