# ==========================================
# 🚚 Last Mile Delivery Analytics — Main App
# ==========================================

import streamlit as st
import importlib

# ---------------------------------------------
# ⚙️ PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="Last Mile Delivery Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# 🏷️ APP HEADER
# ---------------------------------------------
st.title("📦 Last Mile Delivery Analytics Dashboard")
st.markdown(
    """
    Welcome to your **Last Mile Delivery Analytics App**.
    Use the sidebar to navigate between pages for:
    - **Data Overview** (View and clean raw data)
    - **Data Visualization** (Explore trends and patterns)
    - **Model Training** (Train regression & classification models)
    - **Prediction** (Make single or batch predictions)
    """
)

# ---------------------------------------------
# 📝 FOOTER
# ---------------------------------------------
st.sidebar.divider()
st.sidebar.caption("Developed by **Ebenezer Yeboah** — Last Mile Delivery Optimization Project")
