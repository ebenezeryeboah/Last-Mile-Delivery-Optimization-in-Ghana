import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Delivery Analytics & Prediction App",
    page_icon="🚚",
    layout="wide"
)

# ------------------------------------------------
# 📘 Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Data Overview"]
)

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Ebenezer Yeboah**")

# ------------------------------------------------
# 🏠 HOME PAGE
# ------------------------------------------------
if page == "🏠 Home":
    st.title("🚚 Delivery Analytics & Prediction App")
    st.markdown("""
    Welcome to the **Delivery Analytics Platform**, a tool designed to help analyze and predict:
    - **Delivery Duration (Regression)**
    - **Delivery Success (Classification)**

    This app supports both **data exploration** and **machine learning predictions** using 
    real-world delivery datasets.

    **💡 What you can do here:**
    - Upload your delivery dataset (`.xlsx` format)
    - Explore delivery trends and performance metrics
    - Train ML models to predict delivery time or success
    - Input variables to get live predictions

    ---
    """)

    st.subheader("📦 Key Capabilities")
    st.markdown("""
    - Predict *Delivery Duration* (Days)  
    - Predict *Delivery Status* (Delivered / Not Delivered)  
    - Visualize *correlations, distributions, and trends*  
    - Works perfectly for Ghanaian delivery data 🇬🇭
    """)

    st.info("➡️ Use the **sidebar** to navigate to the next section.")

# ------------------------------------------------
# 📊 DATA OVERVIEW PAGE
# ------------------------------------------------
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")

    uploaded_file = st.file_uploader(
        "Upload your delivery dataset (.xlsx file):", type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Data uploaded successfully!")
    else:
        st.info("Using sample dataset (skynet.xlsx)...")
        df = pd.read_excel("skynet.xlsx")  # Make sure this file is in the repo

    # Display DataFrame preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10))

    # Basic info
    st.markdown("### 🧮 Dataset Summary")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.markdown("### ⚠️ Missing Values Summary")
        st.dataframe(missing)
    else:
        st.success("No missing values detected ✅")

    # Quick statistics
    st.markdown("### 📈 Summary Statistics")
    st.dataframe(df.describe().T)

    # Visualization section
    st.markdown("### 📊 Sample Distribution Plot")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        feature = st.selectbox("Select a numerical column to visualize:", num_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[feature], kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numerical columns found to plot.")

