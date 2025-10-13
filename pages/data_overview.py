import streamlit as st
import pandas as pd

# ---- Title ----
st.header("📊 Last Mile Delivery: Data Overview")

# ---- Load Data ----
@st.cache_data  # caches the dataset to avoid reloading every time
def load_data():
    df = pd.read_excel("data/data.xlsx")  # adjust path to your 'data' folder
    df.columns = df.columns.str.strip()   # remove leading/trailing spaces
    return df

df = load_data()

# ---- Define Sensitive and Important Columns ----
sensitive_cols = ['Account Code', 'Docket #', 'Connote #', 'Sender Ref 1', 'POD Name']  # hidden from user
important_cols = ['CManifest Date', 'POD Date',]  # shown first

# ---- Prepare Data for Display ----
other_cols = [col for col in df.columns if col not in sensitive_cols + important_cols]
df_display = df[important_cols + other_cols]

# ---- Display Raw Data ----
st.subheader("Raw Data (Sensitive Info Hidden)")
st.dataframe(df_display.head(50))

# ---- Dataset Info ----
st.subheader("Dataset Summary")
st.write(f"Number of Rows: {df.shape[0]}")
st.write(f"Number of Columns: {df.shape[1]}")
st.write("Column Names (Sensitive Hidden):")
st.write(list(df_display.columns))

# ---- Missing Values ----
st.subheader("Missing Values")
missing = df_display.isnull().sum()
st.dataframe(missing[missing > 0].sort_values(ascending=False))

# ---- Basic Statistics ----
st.subheader("Summary Statistics (Numeric Columns)")
st.dataframe(df_display.describe())

# ---- Column Selection for Preview ----
st.subheader("Preview Specific Columns")
available_cols = [col for col in df_display.columns if col not in sensitive_cols]  # exclude sensitive
cols = st.multiselect("Select columns to preview:", available_cols)
if cols:
    st.dataframe(df_display[cols].head(50))
