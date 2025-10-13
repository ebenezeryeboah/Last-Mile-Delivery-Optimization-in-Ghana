import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------
# 🧼 PAGE TITLE
# ---------------------------------------------
st.header("🧹 Data Cleaning & Feature Engineering")

@st.cache_data
def load_data():
    df = pd.read_excel("data/data.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---- Step 1: Initial Overview ----
st.subheader("Initial Data Snapshot")
st.dataframe(df.head(10))

st.write("### Missing Values Before Cleaning")
missing = df.isnull().sum()
st.dataframe(missing[missing > 0].sort_values(ascending=False))

# ---------------------------------------------
# 🧮 Step 2: Convert Date Columns to Datetime
# ---------------------------------------------
st.write("### Converting Date Columns to Datetime")
date_columns = [
    'CManifest Date', 'Date Linehaul Arrival', 'Date Received from Airline by Customs Agent',
    'Date Customs Duties Paid', 'Date Released From Customs', 'Date Received from Customs Agent',
    'Arrived Hub Date', 'Date In Transit To Destination', 'First OFD Date', 'POD Date',
    'Date Returned to Sender - Delivery Attempts Exceeded'
]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

st.success("Date columns converted successfully!")

# ---------------------------------------------
# ⏱️ Step 3: Feature Engineering
# ---------------------------------------------
st.write("### Creating New Features")

# Helper function
def categorize_time_of_day(hour):
    if pd.isna(hour):
        return np.nan
    elif 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# Compute time-based durations
df['Customs_Clearance_Duration'] = (
    (df['Date Released From Customs'] - df['Date Received from Airline by Customs Agent']) / pd.Timedelta(days=1)
).round(2)

df['Hub_to_Delivery_Duration'] = (
    (df['POD Date'] - df['Arrived Hub Date']) / pd.Timedelta(days=1)
).round(2)

df['Transit_to_Delivery_Duration'] = (
    (df['POD Date'] - df['Date In Transit To Destination']) / pd.Timedelta(days=1)
).round(2)

# Define urban regions
urban_regions = ['Greater Accra', 'Central', 'Ashanti']
df['Is_Urban_Region'] = df['Receiver State'].isin(urban_regions).astype(int)

# Create new categorical/time features
df['CManifest_Day_of_Week'] = df['CManifest Date'].dt.day_name().where(df['CManifest Date'].notna())
df['POD_Day_of_Week'] = df['POD Date'].dt.day_name().where(df['POD Date'].notna())

df['CManifest_Time_of_Day'] = df['CManifest Date'].dt.hour.apply(categorize_time_of_day)
df['POD_Time_of_Day'] = df['POD Date'].dt.hour.apply(categorize_time_of_day)

# Weekend flag
df['Is_Weekend_Delivery'] = (df['POD Date'].dt.dayofweek >= 5).astype(int).where(df['POD Date'].notna())

# Weight categories
df['Weight_Category'] = pd.cut(
    df['Dead Weight'], bins=[0, 1000, 3000, float('inf')],
    labels=['Light', 'Medium', 'Heavy'], include_lowest=True
)

# Delivery status flag
df['Delivery_Status'] = df['POD Name'].notna().astype(int)

# ---------------------------------------------
# 🧾 Step 4: Region Average & Days to Delivered
# ---------------------------------------------
st.write("### Adding Aggregated and Numeric Features")

df['Days to Delivered'] = pd.to_numeric(df['Days to Delivered'], errors='coerce')
df['Region_Delivery_Avg'] = df.groupby('Receiver State')['Days to Delivered'].transform('mean').round(2)

# ---------------------------------------------
# ✅ Step 5: Results
# ---------------------------------------------
st.success("Feature engineering complete!")

st.write("### New Columns Created:")
engineered_cols = [
    'Customs_Clearance_Duration', 'Hub_to_Delivery_Duration', 'Transit_to_Delivery_Duration',
    'Is_Urban_Region', 'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
    'POD_Day_of_Week', 'POD_Time_of_Day', 'Is_Weekend_Delivery',
    'Weight_Category', 'Delivery_Status', 'Region_Delivery_Avg'
]
st.write(engineered_cols)

# Display the cleaned & engineered dataset
st.subheader("Cleaned & Engineered Dataset Preview")
st.dataframe(df.head(20))

# ---------------------------------------------
# 💾 Option to Download Cleaned Data
# ---------------------------------------------
st.write("### Download Cleaned Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Cleaned Dataset",
    data=csv,
    file_name="cleaned_data.csv",
    mime="text/csv"
)
