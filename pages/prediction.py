# ==========================================
# 🚚 Last Mile Delivery Prediction Page — Ghana-Ready Version
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------------------------
# ⚙️ PAGE TITLE & INTRO
# ---------------------------------------------
st.set_page_config(page_title="Last Mile Delivery Prediction", layout="wide")
st.title("🚀 Last-Mile Delivery Prediction Dashboard (Ghana Context)")
st.markdown("""
Use trained ML models to predict:
- **📦 Delivery Duration (Regression)**
- **✅ Delivery Success or Failure (Classification)**
""")

# ---------------------------------------------
# 🇬🇭 OFFICIAL GHANA REGIONS
# ---------------------------------------------
ghana_regions = [
    "Greater Accra", "Ashanti", "Eastern", "Western", "Western North",
    "Central", "Volta", "Oti", "Northern", "Savannah", "North East",
    "Upper East", "Upper West", "Bono", "Bono East", "Ahafo"
]

# ---------------------------------------------
# 📂 LOAD CLEANED DATA TO POPULATE LOCATIONS
# ---------------------------------------------
@st.cache_data
def load_cleaned_data():
    try:
        df = pd.read_csv("cleaned_data.csv")
        df.columns = df.columns.str.strip()
        locations = sorted(df["Receiver Location Name"].dropna().unique().tolist())
        return locations
    except Exception as e:
        st.warning(f"⚠️ Couldn't load locations: {e}")
        return []

receiver_locations = load_cleaned_data()

# ---------------------------------------------
# 1️⃣ SELECT MODEL TYPE
# ---------------------------------------------
model_type = st.radio(
    "Select Prediction Type:",
    ["Regression (Delivery Duration)", "Classification (Delivery Success)"]
)

# ---------------------------------------------
# 2️⃣ LOAD MODEL INFO
# ---------------------------------------------
model_path = (
    "models/regression_model.pkl"
    if "Regression" in model_type
    else "models/classification_model.pkl"
)
if not os.path.exists(model_path):
    st.error("❌ Model file not found! Train it first in the 'Model Training' page.")
    st.stop()

with open(model_path, "rb") as f:
    model_info = pickle.load(f)

if isinstance(model_info, dict) and "model" in model_info:
    model = model_info["model"]
    encoders = model_info.get("encoders", {})
    features = model_info.get("features", [])
    numeric_impute = model_info.get("numeric_impute", {})
else:  # backward compatibility
    model, encoders, features, numeric_impute = model_info, {}, [], {}

st.success(f"✅ {model_type.split('(')[0].strip()} model loaded successfully!")
st.info(f"📊 Model expects {len(features)} features.")

# ---------------------------------------------
# 3️⃣ USER INPUT SECTION
# ---------------------------------------------
st.subheader("📋 Enter Delivery Details")

if "Regression" in model_type:
    col1, col2 = st.columns(2)
    with col1:
        customs_value = st.number_input("Customs Value (GHS)", min_value=0.0, step=1.0)
        dead_weight = st.number_input("Dead Weight (grams)", min_value=0.0, step=100.0)
        clearance_duration = st.number_input("Customs Clearance Duration (days)", min_value=0.0, step=0.1)
        region_avg = st.number_input("Region Delivery Average (days)", min_value=0.0, step=0.1)
        is_urban = st.selectbox("Is Urban Region?", ["Yes", "No"])
    with col2:
        is_weekend = st.selectbox("Is Weekend Delivery?", ["No", "Yes"])
        weight_category = st.selectbox("Weight Category", ["Light", "Medium", "Heavy"])
        receiver_region = st.selectbox("Receiver Region (Ghana)", ghana_regions)
        receiver_location = st.selectbox(
            "Receiver Location Name", receiver_locations if receiver_locations else ["(no data)"]
        )
        cmanifest_day = st.selectbox(
            "Manifest Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        cmanifest_time = st.selectbox("Manifest Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        pod_day = st.selectbox(
            "POD Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        pod_time = st.selectbox("POD Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    is_urban_encoded = 1 if is_urban == "Yes" else 0
    is_weekend_encoded = 1 if is_weekend == "Yes" else 0

    input_dict = {
        "Customs Value": [customs_value],
        "Dead Weight": [dead_weight],
        "Customs_Clearance_Duration": [clearance_duration],
        "Is_Urban_Region": [is_urban_encoded],
        "Region_Delivery_Avg": [region_avg],
        "Is_Weekend_Delivery": [is_weekend_encoded],
        "Weight_Category": [weight_category],
        "Receiver Region": [receiver_region],
        "Receiver Location Name": [receiver_location],
        "CManifest_Day_of_Week": [cmanifest_day],
        "CManifest_Time_of_Day": [cmanifest_time],
        "POD_Day_of_Week": [pod_day],
        "POD_Time_of_Day": [pod_time],
    }
else:
    col1, col2 = st.columns(2)
    with col1:
        customs_value = st.number_input("Customs Value (GHS)", min_value=0.0, step=1.0)
        clearance_duration = st.number_input("Customs Clearance Duration (days)", min_value=0.0, step=0.1)
    with col2:
        region_avg = st.number_input("Region Delivery Average (days)", min_value=0.0, step=0.1)
        cmanifest_day = st.selectbox(
            "Manifest Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        cmanifest_time = st.selectbox("Manifest Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    input_dict = {
        "Customs Value": [customs_value],
        "Region_Delivery_Avg": [region_avg],
        "Customs_Clearance_Duration": [clearance_duration],
        "CManifest_Day_of_Week": [cmanifest_day],
        "CManifest_Time_of_Day": [cmanifest_time],
    }

input_data = pd.DataFrame(input_dict)

# ---------------------------------------------
# 4️⃣ SAFE ENCODING & IMPUTATION
# ---------------------------------------------
def safe_encode(col, encoder, val):
    if encoder is None:
        return val
    if val not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, val)
    return encoder.transform([val])[0]

if encoders:
    for col, enc in encoders.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].apply(lambda x: safe_encode(col, enc, str(x)))
else:
    for col in input_data.select_dtypes(include="object").columns:
        input_data[col] = pd.factorize(input_data[col])[0]

for col, val in numeric_impute.items():
    if col in input_data.columns and (input_data[col].isna().any() or input_data[col].iloc[0] == 0):
        input_data[col] = val

if features:
    for f in features:
        if f not in input_data.columns:
            input_data[f] = 0
    input_data = input_data[features]

# ---------------------------------------------
# 5️⃣ PREDICT
# ---------------------------------------------
st.divider()
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_data)[0]

        if "Regression" in model_type:
            st.success(f"🕒 **Predicted Delivery Duration:** {prediction:.2f} days")
        else:
            result = "✅ Successful Delivery" if prediction == 1 else "❌ Unsuccessful Delivery"
            st.success(result)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
                st.info(f"Confidence: **{prob*100:.2f}%**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------------------------------------
# 6️⃣ FOOTNOTE
# ---------------------------------------------
st.caption(
    "💡 *Receiver Region restricted to Ghana’s 16 official regions. "
    "Receiver Location comes from your dataset to prevent invalid entries.*"
)
