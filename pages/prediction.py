# ==========================================
# 🚚 Last Mile Delivery Prediction Page (Final Version)
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
st.title("🚀 Last Mile Delivery Prediction Dashboard")
st.markdown(
    """
    Use trained machine learning models to predict:
    - **📦 Delivery Duration (Regression)**
    - **✅ Delivery Success or Failure (Classification)**
    """
)

# ---------------------------------------------
# 1️⃣ SELECT MODEL TYPE
# ---------------------------------------------
model_type = st.radio(
    "Select Prediction Type:",
    ["Regression (Delivery Duration)", "Classification (Delivery Success)"]
)

# ---------------------------------------------
# 2️⃣ DEFINE MODEL & ENCODER PATHS + FEATURES
# ---------------------------------------------
if model_type == "Regression (Delivery Duration)":
    model_path = "models/regression_model.pkl"
    encoders_path = "models/regression_encoders.pkl"

    st.info("🧠 Model: Predict **delivery duration (in days)**.")
    features = [
        'Customs Value', 'Dead Weight', 'Customs_Clearance_Duration',
        'Is_Urban_Region', 'Region_Delivery_Avg', 'Is_Weekend_Delivery',
        'Weight_Category', 'Receiver State', 'Receiver Location Name',
        'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
        'POD_Day_of_Week', 'POD_Time_of_Day'
    ]

else:
    model_path = "models/classification_model.pkl"
    encoders_path = "models/classification_encoders.pkl"

    st.info("🧠 Model: Predict whether a delivery will be **Successful** or **Unsuccessful**.")
    features = [
        'Customs Value', 'Region_Delivery_Avg', 'Customs_Clearance_Duration',
        'CManifest_Day_of_Week', 'CManifest_Time_of_Day'
    ]

# ---------------------------------------------
# 3️⃣ LOAD MODEL & ENCODERS
# ---------------------------------------------
if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please train your model in the 'Model Training' page first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)
st.success(f"✅ {model_type.split('(')[0].strip()} model loaded successfully!")

# Load encoders if available
encoders = {}
if os.path.exists(encoders_path):
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    st.info("🔤 Encoders loaded successfully (for categorical features).")
else:
    st.warning("⚠️ Encoder file not found — unseen categories will be auto-encoded (may slightly affect accuracy).")

# ---------------------------------------------
# 4️⃣ USER INPUT SECTION
# ---------------------------------------------
st.subheader("📋 Enter Delivery Details")

if model_type == "Regression (Delivery Duration)":
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
        receiver_state = st.text_input("Receiver State")
        receiver_location = st.text_input("Receiver Location Name")
        cmanifest_day = st.selectbox("Manifest Day of Week",
                                     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        cmanifest_time = st.selectbox("Manifest Time of Day", ["Morning", "Afternoon", "Evening"])
        pod_day = st.selectbox("POD Day of Week",
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        pod_time = st.selectbox("POD Time of Day", ["Morning", "Afternoon", "Evening"])

    is_urban_encoded = 1 if is_urban == "Yes" else 0
    is_weekend_encoded = 1 if is_weekend == "Yes" else 0

    input_dict = {
        'Customs Value': [customs_value],
        'Dead Weight': [dead_weight],
        'Customs_Clearance_Duration': [clearance_duration],
        'Is_Urban_Region': [is_urban_encoded],
        'Region_Delivery_Avg': [region_avg],
        'Is_Weekend_Delivery': [is_weekend_encoded],
        'Weight_Category': [weight_category],
        'Receiver State': [receiver_state],
        'Receiver Location Name': [receiver_location],
        'CManifest_Day_of_Week': [cmanifest_day],
        'CManifest_Time_of_Day': [cmanifest_time],
        'POD_Day_of_Week': [pod_day],
        'POD_Time_of_Day': [pod_time]
    }

else:
    col1, col2 = st.columns(2)
    with col1:
        customs_value = st.number_input("Customs Value (GHS)", min_value=0.0, step=1.0)
        clearance_duration = st.number_input("Customs Clearance Duration (days)", min_value=0.0, step=0.1)
    with col2:
        region_avg = st.number_input("Region Delivery Average (days)", min_value=0.0, step=0.1)
        cmanifest_day = st.selectbox("Manifest Day of Week",
                                     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        cmanifest_time = st.selectbox("Manifest Time of Day", ["Afternoon", "Night"])

    input_dict = {
        'Customs Value': [customs_value],
        'Region_Delivery_Avg': [region_avg],
        'Customs_Clearance_Duration': [clearance_duration],
        'CManifest_Day_of_Week': [cmanifest_day],
        'CManifest_Time_of_Day': [cmanifest_time]
    }

# ---------------------------------------------
# 5️⃣ ENCODE INPUT DATA SAFELY
# ---------------------------------------------
input_data = pd.DataFrame(input_dict)

if encoders:
    for col, encoder in encoders.items():
        if col in input_data.columns:
            # Replace unseen labels with the first known class
            input_data[col] = input_data[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            input_data[col] = encoder.transform(input_data[col].astype(str))
else:
    # Fallback encoding for object-type columns
    for col in input_data.select_dtypes(include='object').columns:
        input_data[col] = pd.factorize(input_data[col])[0]

# ---------------------------------------------
# 6️⃣ PREDICT BUTTON
# ---------------------------------------------
st.divider()
if st.button("🔮 Predict"):
    try:
        # Reorder columns to match training
        input_data = input_data[features]

        # Perform prediction
        prediction = model.predict(input_data)[0]

        if model_type == "Regression (Delivery Duration)":
            st.success(f"🕒 **Predicted Delivery Duration:** {prediction:.2f} days")
        else:
            result = "✅ Successful Delivery" if prediction == 1 else "❌ Unsuccessful Delivery"
            st.success(result)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
                st.info(f"Confidence: **{prob * 100:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ---------------------------------------------
# 7️⃣ FOOTNOTE
# ---------------------------------------------
st.caption(
    "💡 *Tip: Ensure that text inputs (like Receiver State or Location) match known categories from training. "
    "If new categories are entered, the app automatically replaces them with default values to prevent errors.*"
)
