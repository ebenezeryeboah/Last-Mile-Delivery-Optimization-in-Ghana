import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------
# 🎨 PAGE TITLE
# ---------------------------------------------
st.header("📈 Data Visualization: Delivery Insights")

# ---------------------------------------------
# 📂 LOAD CLEANED DATA
# ---------------------------------------------
@st.cache_data
def load_cleaned_data():
    try:
        df = pd.read_csv("cleaned_data.csv")
        df.columns = df.columns.str.strip()
        st.success("✅ Cleaned dataset loaded successfully!")
        return df
    except FileNotFoundError:
        st.error("❌ Cleaned data file not found. Please run 'Data Cleaning & Feature Engineering' first.")
        return None

df = load_cleaned_data()

if df is not None:

    # Sidebar visualization category
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Descriptive", "Relationships", "Correlation & Feature Importance"]
    )

    # ---------------------------------------------
    # 🩵 1️⃣ DESCRIPTIVE VISUALIZATIONS
    # ---------------------------------------------
    if viz_type == "Descriptive":
        st.subheader("📊 Descriptive Visualizations")

        chart_option = st.selectbox(
            "Choose a descriptive chart:",
            [
                "Delivery Duration Distribution",
                "Delivery Status Count",
                "Delivery Duration by Region",
                "Deliveries by Day of Week"
            ]
        )

        if chart_option == "Delivery Duration Distribution":
            st.write("Distribution of delivery times across all shipments.")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Days to Delivered'], bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_title("Distribution of Delivery Duration (Days)")
            ax.set_xlabel("Days to Delivered")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        elif chart_option == "Delivery Status Count":
            st.write("Proportion of deliveries that were successful or not.")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Delivery_Status', data=df, palette='viridis', ax=ax)
            ax.set_title("Delivery Status Count")
            ax.set_xlabel("Delivery Status (0 = Not Delivered, 1 = Delivered)")
            ax.set_ylabel("Number of Deliveries")
            st.pyplot(fig)

        elif chart_option == "Delivery Duration by Region":
            st.write("Comparison of delivery duration across different regions.")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Receiver State', y='Days to Delivered', data=df, palette='coolwarm', ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Delivery Duration by Region")
            st.pyplot(fig)

        elif chart_option == "Deliveries by Day of Week":
            st.write("Which days of the week see the most deliveries.")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='POD_Day_of_Week', data=df, order=[
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ], palette='crest', ax=ax)
            plt.xticks(rotation=30)
            ax.set_title("Deliveries by Day of the Week")
            st.pyplot(fig)

    # ---------------------------------------------
    # 🧡 2️⃣ RELATIONSHIP VISUALIZATIONS
    # ---------------------------------------------
    elif viz_type == "Relationships":
        st.subheader("🔗 Relationship Visualizations")

        chart_option = st.selectbox(
            "Choose a relationship chart:",
            [
                "Dead Weight vs. Delivery Duration (Scatter)",
                "Weight Category vs. Delivery Duration (Violin)",
                "Urban vs. Rural Delivery Duration (Bar)"
            ]
        )

        if chart_option == "Dead Weight vs. Delivery Duration (Scatter)":
            st.write("Relationship between parcel weight and delivery duration.")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x='Dead Weight', y='Days to Delivered', data=df, alpha=0.6, color='teal', ax=ax)
            ax.set_title("Dead Weight vs. Delivery Duration")
            st.pyplot(fig)

        elif chart_option == "Weight Category vs. Delivery Duration (Violin)":
            st.write("How delivery duration varies across weight categories.")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(x='Weight_Category', y='Days to Delivered', data=df, palette='mako', ax=ax)
            ax.set_title("Weight Category vs. Delivery Duration")
            st.pyplot(fig)

        elif chart_option == "Urban vs. Rural Delivery Duration (Bar)":
            st.write("Average delivery duration in urban vs rural regions.")
            avg_duration = df.groupby('Is_Urban_Region')['Days to Delivered'].mean().reset_index()
            avg_duration['Region Type'] = avg_duration['Is_Urban_Region'].map({1: 'Urban', 0: 'Rural'})
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Region Type', y='Days to Delivered', data=avg_duration, palette='coolwarm', ax=ax)
            ax.set_title("Urban vs. Rural Delivery Duration")
            st.pyplot(fig)


    # ---------------------------------------------
    # 💚 3️⃣ CORRELATION & FEATURE IMPORTANCE
    # ---------------------------------------------
    elif viz_type == "Correlation & Feature Importance":
        st.subheader("📊 Correlation and Feature Importance")

        chart_option = st.selectbox(
            "Choose a chart type:",
            [
                "Correlation Heatmap",
                "Correlation with Target Variable",
                "Regression Model Feature Importance",
                "Classification Model Feature Importance"
            ]
        )

        # ---- Correlation Heatmap ----
        if chart_option == "Correlation Heatmap":
            st.write("Shows correlations between all numerical features.")
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
            ax.set_title("Correlation Heatmap of Numerical Features")
            st.pyplot(fig)

        # ---- Correlation with Target ----
        elif chart_option == "Correlation with Target Variable":
            if 'Days to Delivered' in df.columns:
                numeric_df = df.select_dtypes(include=['int64', 'float64'])
                corr = numeric_df.corr()['Days to Delivered'].sort_values(ascending=False)
                st.write("Correlation of each feature with delivery duration:")
                fig, ax = plt.subplots(figsize=(8, 6))
                corr.drop('Days to Delivered').plot(kind='barh', color='teal', ax=ax)
                ax.set_title("Feature Correlation with Delivery Duration (Days to Delivered)")
                ax.set_xlabel("Correlation Coefficient")
                ax.set_ylabel("Feature")
                plt.gca().invert_yaxis()
                st.pyplot(fig)
            else:
                st.warning("'Days to Delivered' column not found in dataset.")

        # ---- Regression Model Feature Importance ----
        elif chart_option == "Regression Model Feature Importance":
            import pickle, os
            model_path = "models/regression_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                # Extract feature names (manually if needed)
                feature_names = [
                    'Customs Value', 'Dead Weight', 'Customs_Clearance_Duration',
                    'Hub_to_Delivery_Duration', 'Is_Urban_Region', 'Region_Delivery_Avg',
                    'Is_Weekend_Delivery', 'Weight_Category', 'Receiver State',
                    'Receiver Location Name', 'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
                    'POD_Day_of_Week', 'POD_Time_of_Day'
                ]
                importance = model.feature_importances_
                feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feat_imp, y=feat_imp.index, palette='viridis', ax=ax)
                ax.set_title("Feature Importance (Regression Model)")
                st.pyplot(fig)
            else:
                st.error("Regression model not found. Please train and save it first.")

        # ---- Classification Model Feature Importance ----
        elif chart_option == "Classification Model Feature Importance":
            import pickle, os
            model_path = "models/classification_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    clf = pickle.load(f)
                feature_names = [
                    'Customs Value', 'Dead Weight', 'Is_Urban_Region', 'Region_Delivery_Avg',
                    'Hub_to_Delivery_Duration', 'Customs_Clearance_Duration',
                    'CManifest_Day_of_Week', 'CManifest_Time_of_Day', 'Weight_Category'
                ]
                importance = clf.feature_importances_
                feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feat_imp, y=feat_imp.index, palette='mako', ax=ax)
                ax.set_title("Feature Importance (Classification Model)")
                st.pyplot(fig)
            else:
                st.error("Classification model not found. Please train and save it first.")
