import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------
# ⚙️ PAGE TITLE
# ---------------------------------------------
st.header("🧠 Model Training & Evaluation — Regression & Classification")

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
        st.error("❌ Cleaned data file not found.")
        return None

df = load_cleaned_data()

if df is not None:
    task = st.radio("Select Modeling Task", ["Regression", "Classification"], index=0)

    # =====================================================================
    # 🔹 REGRESSION MODEL
    # =====================================================================
    if task == "Regression":
        st.subheader("📦 Predict Delivery Duration (Days to Delivered)")

        # target and base features (do NOT include Region_Delivery_Avg here to avoid leakage)
        target = 'Days to Delivered'
        features_base = [
            'Customs Value', 'Dead Weight', 'Customs_Clearance_Duration',
            'Is_Urban_Region', 'Is_Weekend_Delivery',
            'Weight_Category', 'Receiver State', 'Receiver Location Name',
            'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
            'POD_Day_of_Week', 'POD_Time_of_Day'
        ]

        cat_cols = [
            'Weight_Category', 'Receiver State', 'Receiver Location Name',
            'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
            'POD_Day_of_Week', 'POD_Time_of_Day'
        ]

        # quick column presence check
        missing_cols = [c for c in features_base + [target] if c not in df.columns]
        if missing_cols:
            st.error("⚠️ Some required columns are missing from your data: " + ", ".join(missing_cols))
        else:
            # Make a local copy and ensure target numeric
            df_local = df.copy()
            df_local[target] = pd.to_numeric(df_local[target], errors='coerce')

            # Drop rows where target is missing (can't train on those)
            df_local = df_local.dropna(subset=[target]).reset_index(drop=True)

            # ---------- Split BEFORE creating region-level aggregates ----------
            st.info("Splitting data (train/test) BEFORE computing Region_Delivery_Avg to avoid leakage.")
            test_size = st.slider("Select Test Data Size (%)", 10, 40, 20, 5, key="reg_split")

            # Prepare temporary table for splitting
            split_cols = features_base + [target]
            tmp = df_local[split_cols].copy()

            X_temp = tmp.drop(columns=[target])
            y_temp = tmp[target]

            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size / 100.0, random_state=42
            )

            # ---------- Compute Region_Delivery_Avg only on training set ----------
            region_avg_train = pd.concat([X_train_df, y_train], axis=1).groupby('Receiver State')[target].mean()
            # map to train/test; unseen states in test get global training mean
            global_region_mean = region_avg_train.mean()
            X_train_df['Region_Delivery_Avg'] = X_train_df['Receiver State'].map(region_avg_train)
            X_test_df['Region_Delivery_Avg'] = X_test_df['Receiver State'].map(region_avg_train).fillna(
                global_region_mean)

            # ---------- Impute numeric missing values using training stats ----------
            # Use training medians so no information leaks
            numeric_impute = {}
            if 'Customs_Clearance_Duration' in X_train_df.columns:
                numeric_impute['Customs_Clearance_Duration'] = X_train_df['Customs_Clearance_Duration'].median()
                X_train_df['Customs_Clearance_Duration'] = X_train_df['Customs_Clearance_Duration'].fillna(
                    numeric_impute['Customs_Clearance_Duration'])
                X_test_df['Customs_Clearance_Duration'] = X_test_df['Customs_Clearance_Duration'].fillna(
                    numeric_impute['Customs_Clearance_Duration'])

            if 'Is_Weekend_Delivery' in X_train_df.columns:
                X_train_df['Is_Weekend_Delivery'] = X_train_df['Is_Weekend_Delivery'].fillna(0)
                X_test_df['Is_Weekend_Delivery'] = X_test_df['Is_Weekend_Delivery'].fillna(0)

            # ---------- Final feature list (now includes Region_Delivery_Avg computed from train only) ----------
            features = features_base + ['Region_Delivery_Avg']

            # ---------- Encode categoricals: fit on TRAIN and transform both ----------
            encoders = {}
            for col in cat_cols + ['Region_Delivery_Avg']:  # region avg is numeric but safe if present in list
                if col in X_train_df.columns and X_train_df[col].dtype == object:
                    le = LabelEncoder()
                    # fit on training
                    X_train_df[col] = X_train_df[col].astype(str)
                    le.fit(X_train_df[col])
                    X_train_df[col] = le.transform(X_train_df[col])
                    # transform test; unseen -> map to a default (mode) then transform
                    X_test_df[col] = X_test_df[col].astype(str)
                    X_test_df[col] = X_test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X_test_df[col] = le.transform(X_test_df[col])
                    encoders[col] = le
                elif col in X_train_df.columns:
                    # numeric (skip)
                    continue

            # For any remaining object dtypes (like Receiver Location Name), encode similarly
            for col in X_train_df.select_dtypes(include=['object']).columns:
                if col not in encoders:
                    le = LabelEncoder()
                    X_train_df[col] = le.fit_transform(X_train_df[col].astype(str))
                    X_test_df[col] = le.transform(
                        X_test_df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0]))
                    encoders[col] = le

            # Ensure columns order and alignment
            X_train = X_train_df[features].copy()
            X_test = X_test_df[features].copy()

            # ---------- Model training with GridSearch ----------
            if st.button("Train Regression Model"):
                st.info("Training Decision Tree Regressor...")

                # set seeds for reproducibility
                np.random.seed(42)
                import random

                random.seed(42)

                param_grid = {
                    'max_depth': [4, 6, 8, 10],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [3, 5, 10]
                }
                cv = KFold(n_splits=5, shuffle=True, random_state=42)

                grid_search = GridSearchCV(
                    DecisionTreeRegressor(random_state=42),
                    param_grid,
                    cv=cv,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Evaluate (both train & test to check overfitting)
                y_pred_test = best_model.predict(X_test)
                y_pred_train = best_model.predict(X_train)

                st.metric("Test R²", f"{r2_score(y_test, y_pred_test):.4f}")
                st.metric("Test MAE", f"{mean_absolute_error(y_test, y_pred_test):.4f}")
                st.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

                st.metric("Train R²", f"{r2_score(y_train, y_pred_train):.4f}")

                # Save model & encoders and metadata (features)
                os.makedirs("models", exist_ok=True)
                model_info = {
                    "model": best_model,
                    "features": features,
                    "encoders": encoders,
                    "numeric_impute": numeric_impute
                }
                with open("models/regression_model.pkl", "wb") as f:
                    pickle.dump(model_info, f)

                st.success("💾 Regression model, encoders & metadata saved successfully in models/regression_model.pkl!")

                # ---------- Tree visualization ----------
                st.subheader("🧭 Decision Tree Structure (Regression Model)")
                from sklearn.tree import plot_tree
                import io

                fig3, ax3 = plt.subplots(figsize=(24, 12))
                plot_tree(
                    best_model,
                    feature_names=X_train.columns,
                    filled=True,
                    rounded=True,
                    fontsize=9,
                    ax=ax3
                )
                st.pyplot(fig3)

                # Save visualization as PNG for download
                buf = io.BytesIO()
                fig3.savefig(buf, format="png", bbox_inches="tight")
                st.download_button(
                    label="📥 Download Regression Tree as PNG",
                    data=buf.getvalue(),
                    file_name="decision_tree_regression.png",
                    mime="image/png"
                )

                # Insights
                with st.expander("📘 Model Insights", expanded=False):
                    st.markdown("**🧠 Structural Summary of the Decision Tree Regressor**")
                    st.write(f"• **Max Depth:** {best_model.get_depth()}")
                    st.write(f"• **Number of Leaves:** {best_model.get_n_leaves()}")
                    st.write(f"• **Number of Decision Nodes:** {best_model.tree_.node_count}")
                    st.write(f"• **Total Features Used:** {len(X_train.columns)}")

                    top_feature = X_train.columns[np.argmax(best_model.feature_importances_)]
                    st.write(f"• **Most Influential Feature:** `{top_feature}`")

                    interpretability = (
                        "✅ Highly Interpretable" if best_model.get_depth() <= 6 else "⚠️ Moderately Complex"
                    )
                    st.write(f"• **Model Complexity:** {interpretability}")

                    if best_model.get_depth() > 10:
                        st.warning("⚠️ The tree is very deep — potential overfitting risk.")
                    elif best_model.get_depth() <= 4:
                        st.info("ℹ️ The model is shallow and likely generalizes well.")
                    else:
                        st.success("✅ Balanced depth: interpretable and accurate.")


    # =====================================================================
    # 🔸 CLASSIFICATION MODEL (Updated Version)
    # =====================================================================
    elif task == "Classification":
        st.subheader("📦 Predict Delivery Success or Failure")

        target = 'Delivery_Status'
        features = [
            'Customs Value',
            'Region_Delivery_Avg',
            'Customs_Clearance_Duration',
            'CManifest_Day_of_Week',
            'CManifest_Time_of_Day'
        ]

        cat_cols = ['CManifest_Day_of_Week', 'CManifest_Time_of_Day']

        # Check for missing columns
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}")
        else:
            X = df[features].copy()
            y = df[target].copy()

            # Handle missing values
            numeric_impute = {}
            for col in ['Customs_Clearance_Duration', 'Customs Value', 'Region_Delivery_Avg']:
                median_val = X[col].median()
                numeric_impute[col] = median_val
                X[col] = X[col].fillna(median_val)

            X.dropna(inplace=True)
            y = y.loc[X.index]

            st.info(f"📊 Rows used for training: {X.shape[0]} out of {df.shape[0]}")

            # Encode categorical variables
            encoders = {}
            from sklearn.preprocessing import LabelEncoder

            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            # Encode target
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y.astype(str))
            encoders['target'] = y_encoder

            # Train-test split
            test_size = st.slider("Select Test Data Size (%)", 10, 40, 20, 5, key="class_split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=y
            )

            if st.button("Train Classification Model"):
                from sklearn.tree import DecisionTreeClassifier, plot_tree
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                import io

                st.info("Training Decision Tree Classifier... ⏳")

                # Base classifier
                clf = DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                )

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # ---------------------------------------------
                # 📈 PERFORMANCE
                # ---------------------------------------------
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                st.subheader("📊 Classification Model Performance")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{report['1']['precision']:.4f}")
                col3.metric("Recall", f"{report['1']['recall']:.4f}")

                # Confusion matrix
                st.subheader("🔲 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix — Delivery Status")
                st.pyplot(fig)

                # Feature importance
                st.subheader("🌳 Feature Importance")
                feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feat_imp, y=feat_imp.index, palette='mako', ax=ax2)
                ax2.set_title("Feature Importance in Delivery Success Prediction")
                st.pyplot(fig2)

                # ---------------------------------------------
                # 🌳 DECISION TREE VISUALIZATION
                # ---------------------------------------------
                st.subheader("🧭 Decision Tree Structure")
                st.caption("Visual representation of how the model splits features to predict delivery success.")

                fig3, ax3 = plt.subplots(figsize=(22, 12))
                plot_tree(
                    clf,
                    feature_names=X_train.columns,
                    class_names=y_encoder.classes_,
                    filled=True,
                    rounded=True,
                    fontsize=9,
                    ax=ax3
                )
                st.pyplot(fig3)

                # Save visualization as PNG
                buf = io.BytesIO()
                fig3.savefig(buf, format="png", bbox_inches="tight")
                st.download_button(
                    label="📥 Download Decision Tree as PNG",
                    data=buf.getvalue(),
                    file_name="decision_tree_delivery_status.png",
                    mime="image/png"
                )

                # ---------------------------------------------
                # 🧩 MODEL INSIGHTS
                # ---------------------------------------------
                with st.expander("📘 Model Insights", expanded=False):
                    st.markdown("**🧠 Structural Summary of the Decision Tree Model**")
                    st.write(f"• **Max Depth:** {clf.get_depth()}")
                    st.write(f"• **Leaves:** {clf.get_n_leaves()}")
                    st.write(f"• **Nodes:** {clf.tree_.node_count}")
                    st.write(f"• **Features Used:** {len(X_train.columns)}")
                    interpretability = "✅ Highly Interpretable" if clf.get_depth() <= 6 else "⚠️ Moderately Complex"
                    st.write(f"• **Model Complexity:** {interpretability}")

                    if clf.get_depth() > 8:
                        st.warning("⚠️ The tree is quite deep — may overfit.")
                    elif clf.get_depth() <= 4:
                        st.info("ℹ️ The model is shallow and likely generalizes well.")
                    else:
                        st.success("✅ Balanced model depth for interpretability and accuracy.")

                # ---------------------------------------------
                # 💾 SAVE MODEL + METADATA (Unified Format)
                # ---------------------------------------------
                os.makedirs("models", exist_ok=True)
                model_info = {
                    "model": clf,
                    "encoders": encoders,
                    "features": features,
                    "numeric_impute": numeric_impute
                }

                with open("models/classification_model.pkl", "wb") as f:
                    pickle.dump(model_info, f)

                st.success("💾 Model and metadata saved successfully in unified format!")
