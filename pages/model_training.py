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

        target = 'Days to Delivered'
        features = [
            'Customs Value', 'Dead Weight', 'Customs_Clearance_Duration',
            'Is_Urban_Region', 'Region_Delivery_Avg', 'Is_Weekend_Delivery',
            'Weight_Category', 'Receiver State', 'Receiver Location Name',
            'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
            'POD_Day_of_Week', 'POD_Time_of_Day'
        ]

        cat_cols = [
            'Weight_Category', 'Receiver State', 'Receiver Location Name',
            'CManifest_Day_of_Week', 'CManifest_Time_of_Day',
            'POD_Day_of_Week', 'POD_Time_of_Day'
        ]

        if any(col not in df.columns for col in features):
            st.error("⚠️ Some required columns are missing from your data.")
        else:
            X = df[features].copy()
            y = pd.to_numeric(df[target], errors='coerce')

            # Fill missing values
            X['Customs_Clearance_Duration'] = X['Customs_Clearance_Duration'].fillna(X['Customs_Clearance_Duration'].median())
            X['Is_Weekend_Delivery'] = X['Is_Weekend_Delivery'].fillna(0)

            # Drop rows with missing values
            full_data = pd.concat([X, y], axis=1).dropna()
            X, y = full_data[features], full_data[target]

            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            test_size = st.slider("Select Test Data Size (%)", 10, 40, 20, 5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            if st.button("Train Regression Model"):
                st.info("Training Decision Tree Regressor...")

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

                # Evaluate
                y_pred = best_model.predict(X_test)
                st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

                # Save model & encoders
                os.makedirs("models", exist_ok=True)
                with open("models/regression_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                with open("models/regression_encoders.pkl", "wb") as f:
                    pickle.dump(encoders, f)
                st.success("💾 Regression model & encoders saved successfully!")

                # ---------------------------------------------
                # 🌳 DECISION TREE VISUALIZATION FOR REGRESSION
                # ---------------------------------------------
                st.subheader("🧭 Decision Tree Structure (Regression Model)")
                st.caption("Visual representation of how the model predicts delivery duration based on key variables.")

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

                # ---------------------------------------------
                # 🧩 MODEL INSIGHTS PANEL
                # ---------------------------------------------
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


    ## =====================================================================
# 🔸 CLASSIFICATION MODEL
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
            X['Customs_Clearance_Duration'] = X['Customs_Clearance_Duration'].fillna(X['Customs_Clearance_Duration'].median())
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

            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y.astype(str))
            encoders['target'] = y_encoder

            # Train-test split
            test_size = st.slider("Select Test Data Size (%)", 10, 40, 20, 5, key="class_split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=y
            )

            # Train model
            if st.button("Train Classification Model"):
                from sklearn.tree import DecisionTreeClassifier, plot_tree
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                import io

                st.info("Training Decision Tree Classifier... ⏳")

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
                # 📈 EVALUATION
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
                # 🌳 DECISION TREE VISUALIZATION (NEW)
                # ---------------------------------------------
                st.subheader("🧭 Decision Tree Structure")
                st.caption("Visual representation of how the model splits features to make delivery success predictions.")

                fig3, ax3 = plt.subplots(figsize=(22, 12))
                plot_tree(
                    clf,
                    feature_names=X_train.columns,
                    class_names=['Not Delivered', 'Delivered'],
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
                    label="📥 Download Decision Tree as PNG",
                    data=buf.getvalue(),
                    file_name="decision_tree_delivery_status.png",
                    mime="image/png"
                )

                # ---------------------------------------------
                # 🧩 MODEL INSIGHTS PANEL
                # ---------------------------------------------
                with st.expander("📘 Model Insights", expanded=False):
                    st.markdown("**🧠 Structural Summary of the Decision Tree Model**")
                    st.write(f"• **Max Depth:** {clf.get_depth()}")
                    st.write(f"• **Number of Leaves:** {clf.get_n_leaves()}")
                    st.write(f"• **Number of Decision Nodes:** {clf.tree_.node_count}")
                    st.write(f"• **Total Features Used:** {len(X_train.columns)}")

                    interpretability = (
                        "✅ Highly Interpretable" if clf.get_depth() <= 6 else "⚠️ Moderately Complex"
                    )
                    st.write(f"• **Model Complexity:** {interpretability}")

                    if clf.get_depth() > 8:
                        st.warning("⚠️ The tree is quite deep — may overfit.")
                    elif clf.get_depth() <= 4:
                        st.info("ℹ️ The model is shallow and likely generalizes well.")
                    else:
                        st.success("✅ Balanced model depth for interpretability and accuracy.")

                # ---------------------------------------------
                # 💾 SAVE MODEL & ENCODERS
                # ---------------------------------------------
                os.makedirs("models", exist_ok=True)
                with open("models/classification_model.pkl", "wb") as f:
                    pickle.dump(clf, f)
                with open("models/classification_encoders.pkl", "wb") as f:
                    pickle.dump(encoders, f)

                st.success("💾 Model and encoders saved successfully to 'models/'")
