import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import joblib
import os

# ============================================================
# 📂 Ensure folders exist
# ============================================================
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ============================================================
# 📥 Load Dataset
# ============================================================
df = pd.read_excel("data/skynet_cleaned.xlsx")
print("✅ Data loaded successfully:", df.shape)

# ============================================================
# 🧹 Data Preparation (REGRESSION)
# ============================================================
df_reg = df[df['POD Name'].notna()].copy()
df_reg = df_reg[df_reg['Days to Delivered'] > 0]

# Fill missing numerical values
df_reg['Customs_Clearance_Duration'] = df_reg['Customs_Clearance_Duration'].fillna(df_reg['Customs_Clearance_Duration'].median())
df_reg['Hub_to_Delivery_Duration'] = df_reg['Hub_to_Delivery_Duration'].fillna(df_reg['Hub_to_Delivery_Duration'].median())
df_reg['Is_Weekend_Delivery'] = df_reg['Is_Weekend_Delivery'].fillna(0)

features_reg = [
    'Customs Value', 'Dead Weight', 'Customs_Clearance_Duration',
    'Hub_to_Delivery_Duration', 'Is_Urban_Region', 'Region_Delivery_Avg',
    'Is_Weekend_Delivery', 'Weight_Category', 'Receiver State',
    'CManifest_Day_of_Week', 'CManifest_Time_of_Day', 'POD_Day_of_Week',
    'POD_Time_of_Day'
]
target_reg = 'Days to Delivered'

X_reg = df_reg[features_reg].copy()
y_reg = df_reg[target_reg].copy()

# Label encode categorical columns
cat_cols = ['Weight_Category', 'Receiver State', 'CManifest_Day_of_Week',
            'CManifest_Time_of_Day', 'POD_Day_of_Week', 'POD_Time_of_Day']
encoder = LabelEncoder()
for col in cat_cols:
    X_reg[col] = encoder.fit_transform(X_reg[col].astype(str))

# ============================================================
# 🧠 Train Regression Model
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [3, 5, 10]
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_regressor = grid_search.best_estimator_

# Evaluate
y_pred = best_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n🌲 Regression Model Results")
print("R²:", round(r2, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# Save model
joblib.dump(best_regressor, "models/regression_model.pkl")
print("✅ Regression model saved successfully!")

# ============================================================
# 🧹 Data Preparation (CLASSIFICATION)
# ============================================================
df['Delivery_Status'] = df['POD Name'].notna().astype(int)

features_clf = [
    'Customs Value', 'Dead Weight', 'Is_Urban_Region', 'Region_Delivery_Avg',
    'Hub_to_Delivery_Duration', 'Customs_Clearance_Duration', 'Weight_Category'
]
target_clf = 'Delivery_Status'

df_clf = df.dropna(subset=features_clf + [target_clf]).copy()

# Label encode
label_cols = ['Weight_Category']
for col in label_cols:
    df_clf[col] = encoder.fit_transform(df_clf[col].astype(str))

X_clf = df_clf[features_clf]
y_clf = df_clf[target_clf]

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Train classifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42)
clf.fit(X_train_c, y_train_c)

# Evaluate
y_pred_c = clf.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
print("\n✅ Classification Accuracy:", round(acc, 4))
print("\n📋 Classification Report:\n", classification_report(y_test_c, y_pred_c))

# Save model
joblib.dump(clf, "models/classification_model.pkl")
print("✅ Classification model saved successfully!")
