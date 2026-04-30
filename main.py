import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------- UI ----------------
st.title("🎓 Student Marks Predictor")
st.write("Predict marks based on study hours")

# User input
study_hours = st.slider("Select Study Hours", 0.0, 10.0, 5.0)

model_choice = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Decision Tree", "Random Forest"]
)

# ---------------- Load Data ----------------
data = pd.read_csv("StudentMarksDataset.csv").drop(
    ["Std_Name","Std_Branch","Std_Course","Std_RollNo"], axis=1
)

# ---------------- Stratified Split ----------------
data['Study_cat'] = pd.cut(
    data['Std_StudyHours'],
    bins=[0,2,4,6,8,10],
    labels=[1,2,3,4,5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(
    data, data['Study_cat'].cat.add_categories(0).fillna(0)
):
    train_set = data.loc[train_idx].drop("Study_cat", axis=1)

# ---------------- Preprocessing ----------------
X = train_set.drop("Std_Marks", axis=1)
y = train_set["Std_Marks"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, X.columns)
])

X_prepared = full_pipeline.fit_transform(X)


# ---------------- Train Models ----------------
lin_model = LinearRegression().fit(X_prepared, y)
tree_model = DecisionTreeRegressor().fit(X_prepared, y)
rf_model = RandomForestRegressor().fit(X_prepared, y)

# ---------------- Prediction ----------------
input_df = pd.DataFrame({
    "Std_StudyHours": [study_hours]
})

input_prepared = full_pipeline.transform(input_df)

if model_choice == "Linear Regression":
    prediction = lin_model.predict(input_prepared)

elif model_choice == "Decision Tree":
    prediction = tree_model.predict(input_prepared)

else:
    prediction = rf_model.predict(input_prepared)

# ---------------- Output ----------------
st.subheader("📊 Predicted Marks")
st.success(f"{prediction[0]:.2f}")