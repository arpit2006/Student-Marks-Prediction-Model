# Numerical operations (arrays, math, vectorized computation)
import numpy as np
# Data manipulation and analysis (DataFrame, CSV handling, cleaning)
import pandas as pd
# Visualization (plots, graphs, model evaluation visuals)
import matplotlib.pyplot as plt
# Stratified train-test split (keeps category proportions same)
from sklearn.model_selection import StratifiedShuffleSplit
# Handling missing values (mean, median, most_frequent, constant)
from sklearn.impute import SimpleImputer
# Feature scaling (standardization: mean=0, std=1)
from sklearn.preprocessing import StandardScaler
# Creating ML pipelines (clean → scale → model)
from sklearn.pipeline import Pipeline
# Applying different preprocessing to different columns
from sklearn.compose import ColumnTransformer
# Linear regression model (baseline regression algorithm)
from sklearn.linear_model import LinearRegression
# Decision Tree regression model (non-linear, rule-based)
from sklearn.tree import DecisionTreeRegressor
# Random Forest regression (ensemble of decision trees)
from sklearn.ensemble import RandomForestRegressor
# Regression error metric (Root Mean Squared Error)
from sklearn.metrics import root_mean_squared_error
# Cross-validation for robust model evaluation
from sklearn.model_selection import cross_val_score
# R² score (how well the model explains variance)
from sklearn.metrics import r2_score
# Accuracy metric (⚠️ meant for classification, not regression)
from sklearn.metrics import accuracy_score

data = pd.read_csv("StudentMarksDataset.csv").drop(["Std_Name","Std_Branch","Std_Course","Std_RollNo"],axis = 1)

data.head()

data.tail()

data.describe()

#Plotting marks vs Study hours graph
plt.scatter(data['Std_StudyHours'],data['Std_Marks'])
plt.xlabel("Student Marks")
plt.ylabel("Study Hours")
plt.title("Plotting of Marks vs Study Hours")

#plotting Hitograph
data.hist(bins = 10,figsize=(10,3))

np.array(set(data['Std_StudyHours']))

#Stratiffied splitting
data['Study_cat'] = pd.cut(data['Std_StudyHours'],bins = [0,2,4,6,8,10],labels = [1,2,3,4,5])
data_create = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_data,test_data in data_create.split(data,data['Study_cat'].cat.add_categories(0).fillna(0)):
    train_set = data.loc[train_data].drop("Study_cat",axis = 1)
    test_set = data.loc[test_data].to_csv("test_data.csv",index=False)

#Copy
data_copy = train_set.copy()

data_predictions = train_set["Std_Marks"].copy()

num_attributes = data_copy.columns.tolist()

#Dropping the target values
data_copy = train_set.drop("Std_Marks",axis = 1)

#Creating pipeline
num_pipeline = Pipeline([
    ("numerica",SimpleImputer(strategy="median")),
    ("standardize",StandardScaler())
])

#Applying Column Transfer
full_pipline = ColumnTransformer([
    ("cat",num_pipeline,num_attributes)
])

final_data = full_pipline.fit_transform(data_copy)

#Creating Model
print("Linear Model")
lin_reg = LinearRegression()
lin_model = lin_reg.fit(final_data,data_predictions)
lin_predictions = lin_model.predict(final_data)
lin_rmse = root_mean_squared_error(data_predictions, lin_pred)
r2_percent = r2_score(data_predictions, lin_pred) * 100
print(lin_predictions)
print(lin_rmse)
print(r2_percent)

#random Forest
ran_reg = RandomForestRegressor()
ran_model = ran_reg.fit(final_data,data_predictions)
ran_predictions = ran_model.predict(final_data)
per_error = root_mean_squared_error(data_predictions,ran_predictions)
r2_accuracy = r2_score(data_predictions,ran_predictions) * 100
print(np.array(ran_predictions))
print(per_error)
print(r2_accuracy)

#Decision Tree
tree_reg = DecisionTreeRegressor()
tree_model = tree_reg.fit(final_data,data_predictions)
tree_predictions = tree_model.predict(final_data)
per_tree_error = root_mean_squared_error(data_predictions,tree_predictions)
r2_accuracy_tree = r2_score(data_predictions,tree_predictions) * 100
print(np.array(tree_predictions))
print(per_tree_error)
print(r2_accuracy_tree)

data_test = pd.read_csv("test_data.csv").drop("Study_cat",axis = 1)
predictions = full_pipline.fit_transform(data_test)
target_value = data_test["Std_Marks"].copy()
final_predictions = tree_model.predict(predictions)
pd.DataFrame(final_predictions,columns=["predicted_marks"]).to_csv("prediction.csv",index = False)
error = root_mean_squared_error(target_value,final_predictions)
r2_per = r2_score(target_value,final_predictions)