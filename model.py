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

#Plotting marks vs Study hours graph to visualize the relationship between the two variables (how study hours affect marks)
plt.scatter(data['Std_StudyHours'],data['Std_Marks'])
plt.xlabel("Student Marks")
plt.ylabel("Study Hours")
plt.title("Plotting of Marks vs Study Hours")

#plotting Hitograph for all the features
data.hist(bins = 10,figsize=(10,3)) #Creating a histogram for each feature in the dataset to visualize the distribution of values (bins=10 means dividing the range of values into 10 intervals, figsize=(10,3) sets the size of the plot)

np.array(set(data['Std_StudyHours'])) #Finding the unique values in the 'Std_StudyHours' column to understand the range of study hours among students (using set to get unique values and converting it to a numpy array for easier manipulation)

#Stratiffied splitting
data['Study_cat'] = pd.cut(data['Std_StudyHours'],bins = [0,2,4,6,8,10],labels = [1,2,3,4,5])
#Creating an instance of StratifiedShuffleSplit (a method for splitting data into training and testing sets while preserving the distribution of a specified categorical variable, in this case, 'Study_cat')
data_create = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42) 
for train_data,test_data in data_create.split(data,data['Study_cat'].cat.add_categories(0).fillna(0)):
    train_set = data.loc[train_data].drop("Study_cat",axis = 1)
    test_set = data.loc[test_data].to_csv("test_data.csv",index=False)

#Copy
data_copy = train_set.copy() #Creating a copy of the training set to work with (preserving original data)
data_predictions = train_set["Std_Marks"].copy() #Target variable (what we want to predict)
#Dropping the target values
data_copy = train_set.drop("Std_Marks",axis = 1) #Features (what we will use to predict)
num_attributes = data_copy.columns.tolist() #List of feature column names (all columns except target)


#Creating pipeline
num_pipeline = Pipeline([
    ("numerica",SimpleImputer(strategy="median")), #fills na value with median value
    ("standardize",StandardScaler()) # standardizes the data to have mean 0 and std 1
])

#Applying Column Transfer
full_pipline = ColumnTransformer([
    ("cat",num_pipeline,num_attributes) #applies the num_pipeline to the columns specified in num_attributes (all features in this case)
])

final_data = full_pipline.fit_transform(data_copy)

#Creating Model
print("Linear Model")
lin_reg = LinearRegression() #Creating an instance of the Linear Regression model (a simple linear approach to regression tasks)
lin_model = lin_reg.fit(final_data,data_predictions)
lin_predictions = lin_model.predict(final_data)
lin_rmse = root_mean_squared_error(data_predictions, lin_predictions)
r2_percent = r2_score(data_predictions, lin_predictions) * 100
print(lin_predictions)
print(lin_rmse)
print(r2_percent)

#random Forest
ran_reg = RandomForestRegressor()
ran_model = ran_reg.fit(final_data,data_predictions) #Creating an instance of the Random Forest Regressor (an ensemble method that builds multiple decision trees and averages their predictions for regression tasks)
ran_predictions = ran_model.predict(final_data)
per_error = root_mean_squared_error(data_predictions,ran_predictions) #Calculating the Root Mean Squared Error (RMSE) between the actual target values and the predictions made by the Random Forest model (lower RMSE indicates better fit)
r2_accuracy = r2_score(data_predictions,ran_predictions) * 100 #Calculating the R² score (coefficient of determination) for the Random Forest model's predictions, multiplied by 100 to express it as a percentage (higher R² indicates better fit)
print(np.array(ran_predictions))
print(per_error)
print(r2_accuracy)

#Decision Tree
tree_reg = DecisionTreeRegressor() #Creating an instance of the Decision Tree Regressor (a non-linear model that splits the data into branches based on feature values to make predictions for regression tasks)
tree_model = tree_reg.fit(final_data,data_predictions) #Fitting the Decision Tree model to the training data (final_data contains the features and data_predictions contains the target variable)
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