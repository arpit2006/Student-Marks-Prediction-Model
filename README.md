
# 🎓 Student Marks Prediction Model

## 📌 Overview
This project is an end-to-end machine learning regression model that predicts student marks based on study hours. It includes data preprocessing, stratified sampling, pipeline-based feature transformation, and regression models.

## 🧠 Features
- Data cleaning and exploratory analysis  
- Stratified train–test split  
- Preprocessing with `Pipeline` and `ColumnTransformer`  
- Models used:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
- Evaluation using **RMSE** and **R² score**  
- Predictions exported to CSV  

## 📂 Project Structure
student-marks-prediction/
│
├── notebook.ipynb          # Jupyter notebook with EDA, pipeline, and model training
├── model.py                # Python script for training and predicting with models
├── StudentMarksDataset.csv # Original dataset
├── test_data.csv           # Test set generated via stratified split
├── prediction.csv          # Final predictions on test set
├── README.md               # Project overview and instructions
└── LICENSE                 # MIT License file

## ⚙️ Technologies Used
- Python  
- NumPy, Pandas, Matplotlib  
- Scikit-learn  

## 📊 Model Evaluation
- Root Mean Squared Error (RMSE)  
- R² Score  

## 🚀 How to Run
1. Clone the repository  
2. Install dependencies (`pip install -r requirements.txt`)  
3. Run the notebook to train models and generate predictions  

## 📄 License
This project is licensed under the **MIT License**.

## 📌 Learning Outcome
This project demonstrates practical ML skills: preprocessing pipelines, stratified sampling, regression modeling, and evaluation metrics.