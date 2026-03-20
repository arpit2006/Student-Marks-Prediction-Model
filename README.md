# 🎓 Student Marks Prediction Model  

## 📌 Overview  
This project is an **end-to-end machine learning regression model** that predicts student marks based on study hours. It covers the full ML workflow including preprocessing, stratified sampling, pipelines, and multiple regression models.

---

## 🧠 Features  
- Data cleaning and exploratory data analysis (EDA)  
- Stratified train–test split  
- Preprocessing using **Pipeline** and **ColumnTransformer**  
- Multiple regression models:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
- Model evaluation using:
  - RMSE (Root Mean Squared Error)  
  - R² Score  
- Export predictions to CSV  

---

## 📂 Project Structure  
```
student-marks-prediction/
│
├── notebook.ipynb           # EDA, preprocessing, training
├── model.py                 # Script for training & prediction
├── StudentMarksDataset.csv  # Original dataset
├── test_data.csv            # Test dataset (stratified split)
├── prediction.csv           # Model predictions
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

---

## ⚙️ Technologies Used  
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 📊 Model Evaluation  
- **RMSE (Root Mean Squared Error)** → Measures prediction error  
- **R² Score** → Measures model accuracy  

---

## 🚀 How to Run  

```bash
# Clone the repository
git clone https://github.com/your-username/student-marks-prediction.git

# Navigate into the project
cd student-marks-prediction

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook
```

---

## 📄 License  
This project is licensed under the **MIT License**.

---

## 📌 Learning Outcome  
This project demonstrates practical machine learning skills:
- Data preprocessing pipelines  
- Stratified sampling  
- Regression modeling  
- Model evaluation techniques