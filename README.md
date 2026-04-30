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
  - Decision Tree Regressor ✅ (Best Performing Model)  
  - Random Forest Regressor  
- Model evaluation using:
  - RMSE (Root Mean Squared Error)  
  - R² Score  
- Export predictions to CSV  

---

## 🏆 Best Model  
After evaluating all models, the **Decision Tree Regressor** performed the best based on:  
- Lowest RMSE  
- Highest R² Score  

This indicates that the model captures non-linear relationships in the data more effectively than other models.

---

## 📂 Project Structure  

```
student-marks-prediction/
│
├── notebook.ipynb           # EDA, preprocessing, training
├── model.py                 # Script for training & prediction
├── main.py                  # Streamlit UI app
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
- Streamlit  

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
---

```bash
# Clone the repository
- Regression modeling  

# Navigate into the project
cd student-marks-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run main.py

# (Optional) Run Jupyter Notebook for EDA and training
jupyter notebook
```

---

## 🖥️ Streamlit UI
This project now includes a **Streamlit web UI** for easy interaction and predictions. Launch it with:

```bash
streamlit run main.py
```

Open the provided local URL in your browser to use the app.
- Model selection and evaluation  
- Understanding non-linear relationships in data