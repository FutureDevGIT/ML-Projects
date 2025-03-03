# **Machine Learning Projects**

This repository showcases a collection of machine learning projects that demonstrate proficiency and passion for the field. Each project addresses a unique problem statement, utilizing various datasets and machine learning techniques.

---

## **1. Calories Burnt Prediction**

### **Overview**
This project aims to predict the number of calories burnt during physical activities based on individual physiological and activity-related parameters.

### **Dataset**
- **Features:** Age, Gender, Height, Weight, Heart Rate, Duration of Exercise, etc.
- **Target Variable:** Calories Burnt
- **Source:** Publicly available fitness datasets.

### **Approach**
1. **Data Preprocessing:**
   - Handling missing values.
   - Encoding categorical variables (e.g., Gender).
   - Feature scaling to normalize data.

2. **Modeling:**
   - Implemented regression models such as Linear Regression, Random Forest Regressor, and XGBoost Regressor.
   - Performed hyperparameter tuning to optimize model performance.

3. **Evaluation:**
   - Used metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared to evaluate model accuracy.

---

## **2. Diabetes Prediction**

### **Overview**
This project focuses on predicting the likelihood of an individual having diabetes based on specific health indicators.

### **Dataset**
- **Features:** Number of Pregnancies, Glucose Level, Blood Pressure, Skin Thickness, Insulin Level, BMI, Diabetes Pedigree Function, Age.
- **Target Variable:** Diabetes Outcome (1 - Diabetic, 0 - Non-Diabetic)
- **Source:** Pima Indians Diabetes Dataset from the UCI Machine Learning Repository.

### **Approach**
1. **Data Preprocessing:**
   - Addressed missing or zero values in the dataset.
   - Applied feature scaling to standardize data.

2. **Modeling:**
   - Utilized classification algorithms including Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM).
   - Conducted cross-validation and hyperparameter tuning for model optimization.

3. **Evaluation:**
   - Assessed models using accuracy, precision, recall, F1-score, and ROC-AUC metrics.

---

## **3. Gold Price Prediction**

### **Overview**
This project aims to forecast gold prices based on historical data and various economic indicators.

### **Dataset**
- **Features:** Date, Gold Price, Oil Price, Stock Indices, Currency Exchange Rates, etc.
- **Target Variable:** Future Gold Price
- **Source:** Financial datasets from sources like Yahoo Finance and other financial databases.

### **Approach**
1. **Data Preprocessing:**
   - Managed missing values and outliers.
   - Created new features such as moving averages and price differentials.
   - Applied time series decomposition to understand trend and seasonality.

2. **Modeling:**
   - Implemented time series forecasting models like ARIMA, Prophet, and LSTM neural networks.
   - Compared performance of different models to select the best predictor.

3. **Evaluation:**
   - Evaluated models using metrics such as Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).

---

## **4. SONAR Rock vs. Mine Prediction**

### **Overview**
This project involves classifying objects as either rocks or mines based on sonar signal returns, which is crucial for submarine and mine detection operations.

### **Dataset**
- **Features:** 60 sonar frequency readings per observation.
- **Target Variable:** Object Type (R - Rock, M - Mine)
- **Source:** SONAR Dataset from the UCI Machine Learning Repository.

### **Approach**
1. **Data Preprocessing:**
   - Converted categorical labels into numerical format.
   - Applied feature scaling to ensure uniformity across features.

2. **Modeling:**
   - Employed classification algorithms such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Neural Networks.
   - Performed feature selection to identify the most significant frequencies.

3. **Evaluation:**
   - Used confusion matrices, accuracy scores, and F1-scores to evaluate model performance.

---

## **Getting Started**

### **Prerequisites**
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, tensorflow (for LSTM), statsmodels (for ARIMA), fbprophet (for Prophet)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/FutureDevGIT/ML-Projects.git
