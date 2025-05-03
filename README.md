# 🔥 Calorie Predictor (Machine Learning)

This project is a machine learning application built with Python that predicts the number of calories burned based on user attributes like age, gender, height, weight, duration, heart rate, and body temperature. It uses a Random Forest Regressor to perform regression analysis and visualize feature importance and predictions.

---

## 📊 Dataset

The dataset used is `calories_data.csv`, which includes the following columns:

- Gender
- Age
- Height (cm)
- Weight (kg)
- Duration (min)
- Heart_Rate (bpm)
- Body_Temp (C)
- Calories (target variable)

---

## 🚀 Features

- Load and preprocess fitness dataset  
- Encode categorical variables (e.g., Gender)  
- Train/test split and data scaling  
- Model training using **Random Forest Regressor**  
- Evaluate model performance (MSE, R² score)  
- Visualize feature importance  
- Plot actual vs predicted calories  

---

## 📁 File Structure

calorie-predictor/

│

├── calories_data.csv # Input dataset

├── pythonlearn.py # Main Python script

├── README.md # Project documentation
