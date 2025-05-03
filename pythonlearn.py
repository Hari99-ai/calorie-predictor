import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset with tab-separated values
try:
    data = pd.read_csv('calories_data.csv', sep='\t')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'calories_data.csv' file not found. Ensure the file is in the correct directory.")
    exit()

# Display dataset information
print("\nColumns in the dataset:", data.columns.tolist())
print(data.head())

# Check if the target column 'Calories' exists
if 'Calories' not in data.columns:
    print(f"Error: Target column 'Calories' not found. Available columns: {data.columns.tolist()}")
    exit()




# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values. Please clean the data before proceeding.")
    print(data.isnull().sum())
    exit()

# Encode categorical data (e.g., Gender)
if 'Gender' in data.columns:
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])  # M=1, F=0

# Features and target variable
X = data.drop(columns=['Calories'])
y = data['Calories']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Feature importance
feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # 45-degree line
plt.title("Actual vs Predicted Calories Burned")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
