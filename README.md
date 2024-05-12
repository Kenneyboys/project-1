# project-1
Essentials of Statistics and Math for Data Science
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Step 1: Load the data from the Excel file
df = pd.read_excel("Dataset_1.xlsx")

# Step 2: Perform exploratory data analysis (EDA)
print("Exploratory Data Analysis:")
print(df.head())
print(df.info())

# Step 3: Prepare the data
# Select relevant columns for analysis
selected_columns = ['PRICE', 'PRICE_SQFT', 'NO_OF_BATHROOMS', 'FLOOR_SIZE']
df = df[selected_columns]

# Step 4: Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X = df[['PRICE_SQFT', 'NO_OF_BATHROOMS', 'FLOOR_SIZE']]
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Fit a multiple regression model
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()

# Step 6: Evaluate the model's performance
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
