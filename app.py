import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("data/housing.csv")

x = data[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = data["Price"]

# Train-test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Use Linear Regression model for prediction
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)   
r2 = r2_score(y_test, predictions)

print("Predictions:", predictions[:5])


print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
age = st.number_input("Age")

if st.button("Predict"):
    result = model.predict([[area, bedrooms, bathrooms, age]])
    st.success(f"Predicted Price: {result[0]}")

