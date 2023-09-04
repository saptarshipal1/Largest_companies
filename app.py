import streamlit as st
import numpy as np
import pandas as pd
import openpyxl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor



st.title("Company_value_app")

st.write("""
This app predicts the Value of a company using the https://data.world/youngx62/worlds-largest-companies-by-revenue as the training dataset.
""")

st.sidebar.header('User Input Parameters')

df = pd.read_excel('Largest Companies in the World.xlsx')

st.subheader('User Input parameters. This is a small sample')
st.write(df.head())


    
df = df.copy()

# Drop unused columns
df = df.drop(['Global Rank', 'Country','Continent','Latitude','Longitude',], axis=1)

# One-hot encode Categorical feature columns
le = LabelEncoder()
df['Company'] = le.fit_transform(df['Company'])

# Split df into X and y
y = df['Market Value ($billion)']
X = df.drop('Market Value ($billion)', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

model = MLPRegressor()
model.fit(X_train,y_train)
pred = model.predict(X_test)
prediction_proba = model.score(X_test,y_test)
Value_comparison = pd.DataFrame({'Company': le.inverse_transform(X_test['Company']), 'Target Market Value ($billion)': y_test, 'Predicted Market Value ($billion)': pred})


st.subheader('Target data vs. Predicted Data')
st.write(Value_comparison.sort_index(ascending=True))

st.subheader('Prediction Probability')
st.write(prediction_proba)