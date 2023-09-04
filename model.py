import numpy as np
import pandas as pd
import openpyxl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

df = pd.read_excel('Largest Companies in the World.xlsx')

def preprocess(df):
    
    df = df.copy()
    
    # Drop unused columns
    df = df.drop(['Global Rank', 'Company','Continent','Latitude','Longitude',], axis=1)
    
    # One-hot encode Categorical feature columns
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    
    # Split df into X and y
    y = df['Market Value ($billion)']
    X = df.drop('Market Value ($billion)', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess(df)



def model(X_train, X_test, y_train, y_test):
    
    model = GradientBoostingRegressor()
    model.fit(X_train,y_test)
    pred = model.predict(X_test)
    
    return pred
    