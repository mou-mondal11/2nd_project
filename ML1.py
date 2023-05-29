import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

st.title("MPG Prediction App")
dataset= sns.load_dataset('mpg')
st.subheader("Dataset")
Button = st.button('🔎')

if Button:
    st.write('Dataset')
    st.dataframe(dataset)

dataset=dataset.interpolate(method='pad')
x=dataset.iloc[:,1:5].values
y=dataset.iloc[:,0].values

regressor= RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x,y)
y_pred = regressor.predict(x)

print(y_pred)
x=regressor.predict([[303,130,3500,13]])
print(x)


dump(regressor,'mpg.joblib')


st.header("Enter Car Specifications")
cylinders = st.number_input("Cylinders", min_value=0, step=1, value=0)
displacement = st.number_input("Displacement", min_value=0.0, step=0.1, value=0.0)
horsepower = st.number_input("Horsepower", min_value=0.0, step=0.1, value=0.0)
weight = st.number_input("Weight", min_value=0, step=1, value=0)
input_data = [[cylinders, displacement, horsepower, weight]]
prediction = regressor.predict(input_data)

st.subheader("Prediction")
st.write("Predicted MPG:", prediction[0])



