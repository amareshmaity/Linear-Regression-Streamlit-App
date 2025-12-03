import numpy as np
import streamlit as st
from Modules.load_data import load_data
from Modules.train import train_model
from Modules.visualize import visualize_lr_model

with st.sidebar:
    st.subheader("Make Prediction")
    x1 = st.slider(label="X1", min_value=-3.0, max_value=3.0)

st.title("Linear Regression APP")

## Load the data
X_train, y_train, X_test, y_test, df = load_data()

## Train the model
model = train_model(X_train, y_train)
print(X_test[0], y_test[0])

## Prediction
y_train_pred = model.predict(X_train)
y_pred = model.predict(np.array([[x1]]))

## Visualization
visualize_lr_model(X_train, y_train, y_train_pred)



st.write("Predicted Value", y_pred)