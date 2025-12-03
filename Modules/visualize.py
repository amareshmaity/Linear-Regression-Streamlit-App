import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

## function to create visualization
def visualize_lr_model(X_train, y_train, y_train_pred):
    '''Function to draw (X,y) and predicted line'''
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(x=X_train.reshape(-1), y=y_train, color='blue', s=100, label='Actual Data Points')
    ax.plot(X_train.reshape(-1), y_train_pred, color='red', linewidth=2, label='Regression Line')

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("X vs y")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)