import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


## Funciton to load data
def load_data():

    '''This is the function to generate dummy data for regression and return those'''
    X, y = make_regression(n_samples=200, n_features=1, n_targets=1, noise=0.8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df = pd.DataFrame(data=X, columns=['X1'])
    df['target'] = y

    return X_train, y_train, X_test, y_test, df

