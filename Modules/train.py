from sklearn.linear_model import LinearRegression

## Function to train the model
def train_model(X_train, y_train):
    '''This is the function that just return the linear regression model'''

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model