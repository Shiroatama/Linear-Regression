# lin_reg_scikit_sgd.py

# Import libraries necessary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def load(file_name, column):
    # Load data and return 2 columns, the first specified by the user and the second the Price column
    
    data = np.loadtxt(open(file_name, "rb"), dtype="float", delimiter=",")
    data = data[: , [column, 7] ]
    
    validate_data(data)

    return data

def validate_data(data):
    # Validates input data and checks for missing or invalid values.

    assert isinstance(data, np.ndarray), "Data is not a numpy array"
    assert not np.isnan(data).any(), "Data contains NaN values"

def doLinearRegression(column, column_name, alpha_value = None):
    # Performs linear regression on the data provided.
    # Assumes that the data is clean.
    # Parameters:
    #   - column : int - the column of the feature data 
    #   - column_name : string - used in labels when graphing the data
    #   - alpha_value : int - defaults to 0.01. If set to a value, uses that instead. 

    # Load the data
    data = load("realestate.csv", column)

    # Reshaping the data to fit the model since this is only a single feature
    X = data[:, 0].reshape(-1, 1) 
    y = data[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling is applied to make processing faster using SGD
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training is done using the SGDRegressor model.
    # Paremeters: 
    #   - learning_rate : string - set to constant but can also be change to 'optimal', 'invscaling', or 'adaptive'.
    #   - alpha_value : float - if no alpha value was given, the default is used.
    #   - max_iter : int - corresponds to the number of loops or iterations the model goes through.
    #   - tol : float - the stopping criterion. It prevents the code from running indefinitely.  
    model = SGDRegressor(learning_rate='constant', eta0=alpha_value if alpha_value else 0.01, max_iter=10000, tol=1e-3)
    model.fit(X_train_scaled, y_train)

    # Predictions are made on the test data which is then used to compare with the training data
    y_pred = model.predict(X_test_scaled)

    # Print the learned parameters as well as the Mean Squared Error of the model
    # It's worth mentioning that the default parameters or the theta values are both 0
    print(f"Learned parameters: {model.intercept_}, {model.coef_[0]}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    # Inverse transform the scaled features to make reading the data easier
    X_test_orig = scaler.inverse_transform(X_test_scaled)

    # Plots the models to a graph
    plt.scatter(X_test_orig, y_test, label=f"{column_name} vs Price")
    plt.plot(X_test_orig, model.predict(X_test_scaled), label="Learned Hypothesis", color="g")
    
    # Add labels to the axes
    plt.ylabel("Price")
    plt.xlabel(f"{column_name}")

    # Add legends and show the graph.
    plt.legend()
    plt.show()


def main():
    # Main block of code

    # Perform linear regression
    doLinearRegression(2, "Age of House", 0.002)
    # doLinearRegression(3, "Distance to Nearest MRT Station", 0.0000008)
    # doLinearRegression(4, "Number of Nearby Convenience store", 0.07)


if __name__ == "__main__":
    main()