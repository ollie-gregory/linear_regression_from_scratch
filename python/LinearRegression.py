import numpy as np

class LinearRegression():
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a bias (intercept) term by adding a column of ones to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate coefficients using the Normal Equation
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        return self
    
    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' with appropriate data before using 'predict'.")
        
        # Add a bias (intercept) term by adding a column of ones to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate predictions
        predictions = X_b.dot(self.coefficients)
        
        return predictions