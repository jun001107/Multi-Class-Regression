from .GradientDescent import GradientDescent
import numpy as np

class MultiClassLinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias

    def loss_fn_sse(self, X, y, w):
        y_hat = X @ w  # continuous predictions
        return np.sum((y - y_hat) ** 2)

    def gradient_fn_sse(self, X, y, w):
        y_hat = X @ w  # continuous predictions
        return (2 / X.shape[0]) * X.T @ (y_hat - y)

    def fit(self, X, y, gradient_descent=GradientDescent().gradient_descent):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.add_bias:   # Add bias into X
            X = np.column_stack([np.ones(X.shape[0]), X])

        A = X.T @ X
        # y = y.reshape(-1, 1)

        if np.linalg.det(A) < 1e-12:   # Check if matrix A is invertible
            self.w = np.linalg.inv(A) @ X.T @ y
        else: # A is singular, use optimization function to fit.
            self.w, loss_history = gradient_descent(self.gradient_fn_sse, self.loss_fn_sse, X, y)
        return loss_history

    def predict(self, x):
        x = np.asarray(x)

        if self.add_bias: # Add a bias term
            x = np.column_stack([np.ones(x.shape[0]), x])
        y_hat = x @ self.w
        return y_hat

    def accuracy(self, y, y_hat, threshold=0.5):
        y_hat = np.array([1 if val > threshold else 0 for val in y_hat])
        y_hat = np.asarray(y_hat).flatten()
        y = np.asarray(y).flatten()

        if len(y_hat) != len(y):
            print("y_hat and y have different length")
            return

        return np.sum(y_hat == y) / len(y_hat)
