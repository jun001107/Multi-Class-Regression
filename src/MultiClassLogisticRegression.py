from .GradientDescent import GradientDescent
import numpy as np

class MultiClassLogisticRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias

    def softmax(self, x):
        exp_z = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def loss_fn_cce(self, X, y, w):
        eps = 1e-20
        y_hat = self.softmax(X @ w)

        loss = -np.sum(y * np.log(y_hat + eps), axis=1)
        return np.mean(loss)

    def gradient_fn_cce(self, X, y, w):
        y_hat = self.softmax(X @ w)
        return (1 / X.shape[0]) * X.T @ (y_hat - y)

    def fit(self, X, y, gradient_descent=GradientDescent().gradient_descent):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.add_bias:   # Add bias into X
            X = np.column_stack([np.ones(X.shape[0]), X])

        self.w, loss_history = gradient_descent(self.gradient_fn_cce, self.loss_fn_cce, X, y)

        return loss_history

    def predict(self, x):
        x = np.asarray(x)

        if self.add_bias: # Add a bias term
            x = np.column_stack([np.ones(x.shape[0]), x])

        y_hat = self.softmax(x @ self.w)
        return y_hat

    def accuracy(self, y, y_hat, threshold=0.5):
        y_hat = np.array([1 if val > threshold else 0 for val in y_hat])
        y_hat = np.asarray(y_hat).flatten()
        y = np.asarray(y).flatten()

        if len(y_hat) != len(y):
            print("y_hat and y have different length")
            return

        return np.mean(y_hat == y)
