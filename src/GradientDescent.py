import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.005, max_iter=1e5, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def gradient_descent(self, gradient_fn, loss_fn, X, y, w=None):
        _, n_features = X.shape

        # Initialize weights if not provided
        if w is None:
            w = np.random.randn(n_features, 1) # Random Normal Distribution

        prev_loss = loss_fn(X, y, w)
        loss_history = []

        for i in range(int(self.max_iter)):
            gradient = gradient_fn(X, y, w) # Compute gradient
            w_new = w - self.learning_rate * gradient

            cur_loss = loss_fn(X, y, w_new) # Compute new loss
            loss_history.append(cur_loss)
            # Check for convergence
            if  abs(prev_loss - cur_loss) < self.epsilon:
                break

            prev_loss = cur_loss # Update loss value
            w = w_new   # Update weights

        return w, loss_history