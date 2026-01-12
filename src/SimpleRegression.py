import numpy as np

class SimpleRegression:
    def __init__(self):
        self.coef = None

    def fit(self, X_train, y_train):
        '''
        Calculate the coefficient of the simple regression model.

        :param X_train: Standardized feature matrix
        :param y_train: Standardized target vector
        :return: Array of coefficients
        '''
        self.X = np.asarray(X_train)
        self.y = np.asarray(y_train)
        N = self.X.shape[0]

        self.coef = np.dot(self.X.T, self.y) / N

    def get_coef(self):
        # if self.coef.ndim > 1:
        #     return np.mean(np.abs(self.coef), axis=1)

        return self.coef