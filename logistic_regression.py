import numpy as np

class LogisticRegression():
    def __init__(self, fit_intercept = True, lr=1e-3, num_steps=1000, tol=1e-5):
        self.X = None
        self.Y = None
        self.beta = None
        self.num_steps = num_steps
        self.lr = lr
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        if X is None:
            intercept = np.ones((Y.shape[0], 1))
            self.X = np.hstack((intercept)).reshape(-1, 1)
        else:
            if self.fit_intercept:
                intercept = np.ones((X.shape[0], 1))
                self.X = np.hstack((intercept, X.copy()))
            else:
                intercept = np.zeros((X.shape[0], 1))
                self.X = np.hstack((X.copy(),))
        self.Y = Y.copy()
        self.beta = np.ones(self.X.shape[1])
        self.logistic_regression(self.X, self.Y)

    def sigmoid(self, X):
        z = X @ self.beta
        return 1 / (1 + np.exp(-z))

    def loss(self):
        h = self.sigmoid(self.X)
        cost = (((-self.Y).T @ np.log(h))-((1-self.Y).T @ np.log(1-h))).mean()
        return cost

    def logistic_regression(self, X, Y):
        loss_history = [0]
        for step in range(self.num_steps):
            predictions = self.sigmoid(self.X)
            self.beta += self.lr * np.dot(self.X.T, self.Y - predictions)
            loss_step = self.loss()
            if abs(loss_step - loss_history[step]) < self.tol:
                break
            loss_history.append(loss_step)
        return self.beta

    def predict(self, X):
        if isinstance(X, int):
            intercept = np.ones((X, 1))
            X_ = np.hstack((intercept)).reshape(-1, 1)
        else:
            if self.fit_intercept:
                X_ = X.copy()
                intercept = np.ones((X_.shape[0], 1))
                X_ = np.hstack((intercept, X_))
            else:
                X_ = X.copy()
                intercept = np.zeros((X_.shape[0], 1))
                X_ = np.hstack((X_,))

        return np.round(self.sigmoid(X_))

    def predict_proba(self, X):
        if isinstance(X, int):
            intercept = np.ones((X, 1))
            X_ = np.hstack((intercept)).reshape(-1, 1)
        else:
            if self.fit_intercept:
                X_ = X.copy()
                intercept = np.ones((X_.shape[0], 1))
                X_ = np.hstack((intercept, X_))
            else:
                X_ = X.copy()
                intercept = np.zeros((X_.shape[0], 1))
                X_ = np.hstack((X_,))

        return self.sigmoid(X_)
