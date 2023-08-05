import numpy as np
from scipy.linalg import norm


class GK:
    def __init__(self, n_clusters=2, max_iter=100, m=2, max_error=1e-6):
        self.U = None # Membership matrix
        self.A = None # Covariance matrix
        self.F = None # Fuzzy Covariance matrix
        self.V = []   # Cetnroid initialization
        self.c = n_clusters # Number of clusters
        self.max_iter = max_iter # Max iterations
        self.m = m # Fuzzy coefficient
        self.z = 1 # Vector dimension
        self.max_error = max_error # Max error

    def fit(self, X):
        N = X.shape[0]
        self.z = X.shape[1]
        
        V = np.zeros((self.c, X.shape[1]))

        U = np.random.rand(self.c, N)
        U = U/np.sum(U, axis=1).reshape(-1,1) # Normalize
        A = None 

        iteration = 0
        while iteration < self.max_iter:
            U_prev = U.copy()

            V = self.next_centers(X, U)
            A = self._covariance(X, V, U)
            dist = self._distance(X, V, A)
            U = self.next_U(dist)
            iteration += 1

            # Stopping rule
            if norm(U - U_prev) < self.max_error:
                break

        self.V = V
        self.A = A
        self.U = U
        
        return 1

    def next_centers(self, X, U):
        Um = U ** self.m
        return ((Um @ X).T / Um.sum(axis=1)).T

    def _covariance(self, X, V, U):
        Um = U ** self.m
        denominator = Um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(X.reshape(X.shape[0], 1, -1) - V.reshape(1, V.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = Um.transpose().reshape(Um.shape[1], Um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)
        
        A = numerator / denominator

        return A

    def _distance(self, X, V, A):
        dif = np.expand_dims(X.reshape(X.shape[0], 1, -1) - V.reshape(1, V.shape[0], -1), axis=3)
        determ = np.power(np.linalg.det(A), 1 / self.z)
        self.F = determ.reshape(-1, 1, 1) * np.linalg.inv(A)
        temp = dif.transpose((0, 1, 3, 2)) @ self.F
        output = np.matmul(temp, dif).squeeze()
        return np.fmax(output, 1e-8)

    def next_U(self, d):
        w = float(2 / (self.m - 1))
        denominator_ = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        denominator_ = np.power(d[:, None, :] / denominator_.transpose((0, 2, 1)), w)
        denominator_ = 1 / denominator_.sum(1)

        return denominator_.T

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        dist = self._distance(X, self.V, self.A)
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=0)

        U = self.next_U(dist)
        return np.argmax(U, axis=0)