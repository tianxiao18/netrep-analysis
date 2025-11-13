import numpy as np

class DualPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        if not (0 < self.n_components <= min(n_samples, n_features)):
            raise ValueError(f"n_components={self.n_components} must be between 1 and {min(n_samples, n_features)}")

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        G = X_centered @ X_centered.T
        eigvals, U = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals, U = eigvals[idx], U[:, idx]

        U_k = U[:, :self.n_components]
        S = np.sqrt(np.maximum(eigvals[:self.n_components], 0))

        V_k = X_centered.T @ U_k / S

        self.components_ = V_k.T
        self.singular_values_ = S
        self.explained_variance_ = eigvals[:self.n_components] / (X.shape[0] - 1)
        total_variance = np.sum(np.var(X, axis=0, ddof=1))
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


        