"""Linear, Ridge and Lasso regression built from scratch with NumPy.

Also includes the small helpers every ML project needs: a train/test split,
a standard scaler and the common regression metrics.
"""

import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def train_test_split(X, y, test_size=0.2, seed=42):
    """Shuffle the rows, then split them into a train part and a test part."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Rescale each feature to mean 0 and standard deviation 1."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0      # avoid dividing by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """1.0 = perfect, 0.0 = no better than always guessing the average."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def soft_threshold(w, t):
    """Shrink every weight toward 0 by t; anything within t of 0 becomes 0."""
    return np.sign(w) * np.maximum(np.abs(w) - t, 0)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class LinearRegressionGD:
    """Plain linear regression trained with gradient descent."""

    def __init__(self, lr=0.05, n_iters=2000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0
        self.cost_history = []

    def predict(self, X):
        return X @ self.w + self.b

    # Hooks that Ridge and Lasso override -------------------------------
    def _penalty_gradient(self):
        return 0.0

    def _after_step(self):
        pass

    # -------------------------------------------------------------------
    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.cost_history = []

        for _ in range(self.n_iters):
            error = self.predict(X) - y

            dw = (2 / n) * (X.T @ error) + self._penalty_gradient()
            db = (2 / n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db
            self._after_step()

            self.cost_history.append(np.mean(error ** 2))

        return self


class RidgeGD(LinearRegressionGD):
    """Linear regression with an L2 penalty:  MSE + lam * sum(w**2)."""

    def __init__(self, lr=0.05, n_iters=2000, lam=1.0):
        super().__init__(lr, n_iters)
        self.lam = lam

    def _penalty_gradient(self):
        return 2 * self.lam * self.w


class LassoGD(LinearRegressionGD):
    """Linear regression with an L1 penalty:  MSE + lam * sum(|w|)."""

    def __init__(self, lr=0.05, n_iters=2000, lam=1.0):
        super().__init__(lr, n_iters)
        self.lam = lam

    def _after_step(self):
        self.w = soft_threshold(self.w, self.lr * self.lam)

# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------
def cross_val_rmse(make_model, X, y, k=5, seed=42):
    """Average validation RMSE over k folds.

    make_model: a function that returns a fresh, untrained model.
    The scaler is fitted on each training part only, so nothing leaks.
    """
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(X)), k)

    scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = make_model().fit(X_tr, y[train_idx])
        scores.append(rmse(y[val_idx], model.predict(X_val)))

    return np.mean(scores), np.std(scores)