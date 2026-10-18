import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressionGD:
    """Linear regression (many features) trained with gradient descent."""

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0
        self.cost_history = []

    def predict(self, X):
        return X @ self.w + self.b

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)          # one weight per feature
        self.b = 0.0
        self.cost_history = []

        for _ in range(self.n_iters):
            error = self.predict(X) - y

            dw = (2 / n) * (X.T @ error)
            db = (2 / n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.cost_history.append(np.mean(error ** 2))

        return self


# ---- Data: score depends on hours studied and attendance ----
np.random.seed(0)
n = 200
hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(50, 100, n)

X = np.column_stack([hours, attendance])      # shape (200, 2)
y = 4 * hours + 0.5 * attendance + 10 + np.random.normal(0, 3, n)
# true values: w1=4, w2=0.5, b=10

# ---- Attempt 1: NO scaling ----
print("--- Without scaling (lr=0.01) ---")
with np.errstate(all="ignore"):           # hides overflow warnings
    bad_model = LinearRegressionGD(lr=0.01, n_iters=100).fit(X, y)
print("Final cost:", bad_model.cost_history[-1])

# ---- Attempt 2: WITH scaling ----
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

model = LinearRegressionGD(lr=0.1, n_iters=1000).fit(X_scaled, y)
print("\n--- With scaling (lr=0.1) ---")
print("Final cost:", round(model.cost_history[-1], 3))

# ---- Convert weights back to the original units ----
w_original = model.w / std
b_original = model.b - np.sum(model.w * mean / std)

# ---- Compare with scikit-learn ----
sk_model = LinearRegression().fit(X, y)

print("\n--- Comparison (original units) ---")
print("Ours:        w =", np.round(w_original, 4), " b =", round(b_original, 4))
print("scikit-learn w =", np.round(sk_model.coef_, 4), " b =", round(sk_model.intercept_, 4))
print("True values: w = [4, 0.5], b = 10")