import numpy as np
from sklearn.linear_model import Ridge


class RidgeGD:
    """Linear regression with an L2 (Ridge) penalty, trained by gradient descent.

    Cost = MSE + lam * sum(w**2)      (the bias b is NOT penalized)
    lam = 0 gives plain linear regression.
    """

    def __init__(self, lr=0.05, n_iters=5000, lam=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.lam = lam
        self.w = None
        self.b = 0.0

    def predict(self, X):
        return X @ self.w + self.b

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iters):
            error = self.predict(X) - y

            dw = (2 / n) * (X.T @ error) + 2 * self.lam * self.w
            db = (2 / n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ---- Data: 30 features, but only the first 3 really matter ----
np.random.seed(1)
n_train, n_test, d = 40, 200, 30
true_w = np.zeros(d)
true_w[:3] = [5, -3, 2]

def make_data(n):
    X = np.random.normal(0, 1, (n, d))
    y = X @ true_w + 10 + np.random.normal(0, 3, n)
    return X, y

X_train, y_train = make_data(n_train)
X_test, y_test = make_data(n_test)

# ---- Try different penalty strengths ----
print(f"{'lam':>6} | {'train MSE':>9} | {'test MSE':>8}")
print("-" * 32)
for lam in [0, 0.01, 0.1, 0.5, 1, 2, 5]:
    model = RidgeGD(lr=0.05, n_iters=5000, lam=lam).fit(X_train, y_train)
    tr = mse(y_train, model.predict(X_train))
    te = mse(y_test, model.predict(X_test))
    print(f"{lam:>6} | {tr:>9.2f} | {te:>8.2f}")

# ---- Check against scikit-learn ----
lam = 1
ours = RidgeGD(lr=0.05, n_iters=5000, lam=lam).fit(X_train, y_train)
sk = Ridge(alpha=lam * n_train).fit(X_train, y_train)
print("\nlam = 1 comparison (first 3 weights):")
print("Ours:        ", np.round(ours.w[:3], 4), round(ours.b, 4))
print("scikit-learn:", np.round(sk.coef_[:3], 4), round(sk.intercept_, 4))
print("True weights:", true_w[:3])