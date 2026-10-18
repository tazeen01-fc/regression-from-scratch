import numpy as np
from sklearn.linear_model import Lasso


def soft_threshold(w, t):
    """Shrink every weight toward 0 by t; anything within t of 0 becomes exactly 0."""
    return np.sign(w) * np.maximum(np.abs(w) - t, 0)


class LassoGD:
    """Linear regression with an L1 (Lasso) penalty.

    Cost = MSE + lam * sum(|w|)      (the bias b is NOT penalized)
    """

    def __init__(self, lr=0.05, n_iters=5000, lam=0.1):
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

            dw = (2 / n) * (X.T @ error)
            db = (2 / n) * np.sum(error)

            # Step 1: normal gradient step
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Step 2: apply the L1 penalty by soft-thresholding
            self.w = soft_threshold(self.w, self.lr * self.lam)

        return self


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ---- Same data as the Ridge step ----
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
print(f"{'lam':>6} | {'train MSE':>9} | {'test MSE':>8} | {'non-zero weights':>16}")
print("-" * 50)
for lam in [0, 0.1, 0.5, 1, 2, 4]:
    model = LassoGD(lr=0.05, n_iters=5000, lam=lam).fit(X_train, y_train)
    tr = mse(y_train, model.predict(X_train))
    te = mse(y_test, model.predict(X_test))
    nz = np.sum(model.w != 0)
    print(f"{lam:>6} | {tr:>9.2f} | {te:>8.2f} | {nz:>16}")

# ---- Check against scikit-learn ----
lam = 1
ours = LassoGD(lr=0.05, n_iters=5000, lam=lam).fit(X_train, y_train)
sk = Lasso(alpha=lam / 2, max_iter=100000, tol=1e-10).fit(X_train, y_train)
print("\nlam = 1 comparison (first 3 weights):")
print("Ours:        ", np.round(ours.w[:3], 4), round(ours.b, 4))
print("scikit-learn:", np.round(sk.coef_[:3], 4), round(sk.intercept_, 4))
print("True weights:", true_w[:3])
print("\nOurs, all 30 weights (rounded):")
print(np.round(ours.w, 2))