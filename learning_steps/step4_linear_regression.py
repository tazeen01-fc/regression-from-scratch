import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    """Linear regression (one feature) trained with gradient descent."""

    def __init__(self, lr=0.01, n_iters=5000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = 0.0
        self.b = 0.0
        self.cost_history = []

    def predict(self, x):
        return self.w * x + self.b

    def fit(self, x, y):
        n = len(x)
        self.w, self.b = 0.0, 0.0
        self.cost_history = []

        for _ in range(self.n_iters):
            error = self.predict(x) - y

            dw = (2 / n) * np.sum(error * x)
            db = (2 / n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.cost_history.append(np.mean(error ** 2))

        return self


# ---- Messy, realistic data ----
np.random.seed(42)                       # makes results repeatable
x = np.random.uniform(0, 10, 100)        # 100 random x values between 0 and 10
noise = np.random.normal(0, 3, 100)      # random "real-world" scatter
y = 5 * x + 20 + noise                   # true w=5, true b=20, plus noise

# ---- Train ----
model = LinearRegressionGD(lr=0.01, n_iters=5000)
model.fit(x, y)

print(f"Learned: w={model.w:.3f}, b={model.b:.3f}")
print("True values we used to create the data: w=5, b=20")
print(f"Final cost: {model.cost_history[-1]:.3f}")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(x, y, alpha=0.6, label="data")
axes[0].plot(x, model.predict(x), color="red", label="our line")
axes[0].set_title("Data and fitted line")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()

axes[1].plot(model.cost_history)
axes[1].set_title("Cost going down")
axes[1].set_xlabel("iteration")
axes[1].set_ylabel("MSE")

plt.tight_layout()
plt.show()

from sklearn.linear_model import LinearRegression

sk_model = LinearRegression()
sk_model.fit(x.reshape(-1, 1), y)    # sklearn wants x as a 2D column

print("\n--- Comparison ---")
print(f"Ours:         w={model.w:.4f}, b={model.b:.4f}")
print(f"scikit-learn: w={sk_model.coef_[0]:.4f}, b={sk_model.intercept_:.4f}")