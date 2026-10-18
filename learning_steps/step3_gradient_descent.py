import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([35, 45, 55, 65, 75], dtype=float)


def predict(x, w, b):
    return w * x + b


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, lr=0.05, n_iters=1000):
    """Learn w and b by repeatedly stepping downhill on the cost."""
    n = len(x)
    w, b = 0.0, 0.0          # start with a blind guess
    cost_history = []

    for i in range(n_iters):
        y_pred = predict(x, w, b)
        error = y_pred - y

        # Slopes of the cost with respect to w and b
        dw = (2 / n) * np.sum(error * x)
        db = (2 / n) * np.sum(error)

        # Step downhill
        w = w - lr * dw
        b = b - lr * db

        cost_history.append(mse(y, y_pred))

        if i % 100 == 0:
            print(f"iter {i:>4} | w={w:.3f}, b={b:.3f}, cost={cost_history[-1]:.3f}")

    return w, b, cost_history


w, b, history = gradient_descent(x, y)
print(f"\nLearned: w={w:.3f}, b={b:.3f}")
print("Target:  w=10, b=25")