import numpy as np

# Same data as before
x = np.array([1, 2, 3, 4, 5])
y = np.array([35, 45, 55, 65, 75])


def predict(x, w, b):
    """Predict y using the line y = w*x + b."""
    return w * x + b


def mse(y_true, y_pred):
    """Mean Squared Error: the average of squared errors."""
    errors = y_true - y_pred
    return np.mean(errors ** 2)


# Try three different guesses
for w, b in [(10, 25), (8, 30), (0, 0)]:
    predictions = predict(x, w, b)
    cost = mse(y, predictions)
    print(f"w={w}, b={b} -> cost = {cost}")
    
print("\nTrying different slopes (b fixed at 25):")
for w in np.arange(6, 15, 2):
    cost = mse(y, predict(x, w, 25))
    print(f"w={w:>2} -> cost = {cost:.1f}")