import numpy as np

# Our data: hours studied and exam scores
x = np.array([1, 2, 3, 4, 5])
y = np.array([35, 45, 55, 65, 75])

# A guess for w and b (we'll learn these properly later)
w = 10
b = 25

# Predict using the line formula: y = w*x + b
predictions = w * x + b

print("Actual scores:   ", y)
print("Predicted scores:", predictions)