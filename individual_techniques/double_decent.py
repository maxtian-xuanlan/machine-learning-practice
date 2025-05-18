from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# Generates training data for a cubic function with random noise drawn from a
# normal/Gaussian distribution scaled by epsilon with frequency delta

# Error size
epsilon = 1.0
# Error rate
delta = 0.1

# Generate equispaced points from -2 to 2 at 0.1 intervals
X = np.range(-2, 2.1, 0.1).reshape(-1, 1)
# Function: y = x^3 - 3x + epsilon * N(0, 1) * (N(0, 1) < delta)
y = (
    X ** 3
    - 3 * X
    + epsilon * np.random.rand(len(X), 1) * (np.random.rand(len(X), 1) < delta)
)

# Generate 1000 random points from -2 to 2
X_test = 4 * np.random.rand(1000, 1) - 1
y_test = (
    X_test ** 3
    - 3 * X_test
    + epsilon * np.random.rand(len(X_test), 1) * (np.random.rand(len(X_test), 1) < delta)
)

X_cub = np.linspace(-2, 2, 200)
plt.plot(X_cub, X_cub**3 - 3 * X_cub, "k-", label="Cubic Function", alpha=0.6)
plt.ylim(-4, 4)
plt.scatter(X, y, label="Cubic plus noise")
plt.title("Cubic Function with Noise")
plt.legend()
plt.show()

plt.scatter(X_test, y_test, label="Cubic plus noise", s=20)
plt.ylim(-4, 4)
plt.title("Cubic Function with Noise")
plt.legend()
plt.show()
