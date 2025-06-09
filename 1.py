import numpy as np

# Sample input (x), output (y), and initial weights (w)
x = np.array([1, -1, 0, 0.5])   # input vector
y = 1                           # target output
w = np.array([0.2, -0.1, 0.0, 0.1])  # initial weights
lr = 0.1                        # learning rate

# Hebbian Learning Rule
def hebbian(x, y, w):
    return w + lr * x * y

# Perceptron Learning Rule
def perceptron(x, y, w):
    y_pred = np.sign(np.dot(w, x))
    return w + lr * x * (y - y_pred)

# Delta Learning Rule
def delta(x, y, w):
    y_pred = np.dot(w, x)
    return w + lr * (y - y_pred) * x

# Correlation Learning Rule
def correlation(x, y, w):
    return w + lr * x * y

# OutStar Learning Rule
def outstar(x, y, w):
    y_pred = np.dot(w, x)
    return w + lr * (y - y_pred)

# Apply each rule
print("Hebbian:", hebbian(x, y, w))
print("Perceptron:", perceptron(x, y, w))
print("Delta:", delta(x, y, w))
print("Correlation:", correlation(x, y, w))
print("OutStar:", outstar(x, y, w))
