import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)  # Input range

# Activation Functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def softmax(x): 
    e_x = np.exp(x - np.max(x))  # For stability
    return e_x / e_x.sum()

# Plot Function
def plot_activation(name, func):
    y = func(x)
    plt.plot(x, y)
    plt.title(name)
    plt.grid(True)
    plt.show()

# Plot all
plot_activation("Sigmoid", sigmoid)
plot_activation("Tanh", tanh)
plot_activation("ReLU", relu)
plot_activation("Leaky ReLU", leaky_relu)
plot_activation("Softmax (1D sum=1)", softmax)  # Softmax output sums to 1
