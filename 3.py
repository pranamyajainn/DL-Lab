#Q3 Implement a python program for Perceptron Networks by considering the given scenario.
#A student wants to make a decision about whether to go for a movie or not by looking at 3 parameters using a single neuron.
#The three inputs are Favorite hero, heroine, and Climate. Each has weights and we have a bias in the perceptron.
#If the condition is true input is 1 else input is 0, weights for Favorite hero=0.2, heroine=0.4, and Climate=0.2 and bias=-0.5.
#Output is 1. The decision is to go for a movie.Calculate the Accuracy .


def perceptron(inputs, weights, bias):
    activation = sum(i * w for i, w in zip(inputs, weights)) + bias
    
    return 1 if activation >= 0 else 0

inputs = [1, 1, 1]
weights = [0.2, 0.4, 0.2]  # hero, heroine, climate
bias = -0.5
expected_output = 1

output = perceptron(inputs, weights, bias)
accuracy = 100 if output == expected_output else 0

# 📤 Show results
print("Output:", output)
print("Accuracy:", accuracy, "%")
