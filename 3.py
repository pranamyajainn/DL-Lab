# 🎯 Function to simulate a simple perceptron (single neuron)
def perceptron(inputs, weights, bias):
    # 🧠 Multiply each input with its weight and sum them up
    # + Add bias at the end
    activation = sum(i * w for i, w in zip(inputs, weights)) + bias
    
    # 🎬 Decision Rule: If activation is positive or zero → 1 (Yes)
    return 1 if activation >= 0 else 0

# 📥 INPUT: [Favorite hero, Favorite heroine, Good climate] → all true (1)
inputs = [1, 1, 1]

# 🏋️‍♂️ WEIGHTS: Importance given to each input
weights = [0.2, 0.4, 0.2]  # hero, heroine, climate

# ⚖️ BIAS: Built-in resistance (starts negative)
bias = -0.5

# 🎯 Expected output (i.e., student *wants* to go for a movie)
expected_output = 1

# 🔁 Run the perceptron
output = perceptron(inputs, weights, bias)

# ✅ Check if prediction matches expectation
accuracy = 100 if output == expected_output else 0

# 📤 Show results
print("Output:", output)
print("Accuracy:", accuracy, "%")
